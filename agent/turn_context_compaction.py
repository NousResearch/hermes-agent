"""Turn-start compaction for ``build_turn_context`` plus the small compression-attempt
helpers shared with the pre-API / post-tool sites in ``turn_preflight``.

Three passes, in order: idle-triggered compaction (opt-in, wall-clock gap), preflight
context compression (token threshold), and the uncompressed-session overflow-warning
re-arm. ``run_turn_start_compaction`` mutates ``agent`` exactly as the inline prologue
did and returns a ``CompactionOutcome``. Predicates/estimators that tests patch on
``agent.turn_context`` are imported lazily through that module so patches intercept."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from agent.context_engine import automatic_compaction_status_message
from agent.conversation_compression import (
    IDLE_COMPACTION_STATUS_TEMPLATE, PREFLIGHT_COMPRESSION_STATUS_TEMPLATE,
    compression_skipped_due_to_lock, conversation_history_after_compression,
)

logger = logging.getLogger("agent.turn_context")


@dataclass
class CompactionOutcome:
    """Locals rebuilt by turn-start compaction (``build_turn_context`` reads them back)."""

    messages: List[Dict[str, Any]]
    active_system_prompt: Optional[str]
    conversation_history: Optional[List[Dict[str, Any]]]
    current_turn_user_idx: int
    # A preflight pass (threshold or engine-driven) actually rebuilt ``messages``.
    compressed: bool = False
    # Preflight proved an immediate retry ineffective (no progress / insufficient).
    blocked: bool = False


# ── Helpers shared by every compression-attempt site ──


def _clear_overflow_warn(agent: Any) -> None:
    """Re-arm the context-overflow warning dedup (test doubles may lack the method)."""
    # Compression is actually running (block cleared / was never blocked) — reset the blocked-overflow
    # warning dedup so a future blocked-over-threshold turn can warn again. Mirrors the turn-context
    # preflight reset (silent-overflow fix #62625). getattr guard: test doubles built via object.__new__
    # lack the method (gateway test-double pitfall) — treat absence as no-op.
    # Compression is actually running (block cleared / was never blocked) — reset the blocked-overflow
    # warning dedup so a future blocked-over-threshold turn can warn again (silent-overflow fix #62625).
    # getattr guard: test doubles built via object.__new__ lack the method (gateway test-double pitfall) —
    # treat absence as no-op.
    _clear_warn = getattr(agent, "_clear_context_overflow_warn", None)
    if callable(_clear_warn):
        _clear_warn()


def _reset_retry_state_after_compaction(agent: Any) -> None:
    """Give the compacted request a fresh chance: clear retry/empty-response state."""
    agent._empty_content_retries = 0
    agent._thinking_prefill_retries = 0
    agent._last_content_with_tools = None
    agent._last_content_tools_all_housekeeping = False
    agent._mute_post_response = False


def _blocked_compress_reason(
    compressor: Any, tokens: int, attempts_spent: Optional[int] = None
) -> Optional[str]:
    """Why an over-threshold request is blocked (``None`` below threshold or when the
    engine lacks ``should_compress_info`` / raises).

    ``attempts_spent``: when given and the engine says compression SHOULD run
    (``(True, None)``) yet the caller skipped it, the per-turn attempt budget is
    spent — name it ``attempts_exhausted:<n>`` instead of dropping the
    ``(True, None)`` on the floor (silent-lockout case, #101889)."""
    _info = getattr(compressor, "should_compress_info", None)
    if not callable(_info):
        return None
    try:
        _should_now, _reason = _info(tokens)
    except Exception:
        return None
    if attempts_spent is not None and _should_now and not _reason:
        return f"attempts_exhausted:{attempts_spent}"
    return _reason


def _apply_grown_window(agent: Any, compressor: Any, grown: int) -> None:
    """A managed local runtime granted a bigger window: recalibrate the compressor."""
    compressor.update_model(
        agent.model, grown, base_url=getattr(agent, "base_url", "") or "",
        api_key=getattr(agent, "api_key", "") or "",
        provider=getattr(agent, "provider", "") or "",
        api_mode=getattr(agent, "api_mode", "") or "",
    )
    agent._buffer_status(
        f"📈 Context window grown to {grown // 1024}K "
        f"(local model; conversation continues uncompressed)"
    )


def _refund_api_call(agent: Any, api_call_count: int) -> int:
    """A pass that never reached the provider refunds the call count and budget."""
    # Host progress-aware timeout (#98722, salvaged from #98741): this preflight iteration never reached the
    # provider. Refund its provisional call/budget exactly like a successful pre-API compaction, then stop
    # before the unchanged oversized request reaches the provider — its overflow error would only invoke
    # compression again on the same transcript with the wait budget already spent.
    api_call_count -= 1
    agent._api_call_count = api_call_count
    agent.iteration_budget.refund()
    return api_call_count


def _reanchor(agent: Any, messages: List[Any], user_message: Any) -> int:
    """Compaction rebuilt ``messages``: re-anchor this turn's user index so the
    api_content stamp, injection site and persist-override row hit the same dict."""
    from agent.turn_context import reanchor_current_turn_user_idx

    idx = reanchor_current_turn_user_idx(messages, user_message)
    agent._persist_user_message_idx = idx
    return idx


# ── Turn-start passes ──


def run_turn_start_compaction(
    agent: Any, *, messages: List[Dict[str, Any]], system_message: Optional[str],
    active_system_prompt: Optional[str], conversation_history: Optional[List[Dict[str, Any]]],
    current_turn_user_idx: int, user_message: Any, effective_task_id: str,
) -> CompactionOutcome:
    """Idle compaction, then preflight compression (or the uncompressed guard)."""
    out = CompactionOutcome(
        messages=messages, active_system_prompt=active_system_prompt,
        conversation_history=conversation_history, current_turn_user_idx=current_turn_user_idx,
    )
    _idle_compaction(agent, out, system_message, user_message, effective_task_id)
    _preflight_compression(agent, out, system_message, user_message, effective_task_id)
    return out


def _idle_compaction(
    agent: Any, out: CompactionOutcome, system_message: Optional[str], user_message: Any,
    effective_task_id: str,
) -> None:
    """Idle-triggered compaction (opt-in; ``idle_compact_after_seconds``): fires on the
    wall-clock gap since ``_last_activity_ts``; a cheap gap check gates the estimate."""
    from agent import turn_context as _tc

    messages = out.messages
    _idle_after = getattr(agent, "compression_idle_compact_after_seconds", 0)
    if not (agent.compression_enabled and _idle_after > 0 and messages):
        return
    _idle_gap = time.time() - getattr(agent, "_last_activity_ts", time.time())
    if _idle_gap < _idle_after:
        return
    _compressor = agent.context_compressor
    # A live or restored native checkpoint must reach its issuer once so real usage,
    # rather than an opaque ciphertext estimate, decides whether local compression is
    # still needed.  Threshold and post-tool preflight honor the same latch.
    if bool(getattr(_compressor, "awaiting_real_usage_after_compression", False)):
        return
    # Route-aware pressure: on compacted native-Codex sessions the durable figure
    # overstates the wire, so reuse the preflight estimator.
    _idle_tokens = _tc._preflight_request_tokens(
        agent, messages, out.active_system_prompt or ""
    )
    # Don't summarise a thread already below the post-compression target size.
    _idle_floor = int(_compressor.threshold_tokens * _compressor.summary_target_ratio)
    _idle_cooldown = getattr(
        _compressor, "get_active_compression_failure_cooldown", lambda: None
    )()
    # What the previous pass actually produced — the honest floor versus the theoretical
    # ``_idle_floor``. Type pin: compressor doubles expose truthy non-ints here; only a real
    # int may raise the floor, anything else falls back to 0 (original semantics).
    _idle_last_compaction = getattr(_compressor, "last_compression_rough_tokens", 0)
    if not isinstance(_idle_last_compaction, int) or isinstance(_idle_last_compaction, bool):
        _idle_last_compaction = 0
    if not _tc._should_idle_compact(
        enabled=agent.compression_enabled, idle_after_seconds=_idle_after,
        idle_gap_seconds=_idle_gap, tokens=_idle_tokens, floor_tokens=_idle_floor,
        cooldown_active=bool(_idle_cooldown), last_compaction_tokens=_idle_last_compaction,
    ):
        return
    logger.info(
        "Idle compaction: %ss idle >= %ss, ~%s tokens > %s floor (last compaction produced ~%s) (session %s)",
        int(_idle_gap), _idle_after, f"{_idle_tokens:,}", f"{_idle_floor:,}",
        f"{_idle_last_compaction:,}" if _idle_last_compaction > 0 else "n/a",
        agent.session_id or "none",
    )
    _idle_status = automatic_compaction_status_message(
        _compressor,
        phase="idle",
        default_message=IDLE_COMPACTION_STATUS_TEMPLATE.format(
            idle_seconds=int(_idle_gap), tokens=_idle_tokens
        ),
        approx_tokens=_idle_tokens,
        idle_seconds=int(_idle_gap),
        model=agent.model,
    )
    if _idle_status:
        agent._emit_status(_idle_status)
    out.messages, out.active_system_prompt = agent._compress_context(
        messages, system_message, approx_tokens=_idle_tokens, task_id=effective_task_id
    )
    # ``_compress_context`` returns the INPUT list object when it skips; only
    # re-baseline and re-anchor after a real compaction.
    if out.messages is not messages:
        out.conversation_history = conversation_history_after_compression(
            agent, out.messages, out.conversation_history
        )
        out.current_turn_user_idx = _reanchor(agent, out.messages, user_message)


def _codex_native_auto_compaction(agent: Any) -> bool:
    """Codex app-server threads are compacted by the codex agent itself; Hermes only
    initiates compaction in "hermes" mode."""
    return (
        # See #36801.
        getattr(agent, "api_mode", None) == "codex_app_server"
        and str(
            getattr(agent, "codex_app_server_auto_compaction", "native") or "native"
        ).lower()
        in {"native", "off"}
    )


def _preflight_compression(
    agent: Any, out: CompactionOutcome, system_message: Optional[str], user_message: Any,
    effective_task_id: str,
) -> None:
    """Preflight context compression; the cheap pre-check gates the full estimate
    (see ``_should_run_preflight_estimate`` for the OR semantics)."""
    from agent import turn_context as _tc

    agent._turn_received_provider_response = False
    agent._turn_preflight_display_snapshot = None
    if not agent.compression_enabled:
        _rearm_uncompressed_overflow_warn(agent, out.messages, out.active_system_prompt)
        return
    _compressor = agent.context_compressor
    if _tc._review_fork_first_request_pending(agent) or not _tc._should_run_preflight_estimate(
        out.messages, _compressor.protect_first_n, _compressor.protect_last_n,
        _compressor.threshold_tokens,
    ):
        return

    _preflight_tokens = _tc._preflight_request_tokens(
        agent, out.messages, out.active_system_prompt or ""
    )
    # getattr guard: compressor doubles and plugin engines lack this method — absence
    # means no snapshot and the finalizer's rollback stays disarmed.
    _snapshot_fn = getattr(_compressor, "snapshot_preflight_display_tokens", None)
    if callable(_snapshot_fn):
        _snapshot_val = _snapshot_fn()
        # Type pin: MagicMock compressors return truthy Mock objects — only a real int
        # snapshot may arm the interrupted-turn rollback.
        if isinstance(_snapshot_val, int) and not isinstance(_snapshot_val, bool):
            agent._turn_preflight_display_snapshot = _snapshot_val
    _preflight_deferred = getattr(
        _compressor, "should_defer_preflight_to_real_usage", lambda _tokens: False
    )(_preflight_tokens)
    _codex_native_auto = _codex_native_auto_compaction(agent)

    if not _preflight_deferred:
        # Display-only seed: a real provider reading wins and the -1 sentinel stays
        # protected. Also feeds the tool-loop gate on usage-less responses.
        _maybe_seed = getattr(_compressor, "maybe_seed_preflight_display_tokens", None)
        if callable(_maybe_seed):
            _maybe_seed(_preflight_tokens)

    _compression_cooldown = getattr(
        _compressor, "get_active_compression_failure_cooldown", lambda: None
    )()

    _should_compress_now = False
    _compress_block_reason = None
    if _preflight_deferred:
        logger.info(
            "Skipping preflight compression: rough estimate ~%s >= %s, "
            "but last real provider prompt was %s after compression",
            f"{_preflight_tokens:,}", f"{_compressor.threshold_tokens:,}",
            f"{_compressor.last_real_prompt_tokens:,}",
        )
    elif _compression_cooldown:
        logger.info(
            "Skipping preflight compression: same-session cooldown active "
            "(~%s seconds remaining, session %s)",
            int(_compression_cooldown.get("remaining_seconds", 0.0)),
            agent.session_id or "none",
        )
        if _preflight_tokens >= _compressor.threshold_tokens:
            # Over threshold but blocked by the summary-LLM cooldown — surface a warning.
            _cooldown_secs = _compression_cooldown.get("remaining_seconds", 0.0)
            _compress_block_reason = f"cooldown:{_cooldown_secs:.0f}"
    elif _codex_native_auto:
        logger.info(
            "Skipping Hermes preflight compression for codex app-server "
            "(mode=%s); Hermes will not start thread compaction here.",
            getattr(agent, "codex_app_server_auto_compaction", "native"),
        )
    else:
        _should_compress_now = _compressor.should_compress(_preflight_tokens)
        if not _should_compress_now:
            _compress_block_reason = _blocked_compress_reason(_compressor, _preflight_tokens)
    if _should_compress_now:
        # Managed local runtime: growing the window beats compressing (ladder order;
        # same seam as _maybe_grow_local_window in the loop).
        try:
            from agent.conversation_loop import _maybe_grow_local_window

            _grown = _maybe_grow_local_window(agent, _compressor, _preflight_tokens)
        except Exception:
            _grown = None
        if _grown:
            _apply_grown_window(agent, _compressor, _grown)
            _should_compress_now = _compressor.should_compress(_preflight_tokens)
    if _should_compress_now:
        _run_preflight_passes(
            agent, out, _compressor, _preflight_tokens, system_message, effective_task_id
        )
    elif _compress_block_reason:
        # Over threshold but compression blocked: surface a deduped warning so the
        # user can /new or /compress instead of a silent provider limit.
        agent._warn_context_overflow_blocked(
            _compress_block_reason, _preflight_tokens, _compressor.threshold_tokens
        )
    else:
        # Sub-threshold and unblocked — re-arm the overflow warning.
        _clear_overflow_warn(agent)
        # Engine maintenance only when NO skip-branch fired: cooldown, deferred
        # estimate, or codex-native route keep the engine hook unconsulted.
        if not (_compression_cooldown or _preflight_deferred or _codex_native_auto):
            _engine_preflight_maintenance(
                agent, out, _compressor, _preflight_tokens, system_message, effective_task_id
            )

    if out.compressed:
        # Compression rebuilt the list, so the pre-compression user index is stale.
        # Exact-content match first so a todo-snapshot can't steal it.
        out.current_turn_user_idx = _reanchor(agent, out.messages, user_message)


def _run_preflight_passes(
    agent: Any, out: CompactionOutcome, _compressor: Any, _preflight_tokens: int,
    system_message: Optional[str], effective_task_id: str,
) -> None:
    """Threshold-triggered preflight passes (honor ``compression.max_attempts`` like
    the loop's sites, default 3)."""
    from agent import turn_context as _tc

    out.compressed = True
    # Compression is actually running — reset the dedup so a future blocked turn can
    # warn again.
    _clear_overflow_warn(agent)
    logger.info(
        "Preflight compression: ~%s tokens >= %s threshold (model %s, ctx %s)",
        f"{_preflight_tokens:,}", f"{_compressor.threshold_tokens:,}", agent.model,
        f"{_compressor.context_length:,}",
    )
    _preflight_status = automatic_compaction_status_message(
        _compressor,
        phase="preflight",
        default_message=PREFLIGHT_COMPRESSION_STATUS_TEMPLATE.format(
            tokens=_preflight_tokens, threshold=_compressor.threshold_tokens
        ),
        approx_tokens=_preflight_tokens,
        threshold_tokens=_compressor.threshold_tokens,
        context_length=_compressor.context_length,
        model=agent.model,
    )
    if _preflight_status:
        agent._emit_status(_preflight_status)
    _max_preflight_passes = max(1, int(getattr(agent, "max_compression_attempts", 3) or 3))
    for _pass in range(_max_preflight_passes):
        _preflight_input = out.messages
        _orig_len = len(_preflight_input)
        _orig_tokens = _preflight_tokens
        out.messages, out.active_system_prompt = agent._compress_context(
            _preflight_input, system_message, approx_tokens=_preflight_tokens,
            task_id=effective_task_id,
        )
        if out.messages is _preflight_input and compression_skipped_due_to_lock(agent):
            # Lock-skip: another path holds the lock, so this is a DEFER, not proof of
            # incompressibility — don't arm the blocker; stop passes this turn.
            logger.info(
                # That is a temporary DEFER, not proof the transcript cannot compress — do NOT arm the
                # insufficient-progress blocker (the loop's error handlers must keep their provider-proven
                # retry budget) and stop preflight passes for this turn; the lock winner is shrinking the
                # same session concurrently. See #69870.
                "Preflight compression deferred: compression lock "
                "held by another path (session %s)",
                agent.session_id or "none",
            )
            break
        # Re-estimate so size-only compression (same rows, fewer tokens) counts as
        # progress.
        _preflight_tokens = _tc._preflight_request_tokens(
            agent, out.messages, out.active_system_prompt or ""
        )
        if not _tc.compression_made_progress(
            _orig_len, len(out.messages), _orig_tokens, _preflight_tokens
        ):
            _tc._fail_closed_after_preflight_timeout(agent, _preflight_tokens)
            out.blocked = True
            break  # Cannot compress further: neither rows nor tokens moved
        out.conversation_history = conversation_history_after_compression(
            agent, out.messages, out.conversation_history
        )
        _reset_retry_state_after_compaction(agent)
        if not _compressor.should_compress(_preflight_tokens):
            break
        if not _tc._compression_warrants_another_preflight_pass(
            _orig_tokens, _preflight_tokens, _compressor.threshold_tokens
        ):
            out.blocked = True
            logger.warning(
                "Preflight compression made insufficient progress: "
                "~%s -> ~%s request tokens; skipping additional passes",
                f"{_orig_tokens:,}", f"{_preflight_tokens:,}",
            )
            break


def _engine_preflight_maintenance(
    agent: Any, out: CompactionOutcome, _compressor: Any, _preflight_tokens: int,
    system_message: Optional[str], effective_task_id: str,
) -> None:
    """Engine-driven sub-threshold preflight maintenance: engines overriding
    ``should_compress_preflight()`` get exactly ONE ``compress()`` pass; a no-op never
    touches ``blocked``."""
    _engine_preflight = getattr(_compressor, "should_compress_preflight", None)
    if not callable(_engine_preflight):
        return
    try:
        _wants_engine_preflight = bool(_engine_preflight(out.messages))
    except Exception as _preflight_exc:
        # A buggy engine must never break an otherwise-healthy turn.
        logger.debug(
            "should_compress_preflight raised %s; skipping "
            "engine-driven preflight maintenance",
            _preflight_exc,
        )
        return
    if not _wants_engine_preflight:
        return
    logger.info(
        "Engine-driven preflight maintenance: %s requested "
        "compress() at ~%s tokens (below %s threshold)",
        getattr(_compressor, "name", type(_compressor).__name__),
        f"{_preflight_tokens:,}", f"{getattr(_compressor, 'threshold_tokens', 0):,}",
    )
    _engine_input = out.messages
    out.messages, out.active_system_prompt = agent._compress_context(
        _engine_input, system_message, approx_tokens=_preflight_tokens, task_id=effective_task_id
    )
    # ``_compress_context`` returns the INPUT list on every skip path and an engine
    # may no-op; re-baseline/re-anchor only after a REAL compaction.
    if out.messages is not _engine_input:
        out.compressed = True
        out.conversation_history = conversation_history_after_compression(
            agent, out.messages
        )
        _reset_retry_state_after_compaction(agent)


def _rearm_uncompressed_overflow_warn(
    agent: Any, messages: List[Any], active_system_prompt: Optional[str]
) -> None:
    """Uncompressed session guard: the warning fires from the loop's pre-API site;
    here we only RE-ARM the dedup once back under the window."""
    from agent import turn_context as _tc

    _ctx_len = getattr(getattr(agent, "context_compressor", None), "context_length", None)
    if not (isinstance(_ctx_len, int) and _ctx_len > 0):
        return
    _raw_chars = 0
    for _m in messages:
        if not isinstance(_m, dict):
            continue
        _c = _m.get("content")
        if isinstance(_c, str):
            _raw_chars += len(_c)
        elif _c:
            # Non-string, non-empty (multimodal) content defeats a char count — force
            # the real estimate. None/"" contribute nothing.
            _raw_chars = _ctx_len + 1
            break
    # Cheap gate: raw text under ~1/4 of the window (4 chars/token) cannot be over it.
    if _raw_chars <= _ctx_len:
        _clear_overflow_warn(agent)
        return
    # Re-arm with the same route-aware (checkpoint-pruned wire) figure the warn site
    # measures, else a compacted session never clears the dedup and genuine overflow
    # warnings stay suppressed.
    _uncompressed_tokens = _tc._preflight_request_tokens(
        agent, messages, active_system_prompt or ""
    )
    if _uncompressed_tokens <= _ctx_len:
        _clear_overflow_warn(agent)

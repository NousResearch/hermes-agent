"""Per-turn setup for ``run_conversation`` (the turn prologue).

``run_conversation`` opened with ~470 lines of straight-line setup before the
tool-calling loop ever started: stdio guarding, runtime-main wiring, retry-counter
resets, user-message sanitization, todo/nudge-counter hydration, system-prompt
restore-or-build, crash-resilience persistence, preflight context compression, the
``pre_llm_call`` plugin hook, and external-memory prefetch.

All of that is *prologue* — it runs once per turn, has no back-references into the
loop, and produces a fixed set of values the loop then consumes. ``TurnContext``
captures those produced values; ``build_turn_context`` performs the setup work and
returns one. ``run_conversation`` is left to unpack the context and run the loop,
shrinking the orchestrator by the full prologue.

The builder still mutates ``agent`` heavily (counters, thread id, cached prompt,
session DB) exactly as the inline code did — those side effects are the point. The
``TurnContext`` it returns carries only the *locals* the loop reads back.

Behavior is identical to the original inline prologue; this is a pure
move-and-name refactor with no semantic change.
"""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from agent.conversation_compression import conversation_history_after_compression
from agent.iteration_budget import IterationBudget
from agent.model_metadata import (
    estimate_messages_tokens_rough,
    estimate_request_tokens_rough,
)

logger = logging.getLogger(__name__)


def maybe_stamp_empty_resume_row(agent, user_msg: dict) -> bool:
    """Stamp ``user_msg`` ephemeral when the gateway flagged this turn as an
    internal empty-text auto-resume continuation, and CONSUME the flag.

    Extracted as a pure, directly-testable helper (undo-empty-resume pass-3) so
    the safety-critical consume-once behavior — a leaked flag must not drop a
    LATER real user row — is exercised by an executing test, not source
    inspection. Returns True iff the row was stamped.

    Contract: reads ``agent._suppress_user_turn_persist``; if truthy, stamps
    ``user_msg["_empty_resume_synthetic"] = True`` and resets the flag to False
    (consumed once — a single stale flag can drop at most ONE row, and the
    gateway additionally resets it at the top of every turn so it never carries
    across turns). Fail-open on any error (do not stamp → row persists normally;
    losing a real row is worse than a stray empty one).
    """
    try:
        if getattr(agent, "_suppress_user_turn_persist", False):
            user_msg["_empty_resume_synthetic"] = True
            agent._suppress_user_turn_persist = False
            return True
    except Exception:
        pass
    return False


def _compression_made_progress(
    orig_len: int, new_len: int, orig_tokens: int, new_tokens: int
) -> bool:
    """Return ``True`` if a compression pass materially reduced the request.

    Compression can succeed by summarising message contents — reducing the
    estimated request token count — without reducing the message row
    count.  Treating row count as the sole progress signal false-positives
    on size-only wins and surfaces a misleading "Cannot compress further"
    failure even when post-compression tokens are well below the model
    context window.  See issue #39548 for an observed case: 220 → 220
    messages, ~288k → ~183k tokens on a 1M-context model still triggered
    auto-reset.

    The token reduction must be *material* (>5%) to count as progress — the
    same floor the overflow-handler retry path uses (conversation_loop.py,
    #39550) — so a sub-5% wobble doesn't keep the multi-pass loop spinning.
    """
    if new_len < orig_len:
        return True
    return orig_tokens > 0 and new_tokens < orig_tokens * 0.95


def _should_run_preflight_estimate(
    messages: List[Dict[str, Any]],
    protect_first_n: int,
    protect_last_n: int,
    threshold_tokens: int,
) -> bool:
    """Cheap gate for the (expensive) full preflight token estimate.

    Returns ``True`` when either:
      (a) message count exceeds the protected ranges (the historical gate), or
      (b) a cheap char-based estimate already crosses the configured threshold
          — the few-but-huge case from issue #27405 that the count-only gate
          would silently skip (a handful of very large messages never trips
          the count condition, so compression was never attempted and the
          turn hit a hard context-overflow error).

    Branch (b) uses ``estimate_messages_tokens_rough`` (the shared char-based
    estimator) so a single large base64 image isn't mistaken for ~250K tokens.
    It intentionally undercounts vs. the full request estimate — it omits the
    system prompt and tool schemas — because it is only a *hint* deciding
    whether to pay for the authoritative ``estimate_request_tokens_rough``,
    which (together with ``should_compress``) makes the real decision.
    """
    if len(messages) > protect_first_n + protect_last_n + 1:
        return True
    return estimate_messages_tokens_rough(messages) >= threshold_tokens


@dataclass
class TurnContext:
    """Values produced by the turn prologue and consumed by the turn loop."""

    # Sanitized inbound message (surrogates stripped).
    user_message: str
    # Clean message preserved for transcripts / memory queries (no nudge injection).
    original_user_message: Any
    # Working message list for this turn (loop appends to it).
    messages: List[Dict[str, Any]]
    # May be reset to None by preflight compression (new session created).
    conversation_history: Optional[List[Dict[str, Any]]]
    # Cached system prompt active for this turn (may be rebuilt by compression).
    active_system_prompt: Optional[str]
    # Task / turn identifiers.
    effective_task_id: str
    turn_id: str
    # Index of the current user turn within ``messages``.
    current_turn_user_idx: int
    # Whether the post-turn memory review should fire.
    should_review_memory: bool = False
    # Context contributed by ``pre_llm_call`` plugins (appended to user message).
    plugin_user_context: str = ""
    # External-memory prefetch result, reused across loop iterations.
    ext_prefetch_cache: str = ""


def build_turn_context(
    agent,
    user_message: Any,
    system_message: Optional[str],
    conversation_history: Optional[List[Dict[str, Any]]],
    task_id: Optional[str],
    stream_callback,
    persist_user_message: Optional[Any],
    persist_user_timestamp: Optional[float] = None,
    persist_user_platform_id: Optional[str] = None,
    *,
    restore_or_build_system_prompt,
    install_safe_stdio,
    sanitize_surrogates,
    summarize_user_message_for_log,
    set_session_context,
    set_current_write_origin,
    ra,
) -> TurnContext:
    """Run the once-per-turn setup and return the loop's input context.

    The callables/helpers the original prologue referenced from the
    ``conversation_loop`` module are passed in explicitly to keep this module
    free of an import cycle with ``agent.conversation_loop``.
    """
    # Guard stdio against OSError from broken pipes (systemd/headless/daemon).
    install_safe_stdio()

    # NOTE: the DB session row is created later, AFTER the system prompt is
    # restored/built (see _ensure_db_session() below the system-prompt block).
    # Creating it here — before _cached_system_prompt is populated — inserts a
    # row with system_prompt=NULL on a fresh API/gateway agent that carries
    # client-managed history, which then trips the "stored system prompt is
    # null; rebuilding from scratch" warning and a needless first-turn prefix
    # cache miss. (Issue #45499.)

    # Tell auxiliary_client what the live main provider/model are for this turn.
    try:
        from agent.auxiliary_client import set_runtime_main
        set_runtime_main(
            getattr(agent, "provider", "") or "",
            getattr(agent, "model", "") or "",
            base_url=getattr(agent, "base_url", "") or "",
            api_key=getattr(agent, "api_key", "") or "",
            api_mode=getattr(agent, "api_mode", "") or "",
        )
    except Exception:
        pass

    # Tag log records on this thread with the session ID for ``hermes logs``.
    set_session_context(agent.session_id)

    # Bind the skill write-origin ContextVar for this thread.
    set_current_write_origin(getattr(agent, "_memory_write_origin", "assistant_tool"))

    # Clear the route-change announce dedupe at the top of every turn.
    #
    # ``_last_fallback_announced`` de-dupes a re-entrant announce WITHIN one
    # failover episode (I5: announce once per episode). It was reset only inside
    # ``restore_primary_runtime`` — but that reset sits AFTER two early-returns
    # (``auto_recovery=False`` and the primary rate-limit cooldown), so a session
    # that fails over, gets cooldown-pinned, and re-fails-over to the SAME route
    # on a later turn kept the stale transition and SILENTLY SUPPRESSED the repeat
    # announce (the 2026-07-12 "model changed with no 🔄" report). A new turn is a
    # new episode by definition, so clear it unconditionally here — before the
    # restore path can early-return — so a genuinely new failover always announces.
    agent._last_fallback_announced = None

    # Restore the primary runtime if the previous turn activated fallback.
    agent._restore_primary_runtime()

    # Between-turns MCP refresh: an MCP server that finished connecting since
    # the previous turn (slow HTTP/OAuth servers routinely take 2-6s on a cold
    # connect, missing the bounded startup wait) lands in THIS turn's tool
    # snapshot.  This is cache-safe by construction: it runs in the per-turn
    # prologue, before this turn's first API call assembles ``tools=``, so it
    # only ever extends a fresh request prefix — it never mutates the cached
    # prefix of an in-flight turn.  No-op when no MCP servers are registered
    # (the common case, gated by the cheap ``has_registered_mcp_tools`` check)
    # or when the tool set is unchanged (``refresh_agent_mcp_tools`` diffs by
    # name and leaves the snapshot untouched on no-change).
    try:
        if not getattr(agent, "_skip_mcp_refresh", False):
            # Import-cost gate: ``tools.mcp_tool`` pulls in the whole ``mcp``
            # package (~0.4s measured) even when the user has zero MCP servers
            # configured.  MCP tools can only be registered by code that has
            # already imported ``tools.mcp_tool`` (discovery, /reload-mcp,
            # late-binding refresh) — so if it isn't in sys.modules yet, there
            # is nothing to refresh and the import can be skipped outright.
            # This keeps the no-MCP first turn off the heavy import path
            # without changing behavior for MCP users.
            import sys as _sys
            if "tools.mcp_tool" in _sys.modules:
                from tools.mcp_tool import has_registered_mcp_tools, refresh_agent_mcp_tools
                if has_registered_mcp_tools():
                    refresh_agent_mcp_tools(agent, quiet_mode=True)
    except Exception:
        logger.debug("between-turns MCP tool refresh skipped", exc_info=True)

    # Sanitize surrogate characters from user input.
    if isinstance(user_message, str):
        user_message = sanitize_surrogates(user_message)
    if isinstance(persist_user_message, str):
        persist_user_message = sanitize_surrogates(persist_user_message)

    # Store stream callback for _interruptible_api_call to pick up.
    agent._stream_callback = stream_callback
    agent._persist_user_message_idx = None
    agent._persist_user_message_override = persist_user_message
    agent._persist_user_message_timestamp = persist_user_timestamp
    agent._persist_user_message_platform_id = persist_user_platform_id
    # Generate unique task_id if not provided to isolate VMs between tasks.
    effective_task_id = task_id or str(uuid.uuid4())
    agent._current_task_id = effective_task_id
    turn_id = f"{agent.session_id or 'session'}:{effective_task_id}:{uuid.uuid4().hex[:8]}"
    agent._current_turn_id = turn_id
    agent._current_api_request_id = ""
    # Publish this agent's (provider, model) so cronjob(action="create") can
    # resolve model="auto" to the creating agent's own model instead of leaving
    # a new LLM cron unpinned (which inherits the runtime primary — often Opus —
    # at fire time). Module-global backed (NOT a ContextVar — a ContextVar set
    # here is invisible across the tool-executor's asyncio task boundary).
    try:
        from tools.cronjob_tools import set_current_agent_model
        set_current_agent_model(
            getattr(agent, "provider", None), getattr(agent, "model", None)
        )
    except Exception:
        logger.debug("cron auto-model publish skipped", exc_info=True)

    # Reset retry counters and iteration budget at the start of each turn.
    agent._invalid_tool_retries = 0
    agent._invalid_json_retries = 0
    agent._empty_content_retries = 0
    agent._incomplete_scratchpad_retries = 0
    agent._codex_incomplete_retries = 0
    agent._thinking_prefill_retries = 0
    agent._post_tool_empty_retried = False
    agent._last_content_with_tools = None
    agent._last_content_tools_all_housekeeping = False
    agent._mute_post_response = False
    agent._unicode_sanitization_passes = 0
    agent._tool_guardrails.reset_for_turn()
    agent._tool_guardrail_halt_decision = None
    _reset_consol = getattr(agent._memory_store, "reset_consolidation_failures", None)
    if callable(_reset_consol):
        _reset_consol()
    agent._vision_supported = True

    # Pre-turn connection health check: clean up dead TCP connections.
    if agent.api_mode != "anthropic_messages":
        try:
            if agent._cleanup_dead_connections():
                agent._emit_status(
                    "🔌 Detected stale connections from a previous provider "
                    "issue — cleaned up automatically. Proceeding with fresh "
                    "connection."
                )
        except Exception:
            pass
    # Replay compression warning through status_callback for gateway platforms.
    if agent._compression_warning:
        agent._replay_compression_warning()
        agent._compression_warning = None  # send once

    # NOTE: _turns_since_memory and _iters_since_skill are NOT reset here.
    agent.iteration_budget = IterationBudget(agent.max_iterations)

    # Log conversation turn start for debugging/observability.
    _preview_text = summarize_user_message_for_log(user_message)
    _msg_preview = (_preview_text[:80] + "...") if len(_preview_text) > 80 else _preview_text
    _msg_preview = _msg_preview.replace("\n", " ")
    logger.info(
        "conversation turn: session=%s model=%s provider=%s platform=%s history=%d msg=%r",
        agent.session_id or "none", agent.model, agent.provider or "unknown",
        agent.platform or "unknown", len(conversation_history or []),
        _msg_preview,
    )

    # Initialize conversation (copy to avoid mutating the caller's list).
    messages = list(conversation_history) if conversation_history else []

    # The CLI may already have staged this input outside the history passed to
    # ``run_conversation``. Reuse it only when its clean transcript text matches
    # this turn; a stale handoff from a failed prior turn must not replace a
    # later, different user input. Voice turns compare against their explicit
    # clean persistence override rather than the API-only prefixed payload.
    pending_cli_message = getattr(agent, "_pending_cli_user_message", None)
    expected_persist_content = (
        persist_user_message if persist_user_message is not None else user_message
    )
    if (
        isinstance(pending_cli_message, dict)
        and pending_cli_message.get("content") == expected_persist_content
    ):
        user_msg = pending_cli_message
        # The CLI-staged value is the clean transcript text. Restore the
        # API-facing variant (for example, a voice-mode prefix) while retaining
        # the same dict and any close-path durable marker.
        user_msg["content"] = user_message
    else:
        user_msg = {"role": "user", "content": user_message}
        if isinstance(pending_cli_message, dict):
            agent._pending_cli_user_message = None

    # Preserve the original user message (no nudge injection).
    original_user_message = persist_user_message if persist_user_message is not None else user_message

    # An internal auto-resume continuation carries no real user text — the model
    # gets its resume prompt via ``user_message`` but the persisted row would be
    # empty. Stamp it ephemeral so the SessionDB flush drops it (no empty user
    # row in the durable transcript). Set by the gateway on the resume-pending
    # internal-empty path; consumed once here. See _EPHEMERAL_SCAFFOLDING_FLAGS.
    maybe_stamp_empty_resume_row(agent, user_msg)
    # Stamp the platform-side message id (e.g. Discord message.id) as metadata on
    # the user turn so it survives the early crash-resilience persist below
    # (turn-start flush). Load-bearing for restart drain-window recovery: backfill
    # -on-reconnect dedups via has_platform_message_id against this. SPEC D-10.
    if persist_user_platform_id is not None:
        user_msg["platform_message_id"] = persist_user_platform_id
    messages.append(user_msg)
    current_turn_user_idx = len(messages) - 1
    agent._persist_user_message_idx = current_turn_user_idx

    # Hydrate todo store from conversation history.
    if conversation_history and not agent._todo_store.has_items():
        agent._hydrate_todo_store(conversation_history)

    # Hydrate per-session nudge counters from persisted history (issue #22357).
    if conversation_history and agent._user_turn_count == 0:
        prior_user_turns = sum(
            1 for m in conversation_history if m.get("role") == "user"
        )
        if prior_user_turns > 0:
            agent._user_turn_count = prior_user_turns
            if agent._memory_nudge_interval > 0 and agent._turns_since_memory == 0:
                agent._turns_since_memory = prior_user_turns % agent._memory_nudge_interval

    # Track user turns for memory flush and periodic nudge logic.
    agent._user_turn_count += 1
    # Copilot x-initiator: the first API call of this user turn is
    # user-initiated; tool-loop follow-ups revert to "agent" (#3040).
    agent._is_user_initiated_turn = True

    # Reset the streaming context scrubber at the top of each turn.
    scrubber = getattr(agent, "_stream_context_scrubber", None)
    if scrubber is not None:
        scrubber.reset()
    # Reset the think scrubber for the same reason.
    think_scrubber = getattr(agent, "_stream_think_scrubber", None)
    if think_scrubber is not None:
        think_scrubber.reset()

    # Track memory nudge trigger (turn-based, checked here).
    should_review_memory = False
    if (agent._memory_nudge_interval > 0
            and "memory" in agent.valid_tool_names
            and agent._memory_store):
        agent._turns_since_memory += 1
        if agent._turns_since_memory >= agent._memory_nudge_interval:
            should_review_memory = True
            agent._turns_since_memory = 0

    # Cosmetic side-signal: detect an affection "reaction" (ily / <3 / good bot)
    # and notify the host so it can play hearts. Token-free, never touches the
    # conversation, and never fatal — a purely optional UI beat.
    reaction_callback = getattr(agent, "reaction_callback", None)
    if reaction_callback is not None:
        try:
            from agent.reactions import detect_reaction

            kind = detect_reaction(original_user_message)
            if kind:
                reaction_callback(kind)
        except Exception:
            pass

    if not agent.quiet_mode:
        _print_preview = summarize_user_message_for_log(user_message)
        agent._safe_print(
            f"💬 Starting conversation: '{_print_preview[:60]}"
            f"{'...' if len(_print_preview) > 60 else ''}'"
        )

    # ── System prompt (cached per session for prefix caching) ──
    if agent._cached_system_prompt is None:
        restore_or_build_system_prompt(agent, system_message, conversation_history)

    active_system_prompt = agent._cached_system_prompt

    # Create the DB session row now that _cached_system_prompt is populated, so
    # the persisted snapshot is written non-NULL on the first turn (Issue
    # #45499). Keep row creation and the marker-based append in the same
    # per-agent critical section as CLI close persistence.
    persist_lock = getattr(agent, "_session_persist_lock", None)

    def _ensure_and_persist() -> None:
        agent._ensure_db_session()
        agent._persist_session(messages, conversation_history)

    # Crash-resilience: persist the inbound user turn as soon as the session row exists.
    try:
        if persist_lock is None:
            _ensure_and_persist()
        else:
            with persist_lock:
                _ensure_and_persist()
    except Exception:
        logger.warning(
            "Early turn-start session persistence failed for session=%s",
            agent.session_id or "none",
            exc_info=True,
        )
    finally:
        # Keep an unmarked staged input available to a later close retry if the
        # normal persistence attempt failed. Once the marker is present, the
        # close path must no longer treat it as a pre-worker UI input.
        if not isinstance(pending_cli_message, dict) or pending_cli_message.get("_db_persisted"):
            agent._pending_cli_user_message = None

    # ── Preflight context compression ──
    # Gate the (expensive) full token estimate behind a cheap pre-check.
    # See ``_should_run_preflight_estimate`` for the OR semantics that fix
    # issue #27405 (a few very large messages slipping past the count gate).
    if agent.compression_enabled and _should_run_preflight_estimate(
        messages,
        agent.context_compressor.protect_first_n,
        agent.context_compressor.protect_last_n,
        agent.context_compressor.threshold_tokens,
    ):
        _preflight_tokens = estimate_request_tokens_rough(
            messages,
            system_prompt=active_system_prompt or "",
            tools=agent.tools or None,
        )
        _compressor = agent.context_compressor
        # ── P2 "compact on the truth" — calibrate the rough estimate by the
        # provider's measured skew before triggering. note_rough_sent pairs THIS
        # request's rough with the real prompt_tokens that come back, so the next
        # turn's skew is measured on the right message set. should_compress_calibrated
        # compacts when calibrated rough crosses threshold OR raw rough hits the
        # window ceiling (dense-paste / 413 guard). Replaces the removed
        # should_defer_preflight_to_real_usage ratchet.
        _compressor.note_rough_sent(_preflight_tokens)
        _calibrated = _compressor.calibrated_tokens(_preflight_tokens)
        # Cold-start observability (Greptile PR #392 P2): on an empty skew history the
        # DISPLAY calibration (_calibrated, via _current_skew=1.0 identity) can read
        # >= threshold while the TRIGGER decision correctly DEFERS using the
        # conservative cold-start prior (_trigger_skew). Without this note an operator
        # reading the displayed value sees a number above threshold and no explanation
        # for why compaction was skipped. Log ONLY when display and trigger diverge on
        # a deferral (empty history); a normal below-threshold skip stays quiet. Pure
        # logging — no effect on the decision below (INV: never perturb control flow).
        try:
            _trig_cal = _compressor._trigger_calibrated_tokens(_preflight_tokens)
            _thr = _compressor.threshold_tokens
            if _calibrated >= _thr and _trig_cal < _thr:
                logger.debug(
                    "Preflight compression DEFERRED (cold-start skew): display "
                    "calibrated ~%s (skew %.3f) >= %s threshold, but trigger calibrated "
                    "~%s (cold-start skew %.3f) < threshold — skew history empty, using "
                    "conservative prior until a real prompt_tokens pairs (model %s, ctx %s)",
                    f"{_calibrated:,}",
                    _compressor._current_skew(),
                    f"{_thr:,}",
                    f"{_trig_cal:,}",
                    _compressor._trigger_skew(),
                    agent.model,
                    f"{_compressor.context_length:,}",
                )
        except Exception:
            pass
        _last = _compressor.last_prompt_tokens
        # Do NOT overwrite the -1 sentinel (#36718). Store the CALIBRATED estimate,
        # not the raw rough: this value lands in the real-usage slot, so if the
        # provider call fails before update_from_response replaces it, the stale
        # value must reflect the provider's measured accounting — otherwise a later
        # compression check would treat the ~21%-inflated raw as real and compact a
        # request the calibrated path had just decided should fit (Greptile #111).
        if _last >= 0 and _calibrated > _last:
            _compressor.last_prompt_tokens = _calibrated
        # Codex app-server threads are compacted by the codex agent itself;
        # Hermes only initiates compaction in "hermes" mode (#36801).
        _codex_native_auto = (
            getattr(agent, "api_mode", None) == "codex_app_server"
            and str(
                getattr(
                    agent,
                    "codex_app_server_auto_compaction",
                    "native",
                )
                or "native"
            ).lower()
            in {"native", "off"}
        )

        # Upstream d153918f1: same-session compression-failure cooldown. If a
        # recent compression failed, skip the retry for the cooldown window even
        # when the calibrated estimate says compress (avoids tight failure loops).
        _compression_cooldown = getattr(
            _compressor,
            "get_active_compression_failure_cooldown",
            lambda: None,
        )()

        if _compression_cooldown:
            logger.info(
                "Skipping preflight compression: same-session cooldown active "
                "(~%s seconds remaining, session %s)",
                int(_compression_cooldown.get("remaining_seconds", 0.0)),
                agent.session_id or "none",
            )
        elif _codex_native_auto:
            logger.info(
                "Skipping Hermes preflight compression for codex app-server "
                "(mode=%s); Hermes will not start thread compaction here.",
                getattr(agent, "codex_app_server_auto_compaction", "native"),
            )
        elif _compressor.should_compress_calibrated(_preflight_tokens):
            logger.info(
                "Preflight compression: calibrated ~%s (raw ~%s × skew %.3f) >= %s "
                "threshold (model %s, ctx %s)",
                f"{_calibrated:,}",
                f"{_preflight_tokens:,}",
                _compressor._current_skew(),
                f"{_compressor.threshold_tokens:,}",
                agent.model,
                f"{_compressor.context_length:,}",
            )
            agent._emit_status(
                f"📦 Preflight compression: ~{_calibrated:,} tokens "
                f">= {_compressor.threshold_tokens:,} threshold. "
                "This may take a moment."
            )
            for _pass in range(3):
                _orig_len = len(messages)
                _orig_tokens = _preflight_tokens
                messages, active_system_prompt = agent._compress_context(
                    messages, system_message, approx_tokens=_preflight_tokens,
                    task_id=effective_task_id, trigger_reason="threshold",
                )
                # Re-estimate now so size-only compression (same row count,
                # lower token count — e.g. summarising tool outputs) is
                # recognised as progress instead of being misread as
                # "Cannot compress further". Fixes #39548.
                _preflight_tokens = estimate_request_tokens_rough(
                    messages,
                    system_prompt=active_system_prompt or "",
                    tools=agent.tools or None,
                )
                # Upstream anti-infinite-loop guard (#39548 sibling): stop if a pass moved
                # neither rows nor tokens. Then rebuild history + reset retry state.
                if not _compression_made_progress(
                    _orig_len, len(messages), _orig_tokens, _preflight_tokens
                ):
                    break  # Cannot compress further: neither rows nor tokens moved
                conversation_history = conversation_history_after_compression(
                    agent, messages
                )
                agent._empty_content_retries = 0
                agent._thinking_prefill_retries = 0
                agent._last_content_with_tools = None
                agent._last_content_tools_all_housekeeping = False
                agent._mute_post_response = False
                # Fork P2 calibrated re-check (replaces should_defer ratchet): note this
                # pass's rough so skew pairs correctly, then re-check on the calibrated value.
                _compressor.note_rough_sent(_preflight_tokens)
                if not _compressor.should_compress_calibrated(_preflight_tokens):
                    break

    # Plugin hook: pre_llm_call (context injected into user message, not system prompt).
    plugin_user_context = ""
    try:
        from hermes_cli.plugins import invoke_hook as _invoke_hook
        _pre_results = _invoke_hook(
            "pre_llm_call",
            session_id=agent.session_id,
            task_id=effective_task_id,
            turn_id=turn_id,
            user_message=original_user_message,
            conversation_history=list(messages),
            is_first_turn=(not bool(conversation_history)),
            model=agent.model,
            platform=getattr(agent, "platform", None) or "",
            sender_id=getattr(agent, "_user_id", None) or "",
        )
        _ctx_parts: list[str] = []
        # Spill oversized per-hook context to disk so a runaway plugin
        # can't inflate every subsequent turn's prompt. Ported from
        # openai/codex PR #21069 ("Spill large hook outputs from context").
        try:
            from tools.hook_output_spill import (
                get_spill_config as _spill_cfg,
                spill_if_oversized as _spill_if_oversized,
            )
            _spill_config_cached = _spill_cfg()
        except Exception:
            _spill_if_oversized = None  # type: ignore[assignment]
            _spill_config_cached = None
        for r in _pre_results:
            _piece: str = ""
            if isinstance(r, dict) and r.get("context"):
                _piece = str(r["context"])
            elif isinstance(r, str) and r.strip():
                _piece = r
            else:
                continue
            if _spill_if_oversized is not None:
                try:
                    _piece = _spill_if_oversized(
                        _piece,
                        session_id=agent.session_id,
                        source="plugin hook",
                        config=_spill_config_cached,
                    )
                except Exception as _spill_exc:
                    logger.warning("hook context spill failed: %s", _spill_exc)
            _ctx_parts.append(_piece)
        if _ctx_parts:
            plugin_user_context = "\n\n".join(_ctx_parts)
    except Exception as exc:
        logger.warning("pre_llm_call hook failed: %s", exc)

    # Per-turn file-mutation verifier state.
    agent._turn_failed_file_mutations = {}
    agent._turn_file_mutation_paths = set()
    agent._verification_stop_nudges = 0
    agent._pre_verify_nudges = 0

    # Record the execution thread so interrupt()/clear_interrupt() can scope
    # the tool-level interrupt signal to THIS agent's thread only.
    agent._execution_thread_id = threading.current_thread().ident

    # Clear stale per-thread interrupt state, preserving a pending interrupt.
    ra()._set_interrupt(False, agent._execution_thread_id)
    if agent._interrupt_requested:
        ra()._set_interrupt(True, agent._execution_thread_id)
        agent._interrupt_thread_signal_pending = False
    else:
        agent._interrupt_message = None
        agent._interrupt_thread_signal_pending = False

    # Notify memory providers of the new turn (BEFORE prefetch_all).
    if agent._memory_manager:
        try:
            _turn_msg = original_user_message if isinstance(original_user_message, str) else ""
            agent._memory_manager.on_turn_start(agent._user_turn_count, _turn_msg)
        except Exception:
            pass

    # External memory provider: prefetch once before the tool loop.
    ext_prefetch_cache = ""
    if agent._memory_manager:
        try:
            _query = original_user_message if isinstance(original_user_message, str) else ""
            ext_prefetch_cache = agent._memory_manager.prefetch_all(_query) or ""
        except Exception:
            pass

    return TurnContext(
        user_message=user_message,
        original_user_message=original_user_message,
        messages=messages,
        conversation_history=conversation_history,
        active_system_prompt=active_system_prompt,
        effective_task_id=effective_task_id,
        turn_id=turn_id,
        current_turn_user_idx=current_turn_user_idx,
        should_review_memory=should_review_memory,
        plugin_user_context=plugin_user_context,
        ext_prefetch_cache=ext_prefetch_cache,
    )

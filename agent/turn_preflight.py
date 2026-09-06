"""Pre-API preflight compression gate and post-tool compression for the turn loop.

Runs once per API call after the request pressure is measured: grow a managed llama.cpp
window (last resort), or compress when over threshold (deferring on noisy estimates, in
failure cooldown, or while the review fork's first request is pending), handle the
provider-proven overflow re-check fail-closed, and emit the deduped blocked/uncompressed
overflow warnings. Nothing here imports ``agent.conversation_loop`` at module level
(cycle); loop-internal helpers resolve lazily."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from agent.context_engine import automatic_compaction_status_message
from agent.conversation_compression import (
    PRE_API_COMPRESSION_STATUS_TEMPLATE, compression_blocked_transiently,
    compression_skipped_due_to_lock, context_compression_timed_out,
    conversation_history_after_compression,
)
from agent.turn_context import _review_fork_first_request_pending
from agent.turn_context_compaction import (
    _apply_grown_window, _blocked_compress_reason, _clear_overflow_warn, _refund_api_call,
    _reset_retry_state_after_compaction,
)

logger = logging.getLogger("agent.conversation_loop")


@dataclass
class PreflightGateVerdict:
    """``action``: ``"fallthrough"`` (make the API call), ``"continue"`` (window grown or
    history compacted — the call/budget was refunded, re-enter the turn loop and
    re-measure), ``"break"`` (turn ends: compression timeout or non-actionable compaction
    handoff — ``final_response``/``failed``/``_turn_exit_reason`` set) or ``"return"``
    (``result`` is the turn's result dict). The other fields are the loop locals the gate
    rebinds; ``_last_preflight_pressure`` is re-armed only by a compression pass."""

    action: str
    pending_moa_prepared_request: Any
    messages: Any
    active_system_prompt: Any
    conversation_history: Any
    api_call_count: Any
    compression_attempts: Any
    final_response: Any
    failed: Any
    _turn_exit_reason: Any
    _compression_timeout_exhausted: Any
    _preflight_compression_blocked: Any
    _provider_overflow_recovery_pending: Any
    _last_preflight_pressure: Any
    result: Optional[Dict[str, Any]] = None


def run_preflight_compression(
    agent: Any, v: PreflightGateVerdict, *, compressor: Any, request_pressure_tokens: int,
    provider_overflow_preflight: bool, defer_preflight: Any, moa_prepared_request: Any,
    system_message: Any, user_message: Any, max_compression_attempts: int, effective_task_id: Any,
) -> PreflightGateVerdict:
    """Mirror of the turn-prologue guard chain (defer on noisy estimate → skip in failure
    cooldown → ``should_compress``), rebinding the loop locals on ``v`` and setting
    ``v.action``. A compression pass that never reaches the provider refunds the
    call/budget in every branch (skip, re-run, timeout) so ``api_call_count`` never
    over-reports; a lock/transient skip refunds the attempt and leaves the progress
    blocker unarmed. A forced provider-overflow preflight that any gate blocks fails
    closed (llama.cpp may silently truncate)."""
    from agent.conversation_loop import (
        _COMPRESSION_TIMEOUT_FINAL_RESPONSE, _HANDOFF_SKIP_FINAL_RESPONSE,
        _compression_deferred_result, _maybe_grow_local_window, _provider_overflow_exhausted_result,
        _should_skip_model_call_for_reference_handoff,
    )

    def _done(action: str, result: Optional[Dict[str, Any]] = None) -> PreflightGateVerdict:
        v.action, v.result = action, result
        return v

    def _exhausted_result() -> Dict[str, Any]:
        return _provider_overflow_exhausted_result(
            agent, v.messages, v.conversation_history, v.api_call_count, request_pressure_tokens,
            max_compression_attempts,
        )

    _compression_cooldown = getattr(
        compressor, "get_active_compression_failure_cooldown", lambda: None
    )()
    _eligible = (
        agent.compression_enabled
        and len(v.messages) > 1
        and v.compression_attempts < max_compression_attempts
    )
    if (
        _eligible
        and not _review_fork_first_request_pending(agent)
        and (not v._preflight_compression_blocked or provider_overflow_preflight)
        and (not defer_preflight(request_pressure_tokens) or provider_overflow_preflight)
        and not _compression_cooldown
        and compressor.should_compress(request_pressure_tokens)
    ):
        # Managed local runtime: grow the context window before compressing (last
        # resort). Only for a llamacpp provider at the supervised base_url.
        _grown_window = _maybe_grow_local_window(agent, compressor, request_pressure_tokens)
        if _grown_window:
            # Bigger window granted: recalibrate and skip compression this pass. Never
            # reached the provider — refund like the compression path does.
            _apply_grown_window(agent, compressor, _grown_window)
            v.api_call_count = _refund_api_call(agent, v.api_call_count)
            return _done("continue")
        if moa_prepared_request is not None:
            v.pending_moa_prepared_request = moa_prepared_request
        v.compression_attempts += 1
        # Compression is running: reset the blocked-overflow warning dedup so a
        # later blocked turn warns again.
        _clear_overflow_warn(agent)
        _threshold = int(getattr(compressor, "threshold_tokens", 0) or 0)
        _context_length = int(getattr(compressor, "context_length", 0) or 0)
        logger.info(
            "Pre-API compression: ~%s request tokens >= %s threshold "
            "(context=%s, attempt=%s/%s)",
            f"{request_pressure_tokens:,}",
            f"{_threshold:,}",
            f"{_context_length:,}" if getattr(compressor, "context_length", 0) else "unknown",
            v.compression_attempts,
            max_compression_attempts,
        )
        _pre_api_status = automatic_compaction_status_message(
            compressor,
            phase="pre_api",
            default_message=PRE_API_COMPRESSION_STATUS_TEMPLATE.format(
                tokens=request_pressure_tokens
            ),
            approx_tokens=request_pressure_tokens,
            threshold_tokens=_threshold,
            context_length=_context_length,
            model=agent.model,
            attempt=v.compression_attempts,
            max_attempts=max_compression_attempts,
        )
        if _pre_api_status:
            agent._emit_status(_pre_api_status)
        v._last_preflight_pressure = request_pressure_tokens
        _pre_api_input = v.messages
        v.messages, v.active_system_prompt = agent._compress_context(
            v.messages, system_message, approx_tokens=request_pressure_tokens,
            task_id=effective_task_id,
        )
        if context_compression_timed_out(agent):
            # Progress-aware timeout: never reached the provider — refund the
            # call/budget and stop; an overflow retry would only re-compress.
            v.api_call_count = _refund_api_call(agent, v.api_call_count)
            v.final_response = _COMPRESSION_TIMEOUT_FINAL_RESPONSE
            v.failed = True
            v._compression_timeout_exhausted = True
            v._turn_exit_reason = "context_compression_timeout"
            return _done("break")
        if v.messages is _pre_api_input and (
            compression_skipped_due_to_lock(agent) or compression_blocked_transiently(agent)
        ):
            # Temporary DEFER (lock held / cooldown), not evidence about compressibility:
            # refund the attempt, leave the progress blocker unarmed and proceed.
            v.compression_attempts -= 1
            v._last_preflight_pressure = None
            if v.pending_moa_prepared_request is moa_prepared_request:
                v.pending_moa_prepared_request = None
        else:
            _reset_retry_state_after_compaction(agent)
            # Re-baseline the flush cursor: rotation returns None (child flushes
            # whole); in-place returns list(messages) — None would re-append
            # persisted rows. See conversation_history_after_compression().
            v.conversation_history = conversation_history_after_compression(
                agent, v.messages, v.conversation_history
            )
            # Never reaches the provider on skip or re-run — refund the call/budget
            # in BOTH cases, else budget leaks and api_call_count over-reports.
            v.api_call_count = _refund_api_call(agent, v.api_call_count)
            if _should_skip_model_call_for_reference_handoff(v.messages, user_message):
                # Reference-only handoff must not become the active turn after a
                # completed assistant response.
                logger.info(
                    "Skipping post-compaction model call: reference-only "
                    "handoff would be the sole active user turn (#80622)"
                )
                if not v.final_response:
                    v.final_response = _HANDOFF_SKIP_FINAL_RESPONSE
                v._turn_exit_reason = "compaction_handoff_not_actionable"
                return _done("break")
            return _done("continue")
    elif provider_overflow_preflight and _compression_cooldown:
        # Provider proved the request cannot fit and the compressor is unavailable:
        # don't resend; let the next user turn retry after cooldown.
        agent._persist_session(v.messages, v.conversation_history)
        return _done("return", _compression_deferred_result(
            agent, v.messages, v.api_call_count, reason="transient_block"
        ))
    elif provider_overflow_preflight and v.compression_attempts >= max_compression_attempts:
        # All recovery passes consumed and still over threshold: fail closed —
        # llama.cpp may silently truncate an oversized retry.
        return _done("return", _exhausted_result())
    elif _eligible and not defer_preflight(request_pressure_tokens) and _compression_cooldown:
        # Summary-LLM cooldown blocks compression: deduped warning only when over
        # threshold (should_compress_info reason is None below it).
        _block_reason = _blocked_compress_reason(compressor, request_pressure_tokens)
        if _block_reason:
            agent._warn_context_overflow_blocked(
                _block_reason, request_pressure_tokens,
                int(getattr(compressor, "threshold_tokens", 0) or 0),
            )
    elif not agent.compression_enabled and len(v.messages) > 1:
        # Uncompressed session guard: compression is disabled, so warn (deduped) when
        # the request exceeds the context window; the turn-context preflight re-arms.
        _ctx_len = getattr(getattr(agent, "context_compressor", None), "context_length", None)
        if isinstance(_ctx_len, int) and _ctx_len > 0 and request_pressure_tokens > _ctx_len:
            _warn_fn = getattr(agent, "_warn_uncompressed_context_overflow", None)
            if callable(_warn_fn):
                _warn_fn(request_pressure_tokens, _ctx_len)

    if provider_overflow_preflight:
        # Any other gate blocking the forced preflight (e.g. uncompressible one-
        # message request) must fail closed: the request is proven not to fit.
        return _done("return", _exhausted_result())
    return _done("fallthrough")


@dataclass
class PostToolCompressionVerdict:
    """``end_turn`` True → a reference-only compaction handoff would be the sole active
    user turn: stop without another model call (``final_response`` /
    ``turn_exit_reason`` set)."""

    end_turn: bool
    messages: List[Dict[str, Any]]
    active_system_prompt: Any
    conversation_history: Any
    compression_attempts: int
    final_response: Any
    turn_exit_reason: Any


def compress_after_tool_results(
    agent: Any, *, messages: List[Dict[str, Any]], system_message: Any, user_message: Any,
    active_system_prompt: Any, conversation_history: Any, compression_attempts: int,
    max_compression_attempts: int, effective_task_id: Any, final_response: Any,
    turn_exit_reason: Any,
) -> PostToolCompressionVerdict:
    """Post-tool-call compression decision. Pressure comes from API-reported
    ``prompt_tokens`` (a tight lower bound; thinking models inflate completion tokens),
    ``0`` right after compression (no real count yet), else the route-aware
    overhead-inclusive estimate. Over threshold but blocked → deduped warning plus the
    deterministic tool-result-only prune, committed only when the engine returns a NEW
    list (never rebuild ``conversation_history`` for it)."""
    from agent.conversation_loop import (
        _HANDOFF_SKIP_FINAL_RESPONSE, _midturn_request_pressure_tokens,
        _should_skip_model_call_for_reference_handoff,
    )
    from agent.model_metadata import estimate_request_tokens_rough

    def _verdict(end_turn: bool) -> PostToolCompressionVerdict:
        return PostToolCompressionVerdict(
            end_turn=end_turn, messages=messages, active_system_prompt=active_system_prompt,
            conversation_history=conversation_history, compression_attempts=compression_attempts,
            final_response=final_response, turn_exit_reason=turn_exit_reason,
        )

    _compressor = agent.context_compressor
    # Use real token counts from the API response to decide compression.  prompt_tokens + completion_tokens
    # is the actual context size the provider reported plus the assistant turn — a tight lower bound for the
    # next prompt. Tool results appended above aren't counted yet, but the threshold (default 50%) leaves
    # ample headroom; if tool results push past it, the next API call will report the real total and trigger
    # compression then. If last_prompt_tokens is 0 (stale after API disconnect or provider returned no usage
    # data), fall back to rough estimate to avoid missing compression. Without this, a session can grow
    # unbounded after disconnects because should_compress(0) never fires. (#2153)
    if _compressor.last_prompt_tokens > 0:
        # Only prompt_tokens: thinking models inflate completion_tokens with
        # reasoning that uses no context → premature compression.
        # Only use prompt_tokens — completion/reasoning tokens don't consume context window space. (#12026)
        _real_tokens = _compressor.last_prompt_tokens
    elif _compressor.last_prompt_tokens == -1:
        # Compression just ran, no API prompt count yet: don't treat a rough
        # schema-heavy post-compression estimate as real context pressure.
        _real_tokens = 0
    else:
        # Include tool schemas (20-30K tokens the messages-only estimate misses) and
        # stay route-aware: on a compacted native-Codex session the generic
        # durable-history figure would false-trigger.
        # Include tool schemas — with 50+ tools enabled these add 20-30K tokens the messages-only estimate
        # misses, which can skip compression past the configured threshold (#14695). Route-aware
        # (#96995/#97602 class): on a compacted native-Codex session the generic durable-history figure
        # overstates the wire and would false-trigger compression here exactly like the pre-API guard — this
        # fallback runs precisely when no provider usage is available (post-disconnect / gateway restart),
        # the unanchored case from #97602's repro.
        _real_tokens = _midturn_request_pressure_tokens(
            agent, messages, active_system_prompt or "",
            estimate_request_tokens_rough(messages, tools=agent.tools or None),
        )

    if (
        agent.compression_enabled
        and compression_attempts < max_compression_attempts
        and _compressor.should_compress(_real_tokens)
    ):
        compression_attempts += 1
        # Compression is running: reset blocked-overflow warning dedup so a
        # future blocked turn can warn again.
        _clear_overflow_warn(agent)
        agent._safe_print("  ⟳ compacting context…")
        _post_tool_input = messages
        # Pass overhead-aware _real_tokens, not last_prompt_tokens (0 in the
        # no-usage fallback), so the overflow guard sees the true size.
        messages, active_system_prompt = agent._compress_context(
            messages, system_message, approx_tokens=_real_tokens, task_id=effective_task_id
        )
        if messages is _post_tool_input and compression_skipped_due_to_lock(agent):
            # Lock-skip no-op is a temporary defer, not evidence about compressibility:
            # refund so a lock-loser loop doesn't burn the budget toward exhausted.
            # #69870 lock-skip / #97488 transient-block: this pass no-oped for a TEMPORARY reason (another
            # path holds the compression lock, or a timed cooldown/backoff guard is active). That is a
            # temporary DEFER, not evidence about compressibility — refund the attempt (it must not burn the
            # shared overflow-recovery budget toward compression_exhausted → gateway auto-reset,
            # #9893/#35809) and leave the insufficient-progress blocker unarmed. Proceed with the current
            # request: if it truly does not fit, the provider's 413/overflow handler returns the soft
            # compression_deferred result with that stronger signal.
            # #69870 lock-skip: the provider proved the request does not fit, but this compression pass
            # no-oped only because another path holds the session's compression lock. Temporary defer, not
            # exhaustion — refund the attempt and end the turn softly so the gateway does NOT auto-reset the
            # session (#9893/#35809).
            # #97488 transient-block: compression no-oped because a timed guard (host-timeout cooldown /
            # structural backoff) is active — a temporary defer, not evidence of incompressibility. Never
            # classify it as compression_exhausted (gateway auto-reset).
            # bypass_cooldown=True,  # #100661 provider-proven overflow
            # #97488: timed transient guard — defer, never exhaustion (gateway auto-reset).
            # #69870 lock-skip: the provider proved the request does not fit, but this compression pass
            # no-oped only because another path holds the session's compression lock. Temporary defer, not
            # exhaustion — refund the attempt and end the turn softly so the gateway does NOT auto-reset the
            # session (#9893/#35809).
            # #97488 transient-block: a timed guard (host-timeout cooldown / structural backoff) no-oped
            # this pass — defer softly, never compression_exhausted (which would auto-reset the session).
            # #69870 lock-skip: this pass no-oped because another path holds the session's compression lock
            # — a temporary defer, not evidence about compressibility.
            compression_attempts -= 1
        else:
            conversation_history = conversation_history_after_compression(
                agent, messages, conversation_history
            )
            if _should_skip_model_call_for_reference_handoff(messages, user_message):
                logger.info(
                    "Skipping post-tool compaction model call: "
                    "reference-only handoff would be the sole "
                    "active user turn (#80622)"
                )
                if not final_response:
                    final_response = _HANDOFF_SKIP_FINAL_RESPONSE
                turn_exit_reason = "compaction_handoff_not_actionable"
                return _verdict(True)
    elif agent.compression_enabled:
        # Over threshold but compression blocked (cooldown/anti-thrash): deduped
        # warning so context can't silently overflow. ``attempts_spent`` names the
        # attempts_exhausted lockout when the engine says RUN but the per-turn
        # budget is spent (#101889).
        _block_reason = _blocked_compress_reason(
            _compressor, _real_tokens, attempts_spent=compression_attempts
        )
        if _block_reason:
            agent._warn_context_overflow_blocked(
                _block_reason, _real_tokens, int(getattr(_compressor, "threshold_tokens", 0) or 0)
            )
        # Proactive tool-result prune (deterministic, no LLM, keeps tail): no-op unless
        # proactive_prune_tokens is exceeded; commits only past
        # proactive_prune_min_reclaim_tokens so cache breaks stay episodic.
        _prune = getattr(_compressor, "prune_tool_results_only", None)
        if callable(_prune):
            try:
                _pruned_msgs, _pruned_n = _prune(messages, current_tokens=_real_tokens)
            except Exception:
                logger.debug("proactive tool-result prune failed; skipping", exc_info=True)
                _pruned_msgs, _pruned_n = messages, 0
            # Standard no-op caller contract: only commit when the engine returned a
            # NEW list object with a non-zero count. Do NOT rebuild
            # conversation_history: rows already carry _DB_PERSISTED_MARKER, and on a
            # stale in-place flag the helper could seed unpersisted rows.
            if _pruned_n and _pruned_msgs is not messages:
                messages = _pruned_msgs
    return _verdict(False)

"""Pre-API pressure gate for the conversation turn loop: the Ollama runtime-context floor,
the provider-overflow re-check arming, the insufficient-progress blocker (compares fully
assembled requests, not raw ``messages``) and the call into
``turn_preflight.run_preflight_compression``. Nothing here imports
``agent.conversation_loop`` at module level (cycle)."""

from __future__ import annotations

import logging
from contextlib import suppress
from typing import Any

from agent.message_metadata import append_message
from agent.turn_context import _compression_warrants_another_preflight_pass
from agent.turn_preflight import PreflightGateVerdict, run_preflight_compression

logger = logging.getLogger("agent.conversation_loop")

# A request at or above this share of the context window, with no preflight compression
# able to run, is one provider call away from a context-length error.
_CONTEXT_EXHAUSTION_WARN_RATIO = 0.9


def _warn_if_context_exhaustion_unrecoverable(
    agent: Any, v: PreflightGateVerdict, *, compressor: Any, request_pressure_tokens: Any,
    compression_attempts: Any, max_compression_attempts: Any, defer_preflight: Any,
) -> None:
    """Log when a nearly full request has no preflight compression path left.

    Compression is the only thing that shrinks a turn before the provider sees it. When it is
    disabled, blocked for insufficient progress, deferred to real usage, cooling down after a
    failure, or out of attempts, a request at 90%+ of the window reaches the provider as is and
    usually fails there. The warning names the reason so the log explains the failure that
    follows. It never changes the gate's decision."""
    window = int(getattr(compressor, "context_length", 0) or 0)
    try:
        pressure = int(request_pressure_tokens or 0)
    except (TypeError, ValueError):
        return
    if window <= 0 or pressure < int(window * _CONTEXT_EXHAUSTION_WARN_RATIO):
        return
    cooldown = getattr(compressor, "get_active_compression_failure_cooldown", lambda: None)()
    reasons = [name for name, active in (
        ("compression disabled", not getattr(agent, "compression_enabled", True)),
        ("insufficient progress", bool(v._preflight_compression_blocked)),
        ("deferred to provider usage", bool(defer_preflight(pressure))),
        ("failure cooldown", bool(cooldown)),
        ("no compression attempts left", max_compression_attempts - compression_attempts <= 0),
    ) if active]
    if not reasons:
        return
    logger.warning(
        "Request ~%s tokens is %.0f%% of the %s-token context window and preflight "
        "compression cannot run (%s); the provider will likely reject it. Start a new "
        "session or compress manually.",
        f"{pressure:,}", pressure / window * 100, f"{window:,}", ", ".join(reasons),
    )


def run_preflight_gate(
    agent: Any, *, request_pressure_tokens: Any, _moa_prepared_request: Any,
    pending_moa_prepared_request: Any, messages: Any, system_message: Any, user_message: Any,
    active_system_prompt: Any, conversation_history: Any, api_call_count: Any,
    compression_attempts: Any, max_compression_attempts: Any, effective_task_id: Any,
    final_response: Any, failed: Any, _turn_exit_reason: Any, _compression_timeout_exhausted: Any,
    _preflight_compression_blocked: Any, _provider_overflow_recovery_pending: Any,
    _last_preflight_pressure: Any,
) -> PreflightGateVerdict:
    """Run the pre-API guard chain in the original order. ``_last_preflight_pressure`` is
    consumed here (set to None) and re-armed only by a compression pass, so a blocked
    preflight never compares against a stale figure."""
    from agent.conversation_loop import _ollama_context_limit_error

    v = PreflightGateVerdict(
        action="fallthrough", pending_moa_prepared_request=pending_moa_prepared_request,
        messages=messages, active_system_prompt=active_system_prompt,
        conversation_history=conversation_history, api_call_count=api_call_count,
        compression_attempts=compression_attempts, final_response=final_response, failed=failed,
        _turn_exit_reason=_turn_exit_reason,
        _compression_timeout_exhausted=_compression_timeout_exhausted,
        _preflight_compression_blocked=_preflight_compression_blocked,
        _provider_overflow_recovery_pending=_provider_overflow_recovery_pending,
        _last_preflight_pressure=None,
    )

    _runtime_context_error = _ollama_context_limit_error(agent, request_pressure_tokens)
    if _runtime_context_error:
        v.final_response = _runtime_context_error
        v.failed = True
        v._turn_exit_reason = "ollama_runtime_context_too_small"
        append_message(messages, {"role": "assistant", "content": v.final_response})
        agent._emit_diagnostic_status("❌ Ollama runtime context is too small for Hermes tool use")
        v.api_call_count -= 1
        agent._api_call_count = v.api_call_count
        with suppress(Exception):
            agent.iteration_budget.refund()
        v.action = "break"
        return v

    # Pre-API pressure check: tool results grow a turn and last_prompt_tokens lags
    # them. Mirror the turn-prologue guard chain: defer on noisy estimate, skip in
    # failure cooldown, then should_compress().
    _compressor = agent.context_compressor
    _preflight_threshold = int(getattr(_compressor, "threshold_tokens", 0) or 0)
    _provider_overflow_preflight = _provider_overflow_recovery_pending and (
        _preflight_threshold <= 0 or request_pressure_tokens >= _preflight_threshold
    )
    if _provider_overflow_recovery_pending and not _provider_overflow_preflight:
        # The outer-loop rebuild includes system prompt, request-only injections and
        # tool schemas; only that full request with output runway may be sent.
        v._provider_overflow_recovery_pending = False
    # Compare fully assembled requests, not raw ``messages`` (which omit
    # api_content, plugin injections, prefills, MoA context, ephemeral system text).
    if (
        _last_preflight_pressure is not None
        and request_pressure_tokens >= _preflight_threshold
        and not _compression_warrants_another_preflight_pass(
            _last_preflight_pressure, request_pressure_tokens, _preflight_threshold
        )
    ):
        # Stop proactive retries this turn without consuming the shared overflow-
        # recovery budget; the provider's error handler may still compact.
        v._preflight_compression_blocked = True
        logger.warning(
            "Pre-API compression made insufficient progress: ~%s -> "
            "~%s request tokens; skipping additional preflight passes",
            f"{_last_preflight_pressure:,}",
            f"{request_pressure_tokens:,}",
        )
    # An anchored figure is real usage + delta: never deferred. Only a whole-context rough
    # estimate waits for the provider's count.
    _defer_preflight = (
        (lambda _t: False) if getattr(agent, "_request_pressure_anchored", False)
        else getattr(_compressor, "should_defer_preflight_to_real_usage", lambda _t: False)
    )
    _warn_if_context_exhaustion_unrecoverable(
        agent, v, compressor=_compressor, request_pressure_tokens=request_pressure_tokens,
        compression_attempts=compression_attempts, max_compression_attempts=max_compression_attempts,
        defer_preflight=_defer_preflight,
    )
    return run_preflight_compression(
        agent, v, compressor=_compressor, request_pressure_tokens=request_pressure_tokens,
        provider_overflow_preflight=_provider_overflow_preflight,
        defer_preflight=_defer_preflight,
        moa_prepared_request=_moa_prepared_request, system_message=system_message,
        user_message=user_message, max_compression_attempts=max_compression_attempts,
        effective_task_id=effective_task_id,
    )

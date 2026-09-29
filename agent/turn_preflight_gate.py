"""Pre-API pressure gate for the conversation turn loop: the --run-budget deadline stop, the
Ollama runtime-context floor,
the provider-overflow re-check arming, the insufficient-progress blocker (compares fully
assembled requests, not raw ``messages``) and the call into
``turn_preflight.run_preflight_compression``. Nothing here imports
``agent.conversation_loop`` at module level (cycle)."""

from __future__ import annotations

import logging
import time
from contextlib import suppress
from types import SimpleNamespace
from typing import Any, Callable, Optional

from agent.message_metadata import append_message
from agent.turn_context import _compression_warrants_another_preflight_pass
from agent.turn_preflight import PreflightGateVerdict, run_preflight_compression

logger = logging.getLogger("agent.conversation_loop")


def run_budget_deadline(agent) -> Optional[float]:
    """Wall-clock time the current run's --run-budget runs out, or None with no active budget."""
    run_budget = getattr(agent, "run_budget_seconds", None)
    started = getattr(agent, "_run_budget_started_at", None)
    if not run_budget or not started:
        return None
    return float(started) + float(run_budget)


class StreamBudgetCut:
    """The --run-budget deadline as seen by one model stream (#127773). A stream opened before the
    deadline stops at its first chunk after it; one opened after it is the run's one grace call
    (this gate allows exactly one) and finishes its answer. Silence is left to the stale timeout,
    so a local model still prefilling at the deadline keeps its full grace."""

    def __init__(self, agent: Any) -> None:
        deadline = run_budget_deadline(agent)
        self.deadline = deadline if deadline is not None and time.time() < deadline else None
        self.spent = False

    def reached(self) -> bool:
        """True (and stays True) once a chunk arrives past the deadline."""
        self.spent = self.spent or (self.deadline is not None and time.time() >= self.deadline)
        return self.spent

    def mark(self, response: Any) -> Any:
        """Tag a cut response: the length path ends the turn on this mark and keeps the text."""
        if self.spent:
            response._run_budget_spent = True
        return response

    def mark_cut(self, message: Any, runaway: Any, log_runaway: Callable[[Any], None]) -> Any:
        """An Anthropic message built from the stream snapshot after a runaway or deadline cut."""
        if runaway:
            log_runaway(runaway)
            message._runaway_repetition = True
            return message
        # Cut before any block started: an empty content list reads as an invalid response and
        # would be retried past the deadline.
        message.content = message.content or [SimpleNamespace(type="text", text="")]
        return self.mark(message)


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

    # --run-budget deadline (#127773). Past it, the model gets one more call, the way the
    # iteration budget keeps a grace call: it carries the wrap-up notice when the deadline passed
    # inside a tool, and is the run's chance to answer. A later call does not start.
    _deadline = run_budget_deadline(agent)
    if _deadline is not None and time.time() >= _deadline:
        if not getattr(agent, "_run_budget_grace_used", False):
            agent._run_budget_grace_used = True
        else:
            from agent.turn_truncation import end_turn_at_run_budget
            v.api_call_count -= 1
            agent._api_call_count = v.api_call_count
            with suppress(Exception):
                agent.iteration_budget.refund()
            v.result = end_turn_at_run_budget(
                agent, messages, conversation_history, v.api_call_count, effective_task_id)
            v.action = "return"
            return v

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
    return run_preflight_compression(
        agent, v, compressor=_compressor, request_pressure_tokens=request_pressure_tokens,
        provider_overflow_preflight=_provider_overflow_preflight,
        # An anchored figure is real usage + delta: never deferred. Only a whole-context rough
        # estimate waits for the provider's count.
        defer_preflight=(
            (lambda _t: False) if getattr(agent, "_request_pressure_anchored", False)
            else getattr(_compressor, "should_defer_preflight_to_real_usage", lambda _t: False)
        ),
        moa_prepared_request=_moa_prepared_request, system_message=system_message,
        user_message=user_message, max_compression_attempts=max_compression_attempts,
        effective_task_id=effective_task_id,
    )

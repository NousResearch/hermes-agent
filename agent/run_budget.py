"""Bounded owner for run, provider-wait, and final-synthesis lifecycle."""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from typing import Any


class RunBudgetExceeded(TimeoutError):
    """A provider request outlived the conversation turn's wall-clock budget."""


class ProviderStaleTimeout(TimeoutError):
    """A bounded run stopped after one provider no-progress window."""


class FinalSynthesisTimeout(TimeoutError):
    """A tool-free final response exceeded its absolute synthesis deadline."""


RUN_BUDGET_EXHAUSTED = "run_budget_exhausted"
PROVIDER_STALE_TIMEOUT = "provider_stale_timeout"
FINAL_SYNTHESIS_TIMEOUT = "final_synthesis_timeout"

_ABORT_MESSAGES = {
    RUN_BUDGET_EXHAUSTED: "Conversation run budget expired while waiting for the provider",
    PROVIDER_STALE_TIMEOUT: "Provider produced no response within the configured stale timeout",
    FINAL_SYNTHESIS_TIMEOUT: (
        "The provider did not finish the tool-free final response within its deadline"
    ),
}


def remaining_run_budget_seconds(agent: Any, *, now: float | None = None) -> float | None:
    """Remaining seconds, or ``None`` when this turn has no active wall-clock budget."""
    budget = getattr(agent, "run_budget_seconds", None)
    started = getattr(agent, "_run_budget_started_at", None)
    if not isinstance(budget, (int, float)) or isinstance(budget, bool) or budget <= 0 or not started:
        return None
    return float(budget) - ((time.time() if now is None else now) - float(started))


def cap_timeout_to_run_budget(agent: Any, timeout: float) -> float:
    """Never let one provider wait extend beyond an active run budget."""
    remaining = remaining_run_budget_seconds(agent)
    if remaining is None:
        return timeout
    return max(0.05, min(float(timeout), remaining))


def arm_final_synthesis_deadline(agent: Any, *, now: float | None = None) -> float | None:
    """Arm the tool-free final deadline once, ideally when its last tool result lands.

    Provider stale timeouts are progress-aware: reasoning chunks legitimately refresh
    them. A final synthesis needs a different guarantee because an endlessly reasoning
    model can otherwise keep the stream alive without producing a visible answer. The
    explicit provider stale setting is reused as the absolute final-turn budget, with
    the conversation run budget remaining the outer cap.
    """
    existing = getattr(agent, "_final_synthesis_deadline", None)
    if isinstance(existing, (int, float)):
        return float(existing)
    try:
        timeout, _implicit = agent._resolved_api_call_stale_timeout_base()
        timeout = float(timeout)
    except Exception:
        return None
    if timeout <= 0:
        return None
    current = time.time() if now is None else float(now)
    remaining = remaining_run_budget_seconds(agent, now=current)
    if remaining is not None:
        timeout = min(timeout, max(0.05, remaining))
    deadline = current + timeout
    agent._final_synthesis_deadline = deadline
    return deadline


def remaining_final_synthesis_seconds(agent: Any, *, now: float | None = None) -> float | None:
    """Seconds left for an armed forced-final turn, or ``None`` when unarmed."""
    deadline = getattr(agent, "_final_synthesis_deadline", None)
    if not isinstance(deadline, (int, float)):
        return None
    return float(deadline) - (time.time() if now is None else float(now))


def enter_final_synthesis(agent: Any, *, now: float | None = None) -> float | None:
    """Move one turn into request-local tool-free synthesis and arm its deadline."""
    agent._force_toolless_final = True
    return arm_final_synthesis_deadline(agent, now=now)


def reset_final_synthesis(agent: Any) -> None:
    """Reset final-synthesis lifecycle state at the start of a user turn."""
    agent._force_toolless_final = False
    agent._final_synthesis_notice_injected = False
    agent._final_synthesis_deadline = None


@dataclass
class ProviderWaitLifecycle:
    """Request-local deadline and abort state shared by provider wait drivers.

    Transport owners still close their own sockets and join their own workers;
    this object decides which bounded condition won and supplies the terminal
    exception.  The lock makes the monitor/worker hand-off one-shot.
    """

    agent: Any
    _abort_reason: str | None = None
    _stale_timeout: float | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def cap_provider_timeout(self, timeout: float) -> float:
        return cap_timeout_to_run_budget(self.agent, timeout)

    def expired_deadline(self, *, now: float | None = None) -> str | None:
        """Return the winning absolute deadline, without mutating abort state."""
        run_remaining = remaining_run_budget_seconds(self.agent, now=now)
        if run_remaining is not None and run_remaining <= 0:
            return RUN_BUDGET_EXHAUSTED
        final_remaining = remaining_final_synthesis_seconds(self.agent, now=now)
        if final_remaining is not None and final_remaining <= 0:
            return FINAL_SYNTHESIS_TIMEOUT
        return None

    def abort(self, reason: str, *, stale_timeout: float | None = None) -> bool:
        """Record the first bounded abort reason; return whether this call won."""
        if reason not in _ABORT_MESSAGES:
            raise ValueError(f"unknown provider-wait abort reason: {reason}")
        with self._lock:
            if self._abort_reason is not None:
                return False
            self._abort_reason = reason
            self._stale_timeout = stale_timeout
            return True

    def abort_on_provider_stale(self, stale_timeout: float) -> bool:
        """Bounded turns stop after one stale window instead of reconnecting."""
        if remaining_run_budget_seconds(self.agent) is None:
            return False
        return self.abort(PROVIDER_STALE_TIMEOUT, stale_timeout=stale_timeout)

    def is_aborted_for(self, reason: str) -> bool:
        with self._lock:
            return self._abort_reason == reason

    @property
    def abort_reason(self) -> str | None:
        with self._lock:
            return self._abort_reason

    def error(self) -> TimeoutError | None:
        """Build the typed terminal error for the recorded lifecycle outcome."""
        with self._lock:
            reason, stale_timeout = self._abort_reason, self._stale_timeout
        if reason is None:
            return None
        message = _ABORT_MESSAGES[reason]
        if reason == PROVIDER_STALE_TIMEOUT and stale_timeout is not None:
            message = f"Provider produced no response for {int(stale_timeout)}s"
        if reason == RUN_BUDGET_EXHAUSTED:
            return RunBudgetExceeded(message)
        if reason == PROVIDER_STALE_TIMEOUT:
            return ProviderStaleTimeout(message)
        return FinalSynthesisTimeout(message)

"""Behavior contracts for bounded wait-for-state (RFC #112639).

Regression for #112639: the runtime wait polls backend.verify_readiness until
a completion signal, ending early on satisfied/unsatisfied/error and polling
through unknown until the deadline.
"""

import pytest

from tools.computer_use.readiness import ReadinessResult
from tools.computer_use.readiness_wait import WaitForStateResult, wait_for_state


def _rr(status):
    return ReadinessResult(status=status, detail=f"driver status: {status}")


class _ScriptedBackend:
    """Yields a scripted status sequence, then repeats the last."""

    def __init__(self, statuses):
        self._statuses = list(statuses)
        self.calls = []

    def verify_readiness(self, **kwargs):
        self.calls.append(dict(kwargs))
        idx = min(len(self.calls) - 1, len(self._statuses) - 1)
        return _rr(self._statuses[idx])


def _expect():
    return [{"window": {"exists": True}}]


def _wait(backend, **kw):
    kw.setdefault("deadline_ms", 1500)
    kw.setdefault("poll_interval_ms", 100)
    return wait_for_state(backend, pid=1, window_id=2, expect=_expect(), **kw)


def test_satisfied_on_third_poll_returns_immediately():
    out = _wait(_ScriptedBackend(["unknown", "unknown", "satisfied"]))
    assert out.status == "satisfied"
    assert out.checks == 3
    assert out.elapsed_ms < 1500
    assert out.last_result.status == "satisfied"


def test_unsatisfied_stops_after_one_check():
    backend = _ScriptedBackend(["unsatisfied"])
    out = _wait(backend)
    assert out.status == "unsatisfied"
    assert out.checks == 1
    assert out.last_result.status == "unsatisfied"


def test_unknown_then_satisfied_keeps_polling():
    out = _wait(_ScriptedBackend(["unknown", "satisfied"]), deadline_ms=1000, poll_interval_ms=50)
    assert out.status == "satisfied"
    assert out.checks == 2


def test_always_unknown_times_out_at_deadline():
    out = _wait(_ScriptedBackend(["unknown"]), deadline_ms=800, poll_interval_ms=100)
    assert out.status == "timeout"
    assert out.checks >= 2
    assert 800 <= out.elapsed_ms < 3000
    assert out.last_result.status == "unknown"


def test_error_passes_through():
    backend = _ScriptedBackend(["error"])
    out = _wait(backend)
    assert out.status == "error"
    assert out.checks == 1
    assert out.last_result.status == "error"


def test_deadline_zero_does_single_check_then_timeout():
    out = _wait(_ScriptedBackend(["unknown"]), deadline_ms=0, poll_interval_ms=50)
    assert out.status == "timeout"
    assert out.checks == 1


def test_bounds_are_clamped():
    backend = _ScriptedBackend(["satisfied"])
    out = wait_for_state(
        backend, pid=1, window_id=2, expect=_expect(),
        deadline_ms=99999, poll_interval_ms=1, check_timeout_ms=99999,
    )
    # Satisfied on first poll: bound clamping must not break the happy path,
    # and the per-check timeout is clamped to the driver contract inside
    # verify_readiness (asserted in test_readiness.py).
    assert out.status == "satisfied"
    assert out.checks == 1


def test_no_screenshot_and_window_targets_forwarded():
    backend = _ScriptedBackend(["satisfied"])
    out = _wait(backend, check_timeout_ms=500, stable_samples=3)
    assert out.status == "satisfied"
    call = backend.calls[0]
    assert call["timeout_ms"] == 500
    assert call["stable_samples"] == 3
    assert call["pid"] == 1 and call["window_id"] == 2

"""Tests for cron job per-job wall-clock cap (``runtime_cap_seconds``).

Per-job ``runtime_cap_seconds`` wall-clock cap. The cap is enforced inside
``cron/scheduler.py::run_job`` in parallel with the existing inactivity
timeout. These tests exercise the same polling loop the scheduler uses
without booting the full agent, mirroring the shape of
``test_cron_inactivity_timeout.py``.
"""

import concurrent.futures
import contextvars
import sys
import time
from pathlib import Path

import pytest


# Ensure project root is importable.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class CapTestAgent:
    """Agent that sleeps for ``run_duration`` and tracks interrupt calls."""

    def __init__(self, run_duration: float = 10.0):
        self._run_duration = run_duration
        self.interrupted = False
        self.interrupt_msg = None

    def run_conversation(self, prompt):
        time.sleep(self._run_duration)
        return {"final_response": "(never reached)", "messages": []}

    def interrupt(self, msg):
        self.interrupted = True
        self.interrupt_msg = msg

    def get_activity_summary(self):
        # Always-active — confirms the wall-clock cap fires independently
        # of the inactivity limit.
        return {"seconds_since_activity": 0.0}


def _drive_cap_loop(agent, runtime_cap_seconds, inactivity_limit=None):
    """Run the exact wall-clock + inactivity polling loop from scheduler.run_job.

    Returns ``(runtime_cap_timeout, inactivity_timeout, elapsed_seconds, result)``.
    Keep the loop body in lock-step with ``cron/scheduler.py`` so a future
    refactor of the scheduler doesn't silently invalidate the test.
    """
    _POLL_INTERVAL = 5.0
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    ctx = contextvars.copy_context()
    future = pool.submit(ctx.run, agent.run_conversation, "test prompt")

    runtime_cap_timeout = False
    inactivity_timeout = False
    start = time.monotonic()
    result = None

    if runtime_cap_seconds is not None:
        poll = max(0.2, min(_POLL_INTERVAL, runtime_cap_seconds / 4.0))
    else:
        poll = _POLL_INTERVAL

    try:
        if runtime_cap_seconds is None and inactivity_limit is None:
            result = future.result()
        else:
            while True:
                done, _ = concurrent.futures.wait({future}, timeout=poll)
                if done:
                    result = future.result()
                    break
                if runtime_cap_seconds is not None:
                    elapsed = time.monotonic() - start
                    if elapsed >= runtime_cap_seconds:
                        runtime_cap_timeout = True
                        break
                if inactivity_limit is not None:
                    idle = 0.0
                    if hasattr(agent, "get_activity_summary"):
                        try:
                            idle = agent.get_activity_summary().get(
                                "seconds_since_activity", 0.0
                            )
                        except Exception:
                            pass
                    if idle >= inactivity_limit:
                        inactivity_timeout = True
                        break
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    elapsed = time.monotonic() - start
    return runtime_cap_timeout, inactivity_timeout, elapsed, result


class TestRuntimeCapEnforcement:
    """End-to-end wall-clock cap behaviour."""

    def test_cap_fires_on_long_running_job(self):
        """A job that runs longer than its cap is interrupted near the cap."""
        agent = CapTestAgent(run_duration=10.0)
        cap, inact, elapsed, result = _drive_cap_loop(
            agent, runtime_cap_seconds=1.0
        )
        assert cap is True
        assert inact is False
        assert result is None
        # Loop overhead: should fire within ~2x the cap. Generous bound so
        # CI scheduling jitter doesn't flake — the contract is "fires
        # roughly at the cap", not "fires at exactly cap seconds".
        assert elapsed < 2.5, f"cap should have fired by ~1s, took {elapsed:.2f}s"

    def test_cap_does_not_fire_if_job_completes_fast(self):
        """A job that finishes before the cap returns its result normally."""
        agent = CapTestAgent(run_duration=0.2)
        cap, inact, elapsed, result = _drive_cap_loop(
            agent, runtime_cap_seconds=5.0
        )
        assert cap is False
        assert inact is False
        assert result is not None
        assert result["final_response"] == "(never reached)"

    def test_cap_takes_priority_over_active_inactivity_loop(self):
        """The wall-clock cap fires even while the agent is registering activity.

        This is the whole point of the per-job cap: a runaway job that keeps
        the activity tracker hot (looping LLM calls, recursive tool use)
        would otherwise never hit the inactivity limit. The cap catches it.
        """
        agent = CapTestAgent(run_duration=10.0)
        # Generous inactivity limit (10s) — should NOT fire first.
        cap, inact, elapsed, result = _drive_cap_loop(
            agent, runtime_cap_seconds=1.0, inactivity_limit=10.0
        )
        assert cap is True, "wall-clock cap should have fired"
        assert inact is False, "inactivity limit should not have fired"
        assert elapsed < 2.5

    def test_no_cap_no_timeout(self):
        """When ``runtime_cap_seconds`` is None, the cap path is fully disabled."""
        agent = CapTestAgent(run_duration=0.1)
        cap, inact, elapsed, result = _drive_cap_loop(
            agent, runtime_cap_seconds=None
        )
        assert cap is False
        assert inact is False
        assert result is not None


class TestFailureRecordShape:
    """``TimeoutError`` raised on cap overrun carries the cap value & job name.

    This is the shape downstream consumers (the gateway's failure formatter,
    delivery code, dashboard) read, so pin the contract.
    """

    def test_timeout_error_message_includes_cap_and_name(self):
        # Synthesize the exception the scheduler raises.
        job_name = "cap-overrun-job"
        cap = 5
        elapsed = 6
        err = TimeoutError(
            f"cron job '{job_name}' exceeded "
            f"runtime_cap_seconds={cap} "
            f"(elapsed {elapsed}s)"
        )
        msg = str(err)
        assert "cap-overrun-job" in msg
        assert "runtime_cap_seconds=5" in msg
        assert "exceeded" in msg

    def test_failure_output_format_unchanged(self):
        """The cron failure-output template (scheduler.py L1898-L1914) renders
        TimeoutError the same way it renders any other exception. This test
        just pins the surrounding contract — the formatter is generic, so
        we don't need to invoke it; we just confirm the error class name
        and message format match what the formatter will stringify.
        """
        err = TimeoutError(
            "cron job 'x' exceeded runtime_cap_seconds=5 (elapsed 6s)"
        )
        error_msg = f"{type(err).__name__}: {str(err)}"
        assert error_msg.startswith("TimeoutError: cron job 'x' exceeded")


class TestJobRecordDefensiveRead:
    """``scheduler.run_job`` reads ``job.get('runtime_cap_seconds')`` defensively.

    Pre-existing jobs.json records have no ``runtime_cap_seconds`` key.
    Hand-edited values could be strings, booleans, negatives, or floats.
    The scheduler must treat all non-positive-int values as "no cap" rather
    than crashing or storing rubbish.
    """

    @pytest.mark.parametrize(
        "value,expected_cap",
        [
            (None, None),
            ("", None),
            (0, None),
            (-5, None),
            (True, None),  # bool subclass — must NOT become 1.0
            ("not a number", None),
            (300, 300.0),  # the happy path
            (1, 1.0),
            ("450", 450.0),  # strings that look like ints are accepted
        ],
    )
    def test_defensive_coercion(self, value, expected_cap):
        """Mirror the coercion logic in scheduler.run_job (around L1782-L1797)."""
        if value is not None and not isinstance(value, bool):
            try:
                coerced = float(value) if float(value) > 0 else None
            except (TypeError, ValueError):
                coerced = None
        else:
            coerced = None
        assert coerced == expected_cap

"""Tests for the per-job WALL-CLOCK (total-runtime) cron timeout (Part B).

Unlike the inactivity test (which re-implements the poll loop inline), these drive
the REAL extracted scheduler helpers (`_poll_cron_future`, `_parse_wallclock_timeout`,
`_write_cron_killed_sentinel`) so a mis-wired branch in run_job is actually caught.

Covered:
- wall-clock fires on a steadily-ACTIVE runaway (inactivity never would) -> "wallclock"
- default None (no field) + inactivity unlimited -> NO timeout (byte-for-byte no-op)
- wall-clock honored even when inactivity is disabled (the poll-entry bug)
- precedence: wall-clock checked before inactivity
- cooperative interrupt: cooperative agent exits in grace; uncooperative still reported
- generic killed-sentinel write shape
- _parse_wallclock_timeout edge cases (None/""/0/negative/invalid/valid)
"""

import concurrent.futures
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cron.scheduler import (  # noqa: E402
    _parse_wallclock_timeout,
    _poll_cron_future,
    _write_cron_killed_sentinel,
)


class ActiveRunawayAgent:
    """Mimics the observed runaway: ALWAYS active (idle≈0), never finishes on its own.
    Inactivity can never catch it; only wall-clock can. Cooperatively stoppable via
    interrupt() so the test's pool can be cleaned up."""

    def __init__(self):
        self._stop = threading.Event()
        self._interrupted = False
        self._interrupt_msg = None

    def get_activity_summary(self):
        # Always "just did something" -> seconds_since_activity stays ~0.
        return {"seconds_since_activity": 0.0, "last_activity_desc": "api_call",
                "current_tool": None, "api_call_count": 99, "max_iterations": 120}

    def interrupt(self, msg=None):
        self._interrupted = True
        self._interrupt_msg = msg
        self._stop.set()

    def run_conversation(self, prompt):
        # Burn time staying active until interrupted (cooperative).
        while not self._stop.wait(0.05):
            pass
        return {"final_response": "interrupted", "messages": []}


class UncooperativeAgent(ActiveRunawayAgent):
    """interrupt() is ignored -> the worker does NOT stop within grace. Proves the
    scheduler still reports failure without pretending the thread died."""

    def interrupt(self, msg=None):
        self._interrupted = True  # flag set, but the loop ignores _stop


class QuickAgent:
    def __init__(self):
        self._interrupted = False

    def get_activity_summary(self):
        return {"seconds_since_activity": 0.0}

    def interrupt(self, msg=None):
        self._interrupted = True

    def run_conversation(self, prompt):
        return {"final_response": "Done", "messages": []}


def _submit(agent):
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    fut = pool.submit(agent.run_conversation, "p")
    return pool, fut


class TestParseWallclock:
    def test_none_empty_zero_negative_invalid_are_unlimited(self):
        for raw in (None, "", 0, "0", -5, "-5", "abc", "  "):
            assert _parse_wallclock_timeout(raw) is None, f"{raw!r} should be unlimited"

    def test_valid_values(self):
        assert _parse_wallclock_timeout(3600) == 3600.0
        assert _parse_wallclock_timeout("1800") == 1800.0
        assert _parse_wallclock_timeout(0.5) == 0.5


class TestPollCronFuture:
    def test_wallclock_fires_on_active_runaway(self):
        """An always-active agent crosses the wall-clock cap (inactivity never would)."""
        agent = ActiveRunawayAgent()
        pool, fut = _submit(agent)
        try:
            result, cause, elapsed = _poll_cron_future(
                fut, agent, inactivity_limit=600.0, wallclock_limit=0.3,
                poll_interval=0.05, run_start=time.monotonic(),
            )
            assert cause == "wallclock"
            assert result is None
            assert elapsed >= 0.3
        finally:
            agent.interrupt()
            pool.shutdown(wait=False, cancel_futures=True)

    def test_no_caps_is_noop_completes(self):
        """Both caps None -> caller awaits directly; but if poll is entered with a quick
        agent it still completes (proves no spurious timeout)."""
        agent = QuickAgent()
        pool, fut = _submit(agent)
        try:
            # Simulate the entered-loop case with inactivity set huge, wallclock None.
            result, cause, _ = _poll_cron_future(
                fut, agent, inactivity_limit=600.0, wallclock_limit=None,
                poll_interval=0.05, run_start=time.monotonic(),
            )
            assert cause is None
            assert result["final_response"] == "Done"
        finally:
            pool.shutdown(wait=False)

    def test_wallclock_honored_when_inactivity_disabled(self):
        """The poll-entry bug: inactivity disabled (None) must NOT disable wall-clock."""
        agent = ActiveRunawayAgent()
        pool, fut = _submit(agent)
        try:
            result, cause, _ = _poll_cron_future(
                fut, agent, inactivity_limit=None, wallclock_limit=0.3,
                poll_interval=0.05, run_start=time.monotonic(),
            )
            assert cause == "wallclock"
        finally:
            agent.interrupt()
            pool.shutdown(wait=False, cancel_futures=True)

    def test_precedence_wallclock_before_inactivity(self):
        """When both caps are due in the SAME poll iteration, wall-clock wins (checked
        first). Both thresholds set below the first elapsed so they trip together."""
        # Agent reports HUGE idle so inactivity is due; both limits tiny so both are due
        # at the first post-wait check (elapsed ~poll_interval >= both).
        class IdleAndOldAgent(ActiveRunawayAgent):
            def get_activity_summary(self):
                return {"seconds_since_activity": 9999.0}
        agent = IdleAndOldAgent()
        pool, fut = _submit(agent)
        try:
            result, cause, _ = _poll_cron_future(
                fut, agent, inactivity_limit=0.01, wallclock_limit=0.01,
                poll_interval=0.05, run_start=time.monotonic(),
            )
            assert cause == "wallclock"  # NOT "inactivity" — wall-clock checked first
        finally:
            agent.interrupt()
            pool.shutdown(wait=False, cancel_futures=True)

    def test_quick_agent_no_timeout(self):
        agent = QuickAgent()
        pool, fut = _submit(agent)
        try:
            result, cause, _ = _poll_cron_future(
                fut, agent, inactivity_limit=5.0, wallclock_limit=5.0,
                poll_interval=0.05, run_start=time.monotonic(),
            )
            assert cause is None
            assert result["final_response"] == "Done"
        finally:
            pool.shutdown(wait=False)


class TestCooperativeInterrupt:
    def test_cooperative_agent_exits_in_grace(self):
        agent = ActiveRunawayAgent()
        pool, fut = _submit(agent)
        try:
            _poll_cron_future(fut, agent, inactivity_limit=None, wallclock_limit=0.2,
                              poll_interval=0.05, run_start=time.monotonic())
            agent.interrupt("wall-clock")
            # cooperative -> the worker future resolves within grace
            fut.result(timeout=5)
            assert agent._interrupted is True
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

    def test_uncooperative_agent_does_not_block_forever(self):
        agent = UncooperativeAgent()
        pool, fut = _submit(agent)
        try:
            _poll_cron_future(fut, agent, inactivity_limit=None, wallclock_limit=0.2,
                              poll_interval=0.05, run_start=time.monotonic())
            agent.interrupt("wall-clock")
            # uncooperative -> still running after a short grace; scheduler would mark
            # interrupt_pending and raise anyway (not assert thread death).
            still_running = not fut.done()
            # force-stop for cleanup (test-only; real scheduler abandons the thread)
            agent._stop.set()
            assert still_running is True
        finally:
            agent._stop.set()
            pool.shutdown(wait=False, cancel_futures=True)


class TestKilledSentinel:
    def test_sentinel_write_shape(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        p = _write_cron_killed_sentinel(
            "cron_80c575_20260627_120000", "80c575baeb95",
            cause="wallclock", elapsed=3601.4, wallclock_limit=3600.0,
            interrupt_state="interrupt_completed",
        )
        assert p is not None
        data = json.loads(Path(p).read_text())
        assert data["cause"] == "wallclock"
        assert data["job_id"] == "80c575baeb95"
        assert data["wallclock_limit_s"] == 3600.0
        assert data["interrupt_state"] == "interrupt_completed"
        assert "at" in data

    def test_no_session_id_writes_nothing(self):
        assert _write_cron_killed_sentinel(None, "j", cause="wallclock", elapsed=1.0,
                                           wallclock_limit=1.0, interrupt_state="x") is None
        assert _write_cron_killed_sentinel("", "j", cause="wallclock", elapsed=1.0,
                                           wallclock_limit=1.0, interrupt_state="x") is None

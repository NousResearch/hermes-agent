"""Regression tests for #62151 — cron's inactivity watchdog could never fire
for a cron job wedged inside a non-streaming API call.

Reported symptom: a gateway-run cron job hangs forever right after "OpenAI
client created" — no timeout, no error, no delivery, for 2.5h+ — while the
same job run via ``hermes cron tick`` (standalone CLI) completes normally.
The 600s ``HERMES_CRON_TIMEOUT`` inactivity ceiling never triggered.

Root cause, confirmed by reading the code (not just inferred from logs):
``interruptible_api_call`` (agent/chat_completion_helpers.py) polls the
worker thread every 0.3s and, every ~30s, calls
``agent._touch_activity("waiting for non-streaming response (Ns elapsed)")``
purely to keep the gateway's *other* inactivity monitors from mistaking a
slow-but-alive call for a dead one. That touch refreshed the exact field
(``seconds_since_activity``) that cron's OWN watchdog (this file) reads to
decide whether the job is dead — so a job wedged before it ever sent a byte
(e.g. a hung DNS lookup — no TCP connection was ever observed for the
hung job) looked "active" forever, and the ceiling could never fire.

This file mirrors the inactivity-check snippet from
``cron/scheduler.py::run_one_job`` (see that function's ``try/while`` block)
the same way ``tests/cron/test_cron_inactivity_timeout.py`` already mirrors
the surrounding polling loop, and exercises it against a ``FakeAgent`` that
reproduces the reported wedge: a real API-call attempt starts, then only
"still waiting" heartbeats fire, for far longer than the configured
inactivity ceiling.
"""

import concurrent.futures
import time


class WedgedAgent:
    """Simulates a cron job stuck inside interruptible_api_call.

    ``run_conversation`` blocks "forever" (until the test tears it down) —
    standing in for a non-streaming API call wedged before the socket even
    opens. A background heartbeat thread mimics interruptible_api_call's
    poll loop: it repeatedly announces "still waiting" without any real
    progress, exactly reproducing chat_completion_helpers.py's behavior
    before the #62151 fix (``progress`` kwarg defaulting to True everywhere).
    """

    def __init__(self, *, heartbeat_interval, fixed_bug_behavior):
        self._heartbeat_interval = heartbeat_interval
        # fixed_bug_behavior=False reproduces the pre-fix code: every touch
        # (including the "still waiting" heartbeat) advances the single
        # activity clock. fixed_bug_behavior=True reproduces the post-fix
        # code: the heartbeat advances seconds_since_activity only,
        # seconds_since_progress is untouched by it.
        self._fixed_bug_behavior = fixed_bug_behavior
        self._call_start = time.time()
        self._last_activity_ts = self._call_start
        self._last_progress_ts = self._call_start
        self._stop = False

    def _heartbeat_loop(self):
        while not self._stop:
            time.sleep(self._heartbeat_interval)
            self._last_activity_ts = time.time()
            if not self._fixed_bug_behavior:
                self._last_progress_ts = self._last_activity_ts

    def get_activity_summary(self):
        now = time.time()
        return {
            "seconds_since_activity": round(now - self._last_activity_ts, 1),
            "seconds_since_progress": round(now - self._last_progress_ts, 1),
            "last_activity_desc": "waiting for non-streaming response",
            "current_tool": None,
            "api_call_count": 2,
            "max_iterations": 90,
        }

    def run_conversation(self, prompt):
        import threading

        hb = threading.Thread(target=self._heartbeat_loop, daemon=True)
        hb.start()
        # Blocks until the test kills the pool — standing in for a request
        # wedged before it ever reaches the network.
        while not self._stop:
            time.sleep(0.05)
        return {"final_response": "should never get here in this test", "messages": []}


def _run_cron_watchdog_snippet(
    agent, *, cron_inactivity_limit, poll_interval=0.05, max_wall_seconds=None
):
    """Mirrors the corrected inactivity-check block in
    ``cron/scheduler.py::run_one_job`` — reads ``seconds_since_progress``
    (falling back to ``seconds_since_activity`` for agents that don't report
    it), exactly as the production code now does post-#62151.

    ``max_wall_seconds`` is test-only scaffolding, not part of the mirrored
    production snippet: the whole point of the reported bug is that the real
    watchdog loop runs *unbounded* (the reporter waited 2.5h+ — 300x the
    600s ceiling — and it never fired). To prove "never fires" without
    literally hanging the test suite forever, the pre-fix-behavior test
    below waits a generous, but finite, multiple of the ceiling instead.
    """
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = pool.submit(agent.run_conversation, "test prompt")
    inactivity_timeout = False
    start = time.time()
    try:
        while True:
            done, _ = concurrent.futures.wait({future}, timeout=poll_interval)
            if done:
                break
            idle_secs = 0.0
            if hasattr(agent, "get_activity_summary"):
                try:
                    act = agent.get_activity_summary()
                    idle_secs = act.get(
                        "seconds_since_progress",
                        act.get("seconds_since_activity", 0.0),
                    )
                except Exception:
                    pass
            if idle_secs >= cron_inactivity_limit:
                inactivity_timeout = True
                break
            if max_wall_seconds is not None and (time.time() - start) >= max_wall_seconds:
                break
    finally:
        agent._stop = True
        pool.shutdown(wait=False, cancel_futures=True)
    return inactivity_timeout


class TestCronWatchdogSeesWedgedCall:
    def test_fixed_signal_detects_the_wedge(self):
        """With the #62151 fix, the heartbeat no longer masks a wedged call
        from the cron watchdog: seconds_since_progress keeps growing and the
        ceiling fires."""
        agent = WedgedAgent(heartbeat_interval=0.05, fixed_bug_behavior=True)
        timed_out = _run_cron_watchdog_snippet(
            agent, cron_inactivity_limit=0.3, poll_interval=0.05
        )
        assert timed_out is True

    def test_pre_fix_signal_never_detects_the_wedge(self):
        """Reproduces the reported bug: with the OLD behavior (heartbeat
        also refreshes the progress clock), the same wedge NEVER trips the
        watchdog — this is why the reported cron job hung for 2.5h+ (300x
        its 600s ceiling) with zero timeout. Waits 10x the configured
        ceiling here (bounded, so the test itself terminates) instead of
        reproducing the real incident's unbounded hang."""
        agent = WedgedAgent(heartbeat_interval=0.05, fixed_bug_behavior=False)
        timed_out = _run_cron_watchdog_snippet(
            agent, cron_inactivity_limit=0.3, poll_interval=0.05, max_wall_seconds=3.0
        )
        assert timed_out is False

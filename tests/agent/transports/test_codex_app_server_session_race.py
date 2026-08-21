"""Regression test — and exhaustive stress test — for the fix to the race
between CodexAppServerSession.run_turn() and CodexAppServerSession.close().

This is the concrete mechanism behind gateway/run.py's
``_session_expiry_watcher`` tearing a session down mid-turn (see that
function's own comment: "Fall back to _running_agents in case the agent is
still mid-turn when the expiry fires").

Before the fix, ``run_turn()`` re-read ``self._client`` on every iteration
of its polling while-loop (``self._client.is_alive()``,
``self._client.take_server_request(...)``,
``self._client.take_notification(...)``) with no null-guard and no lock,
while ``close()`` set ``self._client = None`` guarded only by
``self._active_turn_lock`` (which protects ``self._active_turn_id`` — not
``self._client``). A ``close()`` landing between two loop iterations made
the very next ``self._client.<method>()`` call raise
``AttributeError: 'NoneType' object has no attribute '...'``.

The fix: ``run_turn()``/``compact_thread()`` now snapshot
``client = self._client`` once and use that local reference for the rest of
the call. ``close()`` still nulls ``self._client`` and kills the
subprocess, but the *local* snapshot stays a valid (now-dead) object, so
the loop's own existing "subprocess exited unexpectedly" dead-process
detection takes over instead of crashing.

Uses a fake in-process client (no subprocess) so it is cheap to run
repeatedly (including the exhaustive/soak variant below) without taxing
the host machine.
"""

from __future__ import annotations

import threading
import time

from agent.transports.codex_app_server_session import CodexAppServerSession


class _RacyFakeClient:
    """Stand-in for CodexAppServerClient with a controllable window
    between is_alive() calls, wide enough for a concurrent close() to
    land in the middle of run_turn()'s poll loop."""

    def __init__(self, ready_for_close: threading.Event, hold: float = 0.05) -> None:
        self._ready_for_close = ready_for_close
        self._hold = hold
        self.alive_calls = 0
        self.closed = False

    def initialize(self, **kwargs):
        return {
            "userAgent": "fake",
            "codexHome": "",
            "platformOs": "test",
            "platformFamily": "test",
        }

    def close(self):
        self.closed = True

    def request(self, method, params=None, timeout=30.0):
        if method == "thread/start":
            return {"thread": {"id": "thread-racy"}}
        if method == "turn/start":
            return {"turn": {"id": "turn-racy"}}
        return {}

    def is_alive(self) -> bool:
        self.alive_calls += 1
        # Signal the closing thread it's safe to race in, then hold the
        # window open so close() has time to run before we check self.closed
        # — mirrors the real CodexAppServerClient, whose is_alive() checks
        # self._proc.poll() and only turns False once close() has actually
        # killed the subprocess.
        self._ready_for_close.set()
        time.sleep(self._hold)
        return not self.closed

    def take_server_request(self, timeout: float = 0.0):
        return None

    def take_notification(self, timeout: float = 0.0):
        time.sleep(0.01)
        return None

    def stderr_tail(self, n: int = 20):
        return []


def _run_turn_vs_close_once(hold: float = 0.02) -> dict:
    """Run one instance of the race: start run_turn() in a thread, close()
    the session from the main thread as soon as run_turn() enters its poll
    loop, and report what happened."""
    ready_for_close = threading.Event()
    fake = _RacyFakeClient(ready_for_close, hold=hold)
    session = CodexAppServerSession(cwd=".", client_factory=lambda **_kw: fake)

    outcome: dict[str, object] = {}

    def _run():
        try:
            outcome["result"] = session.run_turn(
                user_input="hello",
                turn_timeout=5,
                notification_poll_timeout=0.01,
            )
        except BaseException as exc:  # noqa: BLE001 - capturing for assertion
            outcome["error"] = exc

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    if not ready_for_close.wait(timeout=5):
        outcome["error"] = RuntimeError("run_turn never reached is_alive()")
        t.join(timeout=5)
        return outcome

    session.close()
    t.join(timeout=5)
    outcome["thread_alive"] = t.is_alive()
    outcome["client_closed"] = fake.closed
    return outcome


class TestRunTurnClientRace:
    """Proves the fix: session.close() racing session.run_turn() no longer
    crashes the turn. self._client is nulled out on the session, but
    run_turn() is holding its own local snapshot, so the loop's existing
    dead-subprocess detection takes over instead of raising."""

    def test_close_during_run_turn_returns_gracefully(self):
        outcome = _run_turn_vs_close_once()

        assert "error" not in outcome, f"run_turn() raised: {outcome.get('error')!r}"
        assert outcome["thread_alive"] is False, "run_turn thread never returned"
        assert outcome["client_closed"] is True

        result = outcome["result"]
        assert result.should_retire is True
        assert result.error and "unexpectedly" in result.error


class TestRunTurnClientRaceExhaustive:
    """Exhaustion / soak test requested alongside the fix: a single lucky
    pass doesn't prove a race is gone, so this repeats the race hundreds of
    times with jittered timing windows to shake out flakiness. Runs
    sequentially (never more than 2 live threads at once) and every
    session/thread is joined and dropped before the next iteration, so
    memory stays flat regardless of iteration count.
    """

    ITERATIONS = 300

    def test_no_attribute_error_across_many_races(self):
        import random

        rng = random.Random(1234)  # deterministic across CI runs
        failures: list[tuple[int, object]] = []

        for i in range(self.ITERATIONS):
            # Jitter the hold window across a range that puts close() both
            # comfortably before and right on top of the is_alive() check,
            # to probe the boundary rather than just one fixed timing.
            hold = rng.uniform(0.0, 0.03)
            outcome = _run_turn_vs_close_once(hold=hold)
            if "error" in outcome or outcome.get("thread_alive") is not False:
                failures.append((i, outcome.get("error") or outcome))

        assert not failures, (
            f"{len(failures)}/{self.ITERATIONS} iterations hit the race "
            f"(showing up to 5): {failures[:5]}"
        )

"""A bounded delivery turn is limited by IDLE time, not wall-clock (cron/AGENTS.md: "a long-but-active
job is never cut off"). The spawner of a ``hermes chat -Q`` delivery child (cron Bot Chat lane,
bot relay) reads an activity heartbeat the child refreshes beside its turn report; while the
turn keeps making progress the cap must not kill it, and a child that goes silent for the full
cap still is. A child that never heartbeats keeps the wall-clock cap (older children).
"""

import json
import os
import threading
import time
from types import SimpleNamespace

import pytest

from hermes_cli import quiet_single_query as qsq

CAP = 1.5  # seconds; the old wall-clock cap killed here


class _DeliveryChild:
    """Fake Popen: a quiet child whose turn runs until the test ends it. Starts the REAL
    heartbeat writer against an always-advancing fake agent, exactly as the quiet runner does.
    The test process impersonates the child, so its pid IS this process's pid: everything the
    child writes in-process (report, heartbeat) is stamped with it."""

    def __init__(self, argv, *, env=None, _writer_poll=0.15, **kwargs):
        self.argv = argv
        self.pid = os.getpid()
        self.returncode = None
        self.killed = False
        self._exit = threading.Event()
        report = env[qsq.TURN_REPORT_FILE_ENV]
        self.activity_path = qsq.activity_heartbeat_path(report)
        agent = SimpleNamespace(
            get_activity_summary=lambda: {"last_activity_ts": time.time(), "last_activity_desc": "api call"})
        self._heartbeat = qsq.start_turn_activity_heartbeat(
            agent, report, poll_seconds=_writer_poll)
        assert self._heartbeat is not None, "writer refused an agent with an activity clock"

    def communicate(self):
        self._exit.wait()
        return "done", ""

    def end_turn_cleanly(self, report_path):
        qsq.write_turn_report(report_path, exit_code=0, error="", reply="done")
        if self._heartbeat is not None:
            self._heartbeat.set()
        time.sleep(0.3)
        self.returncode = 0
        self._exit.set()

    def kill(self):
        self.killed = True
        self.returncode = -9
        if self._heartbeat is not None:
            self._heartbeat.set()
        self._exit.set()


def test_an_active_turn_is_not_killed_at_the_wall_cap(monkeypatch, tmp_path):
    """The discriminating invariant: a turn that keeps reporting progress (fresh heartbeats)
    must outlive the cap and be booked from its report. Red on a wall-clock spawner."""
    report = str(tmp_path / "turn.json")
    child_holder = {}
    spawned = threading.Event()
    started = time.monotonic()

    def spawn(argv, **kwargs):
        child_holder["proc"] = _DeliveryChild(argv, env=kwargs["env"])
        spawned.set()
        return child_holder["proc"]

    monkeypatch.setattr(qsq.subprocess, "Popen", spawn)

    def end_turn_once_clear_of_the_cap():
        # End the turn only well past the old kill point, and only once the child's
        # heartbeats are flowing (the spawner has had every chance to observe liveness).
        assert spawned.wait(timeout=10.0)
        while time.monotonic() - started < 2.0 * CAP:
            child = child_holder.get("proc")
            if child is not None and os.path.exists(child.activity_path):
                with open(child.activity_path, encoding="utf-8") as fh:
                    if json.load(fh).get("ts"):
                        break
            time.sleep(0.05)
        time.sleep(CAP)  # strictly past the cap, while still active
        child_holder["proc"].end_turn_cleanly(report)

    ender = threading.Thread(target=end_turn_once_clear_of_the_cap, daemon=True)
    ender.start()

    result = qsq.run_reported_turn(["hermes"], env={}, report_path=report, timeout=CAP)

    child = child_holder["proc"]
    assert not child.killed, "an actively reporting turn was killed at the wall cap"
    assert result.returncode == 0
    ender.join(timeout=5.0)


class _SilentChild:
    """A pre-heartbeat child (or one wedged before its first activity): no heartbeat file, no
    report — pure silence from spawn until killed."""

    pid = 8383

    def __init__(self, argv, **kwargs):
        self.argv = argv
        self.returncode = None
        self.killed = False
        self._exit = threading.Event()

    def communicate(self):
        self._exit.wait()
        return "", ""

    def kill(self):
        self.killed = True
        self.returncode = -9
        self._exit.set()


def test_a_silent_child_is_still_killed_at_the_cap(monkeypatch, tmp_path):
    """The counterweight: no heartbeat, no report — the spawner must still hard-interrupt at
    the cap (a stalled child must not monopolise the delivery lane), proving the idle bound
    did not become 'never kill'."""
    report = str(tmp_path / "turn.json")
    child_holder = {}

    def spawn(argv, **kwargs):
        child_holder["proc"] = _SilentChild(argv)
        return child_holder["proc"]

    monkeypatch.setattr(qsq.subprocess, "Popen", spawn)

    outcome = {}

    def run():
        try:
            qsq.run_reported_turn(["hermes"], env={}, report_path=report, timeout=CAP)
            outcome["result"] = "returned"
        except TimeoutExpired:
            outcome["result"] = "timeout"

    from subprocess import TimeoutExpired
    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(timeout=4.0 * CAP)
    assert outcome.get("result") == "timeout", (
        f"a silent child must be interrupted at the cap (got {outcome.get('result')!r})")
    assert child_holder["proc"].killed

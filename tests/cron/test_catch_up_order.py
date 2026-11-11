"""With cron.catch_up_serial a catch-up replays missed cron slots one at a time, in slot order."""

import threading

import pytest

from hermes_constants import get_hermes_home
from tests.waiting import wait_until


def _job(name, kind, slot, schedule="cron"):
    return {
        "id": name,
        "name": name,
        "prompt": "test",
        "schedule": {"kind": schedule, "expr": "0 4 * * *"} if schedule == "cron" else {"kind": schedule},
        "enabled": True,
        "deliver": "local",
        "_scheduled_instant": slot,
        "last_dispatch": {"kind": kind, "scheduled_at": slot},
    }


def _enable_serial_catch_up():
    (get_hermes_home() / "config.yaml").write_text("cron:\n  catch_up_serial: true\n", encoding="utf-8")


class _Runs:
    """Records job starts; every job blocks until the test releases it."""

    def __init__(self):
        self.started, self._release, self._lock = [], {}, threading.Lock()

    def gate(self, name):
        with self._lock:
            return self._release.setdefault(name, threading.Event())

    def process(self, job, *_a, **_kw):
        with self._lock:
            self.started.append(job["name"])
        self.gate(job["name"]).wait(timeout=10)
        return True

    def has_started(self, *names):
        with self._lock:
            return set(names) <= set(self.started)

    def release(self, name):
        self.gate(name).set()

    def release_all(self):
        with self._lock:
            gates = list(self._release.values())
        for gate in gates:
            gate.set()


@pytest.fixture
def sched(monkeypatch):
    import cron.scheduler as sched

    sched._shutdown_parallel_pool()
    sched._running_job_ids.clear()
    runs = _Runs()
    monkeypatch.setattr(sched, "advance_next_runs", lambda *_a, **_kw: 0)
    monkeypatch.setattr(sched, "_process_due_job", runs.process)
    yield sched, runs
    runs.release_all()
    sched._shutdown_parallel_pool()
    sched._running_job_ids.clear()


def _replay_in_order(runs, names):
    """Release each job once it started; the next one may start only after that."""
    for index, name in enumerate(names):
        wait_until(lambda: runs.has_started(name), what=f"{name} started")
        assert [n for n in runs.started if n in names] == names[: index + 1]
        runs.release(name)


def test_missed_cron_slots_replay_serially_in_slot_order(sched, monkeypatch):
    sched, runs = sched
    _enable_serial_catch_up()
    due = [
        _job("coordinator", "late", "2026-11-11T04:35:00+00:00"),
        _job("planner", "catch_up", "2026-11-11T04:20:00+00:00"),
        _job("review-b", "catch_up", "2026-11-11T04:25:00+00:00"),
        _job("review-a", "catch_up", "2026-11-11T04:25:00+00:00"),
        _job("probe", "catch_up", "2026-11-11T07:50:00+00:00", schedule="interval"),
    ]
    monkeypatch.setattr(sched, "get_due_jobs", lambda: due)

    sched.tick(verbose=False, sync=False)

    # The interval job runs beside the replay instead of queueing behind it.
    wait_until(lambda: runs.has_started("planner", "probe"), what="planner and probe started")
    _replay_in_order(runs, ["planner", "review-a", "review-b", "coordinator"])


def test_cron_slot_due_while_draining_queues_behind_the_replay(sched, monkeypatch):
    sched, runs = sched
    _enable_serial_catch_up()
    ticks = iter([
        [_job("planner", "catch_up", "2026-11-11T04:20:00+00:00"),
         _job("review", "catch_up", "2026-11-11T04:25:00+00:00")],
        [_job("briefing", "on_time", "2026-11-11T08:00:00+00:00")],
    ])
    monkeypatch.setattr(sched, "get_due_jobs", lambda: next(ticks))

    sched.tick(verbose=False, sync=False)
    wait_until(lambda: runs.has_started("planner"), what="planner started")
    sched.tick(verbose=False, sync=False)

    _replay_in_order(runs, ["planner", "review", "briefing"])


@pytest.mark.parametrize(("serial", "kinds"), [
    (False, ("catch_up", "catch_up")),
    (True, ("catch_up", "on_time")),
    (True, ("on_time", "on_time")),
])
def test_other_ticks_keep_the_parallel_pool(sched, monkeypatch, serial, kinds):
    sched, runs = sched
    if serial:
        _enable_serial_catch_up()
    due = [_job(f"job-{i}", kind, "2026-11-11T04:00:00+00:00") for i, kind in enumerate(kinds)]
    monkeypatch.setattr(sched, "get_due_jobs", lambda: due)

    sched.tick(verbose=False, sync=False)

    wait_until(lambda: runs.has_started("job-0", "job-1"), what="both jobs running at once")
    assert not sched._catch_up_lanes

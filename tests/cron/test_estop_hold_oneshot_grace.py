"""A one-shot that comes due while ``hermes pause`` holds cron dispatch fires once after resume.

``hermes pause`` (ESTOP) and a gateway drain hold dispatch the same way, but only the drain
stamped ``dispatch_held.json``: the ticker calls ``note_held`` just for ``can_dispatch`` false.
An ESTOP-held tick called ``note_dispatching`` and then returned before the due scan, so the
one-shot grace kept counting from the due time. A reminder due during a four-minute pause — or
a job the operator ran by hand a minute before engaging it — was retired as ``missed`` by the
first scan after ``hermes resume`` and never ran (#153683). The pause now stamps the hold, so
that grace counts from the end of the pause, exactly like #147894's drain. A hold that outlasts
``MAX_EXTENSION_SECONDS`` still retires the one-shot (#93526: no stale reminders).
"""

import threading
import time
from datetime import UTC, datetime, timedelta

import pytest

from agent import estop
from cron.dispatch_hold import MAX_EXTENSION_SECONDS, read
from cron.jobs import create_job, get_due_jobs, load_jobs, save_jobs, trigger_job
from cron.scheduler_provider import InProcessCronScheduler


@pytest.fixture()
def cron_store(tmp_path, monkeypatch):
    """An empty cron store and an ESTOP sentinel inside a temp HERMES_HOME."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    (tmp_path / "cron").mkdir()
    estop._logged_components.clear()
    yield tmp_path
    estop.disengage()
    estop._logged_components.clear()


@pytest.fixture()
def clock(tmp_path, monkeypatch):
    """The due scan's clock and ``dispatch_hold``'s, driven together by ``advance``."""
    state = {"at": datetime.fromtimestamp(time.time(), UTC)}

    class FakeTime:
        def time(self):
            return state["at"].timestamp()

    monkeypatch.setattr("cron.jobs._hermes_now", lambda: state["at"])
    monkeypatch.setattr("cron.dispatch_hold.time", FakeTime())
    return state


def _oneshot(jid, run_at):
    return {
        "id": jid, "name": jid, "prompt": "x",
        "schedule": {"kind": "once", "run_at": run_at.isoformat()},
        "next_run_at": run_at.isoformat(), "last_run_at": None,
        "enabled": True, "state": "scheduled", "repeat": {"times": 1, "completed": 0},
        "deliver": "local", "run_claim": None, "fire_claim": None,
    }


def _paused_ticker_cycle():
    """One cycle of the real in-process ticker while ESTOP — not a drain — holds dispatch."""
    stop = threading.Event()

    def dispensing():
        stop.set()
        return True

    InProcessCronScheduler().start(stop, interval=1, can_dispatch=dispensing)


def _paused_ticks(clock, start, seconds):
    """The pause's own ticker: one held cycle per interval from *start* to *seconds* after it."""
    for elapsed in range(0, seconds + 1, 60):
        clock["at"] = start + timedelta(seconds=elapsed)
        _paused_ticker_cycle()


def test_a_paused_tick_stamps_the_dispatch_hold(cron_store, clock):
    """The ESTOP-held tick records the hold, so ``grace_starts_at`` has something to extend."""
    estop.engage(reason="deploy window")
    _paused_ticker_cycle()

    held = read(cron_store / "cron")
    assert held is not None, "the pause held dispatch without stamping dispatch_held.json"
    assert held["last"] >= held["since"]


def test_oneshot_due_during_a_pause_fires_once_after_resume(cron_store, clock):
    start = clock["at"]
    estop.engage(reason="deploy window")
    _paused_ticks(clock, start, 240)
    estop.disengage()
    # The first scan after `hermes resume`: 250s past a due time only 120s of grace covers.
    clock["at"] = start + timedelta(seconds=250)
    save_jobs([_oneshot("reminder", start - timedelta(seconds=60))])

    assert [d["id"] for d in get_due_jobs()] == ["reminder"]
    assert [j["id"] for j in load_jobs()] == ["reminder"]


def test_a_manual_run_before_the_pause_is_not_retired(cron_store, clock):
    """``hermes cron run`` right before the pause is an explicit request, not a stale schedule."""
    start = clock["at"]
    jid = create_job(prompt="remind me", schedule=(start + timedelta(minutes=30)).isoformat(),
                     name="run by hand", deliver="local")["id"]
    trigger_job(jid)
    estop.engage(reason="deploy window")
    _paused_ticks(clock, start, 240)
    estop.disengage()
    clock["at"] = start + timedelta(seconds=250)

    assert [d["id"] for d in get_due_jobs()] == [jid]
    assert [j["id"] for j in load_jobs()] == [jid]


def test_a_pause_longer_than_the_extension_still_retires(cron_store, clock):
    """The cap is the drain's: a reminder due far into a long pause is stale after resume."""
    start = clock["at"]
    hold = start + timedelta(seconds=MAX_EXTENSION_SECONDS + 600)
    estop.engage(reason="overnight window")
    _paused_ticks(clock, hold, 240)
    estop.disengage()
    clock["at"] = hold + timedelta(seconds=250)
    save_jobs([_oneshot("reminder", start)])

    assert get_due_jobs() == []
    assert load_jobs() == []

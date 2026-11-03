"""A one-shot that comes due while a gateway drain holds cron dispatch fires once afterwards.

A restart or ``hermes update`` drain waits on in-flight turns with the ticker's dispatch held
(``can_dispatch`` false). The one-shot grace (120s) must count from the end of that hold, not
from the due time: an update that waited three minutes on a Kanban turn retired the reminder
the user scheduled for that minute as ``missed`` without ever running it. A hold whose gateway
never came back, or a reminder due hours into an endless external drain, is still retired (#93526).
"""

import threading
import time
from datetime import datetime, timedelta, UTC

import pytest

from cron.dispatch_hold import note_held
from cron.jobs import get_due_jobs, load_jobs, save_jobs
from cron.scheduler_provider import InProcessCronScheduler


@pytest.fixture()
def cron_store(tmp_path, monkeypatch):
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    (tmp_path / "cron").mkdir()
    return tmp_path


def _oneshot(jid, run_at):
    return {
        "id": jid, "name": jid, "prompt": "x",
        "schedule": {"kind": "once", "run_at": run_at.isoformat()},
        "next_run_at": run_at.isoformat(), "last_run_at": None,
        "enabled": True, "state": "scheduled", "repeat": {"times": 1, "completed": 0},
        "deliver": "local", "run_claim": None, "fire_claim": None,
    }


def _scan_at(monkeypatch, when):
    monkeypatch.setattr("cron.jobs._hermes_now", lambda: when)
    return [d["id"] for d in get_due_jobs()]


def _held_ticker_cycle():
    """One cycle of the real in-process ticker while the gateway drains."""
    stop = threading.Event()

    def draining():
        stop.set()
        return False

    InProcessCronScheduler().start(stop, interval=1, can_dispatch=draining)


def test_oneshot_due_during_a_drain_fires_once_the_next_gateway_ticks(cron_store, monkeypatch):
    t0 = time.time()
    note_held(now=t0 - 100)  # the drain began holding dispatch 100s ago ...
    _held_ticker_cycle()     # ... and the ticker is still holding it now
    start = datetime.fromtimestamp(t0, UTC)
    save_jobs([_oneshot("reminder", start - timedelta(seconds=90))])

    # The next gateway's first tick lands 190s after the due time, past the 120s grace.
    assert _scan_at(monkeypatch, start + timedelta(seconds=100)) == ["reminder"]
    assert [j["id"] for j in load_jobs()] == ["reminder"]


@pytest.mark.parametrize("hold_ticks_ago,due_ago", [
    ((3 * 3600 + 60, 3 * 3600), 3 * 3600 + 30),  # the drain's gateway never came back
    (tuple(range(5 * 3600, 89, -60)), 5 * 3600),  # a live hold (external drain) ticking for hours
], ids=["gateway-never-returned", "hours-long-live-hold"])
def test_stale_oneshot_is_retired_despite_a_hold(cron_store, monkeypatch, hold_ticks_ago, due_ago):
    t0 = time.time()
    for ago in hold_ticks_ago:
        note_held(now=t0 - ago)
    start = datetime.fromtimestamp(t0, UTC)
    save_jobs([_oneshot("reminder", start - timedelta(seconds=due_ago))])

    assert _scan_at(monkeypatch, start) == []
    assert load_jobs() == []

"""An auto-started (``--idle-exit``) gateway exits once idle; anything it serves keeps it up.

Real SQLite ledger rows, a real cron store and real asyncio; the runner is a stand-in exposing
only what the idle predicate reads.
"""
from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest

from gateway import run_idle_exit
from gateway.config import Platform
from gateway.runtime_bootstrap import TicketStore
from gateway.session_authorities import SessionAuthorities
from hermes_state import SessionDB
from hermes_state_runtime import admit_session_input, begin_runtime_epoch, cancel_session_input


def _runner(tmp_path, monkeypatch, *, window=0.2):
    home = tmp_path / "home"
    home.mkdir()
    db = SessionDB(db_path=home / "state.db")
    registry = SessionAuthorities(home)
    registry.add(home, SimpleNamespace(profile_id=str(home), db=db))
    stops = []

    async def stop():
        stops.append(True)
        runner._running = False
    runner = SimpleNamespace(
        _running=True, _draining=False, session_authorities=registry, adapters={Platform.LOCAL: object()},
        _profile_adapters={}, _profile_failed_platforms={}, _failed_platforms={},
        config=SimpleNamespace(platforms={}), session_ticket_store=TicketStore("i", frozenset({str(home)})),
        session_runtime_descriptor={"state": "ready", "capabilities": ["x"]}, _exit_reason=None,
        _active_work_count=lambda: 0, _scale_to_zero_has_live_background_work=lambda: False, stop=stop)
    monkeypatch.setattr(run_idle_exit, "idle_exit_seconds", lambda r: window)
    monkeypatch.setattr("gateway.run._cron_tick_profile_homes", lambda config: [("default", home)])
    monkeypatch.setattr(run_idle_exit, "_kanban_busy", lambda: None)
    monkeypatch.setattr(run_idle_exit, "_store_gates_busy", lambda homes: None)
    run_idle_exit.note_client_activity(runner)
    runner._idle_exit_last_activity -= 10
    return runner, home, db, stops


@pytest.mark.asyncio
async def test_idle_auto_started_gateway_exits_and_publishes_why(tmp_path, monkeypatch):
    runner, _home, db, stops = _runner(tmp_path, monkeypatch)
    try:
        await asyncio.wait_for(run_idle_exit.unmanaged_idle_exit_watcher(runner), 10)
        await runner._idle_exit_stop_task
        assert stops == [True]
        assert runner.session_runtime_descriptor["state"] == "draining"
        assert runner.session_runtime_descriptor["drain_reason"] == run_idle_exit.IDLE_EXIT_REASON
    finally:
        db.close()


@pytest.mark.asyncio
async def test_queued_work_clients_adapters_and_cron_each_keep_it_up(tmp_path, monkeypatch):
    """Fail-closed predicate: every class of pending work, a live client and a configured adapter
    keep the gateway up; only when all are gone is it idle."""
    runner, home, db, _stops = _runner(tmp_path, monkeypatch)
    try:
        assert run_idle_exit.busy_reason(runner, window=0.2) is None
        db.create_session("s", source="cli")
        epoch = begin_runtime_epoch(db, instance_id="owner")
        row = admit_session_input(db, epoch=epoch, principal_id="p", session_id="s", request_id="r",
                                  payload={"text": "queued while idle"})
        assert "admissions pending" in run_idle_exit.busy_reason(runner, window=0.2)
        cancel_session_input(db, epoch=epoch, admission_id=row["admission_id"])
        with run_idle_exit.attached_client(runner):
            assert "client(s) attached" in run_idle_exit.busy_reason(runner, window=0.2)
        assert "client activity" in run_idle_exit.busy_reason(runner, window=60)
        runner._idle_exit_last_activity -= 120
        runner.session_ticket_store.mint(profile_id=str(home), subject="u", purpose="interactive")
        assert run_idle_exit.busy_reason(runner, window=0.2) == "client attaching"
        runner.session_ticket_store.revoke()
        runner.config.platforms[Platform.TELEGRAM] = SimpleNamespace(enabled=True)
        assert "messaging" in run_idle_exit.busy_reason(runner, window=0.2)
        runner.config.platforms.clear()
        from cron.jobs import create_job, use_cron_store
        with use_cron_store(home):
            create_job("ping", "every 1h", name="j")
        assert "cron jobs scheduled" in run_idle_exit.busy_reason(runner, window=0.2)
        monkeypatch.setattr("cron.jobs.load_jobs", lambda: (_ for _ in ()).throw(OSError("unreadable")))
        assert "idle probe failed" in run_idle_exit.busy_reason(runner, window=0.2)
    finally:
        db.close()


def test_a_profile_still_starting_leaves_the_served_ledgers_readable(tmp_path, monkeypatch):
    """Hot-serve reserves a starting profile's registry slot before its authority exists. The probe
    still reads every built profile's ledger meanwhile, instead of failing on the slot."""
    runner, _home, db, _stops = _runner(tmp_path, monkeypatch)
    starting = tmp_path / "starting"
    starting.mkdir()
    runner.session_authorities.add(starting, None, name="starting")  # as serve_profile_runtime does
    try:
        db.create_session("s", source="cli")
        epoch = begin_runtime_epoch(db, instance_id="owner")
        admit_session_input(db, epoch=epoch, principal_id="p", session_id="s", request_id="r",
                            payload={"text": "queued while another profile starts"})
        assert "admissions pending" in run_idle_exit.busy_reason(runner, window=0.2)
    finally:
        db.close()


def test_secondary_local_session_adapter_does_not_count_as_messaging(tmp_path, monkeypatch):
    """A secondary local session leaves an adapter map behind after the client disconnects."""
    runner, _home, db, _stops = _runner(tmp_path, monkeypatch)
    secondary_home = tmp_path / "secondary"
    secondary_home.mkdir()
    secondary_db = SessionDB(db_path=secondary_home / "state.db")
    runner.session_authorities.add(
        secondary_home, SimpleNamespace(profile_id=str(secondary_home), db=secondary_db), name="secondary")
    try:
        runner._profile_adapters["secondary"] = {Platform.LOCAL: object()}
        assert run_idle_exit.busy_reason(runner, window=0.2) is None

        runner._profile_adapters["secondary"][Platform.TELEGRAM] = object()
        assert "telegram" in run_idle_exit.busy_reason(runner, window=0.2)
    finally:
        secondary_db.close()
        db.close()

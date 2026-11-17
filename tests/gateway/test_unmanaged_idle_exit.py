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


@pytest.mark.asyncio
async def test_launch_home_deleted_underneath_does_not_hold_it_up(tmp_path, monkeypatch):
    """The client deletes its throwaway home while the gateway it auto-started is still running. The
    deleted-WAL guard then refuses every new SessionDB for that home, so the store gates read busy
    forever, and its queued rows can never run: neither may hold the gateway up."""
    import os
    real_store_gates = run_idle_exit._store_gates_busy
    runner, home, db, stops = _runner(tmp_path, monkeypatch)
    monkeypatch.setattr(run_idle_exit, "_store_gates_busy", real_store_gates)
    try:
        db.create_session("s", source="cli")
        epoch = begin_runtime_epoch(db, instance_id="owner")
        admit_session_input(db, epoch=epoch, principal_id="p", session_id="s", request_id="r",
                            payload={"text": "queued before teardown"})
        for sidecar in ("state.db-wal", "state.db-shm"):
            os.unlink(home / sidecar)
        await asyncio.wait_for(run_idle_exit.unmanaged_idle_exit_watcher(runner), 5)
        await runner._idle_exit_stop_task
        assert stops == [True]
    finally:
        db.close()


@pytest.mark.requires_wal  # a deleted generation exists only for a recorded WAL/SHM pair
def test_retired_home_probe_never_poisons_a_cleanly_closing_store(tmp_path, monkeypatch):
    """The watcher's worker thread probes each store's generation while that store may be closing
    cleanly; the probe must see the settled state, never the teardown window."""
    import threading
    runner, _home, db, _stops = _runner(tmp_path, monkeypatch)
    conn, lock, close, guard = db._conn, db._lock, db._close_connection_quietly, db._raise_if_db_replaced
    closed, probing = threading.Event(), threading.Event()

    class ObservedLock:
        def __enter__(self):
            if threading.current_thread().name == "probe":
                probing.set()
            lock.acquire()

        def __exit__(self, *args):
            lock.release()

    def observed_guard():
        try:
            return guard()
        finally:
            probing.set()

    def paused_close(connection):
        close(connection)
        if connection is conn:  # WAL and SHM are gone; the generation identity is not cleared yet
            closed.set()
            probing.wait(10)

    monkeypatch.setattr(db, "_lock", ObservedLock())
    monkeypatch.setattr(db, "_close_connection_quietly", paused_close)
    monkeypatch.setattr(db, "_raise_if_db_replaced", observed_guard)
    retired = []
    closer = threading.Thread(target=db.close)
    prober = threading.Thread(target=lambda: retired.append(run_idle_exit._retired_homes(runner)), name="probe")
    closer.start()
    assert closed.wait(10)
    prober.start()
    prober.join(15)
    closer.join(15)
    assert not prober.is_alive() and not closer.is_alive()
    assert retired == [set()]
    assert not db._db_wal_generation_lost


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


def test_a_profile_still_starting_keeps_an_otherwise_idle_gateway_up(tmp_path, monkeypatch):
    """With every built ledger empty, a profile hot-serve is still building holds the gateway: its
    queued admissions are recovered only once it starts. The hold ends with the start, filled or failed."""
    runner, _home, db, _stops = _runner(tmp_path, monkeypatch)
    registry = runner.session_authorities
    starting = tmp_path / "starting"
    starting.mkdir()
    try:
        assert run_idle_exit.busy_reason(runner, window=0.2) is None
        registry.add(starting, None, name="starting")  # as serve_profile_runtime does
        assert "profile starting" in run_idle_exit.busy_reason(runner, window=0.2)
        registry.remove(starting)  # the build failed
        assert run_idle_exit.busy_reason(runner, window=0.2) is None
        registry.add(starting, None, name="starting")
        starting_db = SessionDB(db_path=starting / "state.db")
        registry.replace(starting, SimpleNamespace(profile_id=str(starting), db=starting_db))
        try:
            assert run_idle_exit.busy_reason(runner, window=0.2) is None
        finally:
            starting_db.close()
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

"""Routing withdrawal is not proof that the old profile writer released custody.

Exercise the public control handler, real authority/physical-worker retirement, SQLite
queue and profile locks. Only the CLI's external transport and cleanup services are fake.
The process-FD census is empty only for this fixture's fresh databases; deleted-WAL
generation detection across other processes is outside this test's contract.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, closing
from pathlib import Path
import sqlite3
import threading
from types import SimpleNamespace

import pytest

from gateway import run_runtime
from gateway.run_profile_reconcile import unserve_profile_verb
from gateway.runtime_bootstrap import TicketStore
from gateway.runtime_ownership import OwnershipConflict, ProfileOwnership, process_ownership
from gateway.session_authorities import SessionAuthorities, owner_scope
from gateway.session_authority import SessionAuthority
from gateway.session_runtime_workers import start_turn_worker
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from hermes_state_registry import acquire, close_all_under
from hermes_state_runtime import admit_session_input, begin_runtime_epoch
from tests.gateway.test_multiplex_hot_serve import _Adapter, _boot, _mkprofile, _runner


@asynccontextmanager
async def _host(tmp_path, monkeypatch, *, honors_stop, settle_seconds):
    runner, launch = _runner(tmp_path, monkeypatch)
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "runtime"))
    homes = {"default": launch, "worker": _mkprofile(launch, "worker"),
             "sibling": _mkprofile(launch, "sibling")}
    import hermes_state_dbfile
    fresh_databases = {(home / "state.db").resolve() for home in homes.values()}
    assert all(not path.exists() for path in fresh_databases)
    real_census = hermes_state_dbfile.iter_deleted_sqlite_sidecar_holders

    def sidecar_holders(path):
        # These databases did not exist before this test. Avoid macOS's global
        # process-FD scan without replacing SQLite/WAL handling or writer custody.
        return [] if Path(path).resolve() in fresh_databases else real_census(path)

    monkeypatch.setattr(hermes_state_dbfile, "iter_deleted_sqlite_sidecar_holders", sidecar_holders)
    _boot(runner, launch)
    runner._draining, runner._running_agents = False, {}
    runner._executor = ThreadPoolExecutor(max_workers=1)
    runner._note_served_profiles(list(homes.items()))
    sibling_adapter = _Adapter("sibling-test-transport")
    runner._profile_adapters = {"worker": {sibling_adapter.platform: _Adapter("worker-test-transport")},
                                "sibling": {sibling_adapter.platform: sibling_adapter}}
    registry = runner.session_authorities = SessionAuthorities(launch, multiplexed=True)
    for name, home in homes.items():
        db = acquire(home / "state.db")
        db.create_session("s", source="test")
        authority = SessionAuthority(runner, profile_id=str(home), instance_id=f"owner-{name}",
                                     db=db, epoch=begin_runtime_epoch(db, instance_id=f"owner-{name}"))
        registry.add(home, authority, name=name)
    runner.session_runtime_descriptor = {"served_profiles": registry.served_profiles()}
    runner.session_ticket_store = TicketStore("test-owner", registry.profile_ids())
    authority = registry.for_home(homes["worker"])
    # An already accepted follower must remain durable; retiring the physical writer must
    # neither claim nor erase it. No scheduler/agent execution is needed to create this row.
    admit_session_input(authority.db, epoch=authority.epoch, principal_id="human", session_id="s",
                        request_id="follower", payload={"text": "later"}, intent="queue")
    started, release, stopped = (threading.Event() for _ in range(3))
    worker = runner._RunAgentWorker(worker_done=threading.Event())

    def interrupt(message=None, *, tool_reason=None):
        stopped.set()
        if honors_stop:
            release.set()

    def physical_writer():
        started.set()
        try:
            assert release.wait(30), "test did not release its physical writer"
            authority.db.end_session("s", "test_worker_finished")
        finally:
            worker.worker_done.set()

    with owner_scope(authority):
        start_turn_worker(runner, worker, [SimpleNamespace(hard_interrupt=interrupt)], physical_writer)
    host = SimpleNamespace(runner=runner, homes=homes, registry=registry, authority=authority,
                           worker=worker, release=release, stopped=stopped,
                           sibling_adapter=sibling_adapter, releases=[], cleanups=[], unserve_calls=[])
    real_release = run_runtime.release_profile_home

    def release_home(owner, home):
        host.releases.append(Path(home))
        return real_release(owner, home)

    real_unserve = runner._unserve_profile

    async def unserve(name, home):
        host.unserve_calls.append(name)
        return await real_unserve(name, home)

    monkeypatch.setattr(run_runtime, "TURN_SETTLE_SECONDS", settle_seconds)
    monkeypatch.setattr(run_runtime, "release_profile_home", release_home)
    monkeypatch.setattr(runner, "_unserve_profile", unserve)
    monkeypatch.setattr("plugins.memory.import_provider_module", lambda *a: SimpleNamespace(
        MemoryStore=SimpleNamespace(release_all_under=lambda home: None)))
    monkeypatch.setattr("tools.mcp_tool_lifecycle.shutdown_mcp_servers", lambda **kw: None)
    monkeypatch.setattr("hermes_logging.release_profile_log_handlers", lambda home: host.cleanups.append(Path(home)))
    host.stop = unserve_profile_verb(runner)  # built on the actual owning loop
    try:
        assert await asyncio.to_thread(started.wait, 10), "physical worker did not start"
        yield host
    finally:
        release.set()
        await asyncio.wait_for(asyncio.shield(worker.executor_task), 10)
        # A five-second control timeout leaves apply() running. Wait for that same
        # teardown before closing its DB; never send a second unserve to settle it.
        lock = runner._reconcile_lock()
        await asyncio.wait_for(lock.acquire(), 10)
        lock.release()
        runner._executor.shutdown(wait=True)
        for home in homes.values():
            close_all_under(home)
            process_ownership.release(home)


def _queued_rows(home):
    with closing(sqlite3.connect((home / "state.db").as_uri() + "?mode=ro", uri=True)) as db:
        return db.execute("SELECT request_id, status FROM session_admissions ORDER BY request_id").fetchall()


def _cli_restart(host, monkeypatch):
    from gateway import control_socket
    from hermes_cli import gateway as gw
    from hermes_cli.gateway_profile_lifecycle import profile_lifecycle

    calls, answers = [], []
    owner = SimpleNamespace(home=host.homes["default"])
    monkeypatch.setattr(gw, "find_gateway_pids", list)
    monkeypatch.setattr(gw, "_served_by_another_host_gateway", lambda: owner)

    def census(home, command, *, params=None, **kwargs):
        assert home == host.homes["default"] and command == "active-work"
        assert params == {"profile": "worker"}
        # The observed zero snapshot can race a writer. Preserve the real preflight
        # policy and exercise what happens when teardown finds unfinished work.
        return {"active_agents": 0, "active_work": []}

    def unserve(home, name):
        assert home == host.homes["default"] and name == "worker"
        calls.append("unserve")
        answer = host.stop({"name": name})
        answers.append(answer)
        return answer

    def serve(home, name):
        assert home == host.homes["default"] and name == "worker"
        calls.append("serve")
        if process_ownership.owns(host.homes["worker"]):
            return {"error": "old writer still owns the profile"}
        return {"served": "worker", "served_profiles": ["default", "sibling", "worker"]}

    monkeypatch.setattr(control_socket, "query_gateway_control", census)
    monkeypatch.setattr(control_socket, "request_unserve_profile", unserve)
    monkeypatch.setattr(control_socket, "request_serve_profile_hot", serve)

    def invoke():
        token = set_hermes_home_override(host.homes["worker"])
        try:
            return profile_lifecycle("restart", SimpleNamespace())
        finally:
            reset_hermes_home_override(token)

    return invoke, calls, answers


@pytest.mark.asyncio
@pytest.mark.parametrize("honors_stop", [False, True], ids=["retained", "settled"])
async def test_public_retirement_receipt_controls_restart(tmp_path, monkeypatch, capsys, honors_stop):
    """A routing ACK cannot authorize re-serve while the actual old worker owns the home."""
    async with _host(tmp_path, monkeypatch, honors_stop=honors_stop, settle_seconds=2) as host:
        sibling = host.registry.for_home(host.homes["sibling"])
        invoke, calls, answers = _cli_restart(host, monkeypatch)
        assert await asyncio.wait_for(asyncio.to_thread(invoke), 10) is True
        # With the strict CLI consumer fixed first, base4d's missing receipt blocks
        # settled re-serve here; retained reaches the explicit receipt check below.
        assert calls == (["unserve", "serve"] if honors_stop else ["unserve"])
        answer = answers[0]
        assert answer["unserved"] == "worker"
        assert answer["served_profiles"] == ["default", "sibling"]
        assert answer.get("writer_settlement") == ("settled" if honors_stop else "retained")
        assert answer.get("reservation") == ("released" if honors_stop else "retained")
        assert host.stopped.is_set()
        assert host.worker.worker_done.is_set() is honors_stop
        assert process_ownership.owns(host.homes["worker"]) is (not honors_stop)
        assert host.releases == ([host.homes["worker"]] if honors_stop else [])
        assert host.cleanups == ([host.homes["worker"]] if honors_stop else [])
        assert host.registry.for_home(host.homes["worker"]) is (None if honors_stop else host.authority)
        assert _queued_rows(host.homes["worker"]) == [("follower", "queued")]
        contender = ProfileOwnership()
        if honors_stop:
            contender.reserve([host.homes["worker"]])
            contender.release(host.homes["worker"])
        else:
            with pytest.raises(OwnershipConflict):
                contender.reserve([host.homes["worker"]])
            assert host.authority.db.get_session("s")["end_reason"] is None
            assert "restarted by the host gateway" not in capsys.readouterr().out
        assert host.registry.for_home(host.homes["sibling"]) is sibling
        assert sibling.db.get_session("s")["end_reason"] is None
        assert [dict(row) for row in sibling.db._read_all("SELECT epoch, instance_id FROM runtime_epoch")] == [
            {"epoch": 1, "instance_id": "owner-sibling"}]
        assert not host.sibling_adapter.disconnected
        assert process_ownership.owns(host.homes["default"])
        assert process_ownership.owns(host.homes["sibling"])


@pytest.mark.asyncio
async def test_control_timeout_leaves_the_original_teardown_running(tmp_path, monkeypatch):
    """Pending cannot claim release, cancel the old writer, or replay the teardown."""
    async with _host(tmp_path, monkeypatch, honors_stop=False, settle_seconds=30) as host:
        answer = await asyncio.wait_for(asyncio.to_thread(host.stop, {"name": "worker"}), 10)
        assert answer.get("pending") is True
        assert answer.get("writer_settlement") != "settled"
        assert answer.get("reservation") != "released"
        assert host.stopped.is_set() and not host.worker.worker_done.is_set()
        assert process_ownership.owns(host.homes["worker"])
        assert host.unserve_calls == ["worker"] and host.releases == []
        host.release.set()
        lock = host.runner._reconcile_lock()
        await asyncio.wait_for(lock.acquire(), 10)
        lock.release()
        assert host.worker.worker_done.is_set()
        assert host.unserve_calls == ["worker"]
        assert host.releases == [host.homes["worker"]]
        assert host.registry.for_home(host.homes["worker"]) is None
        assert not process_ownership.owns(host.homes["worker"])
        assert _queued_rows(host.homes["worker"]) == [("follower", "queued")]

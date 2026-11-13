"""Lease recovery preserves running capacity and exclusive session ownership."""

import json
import os
from pathlib import Path
import subprocess
import sys
import threading

import pytest

from hermes_cli import active_sessions
from tests.waiting import wait_until


CONFIG = {"max_concurrent_sessions": 1}


def acquire(home, session_id, *, active=True, writer=None):
    lease, refusal = active_sessions.try_acquire_active_session(
        session_id=session_id, surface="desktop", config=CONFIG,
        metadata={"live_session_id": writer or session_id}, registry_home=home,
        track_liveness=True, capacity_active=active,
    )
    assert refusal is None
    assert lease is not None
    return lease


@pytest.mark.parametrize("initially_active", [False, True], ids=["acquired-idle", "settled-idle"])
def test_missing_idle_entry_transfer_preserves_capacity_and_ownership(tmp_path, initially_active):
    home = tmp_path / "registry"
    idle = acquire(home, "idle-old", active=initially_active)
    if initially_active:
        assert idle.set_capacity_active(False) is None
    # Simulate loss of the durable registry; the retained live lease can recover.
    assert idle.state_path is not None
    idle.state_path.unlink()
    running = acquire(home, "running")

    assert active_sessions.transfer_active_session(idle, session_id="idle-new")
    snapshot = active_sessions.active_session_registry_snapshot(home)
    assert sum(entry.get("capacity_active", True) for entry in snapshot) == 1
    refusal = idle.set_capacity_active(True, config=CONFIG)
    assert isinstance(refusal, active_sessions.ActiveSessionRefusal)
    assert refusal.reason == active_sessions.MAX_CONCURRENT_SESSIONS
    foreign, refusal = active_sessions.try_acquire_active_session(
        session_id="idle-new", surface="cli", config={}, registry_home=home,
        metadata={"live_session_id": "foreign-writer"}, capacity_active=False,
    )
    assert foreign is None
    assert isinstance(refusal, active_sessions.ActiveSessionRefusal)
    assert refusal.reason == active_sessions.SESSION_NOT_OWNED
    running.release()
    assert idle.set_capacity_active(True, config=CONFIG) is None
    idle.release()
    assert active_sessions.active_session_registry_snapshot(home) == []


@pytest.mark.parametrize("reclaim_idle", [False, True], ids=["original-active", "idle-reclaim-retains-active"])
def test_missing_active_entry_transfer_cannot_overfill_recovered_registry(tmp_path, reclaim_idle):
    home = tmp_path / "registry"
    original = acquire(home, "original")
    if reclaim_idle:
        original = acquire(home, "original", active=False)
    assert original.state_path is not None
    original.state_path.unlink()
    running = acquire(home, "running")
    before = active_sessions.active_session_registry_snapshot(home)

    assert not active_sessions.transfer_active_session(original, session_id="rotated")
    assert original.session_id == "original"
    assert active_sessions.active_session_registry_snapshot(home) == before
    running.release()
    assert active_sessions.transfer_active_session(original, session_id="rotated")
    blocked, refusal = active_sessions.try_acquire_active_session(
        session_id="blocked", surface="cli", config=CONFIG, registry_home=home,
    )
    assert blocked is None
    assert isinstance(refusal, active_sessions.ActiveSessionRefusal)
    assert refusal.reason == active_sessions.MAX_CONCURRENT_SESSIONS
    original.release()


@pytest.mark.parametrize("missing_source", [False, True], ids=["present-source", "recovered-source"])
def test_transfer_to_owned_destination_refuses_without_changing_either_owner(tmp_path, missing_source):
    home = tmp_path / "registry"
    source = acquire(home, "source", active=False)
    if missing_source:
        assert source.state_path is not None
        source.state_path.unlink()
    destination = acquire(home, "destination", active=False)
    before = active_sessions.active_session_registry_snapshot(home)

    assert not active_sessions.transfer_active_session(
        source, session_id="destination", metadata={"live_session_id": "source-writer"},
    )
    assert source.session_id == "source"
    assert not source.released and not destination.released
    assert active_sessions.active_session_registry_snapshot(home) == before
    assert active_sessions.transfer_active_session(source, session_id="unowned")
    assert active_sessions.transfer_active_session(destination, session_id="destination")
    source.release()
    destination.release()
    assert active_sessions.active_session_registry_snapshot(home) == []


@pytest.mark.parametrize("outage_kind", ["write", "read", "lock"])
def test_failed_idle_transition_is_retried_by_existing_orphan_heartbeat(tmp_path, monkeypatch, outage_kind):
    home = tmp_path / "owner-profile"
    lease = acquire(home, "settled")
    assert lease.state_path is not None and lease.lock_path is not None
    state_before = lease.state_path.read_bytes()

    def unavailable(*args, **kwargs):
        raise OSError("synthetic registry write outage")

    try:
        with monkeypatch.context() as outage:
            if outage_kind == "write":
                outage.setattr(active_sessions, "atomic_json_write", unavailable)
            elif outage_kind == "read":
                lease.state_path.write_text("{synthetic incomplete JSON", encoding="utf-8")
            else:
                lease.lock_path.unlink()
                lease.lock_path.mkdir()
            with pytest.raises((OSError, active_sessions.ActiveSessionRegistryError)):
                lease.set_capacity_active(False)
    finally:
        if outage_kind == "read":
            lease.state_path.write_bytes(state_before)
        elif outage_kind == "lock":
            lease.lock_path.rmdir()

    # The existing backend heartbeat calls this entry point without a turn.
    # It must retry in the lease's owning home, not the current profile.
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "different-profile"))
    active_sessions.release_orphaned_leases({lease.lease_id})
    admitted = acquire(home, "next-turn")
    foreign, refusal = active_sessions.try_acquire_active_session(
        session_id="settled", surface="cli", config={}, registry_home=home,
        metadata={"live_session_id": "foreign"}, capacity_active=False,
    )
    assert foreign is None
    assert isinstance(refusal, active_sessions.ActiveSessionRefusal)
    assert refusal.reason == active_sessions.SESSION_NOT_OWNED
    admitted.release()
    lease.release()


def test_idle_retry_outage_does_not_prevent_other_profile_recovery(tmp_path, monkeypatch):
    blocked_home = tmp_path / "blocked"
    healthy_home = tmp_path / "healthy"
    blocked = acquire(blocked_home, "blocked")
    healthy = acquire(healthy_home, "healthy")
    real_write = active_sessions.atomic_json_write

    def unavailable(path, *args, **kwargs):
        raise OSError("synthetic outage")

    with monkeypatch.context() as outage:
        outage.setattr(active_sessions, "atomic_json_write", unavailable)
        for lease in (blocked, healthy):
            with pytest.raises(OSError, match="synthetic outage"):
                lease.set_capacity_active(False)

    def one_profile_unavailable(path, *args, **kwargs):
        if Path(path) == blocked.state_path:
            raise OSError("still unavailable")
        return real_write(path, *args, **kwargs)

    with monkeypatch.context() as outage:
        outage.setattr(active_sessions, "atomic_json_write", one_profile_unavailable)
        active_sessions.release_orphaned_leases({blocked.lease_id, healthy.lease_id})
    next_healthy = acquire(healthy_home, "next-healthy")
    active_sessions.release_orphaned_leases({blocked.lease_id, healthy.lease_id, next_healthy.lease_id})
    next_blocked = acquire(blocked_home, "next-blocked")
    for lease in (blocked, healthy, next_healthy, next_blocked):
        lease.release()


@pytest.mark.parametrize("reclaim_writer", [False, True], ids=["same-lease-reactivation", "reclaimed-writer"])
def test_inflight_idle_retry_cannot_deactivate_running_writer(tmp_path, monkeypatch, reclaim_writer):
    home = tmp_path / "registry"
    original = acquire(home, "writer")

    def unavailable(*args, **kwargs):
        raise OSError("synthetic outage")

    with monkeypatch.context() as outage:
        outage.setattr(active_sessions, "atomic_json_write", unavailable)
        with pytest.raises(OSError):
            original.set_capacity_active(False)

    retry_waiting = threading.Event()
    allow_retry = threading.Event()
    real_enter = active_sessions._FileLock.__enter__

    def pause_retry(lock):
        if threading.current_thread().name == "idle-retry":
            retry_waiting.set()
            assert allow_retry.wait(10), "test did not release pending idle retry"
        return real_enter(lock)

    monkeypatch.setattr(active_sessions._FileLock, "__enter__", pause_retry)
    worker = threading.Thread(
        target=active_sessions.release_orphaned_leases,
        args=({original.lease_id},), name="idle-retry",
    )
    worker.start()
    try:
        assert retry_waiting.wait(10), "heartbeat did not retry idle intent"
        if reclaim_writer:
            reclaimed = acquire(home, "writer")
            assert reclaimed.lease_id != original.lease_id
        else:
            # The in-flight heartbeat holds the per-lease lock. Resume waits
            # behind that transition and must become the final durable intent.
            reclaimed = original
            allow_retry.set()
            assert original.set_capacity_active(True, config=CONFIG) is None
    finally:
        allow_retry.set()
        worker.join(10)
    assert not worker.is_alive()
    refused, refusal = active_sessions.try_acquire_active_session(
        session_id="other", surface="cli", config=CONFIG, registry_home=home,
    )
    assert refused is None
    assert isinstance(refusal, active_sessions.ActiveSessionRefusal)
    assert refusal.reason == active_sessions.MAX_CONCURRENT_SESSIONS
    reclaimed.release()


def test_cross_process_transfers_atomically_choose_one_destination_owner(tmp_path):
    home = tmp_path / "registry"
    script = """
import json, sys
from pathlib import Path
from hermes_cli import active_sessions
home, source, ready, result = sys.argv[1:]
lease, refusal = active_sessions.try_acquire_active_session(
    session_id=source, surface="desktop", registry_home=home,
    config={"max_concurrent_sessions": 1}, capacity_active=False,
    track_liveness=True, metadata={"live_session_id": source},
)
assert refusal is None
Path(ready).write_text("ready", encoding="utf-8")
input()
transferred = active_sessions.transfer_active_session(lease, session_id="destination")
Path(result).write_text(json.dumps({"transferred": transferred, "session_id": lease.session_id}), encoding="utf-8")
input()
lease.release()
"""
    workers = []
    markers = [(tmp_path / f"ready-{i}", tmp_path / f"result-{i}") for i in range(2)]
    try:
        for i, (ready, result) in enumerate(markers):
            workers.append(subprocess.Popen(
                [sys.executable, "-c", script, str(home), f"source-{i}", str(ready), str(result)],
                cwd=Path(__file__).resolve().parents[2], env={**os.environ, "HERMES_HOME": str(home)},
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            ))
        assert wait_until(lambda: all(ready.exists() for ready, _ in markers), timeout=20)
        for worker in workers:
            assert worker.stdin is not None
            worker.stdin.write("go\n")
            worker.stdin.flush()
        assert wait_until(lambda: all(result.exists() for _, result in markers), timeout=20)
        results = [json.loads(result.read_text(encoding="utf-8")) for _, result in markers]
        assert sum(result["transferred"] for result in results) == 1
        snapshot = active_sessions.active_session_registry_snapshot(home)
        assert len(snapshot) == 2
        assert sum(entry["session_id"] == "destination" for entry in snapshot) == 1
        for i, result in enumerate(results):
            assert result["session_id"] == ("destination" if result["transferred"] else f"source-{i}")
        for worker in workers:
            output, error = worker.communicate("release\n", timeout=10)
            assert worker.returncode == 0, output + error
        assert active_sessions.active_session_registry_snapshot(home) == []
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.kill()
            worker.communicate(timeout=10)

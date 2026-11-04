"""Machine-wide residency admission: records, fencing epochs, leases, recovery.

The store is a pure coordination primitive: it never spawns engines, never
signals processes, never touches boot locks or the network. These tests pin
that contract, with injected ``now``/``boot_id`` so time is deterministic.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from hermes_cli.local_runtime import admission
from hermes_cli.local_runtime.admission import (
    AdmissionError,
    AdmissionStore,
    FencedError,
    Grant,
)

BOOT = "boot-A"
LEASE = 30.0


def _acquire(store, resource="llama:res-1", *, controller="ctrl-1", pid=1000,
             ident="ident-1", boot=BOOT, now=100.0, request_unload=False):
    return store.acquire(
        resource,
        owner_scope="scope:opaque//weird name",
        controller_id=controller,
        pid=pid,
        process_start_identity=ident,
        engine="llama.cpp",
        model="TestModel-Q4",
        boot_id=boot,
        now=now,
        lease_seconds=LEASE,
        request_unload=request_unload,
    )


# -- store location / laziness ------------------------------------------------


def test_default_path_is_machine_root_local_runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    assert admission.default_store_path() == (
        tmp_path / "local_runtime" / "admission.sqlite3")


def test_store_created_lazily_reads_never_create(tmp_path):
    store = AdmissionStore(tmp_path)
    assert not store.path.exists()
    assert store.status("anything") is None
    assert not store.path.exists()
    grant = _acquire(store)
    assert isinstance(grant, Grant)
    assert store.path.exists()
    assert store.path.parent == tmp_path / "local_runtime"


# -- record fields ------------------------------------------------------------


def test_acquire_records_every_required_field(tmp_path):
    store = AdmissionStore(tmp_path)
    grant = _acquire(store, now=100.0)
    assert isinstance(grant, Grant)
    assert grant.resource == "llama:res-1"
    assert grant.owner_scope == "scope:opaque//weird name"
    assert grant.controller_id == "ctrl-1"
    assert grant.pid == 1000
    assert grant.process_start_identity == "ident-1"
    assert grant.engine == "llama.cpp"
    assert grant.model == "TestModel-Q4"
    assert grant.residency == "loading"
    assert grant.epoch == 1
    assert grant.boot_id == BOOT
    assert grant.deadline == pytest.approx(130.0)
    rec = store.status("llama:res-1")
    assert rec is not None
    assert rec.epoch == 1
    assert rec.residency == "loading"
    assert rec.quarantine_reason is None
    assert rec.active_leases == ()


# -- single-owner admission ---------------------------------------------------


def test_second_controller_is_refused_busy(tmp_path):
    store = AdmissionStore(tmp_path)
    assert isinstance(_acquire(store, now=100.0), Grant)
    refusal = _acquire(store, controller="ctrl-2", pid=2000, ident="ident-2",
                       now=110.0)
    assert not isinstance(refusal, Grant)
    assert refusal.reason == "busy"
    assert refusal.retry_after == pytest.approx(20.0)
    # The incumbent is untouched.
    assert store.status("llama:res-1").controller_id == "ctrl-1"


def test_reacquire_same_controller_is_idempotent(tmp_path):
    store = AdmissionStore(tmp_path)
    first = _acquire(store)
    again = _acquire(store)
    assert isinstance(again, Grant)
    assert again.epoch == first.epoch
    assert again.controller_id == first.controller_id


def test_drain_requested_and_idle_yields_unload_required(tmp_path):
    store = AdmissionStore(tmp_path)
    grant = _acquire(store, now=100.0)
    store.add_lease(grant, "job-1", kind="job")
    # Drain requested while the incumbent still holds a lease: still busy.
    refusal = _acquire(store, controller="ctrl-2", pid=2000, ident="ident-2",
                       now=110.0, request_unload=True)
    assert refusal.reason == "busy"
    assert store.status("llama:res-1").residency == "draining"
    # Incumbent goes idle (no leases) -> the caller must unload, not steal.
    store.release_lease(grant, "job-1")
    refusal = _acquire(store, controller="ctrl-2", pid=2000, ident="ident-2",
                       now=115.0)
    assert refusal.reason == "unload_required"
    assert refusal.retry_after is None
    # Incumbent releases -> next controller wins at a higher epoch.
    assert store.release(grant, now=116.0) is True
    nxt = _acquire(store, controller="ctrl-2", pid=2000, ident="ident-2",
                   now=117.0)
    assert isinstance(nxt, Grant) and nxt.epoch == 2


# -- leases --------------------------------------------------------------------


def test_lease_and_job_transitions_are_idempotent(tmp_path):
    store = AdmissionStore(tmp_path)
    grant = _acquire(store)
    a1 = store.add_lease(grant, "job-1", kind="job")
    a2 = store.add_lease(grant, "job-1", kind="job")
    assert a1 == a2 and a1.state == "active"
    assert store.status("llama:res-1").active_leases == ("job-1",)
    r1 = store.release_lease(grant, "job-1")
    r2 = store.release_lease(grant, "job-1")
    assert r1 == r2 and r1.state == "released"
    assert store.status("llama:res-1").active_leases == ()
    c1 = store.complete_job(grant, "job-1")
    c2 = store.complete_job(grant, "job-1")
    assert c1 == c2 and c1.state == "completed"
    # Never raises, even for a lease this grant never had.
    assert store.release_lease(grant, "ghost").state == "released"


# -- fencing --------------------------------------------------------------------


def test_controller_restart_bumps_epoch_and_fences_the_old_grant(tmp_path):
    store = AdmissionStore(tmp_path)
    old = _acquire(store, now=100.0)
    store.add_lease(old, "job-1", kind="job")
    assert store.release(old, now=105.0) is True
    new = _acquire(store, controller="ctrl-2", pid=2000, ident="ident-2",
                   now=110.0)
    assert isinstance(new, Grant) and new.epoch > old.epoch

    with pytest.raises(FencedError):
        store.renew(old, now=120.0)
    with pytest.raises(FencedError):
        store.add_lease(old, "job-2", kind="stream")
    with pytest.raises(FencedError):
        store.release(old, now=121.0)
    with pytest.raises(FencedError):
        store.complete_job(old, "job-1")
    # Nothing changed: the new controller's state is untouched.
    rec = store.status("llama:res-1")
    assert rec.controller_id == "ctrl-2" and rec.epoch == new.epoch
    assert rec.residency == "loading"


def test_release_is_idempotent_for_the_holder(tmp_path):
    store = AdmissionStore(tmp_path)
    grant = _acquire(store)
    assert store.release(grant, now=101.0) is True
    assert store.release(grant, now=102.0) is True
    assert store.status("llama:res-1").residency == "released"


def test_renew_extends_deadline_and_can_advance_residency(tmp_path):
    store = AdmissionStore(tmp_path)
    grant = _acquire(store, now=100.0)
    renewed = store.renew(grant, now=120.0, residency="resident")
    assert renewed.epoch == grant.epoch
    assert renewed.deadline == pytest.approx(150.0)
    assert renewed.residency == "resident"
    assert store.status("llama:res-1").residency == "resident"
    with pytest.raises(AdmissionError):
        store.renew(grant, now=200.0)  # past deadline


# -- stale recovery --------------------------------------------------------------


def test_expired_deadline_is_not_proof_of_release(tmp_path):
    store = AdmissionStore(tmp_path)
    grant = _acquire(store, now=100.0)
    # Expired incumbent: a new controller is refused, memory is not presumed free.
    refusal = _acquire(store, controller="ctrl-2", pid=2000, ident="ident-2",
                       now=200.0)
    assert refusal.reason == "stale_unverified"
    # Recovery without verified exit quarantines, and keeps refusing.
    assert store.recover("llama:res-1", 201.0, BOOT) == "quarantined"
    assert store.recover("llama:res-1", 202.0, BOOT) == "quarantined"
    refusal = _acquire(store, controller="ctrl-2", pid=2000, ident="ident-2",
                       now=203.0)
    assert refusal.reason == "quarantined"
    assert "recover" in refusal.detail
    assert grant.controller_id in refusal.detail
    # Only verified exit unlocks the next grant, at a higher epoch.
    assert store.recover("llama:res-1", 204.0, BOOT,
                         verified_exit=True, expected_epoch=grant.epoch) == "released"
    nxt = _acquire(store, controller="ctrl-2", pid=2000, ident="ident-2",
                   now=205.0)
    assert isinstance(nxt, Grant) and nxt.epoch == grant.epoch + 1


def test_store_uses_wal_journal_mode(tmp_path):
    import sqlite3
    store = AdmissionStore(tmp_path)
    assert isinstance(_acquire(store), Grant)
    conn = sqlite3.connect(store.path)
    try:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    finally:
        conn.close()


def test_old_boot_records_are_quarantined_never_adopted(tmp_path):
    store = AdmissionStore(tmp_path)
    old = _acquire(store, now=100.0)
    # A different boot never adopts the old record, even a live one.
    mismatch = _acquire(store, controller="ctrl-2", pid=2000,
                        ident="ident-2", boot="boot-B", now=101.0)
    assert mismatch.reason == "boot_mismatch"
    assert store.recover("llama:res-1", 50.0, "boot-B") == "quarantined"
    rec = store.status("llama:res-1")
    assert rec.residency == "quarantined"
    assert rec.quarantine_reason == "boot_mismatch"
    refusal = _acquire(store, controller="ctrl-2", pid=2000, ident="ident-2",
                       boot="boot-B", now=51.0)
    assert refusal.reason == "quarantined"
    # A live-looking record from a *different* boot can never be renewed:
    with pytest.raises(FencedError):
        store.renew(old, now=52.0, boot_id="boot-B")
    assert store.recover("llama:res-1", 53.0, "boot-B",
                         verified_exit=True, expected_epoch=old.epoch) == "released"
    nxt = _acquire(store, controller="ctrl-2", pid=2000, ident="ident-2",
                   boot="boot-B", now=54.0)
    assert isinstance(nxt, Grant) and nxt.epoch == old.epoch + 1


def test_pid_reuse_ambiguity_quarantines_without_signalling(tmp_path):
    store = AdmissionStore(tmp_path)
    grant = _acquire(store, pid=4242, ident="start-1")
    # Same pid, different creation identity: ambiguous, refuse + quarantine.
    refusal = _acquire(store, controller="ctrl-2", pid=4242, ident="start-2",
                       now=105.0)
    assert refusal.reason == "identity_ambiguous"
    assert "recover" in refusal.detail
    rec = store.status("llama:res-1")
    assert rec.residency == "quarantined"
    assert rec.quarantine_reason == "identity_ambiguous"
    assert store.recover("llama:res-1", 106.0, BOOT) == "quarantined"
    assert _acquire(store, controller="ctrl-2", pid=5000,
                    ident="start-3", now=107.0).reason == "quarantined"
    assert store.recover("llama:res-1", 108.0, BOOT,
                         verified_exit=True, expected_epoch=grant.epoch) == "released"
    nxt = _acquire(store, controller="ctrl-2", pid=5000, ident="start-3",
                   now=109.0)
    assert isinstance(nxt, Grant) and nxt.epoch == grant.epoch + 1


def test_recover_absent_is_none_and_released_is_no_op(tmp_path):
    store = AdmissionStore(tmp_path)
    assert store.recover("nope", 10.0, BOOT) == "none"
    grant = _acquire(store, now=100.0)
    assert store.release(grant, now=101.0) is True
    # Recovery of a cleanly released record is a terminal no-op.
    assert store.recover("llama:res-1", 102.0, BOOT) == "released"
    assert store.recover("llama:res-1", 103.0, BOOT,
                         verified_exit=True, expected_epoch=grant.epoch) == "released"


def test_recover_of_live_record_still_needs_verified_exit(tmp_path):
    store = AdmissionStore(tmp_path)
    _acquire(store, now=100.0)
    # A live-looking record whose holder is suspect is quarantined first:
    # an unverified recovery never grants permission to take over.
    assert store.recover("llama:res-1", 110.0, BOOT) == "quarantined"
    assert store.status("llama:res-1").quarantine_reason == "stale_unverified"


# -- epoch monotonicity across rounds ------------------------------------------


def test_epochs_strictly_increase_over_release_cycles(tmp_path):
    store = AdmissionStore(tmp_path)
    epochs = []
    now = 100.0
    for i in range(4):
        grant = _acquire(store, controller=f"ctrl-{i}", pid=1000 + i,
                         ident=f"ident-{i}", now=now)
        assert isinstance(grant, Grant)
        epochs.append(grant.epoch)
        assert store.release(grant, now=now + 1.0) is True
        now += 10.0
    assert epochs == sorted(epochs) and len(set(epochs)) == len(epochs)


# -- purity: no process control, no network, no shared globals -------------------


def test_module_never_signals_processes_or_opens_networks():
    src = Path(inspect.getsourcefile(admission)).read_text(encoding="utf-8")
    for banned in ("os.kill", "signal.", "Popen", "subprocess", "socket",
                   "urllib", "requests", "fcntl", "psutil"):
        assert banned not in src, f"admission must never use {banned!r}"


def test_grant_api_defaults_do_not_fabricate_identity():
    params = inspect.signature(AdmissionStore.acquire).parameters
    for name in ("owner_scope", "controller_id", "pid",
                 "process_start_identity", "boot_id"):
        assert params[name].default is inspect.Parameter.empty, name


# -- TOCTOU: verified exit is bound to the epoch that was verified --------------


def test_verified_exit_without_an_epoch_binding_is_refused(tmp_path):
    store = AdmissionStore(tmp_path)
    _acquire(store, now=100.0)
    with pytest.raises(AdmissionError, match="expected_epoch"):
        store.recover("llama:res-1", 110.0, BOOT, verified_exit=True)
    # Nothing changed: the holder is still the live incumbent.
    assert store.status("llama:res-1").residency == "loading"


def test_verified_exit_cannot_release_a_newer_live_holder(tmp_path):
    """The operator verified epoch 1's process is gone; before the recover call
    lands, epoch 2 was legitimately granted. The stale attestation must NOT
    release the live epoch-2 holder (that would allow a double residency)."""
    store = AdmissionStore(tmp_path)
    first = _acquire(store, now=100.0)
    assert store.recover("llama:res-1", 140.0, BOOT,
                         verified_exit=True, expected_epoch=first.epoch) == "released"
    second = _acquire(store, controller="ctrl-2", pid=2000, ident="ident-2", now=141.0)
    assert isinstance(second, Grant) and second.epoch == first.epoch + 1

    # A late, stale recover still carrying the OLD epoch:
    outcome = store.recover("llama:res-1", 142.0, BOOT,
                            verified_exit=True, expected_epoch=first.epoch)
    assert outcome == "stale_epoch"
    live = store.status("llama:res-1")
    assert live.residency == "loading" and live.epoch == second.epoch
    # ... and a third controller is still refused: no double grant.
    third = _acquire(store, controller="ctrl-3", pid=3000, ident="ident-3", now=143.0)
    assert third.reason == "busy"
    # The live holder remains fully functional.
    assert store.renew(second, now=144.0).epoch == second.epoch


# -- lease identity is per epoch -------------------------------------------------


def test_reused_lease_id_in_a_new_epoch_is_tracked_as_active(tmp_path):
    store = AdmissionStore(tmp_path)
    first = _acquire(store, now=100.0)
    store.add_lease(first, "job-1")
    store.complete_job(first, "job-1")
    store.release(first, now=105.0)

    second = _acquire(store, controller="ctrl-2", pid=2000, ident="ident-2", now=106.0)
    assert second.epoch == first.epoch + 1
    result = store.add_lease(second, "job-1")  # same id, NEW grant
    assert result.state == "active" and result.epoch == second.epoch
    # The new job is really tracked: a drain request now sees it as busy.
    refusal = _acquire(store, controller="ctrl-3", pid=3000, ident="ident-3",
                       now=107.0, request_unload=True)
    assert refusal.reason == "busy"
    assert store.status("llama:res-1").active_leases == ("job-1",)

"""Retired accepted runs remain observable before bounded, fenced compaction."""
import time

import pytest

from gateway.platforms.api_server_run_idempotency import RunIdempotencyStore


def prune(store, now):
    with store._immediate_txn():
        store._prune_stale_terminal_locked(now)
        store._conn.commit()


@pytest.mark.parametrize("terminal", ["cancelled", "interrupted", "completed", "failed"])
def test_retired_stopped_receipt_survives_observation_and_restart(tmp_path, monkeypatch, terminal):
    clock = [time.time()]
    monkeypatch.setattr(time, "time", lambda: clock[0])
    path = str(tmp_path / "runs.db")
    authority = ("room-member", 1, "home")
    horizon = clock[0] + 3 * RunIdempotencyStore.RETENTION_SECONDS
    store = RunIdempotencyStore(path)
    try:
        assert store.observe_room_authority("original", authority)
        store.reserve("original", "accepted", "payload", "run", {"status": "running"},
                      room_authority=authority, retention_until=horizon)
        store.request_stop("original", "run")
        assert store.observe_room_authority("successor", (authority[0], 2, "successor"))
        store.update_status("run", {"status": terminal})
        assert store.status_for_run("original", "run")["status"]["status"] == terminal
        store.close()
        store = RunIdempotencyStore(path)
        assert store.status_for_run("original", "run")["status"]["status"] == terminal
        # The signed observation horizon is longer than the settlement grace.
        clock[0] += RunIdempotencyStore.RETENTION_SECONDS + 1
        prune(store, clock[0])
        assert store.status_for_run("original", "run")["status"]["status"] == terminal
        clock[0] = horizon + 1
        prune(store, clock[0])
        assert store.status_for_run("original", "run") is None
        assert store.reserve("original", "accepted", "payload", "late", {"status": "queued"},
                             room_authority=authority) == ("authority_retired", None)
    finally:
        store.close()


@pytest.mark.parametrize("ack_after_settlement", [False, True])
def test_only_terminal_acknowledgement_can_shorten_verified_horizon(tmp_path, monkeypatch, ack_after_settlement):
    clock = [time.time()]
    monkeypatch.setattr(time, "time", lambda: clock[0])
    store = RunIdempotencyStore(str(tmp_path / "runs.db"))
    try:
        authority = ("room-member", 1, "home")
        store.observe_room_authority("original", authority)
        store.reserve("original", "accepted", "payload", "run", {"status": "running"},
                      room_authority=authority, retention_until=clock[0] + 10 * store.RETENTION_SECONDS)
        store.request_stop("original", "run")
        store.observe_room_authority("successor", (authority[0], 2, "successor"))
        clock[0] += 10
        store.update_status("run", {"status": "cancelled"})
        acknowledgement = clock[0] + (1 if ack_after_settlement else -1)
        store._conn.execute("UPDATE run_idempotency SET acknowledged_at=?", (acknowledgement,))
        store._conn.commit()
        clock[0] = acknowledgement + store.ACKNOWLEDGED_RETENTION_SECONDS + 1
        prune(store, clock[0])
        receipt = store.status_for_run("original", "run")
        assert (receipt is None) == ack_after_settlement
    finally:
        store.close()


def test_absent_barriers_compact_immediately_but_accepted_receipts_need_observation(tmp_path):
    store = RunIdempotencyStore(str(tmp_path / "runs.db"))
    try:
        authority = ("room-member", 1, "home")
        store.observe_room_authority("original", authority)
        for index in range(256):
            store.reserve("original", f"absent-{index}", "", f"absent-run-{index}",
                          {"status": "cancelled", "admission_cancelled": True},
                          cancel_if_missing=True, room_authority=authority)
        # A truthy non-boolean claim is not exact proof that no admission happened.
        store.reserve("original", "accepted", "payload", "run",
                      {"status": "cancelled", "admission_cancelled": "true"},
                      room_authority=authority)
        store.request_stop("original", "run")
        store.retire_room_authority("original", authority)
        assert store._conn.execute("SELECT run_id FROM run_idempotency").fetchall() == [("run",)]
        for index in range(256):
            assert store.reserve("original", f"absent-{index}", "late", "late", {"status": "queued"},
                                 room_authority=authority) == ("authority_retired", None)
    finally:
        store.close()


def test_elapsed_observation_without_authority_floor_cannot_forget_stop(tmp_path):
    store = RunIdempotencyStore(str(tmp_path / "runs.db"))
    try:
        authority = ("room-member", 1, "home")
        store.observe_room_authority("original", authority)
        store.reserve("original", "accepted", "payload", "run", {"status": "cancelled"},
                      room_authority=authority)
        store.request_stop("original", "run")
        store._conn.execute("UPDATE run_idempotency SET updated_at=0,retention_until=1")
        store._conn.commit()
        prune(store, time.time())
        assert store.reserve("original", "accepted", "payload", "late", {"status": "queued"},
                             room_authority=authority)[0] == "reused"
    finally:
        store.close()


def test_settlement_starts_recovery_window_after_original_grant_expired(tmp_path, monkeypatch):
    clock = [time.time()]
    monkeypatch.setattr(time, "time", lambda: clock[0])
    path = str(tmp_path / "runs.db")
    store = RunIdempotencyStore(path)
    try:
        authority = ("room-member", 1, "home")
        store.observe_room_authority("original", authority)
        store.reserve("original", "accepted", "payload", "run", {"status": "running"},
                      room_authority=authority, retention_until=clock[0] + 60)
        store.request_stop("original", "run")
        store.observe_room_authority("successor", (authority[0], 2, "successor"))
        clock[0] += store.RETENTION_SECONDS + 120
        prune(store, clock[0])
        assert store.status_for_run("original", "run")["status"]["status"] == "running"
        store.update_status("run", {"status": "cancelled"})
        store.close()
        store = RunIdempotencyStore(path)
        prune(store, clock[0])
        assert store.status_for_run("original", "run")["status"]["status"] == "cancelled"
        clock[0] += store.RETENTION_SECONDS + 1
        prune(store, clock[0])
        assert store.status_for_run("original", "run") is None
        assert store.reserve("original", "accepted", "payload", "late", {"status": "queued"},
                             room_authority=authority) == ("authority_retired", None)
    finally:
        store.close()

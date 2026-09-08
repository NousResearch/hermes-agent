"""One durable winner for passive enrollment replacement versus retirement."""

import sqlite3

import pytest

from gateway import hosted_room_replicas as replicas
from gateway import hosted_room_replica_retirement as retirement
from gateway import hosted_rooms as rooms
from tests.gateway.test_hosted_room_replica_retirement import (
    HOME, TARGET, SECRET, pair, enroll, copied_prefix, close, notice,  # noqa: F401
)
from tests.gateway.test_hosted_room_replica_lineage import (
    SUCCESSOR, transfer, setup_successor, page_v2, ingest,
)


def fork(source, destination):
    with sqlite3.connect(source) as src, sqlite3.connect(destination) as dst:
        src.backup(dst)


@pytest.mark.parametrize("first", ["replace", "retire"])
def test_retirement_and_replacement_have_one_durable_winner(pair, tmp_path, first):
    source, target = pair
    entry = enroll(pair)
    copied_prefix(pair)
    old = tmp_path / "old.db"
    fork(source, old)
    close(old)
    stale = notice(old, entry)
    spans = [{"gateway_id": HOME, "epoch": 1, "from_seq": 0}, transfer(source)]
    if first == "retire":
        retirement.retire_copy(target, payload=stale.payload(), value=stale.value, local_gateway_id=TARGET)
        with pytest.raises(rooms.HostedRoomError):
            setup_successor(source, target, spans, old_id=entry["enrollment_id"])
        assert replicas.replica_state(target, room_id="room")["safety_status"] == "retired"
        return
    current = setup_successor(source, target, spans, old_id=entry["enrollment_id"])
    before = replicas.replica_state(target, room_id="room")
    assert before["lineage_status"] == "pending"
    assert before["authority"] == {"gateway_id": HOME, "epoch": 1}
    with pytest.raises(retirement.RetirementAuthorizationError):
        retirement.retire_copy(target, payload=stale.payload(), value=stale.value, local_gateway_id=TARGET)
    assert replicas.replica_state(target, room_id="room") == before
    with pytest.raises(rooms.HostedRoomError):
        ingest(target, rooms.read_events(old, room_id="room", include_disbanded=True))
    from gateway import hosted_room_link_records as links
    links.begin_room_link_retirement(source, room_id="room", authority_gateway_id=SUCCESSOR, authority_epoch=2)
    links.complete_room_link_retirement(source, room_id="room", authority_gateway_id=SUCCESSOR, authority_epoch=2)
    rooms.disband_room(source, room_id="room", expected_gateway_id=SUCCESSOR, expected_epoch=2)
    closing = retirement.materialize_notice(source, enrollment_id=current["enrollment_id"],
        local_gateway_id=SUCCESSOR, secret_loader=lambda: SECRET)
    result = retirement.retire_copy(target, payload=closing.payload(), value=closing.value, local_gateway_id=TARGET)
    assert result["stored_seq"] == before["last_seq"] == 1
    assert result["source_latest_seq"] == before["latest_seq"]
    assert result["lineage_status"] == "pending"
    assert retirement.retire_copy(target, payload=closing.payload(), value=closing.value, local_gateway_id=TARGET) == result
    with pytest.raises(rooms.HostedRoomError):
        ingest(target, page_v2(source, spans, include_disbanded=True))


@pytest.mark.parametrize("damage", ["known_tail", "quarantine", "late_old_writer"])
def test_replacement_preserves_known_tail_and_unverified_evidence(pair, damage):
    source, target = pair
    entry = enroll(pair)
    copied_prefix(pair)
    spans = [{"gateway_id": HOME, "epoch": 1, "from_seq": 0}, transfer(source)]
    if damage == "known_tail":
        with rooms._transaction(target) as conn:
            conn.execute("UPDATE hosted_room_replicas SET latest_seq=?", (spans[-1]["from_seq"],))
    elif damage == "quarantine":
        with rooms._transaction(target) as conn:
            conn.execute("UPDATE hosted_room_replicas SET quarantined_at=1,quarantine_reason='preserve'")
    else:
        setup_successor(source, target, spans, old_id=entry["enrollment_id"])
        ingest(target, page_v2(source, spans))
        with sqlite3.connect(target) as conn:
            conn.execute("UPDATE hosted_room_replicas SET authority_gateway_id=?,authority_epoch=1", (HOME,))
        state = replicas.replica_state(target, room_id="room")
        assert state["safety_status"] == "quarantined"
        with pytest.raises(rooms.HostedRoomError):
            ingest(target, page_v2(source, spans))
        return
    with pytest.raises(rooms.HostedRoomError):
        setup_successor(source, target, spans, old_id=entry["enrollment_id"])
    assert retirement.current_target_enrollment(target, room_id="room", authority_gateway_id=HOME, authority_epoch=1)["enrollment_id"] == entry["enrollment_id"]


def test_stale_authority_obligations_are_blocked_without_rewriting_scope(pair):
    source, _ = pair
    entry = enroll(pair)
    with rooms._transaction(source) as conn:
        before = dict(conn.execute(f"SELECT * FROM {retirement.HOME_TABLE}").fetchone())
    transfer(source)
    assert retirement.pending_notice_ids(source, local_gateway_id=HOME) == []
    status = retirement.home_status(source)[0]
    assert status["state"] == "blocked_authority" and status["last_error"] == "stale_authority"
    with rooms._transaction(source) as conn:
        after = dict(conn.execute(f"SELECT * FROM {retirement.HOME_TABLE} WHERE enrollment_id=?", (entry["enrollment_id"],)).fetchone())
    for key in ("commitment", "nonce", "closing_value", "authority_epoch", "authority_gateway_id", "roster_sha256"):
        assert after[key] == before[key]


@pytest.mark.parametrize("mode", ["age", "rooms", "bytes"])
@pytest.mark.parametrize("state", ["valid", "quarantined", "wrong_lineage"])
def test_retired_partial_v2_prefix_is_reclaimed_only_for_its_lineage(pair, mode, state):
    from gateway import hosted_room_link_records as links
    from gateway.hosted_room_safety import _prune_disbanded_replicas_locked

    source, target = pair
    old = enroll(pair)
    copied_prefix(pair)
    spans = [{"gateway_id": HOME, "epoch": 1, "from_seq": 0}, transfer(source)]
    entry = setup_successor(source, target, spans, old_id=old["enrollment_id"])
    links.begin_room_link_retirement(source, room_id="room", authority_gateway_id=SUCCESSOR, authority_epoch=2)
    links.complete_room_link_retirement(source, room_id="room", authority_gateway_id=SUCCESSOR, authority_epoch=2)
    rooms.disband_room(source, room_id="room", expected_gateway_id=SUCCESSOR, expected_epoch=2)
    closing = retirement.materialize_notice(source, enrollment_id=entry["enrollment_id"],
        local_gateway_id=SUCCESSOR, secret_loader=lambda: SECRET)
    retired = retirement.retire_copy(target, payload=closing.payload(), value=closing.value, local_gateway_id=TARGET)
    assert retired["lineage_status"] == "pending" and retired["stored_seq"] == 1
    with retirement._transaction(target) as conn:
        if state == "quarantined":
            conn.execute("UPDATE hosted_room_replicas SET quarantine_reason='preserve',quarantined_at=1")
        elif state == "wrong_lineage":
            conn.execute(f"UPDATE {retirement.RETIREMENT_TABLE} SET lineage_sha256=?", ("0" * 64,))
        retired_at = conn.execute(f"SELECT retired_at FROM {retirement.RETIREMENT_TABLE}").fetchone()[0]
        if mode == "age":
            count = _prune_disbanded_replicas_locked(conn, now=retired_at + rooms.DISBANDED_REPLICA_RETENTION_SECONDS + 1)
        elif mode == "rooms":
            count = _prune_disbanded_replicas_locked(conn, now=None, max_replica_rooms=0)
        else:
            count = _prune_disbanded_replicas_locked(conn, now=None, max_replica_event_bytes=0)
        assert count == (1 if state == "valid" else 0)
        assert conn.execute("SELECT COUNT(*) FROM hosted_room_replica_events").fetchone()[0] == (0 if state == "valid" else 1)
        assert conn.execute("SELECT owner_kind FROM hosted_room_id_reservations WHERE room_id='room'").fetchone()[0] == "replica"
    if state == "valid":
        with pytest.raises(replicas.ReplicaHistoryExpiredError):
            replicas.replica_state(target, room_id="room")
        assert retirement.retire_copy(target, payload=closing.payload(), value=closing.value, local_gateway_id=TARGET) == retired
        with pytest.raises(rooms.HostedRoomError):
            ingest(target, page_v2(source, spans, include_disbanded=True))


def test_concurrent_retirement_replacement_has_exactly_one_committed_winner(pair, tmp_path):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    source, target = pair
    entry = enroll(pair)
    copied_prefix(pair)
    old = tmp_path / "old-close.db"
    fork(source, old)
    close(old)
    closing = notice(old, entry)
    spans = [{"gateway_id": HOME, "epoch": 1, "from_seq": 0}, transfer(source)]
    replacement = retirement.prepare_home_enrollment(source, room_id="room", target_install_id=TARGET,
        endpoint="https://participant.example", local_gateway_id=SUCCESSOR, secret=SECRET,
        enrollment_id="concurrent", replace_enrollment_id=entry["enrollment_id"])
    start = threading.Barrier(3)
    def attempt(replace):
        start.wait(timeout=5)
        try:
            if replace:
                retirement.enroll_target(target, enrollment=replacement, target_install_id=TARGET,
                    authority_history=spans, expected_enrollment_id=entry["enrollment_id"])
            else:
                retirement.retire_copy(target, payload=closing.payload(), value=closing.value, local_gateway_id=TARGET)
            return True
        except retirement.RetirementError:
            return False
    with ThreadPoolExecutor(max_workers=2) as pool:
        replaced = pool.submit(attempt, True)
        retired = pool.submit(attempt, False)
        start.wait(timeout=5)
        outcomes = (replaced.result(timeout=10), retired.result(timeout=10))
    assert sum(outcomes) == 1
    state = replicas.replica_state(target, room_id="room")
    assert state["last_seq"] == 1
    assert state["safety_status"] == ("passive" if outcomes[0] else "retired")

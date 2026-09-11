"""Real SQLite retirement invariants; no grants or cleanup authority are inferred."""

import json
import sqlite3
from dataclasses import replace
from contextlib import contextmanager

import pytest

from gateway import hosted_room_link_records as links
from gateway import hosted_room_replicas as replicas
from gateway import hosted_room_replica_retirement as retirement
from gateway import hosted_rooms as rooms

HOME = "install:home"
TARGET = "install:target"
OTHER = "install:other"
SECRET = b"home-only-test-secret-32-bytes-long"
MEMBERS = [
    {
        "member_id": "writer",
        "profile": "default",
        "handle": "writer",
        "target": {"kind": "local", "profile": "default"},
    },
    {
        "member_id": "reviewer",
        "profile": "reviewer",
        "handle": "reviewer",
        "target": {
            "kind": "peer",
            "peer_id": "target",
            "installation_id": TARGET,
            "profile": "reviewer",
            "capability_digest": "a" * 64,
        },
    },
    {
        "member_id": "other",
        "profile": "default",
        "handle": "other",
        "target": {
            "kind": "peer",
            "peer_id": "other",
            "installation_id": OTHER,
            "profile": "default",
            "capability_digest": "b" * 64,
        },
    },
]


@pytest.fixture
def pair(tmp_path):
    home, target = tmp_path / "home.db", tmp_path / "target.db"
    rooms.create_room(
        home,
        room_id="room",
        name="Workshop",
        members=MEMBERS,
        authority_gateway_id=HOME,
    )
    for index in range(3):
        rooms.append_event(
            home,
            room_id="room",
            event_id=f"message-{index}",
            kind="message.user",
            actor={"kind": "user", "id": "owner"},
            payload={"text": f"message {index}"},
            authority_gateway_id=HOME,
            authority_epoch=1,
        )
    return home, target


def prepare(home, **kwargs):
    return retirement.prepare_home_enrollment(
        home,
        room_id="room",
        target_install_id=kwargs.pop("target_install_id", TARGET),
        endpoint=kwargs.pop("endpoint", "https://participant.example"),
        local_gateway_id=HOME,
        secret=kwargs.pop("secret", SECRET),
        **kwargs,
    )


def enroll(pair):
    entry = prepare(pair[0])
    retirement.enroll_target(pair[1], enrollment=entry, target_install_id=TARGET)
    return entry


def freeze(home):
    links.begin_room_link_retirement(
        home, room_id="room", authority_gateway_id=HOME, authority_epoch=1
    )


def close(home):
    freeze(home)
    links.complete_room_link_retirement(
        home, room_id="room", authority_gateway_id=HOME, authority_epoch=1
    )
    return rooms.disband_room(
        home, room_id="room", expected_gateway_id=HOME, expected_epoch=1
    )


def notice(home, entry, loader=lambda: SECRET):
    return retirement.materialize_notice(
        home,
        enrollment_id=entry["enrollment_id"],
        local_gateway_id=HOME,
        secret_loader=loader,
    )


def copied_prefix(pair, count=1):
    page = rooms.read_events(pair[0], room_id="room", limit=count)
    replicas.ingest_page(
        pair[1], room_id="room", room_name="Workshop", members=MEMBERS, page=page
    )


def test_setup_is_idempotent_and_reveal_requires_canonical_close(pair):
    entry = enroll(pair)
    assert prepare(pair[0]) == entry
    assert prepare(pair[0], enrollment_id=entry["enrollment_id"]) == entry
    with pytest.raises(
        retirement.RetirementConflictError, match="disband has not completed"
    ):
        notice(pair[0], entry)
    freeze(pair[0])
    assert retirement.home_status(pair[0])[0]["state"] == "closing"
    with pytest.raises(
        retirement.RetirementConflictError, match="disband has not completed"
    ):
        notice(pair[0], entry)
    assert retirement.pending_notice_ids(pair[0], local_gateway_id=HOME) == []
    with pytest.raises(retirement.RetirementConflictError, match="already started"):
        prepare(pair[0], target_install_id=OTHER)
    close(pair[0])
    outgoing = notice(pair[0], entry)
    assert outgoing.enrollment_id in retirement.pending_notice_ids(
        pair[0], local_gateway_id=HOME
    )
    assert outgoing.value not in repr(outgoing)
    assert outgoing.value not in json.dumps(retirement.home_status(pair[0]))
    assert outgoing.value not in json.dumps(entry)


@pytest.mark.parametrize("count", [0, 1, 3])
def test_retirement_preserves_actual_coverage_and_is_idempotent(pair, count):
    entry = enroll(pair)
    if count:
        copied_prefix(pair, count)
    close(pair[0])
    outgoing = notice(pair[0], entry)
    result = retirement.retire_copy(
        pair[1],
        payload=outgoing.payload(),
        value=outgoing.value,
        local_gateway_id=TARGET,
    )
    assert result["stored_seq"] == count
    assert result["source_latest_seq"] == (3 if count else 0)
    if count:
        state = replicas.replica_state(pair[1], room_id="room")
        assert state["safety_status"] == "retired"
        assert state["disbanded_at"] is None
    assert result == retirement.retire_copy(
        pair[1],
        payload=outgoing.payload(),
        value=outgoing.value,
        local_gateway_id=TARGET,
    )
    with rooms._transaction(pair[1]) as conn:
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM hosted_room_replica_events WHERE kind='room.disbanded'"
            ).fetchone()[0]
            == 0
        )
    retirement.acknowledge_notice(pair[0], notice=outgoing, response=result)
    assert retirement.home_status(pair[0])[0]["state"] == "acknowledged"
    assert retirement.pending_notice_ids(pair[0], local_gateway_id=HOME) == []


def test_retired_copy_rejects_late_pages_and_owner_reenrollment(pair):
    entry = enroll(pair)
    copied_prefix(pair)
    close(pair[0])
    outgoing = notice(pair[0], entry)
    retirement.retire_copy(
        pair[1],
        payload=outgoing.payload(),
        value=outgoing.value,
        local_gateway_id=TARGET,
    )
    with pytest.raises(replicas.ReplicaHistoryExpiredError):
        replicas.ingest_page(
            pair[1],
            room_id="room",
            room_name="Workshop",
            members=MEMBERS,
            page=rooms.read_events(pair[0], room_id="room", include_disbanded=True),
        )
    with pytest.raises(retirement.RetirementConflictError, match="reenrolled"):
        retirement.enroll_target(pair[1], enrollment=entry, target_install_id=TARGET)
    with pytest.raises(rooms.RoomConflictError):
        rooms.create_room(
            pair[1],
            room_id="room",
            name="Resurrected",
            members=MEMBERS,
            authority_gateway_id=TARGET,
        )
    with (
        sqlite3.connect(pair[1]) as conn,
        pytest.raises(sqlite3.IntegrityError, match="retired"),
    ):
        conn.execute("""INSERT INTO hosted_room_replica_events VALUES
            ('room',2,'late','message.user','{}',1,'{}',0)""")


@pytest.mark.parametrize(
    "changed",
    [
        {"room_id": "other"},
        {"authority_gateway_id": OTHER},
        {"authority_epoch": 2},
        {"authority_epoch": True},
        {"target_install_id": OTHER},
        {"enrollment_id": "other"},
    ],
)
def test_retirement_scope_cannot_be_rebound(pair, changed):
    entry = enroll(pair)
    close(pair[0])
    outgoing = notice(pair[0], entry)
    with pytest.raises(retirement.RetirementError):
        retirement.retire_copy(
            pair[1],
            payload={**outgoing.payload(), **changed},
            value=outgoing.value,
            local_gateway_id=TARGET,
        )


def test_commitment_copied_to_another_enrollment_does_not_authorize_it(pair):
    entry = enroll(pair)
    altered = {
        **entry,
        "enrollment_id": "another-enrollment",
        "room_id": "another-room",
    }
    retirement.enroll_target(pair[1], enrollment=altered, target_install_id=TARGET)
    close(pair[0])
    outgoing = notice(pair[0], entry)
    with pytest.raises(retirement.RetirementAuthorizationError):
        retirement.retire_copy(
            pair[1],
            payload={
                **outgoing.payload(),
                "enrollment_id": "another-enrollment",
                "room_id": "another-room",
            },
            value=outgoing.value,
            local_gateway_id=TARGET,
        )


def test_ordinary_grant_value_cannot_retire_a_copy(pair):
    entry = enroll(pair)
    close(pair[0])
    outgoing = notice(pair[0], entry)
    with pytest.raises(retirement.RetirementAuthorizationError):
        retirement.retire_copy(
            pair[1],
            payload=outgoing.payload(),
            value="not-a-retirement-capability",
            local_gateway_id=TARGET,
        )


def test_owner_replacement_rejects_old_value_without_losing_historical_target(pair):
    original = enroll(pair)
    replacement = prepare(
        pair[0],
        enrollment_id="replacement",
        replace_enrollment_id=original["enrollment_id"],
    )
    retirement.enroll_target(
        pair[1],
        enrollment=replacement,
        target_install_id=TARGET,
        expected_enrollment_id=original["enrollment_id"],
    )
    close(pair[0])
    assert len(retirement.pending_notice_ids(pair[0], local_gateway_id=HOME)) == 2
    old = notice(pair[0], original)
    with pytest.raises(retirement.RetirementAuthorizationError, match="replaced"):
        retirement.retire_copy(
            pair[1], payload=old.payload(), value=old.value, local_gateway_id=TARGET
        )
    new = notice(pair[0], replacement)
    assert retirement.retire_copy(
        pair[1], payload=new.payload(), value=new.value, local_gateway_id=TARGET
    )["retired"]


def test_revocation_and_stale_replacement_do_not_reactivate_cleanup(pair):
    original = enroll(pair)
    replacement = prepare(
        pair[0],
        enrollment_id="replacement",
        replace_enrollment_id=original["enrollment_id"],
    )
    retirement.revoke_target_enrollment(
        pair[1], room_id="room", enrollment_id=original["enrollment_id"]
    )
    with pytest.raises(retirement.RetirementConflictError, match="expected state"):
        retirement.enroll_target(
            pair[1],
            enrollment=replacement,
            target_install_id=TARGET,
            expected_enrollment_id=original["enrollment_id"],
        )
    retirement.enroll_target(
        pair[1],
        enrollment=replacement,
        target_install_id=TARGET,
        expected_enrollment_id=original["enrollment_id"],
        expected_state="revoked",
    )
    close(pair[0])
    old = notice(pair[0], original)
    with pytest.raises(retirement.RetirementAuthorizationError):
        retirement.retire_copy(
            pair[1], payload=old.payload(), value=old.value, local_gateway_id=TARGET
        )


def test_lost_ack_reuses_materialized_value_even_after_home_key_loss(pair):
    entry = enroll(pair)
    close(pair[0])
    outgoing = notice(pair[0], entry)
    result = retirement.retire_copy(
        pair[1],
        payload=outgoing.payload(),
        value=outgoing.value,
        local_gateway_id=TARGET,
    )

    def lost():
        raise FileNotFoundError("test key unavailable")

    resumed = notice(pair[0], entry, lost)
    assert resumed == outgoing
    assert result == retirement.retire_copy(
        pair[1], payload=resumed.payload(), value=resumed.value, local_gateway_id=TARGET
    )
    retirement.acknowledge_notice(pair[0], notice=resumed, response=result)


def test_key_rotation_before_close_is_explicit_and_not_a_new_commitment(pair):
    entry = enroll(pair)
    with pytest.raises(retirement.RetirementKeyUnavailable):
        prepare(pair[0], secret=b"different-key-for-test-only-32-bytes")
    close(pair[0])
    with pytest.raises(retirement.RetirementKeyUnavailable):
        notice(pair[0], entry, lambda: b"different-key-for-test-only-32-bytes")
    assert retirement.home_status(pair[0])[0]["state"] == "closed"


def test_wrong_ack_never_completes_the_notice(pair):
    entry = enroll(pair)
    close(pair[0])
    outgoing = notice(pair[0], entry)
    result = retirement.retire_copy(
        pair[1],
        payload=outgoing.payload(),
        value=outgoing.value,
        local_gateway_id=TARGET,
    )
    with pytest.raises(retirement.RetirementConflictError):
        retirement.acknowledge_notice(
            pair[0], notice=replace(outgoing, enrollment_id="another"), response=result
        )
    assert retirement.home_status(pair[0])[0]["state"] == "ready"


def test_pending_obligations_are_not_evicted_for_capacity(pair, monkeypatch):
    entry = enroll(pair)
    monkeypatch.setattr(retirement, "MAX_PENDING_ENROLLMENTS", 1)
    with pytest.raises(retirement.RetirementCapacityError):
        prepare(pair[0], target_install_id=OTHER)
    assert retirement.home_status(pair[0])[0]["enrollment_id"] == entry["enrollment_id"]


def test_canonical_payload_pruning_does_not_erase_close_delivery(pair):
    entry = enroll(pair)
    close(pair[0])
    rooms.prune_disbanded_rooms(pair[0], now=10**12)
    outgoing = notice(pair[0], entry)
    assert outgoing.enrollment_id == entry["enrollment_id"]


def test_exact_canonical_disband_also_closes_preexisting_unfrozen_enrollment(pair):
    entry = enroll(pair)
    rooms.disband_room(
        pair[0], room_id="room", expected_gateway_id=HOME, expected_epoch=1
    )
    assert notice(pair[0], entry).enrollment_id == entry["enrollment_id"]


def test_enrollment_never_overwrites_local_authority_or_quarantine(pair):
    entry = prepare(pair[0])
    rooms.create_room(
        pair[1],
        room_id="room",
        name="Local",
        members=MEMBERS,
        authority_gateway_id=TARGET,
    )
    with pytest.raises(
        retirement.RetirementConflictError, match="locally authoritative"
    ):
        retirement.enroll_target(pair[1], enrollment=entry, target_install_id=TARGET)


def test_reclamation_of_retired_partial_copy_keeps_denial_and_active_history(pair):
    from gateway.hosted_room_safety import _prune_disbanded_replicas_locked

    entry = enroll(pair)
    copied_prefix(pair)
    rooms.create_room(
        pair[0],
        room_id="still-active",
        name="Active",
        members=MEMBERS,
        authority_gateway_id=HOME,
    )
    rooms.append_event(
        pair[0],
        room_id="still-active",
        event_id="keep",
        kind="message.user",
        actor={"kind": "user", "id": "owner"},
        payload={"text": "keep this"},
        authority_gateway_id=HOME,
        authority_epoch=1,
    )
    replicas.ingest_page(
        pair[1],
        room_id="still-active",
        room_name="Active",
        members=MEMBERS,
        page=rooms.read_events(pair[0], room_id="still-active"),
    )
    close(pair[0])
    outgoing = notice(pair[0], entry)
    receipt = retirement.retire_copy(
        pair[1],
        payload=outgoing.payload(),
        value=outgoing.value,
        local_gateway_id=TARGET,
    )
    with rooms._transaction(pair[1], immediate=True) as conn:
        assert (
            _prune_disbanded_replicas_locked(conn, now=None, max_replica_event_bytes=0)
            == 1
        )
    assert replicas.replica_state(pair[1], room_id="still-active")["last_seq"] == 1
    with pytest.raises(replicas.ReplicaHistoryExpiredError):
        replicas.replica_state(pair[1], room_id="room")
    assert receipt == retirement.retire_copy(
        pair[1],
        payload=outgoing.payload(),
        value=outgoing.value,
        local_gateway_id=TARGET,
    )
    with pytest.raises(retirement.RetirementConflictError):
        retirement.enroll_target(pair[1], enrollment=entry, target_install_id=TARGET)


def test_quarantine_remains_non_prunable_after_retirement(pair):
    from gateway.hosted_room_safety import _prune_disbanded_replicas_locked

    entry = enroll(pair)
    copied_prefix(pair)
    close(pair[0])
    outgoing = notice(pair[0], entry)
    retirement.retire_copy(
        pair[1],
        payload=outgoing.payload(),
        value=outgoing.value,
        local_gateway_id=TARGET,
    )
    with rooms._transaction(pair[1], immediate=True) as conn:
        conn.execute(
            "INSERT INTO hosted_room_quarantine VALUES ('room','test-evidence',0)"
        )
        assert (
            _prune_disbanded_replicas_locked(
                conn, now=10**12, max_replica_event_bytes=0
            )
            == 0
        )
        assert (
            conn.execute("SELECT COUNT(*) FROM hosted_room_replica_events").fetchone()[
                0
            ]
            == 1
        )


def test_one_target_ack_supersedes_only_its_historical_enrollments(pair):
    first = enroll(pair)
    second = prepare(
        pair[0],
        enrollment_id="replacement",
        replace_enrollment_id=first["enrollment_id"],
    )
    other_target = prepare(pair[0], target_install_id=OTHER)
    retirement.enroll_target(
        pair[1],
        enrollment=second,
        target_install_id=TARGET,
        expected_enrollment_id=first["enrollment_id"],
    )
    close(pair[0])
    outgoing = notice(pair[0], second)
    response = retirement.retire_copy(
        pair[1],
        payload=outgoing.payload(),
        value=outgoing.value,
        local_gateway_id=TARGET,
    )
    retirement.acknowledge_notice(pair[0], notice=outgoing, response=response)
    states = {row["enrollment_id"]: row for row in retirement.home_status(pair[0])}
    assert states[first["enrollment_id"]]["state"] == "superseded"
    assert states[first["enrollment_id"]]["superseded_by"] == second["enrollment_id"]
    assert states[other_target["enrollment_id"]]["state"] == "closed"


def test_canonical_close_and_journal_readiness_roll_back_together(pair):
    entry = enroll(pair)
    freeze(pair[0])
    links.complete_room_link_retirement(
        pair[0], room_id="room", authority_gateway_id=HOME, authority_epoch=1
    )
    with sqlite3.connect(pair[0]) as conn:
        conn.execute(f"""CREATE TRIGGER fail_retirement_ready BEFORE UPDATE ON {retirement.HOME_TABLE}
            WHEN NEW.state='closed' BEGIN SELECT RAISE(ABORT,'test write failure'); END""")
    with pytest.raises(sqlite3.IntegrityError, match="test write failure"):
        rooms.disband_room(
            pair[0], room_id="room", expected_gateway_id=HOME, expected_epoch=1
        )
    assert rooms.room_state(pair[0], room_id="room").get("disbanded_at") is None
    assert rooms.room_state(pair[0], room_id="room")["latest_seq"] == 3
    assert retirement.pending_notice_ids(pair[0], local_gateway_id=HOME) == []
    with pytest.raises(retirement.RetirementConflictError):
        notice(pair[0], entry)


def test_target_write_failure_does_not_create_denial_or_retirement(pair):
    entry = enroll(pair)
    close(pair[0])
    outgoing = notice(pair[0], entry)
    with sqlite3.connect(pair[1]) as conn:
        conn.execute(f"""CREATE TRIGGER fail_retirement_insert BEFORE INSERT ON {retirement.RETIREMENT_TABLE}
            BEGIN SELECT RAISE(ABORT,'test full disk'); END""")
    with pytest.raises(sqlite3.IntegrityError, match="test full disk"):
        retirement.retire_copy(
            pair[1],
            payload=outgoing.payload(),
            value=outgoing.value,
            local_gateway_id=TARGET,
        )
    with rooms._transaction(pair[1]) as conn:
        assert not retirement.copy_retired_locked(conn, "room")
        assert (
            conn.execute(
                "SELECT 1 FROM hosted_room_id_reservations WHERE room_id='room'"
            ).fetchone()
            is None
        )


def test_invalid_capability_never_takes_the_writer_transaction(pair, monkeypatch):
    entry = enroll(pair)
    close(pair[0])
    outgoing = notice(pair[0], entry)

    def writer_must_not_run(_db):
        raise AssertionError("unauthorized write transaction")

    monkeypatch.setattr(retirement, "_transaction", writer_must_not_run)
    wrong = outgoing.value[:-1] + ("A" if outgoing.value[-1] != "A" else "B")
    with pytest.raises(retirement.RetirementAuthorizationError):
        retirement.retire_copy(
            pair[1], payload=outgoing.payload(), value=wrong, local_gateway_id=TARGET
        )


def test_completed_ack_replay_is_read_only(pair, monkeypatch):
    entry = enroll(pair)
    close(pair[0])
    outgoing = notice(pair[0], entry)
    receipt = retirement.retire_copy(
        pair[1],
        payload=outgoing.payload(),
        value=outgoing.value,
        local_gateway_id=TARGET,
    )

    def writer_must_not_run(_db):
        raise AssertionError("completed replay must be read-only")

    monkeypatch.setattr(retirement, "_transaction", writer_must_not_run)
    assert (
        retirement.retire_copy(
            pair[1],
            payload=outgoing.payload(),
            value=outgoing.value,
            local_gateway_id=TARGET,
        )
        == receipt
    )


def test_revocation_between_read_validation_and_write_wins(pair, monkeypatch):
    entry = enroll(pair)
    close(pair[0])
    outgoing = notice(pair[0], entry)
    original = retirement._transaction

    @contextmanager
    def revoke_then_enter(db):
        monkeypatch.setattr(retirement, "_transaction", original)
        retirement.revoke_target_enrollment(
            db, room_id="room", enrollment_id=entry["enrollment_id"]
        )
        with original(db) as conn:
            yield conn

    monkeypatch.setattr(retirement, "_transaction", revoke_then_enter)
    with pytest.raises(retirement.RetirementAuthorizationError, match="revoked"):
        retirement.retire_copy(
            pair[1],
            payload=outgoing.payload(),
            value=outgoing.value,
            local_gateway_id=TARGET,
        )


def test_late_setup_confirmation_cannot_unfreeze_disband(pair):
    entry = enroll(pair)
    proof = retirement.current_target_enrollment(
        pair[1], room_id="room", authority_gateway_id=HOME, authority_epoch=1
    )
    freeze(pair[0])
    assert not retirement.confirm_home_enrollment(
        pair[0], enrollment_id=entry["enrollment_id"], proof=proof
    )
    assert retirement.home_status(pair[0])[0]["state"] == "closing"


def test_confirmation_is_bound_to_exact_owner_enrollment(pair):
    entry = enroll(pair)
    proof = retirement.current_target_enrollment(
        pair[1], room_id="room", authority_gateway_id=HOME, authority_epoch=1
    )
    assert not retirement.confirm_home_enrollment(
        pair[0],
        enrollment_id=entry["enrollment_id"],
        proof={**proof, "commitment": "a" * 64},
    )
    assert retirement.home_status(pair[0])[0]["state"] == "prepared"
    assert retirement.confirm_home_enrollment(
        pair[0], enrollment_id=entry["enrollment_id"], proof=proof
    )
    assert retirement.home_status(pair[0])[0]["state"] == "enrolled"


def test_owner_enrollment_fences_first_page_scope(pair):
    enroll(pair)
    page = rooms.read_events(pair[0], room_id="room")
    page["authority"] = {"gateway_id": OTHER, "epoch": 1}
    with pytest.raises(replicas.ReplicaError, match="enrollment"):
        replicas.ingest_page(
            pair[1], room_id="room", room_name="Workshop", members=MEMBERS, page=page
        )


def test_corrupt_legacy_copy_is_not_retired_and_made_prunable(pair):
    entry = enroll(pair)
    copied_prefix(pair)
    with sqlite3.connect(pair[1]) as conn:
        conn.execute(
            "UPDATE hosted_room_replicas SET last_seq=99,latest_seq=99 WHERE room_id='room'"
        )
    close(pair[0])
    outgoing = notice(pair[0], entry)
    with pytest.raises(retirement.RetirementConflictError):
        retirement.retire_copy(
            pair[1],
            payload=outgoing.payload(),
            value=outgoing.value,
            local_gateway_id=TARGET,
        )
    with rooms._transaction(pair[1]) as conn:
        assert not retirement.copy_retired_locked(conn, "room")
        assert (
            conn.execute("SELECT COUNT(*) FROM hosted_room_replica_events").fetchone()[
                0
            ]
            == 1
        )


def test_old_writer_cannot_change_a_retired_copy_header_or_event(pair):
    entry = enroll(pair)
    copied_prefix(pair)
    close(pair[0])
    outgoing = notice(pair[0], entry)
    retirement.retire_copy(
        pair[1],
        payload=outgoing.payload(),
        value=outgoing.value,
        local_gateway_id=TARGET,
    )
    for sql in (
        "UPDATE hosted_room_replicas SET authority_epoch=2 WHERE room_id='room'",
        "UPDATE hosted_room_replica_events SET payload_json='{}' WHERE room_id='room'",
    ):
        with (
            sqlite3.connect(pair[1]) as conn,
            pytest.raises(sqlite3.IntegrityError, match="retired"),
        ):
            conn.execute(sql)

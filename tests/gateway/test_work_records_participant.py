"""A participant keeps task evidence only beside an active copy, under a live grant, never as authority."""

import copy
import json
import sqlite3
import time
from contextlib import closing, contextmanager

import pytest

from gateway import hosted_room_peer as peer
from gateway import hosted_room_replica_ingress as ingress
from gateway import hosted_room_replicas as replicas
from gateway import hosted_room_work_records as records
from gateway import hosted_room_work_storage as storage
from gateway import hosted_rooms as rooms
from gateway.hosted_rooms_common import open_sqlite
from tests.gateway.fixtures.passive_copy import (  # noqa: F401
    EVIDENCE, HOME, KEY, MEMBERS, SECRET, TARGET, admit, append, copying, grant, pair, reserve, start)


def rows(path, table=records.TARGET_TABLE):
    with closing(open_sqlite(path)) as conn:
        return [dict(row) for row in conn.execute(f"SELECT * FROM {table} ORDER BY room_id,producer_epoch")]


def evidence(path):
    return replicas.copy_state(path, room_id="room")["work_records"]


@pytest.fixture
def copied(tmp_path):
    """A home with one admitted task, and a participant holding a copy of its history."""
    source, target = tmp_path / "home.db", tmp_path / "participant.db"
    rooms.create_room(source, room_id="room", name="Workshop", members=MEMBERS, authority_gateway_id=HOME)
    append(source, "hello")
    admit(source)
    token = grant(permissions=("status", *EVIDENCE))
    reserve(target, token)
    ingress.ingest_granted_page(target, token=token, secret=SECRET, target_install_id=TARGET, target_profile="default",
                                room_id="room", room_name="Workshop", members=MEMBERS,
                                page=rooms.read_events(source, room_id="room"))
    return source, target, token


def deliver(target, token, record):
    return records.ingest(target, record=record, token=token, secret=SECRET, target_install_id=TARGET,
                          target_profile="default")


def capture(source):
    return records.capture(source, room_id="room", local_gateway_id=HOME)


def test_a_whole_record_is_retained_passively_and_summarized_without_private_content(copied):
    source, target, token = copied
    first = capture(source)
    ack = deliver(target, token, first)
    assert ack == records.acknowledgement(first) == {
        "room_id": "room", "revision": first["revision"], "digest": first["digest"], "passive": True}
    assert deliver(target, token, first) == ack
    summary = evidence(target)
    assert (summary["tasks"][0]["task_id"], summary["phases"], summary["source_loss_safe"]) == ("task", {"queued": 1}, False)
    assert "PRIVATE_PROMPT" not in json.dumps(summary)
    assert rooms.list_rooms(target) == []


@pytest.mark.parametrize("mutation", ["digest", "revision", "prefix", "extra", "authority", "roster"])
def test_revision_scope_prefix_and_privacy_schema_reject_conflicts(copied, mutation):
    source, target, token = copied
    first = capture(source)
    deliver(target, token, first)
    bad = copy.deepcopy(first)
    if mutation == "digest":
        bad["digest"] = "a" * 64
    elif mutation == "revision":
        bad["tasks"][0]["phase"] = "running"
    elif mutation == "prefix":
        bad["history"]["event_sha256"] = "a" * 64
    elif mutation == "extra":
        bad["tasks"][0]["prompt"] = "MUST_NOT_STORE"
    elif mutation == "authority":
        bad["authority"]["epoch"] = 2
    else:
        bad["roster_sha256"] = "a" * 64
    if mutation != "digest":
        bad["digest"] = records.digest({k: v for k, v in bad.items() if k not in {"revision", "digest"}})
    with pytest.raises((records.WorkRecordError, peer.HostedRoomGrantError)):
        deliver(target, token, bad)
    assert evidence(target)["digest"] == first["digest"]


def test_a_stale_revision_cannot_replace_a_newer_record(copied):
    source, target, token = copied
    first = capture(source)
    deliver(target, token, first)
    start(source)
    second = capture(source)
    deliver(target, token, second)
    with pytest.raises(records.WorkRecordError):
        deliver(target, token, first)
    assert evidence(target)["revision"] == second["revision"]


@pytest.mark.parametrize("scope", ["room", "target", "profile", "expired", "history_only"])
def test_a_wrong_expired_or_history_only_grant_never_writes(copied, scope):
    source, target, token = copied
    claims = peer.decode_room_grant(SECRET, token, permission=records.PERMISSION)
    fields = {k: claims[k] for k in ("grant_id", "room_id", "home_install_id", "authority_gateway_id",
              "authority_epoch", "member_id", "target_install_id", "target_profile", "execution_policy_digest",
              "permissions")}
    changes = {"room": {"room_id": "wrong"}, "target": {"target_install_id": "wrong"},
               "profile": {"target_profile": "wrong"}, "expired": {"issued_at": time.time() - 100, "ttl_seconds": 1},
               "history_only": {"permissions": ("status", "replicate")}}
    bad = peer.issue_room_grant(SECRET, **{**fields, **changes[scope]})
    with pytest.raises((peer.HostedRoomGrantError, records.WorkRecordError)):
        deliver(target, bad, capture(source))
    assert evidence(target)["availability"] == "not_retained"


def test_revocation_after_validation_is_refused_inside_the_writer(copied, monkeypatch):
    source, target, token = copied
    claims = peer.decode_room_grant(SECRET, token, permission=records.PERMISSION)
    original = records.validate

    def revoke(value):
        monkeypatch.setattr(records, "validate", original)
        checked = original(value)
        rooms.revoke_room_grant_scope(target, claims=claims, expires_at=claims["status_expires_at"])
        return checked

    monkeypatch.setattr(records, "validate", revoke)
    with pytest.raises(peer.HostedRoomGrantError):
        deliver(target, token, capture(source))
    assert evidence(target)["availability"] == "not_retained"


def test_a_withdrawn_grant_never_reaches_the_replica_writer(copied, monkeypatch):
    source, target, token = copied
    claims = peer.decode_room_grant(SECRET, token, permission=records.PERMISSION)
    rooms.revoke_room_grant_scope(target, claims=claims, expires_at=claims["status_expires_at"])
    entered = []
    original = replicas._replica_transaction

    @contextmanager
    def observe(*args, **kwargs):
        entered.append(True)
        with original(*args, **kwargs) as conn:
            yield conn

    monkeypatch.setattr(replicas, "_replica_transaction", observe)
    with pytest.raises(peer.HostedRoomGrantError, match="revoked|no longer current"):
        deliver(target, token, capture(source))
    assert entered == []


def test_withdrawal_between_preflight_and_writer_is_refused_before_replica_maintenance(copied, monkeypatch):
    """The grant check runs inside the writer, after its lock wait, and before schema or audit work."""
    source, target, token = copied
    record = capture(source)
    claims = peer.decode_room_grant(SECRET, token, permission=records.PERMISSION)
    events = []
    original_authorize = ingress.authorize_granted_room
    original_writer, original_schema = replicas._replica_transaction, replicas._initialize_replica_schema

    def observe_authorize(**kwargs):
        check = original_authorize(**kwargs)

        def checked(conn):
            events.append("grant-check-in-writer" if conn.in_transaction else "grant-preflight")
            return check(conn)
        return checked

    @contextmanager
    def withdraw_then_write(*args, **kwargs):
        rooms.revoke_room_grant_scope(target, claims=claims, expires_at=claims["status_expires_at"])
        events.append("withdrawn")
        with original_writer(*args, **kwargs) as conn:
            yield conn

    def observe_schema(conn):
        events.append("replica-schema")
        return original_schema(conn)

    monkeypatch.setattr(ingress, "authorize_granted_room", observe_authorize)
    monkeypatch.setattr(replicas, "_replica_transaction", withdraw_then_write)
    monkeypatch.setattr(replicas, "_initialize_replica_schema", observe_schema)
    with pytest.raises(peer.HostedRoomGrantError, match="revoked|no longer current"):
        deliver(target, token, record)
    assert events == ["grant-preflight", "withdrawn", "grant-check-in-writer"]


@pytest.mark.parametrize("stale", [False, True], ids=["missing-database", "stale-schema"])
def test_direct_ingress_never_creates_or_migrates_a_store(tmp_path, stale):
    target = tmp_path / "participant.db"
    if stale:
        with sqlite3.connect(target) as conn:
            conn.execute("CREATE TABLE unrelated (id INTEGER)")
    with pytest.raises(peer.HostedRoomGrantError, match="grant state is unavailable"):
        records.ingest(target, record=_unavailable_record(), token=grant(permissions=(records.PERMISSION,)),
                       secret=SECRET, target_install_id=TARGET, target_profile="default")
    if stale:
        with sqlite3.connect(target) as conn:
            assert conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall() == [("unrelated",)]
    else:
        assert not target.exists()


@pytest.mark.parametrize("stale", [False, True], ids=["current-root", "grant-readable-stale-root"])
def test_direct_ingress_requires_a_current_store_without_repairing_it(copied, stale):
    _, target, token = copied
    with sqlite3.connect(target) as conn:
        if stale:
            conn.execute("DROP INDEX idx_hosted_room_events_cursor")
        before = conn.execute("SELECT type,name,sql FROM sqlite_master ORDER BY type,name").fetchall()
    try:
        result = deliver(target, token, _unavailable_record())
    except peer.HostedRoomGrantError as exc:
        result = exc
    if stale:
        # Read with plain SQLite: the room-store connector would itself repair the index.
        with sqlite3.connect(target) as conn:
            assert conn.execute("SELECT type,name,sql FROM sqlite_master ORDER BY type,name").fetchall() == before
        assert "grant state is unavailable" in str(result)
    else:
        assert result["passive"] is True


def _unavailable_record():
    content = {
        "version": 1, "room_id": "room", "home_install_id": HOME, "authority": {"gateway_id": HOME, "epoch": 1},
        "roster_sha256": records.roster_digest(MEMBERS), "history": {"seq": 0, "event_sha256": records.digest([])},
        "availability": "unavailable", "reason": "task_store_missing", "tasks": [], "receipts": [],
        "limitations": records.LIMITATIONS,
        "stop": {"closing": False, "revocation_complete": False, "seq": 0, "cancel_id": None}}
    return {**content, "revision": 1, "digest": records.digest(content)}


def test_a_quarantined_copy_refuses_records_and_reports_them_unavailable(copied):
    source, target, token = copied
    record = capture(source)
    deliver(target, token, record)
    with sqlite3.connect(target) as conn:
        conn.execute("UPDATE hosted_room_replicas SET quarantine_reason='test' WHERE room_id='room'")
    with pytest.raises(records.WorkRecordError):
        deliver(target, token, record)
    assert evidence(target)["availability"] == "unavailable"


def test_an_unaudited_old_copy_is_audited_before_a_record_is_admitted(copied):
    source, target, token = copied
    append(source, "second")
    page = rooms.read_events(source, room_id="room", since_seq=1)
    ingress.ingest_granted_page(target, token=token, secret=SECRET, target_install_id=TARGET, target_profile="default",
                                room_id="room", room_name="Workshop", members=MEMBERS, page=page)
    with sqlite3.connect(target) as old:
        old.execute("DELETE FROM hosted_room_replica_events WHERE room_id='room' AND seq=1")
        assert old.execute("SELECT last_seq,quarantine_reason FROM hosted_room_replicas").fetchone() == (2, None)
    with pytest.raises(records.WorkRecordError):
        deliver(target, token, capture(source))
    with sqlite3.connect(target) as conn:
        assert conn.execute("SELECT quarantine_reason FROM hosted_room_replicas").fetchone()[0] == "non_contiguous_history"
        assert not conn.execute("SELECT 1 FROM sqlite_master WHERE name=?", (records.TARGET_TABLE,)).fetchone()


def test_a_disbanded_copy_reclaims_its_records_and_refuses_late_ones(copied):
    source, target, token = copied
    record = capture(source)
    deliver(target, token, record)
    rooms.disband_room(source, room_id="room", expected_gateway_id=HOME, expected_epoch=1)
    ingress.ingest_granted_page(target, token=token, secret=SECRET, target_install_id=TARGET, target_profile="default",
                                room_id="room", room_name="Workshop", members=MEMBERS,
                                page=rooms.read_events(source, room_id="room", since_seq=1, include_disbanded=True))
    with pytest.raises((records.WorkRecordError, peer.HostedRoomGrantError)):
        deliver(target, token, record)
    state = replicas.copy_state(target, room_id="room")
    assert state["work_records"]["availability"] == "not_retained" and state["disbanded_at"] is not None


@pytest.mark.parametrize("operation", ["disband", "delete"])
def test_an_older_history_writer_reclaims_records_but_not_the_copy_identity(copied, operation):
    source, target, token = copied
    record = capture(source)
    deliver(target, token, record)
    with sqlite3.connect(target) as old:
        old.execute("PRAGMA foreign_keys=OFF")
        old.execute("UPDATE hosted_room_replicas SET disbanded_at=123 WHERE room_id='room'" if operation == "disband"
                    else "DELETE FROM hosted_room_replicas WHERE room_id='room'")
        assert old.execute(f"SELECT COUNT(*) FROM {records.TARGET_TABLE}").fetchone()[0] == 0
        assert old.execute("SELECT owner_kind FROM hosted_room_id_reservations WHERE room_id='room'").fetchone()[0] == "replica"
        with pytest.raises(sqlite3.IntegrityError):
            old.execute(f"INSERT INTO {records.TARGET_TABLE} (room_id,revision,digest,record_json,producer_gateway_id,"
                        "producer_epoch,disposition) VALUES (?,?,?,?,?,?,'current')",
                        ("room", record["revision"], record["digest"], records.encode(record), HOME, 1))


@pytest.mark.parametrize("damage", ["digest", "revision"])
def test_a_damaged_stored_record_is_never_laundered_by_a_new_delivery(copied, damage):
    source, target, token = copied
    record = capture(source)
    deliver(target, token, record)
    with sqlite3.connect(target) as conn:
        conn.execute(f"UPDATE {records.TARGET_TABLE} SET {damage}=?", (0 if damage == "revision" else "wrong",))
        before = conn.execute(f"SELECT revision,digest,record_json FROM {records.TARGET_TABLE}").fetchone()
    with pytest.raises(records.WorkRecordError):
        deliver(target, token, record)
    with sqlite3.connect(target) as conn:
        assert conn.execute(f"SELECT revision,digest,record_json FROM {records.TARGET_TABLE}").fetchone() == before
        assert conn.execute(f"SELECT disposition FROM {records.TARGET_TABLE}").fetchone()[0] == "invalid"


def _damaged_copy(copying):
    """A delivered record, then raw damage to both it and its copy's history."""
    copying.pub._publish_one(KEY)
    copying.pub._publish_one(KEY)
    assert copying.records
    with sqlite3.connect(copying.target) as conn:
        conn.execute(f"UPDATE {records.TARGET_TABLE} SET digest='wrong' WHERE room_id='room'")
        conn.execute("UPDATE hosted_room_replica_events SET payload_json='not-json' WHERE room_id='room'")
    return rows(copying.target)


def test_only_the_inspected_copy_is_classified_and_initialization_rewrites_nothing(copying):
    damaged = _damaged_copy(copying)
    with rooms._transaction(copying.target, immediate=True) as conn:
        records.initialize(conn)
    assert rows(copying.target) == damaged
    assert replicas.copy_state(copying.target, room_id="room")["safety_status"] == "quarantined"
    assert rows(copying.target) == [{**damaged[0], "disposition": "invalid"}]


def test_a_damaged_copy_never_blocks_an_unrelated_room_on_the_same_store(copying):
    before = _damaged_copy(copying)
    rooms.create_room(copying.target, room_id="healthy", name="Unrelated source", members=MEMBERS,
                      authority_gateway_id=HOME)
    assert replicas.copy_state(copying.target, room_id="room")["safety_status"] == "quarantined"
    first = records.capture(copying.target, room_id="healthy", local_gateway_id=HOME)
    assert records.capture(copying.target, room_id="healthy", local_gateway_id=HOME) == first
    assert rows(copying.target) == [{**before[0], "disposition": "invalid"}]


def test_an_inspected_historical_scope_is_projected_invalid_without_rewriting_it(copying):
    _damaged_copy(copying)
    with sqlite3.connect(copying.target) as conn:
        conn.execute(f"UPDATE {records.TARGET_TABLE} SET disposition='historical'")
    before = rows(copying.target)
    assert replicas.copy_state(copying.target, room_id="room")["safety_status"] == "quarantined"
    with closing(open_sqlite(copying.target)) as conn:
        item = records.summary_locked(conn, "room")["scopes"][0]
    assert (item["availability"], item["disposition"], item["incompleteness"]) == (
        "invalid", "historical", ["invalid_work_evidence"])
    assert "tasks" not in item
    assert rows(copying.target) == before


def test_the_summary_revalidates_stored_metadata_not_only_the_record(copying):
    copying.pub._publish_one(KEY)
    copying.pub._publish_one(KEY)
    with rooms._transaction(copying.target, immediate=True) as conn:
        records.initialize(conn)
        conn.execute(f"UPDATE {records.TARGET_TABLE} SET revision=revision+1")
        summary = records.summary_locked(conn, "room")
    assert (summary["availability"], summary["source_loss_safe"], summary["incompleteness"]) == (
        "invalid", False, ["invalid_work_evidence"])
    assert "tasks" not in summary


def _quarantined(copying, *, historical=False, opaque=None):
    copying.pub._publish_one(KEY)
    copying.pub._publish_one(KEY)
    with rooms._transaction(copying.target, immediate=True) as conn:
        # An additive future column must be protected too, NULL and BLOB included.
        conn.execute(f"ALTER TABLE {records.TARGET_TABLE} ADD COLUMN opaque_metadata")
        conn.execute(f"UPDATE {records.TARGET_TABLE} SET opaque_metadata=?", (opaque,))
        if historical:
            conn.execute(f"UPDATE {records.TARGET_TABLE} SET disposition='historical'")
        conn.execute("UPDATE hosted_room_replica_events SET payload_json='not-json'")
    assert replicas.copy_state(copying.target, room_id="room")["safety_status"] == "quarantined"
    return rows(copying.target)[0]


@pytest.mark.parametrize("opaque", [None, b"\x00\xffmetadata"])
def test_invalidation_beside_a_quarantined_copy_is_one_way_and_byte_preserving(copying, opaque):
    before = _quarantined(copying, opaque=opaque)
    with closing(open_sqlite(copying.target)) as conn, conn:
        conn.execute(f"UPDATE {records.TARGET_TABLE} SET disposition='invalid'")
        assert rows_in(conn) == {**before, "disposition": "invalid"}
        for disposition in ("current", "historical"):
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(f"UPDATE {records.TARGET_TABLE} SET disposition=?", (disposition,))
    assert replicas.copy_state(copying.target, room_id="room")["safety_status"] == "quarantined"


def rows_in(conn):
    return dict(conn.execute(f"SELECT * FROM {records.TARGET_TABLE}").fetchone())


@pytest.mark.parametrize("historical", [False, True])
@pytest.mark.parametrize("column,value", [
    ("room_id", "other"), ("producer_gateway_id", "other"), ("producer_epoch", 2), ("revision", 99),
    ("digest", "rewritten"), ("record_json", "{}"), ("opaque_metadata", b"new\x00bytes"), ("rowid", 99),
])
def test_invalidation_cannot_smuggle_any_other_change(copying, historical, column, value):
    before = _quarantined(copying, historical=historical)
    with closing(open_sqlite(copying.target)) as conn:
        for disposition in ("invalid", before["disposition"]):
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(f"UPDATE {records.TARGET_TABLE} SET disposition=?,{column}=?", (disposition, value))
            assert rows_in(conn) == before
        for operation in ("INSERT", "INSERT OR REPLACE"):
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(f"{operation} INTO {records.TARGET_TABLE} ({','.join(before)}) "
                             f"VALUES ({','.join('?' for _ in before)})", tuple(before.values()))
        if historical:
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(f"UPDATE {records.TARGET_TABLE} SET disposition='invalid'")


def test_a_stale_examined_row_cannot_invalidate_its_valid_replacement(copying):
    copying.pub._publish_one(KEY)
    with closing(open_sqlite(copying.source)) as conn, conn:
        original = conn.execute(f"SELECT * FROM {records.PENDING_TABLE}").fetchone()
        conn.execute(f"UPDATE {records.PENDING_TABLE} SET digest='wrong'")
        stale = conn.execute(f"SELECT * FROM {records.PENDING_TABLE}").fetchone()
        conn.execute(f"UPDATE {records.PENDING_TABLE} SET digest=?", (original["digest"],))
    with closing(open_sqlite(copying.source)) as conn:
        with pytest.raises(records.InvalidStoredWorkRecord):
            storage.validate_stored_locked(conn, records.PENDING_TABLE, stale)
    assert rows(copying.source, records.PENDING_TABLE) == [dict(original)]

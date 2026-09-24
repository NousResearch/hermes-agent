"""Regression for #99107: retained quarantine is not disposable history.

All histories are synthetic imported SQLite data. Only schema, retention and
read-only safety helpers run; this does not certify live takeover or recovery.
"""

from contextlib import closing
import sqlite3

import pytest

from gateway import hosted_room_replicas as replicas
from gateway import hosted_rooms_legacy_import as legacy_import

from gateway import hosted_room_safety as safety
from gateway import hosted_rooms as rooms


_TABLES = {
    "authority": ("hosted_rooms", "hosted_room_events"),
    "replica": ("hosted_room_replicas", "hosted_room_replica_events"),
}


def _seed(conn, owner, room_id, *, ended=2, updated=2, incomplete=False, local=False):
    table, events = _TABLES[owner]
    event_id = f"event:{room_id}"
    kind, actor, payload = "message.user", '{"kind":"user","id":"import"}', '{"text":"mémoire"}'
    size = sum(len(value.encode("utf-8")) for value in (event_id, kind, actor, payload))
    counters = "next_seq" if owner == "authority" else "last_seq, latest_seq"
    values = "2" if owner == "authority" else f"1, {2 if incomplete else 1}"
    conn.execute(
        f"""INSERT INTO {table}
            (room_id, name, members_json, authority_gateway_id, authority_epoch,
             event_bytes, created_at, updated_at, disbanded_at, {counters})
            VALUES (?, ?, '[]', 'imported-owner', 1, ?, 1, ?, ?, {values})""",
        (room_id, room_id, size, updated, ended),
    )
    conn.execute(
        f"""INSERT INTO {events}
            (room_id, seq, event_id, kind, actor_json, authority_epoch, payload_json, created_at)
            VALUES (?, 1, ?, ?, ?, 1, ?, 1)""",
        (room_id, event_id, kind, actor, payload),
    )
    if local:
        conn.execute(
            """UPDATE hosted_room_replicas SET quarantine_reason='local_import_error',
               quarantined_at=1 WHERE room_id=?""", (room_id,),
        )
    return size


def _snapshot(conn, owner):
    table, events = _TABLES[owner]
    return {
        "rooms": {row["room_id"]: tuple(row) for row in conn.execute(f"SELECT * FROM {table}")},
        "events": {row["room_id"]: tuple(row) for row in conn.execute(f"SELECT * FROM {events}")},
        "quarantine": [tuple(row) for row in conn.execute("SELECT * FROM hosted_room_quarantine ORDER BY room_id")],
        "reservations": [tuple(row) for row in conn.execute("SELECT * FROM hosted_room_id_reservations ORDER BY room_id")],
        "bytes": conn.execute("SELECT event_bytes FROM hosted_room_event_budget WHERE singleton=1").fetchone()[0],
    }


def test_legacy_import_keeps_passive_and_retired_room_reservations(tmp_path):
    source = tmp_path / "state.db"
    with rooms._transaction(source, immediate=True) as conn:
        authority_bytes = _seed(conn, "authority", "authoritative")
        replica_bytes = _seed(conn, "replica", "passive")
        conn.execute("INSERT INTO hosted_room_id_reservations VALUES ('retired', 'replica', 1)")
    target = tmp_path / "shared-state.db"
    assert [room["room_id"] for room in rooms.list_rooms(target, include_disbanded=True)] == ["authoritative"]
    with closing(rooms._read_connection(target)) as conn:
        assert conn.execute("SELECT room_id FROM hosted_room_replicas").fetchone()[0] == "passive"
        assert {row[0]: row[1] for row in conn.execute(
            "SELECT room_id, owner_kind FROM hosted_room_id_reservations"
        )} == {"authoritative": "authority", "passive": "replica", "retired": "replica"}
        assert conn.execute(
            "SELECT event_bytes FROM hosted_room_event_budget WHERE singleton=1"
        ).fetchone()[0] == authority_bytes + replica_bytes
        assert conn.execute("SELECT rooms FROM hosted_room_legacy_imports").fetchone()[0] == 1
    with pytest.raises(rooms.RoomConflictError):
        rooms.create_room(target, room_id="retired", name="Reuse", members=[],
                          authority_gateway_id="imported-owner")


def test_legacy_import_refuses_conflicting_reservation_owner(tmp_path):
    source = tmp_path / "state.db"
    with rooms._transaction(source, immediate=True) as conn:
        _seed(conn, "authority", "authoritative")
        conn.execute("UPDATE hosted_room_id_reservations SET owner_kind='replica'")
    target = tmp_path / "shared-state.db"
    assert rooms.list_rooms(target) == []
    with closing(rooms._read_connection(target)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM hosted_rooms").fetchone()[0] == 0
        assert conn.execute("SELECT 1 FROM sqlite_master WHERE name='hosted_room_legacy_imports'").fetchone() is None
        assert rooms._schema_is_current(conn)
    # A failed import must restore both triggers, not just roll back its rows.
    with rooms._transaction(target, immediate=True) as conn:
        _seed(conn, "authority", "after-failure", ended=None)
        conn.execute(
            """INSERT INTO hosted_room_events VALUES
               ('after-failure', 2, 'loss', 'authority.lost', '{}', 1, '{}', 2)"""
        )
        assert safety._quarantine_reason_locked(conn, "after-failure") == "unsafe_authority_demotion"
        with pytest.raises(sqlite3.IntegrityError, match="quarantined"):
            conn.execute(
                """INSERT INTO hosted_room_events VALUES
                   ('after-failure', 3, 'new', 'message.user', '{}', 1, '{}', 3)"""
            )


@pytest.mark.parametrize("owner", ["authority", "replica"])
@pytest.mark.parametrize("pressure", ["age", "count", "bytes", "unreclaimable_bytes"])
def test_pruning_keeps_quarantine_and_reclaims_only_terminal_history(tmp_path, monkeypatch, owner, pressure):
    db = tmp_path / "shared-state.db"
    with rooms._transaction(db, immediate=True) as conn:
        sizes = {
            "shared": _seed(conn, owner, "shared", ended=1),
            "old": _seed(conn, owner, "old"),
            "recent": _seed(conn, owner, "recent", ended=100),
            "live": _seed(conn, owner, "live", ended=None),
        }
        if owner == "replica":
            sizes["incomplete"] = _seed(conn, owner, "incomplete", incomplete=True)
            sizes["local"] = _seed(conn, owner, "local", local=True)
        conn.execute(
            "INSERT INTO hosted_room_quarantine VALUES ('shared', 'imported_unsafe_history', 1)"
        )
        before = _snapshot(conn, owner)
    assert before["bytes"] == sum(sizes.values())
    monkeypatch.setattr(rooms, "MAX_DISBANDED_ROOM_TOMBSTONES", 1 if pressure == "count" else 512)
    removed = {"old", "recent"} if pressure == "unreclaimable_bytes" else {"old"}
    budget = 0 if pressure == "unreclaimable_bytes" else before["bytes"] - sizes["old"]
    now = rooms.DISBANDED_ROOM_RETENTION_SECONDS + 50 if pressure == "age" else None
    with rooms._transaction(db, immediate=True) as conn:
        if owner == "authority":
            result = rooms._prune_disbanded_rooms_locked(
                conn, now=now,
                max_gateway_event_bytes=budget if "bytes" in pressure else None,
            )
        else:
            result = safety._prune_disbanded_replicas_locked(
                conn, now=now,
                max_replica_event_bytes=budget if "bytes" in pressure else None,
                max_replica_rooms=len(sizes) - 1 if pressure == "count" else None,
            )
    with closing(rooms._read_connection(db)) as conn:
        after = _snapshot(conn, owner)
        assert after["rooms"] == {key: value for key, value in before["rooms"].items() if key not in removed}
        assert after["events"] == {key: value for key, value in before["events"].items() if key not in removed}
        assert result == len(removed)
        assert after["bytes"] == before["bytes"] - sum(sizes[key] for key in removed)
        assert after["quarantine"] == before["quarantine"]
        assert after["reservations"] == before["reservations"]
        assert safety._room_id_reservation_kind_locked(conn, "shared") == owner
        with pytest.raises(rooms.RoomQuarantinedError, match="imported_unsafe_history"):
            safety._raise_if_quarantined(conn, "shared")
        if owner == "replica":
            assert conn.execute("SELECT quarantine_reason FROM hosted_room_replicas WHERE room_id='shared'").fetchone()[0] is None
        else:
            assert {row[0] for row in conn.execute("SELECT room_id FROM hosted_room_retired_ids")} == removed


@pytest.mark.parametrize("shared_updated", [0, 100])
@pytest.mark.parametrize("budget_floor", [False, True])
def test_first_open_compaction_preserves_shared_only_quarantine(tmp_path, monkeypatch, shared_updated, budget_floor):
    db = tmp_path / "shared-state.db"
    with rooms._transaction(db, immediate=True) as conn:
        sizes = {
            "shared": _seed(conn, "replica", "shared", updated=shared_updated),
            "local": _seed(conn, "replica", "local", local=True),
            "ordinary": _seed(conn, "replica", "ordinary"),
            "authority": _seed(conn, "authority", "authority", ended=None),
        }
        conn.execute("INSERT INTO hosted_room_quarantine VALUES ('shared', 'imported_unsafe_history', 1)")
        before = _snapshot(conn, "replica")
        # Force the real first-open migration, without disabling safety triggers.
        conn.execute("DROP INDEX idx_hosted_room_events_cursor")
    budget = 0 if budget_floor else sum(sizes.values()) - sizes["ordinary"]
    monkeypatch.setattr(rooms, "MAX_GATEWAY_EVENT_BYTES", budget)
    with closing(rooms._connect(db)) as conn:
        assert rooms._schema_is_current(conn)
    # Read again after the migration committed and its connection closed.
    with closing(rooms._read_connection(db)) as conn:
        after = _snapshot(conn, "replica")
        assert after["rooms"] == {key: value for key, value in before["rooms"].items() if key != "ordinary"}
        assert after["events"] == {key: value for key, value in before["events"].items() if key != "ordinary"}
        assert after["reservations"] == before["reservations"]
        assert after["bytes"] == sum(sizes.values()) - sizes["ordinary"]
        assert safety._quarantine_reason_locked(conn, "shared") == "imported_unsafe_history"
        assert safety._quarantine_reason_locked(conn, "ordinary") == "replica_storage_budget_exceeded"
        assert before["quarantine"][0] in after["quarantine"]
        assert conn.execute("SELECT quarantine_reason FROM hosted_room_replicas WHERE room_id='shared'").fetchone()[0] is None
        assert conn.execute("SELECT event_bytes FROM hosted_rooms WHERE room_id='authority'").fetchone()[0] == sizes["authority"]
        assert conn.execute("SELECT COUNT(*) FROM hosted_room_events WHERE room_id='authority'").fetchone()[0] == 1
    # The retained evidence cannot be consumed by the next schema refresh either.
    with rooms._transaction(db, immediate=True) as conn:
        safety.initialize_safety_schema(conn)
        assert _snapshot(conn, "replica") == after


@pytest.mark.parametrize("reclaim", [False, True])
def test_mixed_pruning_uses_only_the_authority_allowance(tmp_path, reclaim):
    db = tmp_path / "mixed.db"
    with rooms._transaction(db, immediate=True) as conn:
        sizes = {name: _seed(conn, "authority", name) for name in ("old", "recent", "shared")}
        _seed(conn, "replica", "live-replica", ended=None)
        conn.execute("INSERT INTO hosted_room_quarantine VALUES ('shared', 'imported_unsafe_history', 1)")
        before = {owner: _snapshot(conn, owner) for owner in _TABLES}
    removed = {"old"} if reclaim else set()
    allowance = sum(sizes.values()) - (sizes["old"] if reclaim else 0)
    with rooms._transaction(db, immediate=True) as conn:
        count = rooms._prune_disbanded_rooms_locked(
            conn, now=None, max_gateway_event_bytes=allowance,
        )
    with closing(rooms._read_connection(db)) as conn:
        for owner in _TABLES:
            after = _snapshot(conn, owner)
            expected = dict(before[owner], bytes=before[owner]["bytes"] - sum(sizes[key] for key in removed))
            for field in ("rooms", "events"):
                expected[field] = {key: value for key, value in before[owner][field].items() if key not in removed}
            assert after == expected
        assert count == len(removed)
        assert {row[0] for row in conn.execute("SELECT room_id FROM hosted_room_retired_ids")} == removed


def test_mixed_append_reclaims_authority_before_refusing_capacity(tmp_path, monkeypatch):
    db = tmp_path / "mixed.db"
    with rooms._transaction(db, immediate=True) as conn:
        old_bytes = _seed(conn, "authority", "old")
        _seed(conn, "authority", "recent", ended=100)
        _seed(conn, "authority", "active", ended=None)
        _seed(conn, "replica", "live-replica", ended=None)
        before = {owner: _snapshot(conn, owner) for owner in _TABLES}
    actor, payload = {"kind": "user", "id": "import"}, {"text": "mémoire" * 20}
    event_id, kind = "new-event", "message.user"
    added = rooms.utf8_len(event_id, kind, rooms._validate_actor(actor, kind=kind)[1], rooms._payload_json(payload))
    limit = before["authority"]["bytes"] + added - old_bytes
    monkeypatch.setattr(rooms, "MAX_GATEWAY_EVENT_BYTES", limit)
    result = rooms.append_event(
        db, room_id="active", event_id=event_id, kind=kind, actor=actor, payload=payload,
        authority_gateway_id="imported-owner", authority_epoch=1, now=200,
    )
    with closing(rooms._read_connection(db)) as conn:
        after = _snapshot(conn, "authority")
        assert set(after["rooms"]) == {"active", "recent"}
        assert after["rooms"]["recent"] == before["authority"]["rooms"]["recent"]
        assert after["events"]["recent"] == before["authority"]["events"]["recent"]
        assert after["quarantine"] == before["authority"]["quarantine"]
        assert after["reservations"] == before["authority"]["reservations"]
        assert _snapshot(conn, "replica") == dict(before["replica"], bytes=limit)
        assert after["bytes"] == rooms._gateway_event_bytes(conn) == limit
        assert conn.execute("SELECT COUNT(*) FROM hosted_room_events WHERE room_id='active'").fetchone()[0] == 2
        assert {row[0] for row in conn.execute("SELECT room_id FROM hosted_room_retired_ids")} == {"old"}
    assert result["seq"] == 2
    assert result["payload"] == payload


def test_mixed_append_rolls_back_when_protected_history_cannot_fit(tmp_path, monkeypatch):
    db = tmp_path / "mixed.db"
    with rooms._transaction(db, immediate=True) as conn:
        old_bytes = _seed(conn, "authority", "old")
        _seed(conn, "authority", "shared")
        _seed(conn, "authority", "active", ended=None)
        _seed(conn, "replica", "live-replica", ended=None)
        _seed(conn, "replica", "local-replica", local=True)
        _seed(conn, "replica", "shared-replica")
        for room_id in ("shared", "shared-replica"):
            conn.execute("INSERT INTO hosted_room_quarantine VALUES (?, 'imported_unsafe_history', 1)", (room_id,))
        before = {owner: _snapshot(conn, owner) for owner in _TABLES}
    actor, payload = {"kind": "user", "id": "import"}, {"text": "mémoire" * 20}
    added = rooms.utf8_len("new-event", "message.user", rooms._validate_actor(actor, kind="message.user")[1], rooms._payload_json(payload))
    limit = before["authority"]["bytes"] + added - old_bytes - 1
    monkeypatch.setattr(rooms, "MAX_GATEWAY_EVENT_BYTES", limit)
    assert rooms.CONTROL_EVENT_BYTE_RESERVE > added
    with pytest.raises(rooms.HostedRoomError, match="storage is full"):
        rooms.append_event(
            db, room_id="active", event_id="new-event", kind="message.user", actor=actor, payload=payload,
            authority_gateway_id="imported-owner", authority_epoch=1, now=200,
        )
    # The attempted ordinary reclamation and its tombstone must roll back too.
    with closing(rooms._read_connection(db)) as conn:
        assert {owner: _snapshot(conn, owner) for owner in _TABLES} == before
        assert conn.execute("SELECT COUNT(*) FROM hosted_room_retired_ids").fetchone()[0] == 0
        assert rooms._gateway_event_bytes(conn) == before["authority"]["bytes"]


def _history(conn, table):
    return [tuple(row) for row in conn.execute(f"SELECT * FROM {table} ORDER BY room_id, seq")]


@pytest.mark.parametrize("lineage", ["demotion", "promotion"])
def test_first_open_import_preserves_quarantined_history_and_live_fence(tmp_path, lineage):
    source, target = tmp_path / "state.db", tmp_path / "shared-state.db"
    with rooms._transaction(source, immediate=True) as conn:
        _seed(conn, "authority", "healthy", ended=None)
        _seed(conn, "authority", "unsafe", ended=None)
        # Model the pre-fence source schema, not permission to append live data.
        conn.execute("DROP TRIGGER trg_hosted_events_reject_quarantined_insert")
        conn.execute("DROP TRIGGER trg_hosted_events_quarantine_unsafe_lineage")
        kind = "authority.lost" if lineage == "demotion" else "authority.claimed"
        payload = '{}' if lineage == "demotion" else '{"promoted_from_replica":true}'
        conn.execute(
            "INSERT INTO hosted_room_events VALUES ('unsafe', 2, 'transition', ?, ?, 1, ?, 2)",
            (kind, '{"kind":"system","id":"legacy"}', payload),
        )
        if lineage == "promotion":
            conn.execute(
                """INSERT INTO hosted_room_events VALUES
                   ('unsafe', 3, 'late', 'message.user', '{"kind":"user","id":"legacy"}', 1, '{}', 3)"""
            )
        conn.execute("UPDATE hosted_rooms SET next_seq=? WHERE room_id='unsafe'", (3 if lineage == "demotion" else 4,))
        conn.execute(rooms._EVENT_BYTES_BACKFILL.format(where="1"))
        if lineage == "demotion":
            conn.execute("INSERT INTO hosted_room_quarantine VALUES ('unsafe', 'original_demotion_evidence', 0)")
        before = _history(conn, "hosted_room_events")
        budget = rooms._gateway_event_bytes(conn)
        reservations = [tuple(row) for row in conn.execute(
            "SELECT * FROM hosted_room_id_reservations ORDER BY room_id"
        )]
    with sqlite3.connect(source) as conn:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    source_bytes = source.read_bytes()
    assert {room["room_id"] for room in rooms.list_rooms(target)} == {"healthy", "unsafe"}
    reason = "original_demotion_evidence" if lineage == "demotion" else "unsafe_replica_promotion"
    with closing(rooms._read_connection(target)) as conn:
        assert _history(conn, "hosted_room_events") == before
        assert tuple(conn.execute("SELECT * FROM hosted_room_quarantine").fetchone()) == (
            "unsafe", reason, 0 if lineage == "demotion" else 2,
        )
        assert [tuple(row) for row in conn.execute(
            "SELECT * FROM hosted_room_id_reservations ORDER BY room_id"
        )] == reservations
        assert conn.execute("SELECT rooms FROM hosted_room_legacy_imports").fetchone()[0] == 2
        assert conn.execute("SELECT event_bytes FROM hosted_room_event_budget").fetchone()[0] == budget
    assert source not in legacy_import._failed_sources
    with pytest.raises(rooms.RoomQuarantinedError):
        rooms.append_event(
            target, room_id="unsafe", event_id="new", kind="message.user",
            actor={"kind": "user", "id": "legacy"}, payload={},
            authority_gateway_id="imported-owner", authority_epoch=1,
        )
    with rooms._transaction(target, immediate=True) as conn:
        with pytest.raises(sqlite3.IntegrityError, match="quarantined"):
            conn.execute(
                """INSERT INTO hosted_room_events VALUES
                   ('unsafe', 4, 'raw-live', 'message.user', '{}', 1, '{}', 4)"""
            )
        # The automatic classifier is restored as well as its live-write fence.
        conn.execute(
            """INSERT INTO hosted_room_events VALUES
               ('healthy', 2, 'new-loss', 'authority.lost', '{}', 1, '{}', 4)"""
        )
        assert safety._quarantine_reason_locked(conn, "healthy") == "unsafe_authority_demotion"
    assert source.read_bytes() == source_bytes


def _terminal_replica(conn, room_id, *, defect=None):
    _seed(conn, "replica", room_id, ended=2)
    conn.execute(
        """INSERT INTO hosted_room_replica_events VALUES
           (?, 2, ?, 'room.disbanded', '{"kind":"system","id":"legacy"}', 1, '{}', 2)""",
        (room_id, f"end:{room_id}"),
    )
    last = 2
    if defect:
        last = 3
        conn.execute(
            """INSERT INTO hosted_room_replica_events VALUES
               (?, 3, ?, 'message.user', '{"kind":"user","id":"legacy"}', 1, '{}', 3)""",
            (room_id, f"event:{room_id}" if defect == "duplicate_event_id" else f"late:{room_id}"),
        )
    size = sum(sum(len(str(row[key]).encode()) for key in (
        "event_id", "kind", "actor_json", "payload_json"
    )) for row in conn.execute("SELECT * FROM hosted_room_replica_events WHERE room_id=?", (room_id,)))
    conn.execute(
        "UPDATE hosted_room_replicas SET last_seq=?, latest_seq=?, event_bytes=? WHERE room_id=?",
        (last, last, size, room_id),
    )
    return size


@pytest.mark.parametrize("defect", ["events_after_disband", "duplicate_event_id"])
def test_first_open_classifies_replica_lineage_before_compacting(tmp_path, monkeypatch, defect):
    db = tmp_path / "replicas.db"
    with rooms._transaction(db, immediate=True) as conn:
        protected = _terminal_replica(conn, "unsafe", defect=defect)
        _terminal_replica(conn, "valid")
        before = _history(conn, "hosted_room_replica_events")
        reservations = [tuple(row) for row in conn.execute("SELECT * FROM hosted_room_id_reservations ORDER BY room_id")]
        conn.execute("DROP INDEX idx_hosted_room_events_cursor")
    monkeypatch.setattr(rooms, "MAX_GATEWAY_EVENT_BYTES", 0)
    with closing(rooms._connect(db)) as conn:
        assert rooms._schema_is_current(conn)
    with closing(rooms._read_connection(db)) as conn:
        assert _history(conn, "hosted_room_replica_events") == [row for row in before if row[0] == "unsafe"]
        row = conn.execute("SELECT quarantine_reason, event_bytes FROM hosted_room_replicas WHERE room_id='unsafe'").fetchone()
        assert tuple(row) == (defect, protected)
        assert safety._quarantine_reason_locked(conn, "unsafe") is None
        assert conn.execute("SELECT event_bytes FROM hosted_room_event_budget").fetchone()[0] == protected
        assert [tuple(row) for row in conn.execute("SELECT * FROM hosted_room_id_reservations ORDER BY room_id")] == reservations


@pytest.mark.parametrize("entry", ["age", "count", "bytes", "append", "refused_append", "refused_ingest"])
def test_late_replica_history_is_audited_at_each_reclamation_boundary(tmp_path, monkeypatch, entry):
    db = tmp_path / "late.db"
    with rooms._transaction(db, immediate=True) as conn:
        _seed(conn, "authority", "active", ended=None)
        protected = _terminal_replica(conn, "unsafe", defect="events_after_disband")
        reclaimable = _terminal_replica(conn, "valid")
        before = _history(conn, "hosted_room_replica_events")
        budget = conn.execute("SELECT event_bytes FROM hosted_room_event_budget").fetchone()[0]
    # No replica observation between the old writer's commit and reclamation.
    if entry == "refused_ingest":
        with pytest.raises(replicas.ReplicaError, match="quarantined"):
            replicas.ingest_page(
                db, room_id="unsafe", room_name="unsafe", members=[], now=4,
                page={"events": [], "authority": {"gateway_id": "imported-owner", "epoch": 1},
                      "cursor": 3, "latest_seq": 3, "has_more": False},
            )
        with closing(rooms._read_connection(db)) as conn:
            # Its transaction rolled back the first audit. Retention must re-audit.
            assert conn.execute("SELECT quarantine_reason FROM hosted_room_replicas WHERE room_id='unsafe'").fetchone()[0] is None
            assert _history(conn, "hosted_room_replica_events") == before
    added = 0
    if entry in {"append", "refused_append"}:
        actor, payload = {"kind": "user", "id": "import"}, {"text": "new"}
        added = rooms.utf8_len("new", "message.user", rooms._validate_actor(actor, kind="message.user")[1], rooms._payload_json(payload))
        limit = budget + added - reclaimable - (1 if entry == "refused_append" else 0)
        monkeypatch.setattr(rooms, "MAX_GATEWAY_EVENT_BYTES", limit)
        def append():
            return rooms.append_event(
                db, room_id="active", event_id="new", kind="message.user", actor=actor,
                payload=payload, authority_gateway_id="imported-owner", authority_epoch=1, now=4,
            )
        if entry == "refused_append":
            with pytest.raises(rooms.HostedRoomError, match="storage is full"):
                append()
            with closing(rooms._read_connection(db)) as conn:
                assert _history(conn, "hosted_room_replica_events") == before
                assert conn.execute("SELECT event_bytes FROM hosted_room_event_budget").fetchone()[0] == budget
            added = 0
        else:
            assert append()["seq"] == 2
    if entry != "append":
        with rooms._transaction(db, immediate=True) as conn:
            assert safety._prune_disbanded_replicas_locked(
                conn, now=rooms.DISBANDED_REPLICA_RETENTION_SECONDS + 3 if entry == "age" else None,
                max_replica_event_bytes=None if entry in {"age", "count"} else 0,
                max_replica_rooms=0 if entry == "count" else None,
            ) == 1
    with closing(rooms._read_connection(db)) as conn:
        assert _history(conn, "hosted_room_replica_events") == [row for row in before if row[0] == "unsafe"]
        row = conn.execute("SELECT quarantine_reason, quarantined_at, event_bytes FROM hosted_room_replicas WHERE room_id='unsafe'").fetchone()
        assert row["quarantine_reason"] == "events_after_disband"
        assert row["quarantined_at"] is not None
        assert row["event_bytes"] == protected
        assert conn.execute("SELECT event_bytes FROM hosted_room_event_budget").fetchone()[0] == budget - reclaimable + added
        assert {row[0] for row in conn.execute("SELECT room_id FROM hosted_room_id_reservations")} == {"active", "unsafe", "valid"}

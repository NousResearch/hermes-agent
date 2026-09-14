"""Regression for #99107: retained quarantine is not disposable history.

All histories are synthetic imported SQLite data. Only schema, retention and
read-only safety helpers run; this does not certify live takeover or recovery.
"""

from contextlib import closing

import pytest

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

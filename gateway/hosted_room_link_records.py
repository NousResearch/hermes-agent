"""Persisted RoomLink route records."""

from typing import Any, Mapping

from gateway.hosted_rooms import HostedRoomError, _transaction
from gateway.hosted_rooms_common import DbPath, clock as _now

_LINK_COLUMNS = (
    "room_id", "member_id", "target_url", "target_profile", "grant", "catalog_json", "cancellation_scope_id",
    "trace_id", "transport_security", "status", "updated_at")



def list_room_link_records(db_path: DbPath) -> list[dict[str, Any]]:
    """Return private RoomLink records without logging or formatting grants."""
    with _transaction(db_path) as conn:
        rows = conn.execute("""SELECT room_id, member_id, target_url, target_profile, grant,
                      catalog_json, cancellation_scope_id, trace_id,
                      transport_security, status, updated_at
                 FROM hosted_room_links
             ORDER BY room_id, member_id""").fetchall()
    return [dict(row) for row in rows]


def upsert_room_link_record(db_path: DbPath, *, record: Mapping[str, Any], max_links: int) -> None:
    """Atomically insert or replace one private RoomLink record."""
    with _transaction(db_path, immediate=True) as conn:
        existing = conn.execute(
            "SELECT 1 FROM hosted_room_links WHERE room_id=? AND member_id=?", (record["room_id"], record["member_id"])
        ).fetchone()
        if existing is None and int(conn.execute("SELECT COUNT(*) FROM hosted_room_links").fetchone()[0]) >= max_links:
            raise HostedRoomError("too many stored room links")
        conn.execute("""INSERT INTO hosted_room_links(
                   room_id, member_id, target_url, target_profile, grant,
                   catalog_json, cancellation_scope_id, trace_id,
                   transport_security, status, updated_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(room_id, member_id) DO UPDATE SET
                   target_url=excluded.target_url,
                   target_profile=excluded.target_profile,
                   grant=excluded.grant,
                   catalog_json=excluded.catalog_json,
                   cancellation_scope_id=excluded.cancellation_scope_id,
                   trace_id=excluded.trace_id,
                   transport_security=excluded.transport_security,
                   status=excluded.status,
                   updated_at=excluded.updated_at""",
            tuple(record[column] for column in _LINK_COLUMNS))


def update_room_link_status(
    db_path: DbPath, *, room_id: str, member_id: str, status: str, now: float | None = None) -> bool:
    """Persist a non-secret route health classification."""
    with _transaction(db_path, immediate=True) as conn:
        return conn.execute(
            "UPDATE hosted_room_links SET status=?, updated_at=? WHERE room_id=? AND member_id=?",
            (status, _now(now), room_id, member_id)).rowcount == 1


def delete_room_link_records(db_path: DbPath, *, room_id: str) -> int:
    """Delete persisted peer routes after their target grants are revoked."""
    with _transaction(db_path, immediate=True) as conn:
        return conn.execute("DELETE FROM hosted_room_links WHERE room_id=?", (room_id,)).rowcount

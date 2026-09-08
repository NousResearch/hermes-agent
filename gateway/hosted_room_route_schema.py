"""Durable no-new-work guard for hosted-room Disband."""

import sqlite3

from gateway.hosted_rooms_common import table_exists


def require_room_work_open(conn: sqlite3.Connection, room_id: str, *, error: type[Exception]) -> None:
    """Check inside the caller's write transaction, after any idempotent replay.

    Driver-only stores may predate the route schema. Its installation and the
    irreversible fence use the same SQLite writer serialization as admission.
    Reads, lease maintenance and accepted-work cleanup must not use this guard.
    """
    if table_exists(conn, "hosted_room_disband_fences") and conn.execute(
        "SELECT 1 FROM hosted_room_disband_fences WHERE room_id=?", (room_id,),
    ).fetchone() is not None:
        raise error("hosted room is being disbanded")

"""Serialize fresh RoomLink run admission with local room-control fences."""

from contextlib import contextmanager, nullcontext
from pathlib import Path
import sqlite3

from gateway import hosted_rooms as rooms
from gateway.hosted_room_peer import decode_room_grant
from gateway.hosted_room_route_schema import require_room_work_open


class RoomAdmissionFenced(ValueError):
    """No new run was reserved; existing runs remain observable and stoppable."""


def _validate(conn, secret, token):
    try:
        claims = decode_room_grant(secret, token, permission="dispatch")
        if rooms.room_grant_is_revoked(None, claims=claims, _conn=conn) or not rooms.peer_room_grant_is_current(None, claims=claims, _conn=conn):
            raise RoomAdmissionFenced("Reconnect this Bot before sending more work.")
        if conn.execute("SELECT 1 FROM hosted_room_quarantine WHERE room_id=?", (claims["room_id"],)).fetchone():
            raise RoomAdmissionFenced("This Group Chat is paused. No new work was started.")
        require_room_work_open(conn, claims["room_id"], error=RoomAdmissionFenced)
        room = conn.execute("SELECT authority_gateway_id,authority_epoch,disbanded_at FROM hosted_rooms WHERE room_id=?",
                            (claims["room_id"],)).fetchone()
        if room is not None and (room["disbanded_at"] is not None
                or (room["authority_gateway_id"], room["authority_epoch"]) != (claims["authority_gateway_id"], claims["authority_epoch"])):
            raise RoomAdmissionFenced("This Group Chat has moved or closed. No new work was started.")
    except ValueError as exc:
        if isinstance(exc, RoomAdmissionFenced):
            raise
        raise RoomAdmissionFenced("Reconnect this Bot before sending more work.") from exc


def _guard_rows(conn, secret, token):
    factory = conn.row_factory
    conn.row_factory = sqlite3.Row
    try:
        _validate(conn, secret, token)
    finally:
        conn.row_factory = factory


@contextmanager
def room_admission_guard(adapter, request):
    """No await may occur in this scope; an existing idempotency replay skips validation."""
    token = adapter._room_grant_token(request)
    if not token:
        yield None
        return
    root = Path(rooms.default_db_path()).resolve()
    stored = adapter._run_idempotency_store._db_path
    same = stored is not None and Path(stored).resolve() == root
    if not same and stored is not None and Path(stored).exists() and root.exists():
        same = Path(stored).samefile(root)
    secret = adapter._room_grant_secret()
    # Embedders can colocate the stores. In that case reserve's own writer lock
    # is already the room lock; opening a second writer would deadlock.
    with (nullcontext(None) if same else rooms._transaction(root, immediate=True)) as room_conn:
        yield lambda run_conn: _guard_rows(room_conn if room_conn is not None else run_conn, secret, token)

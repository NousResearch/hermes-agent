"""RoomLink-authorized ingress to the existing passive replica store."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any

from gateway import hosted_rooms as rooms
from gateway import hosted_room_replicas as replicas
from gateway.hosted_room_peer import HostedRoomGrantError, decode_room_grant


def ingest_granted_page(
    db_path: Path | str, *, token: str, secret: bytes, target_install_id: str,
    target_profile: str, room_id: str, room_name: str, members: list[dict[str, Any]],
    page: dict[str, Any],
) -> dict[str, Any]:
    """Authenticate both the page scope and live permission at the write boundary.

    Replication never confers execution authority. The receiver must have
    explicitly issued this permission to the caller's room-member scope.
    """
    authority = page.get("authority") if isinstance(page, dict) else None
    authorize = authorize_granted_room(
        token=token, secret=secret, target_install_id=target_install_id, target_profile=target_profile,
        room_id=room_id, members=members, authority=authority, permission="replicate")
    return replicas.ingest_page(
        db_path, room_id=room_id, room_name=room_name, members=members, page=page, _authorize=authorize)


def authorize_granted_room(*, token, secret, target_install_id, target_profile, room_id, members, authority, permission):
    """Share exact room authorization; callers recheck under their target write transaction."""
    claims = decode_room_grant(secret, token, permission=permission)
    expected = {"gateway_id": claims["authority_gateway_id"], "epoch": claims["authority_epoch"]}
    if (
        claims["room_id"] != room_id or authority != expected
        or claims["target_install_id"] != target_install_id
        or claims["target_profile"] != target_profile
        or claims["home_install_id"] != claims["authority_gateway_id"]
    ):
        raise HostedRoomGrantError("replica scope does not match its grant")
    matching = [
        member for member in members if isinstance(member, dict)
        and member.get("member_id") == claims["member_id"]
    ] if isinstance(members, list) else []
    target = matching[0].get("target") if len(matching) == 1 else None
    if (
        not isinstance(target, dict) or target.get("kind") != "peer"
        or target.get("installation_id") != target_install_id or target.get("profile") != target_profile
    ):
        raise HostedRoomGrantError("replica does not name the authorized participant")

    def authorize_locked(conn: sqlite3.Connection) -> None:
        # Recheck after waiting for SQLite: expiry/revocation may race body validation.
        now = time.time()
        decode_room_grant(secret, token, permission=permission, now=now)
        reserved = rooms.peer_room_grant_is_current(None, claims=claims, now=now, _conn=conn)
        revoked = rooms.room_grant_is_revoked(None, claims=claims, now=now, _conn=conn)
        if not reserved or revoked:
            raise HostedRoomGrantError("replica grant is revoked or no longer current")

    return authorize_locked

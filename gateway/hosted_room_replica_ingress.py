"""Room-grant authorized ingress into this participant's passive replica store."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any, Callable

from gateway import hosted_rooms as rooms
from gateway import hosted_room_replicas as replicas
from gateway.hosted_room_peer import (
    HostedRoomGrantError, decode_room_grant, issue_room_grant, room_grant_token_digest)

# A copy-only grant is renewed in the acknowledgement once less than this is left of it (or less than
# a quarter of its life), so its host keeps copying long past the first grant's horizon.
COPY_GRANT_RENEWAL_SECONDS = 7 * 24 * 60 * 60


def ingest_granted_page(
    db_path: Path | str, *, token: str, secret: bytes, target_install_id: str,
    target_profile: str, room_id: str, room_name: str, members: list[dict[str, Any]],
    page: dict[str, Any], custody: Any = None, fenced: Callable[[str, int], None] | None = None,
    _verify_transition: Callable[[sqlite3.Connection, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Store one history page sent with a ``replicate`` grant this gateway issued.

    Copying never confers execution authority. The grant must name this gateway, this
    profile and the room's member, and it is checked again inside the replica writer, as is
    ``fenced``: a page stamped with an epoch this gateway fenced for succession never lands.
    ``custody`` is the authority's protection report sent beside the page;
    ``_verify_transition`` lets the copy follow a verified change of host (``ingest_page``).

    The acknowledgment tells the host this installation's consent to continue the group, whether
    it is always on, and, when the report carried a lease request from the authority this copy
    follows, the lease layer's answer (``lease_grant``). Every push stored from that authority
    reaches the lease layer, lease request or not: hearing the host at all is contact. Once the
    copy-only grant such a push used nears its horizon, it also returns a fresh one
    (``renewed_grant``, ``renewed_copy_grant``).
    """
    from gateway import hosted_room_custody as custody_records
    authority = page.get("authority") if isinstance(page, dict) else None
    authorize = authorize_granted_room(
        token=token, secret=secret, target_install_id=target_install_id, target_profile=target_profile,
        room_id=room_id, members=members, authority=authority, permission="replicate")

    def authorize_unfenced(conn: sqlite3.Connection) -> None:
        authorize(conn)
        if fenced is not None:
            fenced(room_id, int(authority["epoch"]))

    result = replicas.ingest_page(
        db_path, room_id=room_id, room_name=room_name, members=members, page=page, _authorize=authorize_unfenced,
        custody_report=custody, _verify_transition=_verify_transition)
    reply = {"allowed": custody_records.local_consent(db_path, room_id), "always_on": custody_records.local_always_on()}
    request = custody.get("lease_request") if isinstance(custody, dict) else None
    if result["copy_authority"] == authority:
        grant = custody_records.lease_grant(room_id, authority["epoch"], authority["gateway_id"], request)
        if grant is not None:
            reply["lease_grant"] = grant
        renewed = renewed_copy_grant(db_path, token=token, secret=secret)
        if renewed is not None:
            reply["renewed_grant"] = renewed
    return {**result, "custody": reply}


def renewed_copy_grant(db_path: Path | str, *, token: str, secret: bytes, now: float | None = None) -> str | None:
    """The fresh copy-only grant to hand back to its host, once the one it pushed with nears its horizon.

    None for any other grant, or with more than ``COPY_GRANT_RENEWAL_SECONDS`` (or a quarter of its
    life) left. The renewal keeps the grant's whole scope (room, host, epoch, this installation and
    profile, policy and permissions) and the length of its life; only that life moves, starting where
    its renewal window starts, so every acknowledgment returns the same grant until the host uses
    it. It is reserved here like the invitation that started it. A revoked grant pushes nothing, so
    its operator's revocation still ends it.
    """
    from gateway.hosted_room_custody import CUSTODY_MEMBER_ID
    claims = decode_room_grant(secret, token, permission="replicate", now=now)
    if claims["member_id"] != CUSTODY_MEMBER_ID:
        return None
    issued, horizon = float(claims["issued_at"]), float(claims.get("status_expires_at", claims["expires_at"]))
    life = horizon - issued
    start = horizon - min(COPY_GRANT_RENEWAL_SECONDS, life / 4)
    if (time.time() if now is None else float(now)) < start:
        return None
    renewed = issue_room_grant(
        secret, grant_id="grant-" + room_grant_token_digest(token)[:32], room_id=claims["room_id"],
        home_install_id=claims["home_install_id"], authority_gateway_id=claims["authority_gateway_id"],
        authority_epoch=claims["authority_epoch"], member_id=CUSTODY_MEMBER_ID,
        target_install_id=claims["target_install_id"], target_profile=claims["target_profile"],
        execution_policy_digest=claims["execution_policy_digest"], permissions=claims["permissions"],
        issued_at=start, ttl_seconds=float(claims["expires_at"]) - issued, status_ttl_seconds=life)
    rooms.reserve_peer_room(db_path, claims=decode_room_grant(secret, renewed, permission="status", now=now),
                            expires_at=start + life, now=now)
    return renewed


def authorize_granted_room(
    *, token: str, secret: bytes, target_install_id: str, target_profile: str, room_id: str,
    members: Any, authority: Any, permission: str,
) -> Callable[[sqlite3.Connection], None]:
    """Check the grant's scope now; return the re-check that runs under the writer lock."""
    claims = decode_room_grant(secret, token, permission=permission)
    expected = {"gateway_id": claims["authority_gateway_id"], "epoch": claims["authority_epoch"]}
    if (
        claims["room_id"] != room_id or authority != expected
        or claims["target_install_id"] != target_install_id
        or claims["target_profile"] != target_profile
        or claims["home_install_id"] != claims["authority_gateway_id"]
    ):
        raise HostedRoomGrantError("replica scope does not match its grant")
    from gateway.hosted_room_custody import CUSTODY_MEMBER_ID
    if claims["member_id"] == CUSTODY_MEMBER_ID:
        # A custodian-only installation has no Bot in the room: its copy-only grant is the consent.
        if "dispatch" in claims["permissions"]:
            raise HostedRoomGrantError("a custodian-only grant never runs work")
        return _recheck(token, secret, claims, permission)
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

    return _recheck(token, secret, claims, permission)


def _recheck(token: str, secret: bytes, claims: dict[str, Any], permission: str) -> Callable[[sqlite3.Connection], None]:
    def authorize_locked(conn: sqlite3.Connection) -> None:
        # Expiry or revocation may land while the request waits for the SQLite writer.
        now = time.time()
        decode_room_grant(secret, token, permission=permission, now=now)
        if (not rooms.peer_room_grant_is_current(None, claims=claims, now=now, conn=conn)
                or rooms.room_grant_is_revoked(None, claims=claims, now=now, token_sha256=room_grant_token_digest(token), conn=conn)):
            raise HostedRoomGrantError("replica grant is revoked or no longer current")

    return authorize_locked

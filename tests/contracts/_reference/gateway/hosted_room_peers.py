"""In-memory reference ``HostedRoomPeerRepository``: links, grant revocations, owed revokes, peer reservations and remote-run receipts over the placement's hosted-room tables (``hosted_room_tables``).

Reference limits: the probe reads memory and cannot fail, so ``RoomProbeUnavailableError`` is never raised; the work-record capture inside ``upsert_remote_run_receipt`` runs only its escaping checks (``hosted_room_work``).
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from typing import Any

from persistence import JsonValue
from persistence.gateway import hosted_rooms as _bounds  # the owed-revoke bounds, read at each call
from persistence.gateway.hosted_rooms import (
    AuthorityConflictError,
    HostedPlacement,
    HostedRoomError,
    OwedRevokesFullError,
    RemoteRunIdentity,
    RemoteRunReceipt,
    RoomGrantClaims,
    RoomLinkRecord,
    grant_digest,
    room_grant_hard_expiry,
)

from ..backend import register_repository
from . import hosted_room_rules as rules
from .hosted_room_tables import HostedTables, peek_tables, read_tables, room_row, write_tables
from .hosted_room_work import capture_transition_locked

__all__ = ["InMemoryHostedRoomPeerRepository"]

# The identity and link columns, the columns of a due owed revoke, and the remote-run table's column order.
_IDENTITY_COLUMNS = (
    "room_id", "home_install_id", "authority_gateway_id", "authority_epoch", "member_id", "target_install_id",
    "target_profile", "task_id", "execution_generation")
_LINK_COLUMNS = (
    "room_id", "member_id", "target_url", "target_profile", "grant", "catalog_json", "cancellation_scope_id",
    "trace_id", "transport_security", "status", "updated_at")
_DUE_COLUMNS = ("room_id", "member_id", "grant_digest", "target_url", "grant", "attempts")
_RECEIPT_COLUMNS = (
    "room_id", "home_install_id", "authority_gateway_id", "authority_epoch", "member_id", "task_id",
    "execution_generation", "target_install_id", "target_profile", "run_id", "session_id")


def _identity(record: RemoteRunIdentity | RemoteRunReceipt) -> tuple[Any, ...]:
    return tuple(getattr(record, column) for column in _IDENTITY_COLUMNS)


def _superseded(row: Mapping[str, Any], gateway_id: str, epoch: int) -> bool:
    """A newer epoch, or the same epoch under another gateway, outranks the claim."""
    row_epoch = int(row["authority_epoch"])
    return row_epoch > epoch or (row_epoch == epoch and str(row["authority_gateway_id"]) != gateway_id)


def _live(row: Mapping[str, Any], now: float) -> bool:
    return float(row["expires_at"]) > now and row["revoked_at"] is None


def _drop_expired_reservations(tables: HostedTables, now: float) -> None:
    for rows in tables["hosted_room_peer_reservations"].values():
        for key in [key for key, row in rows.items() if float(row["expires_at"]) <= now]:
            del rows[key]


def _delete_owed(tables: HostedTables, keep: Any) -> None:
    """Delete every owed-revoke row ``keep(row)`` rejects."""
    owed = tables["hosted_room_owed_revokes"]
    for key in [key for key, row in owed.items() if not keep(row)]:
        del owed[key]


def _owe_revoke(tables: HostedTables, *, room_id: str, member_id: str, target_url: str, grant: str, expires_at: float, now: float, held: bool = False) -> None:
    """One owed revoke: an existing key stays, then the member bound, then the table ceiling, where expired rows go first; a held row past the member bound refuses as a held-grant refusal."""
    owed = tables["hosted_room_owed_revokes"]
    key = (room_id, member_id, grant_digest(grant))
    if key in owed:
        return
    if sum(1 for row in owed.values() if (row["room_id"], row["member_id"]) == (room_id, member_id)) >= _bounds.MAX_OWED_REVOKES_PER_MEMBER:
        if held:
            raise HostedRoomError("too many held group chat grant revokes")
        raise OwedRevokesFullError("too many owed revokes for this group chat member")
    if len(owed) >= _bounds.MAX_OWED_REVOKES:
        _delete_owed(tables, lambda row: float(row["expires_at"]) > now)
        if len(owed) >= _bounds.MAX_OWED_REVOKES:
            raise OwedRevokesFullError("too many owed room grant revokes")
    owed[key] = {
        "room_id": room_id, "member_id": member_id, "grant_digest": key[2], "target_url": target_url, "grant": grant,
        "expires_at": min(float(expires_at), now + _bounds.MAX_OWED_REVOKE_SECONDS), "attempts": 0, "next_attempt_at": now,
        "created_at": now, "held": int(held)}


class InMemoryHostedRoomPeerRepository:
    """See ``persistence.gateway.hosted_rooms.HostedRoomPeerRepository`` for the contract each method implements."""

    def __init__(self, store: Any) -> None:
        self._store = store

    def upsert_room_link_record(self, *, placement: HostedPlacement, record: RoomLinkRecord, max_links: int, hold_replaced: bool = False) -> None:
        with write_tables(self._store, placement) as tables:
            links = tables["hosted_room_links"]
            existing = links.get(record.room_id, {}).get(record.member_id)
            if existing is None and sum(len(rows) for rows in links.values()) >= max_links:
                raise HostedRoomError("too many stored room links")
            if (existing is not None and hold_replaced and existing["grant"] != record.grant
                    and room_grant_hard_expiry(existing["grant"]) > room_grant_hard_expiry(record.grant)):
                _owe_revoke(tables, room_id=record.room_id, member_id=record.member_id, target_url=existing["target_url"],
                            grant=existing["grant"], expires_at=room_grant_hard_expiry(existing["grant"]), now=rules.clock(None), held=True)
            tables.rows("hosted_room_links", record.room_id)[record.member_id] = {column: getattr(record, column) for column in _LINK_COLUMNS}
            key = (record.room_id, record.member_id, grant_digest(record.grant))
            _delete_owed(tables, lambda row: (row["room_id"], row["member_id"], row["grant_digest"]) != key or not row["held"])

    def update_room_link_status(self, *, placement: HostedPlacement, room_id: str, member_id: str, status: str, now: float | None = None) -> bool:
        timestamp = rules.clock(now)
        with write_tables(self._store, placement) as tables:
            row = tables["hosted_room_links"].get(room_id, {}).get(member_id)
            if row is None:
                return False
            row.update(status=status, updated_at=timestamp)
            return True

    def delete_room_link_records(self, *, placement: HostedPlacement, room_id: str) -> int:
        with write_tables(self._store, placement) as tables:
            return len(tables["hosted_room_links"].pop(room_id, {}))

    def revoke_room_grant_scope(self, *, placement: HostedPlacement, claims: RoomGrantClaims, expires_at: float, now: float | None = None) -> None:
        scope_key = rules.room_grant_scope_key(claims)
        timestamp, expiry = rules.clock(now), float(expires_at)
        if expiry <= timestamp:
            return
        with write_tables(self._store, placement) as tables:
            grants = tables["hosted_room_revoked_grants"]
            for key in [key for key, row in grants.items() if float(row["expires_at"]) <= timestamp]:
                del grants[key]
            old = grants.get(scope_key)
            grants[scope_key] = {
                "scope_key": scope_key, "expires_at": max(float(old["expires_at"]), expiry) if old else expiry,
                "revoked_before": max(float(old["revoked_before"]), timestamp) if old else timestamp}
            row = tables["hosted_room_peer_reservations"].get(str(claims.room_id or ""), {}).get((str(claims.member_id or ""), str(claims.target_profile or "")))
            if row is not None and (row["authority_gateway_id"], row["authority_epoch"]) == (str(claims.authority_gateway_id or ""), int(claims.authority_epoch or 0)):
                row.update(revoked_at=timestamp, updated_at=timestamp)

    def reserve_peer_room(self, *, placement: HostedPlacement, claims: RoomGrantClaims, expires_at: float, now: float | None = None) -> None:
        timestamp, expiry = rules.clock(now), float(expires_at)
        if expiry <= timestamp:
            raise HostedRoomError("peer room reservation must expire in the future")
        gateway_id, epoch = claims.authority_gateway_id, int(claims.authority_epoch)
        key = (claims.member_id, claims.target_profile)
        with write_tables(self._store, placement) as tables:
            _drop_expired_reservations(tables, timestamp)
            rows = tables.rows("hosted_room_peer_reservations", claims.room_id)
            same_profile = [row for (_, profile), row in rows.items() if profile == claims.target_profile]
            if any(_live(row, timestamp) and _superseded(row, gateway_id, epoch) for row in same_profile):
                raise AuthorityConflictError("peer room reservation authority changed")
            for row in same_profile:
                if int(row["authority_epoch"]) < epoch and row["revoked_at"] is None:
                    row.update(revoked_at=timestamp, updated_at=timestamp)
            existing = rows.get(key)
            if existing is not None and _superseded(existing, gateway_id, epoch):
                raise AuthorityConflictError("peer room reservation authority changed")
            if existing is None:
                rows[key] = {
                    "room_id": claims.room_id, "member_id": claims.member_id, "target_profile": claims.target_profile,
                    "authority_gateway_id": gateway_id, "authority_epoch": epoch, "expires_at": expiry, "revoked_at": None,
                    "created_at": timestamp, "updated_at": timestamp}
            else:
                existing.update(authority_gateway_id=gateway_id, authority_epoch=epoch, expires_at=max(float(existing["expires_at"]), expiry), revoked_at=None, updated_at=timestamp)

    def upsert_remote_run_receipt(self, *, placement: HostedPlacement, receipt: RemoteRunReceipt, now: float | None = None) -> None:
        timestamp = rules.clock(now)
        identity = _identity(receipt)
        with write_tables(self._store, placement) as tables:
            runs = tables.rows("hosted_room_remote_runs", receipt.room_id)
            existing = runs.get(identity)
            if existing is not None:
                if (existing["run_id"], existing["session_id"]) != (receipt.run_id, receipt.session_id):
                    raise HostedRoomError("remote run receipt conflicts with its logical task")
                existing["updated_at"] = timestamp
                return
            runs[identity] = {**{column: getattr(receipt, column) for column in _RECEIPT_COLUMNS}, "created_at": timestamp, "updated_at": timestamp}
            capture_transition_locked(tables, receipt.room_id)

    def retire_room_link_records(self, *, placement: HostedPlacement, room_id: str, owe_routes: bool = False, revoked_digests: Collection[str] | None = None, now: float | None = None) -> int:
        timestamp = rules.clock(now)
        with write_tables(self._store, placement) as tables:
            links = tables["hosted_room_links"].pop(room_id, {})
            for member_id in sorted(links) if owe_routes else ():
                row = links[member_id]
                _owe_revoke(tables, room_id=room_id, member_id=member_id, target_url=row["target_url"], grant=row["grant"],
                            expires_at=room_grant_hard_expiry(row["grant"]), now=timestamp)
            held = [row for row in tables["hosted_room_owed_revokes"].values() if row["room_id"] == room_id and row["held"]]
            if owe_routes:
                for row in held:
                    row.update(held=0, next_attempt_at=timestamp)
            else:
                released = {id(row) for row in held if revoked_digests is None or row["grant_digest"] in revoked_digests}
                _delete_owed(tables, lambda row: id(row) not in released)
            return len(links)

    def owe_room_grant_revoke(self, *, placement: HostedPlacement, room_id: str, member_id: str, target_url: str, grant: str, expires_at: float, now: float | None = None) -> None:
        timestamp = rules.clock(now)
        with write_tables(self._store, placement) as tables:
            _owe_revoke(tables, room_id=room_id, member_id=member_id, target_url=target_url, grant=grant, expires_at=expires_at, now=timestamp)

    def hold_room_grant_revoke(self, *, placement: HostedPlacement, room_id: str, member_id: str, target_url: str, grant: str, expires_at: float, now: float | None = None) -> None:
        timestamp = rules.clock(now)
        with write_tables(self._store, placement) as tables:
            live = room_row(tables, room_id, include_disbanded=True)["disbanded_at"] is None
            try:
                _owe_revoke(tables, room_id=room_id, member_id=member_id, target_url=target_url, grant=grant, expires_at=expires_at, now=timestamp, held=live)
            except OwedRevokesFullError as exc:
                if live:
                    raise HostedRoomError("too many held group chat grant revokes") from exc
                raise

    def settle_room_grant_revoke(self, *, placement: HostedPlacement, room_id: str, member_id: str, grant_digest: str, retry_at: float | None = None) -> None:
        with write_tables(self._store, placement) as tables:
            owed = tables["hosted_room_owed_revokes"]
            key = (room_id, member_id, grant_digest)
            if retry_at is None:
                owed.pop(key, None)
            elif key in owed:
                owed[key].update(attempts=owed[key]["attempts"] + 1, next_attempt_at=float(retry_at))

    def drop_remote_run_receipt(self, *, placement: HostedPlacement, identity: RemoteRunIdentity, run_id: str) -> bool:
        with write_tables(self._store, placement) as tables:
            runs = tables["hosted_room_remote_runs"].get(identity.room_id, {})
            key = _identity(identity)
            if key not in runs or runs[key]["run_id"] != run_id:
                return False
            del runs[key]
            capture_transition_locked(tables, identity.room_id)
            return True

    def room_has_peer_obligations(self, *, placement: HostedPlacement, room_id: str, now: float | None = None) -> bool:
        timestamp = rules.clock(now)
        tables = read_tables(self._store, placement)
        return bool(tables["hosted_room_links"].get(room_id)) or any(
            row["room_id"] == room_id and float(row["expires_at"]) > timestamp for row in tables["hosted_room_owed_revokes"].values())

    def owes_room_grant_revokes(self, *, placement: HostedPlacement) -> bool:
        return bool(read_tables(self._store, placement)["hosted_room_owed_revokes"])

    def due_room_grant_revokes(self, *, placement: HostedPlacement, limit: int, now: float | None = None) -> Sequence[Mapping[str, JsonValue]]:
        timestamp = rules.clock(now)
        owed = read_tables(self._store, placement)["hosted_room_owed_revokes"]
        due = sorted((row for row in owed.values() if not row["held"] and float(row["next_attempt_at"]) <= timestamp and float(row["expires_at"]) > timestamp),
                     key=lambda row: (row["next_attempt_at"], row["created_at"]))[:int(limit)]
        rows = [{column: row[column] for column in _DUE_COLUMNS} for row in due]
        if any(float(row["expires_at"]) <= timestamp for row in owed.values()):
            with write_tables(self._store, placement) as tables:
                _delete_owed(tables, lambda row: float(row["expires_at"]) > timestamp)
        return rows

    def held_room_grant_revokes(self, *, placement: HostedPlacement, room_id: str, now: float | None = None) -> Sequence[Mapping[str, JsonValue]]:
        timestamp = rules.clock(now)
        with write_tables(self._store, placement) as tables:
            _delete_owed(tables, lambda row: not (row["room_id"] == room_id and row["held"] and float(row["expires_at"]) <= timestamp))
            owed = tables["hosted_room_owed_revokes"]
            return [{"target_url": owed[key]["target_url"], "grant": owed[key]["grant"]} for key in sorted(owed)
                    if key[0] == room_id and owed[key]["held"]]

    def list_room_link_records(self, *, placement: HostedPlacement) -> Sequence[Mapping[str, JsonValue]]:
        links = read_tables(self._store, placement)["hosted_room_links"]
        return [dict(links[room_id][member_id]) for room_id in sorted(links) for member_id in sorted(links[room_id])]

    def room_grant_is_revoked(self, *, placement: HostedPlacement, claims: RoomGrantClaims, now: float | None = None) -> bool:
        timestamp = rules.clock(now)
        scope_key = rules.room_grant_scope_key(claims)
        issued_at = float(claims.issued_at or 0)
        row = read_tables(self._store, placement)["hosted_room_revoked_grants"].get(scope_key)
        return row is not None and float(row["expires_at"]) > timestamp and issued_at <= float(row["revoked_before"])

    def peer_room_grant_is_current(self, *, placement: HostedPlacement, claims: RoomGrantClaims, now: float | None = None) -> bool:
        timestamp = rules.clock(now)
        row = read_tables(self._store, placement)["hosted_room_peer_reservations"].get(claims.room_id, {}).get((claims.member_id, claims.target_profile))
        return (row is not None and (row["authority_gateway_id"], row["authority_epoch"]) == (claims.authority_gateway_id, int(claims.authority_epoch))
                and _live(row, timestamp))

    def probe_peer_room_reservation(self, *, placement: HostedPlacement, room_id: str, target_profile: str, now: float | None = None) -> bool:
        timestamp = rules.clock(now)
        tables = peek_tables(self._store, placement)
        if tables is None:
            return False
        rows = tables["hosted_room_peer_reservations"].get(room_id, {})
        return any(profile == target_profile and _live(row, timestamp) for (_, profile), row in rows.items())

    def remote_run_receipt(self, *, placement: HostedPlacement, identity: RemoteRunIdentity) -> Mapping[str, JsonValue] | None:
        row = read_tables(self._store, placement)["hosted_room_remote_runs"].get(identity.room_id, {}).get(_identity(identity))
        return dict(row) if row is not None else None


register_repository("hosted_room_peers", InMemoryHostedRoomPeerRepository)

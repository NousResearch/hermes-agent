"""In-memory reference ``HostedRoomPeerRepository``: links, grant revocations, peer reservations and remote-run receipts over the placement's hosted-room tables (``hosted_room_tables``).

Reference limits: the probe reads memory and cannot fail, so ``RoomProbeUnavailableError`` is never raised; the work-record capture inside ``upsert_remote_run_receipt`` runs only its escaping checks (``hosted_room_work``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from persistence import JsonValue
from persistence.gateway.hosted_rooms import (
    AuthorityConflictError,
    HostedPlacement,
    HostedRoomError,
    RemoteRunIdentity,
    RemoteRunReceipt,
    RoomGrantClaims,
    RoomLinkRecord,
)

from ..backend import register_repository
from . import hosted_room_rules as rules
from .hosted_room_tables import HostedTables, peek_tables, read_tables, write_tables
from .hosted_room_work import capture_transition_locked

__all__ = ["InMemoryHostedRoomPeerRepository"]

# The identity and link columns and the remote-run table's column order.
_IDENTITY_COLUMNS = (
    "room_id", "home_install_id", "authority_gateway_id", "authority_epoch", "member_id", "target_install_id",
    "target_profile", "task_id", "execution_generation")
_LINK_COLUMNS = (
    "room_id", "member_id", "target_url", "target_profile", "grant", "catalog_json", "cancellation_scope_id",
    "trace_id", "transport_security", "status", "updated_at")
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


class InMemoryHostedRoomPeerRepository:
    """See ``persistence.gateway.hosted_rooms.HostedRoomPeerRepository`` for the contract each method implements."""

    def __init__(self, store: Any) -> None:
        self._store = store

    def upsert_room_link_record(self, *, placement: HostedPlacement, record: RoomLinkRecord, max_links: int) -> None:
        with write_tables(self._store, placement) as tables:
            links = tables["hosted_room_links"]
            if record.member_id not in links.get(record.room_id, {}) and sum(len(rows) for rows in links.values()) >= max_links:
                raise HostedRoomError("too many stored room links")
            tables.rows("hosted_room_links", record.room_id)[record.member_id] = {column: getattr(record, column) for column in _LINK_COLUMNS}

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

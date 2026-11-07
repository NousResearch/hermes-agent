"""``HostedRoomPeerRepository`` on SQLite: peer room links, grant revocations, peer reservations and remote-run
receipts in the file ``placement`` names, each write one transaction and each read one read context on it."""

from __future__ import annotations

import contextlib
import os
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import Any

from persistence import JsonValue
from persistence.gateway.hosted_rooms import (
    AuthorityConflictError, HostedPlacement, HostedRoomError, RemoteRunIdentity, RemoteRunReceipt, RoomGrantClaims,
    RoomLinkRecord,
)
from session_rules.hosted_rooms import _claim_values, clock as _now, room_grant_scope_key

from .._backend import SQLiteStore, register_repository, translated
from ._hosted_rooms_schema import ROOMS_STEP
from ._hosted_store import HostedFiles
from .hosted_rooms import _probe

__all__ = ["SQLiteHostedRoomPeerRepository"]

# The remote-run receipt key, the room-link columns in table order, and the SQL built from them.
_REMOTE_RUN_IDENTITY_COLUMNS = (
    "room_id", "home_install_id", "authority_gateway_id", "authority_epoch", "member_id", "target_install_id",
    "target_profile", "task_id", "execution_generation")
_LINK_COLUMNS = (
    "room_id", "member_id", "target_url", "target_profile", "grant", "catalog_json", "cancellation_scope_id",
    "trace_id", "transport_security", "status", "updated_at")
_REMOTE_RUN_WHERE = " AND ".join(f"{column}=?" for column in _REMOTE_RUN_IDENTITY_COLUMNS)
_SELECT_REMOTE_RUN = f"SELECT * FROM hosted_room_remote_runs WHERE {_REMOTE_RUN_WHERE}"
_LIVE_RESERVATION_WHERE = ("WHERE room_id=? AND target_profile=? AND expires_at>? AND revoked_at IS NULL")
_SELECT_LIVE_RESERVATION = (f"SELECT 1 FROM hosted_room_peer_reservations {_LIVE_RESERVATION_WHERE} LIMIT 1")


def _reservation_superseded(row: sqlite3.Row, gateway_id: str, epoch: int) -> bool:
    """A newer epoch, or the same epoch under another gateway, outranks this claim."""
    row_epoch = int(row["authority_epoch"])
    return row_epoch > epoch or (row_epoch == epoch and str(row["authority_gateway_id"]) != gateway_id)


def _reservation_values(claims: RoomGrantClaims) -> tuple[str, str, str, str, int]:
    """(room_id, member_id, target_profile, authority_gateway_id, authority_epoch) of claims the caller validated."""
    return (claims.room_id, claims.member_id, claims.target_profile, claims.authority_gateway_id,
            int(claims.authority_epoch or 0))


# The receipt key of a typed record or a stored row.
def _remote_run_identity(record: RemoteRunIdentity | RemoteRunReceipt | sqlite3.Row) -> tuple[Any, ...]:
    if isinstance(record, sqlite3.Row):
        return tuple(record[column] for column in _REMOTE_RUN_IDENTITY_COLUMNS)
    return tuple(getattr(record, column) for column in _REMOTE_RUN_IDENTITY_COLUMNS)


class SQLiteHostedRoomPeerRepository:
    """Peer room links, grant revocations, peer reservations and remote-run receipts."""

    def __init__(self, store: SQLiteStore) -> None:
        self._store = store
        self._files = HostedFiles(store, (ROOMS_STEP,))

    # ── writes ──────────────────────────────────────────────────────────────

    # Inserts or replaces the member's link, refusing a new link once ``max_links`` are stored; the file holding the
    # grant is then owner-only.
    def upsert_room_link_record(self, *, placement: HostedPlacement, record: RoomLinkRecord, max_links: int) -> None:
        def write(conn: sqlite3.Connection) -> None:
            existing = conn.execute(
                "SELECT 1 FROM hosted_room_links WHERE room_id=? AND member_id=?", (record.room_id, record.member_id)
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
                tuple(getattr(record, column) for column in _LINK_COLUMNS))
        with translated():
            self._files._hosted_write(placement, write, immediate=True)
        if os.name == "posix":
            with contextlib.suppress(OSError):
                self._files.path(placement).chmod(0o600)

    # True when the link existed and took the new status.
    def update_room_link_status(self, *, placement: HostedPlacement, room_id: str, member_id: str, status: str, now: float | None = None) -> bool:
        with translated():
            return self._files._hosted_write(placement, lambda conn: conn.execute(
                "UPDATE hosted_room_links SET status=?, updated_at=? WHERE room_id=? AND member_id=?",
                (status, _now(now), room_id, member_id)).rowcount == 1, immediate=True)

    # Deletes the room's links and returns how many.
    def delete_room_link_records(self, *, placement: HostedPlacement, room_id: str) -> int:
        with translated():
            return self._files._hosted_write(placement, lambda conn: conn.execute(
                "DELETE FROM hosted_room_links WHERE room_id=?", (room_id,)).rowcount, immediate=True)

    # Revokes the scope's grants issued until now and the matching reservation; expired revocations go first.
    def revoke_room_grant_scope(self, *, placement: HostedPlacement, claims: RoomGrantClaims, expires_at: float, now: float | None = None) -> None:
        fields = asdict(claims)
        scope_key = room_grant_scope_key(fields)
        timestamp = _now(now)
        expiry = float(expires_at)
        if expiry <= timestamp:
            return

        def write(conn: sqlite3.Connection) -> None:
            conn.execute("DELETE FROM hosted_room_revoked_grants WHERE expires_at<=?", (timestamp,))
            conn.execute("""INSERT INTO hosted_room_revoked_grants(
                       scope_key, expires_at, revoked_before
                   ) VALUES (?, ?, ?)
                   ON CONFLICT(scope_key) DO UPDATE SET
                       expires_at=MAX(hosted_room_revoked_grants.expires_at,
                                      excluded.expires_at),
                       revoked_before=MAX(hosted_room_revoked_grants.revoked_before,
                                          excluded.revoked_before)""", (scope_key, expiry, timestamp))
            conn.execute("""UPDATE hosted_room_peer_reservations SET revoked_at=?, updated_at=? WHERE room_id=?
                    AND member_id=? AND target_profile=? AND authority_gateway_id=?
                    AND authority_epoch=?""",
                (
                    timestamp, timestamp,
                    *_claim_values(fields, ("room_id", "member_id", "target_profile", "authority_gateway_id")).values(),
                    int(fields.get("authority_epoch") or 0)))
        with translated():
            self._files._hosted_write(placement, write, immediate=True)

    # The caller validates the claims first. Refuses a reservation a live one outranks, revokes older epochs, upserts.
    def reserve_peer_room(self, *, placement: HostedPlacement, claims: RoomGrantClaims, expires_at: float, now: float | None = None) -> None:
        timestamp = _now(now)
        expiry = float(expires_at)
        if expiry <= timestamp:
            raise HostedRoomError("peer room reservation must expire in the future")
        values = _reservation_values(claims)
        room_id, _, target_profile, gateway_id, epoch = values

        def write(conn: sqlite3.Connection) -> None:
            conn.execute("DELETE FROM hosted_room_peer_reservations WHERE expires_at<=?", (timestamp,))
            authority_rows = conn.execute(
                f"""SELECT authority_gateway_id, authority_epoch
                    FROM hosted_room_peer_reservations {_LIVE_RESERVATION_WHERE}""", (room_id, target_profile, timestamp)
            ).fetchall()
            if any(_reservation_superseded(row, gateway_id, epoch) for row in authority_rows):
                raise AuthorityConflictError("peer room reservation authority changed")
            conn.execute("""UPDATE hosted_room_peer_reservations SET revoked_at=?, updated_at=? WHERE room_id=?
                    AND target_profile=? AND authority_epoch<? AND revoked_at IS NULL""",
                (timestamp, timestamp, room_id, target_profile, epoch))
            existing = conn.execute("""SELECT authority_gateway_id, authority_epoch FROM hosted_room_peer_reservations
                    WHERE room_id=? AND member_id=? AND target_profile=?""", values[:3]).fetchone()
            if existing is not None and _reservation_superseded(existing, gateway_id, epoch):
                raise AuthorityConflictError("peer room reservation authority changed")
            conn.execute("""INSERT INTO hosted_room_peer_reservations(
                       room_id, member_id, target_profile, authority_gateway_id,
                       authority_epoch, expires_at, revoked_at, created_at, updated_at
                   ) VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?)
                   ON CONFLICT(room_id, member_id, target_profile) DO UPDATE SET
                       authority_gateway_id=excluded.authority_gateway_id,
                       authority_epoch=excluded.authority_epoch,
                       expires_at=MAX(hosted_room_peer_reservations.expires_at,
                                      excluded.expires_at),
                       revoked_at=NULL,
                       updated_at=excluded.updated_at""", (*values, expiry, timestamp, timestamp))
        with translated():
            self._files._hosted_write(placement, write, immediate=True)

    # Records a remote run once per receipt key; a repeat with the same run and session only moves ``updated_at``.
    def upsert_remote_run_receipt(self, *, placement: HostedPlacement, receipt: RemoteRunReceipt, now: float | None = None) -> None:
        timestamp = _now(now)
        identity = _remote_run_identity(receipt)
        immutable = (*identity, receipt.run_id, receipt.session_id)

        def write(conn: sqlite3.Connection) -> None:
            existing = conn.execute(_SELECT_REMOTE_RUN, identity).fetchone()
            if existing is not None:
                if (*_remote_run_identity(existing), existing["run_id"], existing["session_id"]) != immutable:
                    raise HostedRoomError("remote run receipt conflicts with its logical task")
                conn.execute(
                    f"UPDATE hosted_room_remote_runs SET updated_at=? WHERE {_REMOTE_RUN_WHERE}", (timestamp, *identity))
                return
            conn.execute("""INSERT INTO hosted_room_remote_runs(
                       room_id, home_install_id, authority_gateway_id,
                       authority_epoch, member_id, target_install_id,
                       target_profile, task_id, execution_generation, run_id,
                       session_id, created_at, updated_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", (*immutable, timestamp, timestamp))
            from ._hosted_work import capture_transition_locked
            capture_transition_locked(conn, receipt.room_id)
        with translated():
            self._files._hosted_write(placement, write, immediate=True)

    # ── reads ───────────────────────────────────────────────────────────────

    # Every stored link, by room and member.
    def list_room_link_records(self, *, placement: HostedPlacement) -> Sequence[Mapping[str, JsonValue]]:
        with translated():
            rows = self._files._hosted_read(placement, lambda conn: conn.execute(
                """SELECT room_id, member_id, target_url, target_profile, grant,
                          catalog_json, cancellation_scope_id, trace_id,
                          transport_security, status, updated_at
                     FROM hosted_room_links
                 ORDER BY room_id, member_id""").fetchall())
        return [dict(row) for row in rows]

    # True when an unexpired revocation of the claims' scope covers their issue time.
    def room_grant_is_revoked(self, *, placement: HostedPlacement, claims: RoomGrantClaims, now: float | None = None) -> bool:
        timestamp = _now(now)
        scope_key = room_grant_scope_key(asdict(claims))
        issued_at = float(claims.issued_at or 0)
        with translated():
            row = self._files._hosted_read(placement, lambda conn: conn.execute(
                """SELECT revoked_before FROM hosted_room_revoked_grants
                WHERE scope_key=? AND expires_at>?""", (scope_key, timestamp)).fetchone())
        return row is not None and issued_at <= float(row["revoked_before"])

    # The caller validates the claims first; true when their reservation is live and unrevoked.
    def peer_room_grant_is_current(self, *, placement: HostedPlacement, claims: RoomGrantClaims, now: float | None = None) -> bool:
        timestamp = _now(now)
        values = _reservation_values(claims)
        with translated():
            return self._files._hosted_read(placement, lambda conn: conn.execute(
                """SELECT 1 FROM hosted_room_peer_reservations WHERE room_id=? AND member_id=?
                AND target_profile=? AND authority_gateway_id=? AND authority_epoch=?
                AND expires_at>? AND revoked_at IS NULL LIMIT 1""", (*values, timestamp)).fetchone()) is not None

    # The caller validates the arguments first; probes for a live reservation with a short lock wait and no schema step.
    def probe_peer_room_reservation(self, *, placement: HostedPlacement, room_id: str, target_profile: str, now: float | None = None) -> bool:
        params = (room_id, target_profile, _now(now))
        return _probe(
            self._files.path(placement), "hosted_room_peer_reservations", _SELECT_LIVE_RESERVATION, params,
            "peer room ownership is temporarily unavailable")

    # The stored receipt for this key, or None.
    def remote_run_receipt(self, *, placement: HostedPlacement, identity: RemoteRunIdentity) -> Mapping[str, JsonValue] | None:
        with translated():
            row = self._files._hosted_read(
                placement, lambda conn: conn.execute(_SELECT_REMOTE_RUN, _remote_run_identity(identity)).fetchone())
        return dict(row) if row is not None else None


register_repository("hosted_room_peers", SQLiteHostedRoomPeerRepository)

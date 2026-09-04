"""Private SQLite storage for negotiated hosted-room links.

Route metadata and its scoped grant share the gateway's private root ``state.db``. SQLite WAL plus
``BEGIN IMMEDIATE`` owns concurrency; grants are never included in reprs, status payloads, or exception messages.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Mapping

from gateway import hosted_room_link_records
from gateway.hosted_room_peer import (
    GatewayRoomCatalog, HostedRoomPeerError, TransportSecurity, validate_room_link_url)
from gateway.hosted_rooms_common import DbPath, compact_json, exact_fields, identifier


MAX_LINKS = 512
MAX_GRANT_CHARS = 16 * 1024
_LEGACY_FIELDS = {
    "room_id", "member_id", "target_url", "target_profile", "grant", "catalog", "cancellation_scope_id", "trace_id",
    "updated_at"}
_OPTIONAL_FIELDS = {"transport_security", "status"}
# SQLite record columns that map 1:1 onto mapping fields, in record order
# (``catalog_json`` is the serialized ``catalog``).
_RECORD_FIELDS = (
    "room_id", "member_id", "target_url", "target_profile", "grant", "cancellation_scope_id", "trace_id",
    "transport_security", "status", "updated_at")
_STATUSES = {"ready", "unavailable", "needs_reauthorization"}


@dataclass(frozen=True)
class StoredRoomLink:
    room_id: str
    member_id: str
    target_url: str
    target_profile: str
    grant: str = field(repr=False)
    catalog: GatewayRoomCatalog
    cancellation_scope_id: str
    trace_id: str
    transport_security: TransportSecurity
    status: str
    updated_at: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "StoredRoomLink":
        _link_fields(value)
        room_id = _short_string(value["room_id"], "room_id")
        member_id = _short_string(value["member_id"], "member_id")
        target_profile = _short_string(value["target_profile"], "target_profile")
        target_url, detected_security = validate_room_link_url(value["target_url"])
        transport_security = str(value.get("transport_security") or detected_security)
        if transport_security != detected_security:
            raise HostedRoomPeerError("transport_security does not match target_url")
        grant = str(value["grant"] or "")
        if not grant or len(grant) > MAX_GRANT_CHARS:
            raise HostedRoomPeerError("room grant is missing or too large")
        status = str(value.get("status") or "ready")
        if status not in _STATUSES:
            raise HostedRoomPeerError("stored room link status is invalid")
        updated_at = float(value["updated_at"])
        if not updated_at > 0:
            raise HostedRoomPeerError("updated_at must be positive")
        return cls(
            room_id=room_id, member_id=member_id, target_url=target_url, target_profile=target_profile,
            grant=grant, catalog=GatewayRoomCatalog.from_mapping(value["catalog"]),
            cancellation_scope_id=_short_string(value["cancellation_scope_id"], "cancellation_scope_id"),
            trace_id=_short_string(value["trace_id"], "trace_id"), transport_security=transport_security,
            status=status, updated_at=updated_at)

    @classmethod
    def from_record(cls, value: Mapping[str, Any]) -> "StoredRoomLink":
        try:
            catalog = json.loads(str(value["catalog_json"]))
        except Exception as exc:
            raise HostedRoomPeerError("stored room link catalog is unreadable") from exc
        return cls.from_mapping({**{name: value[name] for name in _RECORD_FIELDS}, "catalog": catalog})

    def catalog_mapping(self) -> dict[str, Any]:
        return self.catalog.as_mapping()

    def as_record(self) -> dict[str, Any]:
        record = {name: getattr(self, name) for name in _RECORD_FIELDS}
        record["catalog_json"] = compact_json(self.catalog_mapping(), ensure_ascii=False)
        return record


_link_fields = partial(
    exact_fields, label="stored room link", required=_LEGACY_FIELDS, optional=_OPTIONAL_FIELDS,
    error=HostedRoomPeerError, not_object="stored room link fields are invalid",
    missing_fmt="stored room link fields are invalid", unknown_fmt="stored room link fields are invalid")


def _short_string(value: Any, field: str) -> str:
    return identifier(
        str(value or ""), label=field, error=HostedRoomPeerError, max_chars=256, pattern=None,
        invalid=f"{field} is invalid")


def _link_rows(db_path: DbPath) -> list[dict[str, Any]]:
    rows = hosted_room_link_records.list_room_link_records(db_path)
    if len(rows) > MAX_LINKS:
        raise HostedRoomPeerError("stored room link list is invalid")
    return rows


def load_room_links(db_path: DbPath) -> tuple[StoredRoomLink, ...]:
    return tuple(StoredRoomLink.from_record(row) for row in _link_rows(db_path))


def load_room_links_tolerant(db_path: DbPath) -> tuple[tuple[StoredRoomLink, ...], tuple[str, ...]]:
    """Load healthy routes while quarantining malformed rows by identity."""
    links, errors = [], []
    for row in _link_rows(db_path):
        try:
            links.append(StoredRoomLink.from_record(row))
        except Exception:
            errors.append(f"{row.get('room_id') or 'unknown'}:{row.get('member_id') or 'unknown'}:invalid")
    return tuple(links), tuple(errors)


def save_room_link(
    db_path: Path | str,
    link: StoredRoomLink,
    *,
    expected_grant_sha256: str | None = None,
) -> None:
    hosted_room_link_records.upsert_room_link_record(
        db_path,
        record=link.as_record(),
        max_links=MAX_LINKS,
        expected_grant_sha256=expected_grant_sha256,
    )
    if os.name == "posix":
        try:
            Path(db_path).chmod(0o600)
        except OSError:
            pass


def mark_room_link_status(
    db_path: Path | str,
    *,
    room_id: str,
    member_id: str,
    status: str,
    expected_grant_sha256: str | None = None,
) -> bool:
    if status not in _STATUSES:
        raise HostedRoomPeerError("stored room link status is invalid")
    return hosted_room_link_records.update_room_link_status(
        db_path,
        room_id=_short_string(room_id, "room_id"),
        member_id=_short_string(member_id, "member_id"),
        status=status,
        expected_grant_sha256=expected_grant_sha256,
    )


def make_stored_link(
    *, room_id: str, member_id: str, target_url: str, target_profile: str, grant: str, catalog: GatewayRoomCatalog,
    cancellation_scope_id: str, trace_id: str) -> StoredRoomLink:
    target_url, transport_security = validate_room_link_url(target_url)
    return StoredRoomLink.from_mapping({
        "room_id": room_id, "member_id": member_id, "target_url": target_url, "target_profile": target_profile,
        "grant": grant, "catalog": catalog.as_mapping(), "cancellation_scope_id": cancellation_scope_id,
        "trace_id": trace_id, "transport_security": transport_security, "status": "ready", "updated_at": time.time()})


def load_room_link(
    db_path: Path | str,
    *,
    room_id: str,
    member_id: str,
) -> StoredRoomLink | None:
    """Load one exact persisted route without scanning unrelated rooms."""

    row = hosted_room_link_records.room_link_record(
        db_path,
        room_id=_short_string(room_id, "room_id"),
        member_id=_short_string(member_id, "member_id"),
    )
    return StoredRoomLink.from_record(row) if row is not None else None

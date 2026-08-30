"""Private SQLite storage for negotiated hosted-room links.

Route metadata and its scoped grant share the gateway's private root
``state.db``. SQLite WAL plus ``BEGIN IMMEDIATE`` owns concurrency; grants are
never included in reprs, status payloads, or exception messages.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from gateway import hosted_rooms
from gateway.hosted_room_peer import (
    GatewayRoomCatalog,
    HostedRoomPeerError,
    TransportSecurity,
    validate_room_link_url,
)


MAX_LINKS = 512
MAX_GRANT_CHARS = 16 * 1024
_LEGACY_FIELDS = {
    "room_id",
    "member_id",
    "target_url",
    "target_profile",
    "grant",
    "catalog",
    "cancellation_scope_id",
    "trace_id",
    "updated_at",
}
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
        allowed = _LEGACY_FIELDS | {"transport_security", "status"}
        if set(value) - allowed or not _LEGACY_FIELDS.issubset(value):
            raise HostedRoomPeerError("stored room link fields are invalid")
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
            room_id=room_id,
            member_id=member_id,
            target_url=target_url,
            target_profile=target_profile,
            grant=grant,
            catalog=GatewayRoomCatalog.from_mapping(value["catalog"]),
            cancellation_scope_id=_short_string(
                value["cancellation_scope_id"], "cancellation_scope_id"
            ),
            trace_id=_short_string(value["trace_id"], "trace_id"),
            transport_security=transport_security,  # type: ignore[arg-type]
            status=status,
            updated_at=updated_at,
        )

    @classmethod
    def from_record(cls, value: Mapping[str, Any]) -> "StoredRoomLink":
        try:
            catalog = json.loads(str(value["catalog_json"]))
        except Exception as exc:
            raise HostedRoomPeerError("stored room link catalog is unreadable") from exc
        return cls.from_mapping({
            "room_id": value["room_id"],
            "member_id": value["member_id"],
            "target_url": value["target_url"],
            "target_profile": value["target_profile"],
            "grant": value["grant"],
            "catalog": catalog,
            "cancellation_scope_id": value["cancellation_scope_id"],
            "trace_id": value["trace_id"],
            "transport_security": value["transport_security"],
            "status": value["status"],
            "updated_at": value["updated_at"],
        })

    def catalog_mapping(self) -> dict[str, Any]:
        value = {
            "installation_id": self.catalog.installation_id,
            "protocol_versions": list(self.catalog.protocol_versions),
            "link_modes": list(self.catalog.link_modes),
            "persistent_process": self.catalog.persistent_process,
            "text": self.catalog.text,
            "attachments": self.catalog.attachments,
            "execution_policy": self.catalog.execution_policy.as_mapping(),
            "catalog_digest": self.catalog.catalog_digest,
        }
        if (
            self.catalog.endpoint_url is not None
            or self.catalog.endpoint_reason is not None
        ):
            value["endpoint"] = self.catalog.endpoint_mapping()
        return value

    def as_record(self) -> dict[str, Any]:
        return {
            "room_id": self.room_id,
            "member_id": self.member_id,
            "target_url": self.target_url,
            "target_profile": self.target_profile,
            "grant": self.grant,
            "catalog_json": json.dumps(
                self.catalog_mapping(), sort_keys=True, separators=(",", ":")
            ),
            "cancellation_scope_id": self.cancellation_scope_id,
            "trace_id": self.trace_id,
            "transport_security": self.transport_security,
            "status": self.status,
            "updated_at": self.updated_at,
        }


def _short_string(value: Any, field: str) -> str:
    normalized = str(value or "").strip()
    if not normalized or len(normalized) > 256:
        raise HostedRoomPeerError(f"{field} is invalid")
    return normalized


def load_room_links(db_path: Path | str) -> tuple[StoredRoomLink, ...]:
    rows = hosted_rooms.list_room_link_records(db_path)
    if len(rows) > MAX_LINKS:
        raise HostedRoomPeerError("stored room link list is invalid")
    return tuple(StoredRoomLink.from_record(row) for row in rows)


def load_room_links_tolerant(
    db_path: Path | str,
) -> tuple[tuple[StoredRoomLink, ...], tuple[str, ...]]:
    """Load healthy routes while quarantining malformed rows by identity."""
    rows = hosted_rooms.list_room_link_records(db_path)
    if len(rows) > MAX_LINKS:
        raise HostedRoomPeerError("stored room link list is invalid")
    links = []
    errors = []
    for row in rows:
        try:
            links.append(StoredRoomLink.from_record(row))
        except Exception:
            room = str(row.get("room_id") or "unknown")
            member = str(row.get("member_id") or "unknown")
            errors.append(f"{room}:{member}:invalid")
    return tuple(links), tuple(errors)


def save_room_link(db_path: Path | str, link: StoredRoomLink) -> None:
    hosted_rooms.upsert_room_link_record(
        db_path, record=link.as_record(), max_links=MAX_LINKS
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
) -> bool:
    if status not in _STATUSES:
        raise HostedRoomPeerError("stored room link status is invalid")
    return hosted_rooms.update_room_link_status(
        db_path,
        room_id=_short_string(room_id, "room_id"),
        member_id=_short_string(member_id, "member_id"),
        status=status,
    )


def make_stored_link(
    *,
    room_id: str,
    member_id: str,
    target_url: str,
    target_profile: str,
    grant: str,
    catalog: GatewayRoomCatalog,
    cancellation_scope_id: str,
    trace_id: str,
) -> StoredRoomLink:
    target_url, transport_security = validate_room_link_url(target_url)
    return StoredRoomLink.from_mapping({
        "room_id": room_id,
        "member_id": member_id,
        "target_url": target_url,
        "target_profile": target_profile,
        "grant": grant,
        "catalog": {
            "installation_id": catalog.installation_id,
            "protocol_versions": list(catalog.protocol_versions),
            "link_modes": list(catalog.link_modes),
            "persistent_process": catalog.persistent_process,
            "text": catalog.text,
            "attachments": catalog.attachments,
            "execution_policy": catalog.execution_policy.as_mapping(),
            "catalog_digest": catalog.catalog_digest,
            **(
                {"endpoint": catalog.endpoint_mapping()}
                if catalog.endpoint_url is not None
                or catalog.endpoint_reason is not None
                else {}
            ),
        },
        "cancellation_scope_id": cancellation_scope_id,
        "trace_id": trace_id,
        "transport_security": transport_security,
        "status": "ready",
        "updated_at": time.time(),
    })

"""Normalized inert artifact metadata for Mission Control.

This module only reshapes existing Mission Control structured summaries. It
does not open attachment files, inspect paths, fetch URLs, parse payloads, or
perform validation/execution.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _counts(**values: Any) -> dict[str, int]:
    return {key: int(value or 0) for key, value in values.items()}


def _flags(*, dry_run: Any = None, review_required: Any = None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    flags: dict[str, Any] = {}
    if dry_run is not None:
        flags["dry_run"] = dry_run is True
    if review_required is not None:
        flags["review_required"] = review_required is True
    if extra:
        for key, value in extra.items():
            if value not in (None, "", [], {}):
                flags[str(key)] = value
    return flags


def _artifact(
    *,
    source_type: str,
    record_id: Any,
    title: Any,
    project: Any = "",
    status: Any = "",
    kind: Any = "",
    created_at: Any = None,
    updated_at: Any = None,
    archived_at: Any = None,
    counts: dict[str, int] | None = None,
    linked_ids: dict[str, list[Any]] | None = None,
    source_ref_count: int = 0,
    flags: dict[str, Any] | None = None,
    warnings: list[Any] | None = None,
) -> dict[str, Any]:
    return {
        "source_type": source_type,
        "record_id": record_id,
        "title": title,
        "project": project or "",
        "status": status or "",
        "kind": kind or source_type,
        "created_at": created_at,
        "updated_at": updated_at,
        "archived_at": archived_at,
        "counts": counts or {},
        "linked_ids": linked_ids or {},
        "source_ref_count": int(source_ref_count or 0),
        "flags": flags or {},
        "warnings": _as_list(warnings),
        "trusted_for_execution": False,
        "inert_context_only": True,
        "untrusted": True,
    }


def _brief_items() -> list[dict[str, Any]]:
    from hermes_cli.mission_briefs import list_briefs

    items: list[dict[str, Any]] = []
    for brief in _as_list(list_briefs().get("items")):
        if not isinstance(brief, dict):
            continue
        items.append(
            _artifact(
                source_type="mission_brief",
                record_id=brief.get("id"),
                title=brief.get("title"),
                status=brief.get("status"),
                kind="mission_brief",
                created_at=brief.get("created_at"),
                updated_at=brief.get("updated_at"),
                archived_at=brief.get("archived_at"),
                source_ref_count=int(brief.get("reference_count") or 0),
            )
        )
    return items


def _contract_items() -> list[dict[str, Any]]:
    from hermes_cli.mission_control_goal_contracts import list_contracts

    items: list[dict[str, Any]] = []
    for contract in _as_list(list_contracts(include_archived=True).get("items")):
        if not isinstance(contract, dict):
            continue
        linked_brief_id = contract.get("linked_mission_brief_id")
        items.append(
            _artifact(
                source_type="goal_contract",
                record_id=contract.get("id"),
                title=contract.get("title"),
                status=contract.get("status"),
                kind="goal_contract",
                created_at=contract.get("created_at"),
                updated_at=contract.get("updated_at"),
                archived_at=contract.get("archived_at"),
                counts=_counts(
                    success_criteria=contract.get("success_criteria_count"),
                    constraints=contract.get("constraint_count"),
                ),
                linked_ids={"mission_brief_ids": [linked_brief_id]} if linked_brief_id else {},
                source_ref_count=int(contract.get("source_ref_count") or 0),
                flags={"vocabulary_version": contract.get("vocabulary_version")},
            )
        )
    return items


def _packet_items() -> list[dict[str, Any]]:
    from hermes_cli.mission_control import list_packets

    items: list[dict[str, Any]] = []
    for packet in _as_list(list_packets().get("items")):
        if not isinstance(packet, dict):
            continue
        items.append(
            _artifact(
                source_type="mission_packet",
                record_id=packet.get("id"),
                title=packet.get("title"),
                project=packet.get("project"),
                status=packet.get("status"),
                kind=packet.get("kind") or "mission_packet",
                created_at=packet.get("created_at"),
                updated_at=packet.get("updated_at"),
                source_ref_count=int(packet.get("source_ref_count") or 0),
                flags=_flags(dry_run=packet.get("dry_run"), review_required=packet.get("review_required")),
                warnings=_as_list(packet.get("warnings")),
            )
        )
    return items


def _project_room_items() -> list[dict[str, Any]]:
    from hermes_cli.mission_control_project_rooms import list_attachment_metadata, list_rooms

    items: list[dict[str, Any]] = []
    rooms_by_id: dict[str, dict[str, Any]] = {}
    for room in _as_list(list_rooms().get("rooms")):
        if not isinstance(room, dict):
            continue
        room_id = str(room.get("id") or "")
        rooms_by_id[room_id] = room
        items.append(
            _artifact(
                source_type="project_room",
                record_id=room.get("id"),
                title=room.get("title"),
                project=room.get("project_key"),
                status=room.get("status") or "active",
                kind="project_room",
                created_at=room.get("created_at"),
                updated_at=room.get("updated_at"),
                counts=_counts(messages=room.get("message_count"), attachments=room.get("attachment_count")),
            )
        )
    for attachment in _as_list(list_attachment_metadata().get("items")):
        if not isinstance(attachment, dict):
            continue
        room_id = str(attachment.get("room_id") or "")
        room = rooms_by_id.get(room_id, {})
        items.append(
            _artifact(
                source_type="project_room_attachment",
                record_id=attachment.get("id"),
                title=attachment.get("original_filename"),
                project=room.get("project_key") or "",
                status=attachment.get("status") or "active",
                kind="project_room_attachment",
                created_at=attachment.get("created_at"),
                updated_at=attachment.get("updated_at") or attachment.get("created_at"),
                counts=_counts(bytes=attachment.get("size_bytes")),
                linked_ids={"room_ids": [room_id]} if room_id else {},
                flags={"mime_type": attachment.get("mime_type")},
            )
        )
    return items


def list_artifacts(*, kind: str | None = None, status: str | None = None) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    warnings: list[str] = []
    for loader in (_brief_items, _contract_items, _packet_items, _project_room_items):
        try:
            items.extend(loader())
        except Exception as exc:
            warnings.append(f"{loader.__name__} unavailable: {exc}")

    if kind:
        items = [item for item in items if item.get("kind") == kind or item.get("source_type") == kind]
    if status:
        items = [item for item in items if item.get("status") == status]
    items.sort(key=lambda item: str(item.get("updated_at") or item.get("created_at") or ""), reverse=True)
    return {
        "generated_at": _now_iso(),
        "source": "mission_control_artifacts",
        "items": items,
        "warnings": warnings,
    }

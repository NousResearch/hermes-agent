"""Structured delivery and conservative reply intents for staged writes."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence


WRITE_APPROVAL_METADATA_KEY = "write_approval"
WRITE_APPROVAL_REPLY_KEY = "write_approval_reply"
_SUBSYSTEMS = ("memory", "skills")
_PENDING_ID_RE = re.compile(r"^[0-9a-f]{8}$", re.IGNORECASE)


def normalize_surface(surface: Any) -> Optional[dict]:
    """Return a payload-safe surface containing subsystem names and IDs only."""
    if not isinstance(surface, Mapping):
        return None
    raw_items = surface.get("items")
    if not isinstance(raw_items, Mapping):
        return None

    requested = surface.get("subsystems")
    if isinstance(requested, str):
        requested = [requested]
    if not isinstance(requested, (list, tuple)):
        requested = list(raw_items)

    subsystems: list[str] = []
    items: dict[str, list[str]] = {}
    for raw_subsystem in requested:
        subsystem = str(raw_subsystem or "").strip().lower()
        if subsystem not in _SUBSYSTEMS or subsystem in subsystems:
            continue
        raw_ids = raw_items.get(subsystem)
        if not isinstance(raw_ids, (list, tuple)):
            continue
        pending_ids = [
            str(value).lower()
            for value in raw_ids
            if _PENDING_ID_RE.fullmatch(str(value or "").strip())
        ]
        if not pending_ids:
            continue
        subsystems.append(subsystem)
        items[subsystem] = list(dict.fromkeys(pending_ids))

    if not subsystems:
        return None
    normalized = {"subsystems": subsystems, "items": items}
    raw_scope = surface.get("scope")
    if isinstance(raw_scope, Mapping):
        scope = {
            "session_key": str(raw_scope.get("session_key") or ""),
            "profile": str(raw_scope.get("profile") or "default"),
            "chat_id": str(raw_scope.get("chat_id") or ""),
            "thread_id": str(raw_scope.get("thread_id") or ""),
        }
        generation = raw_scope.get("run_generation")
        if isinstance(generation, int):
            scope["run_generation"] = generation
        normalized["scope"] = scope
    return normalized


@dataclass(frozen=True)
class ApprovalCard:
    text: str
    surface: dict


def _target_label(event: Mapping[str, Any]) -> str:
    if event.get("subsystem") == "memory":
        return "USER.md" if event.get("target") == "user" else "MEMORY.md"
    name = str(event.get("target") or "skill library")
    return f"skill {name}"


def _card_lines_for_event(event: Mapping[str, Any]) -> list:
    """Render one staged record: a scannable header, then the preview body."""
    operation = str(event.get("operation") or "change").strip().lower() or "change"
    header = f"**{_target_label(event)}** · {operation} · `{event['pending_id']}`"
    preview = str(event.get("preview") or "").strip() or "(no preview)"
    return [header, *preview.split("\n")]


def approval_card_for_events(events: Sequence[Mapping[str, Any]]) -> Optional[ApprovalCard]:
    """Render one deterministic card from the exact records emitted by a turn."""
    valid = []
    seen = set()
    for event in events:
        if not isinstance(event, Mapping):
            continue
        pending_id = str(event.get("pending_id") or "").lower()
        subsystem = str(event.get("subsystem") or "").lower()
        if (
            subsystem not in _SUBSYSTEMS
            or not _PENDING_ID_RE.fullmatch(pending_id)
            or (subsystem, pending_id) in seen
        ):
            continue
        seen.add((subsystem, pending_id))
        valid.append(dict(event))
    if not valid:
        return None

    items: dict[str, list[str]] = {}
    subsystems: list[str] = []
    for event in valid:
        subsystem = event["subsystem"]
        if subsystem not in subsystems:
            subsystems.append(subsystem)
        items.setdefault(subsystem, []).append(event["pending_id"])

    first = valid[0]
    scope = {
        "session_key": str(first.get("session_key") or ""),
        "run_generation": first.get("run_generation"),
        "profile": str(first.get("profile") or "default"),
    }
    surface = normalize_surface(
        {"subsystems": subsystems, "items": items, "scope": scope}
    )
    if surface is None:
        return None

    lines = ["💾 **Memory proposal**" if subsystems == ["memory"] else "💾 **Write proposal**"]
    for event in valid:
        lines.append("")
        lines.extend(_card_lines_for_event(event))
    return ApprovalCard(text="\n".join(lines), surface=surface)


async def deliver_staged_write_cards(
    *,
    adapter: Any,
    source: Any,
    reply_to_message_id: Optional[str],
    events: Sequence[Mapping[str, Any]],
) -> bool:
    """Send one turn-owned approval card per exact record in the source topic."""
    if adapter is None:
        return False
    from gateway.platforms.base import (
        _mark_notify_metadata,
        _thread_metadata_for_source,
    )

    sent_any = False
    all_succeeded = True
    for event in events:
        card = approval_card_for_events([event])
        if card is None:
            continue
        sent_any = True
        surface = dict(card.surface)
        scope = dict(surface.get("scope") or {})
        scope.update(
            {
                "chat_id": str(getattr(source, "chat_id", "") or ""),
                "thread_id": str(getattr(source, "thread_id", "") or ""),
                "profile": str(
                    getattr(source, "profile", None)
                    or scope.get("profile")
                    or "default"
                ),
            }
        )
        surface["scope"] = scope
        metadata = dict(
            _thread_metadata_for_source(source, reply_to_message_id) or {}
        )
        metadata[WRITE_APPROVAL_METADATA_KEY] = surface
        metadata = _mark_notify_metadata(metadata)
        result = await adapter.send(
            str(getattr(source, "chat_id", "")),
            card.text,
            reply_to=reply_to_message_id,
            metadata=metadata,
        )
        all_succeeded = all_succeeded and bool(getattr(result, "success", False))
    return sent_any and all_succeeded


def build_pending_surface(subsystems: Iterable[str]) -> Optional[dict]:
    """Snapshot pending IDs without exposing summaries or staged payloads."""
    from tools import write_approval as wa

    items: dict[str, list[str]] = {}
    ordered: list[str] = []
    for raw_subsystem in subsystems:
        subsystem = str(raw_subsystem or "").strip().lower()
        if subsystem not in _SUBSYSTEMS or subsystem in ordered:
            continue
        ids = [
            str(record.get("id") or "").lower()
            for record in wa.list_pending(subsystem)
            if isinstance(record, dict)
            and _PENDING_ID_RE.fullmatch(str(record.get("id") or "").strip())
        ]
        if ids:
            ordered.append(subsystem)
            items[subsystem] = list(dict.fromkeys(ids))
    return normalize_surface({"subsystems": ordered, "items": items})


class WriteApprovalReply(str):
    """Text response carrying internal platform-delivery metadata."""

    delivery_metadata: dict

    def __new__(cls, text: str, surface: Any = None):
        instance = super().__new__(cls, text)
        normalized = normalize_surface(surface)
        instance.delivery_metadata = (
            {WRITE_APPROVAL_METADATA_KEY: normalized} if normalized else {}
        )
        return instance


def merge_response_delivery_metadata(
    metadata: Optional[Mapping[str, Any]], response: Any
) -> Optional[dict]:
    """Merge trusted metadata carried by a structured gateway reply."""
    merged = dict(metadata or {})
    extra = getattr(response, "delivery_metadata", None)
    if isinstance(extra, Mapping):
        for key, value in extra.items():
            if isinstance(key, str):
                merged[key] = value
    return merged or None


def _normalize_intent(text: str) -> str:
    normalized = " ".join(str(text or "").strip().casefold().split())
    return normalized.rstrip(".!?。！？")


def _single_subsystem(surface: dict) -> Optional[str]:
    subsystems = surface["subsystems"]
    return subsystems[0] if len(subsystems) == 1 else None


def command_for_reply_intent(text: str, raw_surface: Any) -> Optional[str]:
    """Map an exact intent to an existing slash command.

    The caller must supply a surface recorded from a Hermes-owned outbound
    approval message. Ambiguous or conversational language fails open to the
    normal agent path; this function never infers intent from message text
    alone.
    """
    surface = normalize_surface(raw_surface)
    if surface is None:
        return None
    intent = _normalize_intent(text)
    if not intent:
        return None

    single = _single_subsystem(surface)
    approve_words = {"approve", "одобри", "одобрить"}
    reject_words = {"reject", "deny", "отклони", "отклонить"}
    parts = intent.split()

    if len(parts) == 2 and parts[0] in approve_words | reject_words:
        action = "approve" if parts[0] in approve_words else "reject"
        target_token = parts[1]
        if _PENDING_ID_RE.fullmatch(target_token):
            owners = [
                name for name, ids in surface["items"].items() if target_token in ids
            ]
            if len(owners) == 1:
                return f"/{owners[0]} {action} {target_token}"
        return None

    if intent in {"show pending", "покажи pending", "покажи ожидающие"}:
        return f"/{single} pending" if single else None
    if intent in {
        "show diff",
        "show skill diff",
        "покажи diff",
        "покажи разницу",
    }:
        skill_ids = surface["items"].get("skills", [])
        if len(skill_ids) == 1:
            return f"/skills diff {skill_ids[0]}"
    return None


def command_for_callback_data(data: str) -> Optional[str]:
    """Decode a compact Telegram ``wa:*`` callback into a slash command."""
    match = re.fullmatch(
        r"wa:(m|s):(a|r|p|d|s):(all|[0-9a-f]{8})",
        str(data or ""),
        re.IGNORECASE,
    )
    if match is None:
        return None
    subsystem = "memory" if match.group(1).lower() == "m" else "skills"
    action_code = match.group(2).lower()
    target = match.group(3).lower()
    if action_code == "p":
        return f"/{subsystem} pending"
    if action_code == "d":
        if subsystem != "skills" or target == "all":
            return None
        return f"/skills diff {target}"
    if target == "all":
        return None
    if action_code == "s":
        return f"/{subsystem} approve-session {target}"
    action = "approve" if action_code == "a" else "reject"
    return f"/{subsystem} {action} {target}"


__all__ = [
    "WRITE_APPROVAL_METADATA_KEY",
    "WRITE_APPROVAL_REPLY_KEY",
    "WriteApprovalReply",
    "build_pending_surface",
    "command_for_callback_data",
    "command_for_reply_intent",
    "merge_response_delivery_metadata",
    "normalize_surface",
]

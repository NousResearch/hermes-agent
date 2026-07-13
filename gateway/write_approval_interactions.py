"""Structured delivery and conservative reply intents for staged writes."""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping, Optional


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
    return {"subsystems": subsystems, "items": items}


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


def _explicit_subsystem(token: str) -> Optional[str]:
    token = token.casefold()
    if token in {"memory", "память", "памяти"}:
        return "memory"
    if token in {"skills", "skill", "навыки", "навыков"}:
        return "skills"
    return None


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
    all_words = {"all", "всё", "все"}
    approve_words = {"approve", "одобри", "одобрить"}
    reject_words = {"reject", "deny", "отклони", "отклонить"}
    parts = intent.split()

    if len(parts) == 2 and parts[0] in approve_words and parts[1] in all_words:
        return f"/{single} approve all" if single else None
    if len(parts) == 2 and parts[0] in reject_words and parts[1] in all_words:
        return f"/{single} reject all" if single else None

    if len(parts) in {2, 3} and parts[0] in approve_words | reject_words:
        action = "approve" if parts[0] in approve_words else "reject"
        target_token = parts[-1]
        if len(parts) == 3 and parts[1] not in all_words:
            return None
        subsystem = _explicit_subsystem(target_token)
        if subsystem:
            if subsystem not in surface["subsystems"]:
                return None
            return f"/{subsystem} {action} all"
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
        r"wa:(m|s):(a|r|p|d):(all|[0-9a-f]{8})",
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

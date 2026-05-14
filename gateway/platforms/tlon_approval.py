"""Approval and authorization helpers for Tlon."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


APPROVAL_TTL_SECONDS = 48 * 60 * 60


@dataclass
class PendingApproval:
    id: str
    type: str
    requesting_ship: str
    channel_nest: Optional[str] = None
    group_flag: Optional[str] = None
    group_title: Optional[str] = None
    message_preview: Optional[str] = None
    original_message: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    notification_message_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "requestingShip": self.requesting_ship,
            "channelNest": self.channel_nest,
            "groupFlag": self.group_flag,
            "groupTitle": self.group_title,
            "messagePreview": self.message_preview,
            "originalMessage": self.original_message or None,
            "timestamp": int(self.timestamp * 1000),
            "notificationMessageId": self.notification_message_id,
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "PendingApproval":
        timestamp = raw.get("timestamp")
        if isinstance(timestamp, (int, float)) and timestamp > 10_000_000_000:
            timestamp = timestamp / 1000
        return cls(
            id=str(raw.get("id") or _generate_id(str(raw.get("type") or "dm"), [])),
            type=str(raw.get("type") or "dm"),
            requesting_ship=str(raw.get("requestingShip") or raw.get("requesting_ship") or ""),
            channel_nest=raw.get("channelNest") or raw.get("channel_nest"),
            group_flag=raw.get("groupFlag") or raw.get("group_flag"),
            group_title=raw.get("groupTitle") or raw.get("group_title"),
            message_preview=raw.get("messagePreview") or raw.get("message_preview"),
            original_message=raw.get("originalMessage") or raw.get("original_message") or {},
            timestamp=float(timestamp or time.time()),
            notification_message_id=raw.get("notificationMessageId"),
        )


def create_pending_approval(
    *,
    approval_type: str,
    requesting_ship: str,
    existing_ids: List[str],
    channel_nest: Optional[str] = None,
    group_flag: Optional[str] = None,
    group_title: Optional[str] = None,
    message_preview: Optional[str] = None,
    original_message: Optional[Dict[str, Any]] = None,
) -> PendingApproval:
    return PendingApproval(
        id=_generate_id(approval_type, existing_ids),
        type=approval_type,
        requesting_ship=requesting_ship,
        channel_nest=channel_nest,
        group_flag=group_flag,
        group_title=group_title,
        message_preview=message_preview,
        original_message=original_message or {},
    )


def prune_expired(approvals: List[PendingApproval]) -> List[PendingApproval]:
    now = time.time()
    return [approval for approval in approvals if now - approval.timestamp <= APPROVAL_TTL_SECONDS]


def find_pending_approval(
    approvals: List[PendingApproval],
    approval_id: Optional[str] = None,
) -> Optional[PendingApproval]:
    active = prune_expired(approvals)
    if approval_id:
        exact = next((approval for approval in active if approval.id == approval_id), None)
        if exact:
            return exact
        matches = [approval for approval in active if approval.id.startswith(approval_id)]
        return matches[0] if len(matches) == 1 else None
    return active[-1] if active else None


def has_duplicate_pending(
    approvals: List[PendingApproval],
    *,
    approval_type: str,
    requesting_ship: str,
    channel_nest: Optional[str] = None,
    group_flag: Optional[str] = None,
) -> bool:
    for approval in prune_expired(approvals):
        if approval.type != approval_type or approval.requesting_ship != requesting_ship:
            continue
        if approval_type == "channel" and approval.channel_nest != channel_nest:
            continue
        if approval_type == "group" and approval.group_flag != group_flag:
            continue
        return True
    return False


def format_approval_request(approval: PendingApproval) -> str:
    preview = f'\n"{_truncate(approval.message_preview or "", 140)}"' if approval.message_preview else ""
    if approval.type == "dm":
        subject = f"DM request from {approval.requesting_ship}"
    elif approval.type == "channel":
        subject = (
            f"{approval.requesting_ship} mentioned Hermes in "
            f"{approval.channel_nest or 'a channel'}"
        )
    else:
        group = approval.group_title or approval.group_flag or "a group"
        subject = f"Group invite from {approval.requesting_ship} to join {group}"

    return "\n".join(
        [
            subject,
            preview,
            "",
            f"Pending approval id: {approval.id}",
            "",
            "Use one of:",
            f"/allow {approval.id}",
            f"/reject {approval.id}",
            f"/ban {approval.id}",
        ]
    )


def format_pending_list(approvals: List[PendingApproval]) -> str:
    active = prune_expired(approvals)
    if not active:
        return "No pending Tlon approvals."
    lines = [f"Pending Tlon approvals ({len(active)}):"]
    for approval in active:
        where = approval.channel_nest or approval.group_flag or "DM"
        preview = f" - {_truncate(approval.message_preview, 80)}" if approval.message_preview else ""
        lines.append(f"{approval.id}: {approval.type} from {approval.requesting_ship} in {where}{preview}")
    return "\n".join(lines)


def format_blocked_list(blocked: List[str]) -> str:
    if not blocked:
        return "No Tlon ships are blocked."
    return "Blocked Tlon ships:\n" + "\n".join(f"- {ship}" for ship in sorted(blocked))


def format_confirmation(approval: PendingApproval, action: str) -> str:
    ship = approval.requesting_ship
    if action == "approve":
        if approval.type == "dm":
            return f"Approved DM access for {ship}."
        if approval.type == "channel":
            return f"Approved {ship} in {approval.channel_nest}."
        return f"Approved group invite from {ship}."
    if action == "block":
        return f"Blocked {ship}."
    return f"Rejected request from {ship}."


def _generate_id(approval_type: str, existing_ids: List[str]) -> str:
    prefix = (approval_type or "x")[0]
    for _ in range(10):
        approval_id = f"{prefix}{uuid.uuid4().hex[:4]}"
        if approval_id not in existing_ids:
            return approval_id
    return f"{prefix}{uuid.uuid4().hex[:8]}"


def _truncate(text: Optional[str], max_len: int) -> str:
    text = text or ""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."

"""Role-aware, deduplicated Ship's Crew notification policy."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class NotificationPolicy:
    terminal_events: frozenset[str] = frozenset({"completed", "blocked", "contract_rejected", "review_requested"})
    roles: frozenset[str] = frozenset({"captain", "navigator", "engineer", "pirate"})
    max_chars: int = 1_500
    dedupe_window_seconds: int = 3_600
    suppress_without_evidence: bool = True


def notification_key(*, task_id: str, event_kind: str, event_id: int | str) -> str:
    raw = f"{task_id}|{event_kind}|{event_id}".encode()
    return hashlib.sha256(raw).hexdigest()[:24]


def should_notify(event: Mapping[str, Any], policy: NotificationPolicy = NotificationPolicy()) -> bool:
    kind = str(event.get("kind", ""))
    role = str(event.get("role", ""))
    if kind not in policy.terminal_events or role not in policy.roles:
        return False
    if policy.suppress_without_evidence and kind in {"completed", "review_requested"}:
        evidence = event.get("evidence_sha256") or event.get("evidence_ref")
        if not evidence:
            return False
    return True


def render_notification(event: Mapping[str, Any], policy: NotificationPolicy = NotificationPolicy()) -> str:
    if not should_notify(event, policy):
        return ""
    body = {
        "task_id": event.get("task_id"),
        "event": event.get("kind"),
        "role": event.get("role"),
        "outcome": event.get("outcome"),
        "evidence": event.get("evidence_sha256") or event.get("evidence_ref"),
        "summary": event.get("summary", ""),
    }
    text = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return text if len(text) <= policy.max_chars else text[: policy.max_chars] + "…"

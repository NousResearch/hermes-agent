"""Shared Torben attention-contract helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from torben_open_loops import ACTIVE_STATES, load_loops

BRIEF_SECTIONS = [
    "The Day",
    "The Decisions",
    "The People",
    "The Meetings",
    "The World",
    "The Move",
    "Pending Decisions",
]


def load_pending_decisions(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8") or "[]")
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def load_pattern_proposals(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8") or "{}")
    if not isinstance(payload, dict):
        return []
    try:
        from torben_pattern_miner import proposals_for_weekly_reset
    except Exception:
        return []
    return proposals_for_weekly_reset(payload)


def summarize_open_loops(path: Path, *, limit: int = 10) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "active": [], "counts": {}}
    rows = load_loops(path)
    active = [row for row in rows if row.state in ACTIVE_STATES]
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.state or "missing"] = counts.get(row.state or "missing", 0) + 1
    return {
        "path": str(path),
        "active": [row.to_dict() for row in active[:limit]],
        "counts": counts,
    }


def build_brief_attention_section(
    *,
    pending_decisions: list[dict[str, Any]],
    open_loops: dict[str, Any],
    pattern_proposals: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "sections": BRIEF_SECTIONS,
        "pending_decisions": pending_decisions + list(pattern_proposals or []),
        "open_loops": open_loops,
    }


def meeting_key(alert: dict[str, Any]) -> str:
    return str(
        alert.get("alert_key")
        or alert.get("event_id")
        or "|".join(str(alert.get(key) or "") for key in ("title", "summary", "start_at"))
    )


def dedupe_meeting_alerts(alerts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for alert in alerts:
        key = meeting_key(alert)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(alert)
    return deduped


def should_send_interrupt(signal: dict[str, Any]) -> bool:
    return bool(signal.get("actionable") and signal.get("time_sensitive") and not signal.get("duplicate"))


def silence_on_success(task: str) -> dict[str, Any]:
    return {"task": task, "wakeAgent": False, "reason": "success_no_actionable_payload"}

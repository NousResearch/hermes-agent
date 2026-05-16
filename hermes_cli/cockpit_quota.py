"""Read-only Cockpit quota/session-reset recommendation helpers.

This module intentionally reads only local Hermes session metadata. It does not
reset sessions, mutate provider/model settings, or send messages.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from hermes_state import get_hermes_home

COCKPIT_QUOTA_SCHEMA_VERSION = 1

QUOTA_THRESHOLDS: tuple[tuple[int, str, str], ...] = (
    (130_000, "urgent", "Quota note: this thread is very expensive now. Start a fresh session for the next task; I can carry forward a compact handoff summary."),
    (100_000, "strong", "Quota note: this thread is now expensive to continue. I recommend starting a fresh session before new work; I can carry forward a compact summary."),
    (70_000, "recommend", "Quota note: this thread is carrying a large prompt history. For the next topic, start a fresh session and I will carry forward only the summary."),
    (40_000, "heads_up", "Quota note: this thread is getting large. If we are changing topics, a fresh session will be cheaper."),
)


def _safe_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def build_quota_recommendation(prompt_tokens: int) -> Optional[dict[str, Any]]:
    """Return a quota recommendation for prompt_tokens, or None below threshold."""
    tokens = _safe_int(prompt_tokens)
    for threshold, level, message in QUOTA_THRESHOLDS:
        if tokens >= threshold:
            return {
                "level": level,
                "threshold": threshold,
                "prompt_tokens": tokens,
                "dedupe_key": f"quota-session-reset-{level}-{threshold}",
                "message": message,
                "next_step": "Start a fresh session/thread for the next topic; ask Biff for a compact handoff summary first if needed.",
            }
    return None


def _parse_updated_at(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def _load_sessions(hermes_home: Path) -> dict[str, Any]:
    sessions_file = hermes_home / "sessions" / "sessions.json"
    if not sessions_file.exists():
        return {}
    try:
        data = json.loads(sessions_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _session_prompt_tokens(entry: dict[str, Any]) -> int:
    return _safe_int(entry.get("last_prompt_tokens"))


def get_quota_recommendation_payload(
    *,
    hermes_home: Path | None = None,
    recent_days: int = 2,
) -> dict[str, Any]:
    """Return read-only Cockpit quota/session-reset recommendation payload.

    The payload uses the highest recent ``last_prompt_tokens`` session because
    provider token accounting may be unavailable while prompt size still exposes
    runaway context cost.
    """
    home = Path(hermes_home) if hermes_home is not None else get_hermes_home()
    sessions = _load_sessions(home)
    now = datetime.now()
    cutoff = now - timedelta(days=max(1, int(recent_days or 1)))

    considered: list[dict[str, Any]] = []
    for _, raw in sessions.items():
        if not isinstance(raw, dict):
            continue
        updated_at = _parse_updated_at(raw.get("updated_at"))
        if updated_at and updated_at < cutoff:
            continue
        prompt_tokens = _session_prompt_tokens(raw)
        considered.append({
            "session_id": raw.get("session_id"),
            "platform": raw.get("platform"),
            "display_name": raw.get("display_name"),
            "updated_at": raw.get("updated_at"),
            "last_prompt_tokens": prompt_tokens,
        })

    considered.sort(key=lambda item: item.get("last_prompt_tokens") or 0, reverse=True)
    top = considered[0] if considered else None
    max_prompt_tokens = _safe_int(top.get("last_prompt_tokens") if top else 0)
    recommendation = build_quota_recommendation(max_prompt_tokens)
    if recommendation and top:
        recommendation.update({
            "session_id": top.get("session_id"),
            "platform": top.get("platform"),
            "display_name": top.get("display_name"),
            "updated_at": top.get("updated_at"),
        })

    return {
        "schema_version": COCKPIT_QUOTA_SCHEMA_VERSION,
        "read_only": True,
        "actions_enabled": False,
        "auto_reset_enabled": False,
        "thresholds": [
            {"prompt_tokens": threshold, "level": level}
            for threshold, level, _ in sorted(QUOTA_THRESHOLDS, key=lambda row: row[0])
        ],
        "recent_days": max(1, int(recent_days or 1)),
        "sessions_considered": len(considered),
        "max_prompt_tokens": max_prompt_tokens,
        "recommendation": recommendation,
    }

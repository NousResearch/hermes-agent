"""Profile-aware envelope tracking for local subprocess wrapper foundations."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home

ENVELOPE_STATE_FILENAME = "envelope_tracking.json"
WINDOW_DURATION = timedelta(hours=5)

_DEFAULT_STATE: dict[str, dict[str, Any]] = {
    "anthropic_max": {
        "envelope_total_messages_per_5h": 225,
        "envelope_allocation_hermes_pct": 85,
        "envelope_messages_used_5h": 0,
        "window_start_iso": None,
        "last_invocation_iso": None,
        "halt_flag_active": False,
    },
    "chatgpt_pro": {
        "envelope_total_messages_per_5h": 200,
        "envelope_allocation_hermes_pct": 85,
        "envelope_messages_used_5h": 0,
        "window_start_iso": None,
        "last_invocation_iso": None,
        "halt_flag_active": False,
    },
}


@dataclass(frozen=True)
class EnvelopeDecision:
    allowed: bool
    reason: str
    used: int
    cap: int
    available: int


def envelope_state_path() -> Path:
    return get_hermes_home() / "state" / ENVELOPE_STATE_FILENAME


def default_envelope_state() -> dict[str, dict[str, Any]]:
    return copy.deepcopy(_DEFAULT_STATE)


def allocation_cap(record: dict[str, Any]) -> int:
    total = int(record.get("envelope_total_messages_per_5h", 0))
    pct = int(record.get("envelope_allocation_hermes_pct", 0))
    return (total * pct) // 100


def load_envelope_state(path: Path | None = None) -> dict[str, dict[str, Any]]:
    state_path = path or envelope_state_path()
    if not state_path.exists():
        return default_envelope_state()
    with state_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    state = default_envelope_state()
    for key, defaults in state.items():
        if isinstance(data.get(key), dict):
            defaults.update(data[key])
    return state


def save_envelope_state(state: dict[str, dict[str, Any]], path: Path | None = None) -> None:
    state_path = path or envelope_state_path()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with state_path.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value)


def increment_usage(
    state: dict[str, dict[str, Any]],
    envelope_name: str,
    *,
    now: datetime | None = None,
) -> dict[str, dict[str, Any]]:
    current = now or datetime.now(timezone.utc)
    updated = copy.deepcopy(state)
    record = updated[envelope_name]
    window_start = _parse_iso(record.get("window_start_iso"))
    if window_start is None or current - window_start >= WINDOW_DURATION:
        record["window_start_iso"] = current.isoformat()
        record["envelope_messages_used_5h"] = 0
    record["envelope_messages_used_5h"] = int(record.get("envelope_messages_used_5h", 0)) + 1
    record["last_invocation_iso"] = current.isoformat()
    return updated


def check_budget(
    state: dict[str, dict[str, Any]],
    envelope_name: str,
    *,
    priority: bool = False,
) -> EnvelopeDecision:
    record = state[envelope_name]
    used = int(record.get("envelope_messages_used_5h", 0))
    cap = allocation_cap(record)
    available = max(cap - used, 0)
    if used >= cap and not priority:
        return EnvelopeDecision(False, "budget_blocked", used, cap, available)
    if used >= cap and priority:
        return EnvelopeDecision(True, "priority_override", used, cap, available)
    return EnvelopeDecision(True, "within_budget", used, cap, available)

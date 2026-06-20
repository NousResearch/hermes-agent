"""Affordance gap logging for post-task reflection.

This module is intentionally small: it gives the agent and CLI a structured
place to record missing capabilities before the full reflection lifecycle is
wired into the core agent loop.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from hermes_constants import get_hermes_home


GAP_LOG_FILE = "affordance_gaps.jsonl"


@dataclass
class AffordanceGap:
    goal: str
    missing_capability: str
    failure_description: str = ""
    available_tools: list[str] = field(default_factory=list)
    session_id: str | None = None
    source: str = "manual"
    failure_mode: str = "missing_affordance"
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "failure_mode": self.failure_mode,
            "goal": self.goal,
            "missing_capability": self.missing_capability,
            "failure_description": self.failure_description,
            "available_tools": self.available_tools,
            "session_id": self.session_id,
            "source": self.source,
        }


def default_gap_log_path() -> Path:
    return get_hermes_home() / GAP_LOG_FILE


def log_affordance_gap(
    *,
    goal: str,
    missing_capability: str,
    failure_description: str = "",
    available_tools: Iterable[str] | None = None,
    session_id: str | None = None,
    source: str = "manual",
    log_path: Path | None = None,
) -> dict[str, Any]:
    """Append one missing-affordance event to the persistent gap log."""
    if not goal or not goal.strip():
        raise ValueError("goal is required")
    if not missing_capability or not missing_capability.strip():
        raise ValueError("missing_capability is required")

    gap = AffordanceGap(
        goal=goal.strip(),
        missing_capability=missing_capability.strip(),
        failure_description=(failure_description or "").strip(),
        available_tools=sorted({tool.strip() for tool in (available_tools or []) if tool and tool.strip()}),
        session_id=session_id,
        source=source or "manual",
    )
    target = log_path or default_gap_log_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as f:
        f.write(json.dumps(gap.to_dict(), ensure_ascii=False) + "\n")
    return gap.to_dict()


def load_affordance_gaps(
    *,
    limit: int = 20,
    log_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Load recent gap events, newest first. Invalid JSONL rows are skipped."""
    target = log_path or default_gap_log_path()
    if not target.exists():
        return []

    rows: list[dict[str, Any]] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            rows.append(parsed)

    if limit <= 0:
        return list(reversed(rows))
    return list(reversed(rows))[:limit]


def format_affordance_gaps(gaps: list[dict[str, Any]]) -> str:
    """Render gap events as compact CLI text."""
    if not gaps:
        return "No affordance gaps logged."

    lines = []
    for idx, gap in enumerate(gaps, start=1):
        timestamp = gap.get("timestamp", "")
        goal = gap.get("goal", "")
        missing = gap.get("missing_capability", "")
        failure = gap.get("failure_description", "")
        session_id = gap.get("session_id")
        tools = gap.get("available_tools") or []
        lines.append(f"{idx}. {missing}")
        if goal:
            lines.append(f"   Goal: {goal}")
        if failure:
            lines.append(f"   Failure: {failure}")
        if tools:
            lines.append(f"   Available tools: {', '.join(tools)}")
        if session_id:
            lines.append(f"   Session: {session_id}")
        if timestamp:
            lines.append(f"   Logged: {timestamp}")
    return "\n".join(lines)

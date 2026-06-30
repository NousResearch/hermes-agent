from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _date_name(ts: str) -> str:
    return ts[:10]


@dataclass(frozen=True)
class ObsidianAdapter:
    vault_path: Path

    def __init__(self, vault_path: str | Path):
        object.__setattr__(self, "vault_path", Path(vault_path))

    def append_daily_note(self, relative_dir: str, entry: str, timestamp: str | None = None) -> Path:
        ts = timestamp or _now_iso()
        target_dir = self.vault_path / relative_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{_date_name(ts)}.md"
        with target.open("a", encoding="utf-8") as handle:
            handle.write(entry.rstrip() + "\n\n")
        return target


def append_audit_log(
    adapter: ObsidianAdapter,
    *,
    trigger_type: str,
    decision: str,
    risk_level: str,
    actions_taken: list[str] | None = None,
    tools_used: list[str] | None = None,
    result_summary: str = "",
    follow_up: str = "",
    whether_user_was_notified: bool = False,
    timestamp: str | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    ts = timestamp or _now_iso()
    lines = [
        f"## {ts}",
        f"- timestamp: {ts}",
        f"- trigger_type: {trigger_type}",
        f"- decision: {decision}",
        f"- risk_level: {risk_level}",
        f"- actions_taken: {', '.join(actions_taken or [])}",
        f"- tools_used: {', '.join(tools_used or [])}",
        f"- result_summary: {result_summary}",
        f"- follow_up: {follow_up}",
        f"- whether_user_was_notified: {str(whether_user_was_notified).lower()}",
    ]
    for key, value in (extra or {}).items():
        lines.append(f"- {key}: {value}")
    return adapter.append_daily_note("System/Agent Runs", "\n".join(lines), timestamp=ts)


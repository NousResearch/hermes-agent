from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path


@dataclass(frozen=True)
class ProactiveItem:
    id: str
    kind: str
    status: str
    summary: str
    next_action: str = ""
    source_message: str = ""
    due_at: str = ""


@dataclass(frozen=True)
class ProgressScan:
    waiting_for_kj: list[ProactiveItem]
    active_items: list[ProactiveItem]
    stuck_or_failed: list[ProactiveItem]


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        normalized = value.strip().replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except ValueError:
        return None


def read_state(vault: Path) -> dict[str, str]:
    path = vault / "System" / "Proactive State" / "heartbeat-state.yaml"
    if not path.exists():
        return {}
    state: dict[str, str] = {}
    try:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            if ":" not in raw_line or raw_line.strip().startswith("#"):
                continue
            key, value = raw_line.split(":", 1)
            state[key.strip()] = value.strip()
    except OSError:
        return {}
    return state


def write_state(vault: Path, state: dict[str, str]) -> Path:
    target_dir = vault / "System" / "Proactive State"
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / "heartbeat-state.yaml"
    lines = [f"{key}: {value}" for key, value in sorted(state.items())]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def should_emit(
    state: dict[str, str],
    key: str,
    *,
    interval_minutes: int,
    now: datetime,
) -> bool:
    last = parse_iso(state.get(key))
    if last is None:
        return True
    return now - last >= timedelta(minutes=interval_minutes)


def mark_emitted(state: dict[str, str], key: str, *, now: datetime) -> dict[str, str]:
    updated = dict(state)
    updated[key] = now.astimezone(timezone.utc).replace(microsecond=0).isoformat()
    return updated


def scan_progress(vault: Path) -> ProgressScan:
    waiting: list[ProactiveItem] = []
    active: list[ProactiveItem] = []
    stuck_or_failed: list[ProactiveItem] = []

    for item in _scan_commitments(vault):
        active.append(item)
        if _is_waiting_for_kj(item):
            waiting.append(item)

    for item in _scan_delegated_tasks(vault):
        active.append(item)
        if item.status.lower() in {"failed", "stuck", "blocked", "timeout"}:
            stuck_or_failed.append(item)

    return ProgressScan(
        waiting_for_kj=waiting,
        active_items=active,
        stuck_or_failed=stuck_or_failed,
    )


def _scan_commitments(vault: Path) -> list[ProactiveItem]:
    folder = vault / "System" / "Commitments"
    if not folder.exists():
        return []
    items: list[ProactiveItem] = []
    for path in sorted(folder.glob("*.md")):
        for record in _yaml_blocks(path):
            status = record.get("status", "").strip() or "open"
            if status.lower() in {"done", "closed", "cancelled", "canceled"}:
                continue
            source = record.get("source_message", "").strip()
            items.append(
                ProactiveItem(
                    id=record.get("id", path.stem).strip(),
                    kind="commitment",
                    status=status,
                    summary=source or record.get("inferred_intent", "open commitment"),
                    next_action=record.get("next_action", "").strip(),
                    source_message=source,
                    due_at=record.get("due_at", "").strip(),
                )
            )
    return items


def _scan_delegated_tasks(vault: Path) -> list[ProactiveItem]:
    folder = vault / "System" / "Delegated Tasks"
    if not folder.exists():
        return []
    items: list[ProactiveItem] = []
    for path in sorted(folder.glob("*.md")):
        for record in _yaml_blocks(path):
            status = record.get("status", "").strip().lower()
            if not status or status in {"done", "completed", "success", "cancelled", "canceled"}:
                continue
            task_id = record.get("task_id") or record.get("id") or path.stem
            summary = record.get("summary") or record.get("objective") or "delegated task in progress"
            items.append(
                ProactiveItem(
                    id=task_id.strip(),
                    kind="delegated_task",
                    status=status,
                    summary=summary.strip(),
                    next_action=record.get("recommended_next_action", "").strip(),
                )
            )
    return items


def _yaml_blocks(path: Path) -> list[dict[str, str]]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    records: list[dict[str, str]] = []
    for match in re.finditer(r"```yaml\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE):
        record: dict[str, str] = {}
        for raw_line in match.group(1).splitlines():
            if ":" not in raw_line:
                continue
            key, value = raw_line.split(":", 1)
            key = key.strip()
            if not key:
                continue
            record[key] = value.strip().strip("'").strip('"')
        if record:
            records.append(record)
    return records


def _is_waiting_for_kj(item: ProactiveItem) -> bool:
    haystack = " ".join(
        [item.status, item.next_action, item.summary, item.source_message]
    ).lower()
    return any(
        marker in haystack
        for marker in (
            "waiting_for_kj",
            "needs_kj_input",
            "awaiting_kj_input",
            "ask kj",
            "provide",
            "waiting for kj",
            "補資料",
            "提供",
            "確認",
        )
    )

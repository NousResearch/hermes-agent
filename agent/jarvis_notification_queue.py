from __future__ import annotations

import fcntl
import hashlib
import json
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TERMINAL_STATUSES = {"sent", "failed", "dead_letter"}
NON_TERMINAL_STATUSES = {"pending", "retry_wait"}
ALLOWED_STATUSES = TERMINAL_STATUSES | NON_TERMINAL_STATUSES


@contextmanager
def _locked_file(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.seek(0)
            yield handle
        finally:
            handle.flush()
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _parse_jsonl_lines(lines: list[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def make_notification_id(now: datetime | None = None, seed: str = "") -> str:
    dt = now or datetime.now(timezone.utc)
    stamp = dt.strftime("%Y%m%d_%H%M%S")
    digest = hashlib.sha256(f"{dt.isoformat()}:{seed}".encode("utf-8")).hexdigest()[:12]
    return f"jn_{stamp}_{digest}"


def load_notifications(path: Path) -> list[dict[str, Any]]:
    try:
        return _parse_jsonl_lines(path.read_text(encoding="utf-8").splitlines())
    except FileNotFoundError:
        return []


def latest_notifications(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    for item in items:
        key = item.get("handoff_id") or item.get("notification_id")
        if isinstance(key, str) and key:
            by_key[key] = item
    return list(by_key.values())


def append_notification(path: Path, notification: dict[str, Any]) -> dict[str, Any]:
    record = dict(notification)
    if record.get("status") not in ALLOWED_STATUSES:
        raise ValueError(f"invalid notification status: {record.get('status')}")
    with _locked_file(path) as handle:
        handle.seek(0, 2)
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return record


def mark_notification(path: Path, notification_id: str, status: str, **fields: Any) -> dict[str, Any]:
    if status not in ALLOWED_STATUSES:
        raise ValueError(f"invalid notification status: {status}")
    current = None
    for item in reversed(load_notifications(path)):
        if item.get("notification_id") == notification_id:
            current = item
            break
    if current is None:
        raise KeyError(notification_id)
    marked = dict(current)
    marked.update(fields)
    marked["status"] = status
    marked["updated_at"] = datetime.now(timezone.utc).isoformat()
    return append_notification(path, marked)


def pending_notifications(path: Path) -> list[dict[str, Any]]:
    return [item for item in latest_notifications(load_notifications(path)) if item.get("status") == "pending"]

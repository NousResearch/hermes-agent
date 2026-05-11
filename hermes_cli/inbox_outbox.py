"""Local-first inbox/outbox queue primitives.

This module keeps task intake lightweight and durable using only the local
filesystem under ``{HERMES_HOME}/inbox`` and ``{HERMES_HOME}/outbox``.

The queue layout is intentionally simple:

- inbox/pending/   new tasks waiting to be claimed
- inbox/claimed/   tasks currently in progress
- outbox/completed/terminal records for successful work
- outbox/failed/   terminal records for failures
- outbox/archived/ terminal records retained for history only

Each transition writes a new JSON file whose name includes both the original
creation timestamp and the transition timestamp so the filename itself carries
an audit trail. The JSON payload also retains a ``history`` list of state
changes.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home

INBOX_DIRNAME = "inbox"
OUTBOX_DIRNAME = "outbox"
PENDING_DIRNAME = "pending"
CLAIMED_DIRNAME = "claimed"
COMPLETED_DIRNAME = "completed"
FAILED_DIRNAME = "failed"
ARCHIVED_DIRNAME = "archived"

PENDING_STATE = "pending"
CLAIMED_STATE = "claimed"
COMPLETED_STATE = "completed"
FAILED_STATE = "failed"
ARCHIVED_STATE = "archived"

_TERMINAL_STATES = {COMPLETED_STATE, FAILED_STATE, ARCHIVED_STATE}


class QueueError(RuntimeError):
    """Base exception for inbox/outbox queue operations."""


class TaskNotFoundError(QueueError):
    """Raised when a task identifier cannot be resolved."""


class TaskStateError(QueueError):
    """Raised when a task transition is invalid."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _now_iso() -> str:
    return _now().isoformat().replace("+00:00", "Z")


def _stamp(dt: datetime | None = None) -> str:
    dt = dt or _now()
    return dt.strftime("%Y%m%dT%H%M%SZ")


def _slugify(value: Any, *, fallback: str = "task", max_length: int = 48) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    if not text:
        text = fallback
    return text[:max_length].strip("-") or fallback


def _secure_dir(path: Path) -> None:
    try:
        os.chmod(path, 0o700)
    except (OSError, NotImplementedError):
        pass


def _secure_file(path: Path) -> None:
    try:
        if path.exists():
            os.chmod(path, 0o600)
    except (OSError, NotImplementedError):
        pass


def _queue_root() -> Path:
    return get_hermes_home()


def _inbox_root() -> Path:
    return _queue_root() / INBOX_DIRNAME


def _outbox_root() -> Path:
    return _queue_root() / OUTBOX_DIRNAME


def _state_dir(state: str) -> Path:
    state = state.lower().strip()
    if state == PENDING_STATE:
        return _inbox_root() / PENDING_DIRNAME
    if state == CLAIMED_STATE:
        return _inbox_root() / CLAIMED_DIRNAME
    if state == COMPLETED_STATE:
        return _outbox_root() / COMPLETED_DIRNAME
    if state == FAILED_STATE:
        return _outbox_root() / FAILED_DIRNAME
    if state == ARCHIVED_STATE:
        return _outbox_root() / ARCHIVED_DIRNAME
    raise ValueError(f"Unknown queue state: {state}")


def ensure_queue_dirs() -> dict[str, Path]:
    """Create the local queue directories if they do not already exist."""

    inbox = _inbox_root()
    outbox = _outbox_root()
    dirs = {
        "root": _queue_root(),
        "inbox": inbox,
        "outbox": outbox,
        "pending": inbox / PENDING_DIRNAME,
        "claimed": inbox / CLAIMED_DIRNAME,
        "completed": outbox / COMPLETED_DIRNAME,
        "failed": outbox / FAILED_DIRNAME,
        "archived": outbox / ARCHIVED_DIRNAME,
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
        _secure_dir(path)
    return dirs


def _filename_for(record: dict[str, Any], *, state: str, transition_at: str) -> str:
    created_at = _stamp(_parse_iso(record["created_at"])) if record.get("created_at") else _stamp()
    task_id = record["task_id"]
    slug_source = record.get("title") or record.get("summary") or record.get("metadata", {}).get("subject")
    slug = _slugify(slug_source)
    return f"{created_at}_{task_id}_{slug}_{state}_{transition_at}.json"


def _parse_iso(value: str) -> datetime:
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _secure_dir(path.parent)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=str(path.parent), delete=False) as tmp:
        json.dump(payload, tmp, ensure_ascii=False, indent=2, sort_keys=True)
        tmp.write("\n")
        temp_path = Path(tmp.name)
    os.replace(temp_path, path)
    _secure_file(path)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise QueueError(f"Queue file must contain a JSON object: {path}")
    return data


def _path_for_identifier(identifier: str | Path) -> Path:
    candidate = Path(identifier)
    if candidate.exists():
        return candidate

    task_id = str(identifier)
    search_roots = [
        _inbox_root() / PENDING_DIRNAME,
        _inbox_root() / CLAIMED_DIRNAME,
        _outbox_root() / COMPLETED_DIRNAME,
        _outbox_root() / FAILED_DIRNAME,
        _outbox_root() / ARCHIVED_DIRNAME,
    ]
    for root in search_roots:
        if not root.exists():
            continue
        for path in sorted(root.glob("*.json")):
            try:
                record = _read_json(path)
            except Exception:
                continue
            if record.get("task_id") == task_id:
                return path
    raise TaskNotFoundError(f"Task not found: {identifier}")


def _record_transition(
    record: dict[str, Any],
    *,
    state: str,
    at: str,
    path: Path,
    from_path: Path | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    updated = dict(record)
    updated["state"] = state
    updated["updated_at"] = at
    state_key = f"{state}_at"
    updated[state_key] = at
    updated["current_path"] = str(path)
    updated["file_name"] = path.name
    history = list(updated.get("history") or [])
    history.append(
        {
            "state": state,
            "at": at,
            "path": str(path),
            "from_path": str(from_path) if from_path else None,
            "note": note,
        }
    )
    updated["history"] = history
    return updated


def create_inbox_task(
    task: Any,
    *,
    title: str | None = None,
    task_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a pending inbox task as a durable JSON file.

    The task payload is stored under ``payload`` and can be any JSON-serializable
    object. ``title`` is optional and is used for filename readability.
    """

    ensure_queue_dirs()
    created_at = _now_iso()
    task_id = task_id or uuid.uuid4().hex
    record: dict[str, Any] = {
        "task_id": task_id,
        "title": title,
        "payload": task,
        "metadata": dict(metadata or {}),
        "state": PENDING_STATE,
        "created_at": created_at,
        "updated_at": created_at,
        "history": [],
    }
    target_dir = _state_dir(PENDING_STATE)
    filename = _filename_for(record, state=PENDING_STATE, transition_at=_stamp(_parse_iso(created_at)))
    target_path = target_dir / filename
    record = _record_transition(record, state=PENDING_STATE, at=created_at, path=target_path)
    _atomic_write_json(target_path, record)
    return record


def _iter_state_records(state: str) -> list[dict[str, Any]]:
    ensure_queue_dirs()
    directory = _state_dir(state)
    if not directory.exists():
        return []
    items: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        try:
            record = _read_json(path)
        except Exception:
            continue
        record.setdefault("current_path", str(path))
        record.setdefault("file_name", path.name)
        items.append(record)
    items.sort(key=lambda item: (item.get("created_at", ""), item.get("task_id", "")))
    return items


def list_pending_items() -> list[dict[str, Any]]:
    return _iter_state_records(PENDING_STATE)


def list_claimed_items() -> list[dict[str, Any]]:
    return _iter_state_records(CLAIMED_STATE)


def list_completed_items(outcome: str | None = None) -> list[dict[str, Any]]:
    """Return terminal items from the outbox.

    If ``outcome`` is provided it must be one of ``completed``, ``failed`` or
    ``archived``.
    """

    if outcome is None:
        records = []
        for state in _TERMINAL_STATES:
            records.extend(_iter_state_records(state))
        records.sort(key=lambda item: (item.get("updated_at", ""), item.get("task_id", "")), reverse=True)
        return records
    outcome = outcome.lower().strip()
    if outcome not in _TERMINAL_STATES:
        raise ValueError(f"Unknown terminal outcome: {outcome}")
    return _iter_state_records(outcome)


def list_items(state: str | None = None) -> list[dict[str, Any]]:
    """Generic listing helper for pending, claimed, and terminal records."""

    if state is None:
        items = list_pending_items() + list_claimed_items() + list_completed_items()
        items.sort(key=lambda item: (item.get("updated_at", ""), item.get("task_id", "")), reverse=True)
        return items
    state = state.lower().strip()
    if state == PENDING_STATE:
        return list_pending_items()
    if state == CLAIMED_STATE:
        return list_claimed_items()
    if state in _TERMINAL_STATES:
        return list_completed_items(state)
    raise ValueError(f"Unknown queue state: {state}")


def _transition_task(
    identifier: str | Path,
    *,
    state: str,
    note: str | None = None,
    result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ensure_queue_dirs()
    source_path = _path_for_identifier(identifier)
    record = _read_json(source_path)
    current_state = str(record.get("state") or PENDING_STATE).lower().strip()

    if state == CLAIMED_STATE and current_state != PENDING_STATE:
        raise TaskStateError(f"Can only claim pending tasks (found {current_state})")
    if state in _TERMINAL_STATES and current_state != CLAIMED_STATE:
        raise TaskStateError(f"Can only finalize claimed tasks (found {current_state})")

    transitioned_at = _now_iso()
    if result:
        metadata = dict(record.get("metadata") or {})
        metadata.update(result)
        record["metadata"] = metadata

    target_dir = _state_dir(state)
    target_name = _filename_for(record, state=state, transition_at=_stamp(_parse_iso(transitioned_at)))
    target_path = target_dir / target_name

    updated = _record_transition(
        record,
        state=state,
        at=transitioned_at,
        path=target_path,
        from_path=source_path,
        note=note,
    )
    _atomic_write_json(target_path, updated)
    try:
        source_path.unlink()
    except FileNotFoundError:
        pass
    return updated


def claim_task(identifier: str | Path, *, note: str | None = None) -> dict[str, Any]:
    """Move a pending task into the claimed/in-progress area."""

    return _transition_task(identifier, state=CLAIMED_STATE, note=note)


def complete_task(
    identifier: str | Path,
    *,
    result: dict[str, Any] | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    """Mark a claimed task as completed and move it to the outbox."""

    return _transition_task(identifier, state=COMPLETED_STATE, note=note, result=result)


def fail_task(
    identifier: str | Path,
    *,
    result: dict[str, Any] | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    """Mark a claimed task as failed and move it to the outbox."""

    return _transition_task(identifier, state=FAILED_STATE, note=note, result=result)


def archive_task(
    identifier: str | Path,
    *,
    result: dict[str, Any] | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    """Archive a claimed task into the outbox without marking it failed."""

    return _transition_task(identifier, state=ARCHIVED_STATE, note=note, result=result)


def resolve_task(identifier: str | Path) -> dict[str, Any]:
    """Return the task record for ``identifier`` without moving it."""

    path = _path_for_identifier(identifier)
    record = _read_json(path)
    record.setdefault("current_path", str(path))
    record.setdefault("file_name", path.name)
    return record


__all__ = [
    "ARCHIVED_STATE",
    "CLAIMED_STATE",
    "COMPLETED_STATE",
    "FAILED_STATE",
    "PENDING_STATE",
    "QueueError",
    "TaskNotFoundError",
    "TaskStateError",
    "archive_task",
    "claim_task",
    "complete_task",
    "create_inbox_task",
    "ensure_queue_dirs",
    "fail_task",
    "list_claimed_items",
    "list_completed_items",
    "list_items",
    "list_pending_items",
    "resolve_task",
]
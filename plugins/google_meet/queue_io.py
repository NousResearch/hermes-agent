"""Concurrency-safe JSONL queue helpers for Google Meet realtime speech."""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
from typing import Generator
import uuid

from utils import atomic_write_text


@contextmanager
def locked_queue_file(queue_path: Path) -> Generator[None, None, None]:
    """Hold an exclusive advisory lock shared by every queue operation."""
    queue_path = Path(queue_path)
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = queue_path.with_name(queue_path.name + ".lock")
    with lock_path.open("a", encoding="utf-8") as lock_fp:
        try:
            import fcntl
        except (
            ImportError
        ):  # pragma: no cover - Windows is not a supported realtime host.
            yield
            return
        fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_fp.fileno(), fcntl.LOCK_UN)


def _read_jsonl_unlocked(queue_path: Path) -> tuple[list[dict], bool]:
    """Read valid queue records and assign durable ids to legacy id-less rows."""
    if not queue_path.exists():
        return [], False
    entries: list[dict] = []
    assigned_ids = False
    for line in queue_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if not isinstance(entry, dict):
            continue
        if not entry.get("id"):
            entry["id"] = str(uuid.uuid4())
            assigned_ids = True
        entries.append(entry)
    return entries, assigned_ids


def _write_jsonl_unlocked(queue_path: Path, entries: list[dict]) -> None:
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(json.dumps(entry) + "\n" for entry in entries)
    atomic_write_text(queue_path, content)


def _with_ids(entries: list[dict]) -> list[dict]:
    """Copy entries and make every persisted queue record addressable by id."""
    normalized: list[dict] = []
    for entry in entries:
        record = dict(entry)
        if not record.get("id"):
            record["id"] = str(uuid.uuid4())
        normalized.append(record)
    return normalized


def read_jsonl(queue_path: Path) -> list[dict]:
    """Return valid records, persisting ids for legacy rows before releasing the lock."""
    queue_path = Path(queue_path)
    with locked_queue_file(queue_path):
        entries, assigned_ids = _read_jsonl_unlocked(queue_path)
        if assigned_ids:
            _write_jsonl_unlocked(queue_path, entries)
        return entries


def append_jsonl(queue_path: Path, entry: dict) -> None:
    """Append one addressable JSON object while holding the queue lock."""
    queue_path = Path(queue_path)
    record = _with_ids([entry])[0]
    with locked_queue_file(queue_path):
        queue_path.parent.mkdir(parents=True, exist_ok=True)
        with queue_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(record) + "\n")


def write_jsonl(queue_path: Path, entries: list[dict]) -> None:
    """Replace addressable records under the shared queue lock."""
    queue_path = Path(queue_path)
    with locked_queue_file(queue_path):
        _write_jsonl_unlocked(queue_path, _with_ids(entries))


def remove_jsonl_entry(queue_path: Path, entry_id: str) -> None:
    """Remove only the record addressed by *entry_id*, preserving later appends."""
    queue_path = Path(queue_path)
    entry_id = str(entry_id or "")
    if not entry_id:
        return
    with locked_queue_file(queue_path):
        entries, assigned_ids = _read_jsonl_unlocked(queue_path)
        remaining = [
            entry for entry in entries if str(entry.get("id") or "") != entry_id
        ]
        if assigned_ids or len(remaining) != len(entries):
            _write_jsonl_unlocked(queue_path, remaining)

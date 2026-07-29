"""Concurrency-safe JSONL queue helpers for Google Meet realtime speech."""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
from typing import Iterator
import uuid


@contextmanager
def locked_queue_file(queue_path: Path) -> Iterator[None]:
    """Hold an exclusive advisory lock for queue reads/writes."""
    queue_path = Path(queue_path)
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = queue_path.with_name(queue_path.name + ".lock")
    with lock_path.open("a", encoding="utf-8") as lock_fp:
        try:
            import fcntl

            fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_fp.fileno(), fcntl.LOCK_UN)
        except ImportError:  # pragma: no cover - non-POSIX fallback
            yield


def _read_jsonl_unlocked(queue_path: Path) -> list[dict]:
    if not queue_path.exists():
        return []
    out: list[dict] = []
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
        if "id" not in entry:
            entry["id"] = str(uuid.uuid4())
        out.append(entry)
    return out


def _write_jsonl_unlocked(queue_path: Path, entries: list[dict]) -> None:
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    if not entries:
        queue_path.write_text("", encoding="utf-8")
        return
    queue_path.write_text(
        "\n".join(json.dumps(e) for e in entries) + "\n",
        encoding="utf-8",
    )


def read_jsonl(queue_path: Path) -> list[dict]:
    queue_path = Path(queue_path)
    with locked_queue_file(queue_path):
        return _read_jsonl_unlocked(queue_path)


def append_jsonl(queue_path: Path, entry: dict) -> None:
    queue_path = Path(queue_path)
    with locked_queue_file(queue_path):
        queue_path.parent.mkdir(parents=True, exist_ok=True)
        with queue_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(entry) + "\n")


def write_jsonl(queue_path: Path, entries: list[dict]) -> None:
    queue_path = Path(queue_path)
    with locked_queue_file(queue_path):
        _write_jsonl_unlocked(queue_path, entries)


def remove_jsonl_entry(queue_path: Path, entry_id: str) -> None:
    queue_path = Path(queue_path)
    with locked_queue_file(queue_path):
        latest = _read_jsonl_unlocked(queue_path)
        remaining = [e for e in latest if e.get("id") != entry_id]
        _write_jsonl_unlocked(queue_path, remaining)

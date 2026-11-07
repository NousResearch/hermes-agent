"""Per-job durable KV notepad (cursors, watermarks) carried across cron wake-ups; profile-local
SQLite next to the executions ledger (same connection/pragma pattern as ``cron/executions.py``).

Caps are a documented contract: ``MAX_VALUE_BYTES`` (16 KB per value, UTF-8) and
``MAX_JOB_TOTAL_BYTES`` (64 KB per job, key+value). Oversized writes raise ``ValueError`` and leave
the store untouched — the notepad is prompt-injected each run. Write path is the CLI
(``hermes cron notepad <job_id> set ...``) via the terminal tool; no model tool is added.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from hermes_constants import get_hermes_home
from hermes_time import now as _hermes_now

# Optional test override. Production resolves the path at transaction time so multiplexed profile
# ticks (set_hermes_home_override) cannot leak one profile's notepad rows into the import-time home
# — and remove_job's clear_notepad cannot wipe the wrong profile's DB.
# Same pattern as cron/executions.py. See #86519.
NOTEPAD_FILE: Optional[Path] = None
# Optional test override. Production switches only when the offline converter
# publishes the explicit format marker under <home>/cron/notepad/.
NOTEPAD_DIR: Optional[Path] = None
MAX_VALUE_BYTES = 16 * 1024
MAX_KEY_CHARS = 128
MAX_JOB_TOTAL_BYTES = 64 * 1024
FILE_STORE_MARKER = ".store-format"
FILE_STORE_FORMAT = "kotoba-notepad-v1"
_lock = threading.RLock()


def _current_notepad_file() -> Path:
    return NOTEPAD_FILE or (get_hermes_home().resolve() / "cron" / "notepad.db")


def _current_notepad_root() -> Path:
    return NOTEPAD_DIR or (get_hermes_home().resolve() / "cron" / "notepad")


def _file_store_enabled() -> bool:
    # NOTEPAD_FILE is the legacy SQLite test override and deliberately wins.
    if NOTEPAD_FILE is not None:
        return False
    marker = _current_notepad_root() / FILE_STORE_MARKER
    if not marker.exists():
        return False
    value = marker.read_text(encoding="ascii")
    if value != FILE_STORE_FORMAT + "\n":
        raise ValueError(f"unsupported cron notepad store marker: {value!r}")
    return True


def _file_job_dir(root: Path, job_id: str) -> Path:
    if not job_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in job_id):
        raise ValueError("job_id is not safe for the file-backed notepad")
    return root / job_id


def _file_note_path(job_dir: Path, key: str) -> Path:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest().encode("ascii").hex()
    return job_dir / f"n{digest}.txt"


def _decode_file_entry(text: str) -> dict[str, str]:
    parts = text.split("\n", 3)
    if len(parts) != 4 or parts[0] != "KNP1" or not parts[1] or not parts[2]:
        raise ValueError("malformed KNP1 cron notepad entry")
    if "\n" in parts[2] or "\n" in parts[1]:
        raise ValueError("malformed KNP1 cron notepad key or timestamp")
    return {"updated_at": parts[1], "key": parts[2], "value": parts[3]}


def _encode_file_entry(updated_at: str, key: str, value: str) -> str:
    if not updated_at or "\n" in updated_at or "\n" in key:
        raise ValueError("key or timestamp is not representable in KNP1")
    return f"KNP1\n{updated_at}\n{key}\n{value}"


@contextmanager
def _file_store_lock(root: Path, job_id: str) -> Iterator[None]:
    """Serialize Hermes file-store mutations across processes."""
    lock_dir = root / ".locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    name = hashlib.sha256(job_id.encode("utf-8")).hexdigest() + ".lock"
    with (lock_dir / name).open("a+b") as handle:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            if handle.read(1) == b"":
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _read_file_notes(job_dir: Path, job_id: str) -> list[dict[str, str]]:
    if not job_dir.exists():
        return []
    notes = []
    for path in job_dir.glob("n*.txt"):
        note = _decode_file_entry(path.read_text(encoding="utf-8"))
        if _file_note_path(job_dir, note["key"]).name != path.name:
            raise ValueError(f"cron notepad hash/name mismatch: {path.name}")
        if note["key"].isspace() or len(note["key"]) > MAX_KEY_CHARS:
            raise ValueError("cron notepad key violates the shared file-store contract")
        if len(note["value"].encode("utf-8")) > MAX_VALUE_BYTES:
            raise ValueError("cron notepad value exceeds the shared file-store contract")
        note["job_id"] = job_id
        notes.append(note)
    return sorted(notes, key=lambda note: note["key"])


def _write_file_entry(path: Path, content: str) -> None:
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


def _connect() -> sqlite3.Connection:
    # Late imports: a scheduler daemon that outlives an on-disk upgrade already has the OLD
    # ``hermes_cli.sqlite_util`` / ``cron.jobs`` cached, so new names must be resolved at call time,
    # not at import time (the guarantee cron/ledger.py used to carry, see e24c8499).
    from cron.jobs import _ensure_cron_dir
    from hermes_cli.sqlite_util import open_db

    path = _current_notepad_file()
    _ensure_cron_dir(path.parent)
    return open_db(path, db_label="cron/notepad.db", initialize=_initialize_schema)


def _initialize_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS cron_notepad (
             job_id TEXT NOT NULL,
             key TEXT NOT NULL,
             value TEXT NOT NULL,
             updated_at TEXT NOT NULL,
             PRIMARY KEY (job_id, key)
           )"""
    )


@contextmanager
def _transaction() -> Iterator[sqlite3.Connection]:
    from hermes_cli.sqlite_util import transaction

    with _lock, transaction(_connect()) as conn:
        yield conn


def _validate(job_id: str, key: str, value: str) -> None:
    if not str(job_id):
        raise ValueError("job_id must be non-empty")
    if not key:
        raise ValueError("key must be non-empty")
    if len(key) > MAX_KEY_CHARS:
        raise ValueError(f"key too long (max {MAX_KEY_CHARS} characters)")
    if len(value.encode("utf-8")) > MAX_VALUE_BYTES:
        raise ValueError(f"value too large (max {MAX_VALUE_BYTES} bytes per key)")


def set_note(job_id: str, key: str, value: str) -> dict[str, Any]:
    """Upsert one key. Raises ValueError when a size cap would be exceeded."""
    job_id, key, value = str(job_id), str(key), str(value)
    _validate(job_id, key, value)
    now = _hermes_now().isoformat()
    if _file_store_enabled():
        if key.isspace() or "\n" in key:
            raise ValueError("key is not representable in the Kotoba notepad file store")
        root = _current_notepad_root()
        job_dir = _file_job_dir(root, job_id)
        with _lock, _file_store_lock(root, job_id):
            job_dir.mkdir(parents=True, exist_ok=True)
            notes = _read_file_notes(job_dir, job_id)
            path = _file_note_path(job_dir, key)
            if path.exists():
                existing = _decode_file_entry(path.read_text(encoding="utf-8"))
                if existing["key"] != key:
                    raise ValueError("cron notepad SHA-256 key collision")
            other_bytes = sum(
                len(note["key"].encode("utf-8")) + len(note["value"].encode("utf-8"))
                for note in notes if note["key"] != key
            )
            entry_bytes = len(key.encode("utf-8")) + len(value.encode("utf-8"))
            if other_bytes + entry_bytes > MAX_JOB_TOTAL_BYTES:
                raise ValueError(
                    f"notepad full: job '{job_id}' would exceed "
                    f"{MAX_JOB_TOTAL_BYTES} bytes total; delete unused keys first"
                )
            _write_file_entry(path, _encode_file_entry(now, key, value))
        return {"job_id": job_id, "key": key, "value": value, "updated_at": now}
    with _transaction() as conn:
        row = conn.execute(
            """SELECT COALESCE(SUM(LENGTH(CAST(key AS BLOB))
                 + LENGTH(CAST(value AS BLOB))), 0)
               FROM cron_notepad WHERE job_id=? AND key<>?""",
            (job_id, key),
        ).fetchone()
        other_bytes = int(row[0])
        entry_bytes = len(key.encode("utf-8")) + len(value.encode("utf-8"))
        if other_bytes + entry_bytes > MAX_JOB_TOTAL_BYTES:
            raise ValueError(
                f"notepad full: job '{job_id}' would exceed "
                f"{MAX_JOB_TOTAL_BYTES} bytes total; delete unused keys first"
            )
        conn.execute(
            """INSERT INTO cron_notepad (job_id, key, value, updated_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(job_id, key)
               DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at""",
            (job_id, key, value, now),
        )
    return {"job_id": job_id, "key": key, "value": value, "updated_at": now}


def get_note(job_id: str, key: str) -> Optional[str]:
    if _file_store_enabled():
        job_id, key = str(job_id), str(key)
        job_dir = _file_job_dir(_current_notepad_root(), job_id)
        path = _file_note_path(job_dir, key)
        if not path.exists():
            return None
        note = _decode_file_entry(path.read_text(encoding="utf-8"))
        if note["key"] != key:
            raise ValueError("cron notepad SHA-256 key collision")
        return note["value"]
    with _transaction() as conn:
        row = conn.execute(
            "SELECT value FROM cron_notepad WHERE job_id=? AND key=?",
            (str(job_id), str(key)),
        ).fetchone()
    return None if row is None else row["value"]


def delete_note(job_id: str, key: str) -> bool:
    if _file_store_enabled():
        job_id, key = str(job_id), str(key)
        root = _current_notepad_root()
        job_dir = _file_job_dir(root, job_id)
        path = _file_note_path(job_dir, key)
        with _lock, _file_store_lock(root, job_id):
            if not path.exists():
                return False
            note = _decode_file_entry(path.read_text(encoding="utf-8"))
            if note["key"] != key:
                raise ValueError("cron notepad SHA-256 key collision")
            path.unlink()
            return True
    with _transaction() as conn:
        cur = conn.execute(
            "DELETE FROM cron_notepad WHERE job_id=? AND key=?",
            (str(job_id), str(key)),
        )
    return cur.rowcount > 0


def list_notes(job_id: str) -> list[dict[str, Any]]:
    """All entries for one job, sorted by key."""
    if _file_store_enabled():
        job_id = str(job_id)
        job_dir = _file_job_dir(_current_notepad_root(), job_id)
        return _read_file_notes(job_dir, job_id)
    with _transaction() as conn:
        rows = conn.execute(
            "SELECT job_id, key, value, updated_at FROM cron_notepad "
            "WHERE job_id=? ORDER BY key",
            (str(job_id),),
        ).fetchall()
    return [dict(row) for row in rows]


def clear_notepad(job_id: str) -> int:
    """Delete every key for one job (called from ``cron.jobs.remove_job``). Returns row count;
    no-ops without creating the DB when no notepad file exists yet."""
    if _file_store_enabled():
        job_id = str(job_id)
        root = _current_notepad_root()
        job_dir = _file_job_dir(root, job_id)
        if not job_dir.exists():
            return 0
        with _lock, _file_store_lock(root, job_id):
            notes = _read_file_notes(job_dir, job_id)
            for note in notes:
                _file_note_path(job_dir, note["key"]).unlink()
            return len(notes)
    if not _current_notepad_file().exists():
        return 0
    with _transaction() as conn:
        cur = conn.execute(
            "DELETE FROM cron_notepad WHERE job_id=?", (str(job_id),)
        )
    return cur.rowcount


def render_notepad_section(job_id: str) -> str:
    """Render a job's notepad as a prompt section. An empty notepad MUST return '' so jobs that
    never use the feature get a byte-identical prompt (prompt-cache + drift safety)."""
    try:
        notes = list_notes(job_id)
    except Exception:
        return ""
    if not notes:
        return ""
    lines = [f"- {note['key']}: {note['value']}" for note in notes]
    return (
        "## Job notepad (persistent across runs)\n"
        "This durable scratchpad survives between scheduled runs of this "
        "job. Update it via the CLI, e.g.:\n"
        f"`hermes cron notepad {job_id} set <key> <value>` "
        f"(also: get/delete/list; `hermes cron notepad {job_id} delete "
        "<key>` removes an entry).\n\n" + "\n".join(lines) + "\n\n"
    )

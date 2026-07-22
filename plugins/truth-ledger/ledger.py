from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import sqlite3
import time
from datetime import datetime
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

from jsonschema import Draft202012Validator, FormatChecker

_FORMAT_CHECKER = FormatChecker()
_RFC3339_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}[Tt]\d{2}:\d{2}:\d{2}(?:\.\d+)?"
    r"(?:[Zz]|[+-](?P<offset_hour>\d{2}):(?P<offset_minute>\d{2}))$"
)


@_FORMAT_CHECKER.checks("date-time", raises=(TypeError, ValueError))
def _is_rfc3339_datetime(value: object) -> bool:
    if not isinstance(value, str):
        return False
    match = _RFC3339_RE.fullmatch(value)
    if match is None:
        return False
    if match.group("offset_hour") is not None and (
        int(match.group("offset_hour")) > 23 or int(match.group("offset_minute")) > 59
    ):
        return False
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00").replace("z", "+00:00"))
    return parsed.tzinfo is not None


@lru_cache(maxsize=1)
def _ledger_event_validator() -> Draft202012Validator:
    schema_path = Path(__file__).with_name("schemas") / "ledger-event-v1.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    return Draft202012Validator(schema, format_checker=_FORMAT_CHECKER)


def _valid_ledger_event(document: Dict[str, Any]) -> bool:
    return next(_ledger_event_validator().iter_errors(document), None) is None


def _mkdir_private(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path, 0o700)
    except OSError:
        pass


class _FileLock:
    def __init__(self, path: Path, timeout_seconds: float = 0.2):
        self.path = path
        self.timeout_seconds = timeout_seconds
        self.fd: int | None = None

    def __enter__(self):
        import fcntl

        _mkdir_private(self.path.parent)
        self.fd = os.open(str(self.path), os.O_CREAT | os.O_RDWR, 0o600)
        deadline = time.time() + self.timeout_seconds
        while True:
            try:
                fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return self
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EAGAIN):
                    raise
                if time.time() >= deadline:
                    raise TimeoutError(f"lock timeout: {self.path}")
                time.sleep(0.01)

    def __exit__(self, exc_type, exc, tb):
        import fcntl

        if self.fd is not None:
            try:
                fcntl.flock(self.fd, fcntl.LOCK_UN)
            finally:
                os.close(self.fd)
        self.fd = None


class LedgerStore:
    def __init__(self, root: Path, record_hard_bytes: int = 64 * 1024) -> None:
        self.root = Path(root)
        self.ledger_dir = self.root / "ledger"
        self.state_dir = self.root / "state"
        self.locks_dir = self.state_dir / "locks"
        self.errors_dir = self.root / "errors"
        self.db_path = self.state_dir / "index.sqlite"
        self.append_lock = self.locks_dir / "append.lock"
        self.record_hard_bytes = record_hard_bytes

        _mkdir_private(self.root)
        for d in (self.ledger_dir, self.state_dir, self.locks_dir, self.errors_dir):
            _mkdir_private(d)
        with _FileLock(self.append_lock, timeout_seconds=5.0):
            self._ensure_private_db_file()
            self._init_db()

    def _ensure_private_db_file(self) -> None:
        flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(str(self.db_path), flags, 0o600)
        try:
            if os.fstat(fd).st_mode & 0o777 != 0o600:
                os.fchmod(fd, 0o600)
        finally:
            os.close(fd)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=5.0)
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS event_journal (
                    event_key TEXT PRIMARY KEY,
                    event_id TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL,
                    ledger_file TEXT,
                    ledger_offset INTEGER,
                    checksum TEXT,
                    updated_at REAL NOT NULL
                )
                """
            )
            conn.commit()

    def _find_event_offset_and_checksum(self, ledger_file: Path, event_id: str) -> tuple[int | None, str | None]:
        if not ledger_file.exists():
            return None, None

        offset = 0
        with ledger_file.open("rb") as fh:
            for raw_line in fh:
                offset += len(raw_line)
                try:
                    row = json.loads(raw_line.decode("utf-8"))
                except Exception:
                    continue
                if str(row.get("event_id") or "") == event_id:
                    return offset, hashlib.sha256(raw_line).hexdigest()
        return None, None

    def _repair_tail_for_append(self, ledger_file: Path) -> None:
        """Make the canonical JSONL tail append-safe while preserving evidence."""
        if not ledger_file.exists():
            return
        raw = ledger_file.read_bytes()
        if not raw:
            return

        valid_end = 0
        corrupt_from: int | None = None
        cursor = 0
        for raw_line in raw.splitlines(keepends=True):
            line_end = cursor + len(raw_line)
            has_newline = raw_line.endswith(b"\n")
            candidate = raw_line[:-1] if has_newline else raw_line
            try:
                json.loads(candidate.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                corrupt_from = cursor
                break
            valid_end = line_end
            cursor = line_end
            if not has_newline:
                with ledger_file.open("ab") as fh:
                    fh.write(b"\n")
                    fh.flush()
                    os.fsync(fh.fileno())
                return

        if corrupt_from is None:
            return

        _mkdir_private(self.errors_dir)
        quarantine = self.errors_dir / (
            f"append-corrupt-tail-{ledger_file.stem}-{time.time_ns()}.jsonl"
        )
        with quarantine.open("wb") as fh:
            fh.write(raw[corrupt_from:])
            fh.flush()
            os.fsync(fh.fileno())
        try:
            os.chmod(quarantine, 0o600)
        except OSError:
            pass

        with ledger_file.open("r+b") as fh:
            fh.truncate(valid_end)
            fh.flush()
            os.fsync(fh.fileno())

    def append_event(self, event: Dict[str, Any], event_key: str) -> Dict[str, Any]:
        event_copy = dict(event)
        event_copy.setdefault("occurred_at", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        event_id = str(event_copy.get("event_id") or f"evt_{hashlib.sha256(event_key.encode('utf-8')).hexdigest()[:32]}")
        event_copy["event_id"] = event_id

        line = json.dumps(event_copy, separators=(",", ":"), ensure_ascii=False)
        line_bytes = line.encode("utf-8") + b"\n"
        if len(line_bytes) > self.record_hard_bytes:
            return {"status": "rejected", "reason": "record_hard_cap"}
        if not _valid_ledger_event(event_copy):
            return {"status": "rejected", "reason": "invalid_ledger_event"}

        month = event_copy["occurred_at"][:7]
        ledger_file = self.ledger_dir / f"{month}.jsonl"

        try:
            with self._connect() as conn:
                cur = conn.execute(
                    "SELECT status, event_id, ledger_file FROM event_journal WHERE event_key = ?",
                    (event_key,),
                )
                row = cur.fetchone()
                if row and row[0] == "indexed":
                    return {"status": "duplicate", "event_id": event_id}
                if row and row[0] == "intent":
                    intent_event_id = str(row[1] or event_id)
                    intent_ledger_name = str(row[2] or ledger_file.name)
                    intent_ledger_file = self.ledger_dir / intent_ledger_name
                    event_id = intent_event_id
                    event_copy["event_id"] = event_id
                    line = json.dumps(event_copy, separators=(",", ":"), ensure_ascii=False)
                    line_bytes = line.encode("utf-8") + b"\n"
                    recovered_offset, recovered_checksum = self._find_event_offset_and_checksum(
                        intent_ledger_file,
                        intent_event_id,
                    )
                    if recovered_offset is not None and recovered_checksum is not None:
                        conn.execute(
                            "UPDATE event_journal SET status = ?, ledger_offset = ?, checksum = ?, updated_at = ? WHERE event_key = ?",
                            ("indexed", recovered_offset, recovered_checksum, time.time(), event_key),
                        )
                        conn.commit()
                        return {
                            "status": "indexed",
                            "event_id": intent_event_id,
                            "ledger_file": str(intent_ledger_file),
                            "recovered_from_intent": True,
                        }
                if row is None:
                    conn.execute(
                        "INSERT INTO event_journal(event_key, event_id, status, ledger_file, updated_at) VALUES (?, ?, ?, ?, ?)",
                        (event_key, event_id, "intent", ledger_file.name, time.time()),
                    )
                    conn.commit()
        except sqlite3.OperationalError as exc:
            if "locked" in str(exc).lower() or "busy" in str(exc).lower():
                return {"status": "retry", "reason": "index_db_locked", "event_id": event_id}
            raise

        try:
            with _FileLock(self.append_lock, timeout_seconds=0.2):
                self._repair_tail_for_append(ledger_file)
                with self._connect() as conn:
                    cur = conn.execute(
                        "SELECT status, event_id, ledger_file FROM event_journal WHERE event_key = ?",
                        (event_key,),
                    )
                    row = cur.fetchone()
                    if row and row[0] == "indexed":
                        return {"status": "duplicate", "event_id": str(row[1] or event_id)}
                    if row and row[0] == "intent":
                        intent_event_id = str(row[1] or event_id)
                        intent_ledger_name = str(row[2] or ledger_file.name)
                        intent_ledger_file = self.ledger_dir / intent_ledger_name
                        recovered_offset, recovered_checksum = self._find_event_offset_and_checksum(
                            intent_ledger_file,
                            intent_event_id,
                        )
                        if recovered_offset is not None and recovered_checksum is not None:
                            conn.execute(
                                "UPDATE event_journal SET status = ?, ledger_offset = ?, checksum = ?, updated_at = ? WHERE event_key = ?",
                                ("indexed", recovered_offset, recovered_checksum, time.time(), event_key),
                            )
                            conn.commit()
                            return {
                                "status": "indexed",
                                "event_id": intent_event_id,
                                "ledger_file": str(intent_ledger_file),
                                "recovered_from_intent": True,
                            }

                _mkdir_private(ledger_file.parent)
                with ledger_file.open("ab") as fh:
                    fh.write(line_bytes)
                    fh.flush()
                    os.fsync(fh.fileno())
                    offset = fh.tell()
                try:
                    os.chmod(ledger_file, 0o600)
                except OSError:
                    pass
        except TimeoutError:
            return {"status": "retry", "reason": "append_lock_timeout", "event_id": event_id}

        checksum = hashlib.sha256(line_bytes).hexdigest()
        try:
            with self._connect() as conn:
                conn.execute(
                    "UPDATE event_journal SET status = ?, ledger_offset = ?, checksum = ?, updated_at = ? WHERE event_key = ?",
                    ("indexed", offset, checksum, time.time(), event_key),
                )
                conn.commit()
        except sqlite3.OperationalError as exc:
            if "locked" in str(exc).lower() or "busy" in str(exc).lower():
                return {"status": "retry", "reason": "index_db_locked", "event_id": event_id}
            raise

        return {"status": "indexed", "event_id": event_id, "ledger_file": str(ledger_file)}


def scan_jsonl_with_tail_quarantine(path: Path, quarantine_dir: Path) -> List[Dict[str, Any]]:
    path = Path(path)
    quarantine_dir = Path(quarantine_dir)
    if not path.exists():
        return []

    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    parsed: List[Dict[str, Any]] = []
    quarantine_from = None

    for idx, raw_line in enumerate(lines):
        if not raw_line.endswith(b"\n"):
            quarantine_from = idx
            break
        try:
            text = raw_line.decode("utf-8")
            parsed.append(json.loads(text))
        except Exception:
            quarantine_from = idx
            break

    if quarantine_from is not None:
        _mkdir_private(quarantine_dir)
        suffix = b"".join(lines[quarantine_from:])
        q = quarantine_dir / f"corrupt-tail-{path.stem}-{int(time.time())}.jsonl"
        with q.open("wb") as fh:
            fh.write(suffix)
            fh.flush()
            os.fsync(fh.fileno())
        try:
            os.chmod(q, 0o600)
        except OSError:
            pass

    return parsed

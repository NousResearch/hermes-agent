from __future__ import annotations

import errno
import hashlib
import json
import os
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List


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

        for d in (self.ledger_dir, self.state_dir, self.locks_dir, self.errors_dir):
            _mkdir_private(d)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=250")
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

    def append_event(self, event: Dict[str, Any], event_key: str) -> Dict[str, Any]:
        event_copy = dict(event)
        event_copy.setdefault("occurred_at", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        event_id = str(event_copy.get("event_id") or hashlib.sha256(event_key.encode("utf-8")).hexdigest()[:32])
        event_copy["event_id"] = event_id

        line = json.dumps(event_copy, separators=(",", ":"), ensure_ascii=False)
        line_bytes = line.encode("utf-8") + b"\n"
        if len(line_bytes) > self.record_hard_bytes:
            return {"status": "rejected", "reason": "record_hard_cap"}

        month = event_copy["occurred_at"][:7]
        ledger_file = self.ledger_dir / f"{month}.jsonl"

        with self._connect() as conn:
            cur = conn.execute("SELECT status FROM event_journal WHERE event_key = ?", (event_key,))
            row = cur.fetchone()
            if row and row[0] == "indexed":
                return {"status": "duplicate", "event_id": event_id}
            if row is None:
                conn.execute(
                    "INSERT INTO event_journal(event_key, event_id, status, ledger_file, updated_at) VALUES (?, ?, ?, ?, ?)",
                    (event_key, event_id, "intent", ledger_file.name, time.time()),
                )
                conn.commit()

        try:
            with _FileLock(self.append_lock, timeout_seconds=0.2):
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
        with self._connect() as conn:
            conn.execute(
                "UPDATE event_journal SET status = ?, ledger_offset = ?, checksum = ?, updated_at = ? WHERE event_key = ?",
                ("indexed", offset, checksum, time.time(), event_key),
            )
            conn.commit()

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

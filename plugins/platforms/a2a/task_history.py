"""Bounded terminal A2A facts. Futures, running execution and push credentials stay in memory."""

import json
import sqlite3
from contextlib import closing
from pathlib import Path


class TerminalTaskHistory:
    def __init__(self, home: Path, limit: int):
        # The adapter's owning profile is captured before HTTP worker threads start.
        directory = home / "a2a"
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.path = directory / "terminal_tasks.sqlite3"
        self.limit = limit
        with closing(self._connect()) as db, db:
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1):
                raise RuntimeError("Unsupported A2A terminal task history version; preserve the original database.")
            db.execute("CREATE TABLE IF NOT EXISTS terminal_tasks "
                       "(task_id TEXT PRIMARY KEY, created_at REAL NOT NULL, record TEXT NOT NULL)")
            db.execute("PRAGMA user_version=1")
        self.path.chmod(0o600)

    def _connect(self):
        db = sqlite3.connect(self.path, timeout=5)
        db.execute("PRAGMA synchronous=FULL")
        return db

    def load(self) -> list[dict]:
        with closing(self._connect()) as db:
            rows = db.execute("SELECT record FROM terminal_tasks ORDER BY created_at, task_id").fetchall()
        return [json.loads(row[0]) for row in rows]

    def save(self, record: dict) -> None:
        saved = {key: record[key] for key in (
            "task_id", "context_id", "peer", "agent_slug", "tenant", "input_id", "state",
            "reply", "created_at", "created_iso", "completed_at",
        )}
        with closing(self._connect()) as db, db:
            db.execute("INSERT INTO terminal_tasks VALUES (?, ?, ?)",
                       (record["task_id"], record["created_at"], json.dumps(saved)))
            db.execute("DELETE FROM terminal_tasks WHERE task_id NOT IN "
                       "(SELECT task_id FROM terminal_tasks ORDER BY created_at DESC, task_id DESC LIMIT ?)",
                       (self.limit,))

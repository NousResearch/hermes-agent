"""Persistent deletion receipts for automatically spoken Signal replies."""

import sqlite3
import time
from contextlib import closing
from pathlib import Path

DAY_MS = 86_400_000


def expiry_minutes(value: object) -> int:
    if type(value) is not int or not 0 <= value <= 1380:
        raise ValueError("Signal voice_reply_expiry_minutes must be an integer from 0 to 1380")
    return value


class VoiceReplyExpiry:
    def __init__(self, path: Path, account: str):
        self.path = path
        self.account = account

    def remember(self, target: str, timestamp: int, minutes: int) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("""CREATE TABLE IF NOT EXISTS pending (
                account TEXT NOT NULL, target TEXT NOT NULL, sent INTEGER NOT NULL,
                due INTEGER NOT NULL, PRIMARY KEY (account, target, sent))""")
            db.execute("INSERT OR IGNORE INTO pending VALUES (?, ?, ?, ?)",
                       (self.account, target, timestamp, timestamp + minutes * 60_000))

    async def process_due(self, remove) -> None:
        """Retry pending sends until the sender's 24-hour window expires."""
        if not self.path.exists():
            return
        now = int(time.time() * 1000)
        with closing(sqlite3.connect(self.path)) as db:
            with db:
                db.execute("DELETE FROM pending WHERE account = ? AND sent <= ?", (self.account, now - DAY_MS))
            rows = db.execute("SELECT target, sent FROM pending WHERE account = ? AND due <= ? "
                              "ORDER BY due LIMIT 32", (self.account, now)).fetchall()
            for target, sent in rows:
                key = (self.account, target, sent)
                expired = int(time.time() * 1000) >= sent + DAY_MS
                if expired or await remove(target, sent):
                    with db:
                        db.execute("DELETE FROM pending WHERE account = ? AND target = ? AND sent = ?", key)
                else:
                    with db:
                        db.execute("UPDATE pending SET due = ? WHERE account = ? AND target = ? AND sent = ?",
                                   (int(time.time() * 1000) + 60_000, *key))

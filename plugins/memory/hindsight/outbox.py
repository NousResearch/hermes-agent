"""Crash-safe retain journal. A turn is written here BEFORE it is handed to the
writer thread, so a SIGKILL / os._exit / gateway restart between enqueue and the
network call cannot silently drop it — a fresh provider replays it on the next
``initialize()``.

Legacy retain path only (``HindsightMemoryProvider._retain_batch``): no caller-
supplied idempotency id, so resubmitting a row is an at-least-once operation.
See ``RetainReliability`` in ``reliability.py`` for how that risk is bounded.
"""
import contextlib
import json
import os
from pathlib import Path
import sqlite3
import uuid


class Outbox:
    # Caps how many times a crash-interrupted or explicitly-requeued row may be
    # resubmitted before it is left quarantined for good (owner review).
    MAX_ATTEMPTS = 5

    def __init__(self, home: Path, partition: str):
        root = home / "hindsight" / "outbox"
        for directory in (home / "hindsight", root):
            if directory.is_symlink():
                raise RuntimeError("Unsafe Hindsight journal directory")
            directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            directory.chmod(0o700)
        self.path = root / (partition + ".sqlite3")
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
        try:
            self.path.chmod(0o600)
        finally:
            os.close(fd)
        with self.connect(initialize=True) as db:
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version == 0:
                if db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchone():
                    raise RuntimeError("Unknown Hindsight journal schema; owner review required")
                db.execute("""CREATE TABLE work (
                    id TEXT PRIMARY KEY, document_id TEXT NOT NULL, state TEXT NOT NULL,
                    payload TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0)""")
                db.execute("PRAGMA user_version=1")
            elif version != 1:
                raise RuntimeError("Unknown Hindsight journal version; owner review required")

    @contextlib.contextmanager
    def connect(self, *, initialize=False):
        db = sqlite3.connect(self.path, timeout=1)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA secure_delete=ON")
            db.execute("PRAGMA synchronous=FULL")
            db.execute("BEGIN IMMEDIATE")
            if not initialize and db.execute("PRAGMA user_version").fetchone()[0] != 1:
                raise RuntimeError("Unknown Hindsight journal version; owner review required")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def put(self, payload: dict, *, document_id: str) -> str:
        """Durably record *payload* (the exact ``_retain_batch`` kwargs) as queued
        work and return its identity. Returns BEFORE any network call is made."""
        identity = uuid.uuid4().hex
        with self.connect() as db:
            db.execute(
                "INSERT INTO work(id, document_id, state, payload, attempts) VALUES(?,?,?,?,0)",
                (identity, document_id, "queued", json.dumps(payload, ensure_ascii=False)),
            )
        return identity

    def claim(self, identity: str) -> dict | None:
        """queued -> sending, counted as one attempt. None if already claimed/finished."""
        with self.connect() as db:
            row = db.execute("SELECT state, payload FROM work WHERE id=?", (identity,)).fetchone()
            if row is None or row["state"] != "queued":
                return None
            db.execute("UPDATE work SET state='sending', attempts=attempts+1 WHERE id=?", (identity,))
            return json.loads(row["payload"])

    def finish(self, identity: str, state: str) -> None:
        """sending -> done|quarantined. A late/duplicate call cannot undo a
        terminal acknowledgement (``done`` is final; ``quarantined`` needs the
        explicit :meth:`requeue` path)."""
        with self.connect() as db:
            row = db.execute("SELECT state FROM work WHERE id=?", (identity,)).fetchone()
            if row is None or row["state"] in {"done", "quarantined"}:
                return
            db.execute("UPDATE work SET state=? WHERE id=?", (state, identity))
            if state == "done":
                # Acknowledged; keep the row (for `pending()`/audit) but drop the payload.
                db.execute("UPDATE work SET payload='{}' WHERE id=?", (identity,))

    def pending(self) -> list[dict]:
        """Everything not yet durably acknowledged — queued, mid-send, or quarantined."""
        with self.connect() as db:
            return [dict(row) for row in db.execute("SELECT * FROM work WHERE state!='done' ORDER BY rowid")]

    def recover(self) -> list[str]:
        """Called once on startup. A row left in ``sending`` means a prior process
        died between :meth:`claim` and :meth:`finish` — durability's entire point,
        so (bounded by ``MAX_ATTEMPTS``) it goes back to ``queued`` for a fresh
        provider to retry. ``queued`` rows were never claimed and are returned
        as-is. ``quarantined`` rows are a completed, ambiguous failure and are
        never auto-resubmitted here — see :meth:`requeue`.

        Returns the ids now sitting in ``queued`` state, ready to hand to the writer.
        """
        with self.connect() as db:
            stuck = db.execute("SELECT id, attempts FROM work WHERE state='sending'").fetchall()
            for row in stuck:
                new_state = "queued" if row["attempts"] < self.MAX_ATTEMPTS else "quarantined"
                db.execute("UPDATE work SET state=? WHERE id=?", (new_state, row["id"]))
            return [row["id"] for row in db.execute("SELECT id FROM work WHERE state='queued' ORDER BY rowid")]

    def requeue(self, identity: str) -> bool:
        """Explicit, bounded recovery for a ``quarantined`` row: an operator (or a
        caller who has independently confirmed the earlier send did NOT land)
        asking for one more try. False once ``MAX_ATTEMPTS`` is reached — the row
        stays quarantined for good and needs owner review, not another retry."""
        with self.connect() as db:
            row = db.execute("SELECT state, attempts FROM work WHERE id=?", (identity,)).fetchone()
            if row is None or row["state"] != "quarantined" or row["attempts"] >= self.MAX_ATTEMPTS:
                return False
            db.execute("UPDATE work SET state='queued' WHERE id=?", (identity,))
            return True

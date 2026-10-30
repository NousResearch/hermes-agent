"""Instance-owned model picker preferences, independent of conversations/config.

Each field retains its revision even after cancellation. This makes a first-time
import (base_revision=0) unable to resurrect a choice cancelled on another client.
"""

import os
import sqlite3
from contextlib import closing, contextmanager
from pathlib import Path
from uuid import uuid4


class PreferenceConflict(Exception):
    pass


class ModelPreferences:
    def __init__(self, home: Path):
        self.path = home / "model-preferences.db"

    @contextmanager
    def transaction(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_CREAT | os.O_WRONLY, 0o600)
        os.close(fd)
        with closing(sqlite3.connect(self.path, timeout=5)) as db, db:
            db.row_factory = sqlite3.Row
            db.execute("BEGIN IMMEDIATE")
            db.execute("CREATE TABLE IF NOT EXISTS identity (id TEXT NOT NULL, revision INTEGER NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS preferences (provider TEXT NOT NULL, model TEXT NOT NULL, "
                       "favorite INTEGER NOT NULL DEFAULT 0, hidden INTEGER, "
                       "favorite_revision INTEGER NOT NULL DEFAULT 0, hidden_revision INTEGER NOT NULL DEFAULT 0, "
                       "PRIMARY KEY(provider,model))")
            if db.execute("SELECT 1 FROM identity").fetchone() is None:
                db.execute("INSERT INTO identity VALUES (?,0)", (str(uuid4()),))
            yield db

    @staticmethod
    def snapshot_from(db):
        identity = db.execute("SELECT id,revision FROM identity").fetchone()
        rows = []
        for row in db.execute("SELECT * FROM preferences ORDER BY provider,model"):
            rows.append({"provider": row["provider"], "model": row["model"],
                         "favorite": bool(row["favorite"]),
                         "hidden": None if row["hidden"] is None else bool(row["hidden"]),
                         "favoriteRevision": str(row["favorite_revision"]),
                         "hiddenRevision": str(row["hidden_revision"])})
        return {"version": 1, "instanceId": identity["id"], "revision": str(identity["revision"]), "rows": rows}

    def snapshot(self):
        with self.transaction() as db:
            return self.snapshot_from(db)

    def set(self, instance_id: str, provider: str, model: str, field: str, base: int, value):
        if field not in ("favorite", "hidden") or (type(value) is not bool and not (field == "hidden" and value is None)):
            raise ValueError("Invalid preference")
        with self.transaction() as db:
            identity = db.execute("SELECT id,revision FROM identity").fetchone()
            row = db.execute("SELECT * FROM preferences WHERE provider=? AND model=?", (provider, model)).fetchone()
            if instance_id != identity["id"] or (row[field + "_revision"] if row else 0) != base:
                raise PreferenceConflict()
            if row is None and db.execute("SELECT count(*) FROM preferences").fetchone()[0] >= 5000:
                raise ValueError("Model preference capacity reached")
            revision = identity["revision"] + 1
            if revision >= 2**63:
                raise ValueError("Model preference revision exhausted")
            db.execute("INSERT OR IGNORE INTO preferences(provider,model) VALUES (?,?)", (provider, model))
            # field is a fixed allowlist above, never a SQL identifier from a client.
            db.execute(f"UPDATE preferences SET {field}=?,{field}_revision=? WHERE provider=? AND model=?",
                       (value, revision, provider, model))
            db.execute("UPDATE identity SET revision=?", (revision,))
            return self.snapshot_from(db)

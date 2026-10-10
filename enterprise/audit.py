"""Append-only audit log for the enterprise control plane.

Every control-plane mutation and authorization decision produces one
attributable record. Records never contain credentials, secret values,
provider message contents, or runtime message contents — the writer
actively refuses payloads that look like they do.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from .resources import _find_secretlike_keys  # shared secret-shape detector

_SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL NOT NULL,
    actor       TEXT NOT NULL,
    actor_kind  TEXT NOT NULL,
    action      TEXT NOT NULL,
    kind        TEXT,
    namespace   TEXT,
    resource    TEXT,
    outcome     TEXT NOT NULL,
    reason      TEXT,
    detail      TEXT
);
CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_log(ts);
CREATE INDEX IF NOT EXISTS idx_audit_resource ON audit_log(kind, namespace, resource);
"""

ALLOWED_OUTCOMES = ("allow", "deny", "error", "applied", "rolled-back")


class AuditLog:
    def __init__(self, db_path: str | Path):
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        try:
            self._conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.DatabaseError:
            pass
        with self._lock, self._conn:
            self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def record(
        self,
        *,
        actor: str,
        actor_kind: str,
        action: str,
        outcome: str,
        kind: str | None = None,
        namespace: str | None = None,
        resource: str | None = None,
        reason: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> int:
        if outcome not in ALLOWED_OUTCOMES:
            raise ValueError(f"outcome must be one of {ALLOWED_OUTCOMES}")
        if detail:
            leaked = _find_secretlike_keys(detail)
            if leaked:
                raise ValueError(
                    f"audit detail contains secret-like values at {sorted(leaked)}; "
                    "audit records must never carry secrets"
                )
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO audit_log (ts, actor, actor_kind, action, kind,"
                " namespace, resource, outcome, reason, detail)"
                " VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    time.time(),
                    actor,
                    actor_kind,
                    action,
                    kind,
                    namespace,
                    resource,
                    outcome,
                    reason,
                    json.dumps(detail) if detail else None,
                ),
            )
            return int(cur.lastrowid or 0)

    def query(
        self,
        *,
        kind: str | None = None,
        namespace: str | None = None,
        resource: str | None = None,
        since: float | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        clauses, params = [], []  # type: ignore[var-annotated]
        for col, val in (("kind", kind), ("namespace", namespace),
                         ("resource", resource)):
            if val is not None:
                clauses.append(f"{col}=?")
                params.append(val)
        if since is not None:
            clauses.append("ts>=?")
            params.append(since)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(max(1, min(int(limit), 1000)))
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM audit_log {where} ORDER BY id DESC LIMIT ?", params
            ).fetchall()
        out = []
        for r in rows:
            rec = dict(r)
            if rec.get("detail"):
                rec["detail"] = json.loads(rec["detail"])
            out.append(rec)
        return out

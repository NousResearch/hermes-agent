"""Approval state machine + immutable decision log for the content engine.

Extracted from the t_0553ea32 scratch pipeline (publishing_bridge/approval.py)
and adapted to the LIVE content engine, because the scratch version modelled
real guarantees the live path lacks:

- Every approve/reject/amend/resubmit event is recorded append-only with
  actor, comment, version and timestamp (the live path only stored
  approved_at/rejected_at — no audit trail).
- Transitions are validated against the live ``drafts.status`` vocabulary;
  invalid transitions raise instead of silently overwriting state.
- Version history: each amend/resubmit cycle increments the version.
- Optional actor allowlist: when ``approver_ids`` is provided, every
  transition validates the actor against it.

Status vocabulary (matches the live ``drafts`` table + approval handler):

    draft ──approve()──► approved    (terminal, publishable)
    draft ──reject()──►  rejected    (terminal)
    draft ──amend()──►   amended
    amended ──resubmit()──► draft    (version + 1)

This module is an additive audit/guard layer: it does not change the status
strings the pipeline already uses, so downstream consumers (digest, publisher,
approval handler) are unaffected.
"""

from __future__ import annotations

import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ── DB resolution ──────────────────────────────────────────────────────────

def _default_db_path() -> str:
    """Resolve the content engine DB path (env override, else config)."""
    env = os.environ.get("CONTENT_ENGINE_DB_PATH")
    if env:
        return str(Path(env).resolve())
    try:
        from config import DB_PATH

        return str(DB_PATH)
    except Exception:
        return str(
            Path(__file__).resolve().parent / "db" / "content_engine.db"
        )


DECISION_SCHEMA = """
CREATE TABLE IF NOT EXISTS approval_decisions (
    id          TEXT PRIMARY KEY,
    draft_id    TEXT NOT NULL,
    brand       TEXT,
    platform    TEXT,
    action      TEXT NOT NULL,   -- approve | reject | amend_request | resubmit
    actor       TEXT NOT NULL,
    comment     TEXT,
    version     INTEGER NOT NULL,
    decided_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_approval_decisions_draft
    ON approval_decisions(draft_id, decided_at);
"""


class ApprovalError(Exception):
    """Raised on invalid approval-state transitions."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Live status vocabulary ─────────────────────────────────────────────────

# Maps (current_status, action) -> new_status. None = illegal transition.
_TRANSITIONS: dict[tuple[str, str], str] = {
    ("draft", "approve"): "approved",
    ("draft", "reject"): "rejected",
    ("draft", "amend"): "amended",
    ("amended", "approve"): "approved",
    ("amended", "reject"): "rejected",
    ("amended", "amend"): "amended",
    ("amended", "resubmit"): "draft",
    ("pending", "approve"): "approved",
    ("pending", "reject"): "rejected",
    ("pending", "amend"): "amended",
    ("pending", "resubmit"): "draft",
}

_TERMINAL = {"approved", "rejected", "published"}

_ACTION_NAMES = {
    "approve": "approve",
    "reject": "reject",
    "amend": "amend_request",
    "resubmit": "resubmit",
}


class ApprovalLedger:
    """Append-only approval decision log + transition guard.

    Wraps the existing ``drafts`` table: transitions are validated before the
    status UPDATE is applied, and every decision is written to
    ``approval_decisions`` in the same transaction.
    """

    def __init__(self, db_path: Optional[str] = None, approver_ids: Optional[list[str]] = None):
        self.db_path = db_path or _default_db_path()
        self.approver_ids = frozenset(approver_ids) if approver_ids else None

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _check_actor(self, actor: str) -> None:
        if not actor:
            raise ApprovalError("actor is required for approval actions")
        if self.approver_ids and actor not in self.approver_ids:
            raise ApprovalError(
                f"actor {actor!r} is not an allowed approver "
                f"(allowed: {sorted(self.approver_ids)})"
            )

    def _current(self, conn: sqlite3.Connection, draft_id: str) -> Optional[sqlite3.Row]:
        return conn.execute(
            "SELECT id, status, brand, platform FROM drafts WHERE id = ?",
            (draft_id,),
        ).fetchone()

    def _record(
        self,
        conn: sqlite3.Connection,
        draft_id: str,
        brand: str,
        platform: str,
        action: str,
        actor: str,
        comment: str,
        version: int,
    ) -> None:
        conn.execute(
            """
            INSERT INTO approval_decisions
              (id, draft_id, brand, platform, action, actor, comment, version, decided_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                uuid.uuid4().hex,
                draft_id,
                brand,
                platform,
                _ACTION_NAMES[action],
                actor,
                comment,
                version,
                _utc_now(),
            ),
        )

    def decide(
        self,
        draft_id: str,
        action: str,
        actor: str,
        comment: str = "",
    ) -> bool:
        """Apply one approval transition to a draft.

        Validates the transition, records the decision, and updates the
        drafts.status column atomically. Returns True if applied.

        Raises ApprovalError on an illegal transition or disallowed actor.
        """
        if action not in _ACTION_NAMES:
            raise ApprovalError(f"unknown approval action: {action!r}")
        self._check_actor(actor)

        conn = self._conn()
        try:
            conn.executescript(DECISION_SCHEMA)
            row = self._current(conn, draft_id)
            if row is None:
                return False
            current = row["status"]
            brand = row["brand"]
            platform = row["platform"]

            new_status = _TRANSITIONS.get((current, action))
            if new_status is None:
                if current in _TERMINAL:
                    raise ApprovalError(
                        f"draft {draft_id} is {current!r}; cannot {action}"
                    )
                raise ApprovalError(
                    f"illegal transition {current!r} -> {action!r} for draft {draft_id}"
                )

            now = _utc_now()
            if action == "approve":
                conn.execute(
                    "UPDATE drafts SET status = 'approved', approved_at = ? WHERE id = ?",
                    (now, draft_id),
                )
            elif action == "reject":
                conn.execute(
                    "UPDATE drafts SET status = 'rejected', rejected_at = ? WHERE id = ?",
                    (now, draft_id),
                )
            else:
                # amend / resubmit keep the same column set as the live handler
                conn.execute(
                    "UPDATE drafts SET status = ? WHERE id = ?",
                    (new_status, draft_id),
                )

            version = 1 + conn.execute(
                "SELECT COUNT(*) FROM approval_decisions WHERE draft_id = ? AND action = 'amend_request'",
                (draft_id,),
            ).fetchone()[0]
            self._record(conn, draft_id, brand, platform, action, actor, comment, version)
            conn.commit()
            return True
        finally:
            conn.close()

    def history(self, draft_id: str) -> list[dict]:
        """Return the append-only decision history for a draft."""
        conn = self._conn()
        try:
            conn.executescript(DECISION_SCHEMA)
            rows = conn.execute(
                """SELECT action, actor, comment, version, decided_at
                   FROM approval_decisions WHERE draft_id = ?
                   ORDER BY decided_at ASC""",
                (draft_id,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def current_version(self, draft_id: str) -> int:
        """Return the draft's current approval version (1 + amend count)."""
        history = self.history(draft_id)
        return 1 + sum(1 for d in history if d["action"] == "amend_request")

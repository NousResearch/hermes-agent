"""Explicit delivery-completion labels and the watchdog alert table DDL.

Split out of ``hermes_cli.kanban_db``; this module has no import-time
dependency on it (``_kb`` is reached lazily) so the facade can re-export it.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Optional

# Delivery evidence is deliberately separate from lifecycle status.  A task may
# be done as an implementation handoff without claiming that code is live; an
# aggregate request opts into the verified gate explicitly.
COMPLETION_STATES = {"written", "tested", "deployed", "verified"}

WATCHDOG_ALERTS_DDL = """
-- Bounded state for the detection-only Kanban watchdog.  One row per active
-- (task, condition); resolved rows are pruned by the watchdog after retention.
-- This prevents a periodic scan from producing a notification storm.
CREATE TABLE IF NOT EXISTS kanban_watchdog_alerts (
    task_id       TEXT NOT NULL,
    kind          TEXT NOT NULL,
    first_seen_at INTEGER NOT NULL,
    last_seen_at  INTEGER NOT NULL,
    notified_at   INTEGER NOT NULL,
    resolved_at   INTEGER,
    PRIMARY KEY (task_id, kind)
);
CREATE INDEX IF NOT EXISTS idx_kanban_watchdog_resolved
    ON kanban_watchdog_alerts(resolved_at);
"""


def normalize_verification_owner(
    conn: sqlite3.Connection, owner_id: Optional[str], parents: tuple[str, ...],
) -> Optional[str]:
    """Strip ``owner_id`` and require it to name an existing task (or a parent)."""
    owner_id = str(owner_id).strip() if owner_id else None
    if owner_id and owner_id not in parents:
        row = conn.execute("SELECT 1 FROM tasks WHERE id = ?", (owner_id,)).fetchone()
        if row is None:
            raise ValueError(f"verification_owner_id does not exist: {owner_id}")
    return owner_id


def record_completion_state(
    conn: sqlite3.Connection, task_id: str, state: str, evidence: dict,
) -> bool:
    """Record one explicit delivery label and its proof.

    This is intentionally not a state machine that infers or fills predecessor
    labels: callers must assert each label they mean.  The immutable event log
    retains every assertion while ``tasks`` holds only the current label for
    dashboard/list queries.  ``verified`` is accepted only with a textual proof
    so an empty metadata object cannot masquerade as live observation.
    """
    from hermes_cli import kanban_db as _kb

    normalized = str(state or "").strip().lower()
    if normalized not in COMPLETION_STATES:
        raise ValueError(f"completion state must be one of {sorted(COMPLETION_STATES)}")
    if not isinstance(evidence, dict) or not evidence:
        raise ValueError("completion state evidence must be a non-empty object")
    proof = evidence.get("proof")
    if not isinstance(proof, str) or not proof.strip():
        raise ValueError("completion state evidence requires a non-empty 'proof' string")
    with _kb.write_txn(conn):
        cur = conn.execute(
            "UPDATE tasks SET completion_state = ?, completion_evidence = ? WHERE id = ?",
            (normalized, json.dumps(evidence, ensure_ascii=False), task_id),
        )
        if cur.rowcount != 1:
            return False
        _kb._append_event(
            conn, task_id, "completion_state_recorded",
            {"state": normalized, "evidence": evidence},
        )
    return True

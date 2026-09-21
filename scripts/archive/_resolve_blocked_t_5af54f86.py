#!/usr/bin/env python3
"""Resolve blocked-task audit findings for t_5af54f86.

Uses the canonical kanban_db library (source of truth) directly on the
physical DB paths, because the board metadata in list_boards() is stale
(all boards point at research/kanban.db) and the CLI cannot resolve tasks
that live in the legacy root DB or the retired ops board DB.

Actions (per parent audit t_395fb37e / blocked-task-audit.md):
  1. t_5a4835af (root)   - reassign light-archivist -> octacon (role-forbidden
                            feature build), then unblock so octacon proceeds.
                            Body already has AC + Test Plan, so no contract re-block.
  2. t_7e7fb8b7 (root)   - archive (intentional test probe, served its purpose).
  3. t_9fb932ef (root)   - archive (intentional test probe).
  4-8. ops proposals      - escalate to sahil (drives daily NEEDS-YOU) and
                            schedule (status='scheduled' is invisible to the
                            blocked-unblocker pre-check, killing the auto-recovery
                            churn). Awaiting Sahil's approve/defer/spike decision.
"""
import os
import sqlite3
import time

os.environ.setdefault("HERMES_HOME", "/home/kensei/.hermes")
from hermes_cli import kanban_db as kb

HERMES = "/home/kensei/.hermes"
# W1-G (Batch 1): resolve retired board DB identities via _board_compat.
# default->core, ops->security-ops (reversible: legacy path used if present).
import _board_compat
DEFAULT_DB = _board_compat.resolve_board_db_str("default")
OPS_DB = _board_compat.resolve_board_db_str("ops")
AUTHOR = "kensei-ops-resolver"
NOW = int(time.time())

DECISION_REASON = (
    "Parked awaiting Sahil approve/defer/spike (audit t_395fb37e). "
    "Real blocker is a human decision, not the contract gate. Removed from "
    "auto-recovery churn via scheduled state; escalated to surface in NEEDS-YOU."
)


def _connect(path):
    from pathlib import Path
    # Use kanban_db.connect so isolation_level=None matches write_txn's
    # explicit BEGIN IMMEDIATE (plain sqlite3.connect opens an implicit txn
    # and raises "cannot start a transaction within a transaction").
    conn = kb.connect(db_path=Path(path))
    conn.row_factory = sqlite3.Row
    return conn


def _comment(conn, tid, body):
    kb._add_comment_inline(conn, tid, author=AUTHOR, body=body)


def _snap(conn, tid, label):
    r = conn.execute(
        "SELECT status, assignee, escalation_target, block_kind, block_recurrences "
        "FROM tasks WHERE id=?", (tid,)
    ).fetchone()
    print(f"  [{label}] {tid}: status={r['status']} assignee={r['assignee']} "
          f"escalation={r['escalation_target']} recurrences={r['block_recurrences']}")


def main():
    # ---- ROOT board (kanban.db) ----
    c = _connect(DEFAULT_DB)
    try:
        # 1. t_5a4835af reassign + unblock
        tid = "t_5a4835af"
        _snap(c, tid, "before")
        ok = kb.reassign_task(c, tid, "octacon", reclaim_first=True,
                              reason="Audit t_395fb37e: full feature build mis-assigned to "
                                     "role-forbidden light-archivist (archival-only). Body "
                                     "carries AC+TestPlan so no contract re-block.")
        print(f"  reassign t_5a4835af -> octacon: {ok}")
        if kb.unblock_task(c, tid):
            _comment(c, tid, "Unblocked: wrong-assignee cause resolved. Now owned by octacon; "
                             "dispatcher will proceed (AC+TestPlan present).")
            print("  unblock t_5a4835af: ok")
        _snap(c, tid, "after")

        # 2-3. archive test probes
        for tid in ("t_7e7fb8b7", "t_9fb932ef"):
            _snap(c, tid, "before")
            if kb.archive_task(c, tid):
                _comment(c, tid, "Archived: intentional triage->blocked verification probe "
                                 "(audit t_395fb37e). Served its purpose; removed from churn.")
                print(f"  archive {tid}: ok")
            _snap(c, tid, "after")
    finally:
        c.close()

    # ---- OPS board (resolved via _board_compat; retired ops -> security-ops) ----
    c = _connect(OPS_DB)
    try:
        for tid in ("t_5f038f9f", "t_6c12d09c", "t_a17972a3", "t_eea200d7", "t_96031361"):
            _snap(c, tid, "before")
            res = kb.set_escalation_target(c, tid, target="sahil", author=AUTHOR)
            print(f"  escalate {tid} -> sahil: {res}")
            if kb.schedule_task(c, tid, reason=DECISION_REASON):
                _comment(c, tid, "ESCALATE: " + DECISION_REASON)
                print(f"  schedule {tid}: ok")
            _snap(c, tid, "after")
    finally:
        c.close()

    print("\nDONE. Recommend follow-up: add terminal 'decision-needed' state to stop "
          "auto-recovery churn systemically (see blocked-task-audit.md META-FINDING).")


if __name__ == "__main__":
    main()

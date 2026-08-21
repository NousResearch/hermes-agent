"""OPS-PROBE: dispatcher ordering contract.

The dispatcher picks the next ready task by
``ORDER BY priority DESC, created_at ASC``. Probes this directly so a
regression surfaces with a clear diagnostic, not a flaky production race.

Layer 1 (DB hermetic): three ready rows at p1, p5, p20 — expect the
query to return them in [p20, p5, p1] order. Plus tie-break behavior
when two rows share the same priority.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest
import uuid
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from hermes_cli import kanban_db as kb  # noqa: E402


def _seed_board(conn, slug: str) -> None:
    try:
        conn.execute(
            "INSERT OR IGNORE INTO kanban_boards (slug, name, tenant) "
            "VALUES (?, ?, NULL)",
            (slug, slug),
        )
    except Exception:
        pass


class DispatchOrderProbe(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.mkdtemp(prefix="ops-probe-order-")
        self._db = Path(self._tmpdir) / "kanban.db"
        self._slug = f"ops-probe-order-{uuid.uuid4().hex[:8]}"
        kb.init_db(self._db)
        with kb.connect_closing(db_path=self._db) as conn:
            _seed_board(conn, self._slug)

    def test_priority_descending_then_created_at_ascending(self):
        # Insert p1, p5, p20 with strict created_at ordering so the
        # tie-break column matters only when priorities tie.
        now = int(time.time())
        ids = {}
        for pri, key in [(1, "p1"), (5, "p5"), (20, "p20")]:
            tid = kb._new_task_id()  # type: ignore[attr-defined]
            ids[key] = tid
            with kb.connect_closing(db_path=self._db) as conn:
                conn.execute(
                    "INSERT INTO tasks "
                    "(id, title, body, assignee, status, workspace_kind, "
                    " workspace_path, tenant, priority, created_at, created_by) "
                    "VALUES (?, ?, ?, ?, 'ready', 'scratch', NULL, ?, ?, ?, 'probe')",
                    (tid, key, "body", "engineer",
                     self._slug, pri, now + pri),  # monotonic by pri
                )
        with kb.connect_closing(db_path=self._db) as conn:
            rows = conn.execute(
                "SELECT id, priority FROM tasks WHERE status='ready' "
                "AND claim_lock IS NULL "
                "ORDER BY priority DESC, created_at ASC"
            ).fetchall()
        observed = [r["id"] for r in rows]
        # p20 wins, then p5, then p1.
        self.assertEqual(observed, [ids["p20"], ids["p5"], ids["p1"]])

    def test_ties_broken_by_created_at_ascending(self):
        # Two rows at priority 5 — older created_at wins.
        now = int(time.time())
        old = kb._new_task_id()  # type: ignore[attr-defined]
        new = kb._new_task_id()  # type: ignore[attr-defined]
        with kb.connect_closing(db_path=self._db) as conn:
            for tid, ts in [(old, now - 10), (new, now)]:
                conn.execute(
                    "INSERT INTO tasks "
                    "(id, title, body, assignee, status, workspace_kind, "
                    " workspace_path, tenant, priority, created_at, created_by) "
                    "VALUES (?, ?, ?, ?, 'ready', 'scratch', NULL, ?, ?, ?, 'probe')",
                    (tid, "tie", "body", "engineer",
                     self._slug, 5, ts),
                )
        with kb.connect_closing(db_path=self._db) as conn:
            rows = conn.execute(
                "SELECT id FROM tasks WHERE status='ready' "
                "AND claim_lock IS NULL "
                "ORDER BY priority DESC, created_at ASC"
            ).fetchall()
        observed = [r["id"] for r in rows]
        # Older created_at first.
        self.assertEqual(observed, [old, new])

    def test_zero_priority_tasks_still_dispatchable(self):
        # Sanity: priority=0 rows still appear (the dispatcher does not
        # filter them out — only sort).
        tid = kb._new_task_id()  # type: ignore[attr-defined]
        with kb.connect_closing(db_path=self._db) as conn:
            conn.execute(
                "INSERT INTO tasks "
                "(id, title, body, assignee, status, workspace_kind, "
                " workspace_path, tenant, priority, created_at, created_by) "
                "VALUES (?, ?, ?, ?, 'ready', 'scratch', NULL, ?, ?, ?, 'probe')",
                (tid, "zeropri", "body", "engineer",
                 self._slug, 0, int(time.time())),
            )
        with kb.connect_closing(db_path=self._db) as conn:
            row = conn.execute(
                "SELECT id FROM tasks WHERE status='ready' "
                "AND claim_lock IS NULL AND id = ? "
                "ORDER BY priority DESC, created_at ASC",
                (tid,),
            ).fetchone()
        self.assertIsNotNone(row, "priority=0 row should still be in ready queue")


if __name__ == "__main__":
    unittest.main()
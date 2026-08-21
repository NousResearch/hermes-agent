"""Acceptance tests for parent-priority inheritance and the ``--priority``
flag on ``kanban decompose``. Drives the DB layer directly so the test
is hermetic — no LLM, no watcher tick, no dashboard.

Acceptance contract (from the task body):

  - Parent priority p10, no ``--priority`` flag         -> every child priority p10
  - Parent priority p10, ``--priority p20`` flag        -> every child priority p20
  - Parent priority 0                                   -> children priority 0
  - Per-child override from the decomposer model       -> wins over both
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

# Force a fresh kanban DB per test so each run is hermetic.
_TMPDIR = tempfile.mkdtemp(prefix="kanban-priority-")
os.environ["HERMES_KANBAN_DB"] = os.path.join(_TMPDIR, "kanban.db")
os.environ["HERMES_KANBAN_BOARD"] = "default-test-board"

# Repo root on path so `hermes_cli` resolves cleanly under `python -m pytest`.
_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from hermes_cli import kanban_db as kb  # noqa: E402


def _seed_triage(conn, *, priority: int) -> str:
    """Create one triage task owned by a throwaway board anchor.

    The dispatcher doesn't run in these tests, so we just INSERT a row
    in ``tasks`` with the minimum shape ``decompose_triage_task``
    expects (status='triage', a tenant). Auto-id allocation via
    ``_new_task_id`` is public through ``kb.create_task`` but that one
    forces status='todo'; we want triage, so write the row directly.
    """
    new_id = kb._new_task_id()  # type: ignore[attr-defined]
    import time
    now = int(time.time())
    conn.execute(
        "INSERT INTO tasks "
        "(id, title, body, assignee, status, workspace_kind, "
        " workspace_path, tenant, priority, created_at, created_by) "
        "VALUES (?, ?, ?, ?, 'triage', 'scratch', NULL, ?, ?, ?, 'test')",
        (new_id, "Parent", "body", None, "default-test-board", priority, now),
    )
    return new_id


def _seed_board(conn, slug: str) -> None:
    """Ensure the board row exists (the DB init does this lazily on
    connect, but our connect path may not run before the test reads
    tasks; a direct INSERT is enough)."""
    try:
        conn.execute(
            "INSERT OR IGNORE INTO kanban_boards (slug, name, tenant) "
            "VALUES (?, ?, NULL)",
            (slug, slug),
        )
    except Exception:
        # Table doesn't exist yet; let kb.init_db() create it later.
        pass


class PriorityInheritanceTests(unittest.TestCase):
    def setUp(self) -> None:
        kb.init_db(Path(_TMPDIR) / "kanban.db")
        # ``init_db`` is idempotent; reopen a fresh connection that
        # reads/writes the seeded board.
        with kb.connect_closing() as conn:
            _seed_board(conn, "default-test-board")

    def _children(self, parent_id: str) -> dict[str, int]:
        """Return {child_id: priority} for every direct child of parent_id.

        ``decompose_triage_task`` links the root as a *child* of every
        leaf child (root waits for the whole graph), so direct children
        of the root are rows in ``task_links`` where ``child_id`` is
        the root id.
        """
        with kb.connect_closing() as conn:
            rows = conn.execute(
                "SELECT c.id, c.priority "
                "FROM tasks c "
                "JOIN task_links t ON t.parent_id = c.id "
                "WHERE t.child_id = ?",
                (parent_id,),
            ).fetchall()
        return {r["id"]: r["priority"] for r in rows}

    def test_inherits_parent_priority_when_flag_omitted(self) -> None:
        with kb.connect_closing() as conn:
            parent = _seed_triage(conn, priority=10)
            child_ids = kb.decompose_triage_task(
                conn, parent,
                root_assignee=None,
                children=[
                    {"title": "A"},
                    {"title": "B"},
                    {"title": "C"},
                ],
            )
        self.assertIsNotNone(child_ids)
        priorities = self._children(parent)
        self.assertEqual(set(priorities), set(child_ids))
        self.assertTrue(
            all(p == 10 for p in priorities.values()),
            f"expected all children priority=10, got {priorities}",
        )

    def test_flag_override_applies_to_every_child(self) -> None:
        with kb.connect_closing() as conn:
            parent = _seed_triage(conn, priority=10)
            child_ids = kb.decompose_triage_task(
                conn, parent,
                root_assignee=None,
                children=[
                    {"title": "A"},
                    {"title": "B"},
                ],
                child_priority=20,
            )
        self.assertIsNotNone(child_ids)
        priorities = self._children(parent)
        self.assertTrue(
            all(p == 20 for p in priorities.values()),
            f"expected all children priority=20, got {priorities}",
        )

    def test_parent_priority_zero_propagates_zero(self) -> None:
        """A parent at the schema default 0 must produce children at 0
        — no spurious re-defaulting."""
        with kb.connect_closing() as conn:
            parent = _seed_triage(conn, priority=0)
            child_ids = kb.decompose_triage_task(
                conn, parent,
                root_assignee=None,
                children=[{"title": "Only"}],
            )
        self.assertIsNotNone(child_ids)
        priorities = self._children(parent)
        self.assertTrue(
            all(p == 0 for p in priorities.values()),
            f"expected all children priority=0, got {priorities}",
        )

    def test_per_child_priority_override_wins(self) -> None:
        """Per-child ``priority`` from the decomposer beats the
        effective-child-priority default (root or flag)."""
        with kb.connect_closing() as conn:
            parent = _seed_triage(conn, priority=10)
            child_ids = kb.decompose_triage_task(
                conn, parent,
                root_assignee=None,
                children=[
                    {"title": "A"},                     # inherits -> 10
                    {"title": "B", "priority": 99},     # explicit
                    {"title": "C"},                     # inherits -> 10
                ],
                child_priority=20,
            )
        self.assertIsNotNone(child_ids)
        # Direct INSERT-by-title lookup so we can assert per-child IDs
        # without coupling to insertion order.
        with kb.connect_closing() as conn:
            rows = conn.execute(
                "SELECT title, priority FROM tasks WHERE id IN ("
                + ",".join("?" * len(child_ids))
                + ") ORDER BY title",
                list(child_ids),
            ).fetchall()
        by_title = {r["title"]: r["priority"] for r in rows}
        self.assertEqual(by_title["A"], 20)
        self.assertEqual(by_title["B"], 99)
        self.assertEqual(by_title["C"], 20)


if __name__ == "__main__":
    unittest.main()

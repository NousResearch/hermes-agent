"""Tests for typed block reasons + the unblock-loop breaker.

Covers the built-in fix for the kanban "blocked loop" — a worker blocks a
task, a cron unblocks it, the worker re-blocks for the same reason, repeat
forever. The fix gives ``block_task`` a typed ``kind`` and a persistent
``block_recurrences`` counter:

* ``dependency`` blocks route to ``todo`` (parent-gated, auto-resumed) and
  never enter the human ``blocked`` bucket a cron would keep unblocking.
* ``needs_input`` / ``capability`` / un-typed blocks land in ``blocked``;
  each same-cause re-block after an unblock increments ``block_recurrences``,
  and at ``BLOCK_RECURRENCE_LIMIT`` the task routes to ``triage`` for a human.
* ``unblock_task`` deliberately does NOT reset ``block_recurrences`` (the
  amnesia that let the loop run unbounded).
* A successful ``complete_task`` resets the loop memory.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _running_task(conn, title="t"):
    """Create a task and drive it to ``running`` so block_task can act."""
    tid = kb.create_task(conn, title=title, assignee="worker")
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
    claimed = kb.claim_task(conn, tid, claimer="worker")
    assert claimed is not None
    return tid


def _make_running_again(conn, tid):
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
    assert kb.claim_task(conn, tid, claimer="worker") is not None


# ---------------------------------------------------------------------------
# Loop breaker
# ---------------------------------------------------------------------------










def test_block_loop_detected_event_emitted(kanban_home: Path) -> None:
    with kbc.connect_closing() as conn:
        tid = _running_task(conn)
        kb.block_task(conn, tid, reason="x", kind="capability")
        kb.unblock_task(conn, tid)
        _make_running_again(conn, tid)
        kb.block_task(conn, tid, reason="x", kind="capability")
        events = [e for e in kb.list_events(conn, tid)
                  if e.kind == "block_loop_detected"]
        assert events, "expected a block_loop_detected event"
        payload = events[-1].payload or {}
        assert payload.get("recurrences") == 2
        assert payload.get("kind") == "capability"


def test_block_recurrence_limit_zero_never_routes_to_triage(kanban_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``kanban.block_recurrence_limit: 0`` keeps every same-cause re-block in ``blocked``."""
    monkeypatch.setattr(kb, "block_recurrence_limit", lambda: 0)
    with kbc.connect_closing() as conn:
        tid = _running_task(conn)
        for _ in range(5):
            kb.block_task(conn, tid, reason="x", kind="needs_input")
            assert kb.get_task(conn, tid).status == "blocked"
            kb.unblock_task(conn, tid)
            _make_running_again(conn, tid)
        kb.block_task(conn, tid, reason="x", kind="needs_input")
        task = kb.get_task(conn, tid)
        assert task.status == "blocked"
        assert task.block_recurrences == 6
        assert not [e for e in kb.list_events(conn, tid) if e.kind == "block_loop_detected"]


def test_block_recurrence_limit_configured_value(kanban_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A configured limit replaces the default of 2."""
    monkeypatch.setattr(kb, "block_recurrence_limit", lambda: 3)
    with kbc.connect_closing() as conn:
        tid = _running_task(conn)
        kb.block_task(conn, tid, reason="x", kind="capability")
        kb.unblock_task(conn, tid)
        _make_running_again(conn, tid)
        kb.block_task(conn, tid, reason="x", kind="capability")
        assert kb.get_task(conn, tid).status == "blocked"
        kb.unblock_task(conn, tid)
        _make_running_again(conn, tid)
        kb.block_task(conn, tid, reason="x", kind="capability")
        assert kb.get_task(conn, tid).status == "triage"
        payload = [e for e in kb.list_events(conn, tid) if e.kind == "block_loop_detected"][-1].payload or {}
        assert payload.get("limit") == 3


def test_block_recurrence_limit_reads_config(kanban_home: Path) -> None:
    """The helper reads ``kanban.block_recurrence_limit`` and falls back to the default."""
    for text, expected in (("kanban:\n  block_recurrence_limit: 0\n", 0),
                           ("kanban:\n  block_recurrence_limit: 7\n", 7),
                           ("kanban: {}\n", kb.BLOCK_RECURRENCE_LIMIT),
                           ("kanban:\n  block_recurrence_limit: null\n", kb.BLOCK_RECURRENCE_LIMIT)):
        (kanban_home / "config.yaml").write_text(text)
        assert kb.block_recurrence_limit() == expected


# ---------------------------------------------------------------------------
# Dependency routing
# ---------------------------------------------------------------------------


def test_dependency_then_parent_done_promotes(kanban_home: Path) -> None:
    """A dependency-parked child becomes ready once its parent completes."""
    with kbc.connect_closing() as conn:
        parent = kb.create_task(conn, title="parent", assignee="worker")
        child = _running_task(conn, title="child")
        kb.link_tasks(conn, parent_id=parent, child_id=child)
        kb.block_task(conn, child, reason="wait", kind="dependency")
        assert kb.get_task(conn, child).status == "todo"
        # Finish the parent, then let recompute_ready run.
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (parent,))
        kb.claim_task(conn, parent, claimer="worker")
        kb.complete_task(conn, parent, result="done")
        kb.recompute_ready(conn)
        assert kb.get_task(conn, child).status == "ready"


# ---------------------------------------------------------------------------
# Completion resets loop memory
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Validation + back-compat
# ---------------------------------------------------------------------------



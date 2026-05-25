"""Stress regression: complete_task + recompute_ready must commit atomically.

The historical bug: complete_task ran two SEPARATE write_txn blocks — one for
the parent UPDATE + `completed` event, then a second (via recompute_ready)
for child status='ready' UPDATEs. The inter-transaction gap was a window
in which a WAL auto-checkpoint could partially flush — moving the `tasks`
table page to main-db while leaving `idx_tasks_status` pages in WAL. If the
checkpoint was then interrupted, the index drifted from the table.

These tests assert the atomic-merge property: one COMMIT, no gap, no drift.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def db_path(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    p = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(p.resolve()))
    kb.init_db()
    return p


def _build_parent_with_children(conn, n_children: int = 3):
    parent = kb.create_task(conn, title="parent", assignee="setup")
    children = [
        kb.create_task(conn, title=f"child{i}", parents=[parent], assignee="setup")
        for i in range(n_children)
    ]
    # Bring parent to a completable state.
    conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (parent,))
    return parent, children


def test_complete_task_emits_single_begin_immediate(db_path):
    """complete_task + child promotion must produce exactly ONE BEGIN IMMEDIATE."""
    with kb.connect() as conn:
        parent, children = _build_parent_with_children(conn, n_children=3)
        begins = []

        def trace(stmt):
            if "BEGIN IMMEDIATE" in stmt.upper():
                begins.append(stmt)

        conn.set_trace_callback(trace)
        try:
            ok = kb.complete_task(conn, parent, summary="done")
        finally:
            conn.set_trace_callback(None)
        assert ok is True
        # Exactly one BEGIN IMMEDIATE for the merged complete+promote txn.
        # (Pre-fix: 2 — one for complete_task, one for recompute_ready.)
        assert len(begins) == 1, (
            f"expected 1 BEGIN IMMEDIATE, got {len(begins)}: {begins}"
        )


def test_complete_task_promotes_children_in_same_txn(db_path):
    """After complete_task, children are 'ready' and integrity_check passes."""
    with kb.connect() as conn:
        parent, children = _build_parent_with_children(conn, n_children=3)
        ok = kb.complete_task(conn, parent, summary="done")
        assert ok
        for c in children:
            assert kb.get_task(conn, c).status == "ready"
        row = conn.execute("PRAGMA integrity_check").fetchone()
        assert row[0] == "ok"


def test_repeated_complete_recompute_no_drift_under_load(db_path):
    """50 parent completions in a row — integrity_check must remain clean."""
    with kb.connect() as conn:
        for i in range(50):
            parent, _ = _build_parent_with_children(conn, n_children=2)
            assert kb.complete_task(conn, parent, summary=f"batch{i}")
        row = conn.execute("PRAGMA integrity_check").fetchone()
        assert row[0] == "ok"


def test_two_connection_index_stability(db_path):
    """Concurrent reader sees no drift after writer completes a parent."""
    with kb.connect() as writer:
        for _ in range(20):
            parent, _ = _build_parent_with_children(writer, n_children=2)
            assert kb.complete_task(writer, parent, summary="x")
            with kb.connect() as reader:
                row = reader.execute("PRAGMA integrity_check").fetchone()
                assert row[0] == "ok"

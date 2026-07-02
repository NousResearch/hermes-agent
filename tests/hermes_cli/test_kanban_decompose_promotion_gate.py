"""Tests for the manual-promote gate on decomposed entry children.

When ``decompose_triage_task(..., auto_promote=False)`` fans a triage task
out, its parent-free entry children would otherwise be trivially re-promoted
to ``ready`` by the very next ``recompute_ready`` tick (all-parents-done is
vacuously true for a parent-free task). A persistent ``promotion_gated``
event holds them in ``todo`` until an explicit operator ``promote_task``
(which emits ``promoted_manual``) releases them.

These tests also lock the inert-for-existing-workflows property: nothing
gets a ``promotion_gated`` event under the default auto-promote path, and a
plain parent-free top-level todo still promotes on ``recompute_ready``.

LLM-free by design.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _create_triage(conn, title="rough idea", body=None, assignee=None, tenant=None):
    return kb.create_task(
        conn,
        title=title,
        body=body,
        assignee=assignee,
        tenant=tenant,
        triage=True,
    )


def _event_kinds(conn, task_id):
    return [ev.kind for ev in kb.list_events(conn, task_id)]


# ---------------------------------------------------------------------------
# 1. Gate holds the entry child in todo across a dispatcher tick.
# ---------------------------------------------------------------------------
def test_manual_promote_gate_holds_entry_child_in_todo(kanban_home):
    with kb.connect() as conn:
        tid = _create_triage(conn, title="ship a feature")

    children = [
        {"title": "research", "assignee": "researcher", "parents": []},
        {"title": "build it", "assignee": "engineer", "parents": [0]},
    ]
    with kb.connect() as conn:
        child_ids = kb.decompose_triage_task(
            conn,
            tid,
            root_assignee="orchestrator",
            children=children,
            author="decomposer",
            auto_promote=False,
        )
    assert child_ids is not None and len(child_ids) == 2
    entry_id, dep_id = child_ids

    # Parent-free entry child carries a promotion_gated event and stays todo.
    with kb.connect() as conn:
        assert "promotion_gated" in _event_kinds(conn, entry_id)
        # The dependent child (has sibling parent) is NOT gated — it is
        # already held by its unfinished dependency.
        assert "promotion_gated" not in _event_kinds(conn, dep_id)
        assert kb.get_task(conn, entry_id).status == "todo"
        assert kb.get_task(conn, dep_id).status == "todo"

    # Simulate several dispatcher ticks — the gate must survive all of them.
    with kb.connect() as conn:
        for _ in range(3):
            kb.recompute_ready(conn)
        assert kb.get_task(conn, entry_id).status == "todo"
        assert kb.get_task(conn, dep_id).status == "todo"


# ---------------------------------------------------------------------------
# 2. Manual promotion releases the gate; the cascade then works normally.
# ---------------------------------------------------------------------------
def test_manual_promote_releases_gate_and_cascade_works(kanban_home):
    with kb.connect() as conn:
        tid = _create_triage(conn, title="two-step plan")

    with kb.connect() as conn:
        child_ids = kb.decompose_triage_task(
            conn,
            tid,
            root_assignee="orchestrator",
            children=[
                {"title": "entry", "assignee": "researcher", "parents": []},
                {"title": "dependent", "assignee": "engineer", "parents": [0]},
            ],
            author="decomposer",
            auto_promote=False,
        )
    entry_id, dep_id = child_ids

    # Operator promotes the entry explicitly.
    with kb.connect() as conn:
        ok, err = kb.promote_task(conn, entry_id, actor="op")
        assert ok, err
        assert kb.get_task(conn, entry_id).status == "ready"

    # A subsequent dispatcher tick must NOT re-gate the freshly promoted
    # entry (promoted_manual is the most-recent gate-state event now).
    with kb.connect() as conn:
        kb.recompute_ready(conn)
        assert kb.get_task(conn, entry_id).status == "ready"
        # Its dependent is still blocked by the not-yet-done entry.
        assert kb.get_task(conn, dep_id).status == "todo"

    # Complete the entry; the dependent then promotes normally on the tick.
    with kb.connect() as conn:
        assert kb.complete_task(conn, entry_id)
        kb.recompute_ready(conn)
        assert kb.get_task(conn, entry_id).status == "done"
        assert kb.get_task(conn, dep_id).status == "ready"


# ---------------------------------------------------------------------------
# 3. Regression guard: auto_promote=True emits NO gate; entries promote.
# ---------------------------------------------------------------------------
def test_auto_promote_true_never_gates(kanban_home):
    with kb.connect() as conn:
        tid = _create_triage(conn, title="default fan-out")

    with kb.connect() as conn:
        child_ids = kb.decompose_triage_task(
            conn,
            tid,
            root_assignee="orchestrator",
            children=[
                {"title": "entry-a", "parents": []},
                {"title": "entry-b", "parents": []},
                {"title": "dependent", "parents": [0]},
            ],
            author="decomposer",
            auto_promote=True,
        )
    entry_a, entry_b, dep = child_ids

    with kb.connect() as conn:
        # No promotion_gated event on ANY child.
        for cid in child_ids:
            assert "promotion_gated" not in _event_kinds(conn, cid)
        # Parent-free entries promoted to ready exactly as before.
        assert kb.get_task(conn, entry_a).status == "ready"
        assert kb.get_task(conn, entry_b).status == "ready"
        # The dependent stays todo until its parent completes.
        assert kb.get_task(conn, dep).status == "todo"


# ---------------------------------------------------------------------------
# 4. Inert-for-existing-workflows: a plain parent-free todo still promotes.
# ---------------------------------------------------------------------------
def test_plain_todo_without_gate_still_promotes(kanban_home):
    # Inert-for-existing-workflows: a parent-free task sitting in 'todo'
    # with NO promotion_gated event must still be auto-promoted by
    # recompute_ready, exactly as before this change. We force a genuine
    # gate-less todo (create_task lands parent-free tasks in 'ready', so we
    # push it back to 'todo' without emitting any gate event) and confirm
    # the legacy auto-promote path is byte-identical.
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="ordinary task")
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status = 'todo' WHERE id = ?", (tid,)
            )
        assert kb.get_task(conn, tid).status == "todo"
        # No gate event exists for it (the helper's False common case).
        assert "promotion_gated" not in _event_kinds(conn, tid)
        assert kb._awaiting_manual_promotion(conn, tid) is False

    with kb.connect() as conn:
        kb.recompute_ready(conn)
        # Legacy auto-promote path is unchanged: it becomes ready.
        assert kb.get_task(conn, tid).status == "ready"

"""Tests for kb.decompose_triage_task — the DB-layer atomic fan-out
from the triage column. LLM-free by design.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_workspace as kbw


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


def test_decompose_creates_children_and_promotes_root(kanban_home):
    with kbc.connect() as conn:
        tid = _create_triage(conn, title="ship a feature")
        assert kb.get_task(conn, tid).status == "triage"

    children = [
        {"title": "research", "body": "look at prior art", "assignee": "researcher", "parents": []},
        {"title": "build it", "body": "write code", "assignee": "engineer", "parents": [0]},
    ]
    with kbc.connect() as conn:
        child_ids = kb.decompose_triage_task(
            conn,
            tid,
            root_assignee="orchestrator",
            children=children,
            author="decomposer",
        )
    assert child_ids is not None
    assert len(child_ids) == 2

    with kbc.connect() as conn:
        root = kb.get_task(conn, tid)
        c0 = kb.get_task(conn, child_ids[0])
        c1 = kb.get_task(conn, child_ids[1])

    # Root flipped to todo with orchestrator assignee, gated by children.
    assert root.status == "todo"
    assert root.assignee == "orchestrator"
    # First child has no internal parents → ready on recompute_ready.
    assert c0.status == "ready"
    assert c0.assignee == "researcher"
    # Second child has parents=[0] → stays in todo until c0 completes.
    assert c1.status == "todo"
    assert c1.assignee == "engineer"


def test_decompose_records_audit_comment_and_event(kanban_home):
    with kbc.connect() as conn:
        tid = _create_triage(conn)
        child_ids = kb.decompose_triage_task(
            conn,
            tid,
            root_assignee="orch",
            children=[{"title": "task A", "assignee": "researcher"}],
            author="alice",
        )
    assert child_ids is not None

    with kbc.connect() as conn:
        comments = kb.list_comments(conn, tid)
        events = kb.list_events(conn, tid)

    assert any("Decomposed into" in (c.body or "") for c in comments)
    assert any(ev.kind == "decomposed" for ev in events)


def test_decompose_scratch_children_do_not_inherit_claimed_root_path(kanban_home):
    """Regression: a claimed scratch root has a persisted workspace_path
    (dispatch persists the resolved dir on claim). A fan-out must NOT copy
    that literal path onto scratch children — siblings would all run inside
    the parent's directory. Children stay unset so dispatch materializes a
    fresh ``<workspaces_root>/<child-id>`` per child. See #103303."""
    with kbc.connect() as conn:
        tid = _create_triage(conn, title="fan out research")
        # Simulate a root that has already been claimed by the dispatcher.
        kbw.set_workspace_path(conn, tid, str(kanban_home / "kanban" / "workspaces" / tid))
        child_ids = kb.decompose_triage_task(
            conn,
            tid,
            root_assignee="orchestrator",
            children=[
                {"title": "task A", "assignee": "researcher"},
                {"title": "task B", "assignee": "researcher"},
            ],
            author="decomposer",
        )
    assert child_ids is not None
    assert len(child_ids) == 2
    with kbc.connect() as conn:
        paths = [
            conn.execute("SELECT workspace_path FROM tasks WHERE id = ?", (cid,)).fetchone()[0]
            for cid in child_ids
        ]
    # Each child is unset -> dispatch derives workspaces/<child-id> per child.
    assert paths == [None, None]


def test_decompose_dir_children_keep_inheriting_shared_root_path(kanban_home):
    """'dir' is an explicitly shared persistent checkout: children of a dir
    root inherit its path when kinds match. Guards against over-fixing the
    scratch isolation above (shared checkout is the point of 'dir')."""
    shared = str(kanban_home / "boards" / "default" / "shared-proj")
    with kbc.connect() as conn:
        tid = _create_triage(conn, title="dir fan out")
        conn.execute("UPDATE tasks SET workspace_kind='dir' WHERE id = ?", (tid,))
        kbw.set_workspace_path(conn, tid, shared)
        child_ids = kb.decompose_triage_task(
            conn,
            tid,
            root_assignee="orchestrator",
            children=[{"title": "task A", "assignee": "researcher"}],
            author="decomposer",
        )
    assert child_ids is not None
    with kbc.connect() as conn:
        path = conn.execute(
            "SELECT workspace_path FROM tasks WHERE id = ?", (child_ids[0],)
        ).fetchone()[0]
    assert path == shared



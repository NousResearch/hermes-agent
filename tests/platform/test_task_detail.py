"""Task detail: the runtime's own attempts, notes and artifacts, tenant-scoped.

The endpoint existed and returned the bare task row. Everything the runtime already
records about *how* the work went — the attempts, the notes people left, the files it
produced — was durable and unreachable.
"""

from __future__ import annotations

import json

import pytest

from nova.control.auth import Principal
from nova.runtime.base import TaskDetail, TaskView
from nova.runtime.hermes import work


def _seed(home, *, tenant="acme"):
    """A real board written through the runtime's own API, not hand-rolled SQL."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    conn = kbc.connect()
    task_id = kb.create_task(conn, title="Pull ledger", tenant=tenant, assignee="ops")
    kb.add_comment(conn, task_id, "alice", "use the Q3 export")
    kb.claim_task(conn, task_id)
    kb.complete_task(conn, task_id, result="done", summary="42 rows")
    blob = home / "out.csv"
    blob.write_text("a,b\n")
    kb.add_attachment(
        conn, task_id, filename="out.csv", stored_path=str(blob),
        content_type="text/csv", size=4, uploaded_by="worker",
    )
    conn.commit()
    conn.close()
    return task_id, blob


@pytest.fixture
def board(tmp_path, monkeypatch):
    from hermes_cli import kanban_db as kb

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    kb.init_db()
    return home


def test_detail_carries_runs_notes_and_artifacts(board):
    task_id, _ = _seed(board)
    detail = work.task_detail(board, task_id, tenant_id="acme")

    assert detail is not None
    assert [(r.status, r.outcome, r.summary) for r in detail.runs] == [
        ("done", "completed", "42 rows")
    ]
    assert [(n.author, n.body) for n in detail.notes] == [("alice", "use the Q3 export")]
    assert [(a.filename, a.size_bytes) for a in detail.artifacts] == [("out.csv", 4)]


def test_artifact_paths_never_leave_the_adapter(board):
    """``stored_path`` is an absolute host path: useless to a browser, useful to an
    attacker. It must not appear in any serialised form."""
    task_id, blob = _seed(board)
    payload = json.dumps(work.task_detail(board, task_id, tenant_id="acme").to_dict())

    assert "stored_path" not in payload
    assert str(blob) not in payload
    assert str(board) not in payload


def test_detail_is_tenant_scoped(board):
    task_id, _ = _seed(board, tenant="acme")
    assert work.task_detail(board, task_id, tenant_id="globex") is None
    assert work.task_detail(board, task_id, tenant_id="acme") is not None


def test_task_edges_come_from_the_durable_graph(board):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    conn = kbc.connect()
    parent = kb.create_task(conn, title="Objective", tenant="acme", assignee="ops")
    child = kb.create_task(conn, title="Step", tenant="acme", assignee="ops", parents=[parent])
    conn.commit()
    conn.close()

    assert work.task_detail(board, child, tenant_id="acme").depends_on == (parent,)
    assert work.task_detail(board, parent, tenant_id="acme").blocks == (child,)


def test_missing_task_is_none(board):
    assert work.task_detail(board, "t_nope", tenant_id="acme") is None


# ---------------------------------------------------------------------------
# Contract default
# ---------------------------------------------------------------------------

def test_default_task_detail_degrades_to_the_task_alone():
    """A runtime with no attempt history still answers, rather than obliging every
    adapter to invent a run model."""
    from nova.runtime.base import AgentRuntime

    class Minimal:
        """Stands in for an adapter that implements get_task and nothing else.

        Not a real subclass: AgentRuntime is abstract over nine other methods, and
        stubbing them would test the stubs. The default implementation is called
        unbound, which is exactly how a subclass that does not override it gets it.
        """

        def get_task(self, task_id):
            return TaskView(task_id=task_id, title="x", state="done")

    detail = AgentRuntime.task_detail(Minimal(), "t1")
    assert isinstance(detail, TaskDetail)
    assert detail.task.task_id == "t1"
    assert detail.runs == () and detail.notes == () and detail.artifacts == ()

    class Absent(Minimal):
        def get_task(self, task_id):
            return None

    assert AgentRuntime.task_detail(Absent(), "t1") is None


# ---------------------------------------------------------------------------
# RBAC
# ---------------------------------------------------------------------------

def test_a_viewer_may_open_a_single_task():
    """``ROUTE_ROLES`` is an exact-match table, so ``/tasks`` was readable by a viewer
    while ``/tasks/<id>`` fell through to admin — work you could see but never open."""
    viewer = Principal(name="v", role="viewer", via="token")
    assert viewer.may("/tasks")
    assert viewer.may("/tasks/t_ab12")


def test_the_fail_closed_default_still_holds():
    """Only declared collection prefixes resolve; everything else stays admin-only."""
    viewer = Principal(name="v", role="viewer", via="token")
    assert not viewer.may("/tasks/t_ab12/secrets")   # deeper path is not a task read
    assert not viewer.may("/something-new")
    assert not viewer.may("/something-new/child")
    assert not viewer.may("/decisions")              # still admin, as before


def test_a_single_task_read_is_not_a_write_route():
    viewer = Principal(name="v", role="viewer", via="token")
    admin = Principal(name="a", role="admin", via="token")
    assert not viewer.may_write("/tasks/t_ab12")
    assert not admin.may_write("/tasks/t_ab12")

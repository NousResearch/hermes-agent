"""Regressions for domain-layer descendant invalidation on ancestor reopen.

``kanban_db.invalidate_descendants_for_parent_reopen`` is the single
implementation of "a done ancestor was reopened, retract everything that
assumed its result" (M3). These tests pin:

* done descendants are demoted to ``todo`` with a ``descendant_invalidated``
  event AND a comment naming the ancestor (non-silent),
* running descendants have their audit trail committed BEFORE their worker
  is terminated, and the kill routes through ``_terminate_reclaimed_worker``
  (the same helper the reclaim paths use),
* ``consecutive_failures`` resets to 0 (deliberate operator action —
  opposite of the review-loop rule pinned in M2), and
* the dashboard ``_set_status_direct`` reopen path and the DB function
  produce identical descendant outcomes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def conn(tmp_path: Path):
    db = kb.connect(tmp_path / "kanban.db")
    try:
        yield db
    finally:
        db.close()


def _done_parent_with_done_child(conn):
    parent_id = kb.create_task(conn, title="ancestor", assignee="planner")
    assert kb.complete_task(conn, parent_id)
    child_id = kb.create_task(
        conn, title="child", assignee="builder", parents=[parent_id],
    )
    assert kb.complete_task(conn, child_id)
    return parent_id, child_id


def _reopen_parent_directly(conn, parent_id: str) -> None:
    """Minimal stand-in for a reopen surface: flip done -> todo."""
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET status = 'todo', completed_at = NULL WHERE id = ?",
            (parent_id,),
        )


def test_reopen_demotes_done_descendants_with_events_and_comments(conn):
    parent_id, child_id = _done_parent_with_done_child(conn)
    grandchild_id = kb.create_task(
        conn, title="grandchild", assignee="writer", parents=[child_id],
    )
    assert kb.complete_task(conn, grandchild_id)

    _reopen_parent_directly(conn, parent_id)
    result = kb.invalidate_descendants_for_parent_reopen(
        conn, parent_id, author="operator",
    )

    demoted = {entry["id"]: entry for entry in result["invalidated"]}
    assert set(demoted) == {child_id, grandchild_id}
    for tid in (child_id, grandchild_id):
        assert demoted[tid]["prior_status"] == "done"
        assert demoted[tid]["new_status"] == "todo"
        task = kb.get_task(conn, tid)
        assert task is not None and task.status == "todo"
        assert task.completed_at is None

        events = kb.list_events(conn, tid)
        inval = [e for e in events if e.kind == "descendant_invalidated"]
        assert len(inval) == 1
        payload = inval[0].payload
        assert payload["ancestor"] == parent_id
        assert payload["prior_status"] == "done"
        assert payload["new_status"] == "todo"

        comments = kb.list_comments(conn, tid)
        assert any(
            parent_id in c.body and c.author == "operator" for c in comments
        ), f"no invalidation comment naming {parent_id} on {tid}"

    assert result["terminations"] == []


def test_running_descendant_event_precedes_termination_via_reclaim_helper(
    conn, tmp_path, monkeypatch,
):
    parent_id = kb.create_task(conn, title="ancestor", assignee="planner")
    assert kb.complete_task(conn, parent_id)
    child_id = kb.create_task(
        conn, title="running child", assignee="builder", parents=[parent_id],
    )
    claimed = kb.claim_task(conn, child_id)
    assert claimed is not None and claimed.status == "running"
    kb._set_worker_pid(conn, child_id, 424242)

    kills: list[tuple] = []

    def fake_terminate(pid, claim_lock, **kwargs):
        # The audit trail must already be durable when the kill fires:
        # standalone calls commit before terminating.
        side = kb.connect(tmp_path / "kanban.db")
        try:
            kinds = [e.kind for e in kb.list_events(side, child_id)]
        finally:
            side.close()
        assert "descendant_invalidated" in kinds
        kills.append((pid, claim_lock))
        return {"terminated": True}

    monkeypatch.setattr(kb, "_terminate_reclaimed_worker", fake_terminate)

    _reopen_parent_directly(conn, parent_id)
    result = kb.invalidate_descendants_for_parent_reopen(
        conn, parent_id, author="operator",
    )

    assert kills and kills[0][0] == 424242
    # Terminations are (pid, claim_lock, pid_started_at, worker_scope)
    # tuples; the fake records only the positional prefix.
    assert [(t[0], t[1]) for t in result["terminations"]] == kills
    child = kb.get_task(conn, child_id)
    assert child is not None
    assert child.status == "todo"
    assert child.current_run_id is None
    run = kb.latest_run(conn, child_id)
    assert run is not None and run.outcome == "reclaimed"


def test_counter_reset_on_invalidated_descendants(conn):
    parent_id, child_id = _done_parent_with_done_child(conn)
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET consecutive_failures = 4 WHERE id = ?",
            (child_id,),
        )

    _reopen_parent_directly(conn, parent_id)
    kb.invalidate_descendants_for_parent_reopen(conn, parent_id, author="op")

    child = kb.get_task(conn, child_id)
    assert child is not None
    # Deliberate operator action = fresh start with the breaker; contrast
    # with reopen_review_task, which PRESERVES the counter (M2 rule).
    assert child.consecutive_failures == 0


def test_dashboard_and_db_paths_produce_identical_outcomes(tmp_path, monkeypatch):
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    import importlib.util
    import sys

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()

    repo_root = Path(__file__).resolve().parents[2]
    plugin_file = repo_root / "plugins" / "kanban" / "dashboard" / "plugin_api.py"
    spec = importlib.util.spec_from_file_location(
        "hermes_dashboard_plugin_kanban_m3_test", plugin_file,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    app = fastapi.FastAPI()
    app.include_router(mod.router, prefix="/api/plugins/kanban")
    client = TestClient(app)

    def build_graph(tag: str):
        with kb.connect() as c:
            parent = kb.create_task(c, title=f"{tag}-parent", assignee="planner")
            assert kb.complete_task(c, parent)
            child = kb.create_task(
                c, title=f"{tag}-child", assignee="builder", parents=[parent],
            )
            assert kb.complete_task(c, child)
        return parent, child

    dash_parent, dash_child = build_graph("dash")
    db_parent, db_child = build_graph("db")

    # Surface 1: dashboard drag (done -> todo) via _set_status_direct.
    r = client.patch(
        f"/api/plugins/kanban/tasks/{dash_parent}", json={"status": "todo"},
    )
    assert r.status_code == 200, r.text

    # Surface 2: DB function directly (the single domain implementation).
    with kb.connect() as c:
        with kb.write_txn(c):
            c.execute(
                "UPDATE tasks SET status = 'todo', completed_at = NULL "
                "WHERE id = ?",
                (db_parent,),
            )
        kb.invalidate_descendants_for_parent_reopen(
            c, db_parent, author="dashboard",
        )

    with kb.connect() as c:
        def snapshot(tid: str):
            t = kb.get_task(c, tid)
            assert t is not None
            kinds = sorted(e.kind for e in kb.list_events(c, tid))
            n_comments = len(kb.list_comments(c, tid))
            return (
                t.status,
                t.completed_at,
                t.current_run_id,
                t.consecutive_failures,
                kinds,
                n_comments,
            )

        assert snapshot(dash_child) == snapshot(db_child)
        status, completed_at, _run, failures, kinds, n_comments = snapshot(db_child)
        assert status == "todo"
        assert completed_at is None
        assert failures == 0
        assert "descendant_invalidated" in kinds
        assert n_comments >= 1


# ---------------------------------------------------------------------------
# Gate B pass 4, finding N: the Phase 0 stop verdict is bound to the
# probed run identity — a run replaced between the probe and the
# demotion transaction must never be demoted on the old verdict
# ---------------------------------------------------------------------------


def _arm_scoped_running_descendant(conn, child_id, *, pid=424242, started=55,
                                   scope="old-run.scope"):
    host = kb._claimer_id().split(":", 1)[0]
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET claim_lock = ?, claim_expires = 9999999999, "
            "worker_pid = ?, worker_pid_started_at = ?, "
            "worker_registered_at = 1, worker_scope = ? WHERE id = ?",
            (f"{host}:{pid}", pid, started, scope, child_id),
        )


def test_scoped_descendant_run_replaced_between_probe_and_demotion_is_skipped(
    conn, monkeypatch,
):
    """A concurrent retry replaces the descendant's run between the
    Phase 0 stop probe and the write transaction. The confirmed-empty
    verdict belonged to the OLD run's cgroup; the NEW worker must stay
    running untouched and wait for the next pass to re-probe it."""
    parent_id = kb.create_task(conn, title="ancestor", assignee="planner")
    assert kb.complete_task(conn, parent_id)
    child_id = kb.create_task(
        conn, title="scoped child", assignee="builder", parents=[parent_id],
    )
    assert kb.claim_task(conn, child_id) is not None
    _arm_scoped_running_descendant(conn, child_id)
    host = kb._claimer_id().split(":", 1)[0]

    def replace_run_during_probe(unit_name, *, task_id=None, **_kw):
        assert unit_name == "old-run.scope"
        # The retry lands mid-probe: old run replaced by a new scoped run.
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET current_run_id = 99001, "
                "claim_lock = ?, worker_pid = 424243, "
                "worker_pid_started_at = 56, worker_registered_at = 1, "
                "worker_scope = 'new-run.scope' WHERE id = ?",
                (f"{host}:424243", child_id),
            )
        return True  # the OLD scope's cgroup is confirmed empty

    monkeypatch.setattr(kb, "request_worker_scope_stop", replace_run_during_probe)

    _reopen_parent_directly(conn, parent_id)
    result = kb.invalidate_descendants_for_parent_reopen(
        conn, parent_id, author="operator",
    )

    row = conn.execute(
        "SELECT status, current_run_id, worker_pid, worker_scope, claim_lock "
        "FROM tasks WHERE id = ?",
        (child_id,),
    ).fetchone()
    assert row["status"] == "running"
    assert row["current_run_id"] == 99001
    assert row["worker_pid"] == 424243
    assert row["worker_scope"] == "new-run.scope"
    assert row["claim_lock"] is not None
    assert result["invalidated"] == []
    assert result["terminations"] == []
    assert [
        e for e in kb.list_events(conn, child_id)
        if e.kind == "descendant_invalidated"
    ] == []
    run = kb.latest_run(conn, child_id)
    assert run is not None and run.outcome is None  # no run closed underneath


def test_scoped_descendant_with_stable_identity_demotes_after_verified_stop(
    conn, monkeypatch,
):
    """Control: when the probed identity still matches inside the
    transaction, a confirmed-empty scope demotes in-txn with no kill."""
    parent_id = kb.create_task(conn, title="ancestor", assignee="planner")
    assert kb.complete_task(conn, parent_id)
    child_id = kb.create_task(
        conn, title="scoped child", assignee="builder", parents=[parent_id],
    )
    assert kb.claim_task(conn, child_id) is not None
    _arm_scoped_running_descendant(conn, child_id)
    monkeypatch.setattr(
        kb, "request_worker_scope_stop", lambda *a, **k: True,
    )

    _reopen_parent_directly(conn, parent_id)
    result = kb.invalidate_descendants_for_parent_reopen(
        conn, parent_id, author="operator",
    )

    row = conn.execute(
        "SELECT status, worker_scope, claim_lock FROM tasks WHERE id = ?",
        (child_id,),
    ).fetchone()
    assert row["status"] == "todo"
    assert row["worker_scope"] is None
    assert row["claim_lock"] is None
    assert [e["id"] for e in result["invalidated"]] == [child_id]
    assert result["terminations"] == []
    run = kb.latest_run(conn, child_id)
    assert run is not None and run.outcome == "reclaimed"

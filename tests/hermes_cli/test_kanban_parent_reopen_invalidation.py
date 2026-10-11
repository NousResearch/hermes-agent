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
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def conn(tmp_path: Path):
    db = kbc.connect(tmp_path / "kanban.db")
    try:
        yield db
    finally:
        db.close()


def _done_parent_with_done_child(conn):
    parent_id = kb.create_task(conn, title="ancestor", assignee="planner")
    assert kb.complete_task(conn, parent_id, result="done")
    child_id = kb.create_task(
        conn, title="child", assignee="builder", parents=[parent_id],
    )
    assert kb.complete_task(conn, child_id, result="done")
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
    assert kb.complete_task(conn, grandchild_id, result="done")

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
    assert kb.complete_task(conn, parent_id, result="done")
    child_id = kb.create_task(
        conn, title="running child", assignee="builder", parents=[parent_id],
    )
    claimed = kb.claim_task(conn, child_id)
    assert claimed is not None and claimed.status == "running"
    kbd._set_worker_pid(conn, child_id, 424242)

    kills: list[tuple] = []

    def fake_terminate(pid, claim_lock, started_at=None, **kwargs):
        # The audit trail must already be durable when the kill fires:
        # standalone calls commit before terminating.
        side = kbc.connect(tmp_path / "kanban.db")
        try:
            kinds = [e.kind for e in kb.list_events(side, child_id)]
        finally:
            side.close()
        assert "descendant_invalidated" in kinds
        kills.append((pid, claim_lock, started_at))
        return {"terminated": True}

    monkeypatch.setattr(kb, "_terminate_reclaimed_worker", fake_terminate)

    _reopen_parent_directly(conn, parent_id)
    result = kb.invalidate_descendants_for_parent_reopen(
        conn, parent_id, author="operator",
    )

    assert kills and kills[0][0] == 424242
    assert result["terminations"] == kills
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
        with kbc.connect() as c:
            parent = kb.create_task(c, title=f"{tag}-parent", assignee="planner")
            assert kb.complete_task(c, parent, result="done")
            child = kb.create_task(
                c, title=f"{tag}-child", assignee="builder", parents=[parent],
            )
            assert kb.complete_task(c, child, result="done")
        return parent, child

    dash_parent, dash_child = build_graph("dash")
    db_parent, db_child = build_graph("db")

    # Surface 1: dashboard drag (done -> todo) via _set_status_direct.
    r = client.patch(
        f"/api/plugins/kanban/tasks/{dash_parent}", json={"status": "todo"},
    )
    assert r.status_code == 200, r.text

    # Surface 2: DB function directly (the single domain implementation).
    with kbc.connect() as c:
        with kb.write_txn(c):
            c.execute(
                "UPDATE tasks SET status = 'todo', completed_at = NULL "
                "WHERE id = ?",
                (db_parent,),
            )
        kb.invalidate_descendants_for_parent_reopen(
            c, db_parent, author="dashboard",
        )

    with kbc.connect() as c:
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


@pytest.mark.parametrize("gate_result", ["GATE_FAIL", "PORTABLE_GATE_FAIL"])
@pytest.mark.parametrize("phase", ["ready", "running", "review", "running_review", "done"])
def test_gate_edit_invalidates_descendants_and_restores_phase(
    conn, tmp_path, monkeypatch, gate_result, phase,
):
    parent = kb.create_task(conn, title="gate")
    child = kb.create_task(conn, title="dependent", assignee="builder", parents=[parent])
    assert kb.complete_task(conn, parent, result="GATE_PASS")
    assert kb.get_task(conn, child).status == "ready"
    if phase in {"review", "running_review"}:
        assert kb.request_review(conn, child, summary="implementation", reviewer="reviewer")
    if phase in {"running", "running_review"}:
        claim = kb.claim_review_task if phase == "running_review" else kb.claim_task
        claimed = claim(conn, child)
        assert claimed is not None
        kbd._set_worker_pid(conn, child, 424242)
    if phase == "done":
        assert kb.complete_task(conn, child, result="child evidence", metadata={"proof": "kept"})
    before = kb.get_task(conn, child)
    previous_run = kb.latest_run(conn, child)
    fingerprint = conn.execute("SELECT worker_started_at FROM tasks WHERE id = ?", (child,)).fetchone()[0]
    conn.execute("UPDATE tasks SET consecutive_failures = 1 WHERE id = ?", (child,))
    grandchild = kb.create_task(conn, title="transitive dependent", parents=[child])
    kills = []

    def terminate(pid, claim_lock, started_at=None):
        assert not conn.in_transaction
        with kbc.connect(tmp_path / "kanban.db") as side:
            assert kb.get_task(side, parent).result == gate_result
            assert kb.get_task(side, child).status == "todo"
            assert any(e.kind == "descendant_invalidated" for e in kb.list_events(side, child))
        kills.append((pid, claim_lock, started_at))

    monkeypatch.setattr(kb, "_terminate_reclaimed_worker", terminate)
    assert kb.edit_task(conn, parent, result=gate_result)
    assert kb.get_task(conn, parent).status == "done"
    after = kb.get_task(conn, child)
    assert after.status == "todo" and after.current_run_id is None
    assert after.result == before.result
    assert after.consecutive_failures == 1
    assert kb.get_task(conn, grandchild).status == "todo"
    invalidations = [e for e in kb.list_events(conn, child) if e.kind == "descendant_invalidated"]
    assert len(invalidations) == 1
    assert invalidations[0].payload["reason"] == "ancestor_gate_failed"
    assert invalidations[0].payload["result"] == gate_result
    assert any(parent in c.body for c in kb.list_comments(conn, child))
    if phase in {"running", "running_review"}:
        assert kills == [(424242, before.claim_lock, fingerprint)]
        run = kb.latest_run(conn, child)
        assert run.id == previous_run.id and run.outcome == "reclaimed"
        assert run.ended_at is not None
        assert invalidations[0].run_id == run.id
        assert not kb.complete_task(conn, child, result="stale", expected_run_id=run.id)
    else:
        assert not kills
    if phase == "done":
        assert kb.latest_run(conn, child) == previous_run
        assert any(e.kind == "completed" for e in kb.list_events(conn, child))
        assert after.completed_at is None  # existing descendant-reopen protocol
    assert kb.edit_task(conn, parent, result=gate_result)  # no repeated invalidation
    assert len([e for e in kb.list_events(conn, child) if e.kind == "descendant_invalidated"]) == 1
    assert kb.edit_task(conn, parent, result="PORTABLE_GATE_PASS")
    expected_phase = "review" if phase in {"review", "running_review"} else "ready"
    assert kb.get_task(conn, child).status == expected_phase
    claim = kb.claim_review_task if expected_phase == "review" else kb.claim_task
    assert claim(conn, child) is not None


@pytest.mark.parametrize("failure", ["audit", "commit"])
def test_gate_edit_rollback_never_terminates_worker(conn, monkeypatch, failure):
    parent = kb.create_task(conn, title="gate")
    assert kb.complete_task(conn, parent, result="GATE_PASS")
    child = kb.create_task(conn, title="dependent", parents=[parent])
    assert kb.claim_task(conn, child)
    kbd._set_worker_pid(conn, child, 424242)
    before = kb.get_task(conn, child)
    run = kb.latest_run(conn, child)
    kills = []
    monkeypatch.setattr(kb, "_terminate_reclaimed_worker", lambda *a, **kw: kills.append(a))
    if failure == "audit":
        conn.execute("CREATE TEMP TRIGGER reject_gate_audit BEFORE INSERT ON task_events "
                     "WHEN NEW.kind = 'descendant_invalidated' "
                     "BEGIN SELECT RAISE(ABORT, 'audit failed'); END")
    else:
        boundary = kbc._execute_boundary_with_retry
        def reject_commit(c, statement):
            if statement == "COMMIT":
                raise RuntimeError("commit failed")
            return boundary(c, statement)
        monkeypatch.setattr(kbc, "_execute_boundary_with_retry", reject_commit)
    with pytest.raises(Exception, match="failed"):
        kb.edit_task(conn, parent, result="PORTABLE_GATE_FAIL")
    assert not conn.in_transaction
    assert kb.get_task(conn, parent).result == "GATE_PASS"
    assert kb.get_task(conn, child) == before
    assert kb.latest_run(conn, child) == run
    assert not kills
    assert not any(e.kind == "descendant_invalidated" for e in kb.list_events(conn, child))


@pytest.mark.parametrize("result", ["ordinary result", "BUILD_FAIL", "gate_fail", "XGATE_FAIL"])
def test_non_gate_result_edit_preserves_running_child(conn, monkeypatch, result):
    parent = kb.create_task(conn, title="parent")
    assert kb.complete_task(conn, parent, result="GATE_PASS")
    child = kb.create_task(conn, title="child", parents=[parent])
    assert kb.claim_task(conn, child)
    before = kb.get_task(conn, child)
    kills = []
    monkeypatch.setattr(kb, "_terminate_reclaimed_worker", lambda *a, **kw: kills.append(a))
    assert kb.edit_task(conn, parent, result=result)
    assert kb.get_task(conn, child) == before
    assert not kills


@pytest.mark.parametrize("result", ["GATE_FAIL", "PORTABLE_GATE_FAIL"])
def test_terminal_gate_failure_parent_does_not_promote_child(conn, result):
    parent = kb.create_task(conn, title="failed gate")
    child = kb.create_task(conn, title="dependent", parents=[parent])

    assert kb.complete_task(conn, parent, result=result)

    assert kb.get_task(conn, child).status == "todo"
    assert not kb._parents_satisfied(conn, child)


def test_terminal_successful_parent_promotes_child(conn):
    parent = kb.create_task(conn, title="successful parent")
    child = kb.create_task(conn, title="dependent", parents=[parent])

    assert kb.complete_task(conn, parent, result="PORTABLE_GATE_PASS")

    assert kb.get_task(conn, child).status == "ready"
    assert kb._parents_satisfied(conn, child)


@pytest.mark.parametrize("result", ["GATE_FAIL", "PORTABLE_GATE_FAIL"])
def test_failed_gate_rechecks_creation_unblock_and_review(conn, result):
    parent = kb.create_task(conn, title="gate")
    assert kb.complete_task(conn, parent, result=result)
    child = kb.create_task(conn, title="dependent", parents=[parent])
    assert kb.get_task(conn, child).status == "todo"
    parked = kb.create_task(conn, title="parked", parents=[parent], initial_status="blocked")
    assert kb.unblock_task(conn, parked)
    assert kb.get_task(conn, parked).status == "todo"
    assert not kb.request_review(conn, child, summary="must refuse")
    # Legacy/external writers can still leave a ready/review card: claim fences it.
    for phase in ("ready", "review"):
        conn.execute("UPDATE tasks SET status = ? WHERE id = ?", (phase, child))
        claim = kb.claim_task if phase == "ready" else kb.claim_review_task
        assert claim(conn, child) is None
        assert kb.get_task(conn, child).status == "todo"
    assert kb.edit_task(conn, parent, result="GATE_PASS")
    assert kb.get_task(conn, child).status == "review"
    assert kb.get_task(conn, parked).status == "ready"

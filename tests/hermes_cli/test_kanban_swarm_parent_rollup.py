from __future__ import annotations

import json
from pathlib import Path
from sqlite3 import Connection

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_swarm import SwarmWorkerSpec, create_swarm, latest_blackboard


def _autopilot_parent_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    child_names: tuple[str, ...],
) -> tuple[Connection, str, dict[str, str]]:
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    conn = kb.connect()
    parent = kb.create_task(conn, title="parent", body="parent")
    children: dict[str, str] = {}
    for index, name in enumerate(child_names, start=1):
        child = kb.create_task(conn, title=name, body=name)
        conn.execute("UPDATE tasks SET created_at = ? WHERE id = ?", (index, child))
        kb.link_tasks(conn, parent, child, relation_type="hierarchy")
        children[name] = child
    return conn, parent, children


def _verified_closeout_evidence() -> dict[str, object]:
    return {
        "schema": "kanban_closeout_evidence.v1",
        "verifier_result": {
            "schema": "kanban_verifier_result.v1",
            "verdict": "pass",
        },
        "verification": {"allowed": True},
    }


def _set_task_state(
    conn: Connection,
    task_id: str,
    *,
    status: str,
    review_phase: str | None = None,
    admission_snapshot: dict[str, object] | None = None,
    closeout_evidence: dict[str, object] | None = None,
) -> None:
    conn.execute(
        """
        UPDATE tasks
           SET status = ?,
               review_phase = ?,
               admission_snapshot = ?,
               closeout_evidence = ?
         WHERE id = ?
        """,
        (
            status,
            review_phase,
            json.dumps(admission_snapshot, ensure_ascii=False) if admission_snapshot is not None else None,
            json.dumps(closeout_evidence, ensure_ascii=False) if closeout_evidence is not None else None,
            task_id,
        ),
    )


def test_autopilot_parent_report_preserves_rollup_shape_for_incomplete_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a parent has one completed child and one still-dispatchable child.
    conn, parent, children = _autopilot_parent_tree(tmp_path, monkeypatch, ("complete", "partial"))
    try:
        _set_task_state(
            conn,
            children["complete"],
            status="done",
            review_phase="closed",
            closeout_evidence=_verified_closeout_evidence(),
        )
        _set_task_state(conn, children["partial"], status="ready", admission_snapshot={"autopilot_eligible": True})

        # When: Autopilot builds the parent report.
        report = kb.autopilot_parent_report(conn, parent)

        # Then: the top-level report mirrors the parent-child matrix rollup fields.
        matrix = report["parent_child_matrix"]
        assert report["parentRollupState"] == "partial"
        assert report["countsByRollupState"] == {"complete": 1, "partial": 1}
        assert report["remainingChildren"] == [children["partial"]]
        assert report["nextRequiredChild"] == children["partial"]
        assert matrix["parentRollupState"] == report["parentRollupState"]
        assert matrix["countsByRollupState"] == report["countsByRollupState"]
        assert matrix["remainingChildren"] == report["remainingChildren"]
        assert matrix["nextRequiredChild"] == report["nextRequiredChild"]
    finally:
        conn.close()


def test_create_swarm_preserves_every_worker_as_parent_child_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: an isolated board and a parent swarm with three worker children.
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    conn = kb.connect()
    try:
        workers = [
            SwarmWorkerSpec(profile="alpha", title="first child", body="first"),
            SwarmWorkerSpec(profile="beta", title="second child", body="second"),
            SwarmWorkerSpec(profile="gamma", title="third child", body="third"),
        ]

        # When: the swarm topology is created.
        created = create_swarm(
            conn,
            goal="preserve parent scope",
            workers=workers,
            verifier_assignee="verifier",
            synthesizer_assignee="synthesizer",
        )

        # Then: the root child scope contains every worker and the verifier is separate.
        topology = latest_blackboard(conn, created.root_id)["topology"]
        assert set(kb.child_ids(conn, created.root_id)) == set(created.worker_ids)
        assert topology["worker_ids"] == created.worker_ids
        assert len(created.worker_ids) == 3
        assert kb.parent_ids(conn, created.verifier_id) == created.worker_ids
    finally:
        conn.close()


def test_autopilot_parent_report_distinguishes_rollup_states_with_counts_and_reasons(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn, parent, children = _autopilot_parent_tree(
        tmp_path,
        monkeypatch,
        ("complete", "partial", "blocked", "decision"),
    )
    try:
        _set_task_state(
            conn,
            children["complete"],
            status="done",
            review_phase="closed",
            closeout_evidence=_verified_closeout_evidence(),
        )
        _set_task_state(conn, children["partial"], status="ready", admission_snapshot={"autopilot_eligible": True})
        _set_task_state(conn, children["blocked"], status="blocked", review_phase="worker_done")
        _set_task_state(conn, children["decision"], status="todo", admission_snapshot={"autopilot_eligible": False})

        # When: Autopilot summarizes a mixed parent child-scope.
        report = kb.autopilot_parent_report(conn, parent)

        # Then: all four child states are counted and each incomplete child has an actionable reason.
        matrix = report["parent_child_matrix"]
        matrix_children = {item["task_id"]: item for item in matrix["children"]}
        assert report["parentRollupState"] == "review_blocked"
        assert report["countsByRollupState"] == {
            "complete": 1,
            "partial": 1,
            "review_blocked": 1,
            "needs_user_decision": 1,
        }
        assert set(report["remainingChildren"]) == {
            children["partial"],
            children["blocked"],
            children["decision"],
        }
        assert report["nextRequiredChild"] == children["partial"]
        assert matrix["parentRollupState"] == report["parentRollupState"]
        assert matrix["countsByRollupState"] == report["countsByRollupState"]
        assert matrix_children[children["complete"]]["rollup_state"] == "complete"
        assert matrix_children[children["complete"]]["reason"] is None
        assert matrix_children[children["partial"]]["rollup_state"] == "partial"
        assert matrix_children[children["partial"]]["reason"] == "child_scope_incomplete"
        assert matrix_children[children["blocked"]]["rollup_state"] == "review_blocked"
        assert matrix_children[children["blocked"]]["reason"] == "child_review_blocked"
        assert matrix_children[children["decision"]]["rollup_state"] == "needs_user_decision"
        assert matrix_children[children["decision"]]["reason"] == "child_needs_user_decision"
    finally:
        conn.close()

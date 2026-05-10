from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_work_state import (
    ingest_work_state_run_evidence,
    project_work_state_to_kanban_run,
)


def _record(**overrides):
    values = {
        "work_id": "CH-411",
        "title": "CH-411 title",
        "objective": "Map work_state outcomes into Kanban run metadata",
        "owner": "hermes",
        "executor": "omx",
        "mode": "delegated",
        "owner_session_id": "owner-session",
        "executor_session_id": "exec-session",
        "tmux_session": "omx-CH-411",
        "repo_path": "/repo/hermes-agent",
        "worktree_path": "/repo/hermes-agent/.worktrees/ch411",
        "state": "running",
        "next_action": "Continue bounded slice",
        "proof": "work_state:running",
        "current_lane": "omx_exec",
        "planning_gate": "closed",
        "next_execution_branch": "none",
        "close_authority": "hermes",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _project(record, **kwargs):
    return project_work_state_to_kanban_run(record, **kwargs)


def _run(projection):
    return projection["task_run"]


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def test_running_work_state_maps_to_running_task_and_open_run_metadata():
    projection = _project(_record(state="running"))

    assert projection["status"] == "mapped"
    assert projection["task_status"] == "running"
    assert _run(projection)["status"] == "running"
    assert _run(projection)["outcome"] is None
    metadata = _run(projection)["metadata"]
    assert metadata["schema"] == "work_state_kanban_run_projection.v1"
    assert metadata["work_state"]["work_id"] == "CH-411"
    assert metadata["work_state"]["state"] == "running"
    assert metadata["kanban_done_projection"] == "forbidden"


@pytest.mark.parametrize(
    ("state", "run_status", "outcome"),
    [
        ("blocked", "blocked", "blocked"),
        ("stale", "blocked", "blocked"),
        ("retry-needed", "blocked", "blocked"),
        ("handoff-needed", "blocked", "blocked"),
        ("continuation-needed", "blocked", "blocked"),
        ("worker_done/retry-needed", "blocked", "blocked"),
        ("failed", "failed", "gave_up"),
    ],
)
def test_recoverable_work_state_outcomes_block_task_and_record_canonical_run_outcome(
    state, run_status, outcome
):
    projection = _project(_record(state=state, proof=f"work_state:{state}"))

    assert projection["status"] == "mapped"
    assert projection["task_status"] == "blocked"
    assert _run(projection)["status"] == run_status
    assert _run(projection)["outcome"] == outcome
    metadata = _run(projection)["metadata"]
    assert metadata["work_state"]["state"] == state
    assert metadata["work_state_outcome_preserved_in_metadata"] is True
    assert "metadata" not in projection.get("task", {})


def test_projected_run_statuses_and_outcomes_match_kanban_run_ledger_vocab():
    allowed_status = {"running", "done", "blocked", "crashed", "timed_out", "failed", "released"}
    allowed_outcome = {"completed", "blocked", "crashed", "timed_out", "spawn_failed", "gave_up", "reclaimed", None}

    for state in [
        "active",
        "running",
        "worker_done",
        "finished",
        "blocked",
        "stale",
        "retry-needed",
        "handoff-needed",
        "continuation-needed",
        "failed",
    ]:
        projection = _project(
            _record(
                state=state,
                usable_outcome="usable_output" if state in {"finished", "worker_done"} else None,
                close_disposition="close" if state in {"finished", "worker_done"} else None,
            )
        )
        run = _run(projection)
        assert run["status"] in allowed_status
        assert run["outcome"] in allowed_outcome


def test_finished_usable_output_records_completed_run_without_kanban_done_projection():
    projection = _project(
        _record(
            state="finished",
            usable_outcome="usable_output",
            close_disposition="close",
            next_action="Review output and update Linear SSOT",
            proof="tests passed and artifacts ready",
        )
    )

    assert projection["status"] == "mapped"
    assert projection["task_status"] == "blocked"
    assert _run(projection)["status"] == "done"
    assert _run(projection)["outcome"] == "completed"
    metadata = _run(projection)["metadata"]
    assert metadata["usable_output_recorded"] is True
    assert metadata["kanban_done_projection"] == "forbidden"
    assert metadata["kanban_task_status_reason"] == "linear_ssot_no_kanban_done_projection"


def test_worker_done_records_completed_run_without_kanban_done_projection():
    projection = _project(
        _record(
            state="worker_done",
            next_action="Review worker output before closeout",
            proof="worker completed assigned slice",
            review_closeout={"executor_event": {"status": "executor_finished"}},
        )
    )

    assert projection["status"] == "mapped"
    assert projection["task_status"] == "blocked"
    assert _run(projection)["status"] == "done"
    assert _run(projection)["outcome"] == "completed"
    metadata = _run(projection)["metadata"]
    assert metadata["work_state"]["state"] == "worker_done"
    assert metadata["work_state"]["review_closeout"] == {
        "executor_event": {"status": "executor_finished"}
    }
    assert metadata["kanban_done_projection"] == "forbidden"


def test_ambiguous_resolution_fails_closed_to_blocked_run():
    projection = _project(
        _record(state="retry_needed"),
        resolution={"status": "ambiguous", "matches": [{"work_id": "a"}, {"work_id": "b"}]},
    )

    assert projection["status"] == "fail_closed"
    assert projection["reason"] == "ambiguous_recovery_metadata_fail_closed"
    assert projection["task_status"] == "blocked"
    assert _run(projection)["status"] == "blocked"
    assert _run(projection)["outcome"] == "blocked"
    assert _run(projection)["metadata"]["projection_failed_closed"] is True


def test_missing_recovery_metadata_fails_closed_to_blocked_run():
    projection = _project(
        _record(
            state="stale",
            next_action="",
            proof="",
            executor_session_id=None,
            tmux_session=None,
            repo_path=None,
            worktree_path=None,
        )
    )

    assert projection["status"] == "fail_closed"
    assert projection["reason"] == "missing_recovery_metadata_fail_closed"
    assert projection["task_status"] == "blocked"
    assert _run(projection)["status"] == "blocked"
    assert _run(projection)["outcome"] == "blocked"
    assert _run(projection)["metadata"]["missing"] == [
        "next_action",
        "proof",
        "executor_session_or_workspace_locator",
    ]


def test_missing_record_fails_closed_without_task_metadata():
    projection = _project(None)

    assert projection["status"] == "fail_closed"
    assert projection["reason"] == "missing_work_state_record_fail_closed"
    assert projection["task_status"] == "blocked"
    assert _run(projection)["metadata"]["projection_failed_closed"] is True
    assert "task" not in projection


@pytest.mark.parametrize(
    ("state", "run_status", "outcome", "task_status"),
    [
        ("active", "running", None, "running"),
        ("blocked", "blocked", "blocked", "blocked"),
        ("stale", "blocked", "blocked", "blocked"),
        ("failed", "failed", "gave_up", "blocked"),
        ("handoff-needed", "blocked", "blocked", "blocked"),
        ("continuation-needed", "blocked", "blocked", "blocked"),
        ("worker_done", "done", "completed", "blocked"),
        ("retry-needed", "blocked", "blocked", "blocked"),
    ],
)
def test_ingest_work_state_run_evidence_writes_recoverable_task_run_metadata(
    kanban_home, state, run_status, outcome, task_status
):
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title=f"{state} task",
            idempotency_key=f"linear:CH-420-{state}",
        )
        record = _record(
            work_id=f"CH-420-{state}",
            state=state,
            proof=f"work_state:{state}",
            next_action=f"Handle {state}",
        )

        result = ingest_work_state_run_evidence(conn, record)
        task = kb.get_task(conn, task_id)
        run = kb.latest_run(conn, task_id)

    assert result["status"] == "ingested"
    assert result["side_effects"] == {
        "kanban_task_written": True,
        "task_run_written": True,
        "executor_spawned": False,
        "linear_done_mutated": False,
        "kanban_done_projected_to_linear": False,
    }
    assert task.status == task_status
    assert task.status != "done"
    assert run.status == run_status
    assert run.outcome == outcome
    assert run.metadata["schema"] == "work_state_kanban_run_projection.v1"
    assert run.metadata["source"] == "work_state_run_evidence"
    assert run.metadata["work_state"]["state"] == state
    assert run.metadata["work_state"]["work_id"] == f"CH-420-{state}"
    assert run.metadata["kanban_done_projection"] == "forbidden"
    if state == "active":
        assert run.ended_at is None
    else:
        assert run.ended_at is not None


def test_ingest_updates_existing_active_run_without_spawning_executor(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="active task", idempotency_key="linear:CH-420-active")
        first = ingest_work_state_run_evidence(
            conn,
            _record(work_id="CH-420-active", state="active", proof="first proof"),
        )
        second = ingest_work_state_run_evidence(
            conn,
            _record(work_id="CH-420-active", state="running", proof="second proof"),
        )
        runs = kb.list_runs(conn, task_id)

    assert first["run_id"] == second["run_id"]
    assert len(runs) == 1
    assert runs[0].status == "running"
    assert runs[0].metadata["work_state"]["state"] == "running"
    assert runs[0].metadata["work_state"]["proof"] == "second proof"


def test_ingest_fails_closed_without_writing_when_kanban_correlation_is_ambiguous(kanban_home):
    with kb.connect() as conn:
        a = kb.create_task(conn, title="a", public_id="CH-420")
        b = kb.create_task(conn, title="b", idempotency_key="linear:CH-420")

        result = ingest_work_state_run_evidence(conn, _record(work_id="CH-420", state="blocked"))
        runs_a = kb.list_runs(conn, a)
        runs_b = kb.list_runs(conn, b)

    assert result["status"] == "fail_closed"
    assert result["reason"] == "ambiguous_kanban_task_correlation_fail_closed"
    assert result["side_effects"]["task_run_written"] is False
    assert runs_a == []
    assert runs_b == []


def test_ingest_fails_closed_without_writing_when_kanban_correlation_is_missing(kanban_home):
    with kb.connect() as conn:
        result = ingest_work_state_run_evidence(conn, _record(work_id="CH-420-missing"))

    assert result["status"] == "fail_closed"
    assert result["reason"] == "missing_kanban_task_correlation_fail_closed"
    assert result["side_effects"]["kanban_task_written"] is False


def test_ingest_missing_delegated_locator_writes_blocked_fail_closed_trail(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="locator task", idempotency_key="linear:CH-420")
        result = ingest_work_state_run_evidence(
            conn,
            _record(
                work_id="CH-420",
                state="stale",
                executor_session_id=None,
                tmux_session=None,
                repo_path=None,
                worktree_path=None,
                proof="work_state:stale",
            ),
        )
        task = kb.get_task(conn, task_id)
        run = kb.latest_run(conn, task_id)

    assert result["status"] == "ingested_fail_closed"
    assert result["reason"] == "missing_recovery_metadata_fail_closed"
    assert task.status == "blocked"
    assert run.status == "blocked"
    assert run.outcome == "blocked"
    assert run.metadata["projection_failed_closed"] is True
    assert run.metadata["missing"] == ["executor_session_or_workspace_locator"]
    assert run.metadata["work_state"]["state"] == "stale"

"""Operator recovery from Kanban's triage loop-breaker state."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

from hermes_cli import kanban as kanban_cli
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _trip_block_loop(conn, task_id: str) -> None:
    assert kb.block_task(conn, task_id, reason="waiting on owner", kind="needs_input")
    assert kb.unblock_task(conn, task_id)
    assert kb.block_task(conn, task_id, reason="waiting on owner", kind="needs_input")
    task = kb.get_task(conn, task_id)
    assert task is not None
    assert task.status == "triage"
    assert task.block_recurrences == kb.BLOCK_RECURRENCE_LIMIT


def _mutate_status_during_acceptance(monkeypatch, task_id: str, *statuses: str) -> None:
    from hermes_cli import kanban_pr_acceptance_store as acceptance_store

    def mutate(*args, **kwargs):
        with kbc.connect_closing() as other:
            with acceptance_store.write_txn(other):
                for status in statuses:
                    other.execute("UPDATE tasks SET status=? WHERE id=?", (status, task_id))
                    kb._append_event(other, task_id, f"test_status_{status}")

    monkeypatch.setattr(acceptance_store, "prepare_acceptance", mutate)


def test_triage_task_cannot_complete_without_explicit_operator_override(kanban_home):
    with kbc.connect_closing() as conn:
        task_id = kb.create_task(conn, title="needs reconciliation")
        _trip_block_loop(conn, task_id)

        assert kb.complete_task(conn, task_id, result="adopted verified handoff") is False
        # Existing non-CLI callers use force for unrelated live-claim recovery.
        # That must not silently become permission to bypass triage.
        assert kb.complete_task(
            conn,
            task_id,
            result="adopted verified handoff",
            force=True,
        ) is False
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "triage"


def test_triage_resolution_gate_also_requires_force(kanban_home):
    with kbc.connect_closing() as conn:
        task_id = kb.create_task(conn, title="no force bypass")
        _trip_block_loop(conn, task_id)

        assert not kb.complete_task(
            conn,
            task_id,
            result="attempted bypass",
            _triage_capability=kb._TRIAGE_RESOLUTION_CAPABILITY,
        )
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "triage"


def test_initial_triage_task_is_not_resolvable_as_a_loop_breaker(kanban_home):
    with kbc.connect_closing() as conn:
        task_id = kb.create_task(conn, title="needs specification", triage=True)

    output = kanban_cli.run_slash(
        f"complete {task_id} --force --result 'attempted shortcut'"
    )

    assert "not a loop-breaker" in output
    with kbc.connect_closing() as conn:
        task = kb.get_task(conn, task_id)
        assert task is not None and task.status == "triage"
        assert not any(
            event.kind in {"triage_resolved", "completed"}
            for event in kb.list_events(conn, task_id)
        )

        # Recurrence state without its durable loop event is not provenance.
        conn.execute(
            "UPDATE tasks SET block_recurrences=? WHERE id=?",
            (kb.BLOCK_RECURRENCE_LIMIT, task_id),
        )
        assert not kb.complete_task(
            conn,
            task_id,
            result="attempted direct shortcut",
            force=True,
            _triage_capability=kb._TRIAGE_RESOLUTION_CAPABILITY,
        )


def test_historical_loop_provenance_cannot_resolve_later_dashboard_triage(kanban_home):
    from plugins.kanban.dashboard.plugin_api import _set_status_direct

    with kbc.connect_closing() as conn:
        task_id = kb.create_task(conn, title="historically looped")
        _trip_block_loop(conn, task_id)
        assert kb.specify_triage_task(conn, task_id)
        assert _set_status_direct(conn, task_id, "triage")

    output = kanban_cli.run_slash(
        f"complete {task_id} --force --result 'attempted stale shortcut'"
    )

    assert "not a loop-breaker" in output
    with kbc.connect_closing() as conn:
        task = kb.get_task(conn, task_id)
        assert task is not None and task.status == "triage"
        assert not any(
            event.kind in {"triage_resolved", "completed"}
            for event in kb.list_events(conn, task_id)
        )


def test_cli_explains_how_to_resolve_triage_task(kanban_home):
    with kbc.connect_closing() as conn:
        task_id = kb.create_task(conn, title="needs reconciliation")
        _trip_block_loop(conn, task_id)

    output = kanban_cli.run_slash(
        f"complete {task_id} --result 'adopted verified handoff'"
    )

    assert "triage is a loop-breaker state" in output
    assert "re-run with --force" in output
    with kbc.connect_closing() as conn:
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "triage"


@pytest.mark.parametrize(
    "evidence_args",
    ["", "--result '   '", "--summary '   '", "--result '   ' --summary '   '"],
)
def test_cli_force_preserves_stored_evidence_for_non_triage_task(
    kanban_home, evidence_args,
):
    with kbc.connect_closing() as conn:
        task_id = kb.create_task(conn, title="ordinary forced completion")
        conn.execute(
            "UPDATE tasks SET result=? WHERE id=?",
            ("previously recorded evidence", task_id),
        )

    output = kanban_cli.run_slash(f"complete {task_id} --force {evidence_args}".strip())

    assert output == f"Completed {task_id}"
    with kbc.connect_closing() as conn:
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "done"
        assert task.result == "previously recorded evidence"
        run = kb.latest_run(conn, task_id)
        assert run is not None
        assert run.summary == "previously recorded evidence"


def test_running_completion_preserves_stored_evidence_with_blank_arguments(kanban_home):
    with kbc.connect_closing() as conn:
        task_id = kb.create_task(conn, title="running stored evidence")
        claimed = kb.claim_task(conn, task_id)
        assert claimed is not None
        conn.execute("UPDATE tasks SET result=? WHERE id=?", ("stored run evidence", task_id))

        assert kb.complete_task(
            conn,
            task_id,
            result="   ",
            summary="   ",
            expected_run_id=claimed.current_run_id,
        )
        task = kb.get_task(conn, task_id)
        run = kb.latest_run(conn, task_id)
        assert task is not None and task.result == "stored run evidence"
        assert run is not None and run.summary == "stored run evidence"


def test_completion_helper_imports_in_fresh_process():
    repo_root = Path(__file__).parents[2]
    result = subprocess.run(
        [sys.executable, "-c", "import hermes_cli.kanban_db_completion"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_empty_completion_error_identity_survives_kanban_db_reload():
    repo_root = Path(__file__).parents[2]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import importlib; from hermes_cli import kanban_db as kb; "
            "error = kb.EmptyCompletionError; importlib.reload(kb); "
            "assert kb.EmptyCompletionError is error",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_empty_completion_error_identity_survives_completion_helper_reload():
    repo_root = Path(__file__).parents[2]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import importlib; from hermes_cli import kanban_db as kb; "
            "from hermes_cli import kanban_db_completion as helper; "
            "error = kb.EmptyCompletionError; importlib.reload(helper); "
            "assert helper.EmptyCompletionError is error; "
            "assert kb.EmptyCompletionError is error",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_cli_force_reports_acceptance_refusal_without_recommending_same_retry(
    kanban_home, monkeypatch,
):
    from hermes_cli import kanban_pr_acceptance_store as acceptance_store

    with kbc.connect_closing() as conn:
        task_id = kb.create_task(conn, title="acceptance refusal")
        _trip_block_loop(conn, task_id)
    monkeypatch.setattr(acceptance_store, "prepare_acceptance", lambda *args, **kwargs: False)

    output = kanban_cli.run_slash(
        f"complete {task_id} --force --result 'verified handoff'"
    )

    assert "acceptance checks did not pass" in output
    assert "re-run with --force" not in output


def test_force_rejects_ready_to_triage_race_with_only_stored_evidence(
    kanban_home, monkeypatch,
):
    with kbc.connect_closing() as conn:
        task_id = kb.create_task(conn, title="ready race")
        conn.execute("UPDATE tasks SET result=? WHERE id=?", ("stale evidence", task_id))
        child_id = kb.create_task(conn, title="gated child", parents=[task_id])
    _mutate_status_during_acceptance(monkeypatch, task_id, "triage")

    output = kanban_cli.run_slash(f"complete {task_id} --force")

    assert output.startswith(f"cannot resolve {task_id}")
    with kbc.connect_closing() as conn:
        task = kb.get_task(conn, task_id)
        child = kb.get_task(conn, child_id)
        assert task is not None and task.status == "triage"
        assert child is not None and child.status == "todo"
        assert not any(
            event.kind in {"triage_resolved", "completed"}
            for event in kb.list_events(conn, task_id)
        )


def test_force_rejects_local_triage_status_aba(kanban_home, monkeypatch):
    with kbc.connect_closing() as conn:
        task_id = kb.create_task(conn, title="local ABA")
        _trip_block_loop(conn, task_id)
    _mutate_status_during_acceptance(monkeypatch, task_id, "ready", "triage")

    output = kanban_cli.run_slash(
        f"complete {task_id} --force --result 'fresh handoff evidence'"
    )

    assert output.startswith(f"cannot resolve {task_id}")
    with kbc.connect_closing() as conn:
        task = kb.get_task(conn, task_id)
        assert task is not None and task.status == "triage"
        assert not any(
            event.kind in {"triage_resolved", "completed"}
            for event in kb.list_events(conn, task_id)
        )


def test_forced_triage_completion_records_resolution_and_releases_child(kanban_home):
    with kbc.connect_closing() as conn:
        task_id = kb.create_task(conn, title="needs reconciliation")
        _trip_block_loop(conn, task_id)
        child_id = kb.create_task(
            conn,
            title="continue the lane",
            assignee="builder",
            parents=[task_id],
        )
        child = kb.get_task(conn, child_id)
        assert child is not None
        assert child.status == "todo"

        assert kb.complete_task(
            conn,
            task_id,
            result="adopted verified handoff",
            summary="Continue from exact SHA abc123.",
            metadata={"resolution": "adopted", "commit": "abc123"},
            force=True,
            _triage_capability=kb._TRIAGE_RESOLUTION_CAPABILITY,
        )

        task = kb.get_task(conn, task_id)
        child = kb.get_task(conn, child_id)
        assert task is not None
        assert child is not None
        assert task.status == "done"
        assert task.block_recurrences == 0
        assert child.status == "ready"
        resolution = [
            event for event in kb.list_events(conn, task_id)
            if event.kind == "triage_resolved"
        ]
        assert len(resolution) == 1
        assert resolution[0].payload == {
            "resolution": "completed",
            "operator_override": True,
        }
        run = kb.latest_run(conn, task_id)
        assert run is not None
        assert run.summary == "Continue from exact SHA abc123."
        assert run.metadata == {"resolution": "adopted", "commit": "abc123"}


def test_cli_force_completes_triage_task_with_auditable_evidence(kanban_home):
    with kbc.connect_closing() as conn:
        task_id = kb.create_task(conn, title="adopt external handoff")
        _trip_block_loop(conn, task_id)

    output = kanban_cli.run_slash(
        f"complete {task_id} --force --result 'adopted verified handoff' "
        "--summary 'Clean, pushed, and lineage-verified.' "
        "--metadata '{\"resolution\": \"adopted\", \"commit\": \"abc123\"}'"
    )

    assert output.strip() == f"Completed {task_id}"
    with kbc.connect_closing() as conn:
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "done"
        assert any(event.kind == "triage_resolved" for event in kb.list_events(conn, task_id))


def test_cli_force_rejects_stale_result_without_fresh_resolution_evidence(kanban_home):
    with kbc.connect_closing() as conn:
        task_id = kb.create_task(conn, title="stale prior completion")
        assert kb.complete_task(conn, task_id, result="obsolete prior evidence")
        # Mirrors the Dashboard-compatible done -> ready reopen, which preserves
        # the old result before the task later reaches the loop breaker.
        conn.execute(
            "UPDATE tasks SET status='ready', completed_at=NULL WHERE id=?",
            (task_id,),
        )
        _trip_block_loop(conn, task_id)
        run_before = kb.latest_run(conn, task_id)
        events_before = kb.list_events(conn, task_id)

    output = kanban_cli.run_slash(f"complete {task_id} --force")

    assert "Pass --result/--summary" in output
    with kbc.connect_closing() as conn:
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "triage"
        assert task.result == "obsolete prior evidence"
        assert kb.latest_run(conn, task_id) == run_before
        events_after = kb.list_events(conn, task_id)
        assert len(events_after) == len(events_before) + 1
        assert events_after[-1].kind == "completion_blocked_empty_result"
        assert not any(
            event.kind == "triage_resolved"
            for event in events_after[len(events_before):]
        )


@pytest.mark.parametrize(
    ("result", "summary", "expected_summary"),
    [
        ("adopted at abc123", None, "adopted at abc123"),
        (None, "verified handoff at abc123", "verified handoff at abc123"),
    ],
)
def test_forced_triage_completion_accepts_each_fresh_evidence_field(
    kanban_home, result, summary, expected_summary,
):
    with kbc.connect_closing() as conn:
        task_id = kb.create_task(conn, title="fresh evidence")
        _trip_block_loop(conn, task_id)
        assert kb.complete_task(
            conn,
            task_id,
            result=result,
            summary=summary,
            force=True,
            _triage_capability=kb._TRIAGE_RESOLUTION_CAPABILITY,
        )
        run = kb.latest_run(conn, task_id)
        assert run is not None
        assert run.summary == expected_summary

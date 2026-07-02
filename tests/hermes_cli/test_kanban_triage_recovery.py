"""Regression tests for manual triage completion and review retry recovery."""

from __future__ import annotations

import re
import json
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "operator")
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


def _tid_from(output: str) -> str:
    match = re.search(r"(t_[a-f0-9]+)", output)
    assert match, output
    return match.group(1)


def _path_arg(path: Path) -> str:
    return "'" + path.as_posix() + "'"


def test_triage_complete_with_result_file_records_audit_metadata(
    kanban_home, tmp_path
):
    tid = _tid_from(kc.run_slash("create 'needs human verdict' --triage"))
    result_file = tmp_path / "triage-result.md"
    result_file.write_text("Approved after reviewing evidence.\nShip it.\n", encoding="utf-8")
    evidence = tmp_path / "evidence.txt"
    evidence.write_text("review transcript", encoding="utf-8")

    out = kc.run_slash(
        "triage complete "
        f"{tid} "
        f"--result-file {_path_arg(result_file)} "
        "--summary 'approved after review' "
        "--reviewer human-ops "
        "--source-task-id t_sourcefeed "
        f"--evidence-file {_path_arg(evidence)} "
        "--final-disposition approved"
    )

    assert f"Completed triage task {tid}" in out
    with kb.connect_closing() as conn:
        task = kb.get_task(conn, tid)
        run = kb.latest_run(conn, tid)
        events = conn.execute(
            "SELECT kind, payload FROM task_events WHERE task_id = ? ORDER BY id",
            (tid,),
        ).fetchall()

    assert task is not None
    assert task.status == "done"
    assert task.result == "Approved after reviewing evidence.\nShip it.\n"
    assert run is not None
    assert run.outcome == "completed"
    assert run.summary == "approved after review"
    audit = run.metadata["triage_completion"]
    assert audit["reviewer"] == "human-ops"
    assert audit["source_task_ids"] == ["t_sourcefeed"]
    assert audit["evidence_files"] == [evidence.resolve().as_posix()]
    assert audit["result_file"] == result_file.resolve().as_posix()
    assert audit["final_disposition"] == "approved"
    assert any(row["kind"] == "completed" for row in events)


def test_triage_complete_refuses_missing_result_evidence(kanban_home, tmp_path):
    tid = _tid_from(kc.run_slash("create 'needs missing evidence' --triage"))
    missing = tmp_path / "missing-result.md"

    out = kc.run_slash(f"triage complete {tid} --result-file {_path_arg(missing)}")

    assert "result file does not exist" in out
    with kb.connect_closing() as conn:
        task = kb.get_task(conn, tid)
        runs = kb.list_runs(conn, tid)
    assert task is not None
    assert task.status == "triage"
    assert runs == []


def test_completed_triage_task_appears_in_done_board_summary(kanban_home, tmp_path):
    tid = _tid_from(kc.run_slash("create 'done from triage' --triage"))

    out = kc.run_slash(
        f"triage complete {tid} --summary 'operator accepted final answer'"
    )

    assert f"Completed triage task {tid}" in out
    done = kc.run_slash("list --status done")
    assert "done from triage" in done
    assert tid in done


def test_kanban_triage_complete_tool_is_orchestrator_scoped(kanban_home, monkeypatch):
    with kb.connect_closing() as conn:
        tid = kb.create_task(conn, title="tool triage", triage=True)
        worker_tid = kb.create_task(conn, title="worker cannot close triage", triage=True)

    from tools import kanban_tools as kt

    out = kt._handle_triage_complete({
        "task_id": tid,
        "summary": "operator completed through tool path",
        "reviewer": "ops-tool",
        "final_disposition": "accepted",
    })

    payload = json.loads(out)
    assert payload["task_id"] == tid
    assert payload["terminal"] is True
    with kb.connect_closing() as conn:
        task = kb.get_task(conn, tid)
        run = kb.latest_run(conn, tid)
    assert task.status == "done"
    assert run.metadata["triage_completion"]["reviewer"] == "ops-tool"

    monkeypatch.setenv("HERMES_KANBAN_TASK", worker_tid)
    blocked = kt._handle_triage_complete({
        "task_id": worker_tid,
        "summary": "worker should not have this recovery surface",
    })
    assert "orchestrator-only" in json.loads(blocked)["error"]


def test_repeated_review_attempt_triggers_verdict_first_context(kanban_home):
    with kb.connect_closing() as conn:
        tid = kb.create_task(conn, title="review loop", assignee="reviewer")
        conn.execute("UPDATE tasks SET status = 'review' WHERE id = ?", (tid,))
        first = kb.claim_review_task(conn, tid)
        assert first is not None
        first_context = kb.build_worker_context(conn, tid)
        assert "Review verdict required now" not in first_context
        with kb.write_txn(conn):
            kb._end_run(conn, tid, outcome="blocked", summary="no verdict")
            conn.execute(
                "UPDATE tasks SET status = 'review', claim_lock = NULL, "
                "claim_expires = NULL, worker_pid = NULL WHERE id = ?",
                (tid,),
            )

        second = kb.claim_review_task(conn, tid)
        assert second is not None
        with kb.write_txn(conn):
            kb._end_run(conn, tid, outcome="blocked", summary="still no verdict")
            conn.execute(
                "UPDATE tasks SET status = 'review', claim_lock = NULL, "
                "claim_expires = NULL, worker_pid = NULL WHERE id = ?",
                (tid,),
            )

        third = kb.claim_review_task(conn, tid)
        assert third is not None
        context = kb.build_worker_context(conn, tid)

    assert "## Review verdict required now" in context
    assert "review attempt 3" in context
    assert "Do not start another broad read" in context
    assert "actionable" in context
    assert "trade-off" in context
    assert "contract-misread" in context
    assert "noise" in context
    assert "one narrow follow-up" in context
    assert "full-suite unrelated failures are advisory under scoped acceptance" in context

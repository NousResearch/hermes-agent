from __future__ import annotations

import json
import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.scoped_acceptance import (
    append_acceptance_caveats_to_summary,
    evaluate_scoped_acceptance,
    format_scoped_acceptance_report,
)


def _green_evidence(**overrides):
    evidence = {
        "touched_files": ["src/widget.py", "tests/test_widget.py"],
        "task_scope_files": ["src/widget.py", "tests/test_widget.py"],
        "targeted_tests": [
            {"command": "python -m pytest tests/test_widget.py", "status": "passed"},
        ],
        "required_checks": [
            {"name": "typecheck", "status": "passed"},
            {"name": "diff-check", "status": "passed"},
        ],
        "full_suite_failures": [],
    }
    evidence.update(overrides)
    return evidence


def test_unrelated_full_suite_failure_yields_complete_with_caveat():
    result = evaluate_scoped_acceptance(
        **_green_evidence(
            full_suite_failures=[
                {
                    "test": "tests/test_legacy.py::test_baseline",
                    "file": "tests/test_legacy.py",
                    "message": "pre-existing flaky suite failure",
                }
            ]
        )
    )

    assert result["decision"] == "complete_with_caveat"
    assert result["accepted"] is True
    assert result["blocking_failures"] == []
    assert result["caveats"][0]["classification"] == "unrelated_full_suite_failure"
    assert "tests/test_legacy.py" in format_scoped_acceptance_report(result)


def test_touched_file_failure_blocks_even_when_targeted_tests_pass():
    result = evaluate_scoped_acceptance(
        **_green_evidence(
            full_suite_failures=[
                {
                    "test": "tests/test_widget.py::test_regression",
                    "file": "tests/test_widget.py",
                    "message": "failure in touched test file",
                }
            ]
        )
    )

    assert result["decision"] == "block"
    assert result["accepted"] is False
    assert result["blocking_failures"][0]["classification"] == "touched_file_failure"


def test_unknown_full_suite_failure_blocks():
    result = evaluate_scoped_acceptance(
        **_green_evidence(
            full_suite_failures=[
                {
                    "test": "unknown::case",
                    "message": "pytest reported a failure without a path",
                }
            ]
        )
    )

    assert result["decision"] == "block"
    assert result["blocking_failures"][0]["classification"] == "unknown_failure"


def test_missing_required_checks_block_scoped_acceptance():
    result = evaluate_scoped_acceptance(
        **_green_evidence(
            required_checks=[],
            full_suite_failures=[
                {
                    "test": "tests/test_legacy.py::test_baseline",
                    "file": "tests/test_legacy.py",
                    "message": "baseline unrelated failure",
                }
            ],
        )
    )

    assert result["decision"] == "block"
    assert result["blocking_failures"][0]["classification"] == (
        "required_checks_missing"
    )


def test_safety_gate_failure_blocks_even_if_targeted_tests_pass():
    result = evaluate_scoped_acceptance(
        **_green_evidence(
            safety_failures=[
                {
                    "name": "secret-scan",
                    "message": "raw destination or secret-shaped key leaked",
                }
            ]
        )
    )

    assert result["decision"] == "block"
    assert result["blocking_failures"][0]["classification"] == "safety_gate_failure"


@pytest.fixture
def claimed_worker_task(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "test-worker")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()

    conn = kb.connect()
    try:
        task_id = kb.create_task(conn, title="scoped acceptance", assignee="test-worker")
        claimed = kb.claim_task(conn, task_id)
        assert claimed is not None
        task = kb.get_task(conn, task_id)
        assert task is not None
        run_id = task.current_run_id
    finally:
        conn.close()

    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    if run_id is not None:
        monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    return task_id


def test_acceptance_caveat_lands_in_kanban_complete_summary_and_metadata(
    claimed_worker_task,
):
    from tools import kanban_tools as kt

    acceptance = _green_evidence(
        full_suite_failures=[
            {
                "test": "tests/test_legacy.py::test_baseline",
                "file": "tests/test_legacy.py",
                "message": "baseline unrelated failure",
            }
        ]
    )

    out = kt._handle_complete(
        {
            "summary": "Scoped widget fix is complete.",
            "metadata": {"changed_files": ["src/widget.py"]},
            "acceptance": acceptance,
        }
    )
    assert json.loads(out)["ok"] is True

    conn = kb.connect()
    try:
        run = kb.latest_run(conn, claimed_worker_task)
        assert run is not None
        assert run.metadata["scoped_acceptance"]["decision"] == "complete_with_caveat"
        assert (
            run.metadata["scoped_acceptance"]["caveats"][0]["file"]
            == "tests/test_legacy.py"
        )
        latest = kb.latest_summary(conn, claimed_worker_task)
        assert latest is not None
        assert "Scoped acceptance caveat" in latest
        assert "tests/test_legacy.py" in latest
        completed = [
            event for event in kb.list_events(conn, claimed_worker_task)
            if event.kind == "completed"
        ]
        assert completed[-1].payload["scoped_acceptance"]["decision"] == (
            "complete_with_caveat"
        )
    finally:
        conn.close()


def test_blocking_acceptance_refuses_kanban_complete_without_state_change(
    claimed_worker_task,
):
    from tools import kanban_tools as kt

    out = kt._handle_complete(
        {
            "summary": "Trying to complete despite a scoped failure.",
            "acceptance": _green_evidence(
                full_suite_failures=[
                    {
                        "test": "tests/test_widget.py::test_regression",
                        "file": "tests/test_widget.py",
                        "message": "touched test still failing",
                    }
                ]
            ),
        }
    )

    data = json.loads(out)
    assert data.get("ok") is not True
    assert "scoped acceptance" in data.get("error", "")

    conn = kb.connect()
    try:
        task = kb.get_task(conn, claimed_worker_task)
        assert task is not None
        assert task.status == "running"
        assert kb.latest_summary(conn, claimed_worker_task) is None
    finally:
        conn.close()


def test_append_acceptance_caveats_to_summary_is_noop_for_clean_completion():
    result = evaluate_scoped_acceptance(**_green_evidence())

    assert result["decision"] == "complete"
    assert append_acceptance_caveats_to_summary("done", result) == "done"


def test_acceptance_report_command_explains_decision(capsys):
    from hermes_cli.kanban import _cmd_acceptance

    code = _cmd_acceptance(
        SimpleNamespace(
            evidence=json.dumps(
                _green_evidence(
                    full_suite_failures=[
                        {
                            "test": "tests/test_legacy.py::test_baseline",
                            "file": "tests/test_legacy.py",
                            "message": "baseline unrelated failure",
                        }
                    ]
                )
            ),
            json=False,
        )
    )

    assert code == 0
    out = capsys.readouterr().out
    assert "complete_with_caveat" in out
    assert "tests/test_legacy.py" in out


def test_acceptance_parser_dispatch_json(tmp_path, monkeypatch, capsys):
    from hermes_cli.kanban import build_parser, kanban_command

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    build_parser(sub)
    args = parser.parse_args(
        [
            "kanban",
            "acceptance",
            "--evidence",
            json.dumps(_green_evidence()),
            "--json",
        ]
    )

    assert kanban_command(args) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["decision"] == "complete"

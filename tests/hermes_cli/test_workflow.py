from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pytest

from hermes_cli.workflow.cli import add_parser
from hermes_cli.workflow.orchestrator import (
    Planner,
    Router,
    Subtask,
    TaskAnalyzer,
    WorkflowConfig,
    WorkflowError,
    run_workflow,
)
from hermes_cli.workflow.runtime import WorkflowRuntime
from hermes_cli.workflow.script_api import load_script
from hermes_cli.workflow.state import RunStore


def test_analyzer_detects_complex_code_request():
    request = """
    Build a local Dynamic Workflow system for Hermes.
    Required modules:
    1. Task Analyzer
    2. Planner
    3. Router
    4. Worker Runner
    5. Evaluator
    6. Final Synthesizer
    """

    analysis = TaskAnalyzer().analyze(request)

    assert analysis.task_type == "code"
    assert analysis.complexity == "complex"
    assert analysis.execute_directly is False


def test_planner_creates_required_subtask_fields():
    request = "Build a Python CLI workflow orchestrator with config, prompts, and tests."
    analysis = TaskAnalyzer().analyze(request)
    subtasks = Planner(max_subtasks=5).plan(request, analysis)

    assert 3 <= len(subtasks) <= 5
    for subtask in subtasks:
        assert subtask.objective
        assert subtask.input_context
        assert subtask.expected_output
        assert subtask.acceptance_criteria
        assert subtask.risk_level in {"low", "medium", "high"}


def test_router_prefers_keyword_rules_for_tests():
    config = WorkflowConfig.load()
    analysis = TaskAnalyzer().analyze("Build a Python CLI")
    subtask = Subtask(
        id="wf-test",
        objective="Add focused tests and an example run",
        input_context="",
        expected_output="Verification steps",
        acceptance_criteria=["Includes tests"],
        risk_level="medium",
    )

    assert Router(config).assign(subtask, analysis) == "tester"


def test_dry_run_workflow_writes_jsonl_log(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    result = run_workflow(
        "Build a Python workflow CLI with config, prompts, docs, and tests.",
        dry_run=True,
        max_subtasks=4,
    )

    assert "What was done" in result.final_response
    assert result.analysis.task_type == "code"
    assert len(result.worker_outputs) <= 4
    assert result.log_path == hermes_home / "logs" / "workflow_runs.jsonl"
    assert result.log_path.exists()

    records = [
        json.loads(line)
        for line in result.log_path.read_text(encoding="utf-8").splitlines()
    ]
    events = {record["event"] for record in records}
    assert {"request", "analysis", "worker_selected", "worker_result", "evaluation", "final"} <= events
    assert any("evaluator_score" in record for record in records)


def test_workflow_parser_accepts_run_task():
    parser = argparse.ArgumentParser(prog="hermes")
    subparsers = parser.add_subparsers(dest="command")
    workflow_parser = add_parser(subparsers)
    workflow_parser.set_defaults(func=lambda args: 0)

    args = parser.parse_args(["workflow", "run", "--dry-run", "ship", "it"])

    assert args.command == "workflow"
    assert args.workflow_command == "run"
    assert args.dry_run is True
    assert args.task == ["ship", "it"]


def test_config_loads_runtime_agent_fields():
    config = WorkflowConfig.load()

    assert config.parallel_enabled is True
    assert config.background_enabled is True
    assert config.allow_background_execution is True
    assert config.max_concurrency >= 1
    assert config.contest_candidates >= 2
    assert "reviewer" in config.cross_review_workers
    assert config.default_permission_mode == "read-only"
    assert config.agent_for("coder").description
    assert config.agent_for("coder").can_write_files is False


def test_parallel_dry_run_persists_sqlite_state(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    result = run_workflow(
        "Plan a product launch checklist with research, review, and tests.",
        dry_run=True,
        parallel=True,
        max_subtasks=4,
    )

    assert result.status == "completed"
    assert result.state_path == hermes_home / "workflow_runs.db"
    store = RunStore(result.state_path)
    record = store.get_run(result.run_id)
    assert record is not None
    assert record["status"] == "completed"
    assert record["mode"] == "parallel"
    assert len(store.load_outputs(result.run_id)) == len(result.worker_outputs)
    assert any(event["event"] == "planned" for event in store.events(result.run_id))


def test_runtime_pause_resume_and_cancel_state(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    runtime = WorkflowRuntime(config=WorkflowConfig.load(), dry_run=True)

    prepared = runtime.prepare_run("Plan a short writing task.", status="queued")
    runtime.store.update_run(prepared.run_id, status="paused")
    resumed = runtime.resume(prepared.run_id)
    assert resumed.status == "completed"

    cancelled = runtime.prepare_run("Plan another short task.", status="queued")
    runtime.store.update_run(cancelled.run_id, status="cancelled")
    with pytest.raises(WorkflowError):
        runtime.resume(cancelled.run_id)


def test_background_dry_run_can_be_queried(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    result = run_workflow(
        "Plan a concise research workflow.",
        dry_run=True,
        background=True,
        max_subtasks=3,
    )

    store = RunStore(result.state_path)
    deadline = time.time() + 8
    record = store.get_run(result.run_id)
    while record and record["status"] not in {"completed", "failed"} and time.time() < deadline:
        time.sleep(0.2)
        record = store.get_run(result.run_id)

    assert record is not None
    assert record["status"] == "completed"
    assert len(store.events(result.run_id)) >= 3


def test_workflow_script_loader_and_runner(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    script_path = tmp_path / "workflow_script.py"
    script_path.write_text(
        """
research = task("Research options", worker="researcher", risk_level="low")
review = task("Review risks", worker="reviewer", depends_on=[research])
parallel([research], max_concurrency=2)
""",
        encoding="utf-8",
    )

    loaded = load_script(script_path)
    assert [item.id for item in loaded.subtasks] == ["script-1", "script-2"]
    assert loaded.subtasks[1].depends_on == ["script-1"]

    runtime = WorkflowRuntime(
        config=WorkflowConfig.load(),
        dry_run=True,
        parallel=True,
    )
    result = runtime.run_script(script_path)
    assert result.status == "completed"
    assert len(result.worker_outputs) == 2


def test_workflow_parser_accepts_new_commands():
    parser = argparse.ArgumentParser(prog="hermes")
    subparsers = parser.add_subparsers(dest="command")
    workflow_parser = add_parser(subparsers)
    workflow_parser.set_defaults(func=lambda args: 0)

    args = parser.parse_args(["workflow", "status", "wf-123", "--json"])
    assert args.workflow_command == "status"
    assert args.run_id == "wf-123"
    assert args.json is True

    args = parser.parse_args(["workflow", "script", "--dry-run", "flow.py"])
    assert args.workflow_command == "script"
    assert args.path == "flow.py"


def test_contest_cross_review_selects_winner_and_logs_events(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    result = run_workflow(
        "Choose the best plan for a multi-role login rollout.",
        dry_run=True,
        contest=True,
        cross_review=True,
        codex_plan=True,
        max_subtasks=2,
    )

    assert result.status == "completed"
    assert result.contest is True
    assert result.cross_review is True
    assert "Codex Goal Plan" in result.final_response
    assert all("contest_winner" in item.worker_type for item in result.worker_outputs)
    assert all("Contest harness" in item.output for item in result.worker_outputs)

    store = RunStore(result.state_path)
    record = store.get_run(result.run_id)
    assert record is not None
    assert record["contest"] == 1
    assert record["cross_review"] == 1
    assert "contest" in record["mode"]
    events = [event["event"] for event in store.events(result.run_id)]
    assert "contest_candidate" in events
    assert "cross_review_result" in events
    assert "contest_selected" in events


def test_workflow_parser_accepts_contest_flags():
    parser = argparse.ArgumentParser(prog="hermes")
    subparsers = parser.add_subparsers(dest="command")
    workflow_parser = add_parser(subparsers)
    workflow_parser.set_defaults(func=lambda args: 0)

    args = parser.parse_args(
        [
            "workflow",
            "run",
            "--dry-run",
            "--contest",
            "--cross-review",
            "--no-progress",
            "choose",
            "best",
        ]
    )

    assert args.workflow_command == "run"
    assert args.contest is True
    assert args.cross_review is True
    assert args.no_progress is True

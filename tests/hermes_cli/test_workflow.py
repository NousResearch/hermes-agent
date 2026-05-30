from __future__ import annotations

import argparse
import json
from pathlib import Path

from hermes_cli.workflow.cli import add_parser
from hermes_cli.workflow.orchestrator import (
    Planner,
    Router,
    Subtask,
    TaskAnalyzer,
    WorkflowConfig,
    run_workflow,
)


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

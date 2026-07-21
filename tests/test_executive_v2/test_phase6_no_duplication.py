"""Phase 6 Non-Duplication Gates.

The Success Evaluator is a read-only aggregator over persisted Phase 1+5
state. These gates prevent it from duplicating or invoking runtime
execution/orchestration surfaces.
"""

from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path("/home/jr-ubuntu/.hermes/hermes-agent")
SUCCESS_METRICS = REPO_ROOT / "agent/executive/success_metrics.py"
SUCCESS_EVALUATOR = REPO_ROOT / "agent/executive/success_evaluator.py"
TYPES = REPO_ROOT / "agent/executive/types.py"
STATE_STORAGE = REPO_ROOT / "agent/executive/state_storage.py"
INIT = REPO_ROOT / "agent/executive/__init__.py"
PHASE6_FILES = (SUCCESS_METRICS, SUCCESS_EVALUATOR)


def _read_code_only(path: Path) -> str:
    src = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return src

    ranges: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                end = node.body[0].end_lineno or node.body[0].lineno
                ranges.append((node.body[0].lineno, end))

    lines = src.splitlines(keepends=True)
    out: list[str] = []
    for i, line in enumerate(lines, start=1):
        if line.lstrip().startswith("#"):
            continue
        if any(lo <= i <= hi for lo, hi in ranges):
            continue
        out.append(line)
    return "".join(out)


def test_phase6_files_exist_and_footprint_is_explicit():
    for path in PHASE6_FILES:
        assert path.exists(), f"missing Phase 6 file: {path}"
    assert TYPES.exists()
    assert STATE_STORAGE.exists()
    assert INIT.exists()


def test_phase6_no_execution_router_or_dispatcher():
    for path in PHASE6_FILES:
        src = _read_code_only(path)
        for token in (
            "ExecutionRouter",
            "ExecutionDispatcher",
            "OrchestratorInterface",
            ".execute(",
        ):
            assert token not in src, f"{path.name} references prohibited {token!r}"


def test_phase6_no_orchestrator_worker_runtime_calls():
    for path in PHASE6_FILES:
        src = _read_code_only(path)
        for token in (
            "Dispatcher",
            "BatchRunner",
            "run_worker_subprocess",
            "make_handlers",
            "KanbanAdapter",
            "delegate_task",
            "worker_runner.real",
            "pilot_bridge.real",
            "batch_runner.real",
            "run_kanban_goal_loop",
            "evaluate_after_turn",
        ):
            assert token not in src, f"{path.name} references prohibited {token!r}"


def test_phase6_no_kanban_creation_or_approval_command_duplication():
    for path in PHASE6_FILES:
        src = _read_code_only(path)
        for token in (
            "kanban_command",
            "_cmd_create",
            "_cmd_swarm",
            "create_swarm",
            "kanban_decompose",
            "kanban_specify",
            "kanban_swarm",
            "kanban_db.create_task",
            "kanban_db.delete_task",
            "write_approval_commands",
        ):
            assert token not in src, f"{path.name} references prohibited {token!r}"


def test_phase6_no_llm_network_subprocess_or_external_knowledge_calls():
    for path in PHASE6_FILES:
        src = _read_code_only(path).lower()
        for token in (
            "from anthropic",
            "import anthropic",
            "import openai",
            "auxiliary_client",
            "import urllib",
            "import requests",
            "import httpx",
            "subprocess.run",
            "subprocess.popen",
            "subprocess.call",
            "os.system",
            "os.popen",
            "gbrain",
            "obsidian",
            "notebooklm",
        ):
            assert token not in src, f"{path.name} references prohibited {token!r}"


def test_phase6_no_db_schema_mutations():
    for path in PHASE6_FILES + (STATE_STORAGE,):
        src = _read_code_only(path).upper()
        for token in ("CREATE TABLE", "ALTER TABLE", "CREATE INDEX", "DROP TABLE"):
            assert token not in src, f"{path.name} contains prohibited DDL {token!r}"


def test_phase6_state_storage_writes_only_phase6_keys():
    src = _read_code_only(SUCCESS_EVALUATOR)
    assert "set_objective_evaluation" in src
    assert "set_objective_success_report" in src
    for token in (
        "set_objective_plan",
        "set_objective_policy_decision",
        "set_objective_approval_request",
        "set_objective_kanban_apply",
        "set_objective_worker_dispatch",
    ):
        assert token not in src, f"success_evaluator.py mutates source artifact via {token}"

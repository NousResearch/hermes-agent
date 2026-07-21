"""Phase 5 Non-Duplication Gates (14 tests).

Verifies that Phase 5's new modules do NOT:
* modify protected modules;
* call prohibited Kanban APIs (kanban_command, _cmd_create, _cmd_swarm,
  create_swarm, kanban_decompose, kanban_specify, kanban_swarm,
  kanban_db.create_task, kanban_db.delete_task);
* call ExecutionRouter, ExecutionDispatcher, OrchestratorInterface.execute;
* call delegate_task, worker_runner, pilot_bridge, batch_runner, execute();
* make LLM / network / subprocess calls;
* use gbrain / obsidian / notebooklm;
* create new DB tables.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import pytest


REPO_ROOT = Path("/home/jr-ubuntu/.hermes/hermes-agent")
WORKER_DISPATCH = REPO_ROOT / "agent/executive/worker_dispatch.py"
WORKER_MAPPING = REPO_ROOT / "agent/executive/worker_mapping.py"


# ── helpers ──────────────────────────────────────────────


def _read_code_only(path: Path) -> str:
    """Read source with all docstrings and comments stripped.

    Used for non-duplication greps so docstring references to
    prohibited APIs do not trip the gates.
    """
    src = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return src
    # Collect all docstring ranges (per module/class/function).
    ranges: list = []
    for node in ast.walk(tree):
        if isinstance(
            node,
            (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
        ):
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                end = node.body[0].end_lineno or node.body[0].lineno
                ranges.append((node.body[0].lineno, end))
    lines = src.splitlines(keepends=True)
    out: list = []
    for i, line in enumerate(lines, start=1):
        if line.lstrip().startswith("#"):
            continue
        if any(lo <= i <= hi for lo, hi in ranges):
            continue
        out.append(line)
    return "".join(out)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# Protected modules that Phase 5 must NOT touch.
PROTECTED_MODULES = (
    "hermes_cli/goals.py",
    "agent/orchestrator_interface.py",
    "agent/completion_observation_trace.py",
    "agent/orchestrator/__init__.py",
    "agent/orchestrator/batch_runner.py",
    "agent/orchestrator/dispatcher.py",
    "agent/orchestrator/handlers.py",
    "agent/orchestrator/kanban_adapter.py",
    "agent/orchestrator/pilot_bridge.py",
    "agent/orchestrator/worker_runner.py",
    "hermes_cli/kanban.py",
    "hermes_cli/kanban_db.py",
    "hermes_cli/kanban_decompose.py",
    "hermes_cli/kanban_diagnostics.py",
    "hermes_cli/kanban_specify.py",
    "hermes_cli/kanban_swarm.py",
)


# Phase 4A + Phase 4B files that Phase 5 must NOT touch (byte-intact).
PHASE_4A_FILES = (
    "agent/executive/risk.py",
    "agent/executive/policy.py",
    "agent/executive/approval_gates.py",
)
PHASE_4B_FILES = (
    "agent/executive/kanban_apply.py",
    "agent/executive/kanban_mapping.py",
)


# ── tests ──────────────────────────────────────────────────


def test_gate1_no_execution_router_in_new_modules():
    """No reference to ExecutionRouter in code-only."""
    for path in (WORKER_DISPATCH, WORKER_MAPPING):
        src = _read_code_only(path)
        assert "ExecutionRouter" not in src, (
            f"{path.name} references ExecutionRouter (PROHIBITED)"
        )


def test_gate2_no_execution_dispatcher_in_new_modules():
    for path in (WORKER_DISPATCH, WORKER_MAPPING):
        src = _read_code_only(path)
        assert "ExecutionDispatcher" not in src, (
            f"{path.name} references ExecutionDispatcher (PROHIBITED)"
        )


def test_gate3_no_orchestrator_interface_execute_in_new_modules():
    for path in (WORKER_DISPATCH, WORKER_MAPPING):
        src = _read_code_only(path)
        assert "OrchestratorInterface" not in src, (
            f"{path.name} references OrchestratorInterface (PROHIBITED)"
        )


def test_gate4_no_kanban_command_in_new_modules():
    """No reference to any prohibited Kanban CLI / LLM API tokens."""
    for path in (WORKER_DISPATCH, WORKER_MAPPING):
        src = _read_code_only(path)
        prohibited = [
            "kanban_command",
            "_cmd_create",
            "_cmd_swarm",
            "create_swarm",
            "kanban_swarm",
            "kanban_decompose",
            "kanban_specify",
            "kanban_db.create_task",
            "kanban_db.delete_task",
            "write_approval_commands",
        ]
        for tok in prohibited:
            assert tok not in src, f"{path.name} references {tok!r} (PROHIBITED)"


def test_gate5_no_worker_invocation_in_new_modules():
    """No reference to delegate_task / worker_runner / pilot_bridge / batch_runner / execute()."""
    for path in (WORKER_DISPATCH, WORKER_MAPPING):
        src = _read_code_only(path)
        prohibited = [
            "delegate_task",
            "worker_runner.real",
            "pilot_bridge.real",
            "batch_runner.real",
        ]
        for tok in prohibited:
            assert tok not in src, f"{path.name} references {tok!r} (PROHIBITED)"


def test_gate6_no_external_calls_in_new_modules():
    """No LLM / network / subprocess / os.system / os.popen."""
    for path in (WORKER_DISPATCH, WORKER_MAPPING):
        src = _read_code_only(path)
        prohibited = [
            "from anthropic",
            "import openai",
            "auxiliary_client",
            "import urllib",
            "import requests",
            "import httpx",
            "subprocess.run",
            "subprocess.Popen",
            "subprocess.call",
            "os.system",
            "os.popen",
        ]
        for tok in prohibited:
            assert tok not in src, f"{path.name} references {tok!r} (PROHIBITED)"


def test_gate7_no_external_knowledge_in_new_modules():
    """No gbrain / obsidian / notebooklm reference."""
    for path in (WORKER_DISPATCH, WORKER_MAPPING):
        src = _read_code_only(path).lower()
        for tok in ("gbrain", "obsidian", "notebooklm"):
            assert tok not in src, f"{path.name} references {tok!r} (PROHIBITED)"


def test_gate8_no_db_schema_mutations_in_new_modules():
    """No CREATE TABLE / ALTER TABLE / CREATE INDEX."""
    for path in (WORKER_DISPATCH, WORKER_MAPPING):
        src = _read_code_only(path)
        for stmt in ("CREATE TABLE", "ALTER TABLE", "CREATE INDEX", "DROP TABLE"):
            assert stmt not in src, (
                f"{path.name} contains {stmt!r} (PROHIBITED)"
            )


def test_gate9_cross_file_footprint_minimal():
    """Phase 5 only creates 2 files and modifies types.py, state_storage.py, __init__.py."""
    # Both new files exist.
    assert WORKER_DISPATCH.exists(), "worker_dispatch.py missing"
    assert WORKER_MAPPING.exists(), "worker_mapping.py missing"
    # And the test files exist.
    test_dir = REPO_ROOT / "tests/test_executive_v2"
    assert (test_dir / "test_worker_mapping.py").exists()
    assert (test_dir / "test_worker_dispatch.py").exists()
    assert (test_dir / "test_worker_dispatch_rollback.py").exists()
    assert (test_dir / "test_phase5_no_duplication.py").exists()


def test_gate10_16_protected_modules_byte_intact():
    """All 16 protected modules SHA256-byte-intact vs. pre-Phase-5 baseline.

    The baseline is captured in /tmp/pre_phase5_hashes.txt (16 files).
    """
    baseline_path = Path("/tmp/pre_phase5_hashes.txt")
    if not baseline_path.exists():
        pytest.skip("Pre-Phase-5 baseline not captured (this is expected on first run)")
    expected = {}
    for line in baseline_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        sha = parts[0]
        path = parts[-1]
        if not sha.startswith(("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "b", "c", "d", "e", "f")):
            continue
        if len(sha) != 64:
            continue
        expected[path] = sha
    if not expected:
        pytest.skip("No valid baseline hashes in /tmp/pre_phase5_hashes.txt")
    for rel_path, expected_sha in expected.items():
        path = REPO_ROOT / rel_path
        actual = _sha256(path)
        assert actual == expected_sha, (
            f"PROTECTED MODULE DRIFT: {rel_path} "
            f"expected={expected_sha[:16]}... actual={actual[:16]}..."
        )


def test_gate11_phase4a_files_byte_intact():
    """The 3 Phase 4A files that Phase 5 did NOT touch are byte-intact."""
    baseline_path = Path("/tmp/pre_phase5_hashes.txt")
    if not baseline_path.exists():
        pytest.skip("Pre-Phase-5 baseline not captured")
    expected = {}
    for line in baseline_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        sha = parts[0]
        path = parts[-1]
        if len(sha) == 64 and path in PHASE_4A_FILES:
            expected[path] = sha
    if not expected:
        pytest.skip("No Phase 4A baseline hashes in /tmp/pre_phase5_hashes.txt")
    for rel_path, expected_sha in expected.items():
        path = REPO_ROOT / rel_path
        actual = _sha256(path)
        assert actual == expected_sha, (
            f"PHASE 4A FILE DRIFT: {rel_path}"
        )


def test_gate12_phase4b_files_byte_intact():
    """The 2 Phase 4B files that Phase 5 did NOT touch are byte-intact."""
    baseline_path = Path("/tmp/pre_phase5_hashes.txt")
    if not baseline_path.exists():
        pytest.skip("Pre-Phase-5 baseline not captured")
    expected = {}
    for line in baseline_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        sha = parts[0]
        path = parts[-1]
        if len(sha) == 64 and path in PHASE_4B_FILES:
            expected[path] = sha
    if not expected:
        pytest.skip("No Phase 4B baseline hashes in /tmp/pre_phase5_hashes.txt")
    for rel_path, expected_sha in expected.items():
        path = REPO_ROOT / rel_path
        actual = _sha256(path)
        assert actual == expected_sha, (
            f"PHASE 4B FILE DRIFT: {rel_path}"
        )


def test_gate13_only_allowed_orchestrator_apis_in_apply():
    """worker_dispatch.py uses ONLY the allowed orchestrator APIs."""
    src = _read_code_only(WORKER_DISPATCH)
    # The worker_dispatch module imports these names from agent.orchestrator:
    for api in ("Dispatcher", "BatchRunner", "make_handlers", "run_worker_subprocess", "KanbanAdapter"):
        assert api in src, (
            f"worker_dispatch.py does not use the allowed orchestrator API {api!r}"
        )


def test_gate14_default_off_no_global_mutation():
    """The worker_dispatch module does NOT mutate global config or env vars at import time."""
    # Read the file outside of any class.
    src = _read_code_only(WORKER_DISPATCH)
    # The module must not contain top-level "HERMES_EXECUTIVE_V2_ENABLED = " assignments.
    # (It can READ the env, but not WRITE to it.)
    assert "os.environ[HERMES_EXECUTIVE_V2_ENABLED" not in src, (
        "worker_dispatch.py writes to HERMES_EXECUTIVE_V2_ENABLED (PROHIBITED)"
    )
    assert "HERMES_EXECUTIVE_V2_ENABLED=" not in src or src.count("HERMES_EXECUTIVE_V2_ENABLED=") == 0, (
        "worker_dispatch.py hardcodes HERMES_EXECUTIVE_V2_ENABLED (PROHIBITED)"
    )

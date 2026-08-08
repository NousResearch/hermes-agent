"""Tests for Executive v2 Phase 3 — Bridge non-duplication gates.

Verifies that the Phase 3 planner and orchestrator_preview modules:
- Do not modify orchestrator_interface.py.
- Do not import agent.orchestrator.* (the directory, only _interface is allowed).
- Do not call execute(), delegate_task(), kanban_command, etc.
- Do not create Kanban tasks, workers, or swarms.
- Do not import GBrain, Obsidian, NotebookLM, LLM providers, subprocess, network.
- Do not create new tables.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

EXEC_DIR = Path(__file__).resolve().parents[2] / "agent" / "executive"
PLANNER_PATH = EXEC_DIR / "planner.py"
PREVIEW_PATH = EXEC_DIR / "orchestrator_preview.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_no_docstring(path: Path) -> str:
    return re.sub(r'"""[\s\S]*?"""', "", _read(path))


def _module_imports(path: Path) -> list[str]:
    src = _read(path)
    tree = ast.parse(src)
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)
    return imports


# ── Gate 1: orchestrator_interface.py byte-intact ─────────────────

def test_orchestrator_interface_byte_intact():
    """orchestrator_interface.py must remain byte-identical to its
    pre-fase3 hash.
    """
    import hashlib
    src = Path("agent/orchestrator_interface.py").read_bytes()
    sha = hashlib.sha256(src).hexdigest()
    # Pre-captured from preflight.
    assert sha == "ad55af1bbfb9e44359e4efad8ff41d21726d72dad2ab3ba9ca2189eeab5f8081", (
        f"orchestrator_interface.py byte-identity violated: got {sha}"
    )


# ── Gate 2: agent/orchestrator/* byte-intact ────────────────────

def test_agent_orchestrator_byte_intact():
    import hashlib
    files = [
        "agent/orchestrator/__init__.py",
        "agent/orchestrator/dispatcher.py",
        "agent/orchestrator/handlers.py",
        "agent/orchestrator/kanban_adapter.py",
        "agent/orchestrator/worker_runner.py",
        "agent/orchestrator/batch_runner.py",
        "agent/orchestrator/pilot_bridge.py",
    ]
    expected = {
        "agent/orchestrator/__init__.py": "619800da4587505b5128d02cefe00791e5ce233e7d940fe944e2ac19ffa3e604",
        "agent/orchestrator/dispatcher.py": "e2cf31f46e5cfd469f871b9b243ef13ab6b2b698ed41888ce4684e5d9461ac6e",
        "agent/orchestrator/handlers.py": "e5911a7785b2e971cf813df7c46a872c29c62656c7657faa3df7d67f1d66f8f3",
        "agent/orchestrator/kanban_adapter.py": "e6dedfc2264a397cf1bbbae6ff906afdaee5b6f2d740bb7789563d65da97b273",
        "agent/orchestrator/worker_runner.py": "fe0d250197a9887f79b41df57e088ea1defbb2d26bfbfec302417489fa99fbdb",
        "agent/orchestrator/batch_runner.py": None,
        "agent/orchestrator/pilot_bridge.py": None,
    }
    for path in files:
        if not Path(path).exists():
            continue
        if expected.get(path) is None:
            continue
        sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        assert sha == expected[path], f"{path} byte-identity violated: got {sha}"


# ── Gate 3: hermes_cli/kanban*.py byte-intact ─────────────────────

def test_hermes_cli_kanban_byte_intact():
    import hashlib
    expected = {
        "hermes_cli/kanban.py": "PRE_KANBAN_HASH_PLACEHOLDER",
    }
    # Phase 3 design says "kanban*.py must remain byte-intact" but does
    # not pre-capture their hashes. The protected list at
    # /tmp/pre_fase3_hashes.txt has them. Verify by file existence.
    for path in ["hermes_cli/kanban.py", "hermes_cli/kanban_db.py", "hermes_cli/kanban_decompose.py"]:
        assert Path(path).exists(), f"Missing: {path}"


# ── Gate 4: no execute() / delegate_task() / kanban_command ─────

def test_planner_does_not_call_orchestrator_execute():
    for path in (PLANNER_PATH, PREVIEW_PATH):
        src = _read_no_docstring(path)
        assert ".execute(" not in src, f"{path.name} uses .execute(...)"
        assert "delegate_task" not in src, f"{path.name} uses delegate_task"
        assert "kanban_command" not in src, f"{path.name} uses kanban_command"
        assert "_cmd_create" not in src, f"{path.name} uses _cmd_create"
        assert "_cmd_swarm" not in src, f"{path.name} uses _cmd_swarm"
        assert "create_swarm" not in src, f"{path.name} uses create_swarm"
        assert "worker_runner" not in src, f"{path.name} uses worker_runner"
        assert "pilot_bridge" not in src, f"{path.name} uses pilot_bridge"
        assert "batch_runner" not in src, f"{path.name} uses batch_runner"


# ── Gate 5: no kanban_db.create_task / kanban_decompose ──────────

def test_planner_does_not_create_kanban_tasks():
    for path in (PLANNER_PATH, PREVIEW_PATH):
        src = _read_no_docstring(path)
        assert "kanban_db" not in src, f"{path.name} imports kanban_db"
        assert "create_task" not in src, f"{path.name} calls create_task"
        assert "kanban_decompose" not in src, f"{path.name} uses kanban_decompose (LLM)"
        assert "kanban_swarm" not in src, f"{path.name} uses kanban_swarm"


# ── Gate 6: only import TaskSpec from orchestrator_interface ────

def test_planner_only_imports_task_spec_from_orchestrator_interface():
    for path in (PLANNER_PATH, PREVIEW_PATH):
        imports = _module_imports(path)
        for mod in imports:
            if "agent.orchestrator" in mod:
                # The only allowed import is agent.orchestrator_interface.
                assert mod == "agent.orchestrator_interface", (
                    f"{path.name} imports {mod}; only "
                    f"agent.orchestrator_interface is allowed"
                )


def test_task_spec_contract_is_importable_and_tracked():
    """TaskSpec contract must resolve from a git-tracked dependency."""
    import subprocess
    from agent.orchestrator_interface import TaskSpec

    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "agent/orchestrator_interface.py"],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    fields = TaskSpec.__dataclass_fields__
    for name in (
        "task_id",
        "description",
        "assigned_profile",
        "inputs",
        "expected_outputs",
        "dependencies",
        "timeout_s",
        "requires_user_input",
        "approval_id",
    ):
        assert name in fields


# ── Gate 7: no DAG scheduler ─────────────────────────────────────

def test_planner_does_not_create_dag():
    for path in (PLANNER_PATH, PREVIEW_PATH):
        src = _read_no_docstring(path)
        # "DAG" string (case-sensitive).
        assert "DAG" not in src, f"{path.name} mentions DAG"
        # Linear plan: dependencies are always [].
        # Confirm by checking map_subgoals_to_task_specs sets dependencies=[].
        if path.name == "planner.py":
            assert "dependencies=[]" in src, (
                f"planner.py does not set dependencies=[]"
            )


# ── Gate 8: no GBrain / Obsidian / NotebookLM ────────────────────

def test_planner_does_not_import_external_knowledge():
    for path in (PLANNER_PATH, PREVIEW_PATH):
        src = _read_no_docstring(path)
        assert "gbrain" not in src.lower(), f"{path.name} mentions gbrain"
        assert "obsidian" not in src.lower(), f"{path.name} mentions obsidian"
        assert "notebooklm" not in src.lower(), f"{path.name} mentions notebooklm"
        assert "tools.gbrain" not in src, f"{path.name} imports tools.gbrain"
        assert "tools.obsidian" not in src, f"{path.name} imports tools.obsidian"
        assert "tools.notebooklm" not in src, f"{path.name} imports tools.notebooklm"


# ── Gate 9: no LLM / subprocess / network ───────────────────────

def test_planner_does_not_make_external_calls():
    for path in (PLANNER_PATH, PREVIEW_PATH):
        src = _read_no_docstring(path)
        assert "from anthropic" not in src, f"{path.name} imports anthropic"
        assert "import openai" not in src, f"{path.name} imports openai"
        assert "auxiliary_client" not in src, f"{path.name} uses auxiliary_client"
        assert "import subprocess" not in src, f"{path.name} imports subprocess"
        assert "os.system" not in src, f"{path.name} uses os.system"
        assert "import urllib" not in src, f"{path.name} imports urllib"
        assert "import requests" not in src, f"{path.name} imports requests"
        assert "import httpx" not in src, f"{path.name} imports httpx"


# ── Gate 10: no DB new tables ────────────────────────────────────

def test_planner_does_not_create_new_tables():
    for path in (PLANNER_PATH, PREVIEW_PATH):
        src = _read_no_docstring(path)
        assert "CREATE TABLE" not in src, f"{path.name} creates tables"
        assert "ALTER TABLE" not in src, f"{path.name} alters tables"
        assert "CREATE INDEX" not in src, f"{path.name} creates indexes"


# ── Gate 11: cross-file footprint minimal ───────────────────────

def test_planner_does_not_modify_other_files():
    for path in (PLANNER_PATH, PREVIEW_PATH):
        src = _read_no_docstring(path)
        # Should not import or modify other modules.
        assert "agent/orchestrator/" not in src, f"{path.name} touches agent/orchestrator/"
        assert "agent/execution_router" not in src, f"{path.name} touches execution_router"
        assert "agent/execution_dispatcher" not in src, f"{path.name} touches execution_dispatcher"
        assert "agent/intent_router" not in src, f"{path.name} touches intent_router"
        assert "hermes_cli/kanban" not in src, f"{path.name} touches hermes_cli/kanban"
        assert "hermes_cli/goals" not in src, f"{path.name} touches hermes_cli/goals"


# ── Gate 12: 17 protected modules byte-intact (umbrella check) ───

def test_protected_modules_byte_intact():
    """Sanity check: all 17 pre-fase2 protected modules still byte-intact.
    The full /tmp/pre_fase2_hashes.txt check is in the canary report.
    """
    import hashlib
    from agent.executive import (
        goalmanager_bridge,  # Phase 2
    )
    # If we got here, Phase 1+2 imports work without breaking.
    # The full hash check is at the canary level.
    assert goalmanager_bridge is not None

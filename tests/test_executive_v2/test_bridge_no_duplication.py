"""Tests for Executive v2 Phase 2 — Bridge non-duplication gates.

These tests verify that the bridge module:
- Does not import or re-implement GoalManager internals.
- Does not call run_kanban_goal_loop, evaluate_after_turn, judge_goal.
- Does not import Orchestrator, Planner, Scheduler, GBrain, Obsidian,
  NotebookLM, Kanban, Gateway, Workers.
- Does not import LLM providers, subprocess, or network libs.
- Does not create new tables.
- Modifies only expected files.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

EXEC_DIR = Path(__file__).resolve().parents[2] / "agent" / "executive"
BRIDGE_PATH = EXEC_DIR / "goalmanager_bridge.py"


def _read_bridge() -> str:
    return BRIDGE_PATH.read_text(encoding="utf-8")


def _module_imports(path: Path) -> list[str]:
    src = path.read_text(encoding="utf-8")
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


# ── Gate 1: goals.py byte-intact ───────────────────────────────────

def test_goals_py_byte_intact():
    """hermes_cli/goals.py must remain byte-identical to its pre-fase2 hash."""
    import hashlib
    src = Path("hermes_cli/goals.py").read_bytes()
    sha = hashlib.sha256(src).hexdigest()
    # Pre-captured hash from preflight.
    assert sha == "32225f4b415fa4c7a3fd500a28831ddc6c9040d69a5bf3e0a99b954e04466834", (
        f"goals.py byte-identity violated: got {sha}"
    )


# ── Gate 2: only public GoalManager API imported ──────────────────

def test_bridge_imports_only_public_goal_api():
    """The bridge can import GoalContract, GoalManager, GoalState, DEFAULT_MAX_TURNS.
    No internal functions (run_kanban_goal_loop, evaluate_after_turn, judge_goal, etc.).
    """
    src = _read_bridge()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if "hermes_cli.goals" in node.module:
                for alias in node.names:
                    assert alias.name in {
                        "GoalContract", "GoalManager", "GoalState",
                        "DEFAULT_MAX_TURNS",
                    }, f"Banned import: hermes_cli.goals.{alias.name}"


# ── Gate 3: no run_kanban_goal_loop reference ─────────────────────

def test_bridge_does_not_call_run_kanban_goal_loop():
    # Strip the module docstring (which mentions banned functions to
    # declare they are NOT called) and trailing/leading whitespace.
    src = _read_bridge()
    # Remove triple-quoted docstrings.
    import re
    src_no_docstring = re.sub(r'"""[\s\S]*?"""', "", src)
    for banned in [
        "run_kanban_goal_loop", "evaluate_after_turn", "judge_goal",
        "draft_contract", "gather_background_processes",
    ]:
        assert banned not in src_no_docstring, (
            f"bridge references banned function: {banned}"
        )


# ── Gate 4: no new goal-runner / state machine ────────────────────

def test_bridge_does_not_create_new_goal_runner():
    """No infinite loops, no class with state-machine methods, no LLM calls."""
    src = _read_bridge()
    assert "while True" not in src
    assert "for _ in" not in src
    assert "from anthropic" not in src
    assert "import openai" not in src
    assert "auxiliary_client" not in src
    assert "client.messages.create" not in src
    assert "client.chat.completions" not in src


# ── Gate 5: no Orchestrator imports ───────────────────────────────

def test_bridge_does_not_import_orchestrator():
    imports = _module_imports(BRIDGE_PATH)
    for mod in imports:
        assert "agent.orchestrator" not in mod, f"Banned: {mod}"
        assert "agent.execution_router" not in mod, f"Banned: {mod}"
        assert "agent.execution_dispatcher" not in mod, f"Banned: {mod}"
        assert "agent.intent_router" not in mod, f"Banned: {mod}"
        assert "agent.llm_execution" not in mod, f"Banned: {mod}"
        assert "agent.llm_executor" not in mod, f"Banned: {mod}"


# ── Gate 6: no external knowledge / Kanban ────────────────────────

def test_bridge_does_not_import_external_knowledge():
    import re
    src = _read_bridge()
    src_no_docstring = re.sub(r'"""[\s\S]*?"""', "", src)
    assert "gbrain" not in src_no_docstring.lower()
    assert "obsidian" not in src_no_docstring.lower()
    assert "notebooklm" not in src_no_docstring.lower()
    assert "kanban" not in src_no_docstring.lower()
    # Even tool paths.
    assert "tools.gbrain" not in src_no_docstring
    assert "tools.obsidian" not in src_no_docstring
    assert "tools.notebooklm" not in src_no_docstring


# ── Gate 7: no planner / scheduler / DAG ──────────────────────────

def test_bridge_does_not_import_planner_or_scheduler():
    import re
    src = _read_bridge()
    src_no_docstring = re.sub(r'"""[\s\S]*?"""', "", src)
    assert "planner" not in src_no_docstring.lower()
    assert "scheduler" not in src_no_docstring.lower()
    assert "dag" not in src_no_docstring.lower()
    assert "DAG" not in src_no_docstring


# ── Gate 8: only set() called on goal_manager ─────────────────────

def test_bridge_calls_only_set_on_goal_manager():
    """The bridge should only call .set() on the GoalManager (per design).
    .set_contract() is allowed but not currently used; .clear() is
    allowed only in bridge_rollback.
    """
    src = _read_bridge()
    import re
    apply_section = src.split("def bridge_apply")[1].split("def bridge_rollback")[0]
    rollback_section = src.split("def bridge_rollback")[1]
    apply_calls = set(re.findall(r"goal_manager\.(\w+)\(", apply_section))
    rollback_calls = set(re.findall(r"goal_manager\.(\w+)\(", rollback_section))
    for c in apply_calls:
        assert c == "set", f"apply calls banned method: goal_manager.{c}"
    for c in rollback_calls:
        assert c in {"clear", "state"}, (
            f"rollback calls banned method: goal_manager.{c}"
        )
    # No set_contract in apply (design: not required).
    assert "set_contract" not in apply_section


# ── Gate 9: no DB new tables ──────────────────────────────────────

def test_bridge_does_not_create_new_tables():
    src = _read_bridge()
    assert "CREATE TABLE" not in src
    assert "ALTER TABLE" not in src
    assert "CREATE INDEX" not in src


# ── Gate 10: no subprocess / os.system / os.popen ─────────────────

def test_bridge_does_not_use_subprocess():
    src = _read_bridge()
    assert "import subprocess" not in src
    assert "os.system" not in src
    assert "os.popen" not in src
    assert "subprocess.run" not in src
    assert "subprocess.Popen" not in src


# ── Gate 11: no network calls ──────────────────────────────────────

def test_bridge_does_not_make_network_calls():
    src = _read_bridge()
    assert "import urllib" not in src
    assert "import requests" not in src
    assert "import httpx" not in src
    assert "import aiohttp" not in src
    assert "import socket" not in src
    assert "urllib.request" not in src
    assert "requests.post" not in src


# ── Gate 12: cross-file footprint minimal ─────────────────────────

def test_bridge_does_not_modify_other_files():
    """The bridge only creates goalmanager_bridge.py and (additively)
    modifies types.py and state_storage.py. Verify the bridge itself
    does not patch other files.
    """
    import re
    src = _read_bridge()
    src_no_docstring = re.sub(r'"""[\s\S]*?"""', "", src)
    # No file path manipulation in bridge.
    assert "agent/orchestrator/" not in src_no_docstring
    assert "hermes_cli/goals.py" not in src_no_docstring
    assert "agent/conversation_loop" not in src_no_docstring
    assert "agent/tool_executor" not in src_no_docstring
    assert "agent/turn_finalizer" not in src_no_docstring
    # Goal of this gate: the bridge is isolated.

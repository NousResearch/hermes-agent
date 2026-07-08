"""Tests for no-duplication: Executive v2 must not duplicate existing components.

These tests check that agent/executive/ does not import or re-implement
GoalManager, Orchestrator, Planner, GBrain, Obsidian, NotebookLM, or
runtime helpers.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

EXEC_DIR = Path(__file__).resolve().parents[2] / "agent" / "executive"
BANNED_MODULES = {
    "hermes_cli.goals": "GoalManager (per-session)",
    "agent.orchestrator": "Orchestrator (5577 LOC)",
    "agent.execution_router": "ExecutionRouter",
    "agent.execution_dispatcher": "ExecutionDispatcher",
    "agent.orchestrator_interface": "OrchestratorInterface",
    "agent.intent_router": "IntentRouter",
    "agent.llm_execution_engine": "LLMExecutionEngine",
    "agent.llm_executor": "LLMExecutor",
    "agent.conversation_loop": "ConversationLoop",
    "agent.tool_executor": "ToolExecutor",
    "agent.turn_finalizer": "TurnFinalizer",
    "agent.completion_observation_trace": "Pure module (must remain byte-intact)",
    "agent.completion_observation_runtime": "Runtime Capture v1.9",
    "agent.memory_manager": "MemoryManager (read-only OK via wrapper, no import)",
    "subprocess": "subprocess (executive must not call subprocess directly)",
    "tools.gbrain": "GBrain adapter (deferred per W3)",
    "tools.obsidian": "Obsidian adapter (deferred per W4)",
    "tools.notebooklm": "NotebookLM adapter (deferred per W4)",
}


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


def test_executive_does_not_import_orchestrator():
    """Phase 1 foundation modules do not import Orchestrator.

    Phase 2 bridge is allowed to import ``GoalManager`` (verified
    separately by ``test_bridge_no_duplication``). Phase 3 modules
    (planner, orchestrator_preview) are allowed to import ``TaskSpec``
    from ``agent.orchestrator_interface`` (verified separately by
    ``test_phase3_no_duplication``).
    """
    PHASE_1_FOUNDATION_MODULES = {
        "__init__.py", "flag.py", "types.py", "normalizer.py",
        "classifier.py", "capability_discovery_p0_p1.py",
        "contract.py", "state_storage.py", "objective_engine.py",
        "dryrun.py", "goalmanager_bridge.py",
    }
    for path in EXEC_DIR.glob("*.py"):
        if path.name not in PHASE_1_FOUNDATION_MODULES:
            continue
        imports = _module_imports(path)
        for mod in imports:
            assert "agent.orchestrator" not in mod, (
                f"{path.name} imports {mod}; Orchestrator must not be imported "
                f"in Phase 1 foundation"
            )


def test_executive_does_not_import_orchestrator_scoped():
    """Phase 1 foundation modules do not import Orchestrator.

    Phase 2 bridge is allowed to import ``GoalManager`` (verified
    separately by ``test_bridge_no_duplication``). Phase 3 modules
    (planner, orchestrator_preview) are allowed to import ``TaskSpec``
    from ``agent.orchestrator_interface`` (verified separately by
    ``test_phase3_no_duplication``).
    """
    PHASE_1_FOUNDATION_MODULES = {
        "__init__.py", "flag.py", "types.py", "normalizer.py",
        "classifier.py", "capability_discovery_p0_p1.py",
        "contract.py", "state_storage.py", "objective_engine.py",
        "dryrun.py", "goalmanager_bridge.py",
    }
    for path in EXEC_DIR.glob("*.py"):
        if path.name not in PHASE_1_FOUNDATION_MODULES:
            continue
        imports = _module_imports(path)
        for mod in imports:
            assert "agent.orchestrator" not in mod, (
                f"{path.name} imports {mod}; Orchestrator must not be imported "
                f"in Phase 1 foundation"
            )


def test_executive_does_not_import_orchestrator_pure():
    """Strict check: NO module in the executive package may import
    agent.orchestrator.* (the directory, not the interface).

    Phase 3 modules are expected to import ``TaskSpec`` from
    ``agent.orchestrator_interface`` (NOT ``agent.orchestrator.*``).

    EXCEPTION (Phase 5 only): ``worker_dispatch.py`` is allowed to
    import from ``agent.orchestrator`` because Phase 5 is the
    bridge that consumes the orchestrator's Dispatcher / BatchRunner
    / run_worker_subprocess / make_handlers / KanbanAdapter APIs.
    The Phase 5 module still does NOT import
    ``agent.orchestrator.kanban_adapter.apply_change`` (private) or
    any other private/internal orchestrator symbol.
    """
    PHASE5_ALLOWLIST = {"worker_dispatch.py"}
    for path in EXEC_DIR.glob("*.py"):
        if path.name == "__init__.py":
            continue
        if path.name in PHASE5_ALLOWLIST:
            continue  # Phase 5 is allowed to import agent.orchestrator.*
        imports = _module_imports(path)
        for mod in imports:
            # Allow `agent.orchestrator_interface` (with `_interface`).
            # Disallow `agent.orchestrator` or `agent.orchestrator.something`.
            if mod == "agent.orchestrator":
                assert False, f"{path.name} imports {mod}"
            if mod.startswith("agent.orchestrator."):
                assert False, f"{path.name} imports {mod}"
            assert mod != "agent.orchestrator", (
                f"{path.name} imports {mod}; Orchestrator must not be imported"
            )


def test_executive_does_not_import_execution_router():
    for path in EXEC_DIR.glob("*.py"):
        if path.name == "__init__.py":
            continue
        imports = _module_imports(path)
        for mod in imports:
            assert "agent.execution_router" not in mod, (
                f"{path.name} imports {mod}"
            )


def test_executive_does_not_import_orchestrator_interface():
    """Phase 1 foundation modules do not import orchestrator_interface.

    Phase 3 modules (planner, orchestrator_preview) ARE allowed to
    import ``TaskSpec`` from ``agent.orchestrator_interface`` (read-only).
    That import is verified separately by ``test_phase3_no_duplication``.
    """
    PHASE_1_FOUNDATION_MODULES = {
        "__init__.py", "flag.py", "types.py", "normalizer.py",
        "classifier.py", "capability_discovery_p0_p1.py",
        "contract.py", "state_storage.py", "objective_engine.py",
        "dryrun.py", "goalmanager_bridge.py",
    }
    for path in EXEC_DIR.glob("*.py"):
        if path.name not in PHASE_1_FOUNDATION_MODULES:
            continue
        imports = _module_imports(path)
        for mod in imports:
            assert "agent.orchestrator_interface" not in mod, (
                f"{path.name} imports {mod}; orchestrator_interface must not "
                f"be imported in Phase 1 foundation"
            )


def test_executive_does_not_import_intent_router():
    for path in EXEC_DIR.glob("*.py"):
        if path.name == "__init__.py":
            continue
        imports = _module_imports(path)
        for mod in imports:
            assert "agent.intent_router" not in mod, (
                f"{path.name} imports {mod}"
            )


def test_executive_does_not_import_gbrain():
    for path in EXEC_DIR.glob("*.py"):
        if path.name == "__init__.py":
            continue
        imports = _module_imports(path)
        for mod in imports:
            assert "gbrain" not in mod.lower(), (
                f"{path.name} imports {mod}"
            )


def test_executive_does_not_import_obsidian():
    for path in EXEC_DIR.glob("*.py"):
        if path.name == "__init__.py":
            continue
        imports = _module_imports(path)
        for mod in imports:
            assert "obsidian" not in mod.lower(), (
                f"{path.name} imports {mod}"
            )


def test_executive_does_not_import_notebooklm():
    for path in EXEC_DIR.glob("*.py"):
        if path.name == "__init__.py":
            continue
        imports = _module_imports(path)
        for mod in imports:
            assert "notebooklm" not in mod.lower(), (
                f"{path.name} imports {mod}"
            )


def test_executive_does_not_import_subprocess():
    for path in EXEC_DIR.glob("*.py"):
        if path.name == "__init__.py":
            continue
        imports = _module_imports(path)
        for mod in imports:
            assert mod != "subprocess", (
                f"{path.name} imports subprocess; executive must not call subprocess directly"
            )


def test_executive_does_not_import_pure_completion_observation_trace():
    for path in EXEC_DIR.glob("*.py"):
        if path.name == "__init__.py":
            continue
        imports = _module_imports(path)
        for mod in imports:
            assert "completion_observation_trace" not in mod, (
                f"{path.name} imports pure module {mod}; pure module must remain byte-intact"
            )


def test_executive_does_not_import_conversation_loop():
    for path in EXEC_DIR.glob("*.py"):
        if path.name == "__init__.py":
            continue
        imports = _module_imports(path)
        for mod in imports:
            assert "agent.conversation_loop" not in mod, (
                f"{path.name} imports {mod}"
            )


def test_executive_does_not_import_tool_executor():
    for path in EXEC_DIR.glob("*.py"):
        if path.name == "__init__.py":
            continue
        imports = _module_imports(path)
        for mod in imports:
            assert "agent.tool_executor" not in mod, (
                f"{path.name} imports {mod}"
            )


def test_executive_does_not_import_turn_finalizer():
    for path in EXEC_DIR.glob("*.py"):
        if path.name == "__init__.py":
            continue
        imports = _module_imports(path)
        for mod in imports:
            assert "agent.turn_finalizer" not in mod, (
                f"{path.name} imports {mod}"
            )


def test_executive_does_not_call_subprocess_via_os_system():
    """No calls to os.system or os.popen in the executive package.

    The check is CODE-ONLY: docstrings and comments that list these
    names as documentation of prohibited APIs are skipped.
    """
    import re
    import ast
    for path in EXEC_DIR.glob("*.py"):
        if path.name == "__init__.py":
            continue
        # Strip docstrings + comments before grepping.
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        ranges = []
        for node in ast.walk(tree):
            if isinstance(
                node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                if (
                    node.body
                    and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)
                ):
                    end = node.body[0].end_lineno or node.body[0].lineno
                    ranges.append((node.body[0].lineno, end))
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        code_lines = [
            line
            for i, line in enumerate(lines, start=1)
            if not line.lstrip().startswith("#")
            and not any(lo <= i <= hi for lo, hi in ranges)
        ]
        code_src = "".join(code_lines)
        assert "os.system" not in code_src, f"{path.name} uses os.system"
        assert "os.popen" not in code_src, f"{path.name} uses os.popen"
        assert "subprocess.run" not in code_src, f"{path.name} uses subprocess.run"
        assert "subprocess.Popen" not in code_src, f"{path.name} uses subprocess.Popen"
        assert "subprocess.call" not in code_src, f"{path.name} uses subprocess.call"


def test_executive_does_not_write_to_messages_or_api_kwargs():
    """No writes to messages / api_messages / api_kwargs / tool result messages."""
    import re
    for path in EXEC_DIR.glob("*.py"):
        if path.name == "__init__.py":
            continue
        src = path.read_text(encoding="utf-8")
        # No mutations of .messages
        assert not re.search(r"\.\s*messages\s*\[", src), f"{path.name} touches .messages[...]"
        assert not re.search(r"\.\s*api_messages\s*\[", src), f"{path.name} touches .api_messages[...]"
        assert not re.search(r"\.\s*api_kwargs\s*\[", src), f"{path.name} touches .api_kwargs[...]"
        assert "tool_result_message" not in src, f"{path.name} touches tool_result_message"
        assert "make_tool_result_message" not in src, f"{path.name} calls make_tool_result_message"

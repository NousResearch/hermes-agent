"""Static guards for AIAgent constructor/runtime invariants.

Regression coverage for:
- TypeError: get_tool_definitions() got an unexpected keyword argument 'save_trajectories'
- AttributeError on runtime attributes when the gateway constructs AIAgent directly
"""

from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_INIT = REPO_ROOT / "agent" / "agent_init.py"
RUN_AGENT = REPO_ROOT / "run_agent.py"


def _parse(path: Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8"))


def _find_function(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name}() not found")


def _agent_attr_assign_line(func: ast.FunctionDef, attr_name: str) -> int | None:
    for node in ast.walk(func):
        targets = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        for target in targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "agent"
                and target.attr == attr_name
            ):
                return getattr(node, "lineno", 0)
    return None


def _get_tool_definitions_line(func: ast.FunctionDef) -> int:
    for node in ast.walk(func):
        if not isinstance(node, ast.Call):
            continue
        func_node = node.func
        if isinstance(func_node, ast.Attribute) and func_node.attr == "get_tool_definitions":
            return node.lineno
        if isinstance(func_node, ast.Name) and func_node.id == "get_tool_definitions":
            return node.lineno
    raise AssertionError("get_tool_definitions() call not found in init_agent()")


def _assert_attr_before_get_tool_definitions(attr_name: str) -> None:
    init_agent = _find_function(_parse(AGENT_INIT), "init_agent")
    assign_line = _agent_attr_assign_line(init_agent, attr_name)
    get_tool_line = _get_tool_definitions_line(init_agent)
    assert assign_line is not None, f"agent.{attr_name} is never assigned in init_agent()"
    assert assign_line < get_tool_line, (
        f"agent.{attr_name} assigned at line {assign_line}, but "
        f"get_tool_definitions() is called at line {get_tool_line}"
    )


def _assert_attr_in_init_agent(attr_name: str) -> None:
    init_agent = _find_function(_parse(AGENT_INIT), "init_agent")
    assign_line = _agent_attr_assign_line(init_agent, attr_name)
    assert assign_line is not None, f"agent.{attr_name} is never assigned in init_agent()"


def _attr_used_by_aiagent_outside_init(attr_name: str) -> bool:
    tree = _parse(RUN_AGENT)
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != "AIAgent":
            continue
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                continue
            for child in ast.walk(item):
                if (
                    isinstance(child, ast.Attribute)
                    and isinstance(child.value, ast.Name)
                    and child.value.id == "self"
                    and child.attr == attr_name
                    and isinstance(child.ctx, ast.Load)
                ):
                    return True
    return False


def test_get_tool_definitions_call_uses_only_supported_kwargs():
    init_agent = _find_function(_parse(AGENT_INIT), "init_agent")
    for node in ast.walk(init_agent):
        if not isinstance(node, ast.Call):
            continue
        func_node = node.func
        is_get_tool_definitions = (
            isinstance(func_node, ast.Attribute)
            and func_node.attr == "get_tool_definitions"
        ) or (
            isinstance(func_node, ast.Name)
            and func_node.id == "get_tool_definitions"
        )
        if not is_get_tool_definitions:
            continue
        kw_names = {kw.arg for kw in node.keywords}
        assert "save_trajectories" not in kw_names
        assert kw_names <= {"enabled_toolsets", "disabled_toolsets", "quiet_mode"}


def test_cleanup_session_invariants_seeded_before_tool_loading():
    for attr in (
        "session_id",
        "_session_db",
        "_session_db_created",
        "_parent_session_id",
        "_compression_warning",
    ):
        _assert_attr_before_get_tool_definitions(attr)


def test_gateway_runtime_attrs_are_initialized_by_constructor_authority():
    for attr in (
        "_tool_use_enforcement",
        "_checkpoint_mgr",
        "_session_init_model_config",
        "_last_flushed_db_idx",
        "_primary_runtime",
        "_subdirectory_hints",
    ):
        _assert_attr_in_init_agent(attr)


def test_conditionally_used_runtime_attrs_are_initialized_by_constructor_authority():
    for attr in ("_session_init_model_config", "_last_flushed_db_idx"):
        if _attr_used_by_aiagent_outside_init(attr):
            _assert_attr_in_init_agent(attr)

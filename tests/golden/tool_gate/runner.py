from __future__ import annotations

import builtins
from types import SimpleNamespace

from scripts.refactor_equiv.runner import capture, load_cases


class _InlineToolGateAdapter:
    def __init__(self):
        import agent.tool_executor as tool_executor
        import tools.delegate_tool as delegate_tool

        self._tool_executor = tool_executor
        self._delegate_tool = delegate_tool

    def tool_search_scoped_names(self, agent):
        return self._tool_executor._tool_search_scoped_names(agent)

    def tool_scope_block_message(self, tool_name):
        return (
            f"'{tool_name}' is not available in this session. "
            "Use tool_search to find tools you can call."
        )

    def tool_scope_block_result(self, tool_name):
        import json

        return json.dumps({"error": self.tool_scope_block_message(tool_name)}, ensure_ascii=False)

    def resolve_tool_search_unwrap(self, agent, function_name, function_args):
        from tools import tool_search as _ts

        out_name = function_name
        out_args = function_args
        block_message = None
        block_result = None
        try:
            if function_name == _ts.TOOL_CALL_NAME:
                underlying, underlying_args, err = _ts.resolve_underlying_call(function_args)
                if not err and underlying:
                    if underlying in self.tool_search_scoped_names(agent):
                        out_name = underlying
                        out_args = underlying_args
                    else:
                        block_message = self.tool_scope_block_message(underlying)
                        block_result = self.tool_scope_block_result(underlying)
        except Exception:
            pass
        return out_name, out_args, block_message, block_result

    def pre_tool_block_from_builtin_gate(self, agent, function_name, tool_scope_block):
        from agent.budget_grace_gate import grace_block_message, is_readonly_grace_tool

        if getattr(agent, "_in_budget_grace", False) and not is_readonly_grace_tool(function_name):
            return {
                "message": grace_block_message(function_name),
                "error_type": "budget_grace_block",
            }
        if tool_scope_block is not None:
            return {
                "message": tool_scope_block,
                "error_type": "tool_scope_block",
            }
        return None

    def strip_blocked_delegate_toolsets(self, toolsets):
        return self._delegate_tool._strip_blocked_tools(toolsets)


def _impl():
    try:
        from agent.fork_ext import tool_gate
        return tool_gate
    except ImportError:
        return _InlineToolGateAdapter()


def _agent(payload: dict) -> SimpleNamespace:
    data = dict(payload or {})
    scope_names = data.pop("scope_names", None)
    agent = SimpleNamespace(**data)
    if scope_names is not None:
        import model_tools  # noqa: F401 - initialize registry generation before cache key capture
        from tools.registry import registry

        enabled = getattr(agent, "enabled_toolsets", None)
        disabled = getattr(agent, "disabled_toolsets", None)
        cache_key = (
            getattr(registry, "_generation", 0),
            frozenset(enabled) if enabled is not None else None,
            frozenset(disabled) if disabled is not None else None,
        )
        agent._tool_search_scope_cache = (cache_key, frozenset(scope_names))
    return agent


def _ensure_deferrable_test_tool() -> None:
    from tools.registry import registry

    if registry.get_entry("golden_lookup") is not None:
        return
    registry.register(
        name="golden_lookup",
        toolset="golden_plugin",
        schema={
            "type": "function",
            "function": {
                "name": "golden_lookup",
                "description": "Golden harness deferrable lookup tool.",
                "parameters": {
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                },
            },
        },
        handler=lambda args, **kwargs: args,
    )


def run_case(case: dict):
    _ensure_deferrable_test_tool()
    mod = _impl()
    kind = case["kind"]
    if kind == "unwrap":
        out_name, out_args, block_message, block_result = mod.resolve_tool_search_unwrap(
            _agent(case.get("agent") or {}),
            case["function_name"],
            case.get("function_args") or {},
        )
        return {
            "return": {
                "function_name": out_name,
                "function_args": out_args,
                "scope_block_message": block_message,
                "scope_block_result": block_result,
            },
            "messages": [],
            "db": [],
        }
    if kind == "scope":
        return {
            "return": sorted(mod.tool_search_scoped_names(_agent(case.get("agent") or {}))),
            "messages": [],
            "db": [],
        }
    if kind == "scope_message":
        return {
            "return": mod.tool_scope_block_message(case["tool_name"]),
            "messages": [],
            "db": [],
        }
    if kind == "scope_result":
        return {
            "return": mod.tool_scope_block_result(case["tool_name"]),
            "messages": [],
            "db": [],
        }
    if kind == "pre_block":
        return {
            "return": mod.pre_tool_block_from_builtin_gate(
                _agent(case.get("agent") or {}),
                case["function_name"],
                case.get("tool_scope_block"),
            ),
            "messages": [],
            "db": [],
        }
    if kind == "strip_toolsets":
        return {
            "return": mod.strip_blocked_delegate_toolsets(case["toolsets"]),
            "messages": [],
            "db": [],
        }
    raise AssertionError(f"unknown case kind {kind!r}")


def exercise_coverage():
    mod = _impl()
    capture(load_cases("tests/golden/tool_gate/corpus.json"), run_case)

    original_import = builtins.__import__

    def fail_model_tools(name, *args, **kwargs):
        if name == "model_tools":
            raise RuntimeError("forced import failure")
        return original_import(name, *args, **kwargs)

    builtins.__import__ = fail_model_tools
    try:
        mod.tool_search_scoped_names(SimpleNamespace())
    finally:
        builtins.__import__ = original_import

    import model_tools

    original_get_tool_definitions = model_tools.get_tool_definitions
    model_tools.get_tool_definitions = lambda **kwargs: (_ for _ in ()).throw(RuntimeError("forced defs failure"))
    try:
        mod.tool_search_scoped_names(SimpleNamespace())
    finally:
        model_tools.get_tool_definitions = original_get_tool_definitions

    class NoCacheAgent:
        enabled_toolsets = None
        disabled_toolsets = None

        def __setattr__(self, name, value):
            if name == "_tool_search_scope_cache":
                raise RuntimeError("forced cache failure")
            return super().__setattr__(name, value)

    mod.tool_search_scoped_names(NoCacheAgent())

    from tools import tool_search

    original_resolve = tool_search.resolve_underlying_call
    tool_search.resolve_underlying_call = lambda args: (_ for _ in ()).throw(RuntimeError("forced unwrap failure"))
    try:
        mod.resolve_tool_search_unwrap(SimpleNamespace(), tool_search.TOOL_CALL_NAME, {"name": "x"})
    finally:
        tool_search.resolve_underlying_call = original_resolve

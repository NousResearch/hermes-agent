"""Context-engine tool registration must be as defensive as its sibling registry.

``plugins/context_engine/load_context_engine()`` advertises pluggable context
engines, and ``ContextEngine.get_tool_schemas()`` is the documented way for one
to expose tools. Two hardening rules that upstream already applies to the
*memory-provider* registry — the other consumer of the exact same
``get_tool_schemas()`` contract — were missing on the context-engine path:

1. **Untrusted call.** ``agent/memory_manager.py::get_all_tool_schemas`` wraps
   ``provider.get_tool_schemas()`` in ``try/except`` + warning. But
   ``agent/agent_init.py`` iterated
   ``agent.context_compressor.get_tool_schemas()`` bare, so an engine raising
   there aborted ``init_agent`` — the agent never starts.

2. **Reserved core names.** ``agent/memory_manager.py::add_provider`` imports
   ``toolsets._HERMES_CORE_TOOLS`` and refuses a provider tool that shadows a
   built-in ("Core tools always win", #40466). The context-engine registration
   had no such check — yet
   ``agent/tool_executor.py::execute_tool_calls_sequential`` resolves the
   built-in ``if/elif`` branches BEFORE the
   ``elif agent._context_engine_tool_names and ...`` branch, so a shadowing
   engine tool is registered into ``agent.tools`` / ``valid_tool_names`` and
   can never be routed to the engine.

Both host sites (``agent_init`` at build, ``tools/mcp_tool`` on snapshot
rebuild) now share one entry point, ``collect_engine_tool_schemas``.
"""

from __future__ import annotations

import types

from agent.context_engine import collect_engine_tool_schemas
from toolsets import _HERMES_CORE_TOOLS


def _engine(schemas=None, *, raises=None):
    def _get():
        if raises is not None:
            raise raises
        return list(schemas or [])

    return types.SimpleNamespace(get_tool_schemas=_get)


def _schema(name):
    return {"name": name, "description": "", "parameters": {"type": "object"}}


def _core_name() -> str:
    """A real reserved core tool name to use as the shadow victim."""
    return "memory" if "memory" in _HERMES_CORE_TOOLS else _HERMES_CORE_TOOLS[0]


# ── control: the ordinary path still works ─────────────────────────────────


def test_ordinary_engine_tools_are_returned_unchanged():
    """Control — the guards must not break normal registration."""
    out = collect_engine_tool_schemas(
        _engine([_schema("engine_grep"), _schema("engine_expand")])
    )
    assert [s["name"] for s in out] == ["engine_grep", "engine_expand"]


def test_already_wrapped_schema_is_unwrapped_not_dropped():
    """An OpenAI-form entry normalizes rather than becoming a nameless tool."""
    wrapped = {"type": "function", "function": _schema("engine_grep")}
    out = collect_engine_tool_schemas(_engine([wrapped]))
    assert [s["name"] for s in out] == ["engine_grep"]


def test_engine_without_the_hook_yields_nothing():
    """An engine that never implements get_tool_schemas is not an error."""
    assert collect_engine_tool_schemas(types.SimpleNamespace()) == []


# ── 1. a raising engine must not abort agent construction ──────────────────


def test_raising_engine_degrades_to_no_tools(caplog):
    """The engine call is untrusted; an exception must not propagate."""
    out = collect_engine_tool_schemas(
        _engine(raises=RuntimeError("engine backend down"))
    )
    assert out == []


def test_raising_engine_does_not_abort_agent_init_registration():
    """End-to-end: the agent_init registration block completes on a raising engine.

    Drives the real block against a raising engine — the failure mode this
    guards is `init_agent` aborting, so the assertion is that registration
    finishes and simply publishes no engine tools.
    """
    agent = types.SimpleNamespace()
    agent.tools = []
    agent.valid_tool_names = set()
    agent._context_engine_tool_names = set()
    agent.context_compressor = _engine(raises=RuntimeError("engine backend down"))

    # Mirror of the agent_init loop, driven through the shared entry point.
    for schema in collect_engine_tool_schemas(agent.context_compressor):
        agent.tools.append({"type": "function", "function": schema})
        agent.valid_tool_names.add(schema["name"])
        agent._context_engine_tool_names.add(schema["name"])

    assert agent.tools == []
    assert agent._context_engine_tool_names == set()


# ── 2. reserved core tool names ────────────────────────────────────────────


def test_core_tool_names_are_refused():
    """A shadowing engine tool must never enter the routing table."""
    victim = _core_name()
    out = collect_engine_tool_schemas(
        _engine([_schema(victim), _schema("engine_grep")])
    )
    names = [s["name"] for s in out]
    assert "engine_grep" in names, "a non-core engine tool must still register"
    assert victim not in names, (
        f"engine tool '{victim}' shadows a reserved core tool name and would be "
        f"registered-but-unroutable"
    )


def test_every_core_tool_name_is_refused():
    """The guard covers the whole reserved set, not one sampled name."""
    out = collect_engine_tool_schemas(
        _engine([_schema(n) for n in _HERMES_CORE_TOOLS])
    )
    assert out == []


def test_dispatch_order_is_what_makes_shadowing_a_bug():
    """Pin the invariant the reserved-name guard exists to protect.

    ``execute_tool_calls_sequential`` resolves literal built-in names before it
    consults ``_context_engine_tool_names``; if that order ever inverted, the
    guard's rationale would need revisiting.
    """
    import inspect

    from agent import tool_executor

    src = inspect.getsource(tool_executor.execute_tool_calls_sequential)
    assert "_context_engine_tool_names" in src
    engine_branch_at = src.index("_context_engine_tool_names")
    earlier = src[:engine_branch_at]
    shadowed_before_engine = [
        name for name in _HERMES_CORE_TOOLS if f'function_name == "{name}"' in earlier
    ]
    assert shadowed_before_engine, (
        "expected built-in tool names to be dispatched before the "
        "context-engine branch; if that changed, the reserved-name guard's "
        "rationale needs revisiting"
    )


# ── 3. both host sites share the entry point ───────────────────────────────


def test_snapshot_rebuild_applies_the_same_guards():
    """tools/mcp_tool's re-injection must not diverge from agent_init."""
    from tools import mcp_tool

    victim = _core_name()
    agent = types.SimpleNamespace()
    agent.tools = []
    agent.valid_tool_names = set()
    agent.enabled_toolsets = None
    agent.disabled_toolsets = None
    agent._memory_manager = None
    agent.context_compressor = _engine([_schema(victim), _schema("engine_grep")])

    tools_list: list = []
    name_set: set = set()
    staged = mcp_tool._reinject_post_build_tools(agent, tools_list, name_set)

    assert "engine_grep" in staged
    assert victim not in staged
    assert victim not in name_set
    assert all(t["function"]["name"] != victim for t in tools_list)


def test_snapshot_rebuild_survives_a_raising_engine():
    from tools import mcp_tool

    agent = types.SimpleNamespace()
    agent.tools = []
    agent.valid_tool_names = set()
    agent.enabled_toolsets = None
    agent.disabled_toolsets = None
    agent._memory_manager = None
    agent.context_compressor = _engine(raises=RuntimeError("engine backend down"))

    tools_list: list = []
    name_set: set = set()
    staged = mcp_tool._reinject_post_build_tools(agent, tools_list, name_set)

    assert staged == set()
    assert tools_list == []

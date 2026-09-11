"""Behavioral coverage for the ``skip_tool_search_assembly`` flag.

The flag exists so a caller that already knows its exact toolset can ask
``get_tool_definitions`` for the *pre-assembly* (eager) catalog — the real
tool schemas — instead of the three-tool ``tool_search`` /
``tool_describe`` / ``tool_call`` bridge that progressive disclosure would
otherwise collapse the deferrable tools behind.

These tests exercise the flag's *behavior*, never the shape of the source:

* ``get_tool_definitions(skip_tool_search_assembly=True)`` returns the eager
  catalog even when assembly would otherwise activate; ``False`` lets the
  bridge collapse happen. Same registered catalog, only the flag differs.
* The public constructor (``AIAgent`` → ``init_agent`` →
  ``get_tool_definitions``) actually forwards the caller's value to the
  receiving end, and defaults it to ``False`` so existing callers keep the
  bridge behavior.

The assembly *semantics* (which tools defer, core tools never defer, listing
tiers) live in ``tests/tools/test_tool_search.py``; here we only prove the
flag flips eager↔bridge and reaches the assembler intact.
"""
import json

import pytest

from tools.tool_search import BRIDGE_TOOL_NAMES


def _td(name, properties=None):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"desc for {name}",
            "parameters": {"type": "object", "properties": properties or {}},
        },
    }


def _register_mcp_tool(name, toolset="mcp-skiptest"):
    """Register a real, deferrable MCP-prefixed tool in the live registry.

    Deferrable classification keys off the ``mcp-`` toolset prefix, so these
    tools are exactly the ones progressive disclosure would collapse behind
    the bridge — the surface the flag governs.
    """
    from tools.registry import registry

    def _handler(args, task_id=None, **kw):
        return json.dumps({"ok": True, "tool": name})

    registry.register(
        name=name,
        toolset=toolset,
        schema=_td(name, {"repo": {"type": "string"}}),
        handler=_handler,
        override=True,
    )


@pytest.fixture
def registered_catalog():
    """Three deferrable MCP tools scoped to a private test toolset."""
    names = ["mcp_skip_alpha", "mcp_skip_beta", "mcp_skip_gamma"]
    for n in names:
        _register_mcp_tool(n)
    return "mcp-skiptest", names


def test_skip_true_returns_eager_catalog(registered_catalog):
    """skip_tool_search_assembly=True exposes the real tool schemas.

    The pre-assembly catalog must contain every registered tool and none of
    the bridge tools — the caller gets to call the tools directly.
    """
    import model_tools

    toolset, names = registered_catalog
    defs = model_tools.get_tool_definitions(
        enabled_toolsets=[toolset],
        quiet_mode=True,
        skip_tool_search_assembly=True,
    )
    got = {(t.get("function") or {}).get("name") for t in defs}

    for n in names:
        assert n in got, f"eager catalog dropped real tool {n!r}"
    assert not (BRIDGE_TOOL_NAMES & got), (
        "skip_tool_search_assembly=True must NOT collapse tools behind the "
        f"tool_search bridge, but saw {BRIDGE_TOOL_NAMES & got}"
    )


def test_skip_false_collapses_deferrable_behind_bridge(
    registered_catalog, monkeypatch
):
    """skip_tool_search_assembly=False lets progressive disclosure activate.

    With the tool-search gate open, the same registered tools are replaced
    by the three bridge tools. This is the observable behavior the flag
    turns off — asserted against the identical catalog so the flag is the
    only variable.
    """
    import model_tools
    from tools import tool_search

    # Isolate the flag's effect from the token-threshold heuristic (which has
    # its own coverage): force the gate open and the config on, then let the
    # real assembler run.
    monkeypatch.setattr(tool_search, "should_activate", lambda *a, **k: True)
    monkeypatch.setattr(
        tool_search,
        "load_config",
        lambda: tool_search.ToolSearchConfig.from_raw({"enabled": "on"}),
    )

    toolset, names = registered_catalog
    defs = model_tools.get_tool_definitions(
        enabled_toolsets=[toolset],
        quiet_mode=True,
        skip_tool_search_assembly=False,
    )
    got = {(t.get("function") or {}).get("name") for t in defs}

    assert BRIDGE_TOOL_NAMES <= got, (
        "skip_tool_search_assembly=False with the gate open must expose the "
        f"tool_search bridge, saw only {got}"
    )
    for n in names:
        assert n not in got, (
            f"deferrable tool {n!r} must be collapsed behind the bridge when "
            "the flag is False"
        )


class _FakeOpenAI:
    def __init__(self, **kw):
        self.api_key = kw.get("api_key", "test")
        self.base_url = kw.get("base_url", "http://test")

    def close(self):
        pass


def _build_agent(monkeypatch, captured, **agent_kwargs):
    """Construct a real AIAgent with a spy standing in for the receiving end.

    ``agent/agent_init.py`` calls ``run_agent.get_tool_definitions(...)``, so
    patching that name captures whatever the constructor chain actually
    forwarded — a dropped or coerced value at any hop shows up here.
    """
    def _spy(**kw):
        captured.update(kw)
        return []

    monkeypatch.setattr("run_agent.get_tool_definitions", _spy)
    monkeypatch.setattr("run_agent.check_toolset_requirements", lambda: {})
    monkeypatch.setattr("run_agent.OpenAI", _FakeOpenAI)

    from run_agent import AIAgent

    return AIAgent(
        api_key="test-key",
        base_url="http://test",
        provider="openrouter",
        api_mode="chat_completions",
        max_iterations=1,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        **agent_kwargs,
    )


def test_constructor_forwards_flag_to_get_tool_definitions(monkeypatch, tmp_path):
    """AIAgent → init_agent → get_tool_definitions carries the caller's True."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hm"))
    captured = {}
    _build_agent(monkeypatch, captured, skip_tool_search_assembly=True)
    assert captured.get("skip_tool_search_assembly") is True, (
        "constructor did not forward skip_tool_search_assembly=True to "
        f"get_tool_definitions (saw {captured.get('skip_tool_search_assembly')!r})"
    )


def test_constructor_defaults_flag_to_false(monkeypatch, tmp_path):
    """Callers that don't ask for eager tools keep the bridge behavior."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hm"))
    captured = {}
    _build_agent(monkeypatch, captured)  # flag not passed
    assert captured.get("skip_tool_search_assembly") is False, (
        "default must be False so existing callers keep progressive "
        f"disclosure (saw {captured.get('skip_tool_search_assembly')!r})"
    )

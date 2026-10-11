"""Regression tests for #130153: an unknown tool name that this session can reach
through the tool_search bridge gets bridge-shaped recovery copy instead of the generic
"does not exist" error, which sent models hunting for a tool they had already found.

The deferred hint must be session-scoped (a restricted session is not pointed at a
name its catalog cannot reach), computed lazily (the catalog rebuild is too costly for
every validated batch), and fail-open (any error keeps the legacy generic error).
"""

from __future__ import annotations

import types

from agent.conversation_loop import _invalid_tool_name_error_content, _session_deferred_tool_names
from agent.turn_tool_validation import validate_tool_calls


# --- _invalid_tool_name_error_content ---------------------------------------------------------


def test_generic_error_for_truly_unknown_name():
    content = _invalid_tool_name_error_content("nope_tool", {"tool_call", "tool_search"})
    assert content == "Tool 'nope_tool' does not exist. Available tools: tool_call, tool_search"


def test_deferred_name_gets_bridge_invocation_copy():
    content = _invalid_tool_name_error_content(
        "manage_catalog", {"tool_call", "tool_search"}, frozenset({"manage_catalog"}))
    assert "does not exist" not in content
    assert "deferred" in content
    assert "cannot be called by its bare name" in content
    assert "tool_describe" in content
    assert 'tool_call {"calls": [{"name": "manage_catalog", "arguments": {...}}]}' in content


def test_deferred_name_outside_session_scope_stays_generic():
    # Scope matters: a name another session defers must not get bridge copy here.
    content = _invalid_tool_name_error_content(
        "manage_catalog", {"tool_call"}, frozenset({"some_other_tool"}))
    assert content.startswith("Tool 'manage_catalog' does not exist.")


def test_blank_name_keeps_anti_priming_copy():
    content = _invalid_tool_name_error_content("  ", {"tool_call"}, frozenset({"tool_call"}))
    assert "tool name was empty" in content


# --- _session_deferred_tool_names -------------------------------------------------------------


class _Agent:
    def __init__(self, valid_tool_names):
        self.valid_tool_names = set(valid_tool_names)
        self.enabled_toolsets = None
        self.disabled_toolsets = None


def _bridge_agent() -> _Agent:
    return _Agent({"tool_search", "tool_describe", "tool_call", "terminal_exec"})


def _patch_defer_config(monkeypatch, defer_tools):
    from tools import tool_search as ts
    monkeypatch.setattr(ts, "load_config_readonly",
                        lambda: ts.ToolSearchConfig.from_raw({"defer": sorted(defer_tools)}))


def test_bridge_inactive_returns_empty_without_catalog_rebuild(monkeypatch):
    import model_tools

    def _boom(*_a, **_k):
        raise AssertionError("catalog must not be rebuilt when the bridge is inactive")

    monkeypatch.setattr(model_tools, "get_tool_definitions", _boom)
    agent = _Agent({"terminal_exec", "file_write"})
    assert _session_deferred_tool_names(agent) == frozenset()


def test_bridge_active_collects_scoped_deferred_names(monkeypatch):
    import model_tools

    agent = _bridge_agent()
    monkeypatch.setattr(
        model_tools, "get_tool_definitions",
        lambda **_k: [
            {"type": "function", "function": {"name": "manage_catalog"}},
            {"type": "function", "function": {"name": "terminal_exec"}},
        ])
    _patch_defer_config(monkeypatch, {"manage_catalog"})
    # terminal_exec is not in the defer set, so only manage_catalog is deferred.
    assert _session_deferred_tool_names(agent) == frozenset({"manage_catalog"})


def test_result_is_cached_on_the_agent(monkeypatch):
    import model_tools

    agent = _bridge_agent()
    calls = {"n": 0}

    def _defs(**_k):
        calls["n"] += 1
        return [{"type": "function", "function": {"name": "manage_catalog"}}]

    monkeypatch.setattr(model_tools, "get_tool_definitions", _defs)
    _patch_defer_config(monkeypatch, {"manage_catalog"})
    assert _session_deferred_tool_names(agent) == frozenset({"manage_catalog"})
    assert _session_deferred_tool_names(agent) == frozenset({"manage_catalog"})
    assert calls["n"] == 1


def test_catalog_failure_fails_open(monkeypatch):
    import model_tools

    def _raise(**_k):
        raise RuntimeError("registry unavailable")

    monkeypatch.setattr(model_tools, "get_tool_definitions", _raise)
    agent = _bridge_agent()
    assert _session_deferred_tool_names(agent) == frozenset()


def test_fail_open_empty_result_is_not_cached(monkeypatch):
    # A transient catalog failure must not pin the legacy generic error for the rest of
    # the session: the next call recomputes instead of serving the cached empty set.
    import model_tools

    agent = _bridge_agent()
    defs = [{"type": "function", "function": {"name": "manage_catalog"}}]

    def _flaky(**_k):
        if _flaky.failing:
            raise RuntimeError("registry unavailable")
        return defs

    _flaky.failing = True
    monkeypatch.setattr(model_tools, "get_tool_definitions", _flaky)
    _patch_defer_config(monkeypatch, {"manage_catalog"})
    assert _session_deferred_tool_names(agent) == frozenset()
    _flaky.failing = False
    assert _session_deferred_tool_names(agent) == frozenset({"manage_catalog"})


def test_bridge_inactive_result_is_not_cached(monkeypatch):
    # The bridge may activate between turns (scope widened by bot-chat refresh), so an
    # inactive empty set must not be served once "tool_call" becomes valid.
    import model_tools

    agent = _Agent({"terminal_exec", "file_write"})
    assert _session_deferred_tool_names(agent) == frozenset()
    agent.valid_tool_names = {"tool_search", "tool_describe", "tool_call", "terminal_exec"}
    monkeypatch.setattr(
        model_tools, "get_tool_definitions",
        lambda **_k: [{"type": "function", "function": {"name": "manage_catalog"}}])
    _patch_defer_config(monkeypatch, {"manage_catalog"})
    assert _session_deferred_tool_names(agent) == frozenset({"manage_catalog"})


def test_cache_invalidated_on_registry_mutation(monkeypatch):
    # Mid-session catalog rewrites (MCP connect/reload, tools enable) bump the registry
    # generation; the memo must follow or the session keeps serving pre-rewrite names —
    # the #130153 symptom, one generation later.
    import model_tools
    from tools.registry import registry as _registry

    agent = _bridge_agent()
    defs = {"v1": [{"type": "function", "function": {"name": "manage_catalog"}}],
            "v2": [{"type": "function", "function": {"name": "newly_registered_tool"}}]}
    state = {"version": "v1"}

    def _defs(**_k):
        return defs[state["version"]]

    monkeypatch.setattr(model_tools, "get_tool_definitions", _defs)
    _patch_defer_config(monkeypatch, {"manage_catalog", "newly_registered_tool"})
    assert _session_deferred_tool_names(agent) == frozenset({"manage_catalog"})
    state["version"] = "v2"
    monkeypatch.setattr(_registry, "_generation", getattr(_registry, "_generation", 0) + 1)
    assert _session_deferred_tool_names(agent) == frozenset({"newly_registered_tool"})


# --- validate_tool_calls end-to-end -----------------------------------------------------------


class _ValidationAgent(_Agent):
    """Covers only the surface ``validate_tool_calls`` touches; the deferred-name cache is
    pre-seeded in keyed form (matching the live registry, so it counts as a hit) so the
    test never rebuilds the real catalog."""

    def __init__(self, valid_tool_names, deferred_names):
        super().__init__(valid_tool_names)
        from tools.registry import registry as _registry
        self._bridge_deferred_names = (
            (_registry.current_scope_key(), getattr(_registry, "_generation", 0), None, None),
            frozenset(deferred_names),
        )
        self._invalid_tool_retries = 0
        self._invalid_json_retries = 0
        self.printed = []

    def _uniquify_tool_call_ids(self, tool_calls):
        return tool_calls

    def _repair_tool_call(self, _name):
        return None

    def _vprint(self, msg, **_k):
        self.printed.append(msg)

    def _buffer_vprint(self, msg, **_k):
        self.printed.append(msg)

    def _flush_status_buffer(self):
        pass

    def _build_assistant_message(self, _assistant_message, _finish_reason):
        return {"role": "assistant", "content": None, "tool_calls": []}


def _tool_call(name, call_id="call_1"):
    return types.SimpleNamespace(
        function=types.SimpleNamespace(name=name, arguments="{}"), id=call_id)


def test_deferred_hint_reaches_the_tool_error_result():
    agent = _ValidationAgent(
        {"tool_search", "tool_describe", "tool_call", "terminal_exec"},
        deferred_names={"manage_catalog"},
    )
    messages: list = []
    assistant = types.SimpleNamespace(tool_calls=[_tool_call("manage_catalog")])
    verdict = validate_tool_calls(
        agent, assistant, "tool_calls", messages=messages, conversation_history=[],
        api_call_count=1, effective_task_id=None,
    )
    assert verdict.action == "continue"
    assert agent._invalid_tool_retries == 1  # a deferred bare name is still an invalid call
    error_results = [m for m in messages if m.get("role") == "tool"]
    assert len(error_results) == 1
    content = error_results[0]["content"]
    assert "deferred" in content
    assert 'tool_call {"calls": [{"name": "manage_catalog"' in content
    assert "does not exist" not in content


def test_truly_unknown_name_keeps_generic_error():
    agent = _ValidationAgent(
        {"tool_search", "tool_describe", "tool_call", "terminal_exec"},
        deferred_names={"manage_catalog"},
    )
    messages: list = []
    assistant = types.SimpleNamespace(tool_calls=[_tool_call("nope_tool")])
    verdict = validate_tool_calls(
        agent, assistant, "tool_calls", messages=messages, conversation_history=[],
        api_call_count=1, effective_task_id=None,
    )
    assert verdict.action == "continue"
    error_results = [m for m in messages if m.get("role") == "tool"]
    assert len(error_results) == 1
    assert error_results[0]["content"].startswith("Tool 'nope_tool' does not exist.")

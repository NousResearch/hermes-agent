"""Unknown-name errors for tools reachable only through the Tool Search bridge.

Cases adapted from #130174 (liuhao1024): session scoping, blank-name anti-priming,
fail-open on catalog errors, no catalog rebuild while the bridge is inactive, and
strike counting unchanged for a deferred bare name.
"""

from __future__ import annotations

import types

import pytest

from agent import tool_discovery
from agent.conversation_loop import _invalid_tool_name_error_content
from agent.tool_discovery import invalid_tool_name_error_content
from agent.turn_tool_validation import validate_tool_calls

BRIDGE = {"tool_search", "tool_describe", "tool_call"}


class _Agent:
    def __init__(self, valid_tool_names):
        self.valid_tool_names = set(valid_tool_names)
        self.enabled_toolsets = None
        self.disabled_toolsets = None


@pytest.fixture
def deferred(monkeypatch):
    def _set(names):
        monkeypatch.setattr(tool_discovery, "session_deferred_tools", lambda _agent: frozenset(names))
    return _set


def test_unknown_name_keeps_main_text(deferred):
    deferred({"manage_catalog"})
    agent = _Agent(BRIDGE | {"terminal"})
    assert invalid_tool_name_error_content(agent, "nope_tool") == _invalid_tool_name_error_content(
        "nope_tool", agent.valid_tool_names)


def test_deferred_name_gets_bridge_recipe(deferred):
    deferred({"manage_catalog"})
    content = invalid_tool_name_error_content(_Agent(BRIDGE), "manage_catalog")
    assert "does not exist" not in content
    assert 'tool_describe(names=["manage_catalog"])' in content
    assert 'tool_call(calls=[{"name": "manage_catalog"' in content
    assert "did not execute" in content


def test_name_deferred_outside_this_session_stays_generic(deferred):
    deferred({"some_other_tool"})
    content = invalid_tool_name_error_content(_Agent(BRIDGE), "manage_catalog")
    assert content.startswith("Tool 'manage_catalog' does not exist.")


@pytest.mark.parametrize("name", ["", "  "])
def test_blank_name_keeps_anti_priming_text(deferred, name):
    deferred({name})
    agent = _Agent(BRIDGE)
    assert invalid_tool_name_error_content(agent, name) == _invalid_tool_name_error_content(
        name, agent.valid_tool_names)
    assert "tool name was empty" in invalid_tool_name_error_content(agent, name)


def test_bridge_inactive_never_rebuilds_the_catalog(monkeypatch):
    import model_tools

    def _boom(*_a, **_k):
        raise AssertionError("catalog must not be rebuilt when the bridge is inactive")

    monkeypatch.setattr(model_tools, "get_tool_definitions", _boom)
    agent = _Agent({"terminal", "read_file"})
    assert tool_discovery.session_deferred_tools(agent) == frozenset()
    assert invalid_tool_name_error_content(agent, "todo_list").startswith("Tool 'todo_list' does not exist.")


def test_catalog_failure_fails_open_to_main_text(monkeypatch):
    import model_tools

    def _raise(**_k):
        raise RuntimeError("registry unavailable")

    monkeypatch.setattr(model_tools, "get_tool_definitions", _raise)
    agent = _Agent(BRIDGE)
    assert tool_discovery.session_deferred_tools(agent) == frozenset()
    assert invalid_tool_name_error_content(agent, "todo_list") == _invalid_tool_name_error_content(
        "todo_list", agent.valid_tool_names)


class _ValidationAgent(_Agent):
    """Only the surface ``validate_tool_calls`` touches on the all-invalid path."""

    log_prefix = ""

    def __init__(self, valid_tool_names):
        super().__init__(valid_tool_names)
        self._invalid_tool_retries = 0
        self._invalid_json_retries = 0

    def _uniquify_tool_call_ids(self, tool_calls, taken=None):
        return tool_calls

    def _repair_tool_call(self, _name):
        return None

    def _vprint(self, *_a, **_k):
        pass

    _buffer_vprint = _vprint

    def _flush_status_buffer(self):
        pass

    def _build_assistant_message(self, _assistant_message, _finish_reason):
        return {"role": "assistant", "content": None, "tool_calls": []}


def test_deferred_bare_name_is_still_an_invalid_call(deferred, monkeypatch):
    import hermes_cli.observability.shared_metrics_model as metrics

    monkeypatch.setattr(metrics, "record_tool_call_quality", lambda *_a, **_k: None)
    deferred({"manage_catalog"})
    agent = _ValidationAgent(BRIDGE | {"terminal"})
    messages: list = []
    call = types.SimpleNamespace(id="call_1", function=types.SimpleNamespace(name="manage_catalog", arguments="{}"))
    verdict = validate_tool_calls(
        agent, types.SimpleNamespace(tool_calls=[call]), "tool_calls", messages=messages,
        conversation_history=[], api_call_count=1, effective_task_id=None,
    )
    assert verdict.action == "continue"
    assert agent._invalid_tool_retries == 1  # the call is still rejected and still counts
    (result,) = [m for m in messages if m.get("role") == "tool"]
    assert 'tool_call(calls=[{"name": "manage_catalog"' in result["content"]

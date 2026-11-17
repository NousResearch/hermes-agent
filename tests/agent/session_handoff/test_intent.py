"""The model explicitly chooses handoff; telemetry alone never rotates a session."""
import json
from types import SimpleNamespace

import pytest

from agent.inline_tool_executors import INLINE_TOOL_EXECUTORS
from agent.session_handoff.intent import eligible_intent, inject


def agent():
    return SimpleNamespace(session_id="parent", _fresh_topic_handoff_allowed=True,
                           _fresh_topic_handoff=None, tools=[{"function": {"name": "terminal"}}],
                           valid_tool_names={"terminal"}, context_compressor=SimpleNamespace(
                               last_prompt_tokens=90000, context_length=200000, compression_count=0))


def choose(a, handoff="objective; constraints; decisions; files; checks; next action", call_id="choice"):
    return json.loads(INLINE_TOOL_EXECUTORS["fresh_session"](
        a, {"handoff": handoff}, SimpleNamespace(tool_call_id=call_id)))


def result():
    return {"completed": True, "messages": [{"role": "user", "content": "continue"},
        {"role": "assistant", "tool_calls": [{"id": "choice", "function": {"name": "fresh_session"}}]},
        {"role": "tool", "tool_call_id": "choice", "content": "requested"},
        {"role": "assistant", "content": "Done"}]}


def test_no_choice_and_usage_query_do_not_request_handoff():
    a = agent()
    telemetry = json.loads(INLINE_TOOL_EXECUTORS["fresh_session"](a, {}, SimpleNamespace(tool_call_id="status")))
    assert telemetry["usage"]["last_prompt_tokens"] == 90000
    assert telemetry["usage"]["cached_tokens"] is None
    assert eligible_intent(a, result()) is None


def test_model_choice_preserves_checkpoint_and_is_idempotent():
    a = agent()
    assert choose(a)["status"] == "requested_after_turn"
    assert choose(a, "replacement", "duplicate")["status"] == "already_requested"
    intent = eligible_intent(a, result())
    assert intent["summary"] == "objective; constraints; decisions; files; checks; next action"
    assert intent["session_id"] == "parent"
    json.dumps(intent)  # result receipts must remain wire serializable


@pytest.mark.parametrize("flag", ["failed", "interrupted", "partial", "error"])
def test_unsuccessful_turn_never_settles(flag):
    a = agent()
    choose(a)
    assert eligible_intent(a, {**result(), flag: True}) is None


@pytest.mark.parametrize("change", ["steer", "tool", "rotation", "compression"])
def test_checkpoint_invalidated_by_later_work_or_compression(change):
    a = agent()
    choose(a)
    r = result()
    if change == "steer":
        r["messages"].append({"role": "user", "content": "new task"})
    elif change == "tool":
        r["messages"].append({"role": "assistant", "tool_calls": [{"id": "late", "function": {"name": "terminal"}}]})
    elif change == "rotation":
        a.session_id = "compressed"
    else:
        a.context_compressor.compression_count += 1
    assert eligible_intent(a, r) is None


def test_tool_surface_stable_and_execution_scoped():
    a = agent()
    inject(a)
    tools = json.dumps(a.tools)
    inject(a)
    assert json.dumps(a.tools) == tools
    a._fresh_topic_handoff_allowed = False
    assert "error" in choose(a)


def test_duplicate_intent_keeps_first_checkpoint_and_schema_stable():
    a = agent()
    inject(a)
    initial = json.dumps(a.tools)
    assert choose(a, "first checkpoint")["status"] == "requested_after_turn"
    assert choose(a, "duplicate checkpoint", "duplicate")["status"] == "already_requested"
    inject(a)
    assert json.dumps(a.tools) == initial
    assert eligible_intent(a, result())["summary"] == "first checkpoint"


def test_mixed_parallel_tool_round_cannot_supply_a_trustworthy_checkpoint():
    a = agent()
    choose(a)
    r = result()
    r["messages"][1]["tool_calls"].insert(0, {"id": "concurrent", "function": {"name": "terminal"}})
    assert eligible_intent(a, r) is None


def test_mcp_refresh_preserves_authorized_handoff_and_drops_it_when_revoked(monkeypatch):
    import model_tools
    from tools.mcp_tool_agent import refresh_agent_mcp_tools
    a = agent()
    a.enabled_toolsets = a.disabled_toolsets = None
    a.platform = 'telegram'
    a.session_id = None
    base = {'type': 'function', 'function': {'name': 'terminal', 'parameters': {'type': 'object', 'properties': {}}}}
    mcp = {'type': 'function', 'function': {'name': 'mcp__synthetic__inspect', 'parameters': {'type': 'object', 'properties': {}}}}
    monkeypatch.setattr(model_tools, 'get_tool_definitions', lambda **kwargs: [base, mcp])
    inject(a)
    refresh_agent_mcp_tools(a, preserve_prefix=True)
    snapshot = json.dumps(a.tools)
    assert 'fresh_session' in a.valid_tool_names and 'mcp__synthetic__inspect' in a.valid_tool_names
    refresh_agent_mcp_tools(a, preserve_prefix=True)
    assert json.dumps(a.tools) == snapshot
    a._fresh_topic_handoff_allowed = False
    refresh_agent_mcp_tools(a)
    assert 'fresh_session' not in a.valid_tool_names
    assert all(t['function']['name'] != 'fresh_session' for t in a.tools)


def test_native_codex_does_not_advertise_unavailable_inline_capability():
    a = agent()
    a.api_mode = 'codex_app_server'
    inject(a)
    assert 'fresh_session' not in a.valid_tool_names
    assert 'native Codex is unsupported' in choose(a)['error']
    assert a._fresh_topic_handoff is None

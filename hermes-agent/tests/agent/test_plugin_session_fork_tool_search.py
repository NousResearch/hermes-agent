"""A plugin session fork reaches the plugin's own tool when Tool Search defers it.

With Tool Search on (the default), every plugin tool is deferred: the session's ``tools[]`` holds
the bridges ``tool_search`` / ``tool_describe`` / ``tool_call`` instead of the tool. A fork keeps
those ``tools[]`` byte for byte (cache parity), so a requested tool the session defers is reached
through ``tool_call``: the request is admitted when the name is in the session's deferred catalog,
core appends one line to the prompt naming ``tool_call`` and the tool's parameters, and inside the
fork ``tool_call`` is only a carrier: its target must itself be one of the fork's tools.
"""

from __future__ import annotations

import copy
import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent import background_review, plugin_session_fork, side_question_lease
from agent.side_question_lease import issue_side_question_token
from tests.agent import test_plugin_session_fork as _base
from tests.agent.test_plugin_session_fork import (
    PLUGIN, _call, _ctx, _install_plugin, _job_holding, _load, _reply, _settled, _tool,
)

forks = _base.forks  # the scripted fork builder, shared with the main fork tests

CONNECTOR = "connectors__acme__lookup"
OFFER_PARAMS = {"type": "object", "properties": {"skill": {"type": "string"}}, "required": ["skill"]}

_PLUGIN = '''
import json
import sys

PROBE = sys.modules["_fork_ts_probe"]
PARAMS = {"type": "object", "properties": {"skill": {"type": "string"}}, "required": ["skill"]}


def register(ctx):
    for name, params in (("cw_offer", PARAMS), ("cw_hidden", {"type": "object", "properties": {}})):
        def handle(args, _name=name, turn_origin=None, **kwargs):
            PROBE.seen.append((_name, args, turn_origin))
            return json.dumps({"ok": True})
        ctx.register_tool(name, PROBE.toolset, {"name": name, "description": name, "parameters": params}, handle)
'''

_BYSTANDER = '''
import json
import sys

PROBE = sys.modules["_fork_ts_probe"]


def register(ctx):
    ctx.register_tool("peer_probe", "bystander", {"name": "peer_probe", "description": "peer_probe", "parameters": {
        "type": "object", "properties": {}}}, lambda args, **kw: PROBE.seen.append(("peer_probe", args, None)) or "{}")
'''


@pytest.fixture
def probe(monkeypatch, request):
    """``wisdom`` (granted ``llm.session_fork``) registers ``cw_offer`` and ``cw_hidden``;
    ``bystander`` registers ``peer_probe``. ``probe.seen`` records every handler call, the connector
    included."""
    from tools import connectors

    state = SimpleNamespace(seen=[], toolset=getattr(request, "param", "cw"))
    monkeypatch.setitem(sys.modules, "_fork_ts_probe", state)
    monkeypatch.setattr(connectors, "dispatch_connector_batch",
                        lambda calls, *a, **k: state.seen.append(("connector", calls, None)) or "{}")
    _install_plugin(PLUGIN, declares=True, body=_PLUGIN)
    _install_plugin("bystander", declares=True, body=_BYSTANDER)
    state.manager = _load([PLUGIN, "bystander"], granted=[PLUGIN])
    yield state
    side_question_lease._leases.clear()


def _raw(*names: str) -> list[dict]:
    """Pre-assembly defs: the registered schema when there is one."""
    from tools.registry import registry

    return [{"type": "function", "function": copy.deepcopy(registry.get_schema(n))} if registry.get_schema(n)
            else _tool(n) for n in names]


def _session_defs(names, *, tool_search="auto"):
    """A stand-in for ``model_tools.get_tool_definitions`` over ``names``: raw when asked to skip
    Tool Search, else assembled by the real ``assemble_tool_defs``."""
    from tools.tool_search import ToolSearchConfig, assemble_tool_defs

    def get_tool_definitions(enabled_toolsets=None, disabled_toolsets=None, quiet_mode=False,
                             skip_tool_search_assembly=False):
        raw = _raw(*names)
        if skip_tool_search_assembly:
            return raw
        return assemble_tool_defs(raw, config=ToolSearchConfig.from_raw({"enabled": tool_search})).tool_defs
    return get_tool_definitions


SESSION = ("cw_offer", "peer_probe", "session_search", "terminal")


def _parent(names=SESSION, *, tool_search="auto", tools=None):
    get = _session_defs(names, tool_search=tool_search)
    return _base._agent(tools=get() if tools is None else tools, enabled_toolsets=None, disabled_toolsets=None)


def _names(defs) -> list:
    return [d["function"]["name"] for d in defs]


# ── request time ─────────────────────────────────────────────────────────────────────────

def test_with_tool_search_on_a_deferred_own_tool_is_admitted_and_the_prompt_says_how_to_call_it(forks, probe):
    with patch("model_tools.get_tool_definitions", _session_defs(SESSION)):
        parent = _parent()
        assert "cw_offer" not in _names(parent.tools) and "tool_call" in _names(parent.tools)
        handle = _ctx(probe.manager).fork_session(issue_side_question_token(parent, [], "turn-1"), "Share it?",
                                                 tools=["cw_offer"])
        assert _settled(handle).outcome == "ok"
    prompt = forks.built[0]["fork"].seen["prompt"]
    note = prompt.removeprefix("Share it?")
    assert note != prompt and note.count("\n\n") == 1 and "\n" not in note.strip()
    assert "through tool_call with name=<tool>" in note
    assert f"cw_offer {json.dumps(OFFER_PARAMS, separators=(',', ':'), sort_keys=True)}" in note


def test_with_tool_search_off_the_direct_path_is_unchanged(forks, probe):
    with patch("model_tools.get_tool_definitions", _session_defs(SESSION, tool_search="off")):
        parent = _parent(tool_search="off")
        assert "cw_offer" in _names(parent.tools) and "tool_call" not in _names(parent.tools)
        handle = _ctx(probe.manager).fork_session(issue_side_question_token(parent, [], "turn-1"), "Share it?",
                                                 tools=["cw_offer"])
        assert _settled(handle).outcome == "ok"
    assert forks.built[0]["fork"].seen["prompt"] == "Share it?"


def test_an_own_tool_outside_the_sessions_deferred_catalog_is_refused(forks, probe):
    """``cw_hidden`` is the plugin's own but not in this session's scope: neither advertised nor
    deferred, so the request is refused (token spent, as for any tool the session lacks)."""
    with patch("model_tools.get_tool_definitions", _session_defs(SESSION)):
        parent = _parent()
        token = issue_side_question_token(parent, [], "turn-1")
        assert _ctx(probe.manager).fork_session(token, "q", tools=["cw_offer", "cw_hidden"]) is None
        assert _ctx(probe.manager).fork_session(token, "q", tools=["cw_offer"]) is None
    assert forks.built == []


def test_without_the_bridge_in_the_captured_tools_a_missing_tool_is_refused(forks, probe):
    """The session's catalog classifies ``cw_hidden`` deferrable, but its captured ``tools[]`` carry
    no ``tool_call``: nothing could reach it, so it is not in the session."""
    with patch("model_tools.get_tool_definitions", _session_defs((*SESSION, "cw_hidden"), tool_search="off")):
        parent = _parent(tools=_raw(*SESSION))
        token = issue_side_question_token(parent, [], "turn-1")
        assert _ctx(probe.manager).fork_session(token, "q", tools=["cw_offer", "cw_hidden"]) is None
    assert forks.built == []


def test_another_plugins_deferred_tool_is_still_refused_before_the_lease(forks, probe):
    with patch("model_tools.get_tool_definitions", _session_defs(SESSION)):
        parent = _parent()
        token = issue_side_question_token(parent, [], "turn-1")
        for foreign in ("peer_probe", "tool_call", "tool_describe", "tool_search"):
            assert _ctx(probe.manager).fork_session(token, "q", tools=["cw_offer", foreign]) is None
        assert _settled(_ctx(probe.manager).fork_session(token, "q", tools=["cw_offer"])).outcome == "ok"


@pytest.mark.parametrize("probe", ["skills"], indirect=True)
def test_probe_repro_a_plugin_tool_in_a_core_toolset_is_deferred_and_still_reachable(forks, probe):
    """The live failure, unit-level: the plugin tool sits in the ``skills`` toolset, and the real
    ``get_tool_definitions(["skills"])`` serves the skills tools plus the bridges, never the plugin
    tool. Before the fix the fork was refused ``tools_not_in_session``; now it starts."""
    import model_tools
    from run_agent import AIAgent

    assembled = _names(model_tools.get_tool_definitions(enabled_toolsets=["skills"], quiet_mode=True))
    assert "cw_offer" not in assembled
    assert {"tool_call", "tool_describe", "tool_search"} <= set(assembled)
    assert {n for n in assembled if n.startswith("skill")}  # the core skills tools stay direct
    with patch("model_tools.check_toolset_requirements", return_value={}), patch("agent.process_bootstrap.OpenAI"):
        parent = AIAgent(api_key="k", base_url="https://openrouter.ai/api/v1", model="test/model", quiet_mode=True,
                         skip_context_files=True, skip_memory=True, enabled_toolsets=["skills"])
    parent._cached_system_prompt, parent._current_turn_id = "sys", "turn-1"
    assert _names(parent.tools) == assembled
    handle = _ctx(probe.manager).fork_session(issue_side_question_token(parent, [], "turn-1"), "q",
                                             tools=["cw_offer"])
    assert _settled(handle).outcome == "ok"
    assert "through tool_call" in forks.built[0]["fork"].seen["prompt"]


# ── dispatch: tool_call is only a carrier, sequential and concurrent ────────────────────

def _bridged(call_id: str, name: str, arguments: dict | None = None):
    return SimpleNamespace(id=call_id, type="function", function=SimpleNamespace(
        name="tool_call", arguments=json.dumps({"calls": [{"name": name, "arguments": arguments or {}}]})))


CARRIED = (
    ("own", "cw_offer", {"skill": "x"}),
    ("own_not_requested", "cw_hidden", {}),
    ("another_plugin", "peer_probe", {}),
    ("core_deferred", "session_search", {"query": "x"}),
    ("core", "terminal", {"command": "true"}),
    ("connector", CONNECTOR, {}),
)


def _fork_calls(path):
    return SimpleNamespace(content="", tool_calls=[
        *(_bridged(f"{path}-{label}", name, args) for label, name, args in CARRIED),
        SimpleNamespace(id=f"{path}-describe", type="function",
                        function=SimpleNamespace(name="tool_describe", arguments=json.dumps({"names": ["cw_offer"]}))),
        SimpleNamespace(id=f"{path}-search", type="function",
                        function=SimpleNamespace(name="tool_search", arguments=json.dumps({"queries": ["cw"]})))])


# Connectors are in scope here (``manage_connections``), so a carried connector call would reach one.
DISPATCH_SESSION = (*SESSION, "cw_hidden", "manage_connections")


def _dispatcher():
    """A real agent whose session defers the plugin tools behind Tool Search."""
    from run_agent import AIAgent

    with patch("model_tools.check_toolset_requirements", return_value={}), patch("agent.process_bootstrap.OpenAI"):
        agent = AIAgent(api_key="k", base_url="https://openrouter.ai/api/v1", model="test/model",
                        quiet_mode=True, skip_context_files=True, skip_memory=True)
    agent.client = MagicMock()
    agent._review_input_token_budget = 1000
    return agent


@pytest.mark.parametrize("path", ["sequential", "concurrent"])
def test_inside_the_fork_tool_call_reaches_only_the_requested_own_tool(forks, probe, monkeypatch, path):
    """A prebuilt job for ``cw_offer`` on a fork that defers it. ``tool_call`` runs ``cw_offer``
    once (with the fork's origin); ``tool_call`` to the plugin's unrequested tool, another plugin's
    tool, a deferred core tool, an undeferred core tool or a connector is denied before any handler,
    and so are ``tool_describe`` and ``tool_search``. The fork's ``tools[]`` are the captured ones."""
    monkeypatch.setattr(background_review, "release_cache_parity_fork", lambda *a, **k: {})
    with patch("model_tools.get_tool_definitions", _session_defs(DISPATCH_SESSION)):
        fork = _dispatcher()
        results: list = []

        def _turn(user_message, conversation_history=None):
            messages: list = []
            getattr(fork, f"_execute_tool_calls_{path}")(_fork_calls(path), messages, "task-1")
            results.extend(m for m in messages if m.get("role") == "tool")
            return {"final_response": "done", "messages": []}

        monkeypatch.setattr(fork, "run_conversation", _turn)
        monkeypatch.setattr(background_review, "build_cache_parity_fork",
                            lambda parent, task_cfg, *, max_iterations, write_origin: (fork, {}, False))
        parent = _parent(DISPATCH_SESSION)
        job = _job_holding(parent, ("cw_offer",), _ctx(probe.manager)._own_tool_names)
        plugin_session_fork._launch(job)
        assert _settled(job.handle).outcome == "ok"

    assert json.dumps(fork.tools, sort_keys=False) == json.dumps(parent.tools, sort_keys=False)
    assert probe.seen == [("cw_offer", {"skill": "x"}, f"plugin_fork:{PLUGIN}")]
    by_id = {r["tool_call_id"]: str(r["content"]) for r in results}
    assert "denied" not in by_id[f"{path}-own"]
    for label in ("own_not_requested", "another_plugin", "core_deferred", "core", "connector", "describe", "search"):
        assert "was denied" in by_id[f"{path}-{label}"], (label, by_id)


@pytest.mark.parametrize("path", ["sequential", "concurrent"])
def test_unfenced_the_same_bridged_calls_reach_their_handlers(probe, path):
    """Control for the test above: outside a fork the same ``tool_call`` carriers reach the
    plugin's unrequested tool, the other plugin's tool and the connector."""
    with patch("model_tools.get_tool_definitions", _session_defs(DISPATCH_SESSION)):
        fork = _dispatcher()
        getattr(fork, f"_execute_tool_calls_{path}")(_fork_calls(path), [], "task-1")
    reached = {entry[0] for entry in probe.seen}
    assert {"cw_offer", "cw_hidden", "peer_probe", "connector"} <= reached


# ── end to end: a real turn, a real fork, Tool Search on ────────────────────────────────

def test_real_fork_with_tool_search_calls_its_deferred_tool_through_tool_call(monkeypatch):
    """The live turn's ``tools[]`` hold the bridges, not the plugin tools. The fork sends the same
    ``tools[]`` byte for byte, its prompt carries core's line, and its ``tool_call`` to ``cw_offer``
    runs the handler once while the one to ``cw_other`` (not requested) is denied."""
    from run_agent import AIAgent

    probe = SimpleNamespace(calls=[], events=[], handles=[], prompt="Worth sharing with the team?",
                            attempts=[["cw_offer"]], fork_kwargs={})
    monkeypatch.setitem(sys.modules, "_fork_probe", probe)
    _install_plugin(PLUGIN, declares=True, body=_base._E2E_PLUGIN)
    _load([PLUGIN], granted=[PLUGIN])
    monkeypatch.setattr(background_review, "_record_review_usage_to_parent", lambda *a, **k: None)
    replies = [
        _reply(tool_calls=[_bridged("call_live", "cw_offer")]),
        _reply("The answer is 42."),
        _reply(tool_calls=[_bridged("call_f1", "cw_other"), _bridged("call_f2", "cw_offer")]),
        _reply("Offered."),
    ]
    requests: list = []
    client = MagicMock()
    client.chat.completions.create.side_effect = lambda **kw: (requests.append(copy.deepcopy(kw)), replies.pop(0))[1]
    monkeypatch.setattr("agent.title_generator.maybe_auto_title", lambda *a, **k: None)
    with (
        patch("model_tools.get_tool_definitions", _session_defs(("cw_offer", "cw_other", "delegate_task"))),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI", return_value=client),
        patch("agent.model_metadata.fetch_model_metadata", return_value={}),
    ):
        parent = AIAgent(api_key="k", base_url="https://openrouter.ai/api/v1", model="test/model",
                         quiet_mode=True, skip_context_files=True, skip_memory=True)
        parent._cached_system_prompt = "You are helpful."
        parent.skip_background_review = True
        result = parent.run_conversation("what is the answer?")
        handle = _settled(next((h for h in probe.handles if h is not None), None), 30)
    side_question_lease._leases.clear()

    live, fork = requests[:2], requests[2:]
    assert result["final_response"] == "The answer is 42."
    assert "cw_offer" not in _names(live[-1]["tools"]) and "tool_call" in _names(live[-1]["tools"])
    assert json.dumps(fork[0]["tools"]) == json.dumps(live[-1]["tools"])
    assert fork[0]["messages"][0] == live[-1]["messages"][0]
    content = fork[0]["messages"][-1]["content"]
    assert content.startswith("Worth sharing with the team?\n\nCall cw_offer through tool_call")
    assert handle.outcome == "ok" and handle.result == "Offered." and handle.actions == ("cw_offer",)
    assert probe.calls == [
        {"tool": "cw_offer", "turn_origin": None, "auto_deny": None},
        {"tool": "cw_offer", "turn_origin": f"plugin_fork:{PLUGIN}", "auto_deny": "_bg_review_auto_deny"},
    ]
    denied = [m for m in fork[1]["messages"] if m.get("role") == "tool" and m.get("tool_call_id") == "call_f1"]
    assert denied and "cw_other was denied" in str(denied[0]["content"])

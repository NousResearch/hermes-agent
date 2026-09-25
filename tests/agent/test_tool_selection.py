"""Opt-in selection through native catalog, middleware, dispatcher and SDK serialization."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from agent import tool_selection as selection
from hermes_cli.config import atomic_config_write
from hermes_cli.plugins import PluginContext, get_plugin_manager
from hermes_cli.plugins_manifest import PluginManifest
from hermes_constants import (
    reset_hermes_home_override, set_hermes_home_override,
)
from tools.tool_search import BRIDGE_TOOL_NAMES


CONFIG = {"tools": {"tool_search": {
    "enabled": "on", "defer": "all", "listing": "off",
    "selection": {"enabled": True, "max_tools": 8, "max_schema_tokens": 4096},
}}}


def schema(name):
    return {"name": name, "description": f"{name} capability", "parameters": {
        "type": "object", "properties": {"value": {"type": "string", "enum": ["ok", "other"]}},
        "required": ["value"], "additionalProperties": False,
    }}


def proposal(kw, names):
    return {"selected_tools": names, "catalog_revision": kw["catalog_revision"],
            "source": "test-router", "reason": "matched"}


def tool_names(defs):
    return {s["function"]["name"] for s in defs}


@pytest.fixture
def scoped(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    token = set_hermes_home_override(home)
    atomic_config_write(home / "config.yaml", copy.deepcopy(CONFIG))
    import socket
    monkeypatch.setattr(socket.socket, "connect", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("unexpected network")))
    manager = get_plugin_manager()
    manager._discovered = True
    ctx = PluginContext(PluginManifest(name="working-set-tests"), manager)
    calls = []
    handles = []
    for name in ("selection_alpha", "selection_beta"):
        handles.append(ctx.register_tool(name, "mcp-selection-test", schema(name),
                       lambda args, _name=name, **kw: calls.append((_name, args)) or json.dumps({"ok": _name})))
    yield SimpleNamespace(home=home, ctx=ctx, manager=manager, calls=calls, handles=handles)
    manager.unload()
    reset_hermes_home_override(token)


def make_agent():
    from run_agent import AIAgent
    agent = AIAgent(model="gpt-5", provider="openai", api_mode="codex_responses",
                    api_key="test-key", base_url="https://api.openai.com/v1",
                    enabled_toolsets=["mcp-selection-test"], quiet_mode=True,
                    skip_context_files=True, skip_memory=True, max_iterations=4)
    agent._cached_system_prompt = "Test system stays unchanged."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent._current_turn_id = "turn-one"
    selection.capture_selection_context(agent, "alpha please", [], 0)
    return agent


def dispatch(agent, name, args, *, concurrent=False):
    tc = SimpleNamespace(id="selection-call", type="function",
                         function=SimpleNamespace(name=name, arguments=json.dumps(args)))
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []
    method = agent._execute_tool_calls_concurrent if concurrent else agent._execute_tool_calls_sequential
    method(msg, messages, "selection-task")
    return json.loads(messages[-1]["content"])


def bridge(agent, name, *, concurrent=False, args=None):
    return dispatch(agent, "tool_call", {"calls": [{"name": name, "arguments": args or {"value": "ok"}}]},
                    concurrent=concurrent)


def test_first_codex_http_body_replacement_stability_and_discovery(scoped, monkeypatch):
    """Actual loop -> request assembly -> Codex conversion -> OpenAI SDK -> HTTP bytes."""
    observed = []
    turns = {}
    scoped.ctx.register_hook("pre_llm_call", lambda **kw: turns.__setitem__((kw["session_id"], kw["turn_id"]), kw["user_message"]))
    def route(**kw):
        assert turns[(kw["session_id"], kw["turn_id"])] == kw["task_context"]["current_user_message"]
        observed.append(kw)
        return proposal(kw, ["selection_beta" if "beta" in kw["task_context"]["current_user_message"] else "selection_alpha"])
    scoped.ctx.register_middleware("tool_selection", route)
    bodies = []
    response = {"id": "resp_selection", "object": "response", "created_at": 0,
                "status": "completed", "model": "gpt-5", "output": [{
                    "type": "message", "id": "msg_selection", "role": "assistant", "status": "completed",
                    "content": [{"type": "output_text", "text": "ok", "annotations": []}],
                }]}
    def respond(request):
        body = json.loads(request.content)
        bodies.append(body)
        if body.get("stream"):
            events = [{"type": "response.output_item.done", "output_index": 0, "item": response["output"][0]},
                      {"type": "response.completed", "response": response}]
            return httpx.Response(200, headers={"Content-Type": "text/event-stream"},
                                  content="".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n")
        return httpx.Response(200, json=response)
    from run_agent import AIAgent
    from agent import auxiliary_client
    def http_client(*args, async_mode=False, **kwargs):
        cls = httpx.AsyncClient if async_mode else httpx.Client
        return cls(transport=httpx.MockTransport(respond))
    monkeypatch.setattr(AIAgent, "_build_keepalive_http_client", staticmethod(http_client))
    monkeypatch.setattr(auxiliary_client, "_openai_http_client_kwargs",
                        lambda url, *, async_mode=False: {"http_client": http_client(async_mode=async_mode)})
    agent = make_agent()
    canonical = copy.deepcopy(agent.tools)
    agent._use_prompt_caching = True
    first = agent.run_conversation("alpha please")
    assert bodies
    sent = {s["name"]: s for s in bodies[0]["tools"]}
    assert set(sent) == (BRIDGE_TOOL_NAMES - {"tool_search"}) | {"hermes_tool_search", "selection_alpha"}
    exact = next(r["schema"]["function"] for r in observed[-1]["catalog"] if r["name"] == "selection_alpha")
    assert sent["selection_alpha"]["parameters"] == exact["parameters"]
    assert sent["selection_alpha"]["description"] == exact["description"]
    assert observed[-1]["tool_selection_schema_version"] == "hermes.tool-selection.v1"
    assert observed[-1]["budget"] == {"max_tools": 8, "max_schema_tokens": 4096}
    assert observed[-1]["profile_name"] == scoped.ctx.profile_name == "default"
    assert observed[-1]["task_context"]["current_user_message"] == "alpha please"
    n = len(observed)
    assert selection.tools_for_request(agent) == selection.tools_for_request(agent)
    assert len(observed) == n
    assert agent.tools == canonical
    # A missed tool remains discoverable and executable, without schema accumulation.
    found = dispatch(agent, "tool_search", {"queries": ["selection_beta"]})
    assert "selection_beta" in found["tools"]
    described = dispatch(agent, "tool_describe", {"names": ["selection_beta"]})
    assert described["tools"]["selection_beta"]["parameters"] == schema("selection_beta")["parameters"]
    assert bridge(agent, "selection_beta") == {"ok": "selection_beta"}
    assert "selection_beta" not in tool_names(selection.tools_for_request(agent))
    agent.request_overrides = {"tool_choice": {"type": "function", "name": "selection_beta"}}
    agent.run_conversation("beta please", conversation_history=first["messages"])
    assert bodies[-1]["tool_choice"] == {"type": "function", "name": "selection_beta"}
    assert bodies[-1]["prompt_cache_key"] != bodies[0]["prompt_cache_key"]
    assert len(observed) == n + 1
    assert tool_names(selection.tools_for_request(agent)) == BRIDGE_TOOL_NAMES | {"selection_beta"}
    assert agent._cached_system_prompt == "Test system stays unchanged."
    assert agent.tools == canonical
    agent.client.close()


@pytest.mark.parametrize("bad", [None, {}, False, "bad", ["selection_alpha"],
                                   ["unknown"], ["selection_alpha", "selection_alpha"]])
def test_no_match_and_invalid_results_are_discovery_only(scoped, bad):
    def route(**kw):
        if isinstance(bad, list):
            result = proposal(kw, bad)
            if bad == ["selection_alpha"]:
                result["catalog_revision"] = "stale"
            return result
        return bad
    scoped.ctx.register_middleware("tool_selection", route)
    assert tool_names(selection.tools_for_request(make_agent())) == BRIDGE_TOOL_NAMES


def test_bounds_exception_arbitration_and_no_private_logs(scoped, monkeypatch, caplog):
    agent = make_agent()
    private = "private-task-value"
    def fail(**kw):
        raise RuntimeError(private)
    handle = scoped.ctx.register_middleware("tool_selection", fail)
    assert tool_names(selection.tools_for_request(agent)) == BRIDGE_TOOL_NAMES
    assert private not in caplog.text
    handle.release()
    scoped.ctx.register_middleware("tool_selection", lambda **kw: proposal(kw, ["selection_alpha"]))
    scoped.ctx.register_middleware("tool_selection", lambda **kw: proposal(kw, ["selection_beta"]))
    selection.capture_selection_context(agent, "new", [], 0)
    assert tool_names(selection.tools_for_request(agent)) == BRIDGE_TOOL_NAMES
    # The cooperative deadline refuses late returns, without threads or sleep.
    clock = iter([0.0, 0.0, 1.0])
    monkeypatch.setattr(selection.time, "monotonic", lambda: next(clock))
    assert selection.invoke_selection_callbacks([lambda **kw: {"late": True}], {}) == [False]


def test_context_bounds_omit_internal_content_and_do_not_mutate(scoped):
    from agent.context_compressor import SUMMARY_PREFIX
    rows = [{"role": "system", "content": "system secret"},
            {"role": "user", "content": SUMMARY_PREFIX + " synthetic"},
            {"role": "assistant", "content": "hidden", "reasoning": "secret"},
            {"role": "tool", "content": "tool secret"},
            {"role": "user", "content": "plain"},
            {"role": "assistant", "content": "answer"}]
    original = copy.deepcopy(rows)
    context = selection.bounded_task_context("current", rows, len(rows))
    assert context == {"current_user_message": "current", "partial": True,
                       "recent_messages": [{"role": "user", "content": "plain"}, {"role": "assistant", "content": "answer"}]}
    assert rows == original
    assert selection.bounded_task_context("x" * (128 * 1024 + 1), rows, len(rows)) is None
    assert selection.bounded_task_context([{"type": "text", "text": "x"}], rows, len(rows)) is None
    assert selection.bounded_task_context("\ud800", [], 0) is None
    many = [{"role": "user", "content": "x" * 9000}] * 9
    ctx = selection.bounded_task_context("current", many, len(many))
    assert ctx["partial"] and len(ctx["recent_messages"]) <= 4
    assert sum(len(r["content"]) for r in ctx["recent_messages"]) <= 32768
    seen = []
    scoped.ctx.register_middleware("tool_selection", lambda **kw: seen.append(kw) or proposal(kw, ["selection_alpha"]))
    agent = make_agent()
    selection.capture_selection_context(agent, "x" * (128 * 1024 + 1), [], 0)
    assert tool_names(selection.tools_for_request(agent)) == BRIDGE_TOOL_NAMES
    assert seen == []


@pytest.mark.parametrize("concurrent", [False, True])
def test_dynamic_routes_scope_and_native_veto(scoped, monkeypatch, concurrent):
    agent = make_agent()
    agent.enabled_toolsets += ["memory", "context_engine"]
    invoked = []
    agent._memory_manager = SimpleNamespace(
        get_all_tool_schemas=lambda: [schema("dynamic_memory")],
        has_tool=lambda name: name == "dynamic_memory",
        handle_tool_call=lambda name, args: invoked.append(name) or json.dumps({"ok": name}),
    )
    agent.context_compressor = SimpleNamespace(
        get_tool_schemas=lambda: [schema("dynamic_context")],
        handle_tool_call=lambda name, args, **kw: invoked.append(name) or json.dumps({"ok": name}),
    )
    from tools import bot_mode_dm
    monkeypatch.setattr(bot_mode_dm, "message_agent_authorized", lambda a: True)
    monkeypatch.setattr(bot_mode_dm, "message_agent_tool", lambda **kw: invoked.append("message_agent") or '{"ok":"message_agent"}')
    scoped.ctx.register_middleware("tool_selection", lambda **kw: proposal(kw, ["dynamic_memory", "dynamic_context", "message_agent"]))
    offered = selection.tools_for_request(agent)
    assert tool_names(offered) == BRIDGE_TOOL_NAMES | {"dynamic_memory", "dynamic_context", "message_agent"}
    for name in ("dynamic_memory", "dynamic_context", "message_agent"):
        result = dispatch(agent, "tool_describe", {"names": [name]}, concurrent=concurrent)
        assert name in result["tools"]
        args = {"target": "other", "message": "hello"} if name == "message_agent" else {"value": "ok"}
        assert bridge(agent, name, concurrent=concurrent, args=args)["ok"] == name
    assert invoked == ["dynamic_memory", "dynamic_context", "message_agent"]
    invoked.clear()
    veto = scoped.ctx.register_hook("pre_tool_call", lambda **kw: {"action": "block", "message": "test veto"})
    # Native pre_tool_call consumes the underlying name, not the tool_call wrapper.
    result = bridge(agent, "dynamic_memory", concurrent=concurrent)
    assert "error" in result and invoked == []
    veto.release()
    agent.disabled_toolsets = ["memory", "context_engine"]
    monkeypatch.setattr(bot_mode_dm, "message_agent_authorized", lambda a: False)
    for name in ("dynamic_memory", "dynamic_context", "message_agent"):
        assert "error" in bridge(agent, name, concurrent=concurrent)
    assert invoked == []


def test_revocation_budget_explicit_choice_and_default(scoped):
    observed = []
    scoped.ctx.register_middleware("tool_selection", lambda **kw: observed.append(kw) or proposal(kw, ["selection_alpha"]))
    agent = make_agent()
    assert "selection_alpha" in tool_names(selection.tools_for_request(agent))
    scoped.handles[0].release()
    assert tool_names(selection.tools_for_request(agent)) == BRIDGE_TOOL_NAMES
    assert observed[0]["catalog_revision"] != observed[-1]["catalog_revision"]
    assert "error" in bridge(agent, "selection_alpha") and scoped.calls == []
    agent.request_overrides = {"tool_choice": {"type": "function", "name": "selection_beta"}}
    assert "selection_beta" in tool_names(selection.tools_for_request(agent))
    agent.request_overrides["tool_choice"]["name"] = "unknown"
    with pytest.raises(ValueError, match="Unauthorized"):
        selection.tools_for_request(agent)
    agent.request_overrides = {}
    cfg = copy.deepcopy(CONFIG)
    cfg["tools"]["tool_search"]["selection"]["max_schema_tokens"] = 1
    atomic_config_write(scoped.home / "config.yaml", cfg)
    assert tool_names(selection.tools_for_request(agent)) == BRIDGE_TOOL_NAMES
    cfg["tools"]["tool_search"]["selection"]["enabled"] = False
    atomic_config_write(scoped.home / "config.yaml", cfg)
    assert tool_names(selection.tools_for_request(agent)) == BRIDGE_TOOL_NAMES
    cfg["tools"]["tool_search"]["defer"] = []
    atomic_config_write(scoped.home / "config.yaml", cfg)
    assert selection.tools_for_request(agent) is agent.tools


def test_profile_a_b_a_uses_own_native_manager_catalog_and_context(scoped, tmp_path):
    from agent.secret_scope import set_multiplex_active
    seen = []
    scoped.ctx.register_middleware("tool_selection", lambda **kw: seen.append(kw) or proposal(kw, ["selection_alpha"]))
    a = make_agent()
    first = selection.tools_for_request(a)
    b_home = scoped.home / "profiles" / "b"
    b_home.mkdir(parents=True)
    atomic_config_write(b_home / "config.yaml", copy.deepcopy(CONFIG))
    set_multiplex_active(True)
    token = set_hermes_home_override(b_home)
    try:
        manager = get_plugin_manager()
        manager._discovered = True
        ctx = PluginContext(PluginManifest(name="working-set-tests-b"), manager)
        ctx.register_tool("selection_gamma", "mcp-selection-test", schema("selection_gamma"), lambda args, **kw: "{}")
        ctx.register_middleware("tool_selection", lambda **kw: seen.append(kw) or proposal(kw, ["selection_gamma"]))
        b = make_agent()
        assert tool_names(selection.tools_for_request(b)) == BRIDGE_TOOL_NAMES | {"selection_gamma"}
        assert seen[-1]["profile_name"] == ctx.profile_name == "b"
        assert {r["name"] for r in seen[-1]["catalog"]} == {"selection_gamma"}
        manager.unload()
    finally:
        reset_hermes_home_override(token)
        set_multiplex_active(False)
    assert selection.tools_for_request(a) == first
    assert len(seen) == 2


@pytest.mark.parametrize("concurrent", [False, True])
def test_selected_and_missed_calls_keep_native_approval_guardrails_and_revocation(scoped, concurrent):
    from agent.tool_guardrails import ToolCallGuardrailConfig, ToolCallGuardrailController
    agent = make_agent()
    scoped.ctx.register_middleware("tool_selection", lambda **kw: proposal(kw, ["selection_alpha"]))
    selection.tools_for_request(agent)
    observed = []
    def approve(**kw):
        observed.append(kw["tool_name"])
        return {"action": "approve", "message": "requires human confirmation"}
    handle = scoped.ctx.register_hook("pre_tool_call", approve)
    # The real approval gate fails closed with no interactive human in this fixture.
    direct = dispatch(agent, "selection_alpha", {"value": "ok"}, concurrent=concurrent)
    missed = bridge(agent, "selection_beta", concurrent=concurrent)
    assert "BLOCKED" in direct["error"] and "BLOCKED" in missed["error"]
    assert observed == ["selection_alpha", "selection_beta"]
    assert scoped.calls == []
    handle.release()
    agent._tool_guardrails = ToolCallGuardrailController(ToolCallGuardrailConfig(
        hard_stop_enabled=True, exact_failure_block_after=1))
    agent._tool_guardrails.after_call("selection_beta", {"value": "ok"}, '{"error":"failed"}', failed=True)
    assert "error" in bridge(agent, "selection_beta", concurrent=concurrent)
    assert scoped.calls == []
    # Revoke after the opening scope check, inside the real native policy hook.
    scoped.ctx.register_hook("pre_tool_call", lambda **kw: scoped.handles[0].release())
    assert "error" in dispatch(agent, "selection_alpha", {"value": "ok"}, concurrent=concurrent)
    assert scoped.calls == []


def test_catalog_mutation_hard_caps_dynamic_errors_and_empty_selection(scoped, monkeypatch):
    agent = make_agent()
    assert tool_names(selection.tools_for_request(agent)) == BRIDGE_TOOL_NAMES  # no callback
    def mutate(**kw):
        row = next(r for r in kw["catalog"] if r["name"] == "selection_alpha")
        row["schema"]["function"]["parameters"]["required"] = []
        kw["task_context"]["current_user_message"] = "invented"
        return proposal(kw, ["selection_alpha"])
    scoped.ctx.register_middleware("tool_selection", mutate)
    selection.capture_selection_context(agent, "real input", [], 0)
    offered = selection.tools_for_request(agent)
    selected = next(s for s in offered if s["function"]["name"] == "selection_alpha")
    assert selected["function"]["parameters"]["required"] == ["value"]
    assert agent._tool_selection_context["current_user_message"] == "real input"
    assert "error" in bridge(agent, "selection_alpha", args={"wrong": "ok"})
    assert scoped.calls == []
    monkeypatch.setattr(selection, "MAX_CATALOG_TOOLS", 1)
    assert tool_names(selection.tools_for_request(agent)) == BRIDGE_TOOL_NAMES
    assert "error" in bridge(agent, "selection_alpha")
    monkeypatch.setattr(selection, "MAX_CATALOG_TOOLS", 4096)
    agent.enabled_toolsets += ["context_engine"]
    agent.context_compressor = SimpleNamespace(get_tool_schemas=lambda: [{"invalid": True}])
    assert tool_names(selection.tools_for_request(agent)) == BRIDGE_TOOL_NAMES
    assert "error" in bridge(agent, "selection_beta")
    assert scoped.calls == []


def test_callback_revocation_and_actual_host_budget(scoped):
    agent = make_agent()
    def revoke(**kw):
        scoped.handles[0].release()
        return proposal(kw, ["selection_alpha"])
    handle = scoped.ctx.register_middleware("tool_selection", revoke)
    assert tool_names(selection.tools_for_request(agent)) == BRIDGE_TOOL_NAMES
    assert "selection_alpha" not in agent.valid_tool_names
    handle.release()
    scoped.ctx.register_middleware("tool_selection", lambda **kw: proposal(kw, ["selection_beta"]))
    cfg = copy.deepcopy(CONFIG)
    cfg["tools"]["tool_search"]["selection"]["max_schema_tokens"] = 1
    atomic_config_write(scoped.home / "config.yaml", cfg)
    selection.capture_selection_context(agent, "beta", [], 0)
    assert tool_names(selection.tools_for_request(agent)) == BRIDGE_TOOL_NAMES
    cfg["tools"]["tool_search"]["selection"]["max_schema_tokens"] = 4096
    cfg["tools"]["tool_search"]["selection"]["max_tools"] = 0
    atomic_config_write(scoped.home / "config.yaml", cfg)
    assert tool_names(selection.tools_for_request(agent)) == BRIDGE_TOOL_NAMES


def test_execute_code_reachability_uses_authorized_inventory_not_working_set(scoped, monkeypatch):
    from tools import code_execution_tool as code
    agent = make_agent()
    agent.enabled_toolsets += ["file", "code_execution"]
    assert tool_names(selection.tools_for_request(agent)) == BRIDGE_TOOL_NAMES
    seen = []
    def execute(**kw):
        seen.append(kw["enabled_tools"])
        return json.dumps({"ok": sorted(code._sandbox_tools_for(kw["enabled_tools"]))})
    monkeypatch.setattr(code, "execute_code", execute)
    result = dispatch(agent, "execute_code", {"code": "print('test')"})
    assert "read_file" in result["ok"] and "read_file" in seen[0]
    assert "read_file" not in agent._request_visible_tool_names
    assert code._sandbox_tools_for(["execute_code"]) == frozenset()
    assert code._sandbox_tools_for([]) == frozenset()
    # No authority may be supplied through bridge arguments.
    agent.enabled_toolsets = []
    assert "error" in dispatch(agent, "tool_call", {
        "calls": [{"name": "execute_code", "arguments": {"code": "print('test')"}}],
        "current_tool_defs": [{"type": "function", "function": schema("execute_code")}],
    })
    assert len(seen) == 1


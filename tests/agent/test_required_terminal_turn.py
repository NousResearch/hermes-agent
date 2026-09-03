"""Required terminal-tool integration tests for the real agent loop."""

from __future__ import annotations

import json
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest

from agent.required_terminal_turn import (
    DEFAULT_FAILURE_RESPONSE,
    TerminalToolResult,
    _mint_natural_gateway_turn,
)
from agent.transports.types import NormalizedResponse, ToolCall
from hermes_cli.middleware import (
    run_llm_execution_middleware as ORIGINAL_LLM_EXECUTION_MIDDLEWARE,
)
from hermes_state import SessionDB
from run_agent import AIAgent
from tools.registry import registry

TOOL_NAME = "test_terminal_turn"
TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": "Resolve one terminal test turn.",
        "parameters": {
            "type": "object",
            "properties": {"intent": {"type": "string"}},
            "required": ["intent"],
        },
    },
}

DOMAIN = {
    "mode": "success",
    "mutations": 0,
    "calls": 0,
    "receipts": {},
}
_AUTO_GATEWAY_TURN = object()


def _domain_handler(args, **kwargs):
    DOMAIN["calls"] += 1
    mode = DOMAIN["mode"]
    if mode == "error":
        raise RuntimeError("domain handler failed")
    if mode == "nonterminal":
        return json.dumps({"status": "ok"})

    request_id = str(kwargs.get("request_id") or "")
    intent = str(args["intent"])
    prior = DOMAIN["receipts"].get(request_id)
    if prior is None:
        DOMAIN["mutations"] += 1
        prior = {
            "intent": intent,
            "response_text": f"committed:{request_id}:{intent}",
            "receipt": {"request_id": request_id, "mutation": DOMAIN["mutations"]},
        }
        DOMAIN["receipts"][request_id] = prior
    elif prior["intent"] != intent:
        raise RuntimeError("request id reused with a different intent")

    turn_id = str(kwargs.get("turn_id") or "")
    tool_call_id = str(kwargs.get("tool_call_id") or "")
    if mode == "wrong_turn":
        turn_id = "wrong-turn"
    elif mode == "wrong_call":
        tool_call_id = "wrong-call"

    receipt = {"invalid": {"set"}} if mode == "invalid_receipt" else prior["receipt"]

    return TerminalToolResult(
        turn_id=turn_id,
        tool_call_id=tool_call_id,
        request_id="wrong-request" if mode == "wrong_request" else request_id,
        response_text=prior["response_text"],
        receipt=receipt,
    )


@pytest.fixture(autouse=True)
def registered_terminal_tool():
    previous = registry.get_entry(TOOL_NAME)
    DOMAIN.update(mode="success", mutations=0, calls=0, receipts={})
    registry.register(
        name=TOOL_NAME,
        toolset="test-terminal",
        schema=TOOL_SCHEMA["function"],
        handler=_domain_handler,
        check_fn=lambda: True,
        terminal=True,
    )
    current = registry.get_entry(TOOL_NAME)
    try:
        yield
    finally:
        if current is not None:
            registry.restore_registration(TOOL_NAME, current, previous)


class FakeCodexTransport:
    def preflight_kwargs(self, kwargs, **_ignored):
        return dict(kwargs)

    def validate_response(self, response):
        return bool(response and response.output)

    def normalize_response(self, response, **_ignored):
        return response.normalized

    def normalize_usage(self, _response):
        return None


def _raw_response(normalized):
    return SimpleNamespace(
        id="response-test",
        model="gpt-5.6-luna",
        status="completed",
        incomplete_details=None,
        output=[{"type": "function_call"}],
        usage=None,
        normalized=normalized,
    )


def _tool_call(name=TOOL_NAME, call_id="call-test", *, intent="reply"):
    return ToolCall(
        id=call_id,
        name=name,
        arguments=json.dumps({"intent": intent}),
        provider_data={"call_id": call_id, "response_item_id": f"fc-{call_id}"},
    )


def _make_agent(
    monkeypatch,
    tmp_path,
    case_name,
    responses,
    *,
    fail_flush_role=None,
    required_tool=TOOL_NAME,
    provider="openai-codex",
    api_mode="codex_responses",
    platform="telegram",
    tamper_execution=False,
):
    from hermes_cli import config as config_module
    from hermes_cli import lifecycle
    from hermes_cli import middleware as middleware_module

    monkeypatch.setattr(
        config_module,
        "load_config_readonly",
        lambda: (
            {}
            if required_tool is None
            else {
                "agent": {
                    "required_terminal_tool": {
                        "name": required_tool,
                        "surface": "gateway",
                        "failure_response": DEFAULT_FAILURE_RESPONSE,
                    }
                }
            }
        ),
    )

    # Prove the host policy runs after public middleware: this deliberately
    # weakens the request, and the provider capture must still see the exact
    # required choice with parallel calls disabled.
    monkeypatch.setattr(
        middleware_module,
        "apply_llm_request_middleware",
        lambda payload, **_kwargs: SimpleNamespace(
            payload={
                **payload,
                "tool_choice": "auto",
                "parallel_tool_calls": True,
            },
            original_payload=dict(payload),
            trace=["tampered-by-test"],
        ),
    )
    execution_middleware = ORIGINAL_LLM_EXECUTION_MIDDLEWARE
    if tamper_execution:
        def execution_middleware(request, next_call, **_kwargs):
            return next_call(
                {
                    **request,
                    "tool_choice": "auto",
                    "parallel_tool_calls": True,
                }
            )

    monkeypatch.setattr(
        middleware_module,
        "run_llm_execution_middleware",
        execution_middleware,
    )

    db_path = tmp_path / f"{case_name}.sqlite3"
    session_db = SessionDB(db_path)
    agent = AIAgent(
        api_key="sentinel-not-a-secret",
        base_url="http://127.0.0.1:9/v1",
        model="test-model",
        provider="custom",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        session_db=session_db,
        platform=platform,
    )
    session_id = f"session-{case_name}"
    session_db.ensure_session(session_id, source="test")
    agent.session_id = session_id
    agent.provider = provider
    agent.api_mode = api_mode
    agent.model = "gpt-5.6-luna"
    agent.client = Mock()
    agent.tools = [TOOL_SCHEMA]
    agent.valid_tool_names = {TOOL_NAME}
    agent.max_iterations = 4
    agent._api_max_retries = 1

    transport = FakeCodexTransport()
    agent._get_transport = MethodType(lambda _self: transport, agent)
    agent._build_api_kwargs = MethodType(
        lambda _self, api_messages, tools_for_api=None: {
            "model": agent.model,
            "input": api_messages,
            "tools": [TOOL_SCHEMA],
            "tool_choice": "auto",
            "parallel_tool_calls": True,
        },
        agent,
    )

    requests = []
    responses = list(responses)

    def provider_call(_self, kwargs):
        requests.append(dict(kwargs))
        if not responses:
            raise AssertionError("unexpected second provider request")
        return _raw_response(responses.pop(0))

    agent._interruptible_api_call = MethodType(provider_call, agent)

    streamed = []

    def streaming_call(_self, kwargs, *, on_first_delta=None):
        if on_first_delta:
            on_first_delta()
        agent._fire_stream_delta("UNTRUSTED STREAM")
        return provider_call(agent, kwargs)

    agent._interruptible_streaming_api_call = MethodType(streaming_call, agent)

    transforms = []
    monkeypatch.setattr(
        lifecycle,
        "has_hook",
        lambda hook_name: hook_name == "transform_llm_output",
    )

    def invoke_hook(hook_name, **_kwargs):
        if hook_name == "transform_llm_output":
            transforms.append(hook_name)
            return ["UNTRUSTED TRANSFORM"]
        return []

    monkeypatch.setattr(lifecycle, "invoke_hook", invoke_hook)

    if fail_flush_role is not None:
        real_flush = agent._flush_messages_to_session_db

        def guarded_flush(_self, messages, history=None):
            if messages and messages[-1].get("role") == fail_flush_role:
                if fail_flush_role != "assistant" or (
                    len(messages) >= 2 and messages[-2].get("role") == "tool"
                ):
                    return False
            return real_flush(messages, history)

        agent._flush_messages_to_session_db = MethodType(guarded_flush, agent)

    return agent, session_db, requests, streamed, transforms


def _run_case(
    monkeypatch,
    tmp_path,
    case_name,
    response,
    *,
    platform_message_id="message-1",
    fail_flush_role=None,
    required_tool=TOOL_NAME,
    provider="openai-codex",
    api_mode="codex_responses",
    platform="telegram",
    tamper_execution=False,
    relay_rewrite=None,
    gateway_turn=_AUTO_GATEWAY_TURN,
):
    if gateway_turn is _AUTO_GATEWAY_TURN:
        gateway_turn = _mint_natural_gateway_turn(
            platform_message_id, internal=False
        )
    agent, session_db, requests, streamed, transforms = _make_agent(
        monkeypatch,
        tmp_path,
        case_name,
        [response],
        fail_flush_role=fail_flush_role,
        required_tool=required_tool,
        provider=provider,
        api_mode=api_mode,
        platform=platform,
        tamper_execution=tamper_execution,
    )
    if relay_rewrite is not None:
        from agent import relay_llm

        setattr(agent, "_disable_streaming", True)

        def relay_execute(request, callback, **_kwargs):
            return callback(relay_rewrite(dict(request)))

        monkeypatch.setattr(relay_llm, "execute", relay_execute)
    try:
        result = agent.run_conversation(
            "learner input",
            conversation_history=[],
            persist_user_platform_id=platform_message_id,
            stream_callback=streamed.append,
            gateway_turn=gateway_turn,
        )
        persisted = session_db.get_messages(agent.session_id)
        return result, persisted, requests, streamed, transforms, agent
    finally:
        agent.close()


def test_real_loop_success_is_terminal_and_persisted(monkeypatch, tmp_path):
    DOMAIN.update(mode="success", calls=0, mutations=0, receipts={})
    response = NormalizedResponse(
        content=None,
        tool_calls=[_tool_call()],
        finish_reason="stop",
    )
    result, persisted, requests, streamed, transforms, _agent = _run_case(
        monkeypatch,
        tmp_path,
        "success", response, tamper_execution=True
    )
    expected = "committed:session-success:message-1:reply"
    assert result["final_response"] == expected, result
    assert len(requests) == 1
    assert requests[0]["tool_choice"] == {
        "type": "function",
        "name": TOOL_NAME,
    }
    assert requests[0]["parallel_tool_calls"] is False
    assert streamed == []
    assert transforms == []
    assert DOMAIN["calls"] == 1 and DOMAIN["mutations"] == 1
    roles = [row["role"] for row in persisted]
    assert roles[-4:] == ["user", "assistant", "tool", "assistant"]
    assert persisted[-1]["content"] == expected


def test_relay_final_request_cannot_relax_terminal_constraint(
    monkeypatch, tmp_path
):
    response = NormalizedResponse(None, [_tool_call()], "stop")

    result, _persisted, requests, _streamed, _transforms, _agent = _run_case(
        monkeypatch,
        tmp_path,
        "relay-relaxes-constraint",
        response,
        relay_rewrite=lambda request: {
            **request,
            "tool_choice": "auto",
            "parallel_tool_calls": True,
        },
    )
    assert result["completed"] is True
    assert requests[0]["tool_choice"] == {
        "type": "function",
        "name": TOOL_NAME,
    }
    assert requests[0]["parallel_tool_calls"] is False


def test_relay_final_request_cannot_remove_terminal_tool(monkeypatch, tmp_path):
    response = NormalizedResponse(None, [_tool_call()], "stop")
    DOMAIN.update(mode="success", calls=0, mutations=0, receipts={})
    result, _persisted, requests, _streamed, _transforms, _agent = _run_case(
        monkeypatch,
        tmp_path,
        "relay-removes-tool",
        response,
        relay_rewrite=lambda request: {
            **request,
            "tools": [],
            "tool_choice": "auto",
            "parallel_tool_calls": True,
        },
    )
    assert result["final_response"] == DEFAULT_FAILURE_RESPONSE
    assert requests == []
    assert DOMAIN["calls"] == 0


def test_wrong_missing_multiple_and_mixed_prose_fail_before_execution(
    monkeypatch, tmp_path
):
    cases = {
        "missing": NormalizedResponse("UNTRUSTED", None, "stop"),
        "wrong": NormalizedResponse(None, [_tool_call("wrong_tool")], "stop"),
        "invalid_args": NormalizedResponse(
            None,
            [ToolCall(id="call-invalid", name=TOOL_NAME, arguments="{")],
            "stop",
        ),
        "multiple": NormalizedResponse(
            None, [_tool_call(call_id="one"), _tool_call(call_id="two")], "stop"
        ),
        "mixed": NormalizedResponse("UNTRUSTED", [_tool_call()], "stop"),
    }
    for name, response in cases.items():
        DOMAIN.update(mode="success", calls=0, mutations=0, receipts={})
        result, persisted, requests, streamed, transforms, _agent = _run_case(
            monkeypatch,
            tmp_path,
            name, response
        )
        assert result["final_response"] == DEFAULT_FAILURE_RESPONSE
        assert len(requests) == 1
        assert DOMAIN["calls"] == 0 and DOMAIN["mutations"] == 0
        assert streamed == [] and transforms == []
        assert "UNTRUSTED" not in json.dumps(persisted)


def test_bad_results_and_persistence_fail_closed(monkeypatch, tmp_path):
    response = NormalizedResponse(None, [_tool_call()], "stop")
    for mode in (
        "error",
        "nonterminal",
        "wrong_turn",
        "wrong_call",
        "wrong_request",
        "invalid_receipt",
    ):
        DOMAIN.update(mode=mode, calls=0, mutations=0, receipts={})
        result, persisted, _requests, streamed, transforms, _agent = _run_case(
            monkeypatch,
            tmp_path,
            f"bad-{mode}", response
        )
        assert result["final_response"] == DEFAULT_FAILURE_RESPONSE
        assert streamed == [] and transforms == []
        assert "UNTRUSTED" not in json.dumps(persisted)

    DOMAIN.update(mode="success", calls=0, mutations=0, receipts={})
    result, persisted, _requests, streamed, transforms, _agent = _run_case(
        monkeypatch,
        tmp_path,
        "fail-tool-persist", response, fail_flush_role="tool"
    )
    assert not result["final_response"].startswith("committed:")
    assert streamed == [] and transforms == []
    assert "UNTRUSTED" not in json.dumps(persisted)

    DOMAIN.update(mode="success", calls=0, mutations=0, receipts={})
    result, persisted, _requests, streamed, transforms, _agent = _run_case(
        monkeypatch,
        tmp_path,
        "fail-final-persist", response, fail_flush_role="assistant"
    )
    assert not result["final_response"].startswith("committed:")
    assert result["completed"] is False
    assert streamed == [] and transforms == []
    assert "UNTRUSTED" not in json.dumps(persisted)
    assistant_rows = [
        row for row in result["messages"] if row.get("role") == "assistant"
    ]
    assert assistant_rows[-1]["content"] == DEFAULT_FAILURE_RESPONSE
    assert not any(
        str(row.get("content") or "").startswith("committed:")
        for row in assistant_rows
    )


def test_cached_agent_resets_receipt_and_domain_replays_idempotently(
    monkeypatch, tmp_path
):
    DOMAIN.update(mode="success", calls=0, mutations=0, receipts={})
    success = NormalizedResponse(None, [_tool_call()], "stop")
    conflict = NormalizedResponse(
        None,
        [_tool_call(intent="different")],
        "stop",
    )
    missing = NormalizedResponse("UNTRUSTED", None, "stop")
    agent, session_db, requests, streamed, transforms = _make_agent(
        monkeypatch,
        tmp_path,
        "consecutive", [success, success, conflict, missing]
    )
    try:
        first = agent.run_conversation(
            "first",
            conversation_history=[],
            persist_user_platform_id="same",
            gateway_turn=_mint_natural_gateway_turn("same", internal=False),
        )
        second = agent.run_conversation(
            "retry",
            conversation_history=[],
            persist_user_platform_id="same",
            gateway_turn=_mint_natural_gateway_turn("same", internal=False),
        )
        conflict_result = agent.run_conversation(
            "conflict",
            conversation_history=[],
            persist_user_platform_id="same",
            gateway_turn=_mint_natural_gateway_turn("same", internal=False),
        )
        third = agent.run_conversation(
            "new",
            conversation_history=[],
            persist_user_platform_id="new",
            gateway_turn=_mint_natural_gateway_turn("new", internal=False),
        )
        assert first["final_response"] == second["final_response"]
        assert DOMAIN["mutations"] == 1 and DOMAIN["calls"] == 3
        assert conflict_result["final_response"] == DEFAULT_FAILURE_RESPONSE
        assert third["final_response"] == DEFAULT_FAILURE_RESPONSE
        assert len(requests) == 4 and streamed == [] and transforms == []
    finally:
        agent.close()


def test_plugin_register_tool_exposes_terminal_metadata(tmp_path):
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

    manager = PluginManager(scope_key=str(tmp_path / "plugin-scope"))
    context = PluginContext(
        PluginManifest(name="terminal-test", key="terminal-test"), manager
    )
    context.register_tool(
        name="plugin_terminal_turn",
        toolset="test-terminal",
        schema={"name": "plugin_terminal_turn"},
        handler=lambda _args, **_kwargs: "unused",
        terminal=True,
    )
    try:
        entry = registry.get_entry("plugin_terminal_turn", scope=manager.scope_key)
        assert entry is not None and entry.terminal is True
    finally:
        registry.deregister("plugin_terminal_turn", scope=manager.scope_key)


def test_policy_is_opt_in_and_unsupported_boundaries_fail_preflight(
    monkeypatch, tmp_path
):
    ordinary = NormalizedResponse(
        content="ordinary provider text",
        tool_calls=None,
        finish_reason="stop",
    )

    normal, _persisted, requests, streamed, transforms, _agent = _run_case(
        monkeypatch,
        tmp_path,
        "opt-in-off", ordinary, required_tool=None
    )
    assert normal["final_response"] == "UNTRUSTED TRANSFORM"
    assert len(requests) == 1
    assert requests[0]["tool_choice"] == "auto"
    assert streamed == ["UNTRUSTED STREAM"]
    assert transforms == ["transform_llm_output"]

    for surface in ("tui", "desktop"):
        direct, _persisted, requests, streamed, transforms, _agent = _run_case(
            monkeypatch,
            tmp_path,
            f"direct-{surface}",
            ordinary,
            platform=surface,
            gateway_turn=False,
        )
        assert direct["final_response"] == "UNTRUSTED TRANSFORM"
        assert requests[0]["tool_choice"] == "auto"
        assert streamed == ["UNTRUSTED STREAM"]
        assert transforms == ["transform_llm_output"]

    internal_marker = _mint_natural_gateway_turn(
        "internal-message", internal=True
    )
    assert internal_marker is None
    internal, _persisted, requests, streamed, transforms, _agent = _run_case(
        monkeypatch,
        tmp_path,
        "internal-gateway-event",
        ordinary,
        platform_message_id="internal-message",
        gateway_turn=internal_marker,
    )
    assert internal["final_response"] == "UNTRUSTED TRANSFORM"
    assert requests[0]["tool_choice"] == "auto"
    assert streamed == ["UNTRUSTED STREAM"]
    assert transforms == ["transform_llm_output"]

    forged, _persisted, requests, streamed, transforms, _agent = _run_case(
        monkeypatch,
        tmp_path,
        "forged-gateway-bool",
        ordinary,
        gateway_turn=True,
    )
    assert forged["final_response"] == "UNTRUSTED TRANSFORM"
    assert requests[0]["tool_choice"] == "auto"
    assert streamed == ["UNTRUSTED STREAM"]
    assert transforms == ["transform_llm_output"]

    mismatched, _persisted, requests, streamed, transforms, _agent = _run_case(
        monkeypatch,
        tmp_path,
        "mismatched-gateway-capability",
        ordinary,
        platform_message_id="message-1",
        gateway_turn=_mint_natural_gateway_turn(
            "different-message", internal=False
        ),
    )
    assert mismatched["final_response"] == DEFAULT_FAILURE_RESPONSE
    assert requests == [] and streamed == [] and transforms == []

    unsupported, _persisted, requests, streamed, transforms, _agent = _run_case(
        monkeypatch,
        tmp_path,
        "unsupported-provider",
        ordinary,
        provider="custom",
    )
    assert unsupported["final_response"] == DEFAULT_FAILURE_RESPONSE
    assert requests == [] and streamed == [] and transforms == []

    unsupported_mode, _persisted, requests, streamed, transforms, _agent = (
        _run_case(
            monkeypatch,
            tmp_path,
            "unsupported-mode",
            ordinary,
            api_mode="chat_completions",
        )
    )
    assert unsupported_mode["final_response"] == DEFAULT_FAILURE_RESPONSE
    assert requests == [] and streamed == [] and transforms == []

    fallback_agent, fallback_db, *_ = _make_agent(
        monkeypatch,
        tmp_path,
        "fallback-blocked", [ordinary]
    )
    try:
        setattr(fallback_agent, "_required_terminal_turn_active", True)
        setattr(
            fallback_agent,
            "_fallback_config",
            [{"provider": "other", "model": "other"}],
        )
        setattr(fallback_agent, "_fallback_index", 0)
        assert fallback_agent._try_activate_fallback() is False
        assert getattr(fallback_agent, "_fallback_index") == 0
    finally:
        fallback_db.close()

    missing, _persisted, requests, streamed, transforms, _agent = _run_case(
        monkeypatch,
        tmp_path,
        "missing-tool",
        ordinary,
        required_tool="absent_terminal_tool",
    )
    assert missing["final_response"] == DEFAULT_FAILURE_RESPONSE
    assert requests == [] and streamed == [] and transforms == []

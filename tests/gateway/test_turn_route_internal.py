"""Internal Gateway events must bypass external-user turn routing."""

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from gateway.config import Platform
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from gateway.run_turn_runner import TurnRunner
from gateway.turn_context import TurnContext


class _StopAfterRoute(BaseException):
    pass


@pytest.mark.parametrize(("internal", "expected_callbacks"), [(True, 0), (False, 1)])
def test_gateway_message_event_internal_identity_controls_turn_route(monkeypatch, internal, expected_callbacks):
    runner = object.__new__(GatewayRunner)
    runner.config = {}
    runner._service_tier = None
    source = SessionSource(platform=Platform.LOCAL, chat_id="chat-1", user_id="user-1")
    event = MessageEvent(text="synthetic notification", source=source, internal=internal)
    callbacks = []
    session_entry = SimpleNamespace(session_id="physical-session")
    prepared = runner._PreparedTurn(
        history=[], context_prompt="", message_text=event.text,
        persist_user_message=event.text, persist_user_timestamp=1.0,
        persist_user_display_kind="internal_notification" if internal else None,
        persistence_session_id=session_entry.session_id, persistence_owner="owner-1",
    )

    async def resolve_session(_event, resolved_source):
        return resolved_source, session_entry, "durable-session"

    async def prepare_turn(*_args):
        return prepared, None

    async def emit_hook(*_args):
        return None

    def apply_route(route, **metadata):
        callbacks.append(metadata)
        return SimpleNamespace(changed=False, payload=None, trace=[])

    async def run_agent(**kwargs):
        runner._resolve_turn_agent_config(
            kwargs["message"], "test-model",
            {
                "api_key": "test-key", "base_url": "https://example.invalid/v1",
                "provider": "custom", "requested_provider": "custom:alpha",
                "api_mode": "chat_completions", "args": [], "capabilities": {},
            },
            session_id=kwargs["session_id"], session_key=kwargs["session_key"],
            source=source, conversation_history=[], internal=kwargs["internal"],
        )
        raise _StopAfterRoute

    runner._hmwa_resolve_session = resolve_session
    runner._hmwa_prepare_turn = prepare_turn
    runner.hooks = SimpleNamespace(emit=emit_hook)
    runner._reply_anchor_for_event = lambda _event: None
    runner._run_agent = run_agent
    runner._clear_session_env = lambda _tokens: None
    monkeypatch.setattr("gateway.run_heartbeat_acceptance.heartbeat_owner_is_current", lambda *_args: True)
    monkeypatch.setattr("hermes_cli.middleware.apply_turn_route_middleware", apply_route)

    with pytest.raises(_StopAfterRoute):
        asyncio.run(runner._handle_message_with_agent(event, source, "quick-key", 1))

    assert len(callbacks) == expected_callbacks
    if callbacks:
        assert callbacks[0]["is_user_turn"] is True
        assert callbacks[0]["internal"] is False
        assert callbacks[0]["session_id"] == "physical-session"
        assert callbacks[0]["session_key"] == "durable-session"


def test_gateway_turn_context_carries_internal_identity():
    assert TurnContext().internal is False
    assert TurnContext(internal=True).internal is True


def test_gateway_route_middleware_redacts_acp_arguments(monkeypatch):
    runner = object.__new__(GatewayRunner)
    runner._service_tier = None
    captured = {}

    def apply_route(route, **metadata):
        captured["route"] = route
        captured["metadata"] = metadata
        return SimpleNamespace(changed=False, payload=route, trace=[])

    monkeypatch.setattr("hermes_cli.middleware.apply_turn_route_middleware", apply_route)
    acp_argument = "gateway-acp-token-value"
    route = runner._resolve_turn_agent_config(
        "hello",
        "test-model",
        {
            "api_key": "provider-token-value",
            "base_url": "https://example.invalid/v1",
            "provider": "custom",
            "requested_provider": "custom:alpha",
            "api_mode": "chat_completions",
            "command": "hermes-acp",
            "args": ["--api-key", acp_argument],
            "capabilities": {"secret": "capability-token-value"},
        },
        session_id="physical-session",
        session_key="durable-session",
        internal=False,
    )

    assert route["model"] == "test-model"
    assert captured["metadata"]["session_id"] == "physical-session"
    assert captured["metadata"]["session_key"] == "durable-session"
    assert acp_argument not in repr(captured["route"])
    assert "provider-token-value" not in repr(captured["route"])
    assert "capability-token-value" not in repr(captured["route"])


@pytest.mark.parametrize("internal", [True, False])
def test_turn_runner_passes_context_internal_identity_to_route_resolution(internal):
    runner = SimpleNamespace(
        _pre_agent_fallback_notice=None,
        _provider_routing={},
        _resolve_session_agent_runtime=lambda **_kwargs: ("test-model", {"provider": "custom"}),
        _resolve_session_reasoning_config=lambda **_kwargs: None,
        _resolve_session_service_tier=lambda **_kwargs: None,
    )
    # A BaseException sentinel stops run_sync immediately after the route call.
    class _StopAtRoute(BaseException):
        pass

    runner._resolve_turn_agent_config = Mock(side_effect=_StopAtRoute)
    ctx = TurnContext(
        source=SessionSource(platform=Platform.LOCAL, chat_id="chat-1", user_id="user-1"),
        message="message", session_id="physical-session", session_key="durable-session",
        history=[], user_config={}, internal=internal,
    )
    turn_runner = TurnRunner(runner, ctx)
    turn_runner._combined_ephemeral_prompt = lambda: ""
    turn_runner._setup_stream_consumer = lambda *_args: (None, None, None, False)

    with pytest.raises(_StopAtRoute):
        turn_runner.run_sync()

    assert runner._resolve_turn_agent_config.call_args.kwargs["internal"] is internal


def test_turn_route_trace_reaches_api_observer_and_refreshes_on_reuse(monkeypatch, tmp_path):
    """Route-stage entries merge with request-stage entries for pre_api_request observers."""
    from agent.turn_api_request import build_api_request
    from gateway.config import Platform
    from gateway.run_turn_routing import GatewayTurnRoutingMixin
    from gateway.run_turn_runner import TurnRunner
    from gateway.session import SessionSource
    from gateway.turn_context import TurnContext
    from hermes_cli import plugins

    manager = plugins.PluginManager(scope_key=str(tmp_path / "plugin-home"))
    manager._discovered = True
    context = plugins.PluginContext(
        plugins.PluginManifest(name="review-router", key="review-router", source="user"),
        manager,
    )
    monkeypatch.setattr(plugins, "_delivery_manager", lambda: manager)
    observed, resolver_calls = [], []
    decision = {"model": "selected-model", "reason": "route-decision"}
    context.register_middleware("turn_route", lambda route, **kw: {
        "route": {**route, "model": decision["model"]}, "reason": decision["reason"],
    })
    context.register_middleware("llm_request", lambda request, **kw: {
        "request": dict(request), "reason": "request-decision",
    })
    context.register_hook("pre_api_request", lambda **kw: observed.append(kw))
    runtime = {
        "provider": "custom", "requested_provider": "custom:local",
        "api_mode": "chat_completions", "base_url": "https://route-test.invalid/v1",
        "api_key": "inert-placeholder", "command": None, "args": [],
        "credential_pool": None,
    }

    def resolve_provider(requested, *, target_model=None, **kw):
        resolver_calls.append((requested, target_model))
        return dict(runtime)

    monkeypatch.setattr("gateway.run._resolve_runtime_agent_kwargs_for_provider", resolve_provider)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="chat", user_id="user")
    host = GatewayTurnRoutingMixin()
    host._service_tier = None
    route = host._resolve_turn_agent_config(
        "hello", "configured-model", runtime, session_id="physical", session_key="durable",
        source=source, conversation_history=[], internal=False,
    )
    route_entry = {"plugin": "review-router", "reason": "route-decision"}
    assert route["middleware_trace"] == [route_entry]
    assert resolver_calls == [("custom:local", "selected-model")]

    # Only inference-client construction is inert; host/request/hook code stays real.
    def inert_agent(**kw):
        agent = SimpleNamespace(**kw)
        agent.tools, agent._image_rejecting_models = [], set()
        agent._use_prompt_caching = agent._force_ascii_payload = False
        agent._empty_content_retries, agent.max_tokens = 0, 1024
        agent._reset_stream_delivery_tracking = lambda: None
        agent._reapply_reasoning_echo_for_provider = lambda messages: None
        agent._is_copilot_url = lambda: False
        agent._build_api_kwargs = lambda messages, **kw: {"model": agent.model, "messages": messages}
        agent._api_request_payload_for_hook = dict
        return agent

    ctx = TurnContext(
        source=source, session_id="physical", session_key="durable", user_config={},
        enabled_toolsets=[], disabled_toolsets=[], AIAgent=inert_agent,
    )
    runner = SimpleNamespace(
        _prefill_messages=None, _service_tier=None, _session_db=None,
        _refresh_fallback_model=lambda: None,
    )
    agent = TurnRunner(runner, ctx)._build_fresh_agent(route, "telegram", "", 1, None, {}, True)
    messages = [{"role": "user", "content": "hello"}]
    build_api_request(
        agent, api_messages=messages, _moa_prepared_request=None, tools_for_api=[],
        system_message="", messages=messages, original_user_message="hello",
        approx_tokens=1, total_chars=5, retry_count=0, api_call_count=1,
        api_request_id="request", api_start_time=0.0, effective_task_id="task", turn_id="turn",
    )
    assert len(observed) == 1
    assert observed[0]["model"] == "selected-model"
    assert {"reason": "request-decision"} in observed[0]["middleware_trace"]
    assert observed[0]["middleware_trace"] == [route_entry, {"reason": "request-decision"}]

    monkeypatch.setattr(TurnRunner, "_skip_context_files", lambda self, platform_key: False)
    monkeypatch.setattr(TurnRunner, "_cached_sid_is_dead", lambda self, lock, cache: (None, False))
    monkeypatch.setattr(TurnRunner, "_current_message_count", lambda self: 0)
    monkeypatch.setattr(
        TurnRunner, "_lookup_cached_agent",
        lambda self, *args: SimpleNamespace(agent=agent, reused=True, evicted=None),
    )
    runner._agent_config_signature = lambda *args, **kw: "sig"
    runner._extract_cache_busting_config = lambda cfg: {}
    runner._apply_fallback_chain_to_agent = lambda agent, chain: None

    def reused_turn_trace(next_route, turn):
        reused, was_reused = TurnRunner(runner, ctx)._resolve_turn_agent(next_route, "telegram", "", 1, None, {})
        assert reused is agent and was_reused
        observed.clear()
        build_api_request(
            agent, api_messages=messages, _moa_prepared_request=None, tools_for_api=[],
            system_message="", messages=messages, original_user_message="hello",
            approx_tokens=1, total_chars=5, retry_count=0, api_call_count=turn,
            api_request_id=f"request-{turn}", api_start_time=0.0, effective_task_id="task", turn_id=f"turn-{turn}",
        )
        return observed[0]["middleware_trace"]

    # A router that keeps the configured route still made a decision; its reason must reach observers.
    decision.update(model="configured-model", reason="keep-configured-route")
    resolver_calls.clear()
    kept = host._resolve_turn_agent_config(
        "hello", "configured-model", runtime, session_id="physical", session_key="durable",
        source=source, conversation_history=[{"role": "user", "content": "hello"}], internal=False,
    )
    keep_entry = {"plugin": "review-router", "reason": "keep-configured-route"}
    assert kept["model"] == "configured-model" and resolver_calls == []
    assert kept["middleware_trace"] == [keep_entry]
    assert reused_turn_trace(kept, 2) == [keep_entry, {"reason": "request-decision"}]

    # Reuse refreshes the trace: a later turn without a route decision must not keep the old reason.
    absent = {"model": "configured-model", "runtime": runtime, "middleware_trace": []}
    assert reused_turn_trace(absent, 3) == [{"reason": "request-decision"}]


def test_unusable_turn_route_falls_back_to_configured_route_with_warning(monkeypatch, tmp_path, caplog):
    """A changed but unusable route is rejected visibly: configured route, no trace, a warning."""
    import logging

    from gateway.config import Platform
    from gateway.run_turn_routing import GatewayTurnRoutingMixin
    from gateway.session import SessionSource
    from hermes_cli import plugins

    manager = plugins.PluginManager(scope_key=str(tmp_path / "plugin-home"))
    manager._discovered = True
    context = plugins.PluginContext(
        plugins.PluginManifest(name="review-router", key="review-router", source="user"),
        manager,
    )
    monkeypatch.setattr(plugins, "_delivery_manager", lambda: manager)
    context.register_middleware("turn_route", lambda route, **kw: {
        "route": {**route, "model": "   "}, "reason": "blank-model",
    })
    resolver_calls = []
    monkeypatch.setattr(
        "gateway.run._resolve_runtime_agent_kwargs_for_provider",
        lambda *args, **kw: resolver_calls.append(args) or {},
    )
    runtime = {
        "provider": "custom", "requested_provider": "custom:local",
        "api_mode": "chat_completions", "base_url": "https://route-test.invalid/v1",
        "api_key": "inert-placeholder", "command": None, "args": [],
        "credential_pool": None,
    }
    host = GatewayTurnRoutingMixin()
    host._service_tier = None
    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        route = host._resolve_turn_agent_config(
            "hello", "configured-model", runtime, session_id="physical", session_key="durable",
            source=SessionSource(platform=Platform.TELEGRAM, chat_id="chat", user_id="user"),
            conversation_history=[], internal=False,
        )
    assert route["model"] == "configured-model"
    assert route["runtime"]["requested_provider"] == "custom:local"
    assert "middleware_trace" not in route
    assert resolver_calls == []
    assert "unusable route" in caplog.text

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


def test_gateway_turn_context_carries_internal_identity():
    assert TurnContext().internal is False
    assert TurnContext(internal=True).internal is True


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

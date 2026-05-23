"""Tests for the local Hermes agent bridge plugin/server."""

import json
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform
from hermes_cli.plugins import VALID_HOOKS
from plugins.agent_bridge import (
    BridgeConfig,
    RoomConfig,
    _FORMAT_WAKE_TOOL_NAME,
    _agent_bridge_format_wake_message,
    _agent_bridge_llm_context,
    _escape_gateway_command,
    _format_wake_message_payload,
    _is_gateway_slash_command,
    _on_pre_llm_call,
    _wakes_self,
)
from plugins.agent_bridge.server import BridgeState


def _room(max_bot_messages: int = 16):
    return {
        "room_id": "pig_king",
        "external_targets": [{"platform": "wecom", "chat_id": "room-1"}],
        "participants": [
            {"agent_id": "agent-a", "display_name": "Agent A", "mention_names": ["@AgentA"]},
            {"agent_id": "agent-b", "display_name": "Agent B", "mention_names": ["@AgentB"]},
        ],
        "max_bot_messages": max_bot_messages,
        "idle_timeout_seconds": 1800,
    }


def test_agent_bridge_hooks_are_declared():
    assert "gateway_startup" in VALID_HOOKS
    assert "post_gateway_response" in VALID_HOOKS


def test_bridge_server_queues_human_message_only_for_other_agents():
    state = BridgeState()
    state.register({"agent_id": "agent-a", "display_name": "Agent A", "rooms": {"pig_king": _room()}})
    state.register({"agent_id": "agent-b", "display_name": "Agent B", "rooms": {"pig_king": _room()}})

    result = state.publish({
        "id": "human-1",
        "room_id": "pig_king",
        "room": _room(),
        "origin_agent_id": "agent-a",
        "author_id": "human-1",
        "author_name": "Human",
        "author_type": "human",
        "text": "hello @AgentB",
        "platform": "wecom",
        "chat_id": "room-1",
    })

    assert result["thread_id"] == "human-1"
    assert state.poll("agent-a", timeout=0)["events"] == []
    events = state.poll("agent-b", timeout=0)["events"]
    assert len(events) == 1
    assert events[0]["allow_auto_reply"] is True
    assert events[0]["text"] == "hello @AgentB"


def test_bridge_server_caps_bot_to_bot_thread():
    state = BridgeState()
    room = _room(max_bot_messages=2)
    state.register({"agent_id": "agent-a", "display_name": "Agent A", "rooms": {"pig_king": room}})
    state.register({"agent_id": "agent-b", "display_name": "Agent B", "rooms": {"pig_king": room}})

    human = state.publish({
        "id": "human-1",
        "room_id": "pig_king",
        "room": room,
        "origin_agent_id": "agent-a",
        "author_id": "human",
        "author_type": "human",
        "text": "start @AgentB",
        "platform": "wecom",
        "chat_id": "room-1",
    })
    first_bot = state.publish({
        "id": "bot-1",
        "room_id": "pig_king",
        "room": room,
        "origin_agent_id": "agent-a",
        "author_id": "agent-a",
        "author_type": "agent",
        "thread_id": human["thread_id"],
        "text": "@AgentB your turn",
        "platform": "wecom",
        "chat_id": "room-1",
    })
    second_bot = state.publish({
        "id": "bot-2",
        "room_id": "pig_king",
        "room": room,
        "origin_agent_id": "agent-b",
        "author_id": "agent-b",
        "author_type": "agent",
        "thread_id": human["thread_id"],
        "text": "@AgentA back to you",
        "platform": "wecom",
        "chat_id": "room-1",
    })

    assert first_bot["allow_auto_reply"] is True
    assert second_bot["allow_auto_reply"] is False


def test_agent_bridge_wake_names_and_command_escape():
    room = RoomConfig(
        room_id="pig_king",
        external_targets=[{"platform": "wecom", "chat_id": "room-1"}],
        participants=[{"agent_id": "agent-b", "display_name": "Agent B", "mention_names": ["Agent B", "@AgentB"]}],
    )
    cfg = BridgeConfig(
        enabled=True,
        agent_id="agent-b",
        display_name="Agent B",
        server_url="http://127.0.0.1:8791",
        token="",
        rooms={"pig_king": room},
    )

    assert _wakes_self("hello @AgentB", room, cfg)
    assert not _wakes_self("hello Agent B", room, cfg)
    assert not _wakes_self("hello @AgentA", room, cfg)
    assert _escape_gateway_command("/reset") == f"{chr(0x200B)}/reset"
    assert _is_gateway_slash_command("/new", room, cfg)
    assert _is_gateway_slash_command("  /approve", room, cfg)
    assert _is_gateway_slash_command("@AgentB /new", room, cfg)
    assert not _is_gateway_slash_command("please discuss /new behavior", room, cfg)


def test_agent_bridge_pre_dispatch_skips_slash_commands(monkeypatch):
    import plugins.agent_bridge as bridge

    room = RoomConfig(
        room_id="pig_king",
        external_targets=[{"platform": "wecom", "chat_id": "room-1"}],
        participants=[
            {"agent_id": "agent-a", "display_name": "Agent A", "mention_names": ["@AgentA"]},
            {"agent_id": "agent-b", "display_name": "Agent B", "mention_names": ["@AgentB"]},
        ],
    )
    cfg = BridgeConfig(
        enabled=True,
        agent_id="agent-a",
        display_name="Agent A",
        server_url="http://127.0.0.1:8791",
        token="",
        rooms={"pig_king": room},
    )
    runtime = SimpleNamespace(
        last_thread_by_session={},
        _log_error=lambda message, exc: None,
    )
    source = SimpleNamespace(
        platform=Platform.WECOM,
        chat_id="room-1",
        chat_type="group",
        thread_id="",
        user_id="human-1",
        user_name="Human",
        is_bot=False,
    )
    event = SimpleNamespace(source=source, text="/new", message_id="msg-1")
    gateway = SimpleNamespace(_session_key_for_source=lambda source: "session-1")
    published = []

    monkeypatch.setattr(bridge, "_current_runtime", lambda gateway=None: (runtime, cfg))
    monkeypatch.setattr(bridge, "_publish_event", lambda **kwargs: published.append(kwargs))

    assert bridge._on_pre_gateway_dispatch(event, gateway=gateway) is None
    assert published == []


def test_agent_bridge_format_wake_message_uses_current_config_identity():
    room = RoomConfig(
        room_id="pig_king",
        external_targets=[{"platform": "wecom", "chat_id": "room-1"}],
        participants=[
            {"agent_id": "agent-a", "display_name": "Agent A", "mention_names": ["Agent A", "@AgentA"]},
            {"agent_id": "agent-b", "display_name": "Agent B", "mention_names": ["Agent B", "@AgentB"]},
        ],
    )
    cfg_a = BridgeConfig(
        enabled=True,
        agent_id="agent-a",
        display_name="Agent A",
        server_url="http://127.0.0.1:8791",
        token="",
        rooms={"pig_king": room},
    )
    cfg_b = BridgeConfig(
        enabled=True,
        agent_id="agent-b",
        display_name="Agent B",
        server_url="http://127.0.0.1:8791",
        token="",
        rooms={"pig_king": room},
    )

    from_a = _format_wake_message_payload({"target_agent_id": "agent-b", "message": "what do you think?"}, cfg_a)
    from_b = _format_wake_message_payload({"target_agent_id": "agent-a", "message": "your turn"}, cfg_b)
    already_tagged = _format_wake_message_payload({"target_agent_id": "agent-b", "message": "@AgentB already tagged"}, cfg_a)

    assert from_a["success"] is True
    assert from_a["wake_text"] == "@AgentB what do you think?"
    assert from_b["success"] is True
    assert from_b["wake_text"] == "@AgentA your turn"
    assert already_tagged["wake_text"] == "@AgentB already tagged"


def test_agent_bridge_format_wake_message_errors_are_actionable():
    room = RoomConfig(
        room_id="pig_king",
        external_targets=[{"platform": "wecom", "chat_id": "room-1"}],
        participants=[
            {"agent_id": "agent-a", "display_name": "Agent A", "mention_names": ["@AgentA"]},
            {"agent_id": "agent-b", "display_name": "Agent B", "mention_names": ["@AgentB"]},
        ],
    )
    cfg = BridgeConfig(
        enabled=True,
        agent_id="agent-a",
        display_name="Agent A",
        server_url="http://127.0.0.1:8791",
        token="",
        rooms={"pig_king": room},
    )
    multi_room_cfg = BridgeConfig(
        enabled=True,
        agent_id="agent-a",
        display_name="Agent A",
        server_url="http://127.0.0.1:8791",
        token="",
        rooms={"pig_king": room, "second": room},
    )

    unknown = _format_wake_message_payload({"target_agent_id": "agent-x", "message": "hello"}, cfg)
    missing_room = _format_wake_message_payload({"target_agent_id": "agent-b", "message": "hello"}, multi_room_cfg)

    assert unknown["success"] is False
    assert unknown["available_agents"][0]["agent_id"] == "agent-b"
    assert missing_room["success"] is False
    assert "room_id is required" in missing_room["error"]


def test_agent_bridge_format_wake_message_tool_wrapper(monkeypatch):
    import plugins.agent_bridge as bridge

    room = RoomConfig(
        room_id="pig_king",
        external_targets=[{"platform": "wecom", "chat_id": "room-1"}],
        participants=[
            {"agent_id": "agent-a", "display_name": "Agent A", "mention_names": ["@AgentA"]},
            {"agent_id": "agent-b", "display_name": "Agent B", "mention_names": ["@AgentB"]},
        ],
    )
    cfg = BridgeConfig(
        enabled=True,
        agent_id="agent-a",
        display_name="Agent A",
        server_url="http://127.0.0.1:8791",
        token="",
        rooms={"pig_king": room},
    )
    monkeypatch.setattr(bridge, "_load_bridge_config", lambda: cfg)

    result = json.loads(_agent_bridge_format_wake_message({"target_agent_id": "agent-b", "message": "please answer"}))

    assert result["success"] is True
    assert result["wake_text"] == "@AgentB please answer"
    assert result["instruction"]


def test_agent_bridge_llm_context_mentions_tool_and_peers(monkeypatch):
    import plugins.agent_bridge as bridge

    room = RoomConfig(
        room_id="pig_king",
        external_targets=[{"platform": "wecom", "chat_id": "room-1"}],
        participants=[
            {"agent_id": "agent-a", "display_name": "Agent A", "mention_names": ["@AgentA"]},
            {"agent_id": "agent-b", "display_name": "Agent B", "mention_names": ["@AgentB"]},
        ],
    )
    cfg = BridgeConfig(
        enabled=True,
        agent_id="agent-a",
        display_name="Agent A",
        server_url="http://127.0.0.1:8791",
        token="",
        rooms={"pig_king": room},
    )
    monkeypatch.setattr(bridge, "_load_bridge_config", lambda: cfg)

    context = _agent_bridge_llm_context(cfg)
    hook_result = _on_pre_llm_call()

    assert _FORMAT_WAKE_TOOL_NAME in context
    assert "agent-a" in context
    assert "agent-b" in context
    assert "@AgentB" in context
    assert hook_result == {"context": context}


@pytest.mark.asyncio
async def test_bridge_event_mention_injects_internal_adapter_event(monkeypatch):
    import plugins.agent_bridge as bridge

    room = RoomConfig(
        room_id="pig_king",
        external_targets=[{"platform": "wecom", "chat_id": "room-1"}],
        participants=[
            {"agent_id": "agent-a", "display_name": "Agent A", "mention_names": ["@AgentA"]},
            {"agent_id": "agent-b", "display_name": "Agent B", "mention_names": ["@AgentB"]},
        ],
    )
    cfg = BridgeConfig(
        enabled=True,
        agent_id="agent-b",
        display_name="Agent B",
        server_url="http://127.0.0.1:8791",
        token="",
        rooms={"pig_king": room},
    )
    adapter = SimpleNamespace(handle_message=AsyncMock())
    gateway = SimpleNamespace(
        adapters={Platform.WECOM: adapter},
        config=GatewayConfig(group_sessions_per_user=False),
    )
    runtime = SimpleNamespace(
        lock=threading.RLock(),
        seen_event_ids=set(),
        gateway=gateway,
    )
    monkeypatch.setattr(bridge, "_current_runtime", lambda gateway=None: (runtime, cfg))

    await bridge._handle_bridge_event({
        "id": "evt-1",
        "room_id": "pig_king",
        "author_id": "agent-a",
        "author_name": "Agent A",
        "author_type": "agent",
        "text": "please answer @AgentB",
        "platform": "wecom",
        "chat_id": "room-1",
        "chat_type": "group",
        "allow_auto_reply": True,
    })

    adapter.handle_message.assert_awaited_once()
    injected = adapter.handle_message.await_args.args[0]
    assert injected.internal is True
    assert injected.source.user_name == "Agent A"
    assert injected.text == "please answer @AgentB"


@pytest.mark.asyncio
async def test_bridge_human_mention_is_observed_not_injected(monkeypatch):
    import plugins.agent_bridge as bridge

    room = RoomConfig(
        room_id="pig_king",
        external_targets=[{"platform": "wecom", "chat_id": "room-1"}],
        participants=[
            {"agent_id": "agent-a", "display_name": "Agent A", "mention_names": ["@AgentA"]},
            {"agent_id": "agent-b", "display_name": "Agent B", "mention_names": ["@AgentB"]},
        ],
    )
    cfg = BridgeConfig(
        enabled=True,
        agent_id="agent-b",
        display_name="Agent B",
        server_url="http://127.0.0.1:8791",
        token="",
        rooms={"pig_king": room},
    )
    adapter = SimpleNamespace(handle_message=AsyncMock())
    gateway = SimpleNamespace(
        adapters={Platform.WECOM: adapter},
        config=GatewayConfig(group_sessions_per_user=False),
    )
    runtime = SimpleNamespace(
        lock=threading.RLock(),
        seen_event_ids=set(),
        gateway=gateway,
    )
    observed = []
    monkeypatch.setattr(bridge, "_current_runtime", lambda gateway=None: (runtime, cfg))
    monkeypatch.setattr(bridge, "_append_observed_event", lambda payload, gateway: observed.append(payload))

    await bridge._handle_bridge_event({
        "id": "evt-human-1",
        "room_id": "pig_king",
        "author_id": "human-1",
        "author_name": "Human",
        "author_type": "human",
        "text": "@AgentA please call @AgentB",
        "platform": "wecom",
        "chat_id": "room-1",
        "chat_type": "group",
        "allow_auto_reply": True,
    })

    adapter.handle_message.assert_not_awaited()
    assert [event["id"] for event in observed] == ["evt-human-1"]


@pytest.mark.asyncio
async def test_bridge_bot_bare_name_is_observed_not_injected(monkeypatch):
    import plugins.agent_bridge as bridge

    room = RoomConfig(
        room_id="pig_king",
        external_targets=[{"platform": "wecom", "chat_id": "room-1"}],
        participants=[
            {"agent_id": "agent-a", "display_name": "Agent A", "mention_names": ["@AgentA"]},
            {"agent_id": "agent-b", "display_name": "Agent B", "mention_names": ["Agent B", "@AgentB"]},
        ],
    )
    cfg = BridgeConfig(
        enabled=True,
        agent_id="agent-b",
        display_name="Agent B",
        server_url="http://127.0.0.1:8791",
        token="",
        rooms={"pig_king": room},
    )
    adapter = SimpleNamespace(handle_message=AsyncMock())
    gateway = SimpleNamespace(
        adapters={Platform.WECOM: adapter},
        config=GatewayConfig(group_sessions_per_user=False),
    )
    runtime = SimpleNamespace(
        lock=threading.RLock(),
        seen_event_ids=set(),
        gateway=gateway,
    )
    observed = []
    monkeypatch.setattr(bridge, "_current_runtime", lambda gateway=None: (runtime, cfg))
    monkeypatch.setattr(bridge, "_append_observed_event", lambda payload, gateway: observed.append(payload))

    await bridge._handle_bridge_event({
        "id": "evt-bot-bare-1",
        "room_id": "pig_king",
        "author_id": "agent-a",
        "author_name": "Agent A",
        "author_type": "agent",
        "text": "Agent B was mentioned as a topic, not addressed.",
        "platform": "wecom",
        "chat_id": "room-1",
        "chat_type": "group",
        "allow_auto_reply": True,
    })

    adapter.handle_message.assert_not_awaited()
    assert [event["id"] for event in observed] == ["evt-bot-bare-1"]

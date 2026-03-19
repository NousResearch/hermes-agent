"""Tests for Discord runtime control commands added in slice 0025."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, HomeChannel, Platform, PlatformConfig
from gateway.delivery import DeliveryTarget
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key


def _make_source(**overrides) -> SessionSource:
    data = {
        "platform": Platform.DISCORD,
        "user_id": "u1",
        "user_name": "alan",
        "chat_id": "c1",
        "chat_name": "Hermes / #general",
        "chat_type": "group",
    }
    data.update(overrides)
    return SessionSource(**data)


def _make_event(
    text: str,
    *,
    source: SessionSource | None = None,
    session_source: SessionSource | None = None,
    target_source: SessionSource | None = None,
    raw_channel=None,
) -> MessageEvent:
    source = source or target_source or _make_source()
    metadata = {}
    if session_source is not None:
        metadata["session_source"] = session_source
    if target_source is not None:
        metadata["command_target_source"] = target_source
    raw_message = SimpleNamespace(channel=raw_channel) if raw_channel is not None else None
    return MessageEvent(
        text=text,
        message_type=MessageType.COMMAND,
        source=source,
        raw_message=raw_message,
        message_id="m1",
        metadata=metadata,
    )


def _make_binding(**overrides):
    data = {
        "thread_id": "thr-1",
        "session_key": "discord:thread:thr-1",
        "chat_name": "Release Thread",
        "bound_by": "alan",
        "parent_chat_id": "c1",
        "idle_timeout_minutes": 1440,
        "max_age_minutes": 0,
    }
    data.update(overrides)
    return SimpleNamespace(**data)


def _make_runner():
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={
            Platform.DISCORD: PlatformConfig(enabled=True, token="***"),
            Platform.TELEGRAM: PlatformConfig(
                enabled=True,
                token="***",
                home_channel=HomeChannel(platform=Platform.TELEGRAM, chat_id="tg-home", name="Ops Feed"),
            ),
            Platform.SLACK: PlatformConfig(
                enabled=True,
                token="***",
                home_channel=HomeChannel(platform=Platform.SLACK, chat_id="slack-home", name="Slack Ops"),
            ),
        }
    )
    adapter = MagicMock()
    adapter._get_parent_channel_id = MagicMock(return_value="c1")
    runner.adapters = {Platform.DISCORD: adapter}
    runner.session_store = MagicMock()
    runner.session_store._generate_session_key = lambda source: build_session_key(source)
    runner._running_agents = {}
    runner._voice_mode = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_send_policies = {}
    runner._session_docks = {}
    runner._save_runtime_controls = MagicMock()
    runner._schedule_gateway_restart = MagicMock()
    runner.delivery_router = SimpleNamespace(deliver=AsyncMock())
    return runner, adapter


@pytest.mark.asyncio
async def test_focus_command_binds_current_thread():
    runner, adapter = _make_runner()
    target_source = _make_source(
        chat_id="thr-1",
        chat_name="Hermes / #general / Release Thread",
        chat_type="thread",
        thread_id="thr-1",
    )
    adapter.focus_thread_binding.return_value = _make_binding(session_key=build_session_key(target_source))
    raw_channel = SimpleNamespace(parent_id="c1")

    result = await runner._handle_focus_command(
        _make_event("/focus release", target_source=target_source, raw_channel=raw_channel)
    )

    assert "Thread focused" in result
    adapter.focus_thread_binding.assert_called_once()
    kwargs = adapter.focus_thread_binding.call_args.kwargs
    assert kwargs["thread_id"] == "thr-1"
    assert kwargs["parent_chat_id"] == "c1"
    assert kwargs["bound_by"] == "alan"


@pytest.mark.asyncio
async def test_unfocus_command_removes_current_thread_binding():
    runner, adapter = _make_runner()
    target_source = _make_source(
        chat_id="thr-1",
        chat_type="thread",
        thread_id="thr-1",
    )
    adapter.unfocus_thread_binding.return_value = _make_binding()

    result = await runner._handle_unfocus_command(
        _make_event("/unfocus", target_source=target_source, raw_channel=SimpleNamespace(parent_id="c1"))
    )

    assert "Thread unfocused" in result
    adapter.unfocus_thread_binding.assert_called_once_with("thr-1")


@pytest.mark.asyncio
async def test_session_command_updates_idle_timeout():
    runner, adapter = _make_runner()
    target_source = _make_source(
        chat_id="thr-1",
        chat_type="thread",
        thread_id="thr-1",
    )
    adapter.get_thread_binding.return_value = _make_binding()
    adapter.update_thread_binding_limits.return_value = _make_binding(idle_timeout_minutes=120)

    result = await runner._handle_session_command(
        _make_event("/session idle 2h", target_source=target_source, raw_channel=SimpleNamespace(parent_id="c1"))
    )

    assert "Updated thread session `idle` to `2h`" in result
    adapter.update_thread_binding_limits.assert_called_once_with("thr-1", idle_timeout_minutes=120)


@pytest.mark.asyncio
async def test_agents_command_lists_focused_threads_for_current_chat():
    runner, adapter = _make_runner()
    target_source = _make_source(chat_id="c1", chat_type="group")
    binding = _make_binding(session_key="discord:thread:thr-1")
    adapter.list_thread_bindings.return_value = [binding]
    runner._running_agents[binding.session_key] = object()

    result = await runner._handle_agents_command(_make_event("/agents", target_source=target_source))

    assert "Focused Discord Threads" in result
    assert "`thr-1`" in result
    assert "running: yes" in result


@pytest.mark.asyncio
async def test_send_command_persists_session_policy():
    runner, _adapter = _make_runner()
    target_source = _make_source(chat_id="c9", chat_type="group")
    session_key = build_session_key(target_source)

    result = await runner._handle_send_command(_make_event("/send off", target_source=target_source))

    assert result == "📮 Send policy for this session is now `off`."
    assert runner._session_send_policies[session_key] == "off"
    runner._save_runtime_controls.assert_called_once()


@pytest.mark.asyncio
async def test_activation_command_sets_discord_chat_override():
    runner, adapter = _make_runner()
    target_source = _make_source(chat_id="c7", chat_type="group")

    result = await runner._handle_activation_command(
        _make_event("/activation always", target_source=target_source)
    )

    assert result == "🎛️ Activation mode for this chat is now `always`."
    adapter.set_activation_mode.assert_called_once_with("c7", "always")


@pytest.mark.asyncio
async def test_dock_command_persists_home_channel_target():
    runner, _adapter = _make_runner()
    target_source = _make_source(chat_id="c1", chat_type="group")
    session_key = build_session_key(target_source)

    result = await runner._handle_dock_command(
        _make_event("/dock-telegram", target_source=target_source),
        Platform.TELEGRAM,
    )

    assert "Replies for this session are now docked to telegram home channel" in result
    assert runner._session_docks[session_key] == {
        "platform": "telegram",
        "chat_id": "tg-home",
        "thread_id": None,
        "name": "Ops Feed",
    }
    runner._save_runtime_controls.assert_called_once()


@pytest.mark.asyncio
async def test_restart_command_schedules_restart():
    runner, _adapter = _make_runner()

    result = await runner._handle_restart_command(_make_event("/restart"))

    assert result == "♻️ Gateway restart scheduled. Hermes will reconnect shortly."
    runner._schedule_gateway_restart.assert_called_once_with()


@pytest.mark.asyncio
async def test_deliver_docked_response_routes_to_dock_target_and_suppresses_echo():
    runner, _adapter = _make_runner()
    session_key = "discord:session:1"
    runner._session_docks[session_key] = {
        "platform": "telegram",
        "chat_id": "tg-home",
        "thread_id": None,
        "name": "Ops Feed",
    }
    dock_target = DeliveryTarget(platform=Platform.TELEGRAM, chat_id="tg-home", thread_id=None)
    runner.delivery_router.deliver.return_value = {
        dock_target.to_string(): {"success": True},
    }

    result = await runner._deliver_docked_response(
        session_key=session_key,
        response="hello world",
        source=_make_source(chat_id="c1", chat_type="group"),
    )

    assert result is None
    runner.delivery_router.deliver.assert_awaited_once()

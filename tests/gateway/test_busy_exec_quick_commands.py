"""Exec quick commands must not become queued agent input while a turn runs."""
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource
from tests.gateway.test_slash_access_dispatch import _make_runner


class _Adapter(BasePlatformAdapter):
    """Concrete adapter whose final-reply send can be held open."""

    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="t"), Platform.SIGNAL)
        self.sent: list[str] = []

    async def connect(self, *, is_reconnect: bool = False):
        return True

    async def disconnect(self):
        pass

    async def get_chat_info(self, chat_id):
        return {}

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append(content)
        return SendResult(success=True, message_id=f"out-{len(self.sent)}")


def _event(text: str) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.SIGNAL,
            chat_id="+15550001",
            chat_type="dm",
            user_id="+15550001",
            user_name="owner",
        ),
        message_id=f"in-{text}",
    )


@pytest.fixture
def busy_gateway():
    runner = _make_runner(platform=Platform.SIGNAL)
    adapter = _Adapter()
    runner.adapters = {Platform.SIGNAL: adapter}
    runner._draining = False
    runner._restart_requested = False
    runner._busy_input_mode = "queue"
    runner._busy_text_mode = "interrupt"
    runner._session_db = None
    runner.config.quick_commands = {"limits": {"type": "exec", "command": "fixture-command"}}
    runner._hm_run_exec_quick_command = AsyncMock(return_value="exec-result")
    runner._wire_adapter_handlers(adapter, message_handler=runner._handle_message)
    event = _event("/limits")
    key = adapter._event_session_key(event)
    agent = MagicMock()
    runner._session_state(key).turn.agent = agent
    guard = asyncio.Event()
    adapter._active_sessions[key] = guard
    return runner, adapter, event, key, agent, guard


async def _dispatch(entry, runner, adapter, event):
    if entry == "adapter":
        await adapter.handle_message(event)
        return "\n".join(adapter.sent)
    return await runner._handle_message(event)


def _assert_turn_unchanged(runner, adapter, key, agent, guard):
    assert key not in adapter._pending_messages
    assert not runner._pending_messages
    assert runner._session_state(key).turn.agent is agent
    assert adapter._active_sessions[key] is guard
    agent.interrupt.assert_not_called()


@pytest.mark.asyncio
async def test_busy_exec_enters_routed_runtime_scope(busy_gateway, tmp_path):
    from hermes_constants import get_hermes_home

    runner, adapter, event, key, agent, guard = busy_gateway
    home = tmp_path / "routed-profile"
    home.mkdir()
    runner.config.multiplex_profiles = True
    runner._resolve_profile_home_for_source = lambda source: home
    original_home = get_hermes_home()

    async def execute(command, snippet):
        assert get_hermes_home() == home
        return "scoped-result"

    runner._hm_run_exec_quick_command = AsyncMock(side_effect=execute)
    assert await runner._hm_try_busy_exec_quick_command(event, event.source) == (True, "scoped-result")
    assert get_hermes_home() == original_home
    _assert_turn_unchanged(runner, adapter, key, agent, guard)


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["adapter", "runner"])
@pytest.mark.parametrize("mode", ["queue", "steer", "interrupt"])
async def test_exec_quick_command_during_active_turn(busy_gateway, entry, mode):
    runner, adapter, event, key, agent, guard = busy_gateway
    runner._busy_input_mode = mode
    result = await _dispatch(entry, runner, adapter, event)
    runner._hm_run_exec_quick_command.assert_awaited_once_with("limits", "fixture-command")
    assert result == "exec-result"
    _assert_turn_unchanged(runner, adapter, key, agent, guard)


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["adapter", "runner"])
@pytest.mark.parametrize("allowed", [False, True])
async def test_busy_exec_preserves_slash_access(busy_gateway, entry, allowed):
    runner, adapter, event, key, agent, guard = busy_gateway
    runner.config.platforms[Platform.SIGNAL].extra = {
        "allow_admin_from": ["admin"],
        "user_allowed_commands": ["limits"] if allowed else [],
    }
    result = await _dispatch(entry, runner, adapter, event)
    assert (result == "exec-result") is allowed
    assert runner._hm_run_exec_quick_command.await_count == int(allowed)
    if not allowed:
        assert "⛔" in result
    _assert_turn_unchanged(runner, adapter, key, agent, guard)


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["adapter", "runner"])
async def test_busy_exec_rejects_unauthorized_sender(busy_gateway, entry):
    runner, adapter, event, key, agent, guard = busy_gateway
    runner._is_user_authorized = lambda source: False
    await _dispatch(entry, runner, adapter, event)
    runner._hm_run_exec_quick_command.assert_not_awaited()
    _assert_turn_unchanged(runner, adapter, key, agent, guard)


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["adapter", "runner"])
async def test_busy_exec_uses_real_subprocess(busy_gateway, entry, tmp_path):
    import sys
    from gateway.run import GatewayRunner

    runner, adapter, event, key, agent, guard = busy_gateway
    script = tmp_path / "receipt.py"
    receipt = tmp_path / "receipt.txt"
    script.write_text(
        "from pathlib import Path\n"
        f"p = Path({str(receipt)!r})\n"
        "with p.open('a') as f: f.write('executed\\n')\n"
        "print('real-exec-result')\n",
        encoding="utf-8",
    )
    runner.config.quick_commands["limits"]["command"] = f'"{sys.executable}" "{script}"'
    runner._hm_run_exec_quick_command = GatewayRunner._hm_run_exec_quick_command.__get__(runner)
    result = await _dispatch(entry, runner, adapter, event)
    assert result == "real-exec-result"
    assert receipt.read_text() == "executed\n"
    _assert_turn_unchanged(runner, adapter, key, agent, guard)


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["adapter", "runner"])
async def test_busy_exec_does_not_bypass_draining(busy_gateway, entry):
    runner, adapter, event, key, agent, guard = busy_gateway
    runner._draining = True
    await _dispatch(entry, runner, adapter, event)
    runner._hm_run_exec_quick_command.assert_not_awaited()
    agent.interrupt.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["adapter", "runner"])
@pytest.mark.parametrize("kind", ["control-disabled", "unknown", "text"])
async def test_non_commands_keep_busy_queue_semantics(busy_gateway, entry, kind):
    runner, adapter, event, key, agent, guard = busy_gateway
    if kind == "control-disabled":
        event.allow_gateway_control = False
    else:
        event.text = "/unknown" if kind == "unknown" else "ordinary follow-up"
    await _dispatch(entry, runner, adapter, event)
    runner._hm_run_exec_quick_command.assert_not_awaited()
    assert key in adapter._pending_messages or runner._pending_messages
    agent.interrupt.assert_not_called()


@pytest.mark.asyncio
async def test_delivery_failure_cannot_replay_executed_command(busy_gateway):
    runner, adapter, event, key, agent, guard = busy_gateway
    runner._send_busy_ack_reply = AsyncMock(side_effect=RuntimeError("transport down"))
    await adapter.handle_message(event)
    runner._hm_run_exec_quick_command.assert_awaited_once()
    _assert_turn_unchanged(runner, adapter, key, agent, guard)


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["adapter", "runner"])
async def test_empty_command_reports_existing_error_without_queue(busy_gateway, entry):
    runner, adapter, event, key, agent, guard = busy_gateway
    runner.config.quick_commands["limits"]["command"] = ""
    result = await _dispatch(entry, runner, adapter, event)
    assert "no command" in result.lower()
    runner._hm_run_exec_quick_command.assert_not_awaited()
    _assert_turn_unchanged(runner, adapter, key, agent, guard)

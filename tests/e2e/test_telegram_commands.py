"""E2E tests for Telegram gateway slash commands.

Each test drives a message through the full async pipeline:
    adapter.handle_message(event)
        → BasePlatformAdapter._process_message_background()
        → GatewayRunner._handle_message() (command dispatch)
        → adapter.send() (captured for assertions)

No LLM involved — only gateway-level commands are tested.
"""

import pytest

from tests.e2e.conftest import (
    make_adapter,
    make_runner,
    make_session_entry,
    make_source,
    send_and_capture,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def source():
    return make_source()


@pytest.fixture()
def session_entry(source):
    return make_session_entry(source)


@pytest.fixture()
def runner(session_entry):
    return make_runner(session_entry)


@pytest.fixture()
def adapter(runner):
    return make_adapter(runner)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTelegramSlashCommands:
    """Gateway slash commands dispatched through the full adapter pipeline."""

    @pytest.mark.asyncio
    async def test_help_returns_command_list(self, adapter):
        send = await send_and_capture(adapter, "/help")

        send.assert_called_once()
        response_text = send.call_args[1].get("content") or send.call_args[0][1]
        assert "/new" in response_text
        assert "/status" in response_text

    @pytest.mark.asyncio
    async def test_status_shows_session_info(self, adapter):
        send = await send_and_capture(adapter, "/status")

        send.assert_called_once()
        response_text = send.call_args[1].get("content") or send.call_args[0][1]
        # Status output includes session metadata
        assert "session" in response_text.lower() or "Session" in response_text

    @pytest.mark.asyncio
    async def test_new_resets_session(self, adapter, runner):
        send = await send_and_capture(adapter, "/new")

        send.assert_called_once()
        runner.session_store.reset_session.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_when_no_agent_running(self, adapter):
        send = await send_and_capture(adapter, "/stop")

        send.assert_called_once()
        response_text = send.call_args[1].get("content") or send.call_args[0][1]
        response_lower = response_text.lower()
        assert "no" in response_lower or "stop" in response_lower or "not running" in response_lower

    @pytest.mark.asyncio
    async def test_commands_shows_listing(self, adapter):
        send = await send_and_capture(adapter, "/commands")

        send.assert_called_once()
        response_text = send.call_args[1].get("content") or send.call_args[0][1]
        # Should list at least some commands
        assert "/" in response_text

    @pytest.mark.asyncio
    async def test_sequential_commands_share_session(self, adapter):
        """Two commands from the same chat_id should both succeed."""
        send_help = await send_and_capture(adapter, "/help")
        send_help.assert_called_once()

        send_status = await send_and_capture(adapter, "/status")
        send_status.assert_called_once()

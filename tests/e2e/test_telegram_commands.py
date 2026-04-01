"""E2E tests for Telegram gateway slash commands.

Each test drives a message through the full async pipeline:
    adapter.handle_message(event)
        → BasePlatformAdapter._process_message_background()
        → GatewayRunner._handle_message() (command dispatch)
        → adapter.send() (captured for assertions)

No LLM involved — only gateway-level commands are tested.
"""

import asyncio
from unittest.mock import AsyncMock

import pytest

from gateway.platforms.base import SendResult
from tests.e2e.conftest import (
    make_adapter,
    make_event,
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

    @pytest.mark.asyncio
    @pytest.mark.xfail(
        reason="Bug: _handle_provider_command references unbound model_cfg when config.yaml is absent",
        strict=False,
    )
    async def test_provider_shows_current_provider(self, adapter):
        send = await send_and_capture(adapter, "/provider")

        send.assert_called_once()
        response_text = send.call_args[1].get("content") or send.call_args[0][1]
        assert "provider" in response_text.lower()

    @pytest.mark.asyncio
    async def test_verbose_responds(self, adapter):
        send = await send_and_capture(adapter, "/verbose")

        send.assert_called_once()
        response_text = send.call_args[1].get("content") or send.call_args[0][1]
        # Either shows the mode cycle or tells user to enable it in config
        assert "verbose" in response_text.lower() or "tool_progress" in response_text

    @pytest.mark.asyncio
    async def test_personality_lists_options(self, adapter):
        send = await send_and_capture(adapter, "/personality")

        send.assert_called_once()
        response_text = send.call_args[1].get("content") or send.call_args[0][1]
        assert "personalit" in response_text.lower()  # matches "personality" or "personalities"

    @pytest.mark.asyncio
    async def test_yolo_toggles_mode(self, adapter):
        send = await send_and_capture(adapter, "/yolo")

        send.assert_called_once()
        response_text = send.call_args[1].get("content") or send.call_args[0][1]
        assert "yolo" in response_text.lower()


class TestSessionLifecycle:
    """Verify session state changes across command sequences."""

    @pytest.mark.asyncio
    async def test_new_then_status_reflects_reset(self, adapter, runner, session_entry):
        """After /new, /status should report the fresh session."""
        await send_and_capture(adapter, "/new")
        runner.session_store.reset_session.assert_called_once()

        send = await send_and_capture(adapter, "/status")
        send.assert_called_once()
        response_text = send.call_args[1].get("content") or send.call_args[0][1]
        # Session ID from the entry should appear in the status output
        assert session_entry.session_id[:8] in response_text

    @pytest.mark.asyncio
    async def test_new_is_idempotent(self, adapter, runner):
        """/new called twice should not crash."""
        await send_and_capture(adapter, "/new")
        await send_and_capture(adapter, "/new")
        assert runner.session_store.reset_session.call_count == 2


class TestAuthorization:
    """Verify the pipeline handles unauthorized users."""

    @pytest.mark.asyncio
    async def test_unauthorized_user_gets_pairing_response(self, adapter, runner):
        """Unauthorized DM should trigger pairing code, not a command response."""
        runner._is_user_authorized = lambda _source: False

        event = make_event("/help")
        adapter.send.reset_mock()
        await adapter.handle_message(event)
        await asyncio.sleep(0.3)

        # The adapter.send is called directly by the authorization path
        # (not via _send_with_retry), so check it was called with a pairing message
        adapter.send.assert_called()
        response_text = adapter.send.call_args[0][1] if len(adapter.send.call_args[0]) > 1 else ""
        assert "recognize" in response_text.lower() or "pair" in response_text.lower() or "ABC123" in response_text

    @pytest.mark.asyncio
    async def test_unauthorized_user_does_not_get_help(self, adapter, runner):
        """Unauthorized user should NOT see the help command output."""
        runner._is_user_authorized = lambda _source: False

        event = make_event("/help")
        adapter.send.reset_mock()
        await adapter.handle_message(event)
        await asyncio.sleep(0.3)

        # If send was called, it should NOT contain the help text
        if adapter.send.called:
            response_text = adapter.send.call_args[0][1] if len(adapter.send.call_args[0]) > 1 else ""
            assert "/new" not in response_text


class TestSendFailureResilience:
    """Verify the pipeline handles send failures gracefully."""

    @pytest.mark.asyncio
    async def test_send_failure_does_not_crash_pipeline(self, adapter):
        """If send() returns failure, the pipeline should not raise."""
        adapter.send = AsyncMock(return_value=SendResult(success=False, error="network timeout"))
        adapter.set_message_handler(adapter._message_handler)  # re-wire with same handler

        event = make_event("/help")
        # Should not raise — pipeline handles send failures internally
        await adapter.handle_message(event)
        await asyncio.sleep(0.3)

        adapter.send.assert_called()

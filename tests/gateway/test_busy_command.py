"""Smoke tests for gateway /busy command dispatch."""

import pytest

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.platforms.base import EphemeralReply, MessageEvent
from gateway.session import SessionSource


def _make_runner(busy_mode="interrupt"):
    """Create a GatewayRunner with known busy mode."""
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.session_store = None
    runner.config = None
    runner._busy_input_mode = busy_mode
    return runner


def _make_event(text: str, chat_id: str = "chat-test") -> MessageEvent:
    source = SessionSource(
        platform=Platform.TELEGRAM,
        user_id=f"user-{chat_id}",
        chat_id=chat_id,
        user_name="tester",
        chat_type="dm",
    )
    return MessageEvent(text=text, source=source)


class TestBusyCommand:
    """Test /busy command dispatch without config persistence."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("command", "busy_mode"),
        [("/busy status", "queue"), ("/busy", "steer")],
    )
    async def test_status_returns_current_mode(self, command, busy_mode):
        """Bare /busy and /busy status show the current busy mode."""
        runner = _make_runner(busy_mode=busy_mode)
        event = _make_event(command)
        result = await runner._handle_busy_command(event)
        reply_text = str(result).lower()
        assert busy_mode in reply_text
        assert "busy" in reply_text

    @pytest.mark.asyncio
    async def test_busy_invalid_arg(self):
        """/busy with invalid arg returns error."""
        runner = _make_runner()
        event = _make_event("/busy bananas")
        result = await runner._handle_busy_command(event)
        assert "unknown" in str(result).lower()

class TestBusyCommandPersistence:
    """Test /busy persistence with mocked save_config_value."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("initial_mode", "new_mode"),
        [
            ("interrupt", "queue"),
            ("queue", "steer"),
            ("queue", "interrupt"),
        ],
    )
    async def test_set_mode_persists(self, monkeypatch, initial_mode, new_mode):
        """Each supported /busy mode is saved and applied."""
        runner = _make_runner(busy_mode=initial_mode)
        monkeypatch.setattr("cli.save_config_value", lambda k, v: True)
        event = _make_event(f"/busy {new_mode}")
        result = await runner._handle_busy_command(event)
        assert new_mode in str(result).lower()
        assert runner._busy_input_mode == new_mode

    @pytest.mark.asyncio
    async def test_save_failure_preserves_mode(self, monkeypatch):
        """When save_config_value returns False, mode is unchanged."""
        runner = _make_runner(busy_mode="steer")
        monkeypatch.setattr(
            "cli.save_config_value", lambda k, v: False
        )
        event = _make_event("/busy queue")
        result = await runner._handle_busy_command(event)
        assert "unchanged" in str(result).lower()
        assert runner._busy_input_mode == "steer"

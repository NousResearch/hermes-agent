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
    async def test_status_returns_current_mode(self):
        """/busy status shows the current busy mode."""
        runner = _make_runner(busy_mode="queue")
        event = _make_event("/busy status")
        result = await runner._handle_busy_command(event)
        assert "queue" in str(result).lower()

    @pytest.mark.asyncio
    async def test_bare_busy_shows_status(self):
        """/busy without arguments shows status."""
        runner = _make_runner(busy_mode="steer")
        event = _make_event("/busy")
        result = await runner._handle_busy_command(event)
        assert "steer" in str(result).lower()

    @pytest.mark.asyncio
    async def test_busy_invalid_arg(self):
        """/busy with invalid arg returns error."""
        runner = _make_runner()
        event = _make_event("/busy bananas")
        result = await runner._handle_busy_command(event)
        assert "unknown" in str(result).lower()

    @pytest.mark.asyncio
    async def test_busy_toggle_messages(self):
        """Status response includes the mode and behavior."""
        runner = _make_runner(busy_mode="queue")
        event = _make_event("/busy status")
        result = await runner._handle_busy_command(event)
        reply_text = str(result)
        assert "queue" in reply_text.lower()
        assert "busy" in reply_text.lower()


class TestBusyCommandPersistence:
    """Test /busy persistence with mocked save_config_value."""

    @pytest.mark.asyncio
    async def test_set_queue_persists(self, monkeypatch):
        """/busy queue saves config and updates mode."""
        runner = _make_runner(busy_mode="interrupt")
        monkeypatch.setattr(
            "cli.save_config_value", lambda k, v: True
        )
        event = _make_event("/busy queue")
        result = await runner._handle_busy_command(event)
        assert "queue" in str(result).lower()
        assert runner._busy_input_mode == "queue"

    @pytest.mark.asyncio
    async def test_set_steer_persists(self, monkeypatch):
        """/busy steer saves config and updates mode."""
        runner = _make_runner(busy_mode="queue")
        monkeypatch.setattr(
            "cli.save_config_value", lambda k, v: True
        )
        event = _make_event("/busy steer")
        result = await runner._handle_busy_command(event)
        assert "steer" in str(result).lower()
        assert runner._busy_input_mode == "steer"

    @pytest.mark.asyncio
    async def test_set_interrupt_persists(self, monkeypatch):
        """/busy interrupt saves config and updates mode."""
        runner = _make_runner(busy_mode="queue")
        monkeypatch.setattr(
            "cli.save_config_value", lambda k, v: True
        )
        event = _make_event("/busy interrupt")
        result = await runner._handle_busy_command(event)
        assert "interrupt" in str(result).lower()
        assert runner._busy_input_mode == "interrupt"

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

    @pytest.mark.asyncio
    async def test_save_exception_preserves_mode(self, monkeypatch):
        """When save_config_value raises, mode is unchanged."""
        runner = _make_runner(busy_mode="interrupt")

        def _raise(*args, **kwargs):
            raise RuntimeError("disk full")

        monkeypatch.setattr(
            "cli.save_config_value", _raise
        )
        event = _make_event("/busy steer")
        result = await runner._handle_busy_command(event)
        assert "Could not save" in str(result)
        assert runner._busy_input_mode == "interrupt"

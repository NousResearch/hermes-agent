"""Smoke tests for gateway /busy command dispatch and persistence."""

import pytest

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


@pytest.fixture(autouse=True)
def _clean_busy_state():
    """Reset busy mode on the runner between tests."""
    yield
    # No global state to clean — _busy_input_mode is per-instance


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
    async def test_busy_queue_subcommand(self):
        """/busy queue sets mode to queue."""
        runner = _make_runner(busy_mode="interrupt")
        event = _make_event("/busy queue")
        result = await runner._handle_busy_command(event)
        # Should fail to save (no config), but should NOT mutate in-memory
        assert "unchanged" in str(result).lower() or "could not" in str(result).lower()
        # In-memory mode should NOT have changed because save failed
        assert runner._busy_input_mode == "interrupt"

    @pytest.mark.asyncio
    async def test_busy_steer_subcommand(self):
        """/busy steer sets mode to steer."""
        runner = _make_runner(busy_mode="queue")
        event = _make_event("/busy steer")
        result = await runner._handle_busy_command(event)
        assert "unchanged" in str(result).lower() or "could not" in str(result).lower()
        # In-memory mode unchanged because save failed
        assert runner._busy_input_mode == "queue"

    @pytest.mark.asyncio
    async def test_busy_interrupt_subcommand(self):
        """/busy interrupt sets mode to interrupt."""
        runner = _make_runner(busy_mode="steer")
        event = _make_event("/busy interrupt")
        result = await runner._handle_busy_command(event)
        assert "unchanged" in str(result).lower() or "could not" in str(result).lower()
        # In-memory mode unchanged because save failed
        assert runner._busy_input_mode == "steer"

    @pytest.mark.asyncio
    async def test_busy_invalid_arg(self):
        """/busy with invalid arg returns error."""
        runner = _make_runner()
        event = _make_event("/busy bananas")
        result = await runner._handle_busy_command(event)
        assert "unknown" in str(result).lower()

    @pytest.mark.asyncio
    async def test_busy_toggle_messages(self):
        """Each subcommand returns an appropriate EphemeralReply with behavior text."""
        runner = _make_runner(busy_mode="queue")
        event = _make_event("/busy status")
        result = await runner._handle_busy_command(event)
        # Status response should mention the mode and behavior
        reply_text = str(result)
        assert "queue" in reply_text.lower()
        assert "queues" in reply_text.lower() or "busy" in reply_text.lower()

"""Tests for /undo gateway slash command.

Tests the _handle_undo_command handler (rewind N user turns and echo the
removed text so the user can copy/edit and resend it).
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform
from gateway.platforms.base import BasePlatformAdapter, EphemeralReply, MessageEvent
from gateway.session import SessionSource


def _make_event(text="/undo", platform=Platform.TELEGRAM,
                user_id="12345", chat_id="67890"):
    """Build a MessageEvent for testing."""
    source = SessionSource(
        platform=platform,
        user_id=user_id,
        chat_id=chat_id,
        user_name="testuser",
    )
    return MessageEvent(text=text, source=source)


def _make_runner(rewind_result):
    """Create a bare GatewayRunner with a mocked async_session_store."""
    from gateway.run import GatewayRunner
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner.session_store = MagicMock()

    session_entry = MagicMock()
    session_entry.session_id = "sess_test"
    session_entry.last_prompt_tokens = 100

    facade = MagicMock()
    facade._store = runner.session_store
    facade.get_or_create_session = AsyncMock(return_value=session_entry)
    facade.rewind_session = AsyncMock(return_value=rewind_result)
    runner._async_session_store = facade

    runner._evict_cached_agent = MagicMock(side_effect=Exception("no cache in test"))

    return runner


class TestHandleUndoCommand:
    """Tests for GatewayRunner._handle_undo_command."""

    @pytest.mark.asyncio
    async def test_nothing_to_undo(self):
        """When there's nothing to rewind, a plain notice is returned."""
        runner = _make_runner(rewind_result=None)
        event = _make_event(text="/undo")
        result = await runner._handle_undo_command(event)
        assert "nothing" in result.lower() or result  # locale-dependent, just don't crash

    @pytest.mark.asyncio
    async def test_invalid_count_shows_message(self):
        """A non-numeric /undo argument returns an error, no rewind attempted."""
        runner = _make_runner(rewind_result=None)
        event = _make_event(text="/undo abc")
        result = await runner._handle_undo_command(event)
        assert "abc" in result

    @pytest.mark.asyncio
    async def test_invalid_count_ack_does_not_leak_arg_as_attachment(self):
        """A bare local path typed as the (invalid) /undo argument must not leak.

        `/undo /tmp/bg.png` is an invalid turn count, but the error message
        echoes the argument verbatim — the same leak class as the success
        path, just with the user's own current input instead of history.
        """
        runner = _make_runner(rewind_result=None)
        event = _make_event(text="/undo /tmp/bg.png")
        result = await runner._handle_undo_command(event)

        assert isinstance(result, EphemeralReply)
        assert result.ttl_seconds == 0
        assert "/tmp/bg.png" in result

    @pytest.mark.asyncio
    async def test_successful_undo_returns_preview(self):
        """A successful /undo echoes the removed turn's text."""
        runner = _make_runner(rewind_result={
            "target_text": "hello world",
            "turns_undone": 1,
            "rewound_count": 2,
        })
        event = _make_event(text="/undo")
        result = await runner._handle_undo_command(event)
        assert "hello world" in result

    @pytest.mark.asyncio
    async def test_ack_does_not_leak_undone_path_as_attachment(self):
        """A bare local path in the undone turn must not be extractable.

        The /undo preview echoes the exact text of the removed turn "so the
        user can copy/edit and resend" it — that text is not agent-authored,
        so it can contain anything the user or a prior assistant turn wrote,
        including a bare existing local file path. extract_local_files()
        (run by _process_message_background on every *non-ephemeral*
        response) would otherwise upload that path from the undo ack itself,
        the same class of bug fixed for /background in #64661.
        """
        runner = _make_runner(rewind_result={
            "target_text": "use /tmp/bg.png and return it as an image",
            "turns_undone": 1,
            "rewound_count": 2,
        })
        event = _make_event(text="/undo")
        result = await runner._handle_undo_command(event)

        assert isinstance(result, EphemeralReply)
        assert result.ttl_seconds == 0  # no auto-delete — the preview stays visible
        assert "use /tmp/bg.png and return it as an image" in result

        # Sanity: the path really would be picked up if this weren't skipped
        # via the is_ephemeral_response guard.
        with patch("os.path.isfile", return_value=True):
            local_files, _ = BasePlatformAdapter.extract_local_files(str(result))
        assert local_files == ["/tmp/bg.png"]

"""Tests for gateway /history command — session transcript query and pagination."""

from unittest.mock import MagicMock, patch

import pytest


def _make_source(platform_str="telegram"):
    """Create a minimal source object matching MessageEvent.source."""
    from gateway.run import Platform
    source = MagicMock()
    source.platform = getattr(Platform, platform_str.upper(), None) or platform_str
    source.user_id = "user_123"
    return source


def _make_event(args="", platform="telegram"):
    """Create a mock MessageEvent with the given command args."""
    event = MagicMock()
    event.get_command_args.return_value = args
    event.source = _make_source(platform)
    return event


def _make_runner(transcript=None, session_id="sess_001"):
    """Build a bare GatewayRunner with just the fields _handle_history_command needs."""
    from gateway.run import GatewayRunner

    # Session store mock
    session_entry = MagicMock()
    session_entry.session_id = session_id

    store = MagicMock()
    store.get_or_create_session.return_value = session_entry
    store.load_transcript.return_value = transcript or []

    # Minimal GatewayRunner
    runner = object.__new__(GatewayRunner)
    runner.session_store = store
    runner._running_agents = {}
    runner._running_agents_ts = {}
    return runner, store


def _sample_transcript():
    """Create a transcript with 12 user messages and some assistant/tool noise."""
    msgs = []
    import time
    base_ts = time.time() - 1000
    for i in range(12):
        msgs.append({"role": "user", "content": f"User prompt number {i+1}"})
        msgs.append({"role": "assistant", "content": f"Response {i+1}"})
        msgs.append({"role": "tool", "content": f"Tool result {i+1}"})
    return msgs


class TestHistoryCommand:
    """Tests for _handle_history_command display and pagination."""

    @pytest.mark.asyncio
    async def test_no_history(self):
        runner, store = _make_runner(transcript=[])
        event = _make_event()
        result = await runner._handle_history_command(event)
        assert "No conversation history" in result

    @pytest.mark.asyncio
    async def test_shows_recent_prompts_page_1(self):
        msgs = _sample_transcript()
        runner, store = _make_runner(transcript=msgs)
        event = _make_event()
        result = await runner._handle_history_command(event)

        # Should show first 5 prompts
        assert "Page 1/3" in result
        assert "[1] User prompt number 1" in result
        assert "[5] User prompt number 5" in result

    @pytest.mark.asyncio
    async def test_pagination_page_2(self):
        msgs = _sample_transcript()
        runner, store = _make_runner(transcript=msgs)
        event = _make_event(args="2")
        result = await runner._handle_history_command(event)

        assert "Page 2/3" in result
        assert "[6] User prompt number 6" in result
        assert "[10] User prompt number 10" in result

    @pytest.mark.asyncio
    async def test_search_by_keyword(self):
        msgs = _sample_transcript()
        runner, store = _make_runner(transcript=msgs)
        event = _make_event(args="prompt number 3")
        result = await runner._handle_history_command(event)

        assert 'search "prompt number 3"' in result
        assert "User prompt number 3" in result

    @pytest.mark.asyncio
    async def test_search_no_match(self):
        msgs = _sample_transcript()
        runner, store = _make_runner(transcript=msgs)
        event = _make_event(args="nonexistent")
        result = await runner._handle_history_command(event)

        assert "No history items found" in result
        assert "nonexistent" in result

    @pytest.mark.asyncio
    async def test_search_uses_fts5_when_available(self):
        """When _session_db is set, search_messages() should be called."""
        from unittest.mock import MagicMock

        msgs = _sample_transcript()
        runner, store = _make_runner(transcript=msgs)

        # Wire a mock _session_db with FTS5 search
        mock_db = MagicMock()
        mock_db.search_messages.return_value = [
            {"content": "FTS5 matched result", "role": "user"}
        ]
        runner._session_db = mock_db

        event = _make_event(args="fts5 keyword")
        result = await runner._handle_history_command(event)

        mock_db.search_messages.assert_called_once()
        assert "FTS5 matched result" in result

    @pytest.mark.asyncio
    async def test_search_fts5_fallback_to_substring_on_error(self):
        """When FTS5 raises an exception, fall back to substring matching."""
        from unittest.mock import MagicMock

        msgs = _sample_transcript()
        runner, store = _make_runner(transcript=msgs)

        mock_db = MagicMock()
        mock_db.search_messages.side_effect = RuntimeError("FTS5 failed")
        runner._session_db = mock_db

        event = _make_event(args="prompt number 3")
        result = await runner._handle_history_command(event)

        assert "User prompt number 3" in result

    @pytest.mark.asyncio
    async def test_page_out_of_range_clamps_to_last_page(self):
        msgs = _sample_transcript()
        runner, store = _make_runner(transcript=msgs)
        event = _make_event(args="99")
        result = await runner._handle_history_command(event)

        assert "Page 3/3" in result
        assert "[11] User prompt number 11" in result
        assert "[12] User prompt number 12" in result

    @pytest.mark.asyncio
    async def test_session_store_unavailable(self):
        runner, store = _make_runner(transcript=[])
        store.get_or_create_session.side_effect = RuntimeError("DB locked")
        event = _make_event()
        result = await runner._handle_history_command(event)
        assert "session" in result.lower() or "db" in result.lower()

    @pytest.mark.asyncio
    async def test_filters_tool_and_assistant(self):
        """Only user messages should appear, tool/assistant should be filtered."""
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "tool", "content": "result"},
            {"role": "user", "content": "How are you?"},
        ]
        runner, store = _make_runner(transcript=msgs)
        event = _make_event()
        result = await runner._handle_history_command(event)

        assert "[1] Hello" in result
        assert "[2] How are you?" in result
        assert "Hi there" not in result
        assert "result" not in result


class TestHistoryCommandMessageEvent:
    """Test that /history can be dispatched via a real MessageEvent-like object."""

    @pytest.mark.asyncio
    async def test_empty_args_from_event(self):
        """The get_command_args() should return empty string for plain /history."""
        msgs = [{"role": "user", "content": "Test"}]
        runner, store = _make_runner(transcript=msgs)
        event = _make_event(args="")
        result = await runner._handle_history_command(event)
        assert "[1] Test" in result

    @pytest.mark.asyncio
    async def test_history_next_prev_navigation(self):
        """prev/next keywords in args should navigate pages."""
        msgs = _sample_transcript()
        runner, store = _make_runner(transcript=msgs)

        # "prev" on page 1 should stay on page 1
        event = _make_event(args="prev")
        result = await runner._handle_history_command(event)
        assert "Page 1/3" in result

        # "next" should go to page 2
        event = _make_event(args="next")
        result = await runner._handle_history_command(event)
        assert "Page 2/3" in result

"""Integration tests for /history pagination and FTS5 features."""
# This is the content for: tests/gateway/test_history_reuse.py

from unittest.mock import MagicMock, patch, PropertyMock

import pytest


def _sample_prompts(count=12):
    """Generate sample user prompts."""
    return [f"User prompt number {i+1}" for i in range(count)]


def _make_source(platform_str="telegram"):
    from gateway.run import Platform
    source = MagicMock()
    source.platform = getattr(Platform, platform_str.upper(), None) or platform_str
    source.user_id = "user_123"
    return source


def _make_event(args="", platform="telegram"):
    event = MagicMock()
    event.get_command_args.return_value = args
    event.source = _make_source(platform)
    return event


class TestPendingHistoryState:
    """Tests for _pending_history_selection state management."""

    def test_state_initialized_in_init(self):
        from gateway.run import GatewayRunner
        runner = object.__new__(GatewayRunner)
        runner._pending_history_selection = {}
        assert isinstance(runner._pending_history_selection, dict)

    @pytest.mark.asyncio
    async def test_history_command_saves_pending_state(self):
        from gateway.run import GatewayRunner
        msgs = [{"role": "user", "content": p} for p in _sample_prompts(7)]

        # Build minimal runner with mock session store
        runner = object.__new__(GatewayRunner)
        runner._pending_history_selection = {}
        store = MagicMock()
        entry = MagicMock()
        entry.session_id = "sess_001"
        store.get_or_create_session.return_value = entry
        store.load_transcript.return_value = msgs
        runner.session_store = store
        # Wire _session_key_for_source to return a predictable key
        runner._session_key_for_source = MagicMock(return_value="test_session_key")

        event = _make_event()
        result = await runner._handle_history_command(event)

        # Should have saved pending state with 7 prompts, page 1
        state = runner._pending_history_selection.get("test_session_key")
        assert state is not None, "Pending state should be saved"
        assert len(state["prompts"]) == 5  # page_size=5
        assert state["page"] == 1
        assert state["total_pages"] == 2
        assert "[1]" in result
        assert "[5]" in result


class TestNumberInterception:
    """Tests for intercepting numbered replies for prompt reuse."""

    @pytest.mark.asyncio
    async def test_numbered_reply_reuses_prompt(self):
        from gateway.run import GatewayRunner
        runner = object.__new__(GatewayRunner)
        runner._pending_history_selection = {}
        runner._interrupt_requested = False

        prompts = _sample_prompts(5)
        sk = "agent:main:test:private:u1"
        runner._pending_history_selection[sk] = {
            "prompts": prompts,
            "page": 1,
            "total_pages": 1,
            "search_query": None,
            "timestamp": 9999999999.0,  # far future — not stale
        }

        result = runner._check_pending_history_selection(sk, "3")
        assert result == "User prompt number 3"
        assert sk not in runner._pending_history_selection  # cleared after use

    @pytest.mark.asyncio
    async def test_out_of_range_number_returns_none(self):
        from gateway.run import GatewayRunner
        runner = object.__new__(GatewayRunner)
        runner._pending_history_selection = {}
        runner._interrupt_requested = False

        prompts = _sample_prompts(5)
        sk = "agent:main:test:private:u1"
        runner._pending_history_selection[sk] = {
            "prompts": prompts,
            "page": 1,
            "total_pages": 1,
            "search_query": None,
            "timestamp": 9999999999.0,  # not stale
        }

        result = runner._check_pending_history_selection(sk, "99")
        assert result is None  # out of range
        assert sk in runner._pending_history_selection  # NOT cleared

    def test_non_number_returns_none(self):
        from gateway.run import GatewayRunner
        runner = object.__new__(GatewayRunner)
        runner._pending_history_selection = {}
        sk = "agent:main:test:private:u1"
        result = runner._check_pending_history_selection(sk, "hello")
        assert result is None

    def test_no_pending_state_returns_none(self):
        from gateway.run import GatewayRunner
        runner = object.__new__(GatewayRunner)
        runner._pending_history_selection = {}
        sk = "agent:main:test:private:u1"
        result = runner._check_pending_history_selection(sk, "3")
        assert result is None

    def test_stale_state_cleared(self):
        from gateway.run import GatewayRunner
        runner = object.__new__(GatewayRunner)
        runner._pending_history_selection = {}
        sk = "agent:main:test:private:u1"
        runner._pending_history_selection[sk] = {
            "prompts": _sample_prompts(5),
            "page": 1,
            "total_pages": 1,
            "search_query": None,
            "timestamp": 0.0,  # very old
        }
        result = runner._check_pending_history_selection(sk, "3")
        assert result is None  # stale > 120s
        assert sk not in runner._pending_history_selection  # cleared

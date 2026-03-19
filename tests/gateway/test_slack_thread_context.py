"""
Tests for Slack thread context fetching.

Issue #1953: Slack bot doesn't fetch the thread when it gets mentioned.

When a user @mentions the bot in an existing thread, the bot should fetch
prior messages from that thread to provide context for its response.

This enables use cases like:
- "Summarize this thread"
- "Review this PR discussion"
- "What did we decide about X?"
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch


class TestFetchThreadContext:
    """Tests for _fetch_thread_context method."""

    @pytest.fixture
    def slack_adapter(self):
        """Create a mock SlackAdapter with a mocked client."""
        from gateway.platforms.slack import SlackAdapter, SLACK_AVAILABLE
        
        if not SLACK_AVAILABLE:
            pytest.skip("slack-bolt not installed")
            
        from gateway.config import PlatformConfig, Platform
        
        config = MagicMock(spec=PlatformConfig)
        config.token = "xoxb-test-token"
        config.platform = Platform.SLACK
        
        adapter = SlackAdapter(config)
        adapter._app = MagicMock()
        adapter._app.client = AsyncMock()
        adapter._bot_user_id = "U12345BOT"
        adapter._user_name_cache = {}
        
        return adapter

    @pytest.mark.asyncio
    async def test_fetch_thread_context_success(self, slack_adapter):
        """Test successful thread context fetching."""
        # Mock conversations.replies response
        slack_adapter._app.client.conversations_replies = AsyncMock(return_value={
            "messages": [
                {"user": "U111", "text": "Hey team, should we refactor this?"},
                {"user": "U222", "text": "Yes, I think we should use the new API"},
                {"user": "U333", "text": "Agreed. What about the timeline?"},
                {"user": "U111", "text": f"<@{slack_adapter._bot_user_id}> can you help?"},  # trigger
            ]
        })
        
        # Mock user name resolution
        async def resolve_user(user_id):
            names = {"U111": "Alice", "U222": "Bob", "U333": "Charlie"}
            return names.get(user_id, user_id)
        slack_adapter._resolve_user_name = resolve_user
        
        context = await slack_adapter._fetch_thread_context("C123", "1234567890.000001")
        
        # Should include all messages except the last (trigger)
        assert "[Thread context" in context
        assert "[Alice]:" in context
        assert "[Bob]:" in context
        assert "[Charlie]:" in context
        assert "refactor" in context
        assert "new API" in context
        # Should not include the @bot mention message
        assert "can you help" not in context

    @pytest.mark.asyncio
    async def test_fetch_thread_context_empty(self, slack_adapter):
        """Test handling empty thread."""
        slack_adapter._app.client.conversations_replies = AsyncMock(return_value={
            "messages": []
        })
        
        context = await slack_adapter._fetch_thread_context("C123", "1234567890.000001")
        assert context == ""

    @pytest.mark.asyncio
    async def test_fetch_thread_context_no_thread_ts(self, slack_adapter):
        """Test handling missing thread_ts."""
        context = await slack_adapter._fetch_thread_context("C123", None)
        assert context == ""

    @pytest.mark.asyncio
    async def test_fetch_thread_context_api_error(self, slack_adapter):
        """Test graceful handling of API errors."""
        slack_adapter._app.client.conversations_replies = AsyncMock(
            side_effect=Exception("API error")
        )
        
        context = await slack_adapter._fetch_thread_context("C123", "1234567890.000001")
        assert context == ""  # Should fail gracefully

    @pytest.mark.asyncio
    async def test_fetch_thread_context_skips_bot_messages(self, slack_adapter):
        """Test that bot messages are excluded from context."""
        slack_adapter._app.client.conversations_replies = AsyncMock(return_value={
            "messages": [
                {"user": "U111", "text": "Question about deployment"},
                {"bot_id": "B123", "text": "I can help with that!"},  # bot message
                {"user": "U111", "text": f"<@{slack_adapter._bot_user_id}> now what?"},
            ]
        })
        
        async def resolve_user(user_id):
            return "Alice" if user_id == "U111" else user_id
        slack_adapter._resolve_user_name = resolve_user
        
        context = await slack_adapter._fetch_thread_context("C123", "1234567890.000001")
        
        assert "[Alice]:" in context
        assert "Question about deployment" in context
        assert "I can help with that" not in context  # bot message excluded

    @pytest.mark.asyncio
    async def test_fetch_thread_context_respects_limit(self, slack_adapter):
        """Test that limit parameter is passed to API."""
        slack_adapter._app.client.conversations_replies = AsyncMock(return_value={
            "messages": []
        })
        
        await slack_adapter._fetch_thread_context("C123", "1234567890.000001", limit=5)
        
        slack_adapter._app.client.conversations_replies.assert_called_once_with(
            channel="C123",
            ts="1234567890.000001",
            limit=5,
        )


class TestMessageHandlerThreadIntegration:
    """Integration tests for thread context in message handler."""

    @pytest.fixture
    def slack_adapter(self):
        """Create a SlackAdapter with full mocking."""
        from gateway.platforms.slack import SlackAdapter, SLACK_AVAILABLE
        
        if not SLACK_AVAILABLE:
            pytest.skip("slack-bolt not installed")
            
        from gateway.config import PlatformConfig, Platform
        
        config = MagicMock(spec=PlatformConfig)
        config.token = "xoxb-test-token"
        config.platform = Platform.SLACK
        
        adapter = SlackAdapter(config)
        adapter._app = MagicMock()
        adapter._app.client = AsyncMock()
        adapter._bot_user_id = "U12345BOT"
        adapter._user_name_cache = {}
        adapter.handle_message = AsyncMock()
        adapter._add_reaction = AsyncMock(return_value=True)
        adapter._remove_reaction = AsyncMock(return_value=True)
        
        return adapter

    @pytest.mark.asyncio
    async def test_message_in_thread_includes_context(self, slack_adapter):
        """Test that messages in threads include fetched context."""
        # Mock thread context
        slack_adapter._app.client.conversations_replies = AsyncMock(return_value={
            "messages": [
                {"user": "U111", "text": "We need to review the PR"},
                {"user": "U222", "text": "It has some issues"},
                {"user": "U111", "text": f"<@{slack_adapter._bot_user_id}> review this"},
            ]
        })
        
        async def resolve_user(user_id):
            return {"U111": "Alice", "U222": "Bob"}.get(user_id, user_id)
        slack_adapter._resolve_user_name = resolve_user
        
        # Simulate incoming message event in a thread
        event = {
            "text": f"<@{slack_adapter._bot_user_id}> summarize",
            "user": "U111",
            "channel": "C123",
            "ts": "1234567890.000010",
            "thread_ts": "1234567890.000001",  # In a thread
            "channel_type": "channel",
        }
        
        await slack_adapter._handle_slack_message(event)
        
        # Verify handle_message was called
        assert slack_adapter.handle_message.called
        
        # Check the text includes thread context
        call_args = slack_adapter.handle_message.call_args
        msg_event = call_args[0][0]
        
        # The text should include thread context + the summarize request
        assert "[Thread context" in msg_event.text
        assert "summarize" in msg_event.text

    @pytest.mark.asyncio
    async def test_dm_message_no_thread_context(self, slack_adapter):
        """Test that DMs don't try to fetch thread context."""
        slack_adapter._app.client.conversations_replies = AsyncMock()
        
        event = {
            "text": "Hello bot",
            "user": "U111",
            "channel": "D123",
            "ts": "1234567890.000010",
            "channel_type": "im",  # DM
        }
        
        await slack_adapter._handle_slack_message(event)
        
        # Should NOT call conversations.replies for DMs
        slack_adapter._app.client.conversations_replies.assert_not_called()

    @pytest.mark.asyncio
    async def test_top_level_message_no_thread_context(self, slack_adapter):
        """Test that top-level channel messages don't fetch thread context."""
        slack_adapter._app.client.conversations_replies = AsyncMock()
        
        event = {
            "text": f"<@{slack_adapter._bot_user_id}> hello",
            "user": "U111",
            "channel": "C123",
            "ts": "1234567890.000010",
            # No thread_ts = top-level message
            "channel_type": "channel",
        }
        
        await slack_adapter._handle_slack_message(event)
        
        # Should NOT call conversations.replies for top-level messages
        slack_adapter._app.client.conversations_replies.assert_not_called()

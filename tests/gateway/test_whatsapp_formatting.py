"""Tests for WhatsApp message formatting and chunking.

Covers:
- format_message(): markdown → WhatsApp syntax conversion
- send(): message chunking for long responses
- MAX_MESSAGE_LENGTH: practical UX limit
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform, PlatformConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adapter():
    """Create a WhatsAppAdapter with test attributes (bypass __init__)."""
    from gateway.platforms.whatsapp import WhatsAppAdapter

    adapter = WhatsAppAdapter.__new__(WhatsAppAdapter)
    adapter.platform = Platform.WHATSAPP
    adapter.config = MagicMock()
    adapter.config.extra = {}
    adapter._bridge_port = 3000
    adapter._bridge_script = "/tmp/test-bridge.js"
    adapter._session_path = MagicMock()
    adapter._bridge_log_fh = None
    adapter._bridge_log = None
    adapter._bridge_process = None
    adapter._reply_prefix = None
    adapter._running = True
    adapter._message_handler = None
    adapter._fatal_error_code = None
    adapter._fatal_error_message = None
    adapter._fatal_error_retryable = True
    adapter._fatal_error_handler = None
    adapter._active_sessions = {}
    adapter._pending_messages = {}
    adapter._background_tasks = set()
    adapter._auto_tts_disabled_chats = set()
    adapter._message_queue = asyncio.Queue()
    adapter._http_session = MagicMock()
    adapter._mention_patterns = []
    return adapter


class _AsyncCM:
    """Minimal async context manager returning a fixed value."""

    def __init__(self, value):
        self.value = value

    async def __aenter__(self):
        return self.value

    async def __aexit__(self, *exc):
        return False


# ---------------------------------------------------------------------------
# format_message tests
# ---------------------------------------------------------------------------

class TestFormatMessage:
    """WhatsApp markdown conversion."""

    def test_bold_double_asterisk(self):
        adapter = _make_adapter()
        assert adapter.format_message("**hello**") == "*hello*"

    def test_bold_double_underscore(self):
        adapter = _make_adapter()
        assert adapter.format_message("__hello__") == "*hello*"

    def test_strikethrough(self):
        adapter = _make_adapter()
        assert adapter.format_message("~~deleted~~") == "~deleted~"

    def test_headers_converted_to_bold(self):
        adapter = _make_adapter()
        assert adapter.format_message("# Title") == "*Title*"
        assert adapter.format_message("## Subtitle") == "*Subtitle*"
        assert adapter.format_message("### Deep") == "*Deep*"

    def test_links_converted(self):
        adapter = _make_adapter()
        result = adapter.format_message("[click here](https://example.com)")
        assert result == "click here (https://example.com)"

    def test_code_blocks_protected(self):
        """Code blocks should not have their content reformatted."""
        adapter = _make_adapter()
        content = "before **bold** ```python\n**not bold**\n``` after **bold**"
        result = adapter.format_message(content)
        assert "```python\n**not bold**\n```" in result
        assert result.startswith("before *bold*")
        assert result.endswith("after *bold*")

    def test_inline_code_protected(self):
        """Inline code should not have its content reformatted."""
        adapter = _make_adapter()
        content = "use `**raw**` here"
        result = adapter.format_message(content)
        assert "`**raw**`" in result
        assert result.startswith("use ")

    def test_empty_content(self):
        adapter = _make_adapter()
        assert adapter.format_message("") == ""
        assert adapter.format_message(None) is None

    def test_plain_text_unchanged(self):
        adapter = _make_adapter()
        assert adapter.format_message("hello world") == "hello world"

    def test_already_whatsapp_italic(self):
        """Single *italic* should pass through unchanged."""
        adapter = _make_adapter()
        # After bold conversion, *text* is WhatsApp italic
        assert adapter.format_message("*italic*") == "*italic*"

    def test_multiline_mixed(self):
        adapter = _make_adapter()
        content = "# Header\n\n**Bold text** and ~~strike~~\n\n```\ncode\n```"
        result = adapter.format_message(content)
        assert "*Header*" in result
        assert "*Bold text*" in result
        assert "~strike~" in result
        assert "```\ncode\n```" in result


# ---------------------------------------------------------------------------
# MAX_MESSAGE_LENGTH tests
# ---------------------------------------------------------------------------

class TestMessageLimits:
    """WhatsApp message length limits."""

    def test_max_message_length_is_practical(self):
        from gateway.platforms.whatsapp import WhatsAppAdapter
        assert WhatsAppAdapter.MAX_MESSAGE_LENGTH == 4096

    def test_chunk_limit_reserves_default_self_chat_prefix(self, monkeypatch):
        adapter = _make_adapter()
        monkeypatch.delenv("WHATSAPP_REPLY_PREFIX", raising=False)
        monkeypatch.setenv("WHATSAPP_MODE", "self-chat")

        assert adapter._outgoing_chunk_limit() == (
            adapter.MAX_MESSAGE_LENGTH - len(adapter.DEFAULT_REPLY_PREFIX)
        )

    def test_chunk_limit_does_not_reserve_prefix_in_bot_mode(self, monkeypatch):
        adapter = _make_adapter()
        monkeypatch.setenv("WHATSAPP_MODE", "bot")

        assert adapter._outgoing_chunk_limit() == adapter.MAX_MESSAGE_LENGTH


# ---------------------------------------------------------------------------
# send() chunking tests
# ---------------------------------------------------------------------------

class TestSendChunking:
    """WhatsApp send() splits long messages into chunks."""

    @pytest.mark.asyncio
    async def test_short_message_single_send(self):
        adapter = _make_adapter()
        resp = MagicMock(status=200)
        resp.json = AsyncMock(return_value={"messageId": "msg1"})
        adapter._http_session.post = MagicMock(return_value=_AsyncCM(resp))

        result = await adapter.send("chat1", "short message")
        assert result.success
        # Only one call to bridge /send
        assert adapter._http_session.post.call_count == 1

    @pytest.mark.asyncio
    async def test_long_message_chunked(self):
        adapter = _make_adapter()
        resp = MagicMock(status=200)
        resp.json = AsyncMock(return_value={"messageId": "msg1"})
        adapter._http_session.post = MagicMock(return_value=_AsyncCM(resp))

        # Create a message longer than MAX_MESSAGE_LENGTH (4096)
        long_msg = "a " * 3000  # ~6000 chars

        result = await adapter.send("chat1", long_msg)
        assert result.success
        # Should have made multiple calls
        assert adapter._http_session.post.call_count > 1

    @pytest.mark.asyncio
    async def test_chunks_leave_room_for_bridge_prefix(self, monkeypatch):
        adapter = _make_adapter()
        monkeypatch.delenv("WHATSAPP_REPLY_PREFIX", raising=False)
        monkeypatch.setenv("WHATSAPP_MODE", "self-chat")
        resp = MagicMock(status=200)
        resp.json = AsyncMock(return_value={"messageId": "msg1"})
        adapter._http_session.post = MagicMock(return_value=_AsyncCM(resp))

        long_msg = "a " * 3000

        await adapter.send("chat1", long_msg)

        for call in adapter._http_session.post.call_args_list:
            payload = call.kwargs.get("json") or call[1].get("json")
            final_text = adapter.DEFAULT_REPLY_PREFIX + payload["message"]
            assert len(final_text) <= adapter.MAX_MESSAGE_LENGTH

    @pytest.mark.asyncio
    async def test_empty_message_no_send(self):
        adapter = _make_adapter()
        result = await adapter.send("chat1", "")
        assert result.success
        assert adapter._http_session.post.call_count == 0

    @pytest.mark.asyncio
    async def test_whitespace_only_no_send(self):
        adapter = _make_adapter()
        result = await adapter.send("chat1", "   \n  ")
        assert result.success
        assert adapter._http_session.post.call_count == 0

    @pytest.mark.asyncio
    async def test_format_applied_before_send(self):
        """Markdown should be converted to WhatsApp format before sending."""
        adapter = _make_adapter()
        resp = MagicMock(status=200)
        resp.json = AsyncMock(return_value={"messageId": "msg1"})
        adapter._http_session.post = MagicMock(return_value=_AsyncCM(resp))

        await adapter.send("chat1", "**bold text**")

        # Check the payload sent to the bridge
        call_args = adapter._http_session.post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        assert payload["message"] == "*bold text*"

    @pytest.mark.asyncio
    async def test_reply_to_only_on_first_chunk(self):
        """reply_to should only be set on the first chunk."""
        adapter = _make_adapter()
        resp = MagicMock(status=200)
        resp.json = AsyncMock(return_value={"messageId": "msg1"})
        adapter._http_session.post = MagicMock(return_value=_AsyncCM(resp))

        long_msg = "word " * 2000  # ~10000 chars, multiple chunks

        await adapter.send("chat1", long_msg, reply_to="orig123")

        calls = adapter._http_session.post.call_args_list
        assert len(calls) > 1

        # First chunk should have replyTo
        first_payload = calls[0].kwargs.get("json") or calls[0][1].get("json")
        assert first_payload.get("replyTo") == "orig123"

        # Subsequent chunks should NOT have replyTo
        for call in calls[1:]:
            payload = call.kwargs.get("json") or call[1].get("json")
            assert "replyTo" not in payload

    @pytest.mark.asyncio
    async def test_bridge_error_returns_failure(self):
        adapter = _make_adapter()
        resp = MagicMock(status=500)
        resp.text = AsyncMock(return_value="Internal Server Error")
        adapter._http_session.post = MagicMock(return_value=_AsyncCM(resp))

        result = await adapter.send("chat1", "hello")
        assert not result.success
        assert "Internal Server Error" in result.error

    @pytest.mark.asyncio
    async def test_not_connected_returns_failure(self):
        adapter = _make_adapter()
        adapter._running = False

        result = await adapter.send("chat1", "hello")
        assert not result.success
        assert "Not connected" in result.error


# ---------------------------------------------------------------------------
# display_config tier classification
# ---------------------------------------------------------------------------

class TestWhatsAppTier:
    """WhatsApp should be classified as TIER_MEDIUM."""

    def test_whatsapp_streaming_follows_global(self):
        from gateway.display_config import resolve_display_setting
        # TIER_MEDIUM has streaming: None (follow global), not False
        assert resolve_display_setting({}, "whatsapp", "streaming") is None

    def test_whatsapp_tool_progress_is_new(self):
        from gateway.display_config import resolve_display_setting
        assert resolve_display_setting({}, "whatsapp", "tool_progress") == "new"

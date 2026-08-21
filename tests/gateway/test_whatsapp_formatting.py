"""Tests for WhatsApp message formatting and chunking.

Covers:
- format_message(): markdown → WhatsApp syntax conversion
- send(): message chunking for long responses
- MAX_MESSAGE_LENGTH: practical UX limit
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform


@pytest.fixture(autouse=True)
def _whatsapp_open_optin(monkeypatch):
    """Opt into WhatsApp allow-all so ``dm_policy: open`` dispatch tests run.

    The adapter fails closed on ``open`` without an allow-all opt-in
    (SECURITY.md 2.6); these formatting/dispatch-mechanics tests set
    ``_dm_policy = "open"`` as a stand-in for "process this DM".
    """
    monkeypatch.setenv("WHATSAPP_ALLOW_ALL_USERS", "true")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adapter():
    """Create a WhatsAppAdapter with test attributes (bypass __init__)."""
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

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
    adapter._dm_policy = "open"
    adapter._allow_from = set()
    adapter._group_policy = "open"
    adapter._group_allow_from = set()
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


    def test_strikethrough(self):
        adapter = _make_adapter()
        assert adapter.format_message("~~deleted~~") == "~deleted~"

    def test_headers_converted_to_bold(self):
        adapter = _make_adapter()
        assert adapter.format_message("# Title") == "*Title*"
        assert adapter.format_message("## Subtitle") == "*Subtitle*"
        assert adapter.format_message("### Deep") == "*Deep*"

    def test_bold_header_does_not_double_wrap(self):
        """"# **Title**" must become *Title*, not **Title** (WhatsApp would
        render the doubled asterisks literally)."""
        adapter = _make_adapter()
        assert adapter.format_message("# **Title**") == "*Title*"
        assert adapter.format_message("## __Strong__") == "*Strong*"


    def test_already_whatsapp_italic(self):
        """Markdown *italic* converts to WhatsApp _italic_ (PR #58704)."""
        adapter = _make_adapter()
        assert adapter.format_message("*italic*") == "_italic_"
        # Already-WhatsApp _italic_ passes through unchanged
        assert adapter.format_message("_italic_") == "_italic_"


# ---------------------------------------------------------------------------
# MAX_MESSAGE_LENGTH tests
# ---------------------------------------------------------------------------

class TestMessageLimits:
    """WhatsApp message length limits."""


    def test_chunk_limit_reserves_default_self_chat_prefix(self, monkeypatch):
        adapter = _make_adapter()
        monkeypatch.delenv("WHATSAPP_REPLY_PREFIX", raising=False)
        monkeypatch.setenv("WHATSAPP_MODE", "self-chat")

        assert adapter._outgoing_chunk_limit() == (
            adapter.MAX_MESSAGE_LENGTH - len(adapter.DEFAULT_REPLY_PREFIX)
        )


# ---------------------------------------------------------------------------
# edit_message() formatting tests (#80061)
# ---------------------------------------------------------------------------

def _edit_adapter():
    """Adapter wired so edit_message reaches the bridge POST."""
    adapter = _make_adapter()
    adapter._running = True
    adapter._check_managed_bridge_exit = AsyncMock(return_value=None)
    resp = MagicMock(status=200)
    resp.json = AsyncMock(return_value={"messageId": "msg1"})
    adapter._http_session.post = MagicMock(return_value=_AsyncCM(resp))
    return adapter


def _posted_message(adapter):
    return adapter._http_session.post.call_args.kwargs["json"]["message"]


class TestEditMessageFormatting:
    """Streaming replies are delivered as edits, so edits must format too.

    ``send()`` converts markdown to WhatsApp syntax before posting; before
    #80061 ``edit_message`` posted the raw model output, so every progressive
    frame of a streamed reply showed literal ``**asterisks`` until (and unless)
    a plain send happened to replace it.
    """

    @pytest.mark.asyncio
    async def test_edit_converts_markdown_to_whatsapp_syntax(self):
        adapter = _edit_adapter()

        result = await adapter.edit_message("chat1", "msg1", "**bold** and *italic*")

        assert result.success
        assert _posted_message(adapter) == "*bold* and _italic_"

    @pytest.mark.asyncio
    async def test_edit_applies_the_full_conversion(self):
        """Asserted against literal expected output, not against
        format_message() itself -- comparing the two would pass for any
        implementation, including a broken one."""
        adapter = _edit_adapter()

        await adapter.edit_message(
            "chat1", "msg1", "# Title\n**bold** ~~struck~~ [text](http://x)"
        )

        assert _posted_message(adapter) == "*Title*\n*bold* ~struck~ text (http://x)"

    @pytest.mark.asyncio
    async def test_edit_preserves_code_fences(self):
        """Fenced blocks are protected by format_message and must survive."""
        adapter = _edit_adapter()
        raw = "before\n```\n**not bold**\n```\nafter"

        await adapter.edit_message("chat1", "msg1", raw)

        assert "```\n**not bold**\n```" in _posted_message(adapter)

    @pytest.mark.asyncio
    async def test_streaming_prefixes_never_corrupt_the_frame(self):
        """edit_message is the only caller fed PARTIAL text.

        send() only ever sees a complete message; a streamed edit sees
        prefixes, which routinely carry an unmatched ``**`` or an unclosed
        backtick. Those must pass through untouched rather than emit a stray
        delimiter, and the final frame must equal the fully converted message.
        """
        adapter = _edit_adapter()
        full = "Here is **bold** and `code` done"

        for cut in range(1, len(full) + 1):
            adapter._http_session.post.reset_mock()
            await adapter.edit_message("chat1", "msg1", full[:cut])
            frame = _posted_message(adapter)
            # A prefix may be converted or verbatim, but never gains a
            # delimiter that was not derivable from the text it was given.
            assert frame.count("_") <= full[:cut].count("_") + full[:cut].count("*")
            assert "**" not in frame or "**" in full[:cut]

        assert _posted_message(adapter) == "Here is *bold* and `code` done"

    @pytest.mark.parametrize("partial", ["**bold", "*ital", "`code", "a ** b"])
    def test_unbalanced_delimiters_pass_through(self, partial):
        """Mid-stream fragments with unclosed markers are left alone."""
        adapter = _make_adapter()
        assert adapter.format_message(partial) == partial

    @pytest.mark.asyncio
    async def test_edit_does_not_double_convert(self):
        """format_message is not idempotent (*x* -> _x_), so the adapter must
        format exactly once. Text already in WhatsApp italic stays italic."""
        adapter = _edit_adapter()

        await adapter.edit_message("chat1", "msg1", "_italic_")

        assert _posted_message(adapter) == "_italic_"


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


# ---------------------------------------------------------------------------
# bridge event metadata
# ---------------------------------------------------------------------------

class TestBridgeEventMetadata:
    """WhatsApp bridge metadata is preserved for downstream consumers."""

    @pytest.mark.asyncio
    async def test_quoted_reply_metadata_is_preserved_in_raw_message(self):
        adapter = _make_adapter()
        data = {
            "messageId": "incoming-msg",
            "chatId": "15551234567@s.whatsapp.net",
            "senderId": "15551234567@s.whatsapp.net",
            "senderName": "Tester",
            "chatName": "Tester",
            "isGroup": False,
            "body": "approved",
            "hasMedia": False,
            "mediaUrls": [],
            "quotedMessageId": "outbound-msg",
            "quotedParticipant": "99999999999@s.whatsapp.net",
            "quotedRemoteJid": "15551234567@s.whatsapp.net",
            "hasQuotedMessage": True,
        }

        event = await adapter._build_message_event(data)

        assert event is not None
        assert event.raw_message["quotedMessageId"] == "outbound-msg"
        assert event.raw_message["quotedParticipant"] == "99999999999@s.whatsapp.net"
        assert event.raw_message["quotedRemoteJid"] == "15551234567@s.whatsapp.net"
        assert event.raw_message["hasQuotedMessage"] is True


# ---------------------------------------------------------------------------
# display_config tier classification
# ---------------------------------------------------------------------------

class TestWhatsAppTier:
    """WhatsApp should be classified as TIER_MEDIUM."""

    def test_whatsapp_streaming_follows_global(self):
        from gateway.display_config import resolve_display_setting
        # TIER_MEDIUM has streaming: None (follow global), not False
        assert resolve_display_setting({}, "whatsapp", "streaming") is None


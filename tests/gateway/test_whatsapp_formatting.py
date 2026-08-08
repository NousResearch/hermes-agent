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

    @pytest.mark.asyncio
    async def test_reply_to_uncaptioned_image_attaches_quoted_media(self, tmp_path, monkeypatch):
        # contextInfo.quotedMessage only ever carries a thumbnail-sized stub
        # for media (or nothing for an uncaptioned attachment). The bridge
        # resolves the quoted message's already-downloaded media via its own
        # cache (createQuotedMediaCache) and hands back the real cached path
        # in quotedMediaUrls. The adapter must fold that into this event's
        # own media_urls/media_types so the existing vision pipeline picks it
        # up — otherwise a reply like "save this" to an uncaptioned photo
        # someone else sent looks to the agent like there is no image at all.
        adapter = _make_adapter()

        cache_dir = tmp_path / "cache" / "image"
        cache_dir.mkdir(parents=True)
        quoted_image_path = cache_dir / "img_original.jpg"
        quoted_image_path.write_bytes(b"fake-jpeg-bytes")

        from plugins.platforms.whatsapp import adapter as adapter_module
        monkeypatch.setattr(
            adapter_module, "_is_allowed_bridge_path", lambda url: True,
        )

        data = {
            "messageId": "reply-msg",
            "chatId": "15551234567@s.whatsapp.net",
            "senderId": "15551234567@s.whatsapp.net",
            "senderName": "Ananya",
            "chatName": "Family",
            "isGroup": True,
            "body": "did you save this wedding invite?",
            "hasMedia": False,
            "mediaUrls": [],
            "mediaType": "",
            "quotedMessageId": "original-image-msg",
            "quotedParticipant": "99999999999@s.whatsapp.net",
            "quotedRemoteJid": "15551234567@s.whatsapp.net",
            "hasQuotedMessage": True,
            "quotedText": "",
            "quotedMediaUrls": [str(quoted_image_path)],
            "quotedMediaType": "image",
        }

        event = await adapter._build_message_event(data)

        assert event is not None
        assert str(quoted_image_path) in event.media_urls
        idx = event.media_urls.index(str(quoted_image_path))
        assert event.media_types[idx] == "image/jpeg"

    @pytest.mark.asyncio
    async def test_quoted_media_path_outside_cache_dir_is_rejected(self, monkeypatch):
        # _is_allowed_bridge_path guards against a compromised/buggy bridge
        # handing back an arbitrary absolute path; quoted-media handling must
        # respect the same guard as direct media, not bypass it.
        adapter = _make_adapter()

        from plugins.platforms.whatsapp import adapter as adapter_module
        monkeypatch.setattr(
            adapter_module, "_is_allowed_bridge_path", lambda url: False,
        )

        data = {
            "messageId": "reply-msg-2",
            "chatId": "15551234567@s.whatsapp.net",
            "senderId": "15551234567@s.whatsapp.net",
            "senderName": "Ananya",
            "chatName": "Family",
            "isGroup": True,
            "body": "did you save this?",
            "hasMedia": False,
            "mediaUrls": [],
            "mediaType": "",
            "quotedMessageId": "original-image-msg",
            "quotedParticipant": "99999999999@s.whatsapp.net",
            "quotedRemoteJid": "15551234567@s.whatsapp.net",
            "hasQuotedMessage": True,
            "quotedText": "",
            "quotedMediaUrls": ["/etc/passwd"],
            "quotedMediaType": "image",
        }

        event = await adapter._build_message_event(data)

        assert event is not None
        assert "/etc/passwd" not in event.media_urls


# ---------------------------------------------------------------------------
# display_config tier classification
# ---------------------------------------------------------------------------

class TestWhatsAppTier:
    """WhatsApp should be classified as TIER_MEDIUM."""

    def test_whatsapp_streaming_follows_global(self):
        from gateway.display_config import resolve_display_setting
        # TIER_MEDIUM has streaming: None (follow global), not False
        assert resolve_display_setting({}, "whatsapp", "streaming") is None


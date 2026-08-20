"""Tests for WhatsApp message formatting and chunking.

Covers:
- format_message(): markdown → WhatsApp syntax conversion
- send(): message chunking for long responses
- MAX_MESSAGE_LENGTH: practical UX limit
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform, PlatformConfig


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
    adapter._split_outgoing_on_blank_lines = False
    adapter._split_outgoing_delay_seconds = 0.6
    adapter._split_outgoing_max_parts = 4
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


class TestOutgoingSplitConfig:
    def test_defaults_are_opt_out_with_documented_values(self):
        from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

        adapter = WhatsAppAdapter(PlatformConfig(enabled=True))

        assert adapter._split_outgoing_on_blank_lines is False
        assert adapter._split_outgoing_delay_seconds == 0.6
        assert adapter._split_outgoing_max_parts == 4

    def test_values_are_config_backed_and_invalid_values_fall_back_safely(self):
        from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

        configured = WhatsAppAdapter(
            PlatformConfig(
                enabled=True,
                extra={
                    "split_outgoing_on_blank_lines": "yes",
                    "split_outgoing_delay_seconds": "1.25",
                    "split_outgoing_max_parts": "7",
                },
            )
        )
        invalid = WhatsAppAdapter(
            PlatformConfig(
                enabled=True,
                extra={
                    "split_outgoing_on_blank_lines": "sometimes",
                    "split_outgoing_delay_seconds": "nan",
                    "split_outgoing_max_parts": -2,
                },
            )
        )
        malformed_max = WhatsAppAdapter(
            PlatformConfig(
                enabled=True,
                extra={"split_outgoing_max_parts": "many"},
            )
        )

        assert configured._split_outgoing_on_blank_lines is True
        assert configured._split_outgoing_delay_seconds == 1.25
        assert configured._split_outgoing_max_parts == 7
        assert invalid._split_outgoing_on_blank_lines is False
        assert invalid._split_outgoing_delay_seconds == 0.6
        assert invalid._split_outgoing_max_parts == 4
        assert malformed_max._split_outgoing_max_parts == 4


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
    async def test_blank_line_split_is_opt_in(self):
        adapter = _make_adapter()
        resp = MagicMock(status=200)
        resp.json = AsyncMock(return_value={"messageId": "msg1"})
        adapter._http_session.post = MagicMock(return_value=_AsyncCM(resp))

        await adapter.send("chat1", "one\n\ntwo")

        payloads = [call.kwargs["json"] for call in adapter._http_session.post.call_args_list]
        assert [payload["message"] for payload in payloads] == ["one\n\ntwo"]

    @pytest.mark.asyncio
    async def test_blank_lines_split_bubbles_but_single_newlines_do_not(self):
        adapter = _make_adapter()
        adapter._split_outgoing_on_blank_lines = True
        resp = MagicMock(status=200)
        resp.json = AsyncMock(return_value={"messageId": "msg1"})
        adapter._http_session.post = MagicMock(return_value=_AsyncCM(resp))

        await adapter.send("chat1", "one\n\ntwo\nthree")

        payloads = [call.kwargs["json"] for call in adapter._http_session.post.call_args_list]
        assert [payload["message"] for payload in payloads] == ["one", "two\nthree"]

    @pytest.mark.asyncio
    async def test_blank_line_split_merges_remainder_at_max_parts(self):
        adapter = _make_adapter()
        adapter._split_outgoing_on_blank_lines = True
        adapter._split_outgoing_max_parts = 3
        resp = MagicMock(status=200)
        resp.json = AsyncMock(return_value={"messageId": "msg1"})
        adapter._http_session.post = MagicMock(return_value=_AsyncCM(resp))

        await adapter.send("chat1", "one\n\ntwo\n\nthree\n\nfour")

        payloads = [call.kwargs["json"] for call in adapter._http_session.post.call_args_list]
        assert [payload["message"] for payload in payloads] == [
            "one",
            "two",
            "three\n\nfour",
        ]

    @pytest.mark.asyncio
    async def test_legacy_chunk_delay_remains_point_three_when_opted_out(self, monkeypatch):
        adapter = _make_adapter()
        adapter._split_outgoing_delay_seconds = 1.25
        resp = MagicMock(status=200)
        resp.json = AsyncMock(return_value={"messageId": "msg1"})
        adapter._http_session.post = MagicMock(return_value=_AsyncCM(resp))
        sleep = AsyncMock()
        monkeypatch.setattr(asyncio, "sleep", sleep)

        await adapter.send("chat1", "a " * 3000)

        assert adapter._http_session.post.call_count > 1
        sleep.assert_awaited()
        assert {call.args[0] for call in sleep.await_args_list} == {0.3}

    @pytest.mark.asyncio
    async def test_configured_split_delay_is_used_when_opted_in(self, monkeypatch):
        adapter = _make_adapter()
        adapter._split_outgoing_on_blank_lines = True
        adapter._split_outgoing_delay_seconds = 1.25
        resp = MagicMock(status=200)
        resp.json = AsyncMock(return_value={"messageId": "msg1"})
        adapter._http_session.post = MagicMock(return_value=_AsyncCM(resp))
        sleep = AsyncMock()
        monkeypatch.setattr(asyncio, "sleep", sleep)

        await adapter.send("chat1", "one\n\ntwo")

        sleep.assert_awaited()
        assert {call.args[0] for call in sleep.await_args_list} == {1.25}

    @pytest.mark.asyncio
    async def test_blank_lines_inside_fenced_code_stay_in_one_bubble(self):
        adapter = _make_adapter()
        adapter._split_outgoing_on_blank_lines = True
        resp = MagicMock(status=200)
        resp.json = AsyncMock(return_value={"messageId": "msg1"})
        adapter._http_session.post = MagicMock(return_value=_AsyncCM(resp))
        content = "intro\n\n```python\nfirst()\n\nsecond()\n```\n\noutro"

        await adapter.send("chat1", content)

        payloads = [call.kwargs["json"] for call in adapter._http_session.post.call_args_list]
        assert [payload["message"] for payload in payloads] == [
            "intro",
            "```python\nfirst()\n\nsecond()\n```",
            "outro",
        ]

    @pytest.mark.asyncio
    async def test_length_chunking_still_balances_fences_after_bubble_split(self):
        adapter = _make_adapter()
        adapter._split_outgoing_on_blank_lines = True
        resp = MagicMock(status=200)
        resp.json = AsyncMock(return_value={"messageId": "msg1"})
        adapter._http_session.post = MagicMock(return_value=_AsyncCM(resp))
        code = "```python\n" + ("print('hello')\n" * 400) + "\nprint('done')\n```"

        await adapter.send("chat1", f"intro\n\n{code}")

        messages = [
            call.kwargs["json"]["message"]
            for call in adapter._http_session.post.call_args_list
        ]
        assert messages[0] == "intro"
        assert len(messages) > 2
        assert all(message.startswith("```python\n") for message in messages[1:])
        assert all("```" in message.removeprefix("```python\n") for message in messages[1:])


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


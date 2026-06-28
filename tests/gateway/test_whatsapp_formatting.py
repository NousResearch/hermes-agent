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
    adapter._human_cascade_messages = True
    adapter._human_cascade_max_bubbles = 3
    adapter._human_cascade_delay_seconds = 0
    adapter._human_cascade_delay_jitter_seconds = 0
    adapter._human_cascade_typing_indicators = False
    adapter._human_cascade_max_total_chars = 900
    adapter._human_cascade_min_total_chars = 320
    adapter._human_cascade_min_lead_chars = 40
    adapter._human_cascade_max_bubble_chars = 320
    adapter._human_cascade_max_merged_bubble_chars = 640
    adapter._human_cascade_groups = False
    adapter._chunk_delay_seconds = 0
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

    def test_bold_header_does_not_double_wrap(self):
        """"# **Title**" must become *Title*, not **Title** (WhatsApp would
        render the doubled asterisks literally)."""
        adapter = _make_adapter()
        assert adapter.format_message("# **Title**") == "*Title*"
        assert adapter.format_message("## __Strong__") == "*Strong*"

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
        from plugins.platforms.whatsapp.adapter import WhatsAppAdapter
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
        assert result.raw_response["human_cascade"] is False

    @pytest.mark.asyncio
    async def test_short_blank_line_reply_stays_single(self):
        adapter = _make_adapter()
        resp = MagicMock(status=200)
        resp.json = AsyncMock(return_value={"messageId": "msg1"})
        adapter._http_session.post = MagicMock(return_value=_AsyncCM(resp))

        content = "lol yeah that queued notice is separate\n\nbut the actual reply cascaded correctly"
        result = await adapter.send("chat1", content)

        assert result.success
        assert adapter._http_session.post.call_count == 1
        payload = adapter._http_session.post.call_args.kwargs["json"]
        assert payload["message"] == content
        assert result.raw_response["human_cascade"] is False

    @pytest.mark.asyncio
    async def test_substantive_blank_line_paragraphs_send_as_human_cascade(self):
        adapter = _make_adapter()
        responses = []
        for msg_id in ("msg1", "msg2", "msg3"):
            resp = MagicMock(status=200)
            resp.json = AsyncMock(return_value={"messageId": msg_id})
            responses.append(_AsyncCM(resp))
        adapter._http_session.post = MagicMock(side_effect=responses)

        messages = [
            "First thought has enough actual content to deserve its own bubble, because it frames the answer before the heavier detail arrives and gives the reader a clean starting point.",
            "Second thought is also a real sentence with useful substance, so it reads like a natural follow-up instead of a random tiny ack split for no reason.",
            "Third thought closes the answer with enough context for the reader, proving the cascade is being used for cadence rather than just reacting to blank lines.",
        ]
        result = await adapter.send("chat1", "\n\n".join(messages))

        assert result.success
        assert adapter._http_session.post.call_count == 3
        payloads = [call.kwargs["json"] for call in adapter._http_session.post.call_args_list]
        assert [payload["message"] for payload in payloads] == messages
        assert result.message_id == "msg3"
        assert result.continuation_message_ids == ("msg1", "msg2")
        assert result.raw_response["human_cascade"] is True

    @pytest.mark.asyncio
    async def test_extra_paragraphs_merge_into_final_cascade_bubble(self):
        adapter = _make_adapter()
        responses = []
        for msg_id in ("msg1", "msg2", "msg3"):
            resp = MagicMock(status=200)
            resp.json = AsyncMock(return_value={"messageId": msg_id})
            responses.append(_AsyncCM(resp))
        adapter._http_session.post = MagicMock(side_effect=responses)

        messages = [
            "First paragraph is substantive enough to be worth a separate WhatsApp bubble.",
            "Second paragraph keeps the cadence readable before the merged tail arrives.",
            "Third paragraph is still short enough to fit as a clean lead-in bubble.",
            ("The fourth paragraph is deliberately longer and should be merged with the final paragraph instead of creating a fifth bubble. " * 3).strip(),
            ("The fifth paragraph continues the tail so the adapter proves it caps cascade count without collapsing everything into one wall. " * 3).strip(),
        ]
        result = await adapter.send("chat1", "\n\n".join(messages))

        assert result.success
        assert adapter._http_session.post.call_count == 3
        payloads = [call.kwargs["json"] for call in adapter._http_session.post.call_args_list]
        assert [payload["message"] for payload in payloads] == [
            messages[0],
            messages[1],
            messages[2] + "\n\n" + messages[3] + "\n\n" + messages[4],
        ]
        assert result.message_id == "msg3"
        assert result.continuation_message_ids == ("msg1", "msg2")
        assert result.raw_response["human_cascade"] is True

    @pytest.mark.asyncio
    async def test_human_cascade_sends_typing_indicator_between_bubbles(self):
        adapter = _make_adapter()
        adapter._human_cascade_typing_indicators = True
        responses = []
        for msg_id in ("msg1", None, "msg2"):
            resp = MagicMock(status=200)
            resp.json = AsyncMock(return_value={"success": True} if msg_id is None else {"messageId": msg_id})
            responses.append(_AsyncCM(resp))
        adapter._http_session.post = MagicMock(side_effect=responses)

        first = "First paragraph is substantial and useful enough to land immediately as a natural WhatsApp bubble, giving the reader a clean opening thought before the follow-up arrives."
        second = "Second paragraph is also substantial enough to follow after a typing presence event, making the delivery feel typed instead of sprayed into the chat, and it pushes the combined text over the mobile split threshold."
        result = await adapter.send("chat1", first + "\n\n" + second)

        assert result.success
        urls = [call.args[0] for call in adapter._http_session.post.call_args_list]
        assert urls[0].endswith("/send")
        assert urls[1].endswith("/typing")
        assert urls[2].endswith("/send")
        assert adapter._http_session.post.call_args_list[1].kwargs["json"] == {"chatId": "chat1"}
        assert result.raw_response["human_cascade"] is True

    @pytest.mark.asyncio
    async def test_human_cascade_can_be_disabled(self):
        adapter = _make_adapter()
        adapter._human_cascade_messages = False
        resp = MagicMock(status=200)
        resp.json = AsyncMock(return_value={"messageId": "msg1"})
        adapter._http_session.post = MagicMock(return_value=_AsyncCM(resp))

        await adapter.send("chat1", "first\n\nsecond")

        assert adapter._http_session.post.call_count == 1
        payload = adapter._http_session.post.call_args.kwargs["json"]
        assert payload["message"] == "first\n\nsecond"

    @pytest.mark.asyncio
    async def test_metadata_delivery_style_single_suppresses_human_cascade(self):
        adapter = _make_adapter()
        resp = MagicMock(status=200)
        resp.json = AsyncMock(return_value={"messageId": "msg1"})
        adapter._http_session.post = MagicMock(return_value=_AsyncCM(resp))

        await adapter.send("chat1", "thinking\n\nstill checking", metadata={"delivery_style": "single"})

        assert adapter._http_session.post.call_count == 1
        payload = adapter._http_session.post.call_args.kwargs["json"]
        assert payload["message"] == "thinking\n\nstill checking"

    @pytest.mark.asyncio
    async def test_metadata_delivery_style_cascade_can_force_structured_message(self):
        adapter = _make_adapter()
        responses = []
        for msg_id in ("msg1", "msg2", "msg3"):
            resp = MagicMock(status=200)
            resp.json = AsyncMock(return_value={"messageId": msg_id})
            responses.append(_AsyncCM(resp))
        adapter._http_session.post = MagicMock(side_effect=responses)

        await adapter.send(
            "chat1",
            "done\n\n- one\n- two\n- three\n\nnext",
            metadata={"delivery_style": "cascade"},
        )

        assert adapter._http_session.post.call_count == 3

    @pytest.mark.asyncio
    async def test_machine_artifacts_do_not_human_cascade(self):
        adapter = _make_adapter()
        resp = MagicMock(status=200)
        resp.json = AsyncMock(return_value={"messageId": "msg1"})
        adapter._http_session.post = MagicMock(return_value=_AsyncCM(resp))

        content = "Here is the config to verify as one unit.\n\napiVersion: v1\nkind: ConfigMap\nmetadata:\n  name: demo\n\nCopy this exactly into the config file."
        await adapter.send("chat1", content)

        assert adapter._http_session.post.call_count == 1
        payload = adapter._http_session.post.call_args.kwargs["json"]
        assert payload["message"] == content

    @pytest.mark.asyncio
    async def test_mixed_prose_and_json_artifact_splits_only_before_artifact_tail(self):
        adapter = _make_adapter()
        responses = []
        for msg_id in ("msg1", "msg2"):
            resp = MagicMock(status=200)
            resp.json = AsyncMock(return_value={"messageId": msg_id})
            responses.append(_AsyncCM(resp))
        adapter._http_session.post = MagicMock(side_effect=responses)

        lead = "This is the human-readable setup context, long enough to deserve its own bubble before the machine-readable payload arrives for verification, and it explains why the next blob is deliberately kept intact instead of chopped into mobile confetti that would be annoying to copy, audit, or compare against the file on disk."
        artifact = '{\n  "service": "gateway",\n  "mode": "safe",\n  "retries": 3\n}'
        await adapter.send("chat1", lead + "\n\n" + artifact)

        assert adapter._http_session.post.call_count == 2
        payloads = [call.kwargs["json"] for call in adapter._http_session.post.call_args_list]
        assert payloads[0]["message"] == lead
        assert payloads[1]["message"] == artifact

    @pytest.mark.asyncio
    async def test_code_blocks_do_not_human_cascade(self):
        adapter = _make_adapter()
        resp = MagicMock(status=200)
        resp.json = AsyncMock(return_value={"messageId": "msg1"})
        adapter._http_session.post = MagicMock(return_value=_AsyncCM(resp))

        content = "copy this\n\n```bash\necho hi\n```\n\nthen run it"
        await adapter.send("chat1", content)

        assert adapter._http_session.post.call_count == 1
        payload = adapter._http_session.post.call_args.kwargs["json"]
        assert "```bash\necho hi\n```" in payload["message"]

    @pytest.mark.asyncio
    async def test_approval_gates_do_not_human_cascade(self):
        adapter = _make_adapter()
        resp = MagicMock(status=200)
        resp.json = AsyncMock(return_value={"messageId": "msg1"})
        adapter._http_session.post = MagicMock(return_value=_AsyncCM(resp))

        content = "I can restart the gateway.\n\nReply `/approve` to restart.\n\nRisk: brief downtime."
        await adapter.send("chat1", content)

        assert adapter._http_session.post.call_count == 1
        payload = adapter._http_session.post.call_args.kwargs["json"]
        assert "Reply `/approve`" in payload["message"]

    @pytest.mark.asyncio
    async def test_substantive_structured_tail_cascades_plain_lead_in_only(self):
        adapter = _make_adapter()
        responses = []
        for msg_id in ("msg1", "msg2"):
            resp = MagicMock(status=200)
            resp.json = AsyncMock(return_value={"messageId": msg_id})
            responses.append(_AsyncCM(resp))
        adapter._http_session.post = MagicMock(side_effect=responses)

        lead = "Here’s the useful summary before the list, with enough context to deserve its own bubble."
        tail = "\n".join([
            "- changed adapter so short acknowledgements do not split into fake-human theatre bubbles",
            "- added tests for short replies, structured reports, substantive lead-ins, and long prose tails",
            "- kept approval prompts, code blocks, and group chats conservative so safety/copyability still wins",
            "- ran the focused gateway suite after the heuristic change to catch regressions",
        ])
        await adapter.send("chat1", lead + "\n\n" + tail)

        assert adapter._http_session.post.call_count == 2
        payloads = [call.kwargs["json"] for call in adapter._http_session.post.call_args_list]
        assert payloads[0]["message"] == lead
        assert payloads[1]["message"] == tail

    @pytest.mark.asyncio
    async def test_short_opener_before_structured_tail_stays_single(self):
        adapter = _make_adapter()
        resp = MagicMock(status=200)
        resp.json = AsyncMock(return_value={"messageId": "msg1"})
        adapter._http_session.post = MagicMock(return_value=_AsyncCM(resp))

        content = "yep found it\n\n- changed adapter\n- added tests\n- ran suite\n\n139 passed"
        await adapter.send("chat1", content)

        assert adapter._http_session.post.call_count == 1
        payload = adapter._http_session.post.call_args.kwargs["json"]
        assert payload["message"] == content

    @pytest.mark.asyncio
    async def test_report_heading_before_structured_tail_stays_single(self):
        adapter = _make_adapter()
        resp = MagicMock(status=200)
        resp.json = AsyncMock(return_value={"messageId": "msg1"})
        adapter._http_session.post = MagicMock(return_value=_AsyncCM(resp))

        content = "Summary\n\n- changed adapter\n- added tests\n- ran suite\n\nRisk\n\n- gateway restart needed"
        await adapter.send("chat1", content)

        assert adapter._http_session.post.call_count == 1
        payload = adapter._http_session.post.call_args.kwargs["json"]
        assert payload["message"] == content

    @pytest.mark.asyncio
    async def test_structured_report_without_plain_lead_in_stays_single(self):
        adapter = _make_adapter()
        resp = MagicMock(status=200)
        resp.json = AsyncMock(return_value={"messageId": "msg1"})
        adapter._http_session.post = MagicMock(return_value=_AsyncCM(resp))

        content = "- changed adapter\n- added tests\n- ran suite\n\n44 passed"
        await adapter.send("chat1", content)

        assert adapter._http_session.post.call_count == 1
        payload = adapter._http_session.post.call_args.kwargs["json"]
        assert payload["message"].startswith("- changed adapter")

    @pytest.mark.asyncio
    async def test_long_briefs_cascade_lead_in_then_chunk_tail(self):
        adapter = _make_adapter()
        responses = []
        for msg_id in ("msg1", "msg2"):
            resp = MagicMock(status=200)
            resp.json = AsyncMock(return_value={"messageId": msg_id})
            responses.append(_AsyncCM(resp))
        adapter._http_session.post = MagicMock(side_effect=responses)

        lead = "Here’s the daily brief with enough context to make the first bubble useful before the longer body lands."
        content = lead + "\n\n" + ("Lots of info here. " * 80)
        result = await adapter.send("chat1", content)

        assert result.success
        assert adapter._http_session.post.call_count == 2
        payloads = [call.kwargs["json"] for call in adapter._http_session.post.call_args_list]
        assert payloads[0]["message"] == lead
        assert payloads[1]["message"].startswith("Lots of info here.")
        assert result.raw_response["human_cascade"] is True

    @pytest.mark.asyncio
    async def test_active_question_before_tail_stays_single(self):
        adapter = _make_adapter()
        resp = MagicMock(status=200)
        resp.json = AsyncMock(return_value={"messageId": "msg1"})
        adapter._http_session.post = MagicMock(return_value=_AsyncCM(resp))

        content = "Do you want me to restart the gateway?\n\n" + (
            "These extra diagnostic details would bury the active question if they landed in a later bubble. " * 5
        ).strip()
        await adapter.send("chat1", content)

        assert adapter._http_session.post.call_count == 1
        payload = adapter._http_session.post.call_args.kwargs["json"]
        assert payload["message"] == content

    @pytest.mark.asyncio
    async def test_link_tail_stays_with_immediate_context(self):
        adapter = _make_adapter()
        responses = []
        for msg_id in ("msg1", "msg2"):
            resp = MagicMock(status=200)
            resp.json = AsyncMock(return_value={"messageId": msg_id})
            responses.append(_AsyncCM(resp))
        adapter._http_session.post = MagicMock(side_effect=responses)

        intro = "I found the relevant preview and this opening bubble is useful context before the actual action link arrives, rather than dropping a naked URL into the chat like a suspicious little phishing pellet."
        context = "This link is for the draft preview, so it needs to travel with the sentence explaining what it is and why the user would tap it."
        link = "Preview: https://example.com/draft/abc123"
        await adapter.send("chat1", "\n\n".join([intro, context, link]))

        assert adapter._http_session.post.call_count == 2
        payloads = [call.kwargs["json"] for call in adapter._http_session.post.call_args_list]
        assert payloads[0]["message"] == intro
        assert payloads[1]["message"] == context + "\n\n" + link

    @pytest.mark.asyncio
    async def test_group_chats_do_not_human_cascade_by_default(self):
        adapter = _make_adapter()
        resp = MagicMock(status=200)
        resp.json = AsyncMock(return_value={"messageId": "msg1"})
        adapter._http_session.post = MagicMock(return_value=_AsyncCM(resp))

        await adapter.send("120363001234567890@g.us", "first\n\nsecond")

        assert adapter._http_session.post.call_count == 1
        payload = adapter._http_session.post.call_args.kwargs["json"]
        assert payload["message"] == "first\n\nsecond"

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

    def test_whatsapp_tool_progress_is_new(self):
        from gateway.display_config import resolve_display_setting
        assert resolve_display_setting({}, "whatsapp", "tool_progress") == "new"

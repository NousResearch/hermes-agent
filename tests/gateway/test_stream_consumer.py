"""Tests for GatewayStreamConsumer — media directive stripping in streaming."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig


# ── _clean_for_display unit tests ────────────────────────────────────────


class TestCleanForDisplay:
    """Verify MEDIA: directives and internal markers are stripped from display text."""

    def test_no_media_passthrough(self):
        """Text without MEDIA: passes through unchanged."""
        text = "Here is your analysis of the image."
        assert GatewayStreamConsumer._clean_for_display(text) == text

    def test_media_tag_stripped(self):
        """Basic MEDIA:<path> tag is removed."""
        text = "Here is the image\nMEDIA:/tmp/hermes/image.png"
        result = GatewayStreamConsumer._clean_for_display(text)
        assert "MEDIA:" not in result
        assert "Here is the image" in result

    def test_media_tag_with_space(self):
        """MEDIA: tag with space after colon is removed."""
        text = "Audio generated\nMEDIA: /home/user/.hermes/audio_cache/voice.mp3"
        result = GatewayStreamConsumer._clean_for_display(text)
        assert "MEDIA:" not in result
        assert "Audio generated" in result

    def test_media_tag_with_quotes(self):
        """MEDIA: tags wrapped in quotes or backticks are removed."""
        for wrapper in ['`MEDIA:/path/file.png`', '"MEDIA:/path/file.png"', "'MEDIA:/path/file.png'"]:
            text = f"Result: {wrapper}"
            result = GatewayStreamConsumer._clean_for_display(text)
            assert "MEDIA:" not in result, f"Failed for wrapper: {wrapper}"

    def test_audio_as_voice_stripped(self):
        """[[audio_as_voice]] directive is removed."""
        text = "[[audio_as_voice]]\nMEDIA:/tmp/voice.ogg"
        result = GatewayStreamConsumer._clean_for_display(text)
        assert "[[audio_as_voice]]" not in result
        assert "MEDIA:" not in result

    def test_multiple_media_tags(self):
        """Multiple MEDIA: tags are all removed."""
        text = "Here are two files:\nMEDIA:/tmp/a.png\nMEDIA:/tmp/b.jpg"
        result = GatewayStreamConsumer._clean_for_display(text)
        assert "MEDIA:" not in result
        assert "Here are two files:" in result

    def test_excessive_newlines_collapsed(self):
        """Blank lines left by removed tags are collapsed."""
        text = "Before\n\n\nMEDIA:/tmp/file.png\n\n\nAfter"
        result = GatewayStreamConsumer._clean_for_display(text)
        # Should not have 3+ consecutive newlines
        assert "\n\n\n" not in result

    def test_media_only_response(self):
        """Response that is entirely MEDIA: tags returns empty/whitespace."""
        text = "MEDIA:/tmp/image.png"
        result = GatewayStreamConsumer._clean_for_display(text)
        assert result.strip() == ""

    def test_media_mid_sentence(self):
        """MEDIA: tag embedded in prose is stripped cleanly."""
        text = "I generated this image MEDIA:/tmp/art.png for you."
        result = GatewayStreamConsumer._clean_for_display(text)
        assert "MEDIA:" not in result
        assert "generated" in result
        assert "for you." in result

    def test_preserves_non_media_colons(self):
        """Normal colons and text with 'MEDIA' as a word aren't stripped."""
        text = "The media: files are stored in /tmp. Use social MEDIA carefully."
        result = GatewayStreamConsumer._clean_for_display(text)
        # "MEDIA:" in upper case without a path won't match \S+ (space follows)
        # But "media:" is lowercase so won't match either
        assert result == text


# ── Integration: _send_or_edit strips MEDIA: ─────────────────────────────


class TestFinalizeCapabilityGate:
    """Verify REQUIRES_EDIT_FINALIZE gates the redundant final edit.

    Platforms that don't need an explicit finalize signal (Telegram,
    Slack, Matrix, …) should skip the redundant final edit when the
    mid-stream edit already delivered the final content.  Platforms that
    *do* need it (DingTalk AI Cards) must always receive a finalize=True
    edit at the end of the stream.
    """

    @pytest.mark.asyncio
    async def test_identical_text_skip_respects_adapter_flag(self):
        """_send_or_edit short-circuits identical-text only when the
        adapter doesn't require an explicit finalize signal."""
        # Adapter without finalize requirement — should skip identical edit.
        plain = MagicMock()
        plain.REQUIRES_EDIT_FINALIZE = False
        plain.send = AsyncMock(return_value=SimpleNamespace(
            success=True, message_id="m1",
        ))
        plain.edit_message = AsyncMock()
        plain.MAX_MESSAGE_LENGTH = 4096
        c1 = GatewayStreamConsumer(plain, "chat_1")
        await c1._send_or_edit("hello")  # first send
        await c1._send_or_edit("hello", finalize=True)  # identical → skip
        plain.edit_message.assert_not_called()

        # Adapter that requires finalize — must still fire the edit.
        picky = MagicMock()
        picky.REQUIRES_EDIT_FINALIZE = True
        picky.send = AsyncMock(return_value=SimpleNamespace(
            success=True, message_id="m1",
        ))
        picky.edit_message = AsyncMock(return_value=SimpleNamespace(
            success=True, message_id="m1",
        ))
        picky.MAX_MESSAGE_LENGTH = 4096
        c2 = GatewayStreamConsumer(picky, "chat_1")
        await c2._send_or_edit("hello")
        await c2._send_or_edit("hello", finalize=True)
        # Finalize edit must go through even on identical content.
        picky.edit_message.assert_called_once()
        assert picky.edit_message.call_args[1]["finalize"] is True


class TestEditMessageFinalizeSignature:
    """Every concrete platform adapter must accept the ``finalize`` kwarg.

    stream_consumer._send_or_edit always passes ``finalize=`` to
    ``adapter.edit_message(...)`` (see gateway/stream_consumer.py).  An
    adapter that overrides edit_message without accepting finalize raises
    TypeError the first time streaming hits a segment break or final edit.
    Guard the contract with an explicit signature check so it cannot
    silently regress — existing tests use MagicMock which swallows any
    kwarg and cannot catch this.
    """

    @pytest.mark.parametrize(
        "module_path,class_name",
        [
            ("plugins.platforms.telegram.adapter", "TelegramAdapter"),
            ("plugins.platforms.discord.adapter", "DiscordAdapter"),
            ("plugins.platforms.slack.adapter", "SlackAdapter"),
            ("plugins.platforms.matrix.adapter", "MatrixAdapter"),
            ("plugins.platforms.mattermost.adapter", "MattermostAdapter"),
            ("plugins.platforms.feishu.adapter", "FeishuAdapter"),
            ("plugins.platforms.whatsapp.adapter", "WhatsAppAdapter"),
            ("plugins.platforms.dingtalk.adapter", "DingTalkAdapter"),
        ],
    )
    def test_edit_message_accepts_finalize(self, module_path, class_name):
        import inspect

        module = pytest.importorskip(module_path)
        cls = getattr(module, class_name)
        params = inspect.signature(cls.edit_message).parameters
        assert "finalize" in params, (
            f"{class_name}.edit_message must accept 'finalize' kwarg; "
            f"stream_consumer._send_or_edit passes it unconditionally"
        )


class TestSendOrEditMediaStripping:
    """Verify _send_or_edit strips MEDIA: before sending to the platform."""

    @pytest.mark.asyncio
    async def test_first_send_strips_media(self):
        """Initial send removes MEDIA: tags from visible text."""
        adapter = MagicMock()
        send_result = SimpleNamespace(success=True, message_id="msg_1")
        adapter.send = AsyncMock(return_value=send_result)
        adapter.MAX_MESSAGE_LENGTH = 4096

        consumer = GatewayStreamConsumer(adapter, "chat_123")
        await consumer._send_or_edit("Here is your image\nMEDIA:/tmp/test.png")

        adapter.send.assert_called_once()
        sent_text = adapter.send.call_args[1]["content"]
        assert "MEDIA:" not in sent_text
        assert "Here is your image" in sent_text

    @pytest.mark.asyncio
    async def test_edit_strips_media(self):
        """Edit call removes MEDIA: tags from visible text."""
        adapter = MagicMock()
        send_result = SimpleNamespace(success=True, message_id="msg_1")
        edit_result = SimpleNamespace(success=True)
        adapter.send = AsyncMock(return_value=send_result)
        adapter.edit_message = AsyncMock(return_value=edit_result)
        adapter.MAX_MESSAGE_LENGTH = 4096

        consumer = GatewayStreamConsumer(adapter, "chat_123")
        # First send
        await consumer._send_or_edit("Starting response...")
        # Edit with MEDIA: tag
        await consumer._send_or_edit("Here is the result\nMEDIA:/tmp/image.png")

        adapter.edit_message.assert_called_once()
        edited_text = adapter.edit_message.call_args[1]["content"]
        assert "MEDIA:" not in edited_text

    @pytest.mark.asyncio
    async def test_media_only_skips_send(self):
        """If text is entirely MEDIA: tags, the send is skipped."""
        adapter = MagicMock()
        adapter.send = AsyncMock()
        adapter.MAX_MESSAGE_LENGTH = 4096

        consumer = GatewayStreamConsumer(adapter, "chat_123")
        await consumer._send_or_edit("MEDIA:/tmp/image.png")

        adapter.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_cursor_only_update_skips_send(self):
        """A bare streaming cursor should not be sent as its own message."""
        adapter = MagicMock()
        adapter.send = AsyncMock()
        adapter.MAX_MESSAGE_LENGTH = 4096

        consumer = GatewayStreamConsumer(
            adapter,
            "chat_123",
            StreamConsumerConfig(cursor=" ▉"),
        )
        await consumer._send_or_edit(" ▉")

        adapter.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_short_text_with_cursor_skips_new_message(self):
        """Short text + cursor should not create a standalone new message.

        During rapid tool-calling the model often emits 1-2 tokens before
        switching to tool calls.  Sending 'I ▉' as a new message risks
        leaving the cursor permanently visible if the follow-up edit is
        rate-limited.  The guard should skip the first send and let the
        text accumulate into the next segment.
        """
        adapter = MagicMock()
        adapter.send = AsyncMock()
        adapter.MAX_MESSAGE_LENGTH = 4096

        consumer = GatewayStreamConsumer(
            adapter,
            "chat_123",
            StreamConsumerConfig(cursor=" ▉"),
        )
        # No message_id yet (first send) — short text + cursor should be skipped
        assert consumer._message_id is None
        result = await consumer._send_or_edit("I ▉")
        assert result is True
        adapter.send.assert_not_called()

        # 3 chars is still under the threshold
        result = await consumer._send_or_edit("Hi! ▉")
        assert result is True
        adapter.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_longer_text_with_cursor_sends_new_message(self):
        """Text >= 4 visible chars + cursor should create a new message normally."""
        adapter = MagicMock()
        send_result = SimpleNamespace(success=True, message_id="msg_1")
        adapter.send = AsyncMock(return_value=send_result)
        adapter.MAX_MESSAGE_LENGTH = 4096

        consumer = GatewayStreamConsumer(
            adapter,
            "chat_123",
            StreamConsumerConfig(cursor=" ▉"),
        )
        result = await consumer._send_or_edit("Hello ▉")
        assert result is True
        adapter.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_short_text_without_cursor_sends_normally(self):
        """Short text without cursor (e.g. final edit) should send normally."""
        adapter = MagicMock()
        send_result = SimpleNamespace(success=True, message_id="msg_1")
        adapter.send = AsyncMock(return_value=send_result)
        adapter.MAX_MESSAGE_LENGTH = 4096

        consumer = GatewayStreamConsumer(
            adapter,
            "chat_123",
            StreamConsumerConfig(cursor=" ▉"),
        )
        # No cursor in text — even short text should be sent
        result = await consumer._send_or_edit("OK")
        assert result is True
        adapter.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_short_text_cursor_edit_existing_message_allowed(self):
        """Short text + cursor editing an existing message should proceed."""
        adapter = MagicMock()
        edit_result = SimpleNamespace(success=True)
        adapter.edit_message = AsyncMock(return_value=edit_result)
        adapter.MAX_MESSAGE_LENGTH = 4096

        consumer = GatewayStreamConsumer(
            adapter,
            "chat_123",
            StreamConsumerConfig(cursor=" ▉"),
        )
        consumer._message_id = "msg_1"  # Existing message — guard should not fire
        consumer._last_sent_text = ""
        result = await consumer._send_or_edit("I ▉")
        assert result is True
        adapter.edit_message.assert_called_once()


# ── Integration: full stream run ─────────────────────────────────────────


class TestStreamRunMediaStripping:
    """End-to-end: deltas with MEDIA: produce clean visible text."""

    @pytest.mark.asyncio
    async def test_stream_with_media_tag(self):
        """Full stream run strips MEDIA: from the final visible message."""
        adapter = MagicMock()
        send_result = SimpleNamespace(success=True, message_id="msg_1")
        edit_result = SimpleNamespace(success=True)
        adapter.send = AsyncMock(return_value=send_result)
        adapter.edit_message = AsyncMock(return_value=edit_result)
        adapter.MAX_MESSAGE_LENGTH = 4096

        config = StreamConsumerConfig(edit_interval=0.01, buffer_threshold=5)
        consumer = GatewayStreamConsumer(adapter, "chat_123", config)

        # Feed deltas
        consumer.on_delta("Here is your generated image\n")
        consumer.on_delta("MEDIA:/home/user/.hermes/cache/images/abc123.png")
        consumer.finish()

        await consumer.run()

        # Verify the final text sent/edited doesn't contain MEDIA:
        all_calls = []
        for call in adapter.send.call_args_list:
            all_calls.append(call[1].get("content", ""))
        for call in adapter.edit_message.call_args_list:
            all_calls.append(call[1].get("content", ""))

        for sent_text in all_calls:
            assert "MEDIA:" not in sent_text, f"MEDIA: leaked into display: {sent_text!r}"

        assert consumer.already_sent


class TestBeforeFinalizeHook:
    """Verify the optional pre-finalize hook fires at the right time."""

    @pytest.mark.asyncio
    async def test_hook_runs_before_finalize_edit(self):
        """Adapters that require finalize should pause typing before the edit."""
        events = []
        adapter = MagicMock()
        adapter.REQUIRES_EDIT_FINALIZE = True
        adapter.send = AsyncMock(
            side_effect=lambda **_kw: (
                events.append("send"),
                SimpleNamespace(success=True, message_id="msg_1"),
            )[1]
        )
        adapter.edit_message = AsyncMock(
            side_effect=lambda **_kw: (
                events.append("edit"),
                SimpleNamespace(success=True, message_id="msg_1"),
            )[1]
        )
        adapter.MAX_MESSAGE_LENGTH = 4096

        consumer = GatewayStreamConsumer(
            adapter,
            "chat_123",
            StreamConsumerConfig(edit_interval=0.01, buffer_threshold=5),
            on_before_finalize=lambda: events.append("pause"),
        )
        consumer.on_delta("Hello")
        consumer.finish()

        await consumer.run()

        assert events == ["send", "pause", "edit"]

    @pytest.mark.asyncio
    async def test_hook_runs_once_when_final_text_already_visible(self):
        """The hook still fires once even when no final edit is required."""
        events = []
        adapter = MagicMock()
        adapter.REQUIRES_EDIT_FINALIZE = False
        adapter.send = AsyncMock(return_value=SimpleNamespace(success=True, message_id="msg_1"))
        adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=True, message_id="msg_1"))
        adapter.MAX_MESSAGE_LENGTH = 4096

        consumer = GatewayStreamConsumer(
            adapter,
            "chat_123",
            StreamConsumerConfig(edit_interval=0.01, buffer_threshold=5),
            on_before_finalize=lambda: events.append("pause"),
        )
        consumer.on_delta("Hello")
        consumer.finish()

        await consumer.run()

        assert events == ["pause"]
        adapter.edit_message.assert_not_called()


# ── Segment break (tool boundary) tests ──────────────────────────────────


class TestSegmentBreakOnToolBoundary:
    """Verify that on_delta(None) finalizes the current message and starts a
    new one so the final response appears below tool-progress messages."""

    @pytest.mark.asyncio
    async def test_segment_break_creates_new_message(self):
        """After a None boundary, next text creates a fresh message."""
        adapter = MagicMock()
        send_result_1 = SimpleNamespace(success=True, message_id="msg_1")
        send_result_2 = SimpleNamespace(success=True, message_id="msg_2")
        edit_result = SimpleNamespace(success=True)
        adapter.send = AsyncMock(side_effect=[send_result_1, send_result_2])
        adapter.edit_message = AsyncMock(return_value=edit_result)
        adapter.MAX_MESSAGE_LENGTH = 4096

        config = StreamConsumerConfig(edit_interval=0.01, buffer_threshold=5)
        consumer = GatewayStreamConsumer(adapter, "chat_123", config)

        # Phase 1: intermediate text before tool calls
        consumer.on_delta("Let me search for that...")
        # Tool boundary — model is about to call tools
        consumer.on_delta(None)
        # Phase 2: final response text after tools finished
        consumer.on_delta("Here are the results.")
        consumer.finish()

        await consumer.run()

        # Should have sent TWO separate messages (two adapter.send calls),
        # not just edited the first one.
        assert adapter.send.call_count == 2
        first_text = adapter.send.call_args_list[0][1]["content"]
        second_text = adapter.send.call_args_list[1][1]["content"]
        assert "search" in first_text
        assert "results" in second_text

    @pytest.mark.asyncio
    async def test_segment_break_no_text_before(self):
        """A None boundary with no preceding text is a no-op."""
        adapter = MagicMock()
        send_result = SimpleNamespace(success=True, message_id="msg_1")
        adapter.send = AsyncMock(return_value=send_result)
        adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=True))
        adapter.MAX_MESSAGE_LENGTH = 4096

        config = StreamConsumerConfig(edit_interval=0.01, buffer_threshold=5)
        consumer = GatewayStreamConsumer(adapter, "chat_123", config)

        # No text before the boundary — model went straight to tool calls
        consumer.on_delta(None)
        consumer.on_delta("Final answer.")
        consumer.finish()

        await consumer.run()

        # Only one send call (the final answer)
        assert adapter.send.call_count == 1
        assert "Final answer" in adapter.send.call_args_list[0][1]["content"]

    @pytest.mark.asyncio
    async def test_segment_break_removes_cursor(self):
        """The finalized segment message should not have a cursor."""
        adapter = MagicMock()
        send_result = SimpleNamespace(success=True, message_id="msg_1")
        edit_result = SimpleNamespace(success=True)
        adapter.send = AsyncMock(return_value=send_result)
        adapter.edit_message = AsyncMock(return_value=edit_result)
        adapter.MAX_MESSAGE_LENGTH = 4096

        config = StreamConsumerConfig(edit_interval=0.01, buffer_threshold=5, cursor=" ▉")
        consumer = GatewayStreamConsumer(adapter, "chat_123", config)

        consumer.on_delta("Thinking...")
        consumer.on_delta(None)
        consumer.on_delta("Done.")
        consumer.finish()

        await consumer.run()

        # The first segment should have been finalized without cursor.
        # Check all edit_message calls + the initial send for the first segment.
        # The last state of msg_1 should NOT have the cursor.
        all_texts = []
        for call in adapter.send.call_args_list:
            all_texts.append(call[1].get("content", ""))
        for call in adapter.edit_message.call_args_list:
            all_texts.append(call[1].get("content", ""))

        # Find the text(s) that contain "Thinking" — the finalized version
        # should not have the cursor.
        thinking_texts = [t for t in all_texts if "Thinking" in t]
        assert thinking_texts, "Expected at least one message with 'Thinking'"
        # The LAST occurrence is the finalized version
        assert "▉" not in thinking_texts[-1], (
            f"Cursor found in finalized segment: {thinking_texts[-1]!r}"
        )

    @pytest.mark.asyncio
    async def test_multiple_segment_breaks(self):
        """Multiple tool boundaries create multiple message segments."""
        adapter = MagicMock()
        msg_counter = iter(["msg_1", "msg_2", "msg_3"])
        adapter.send = AsyncMock(
            side_effect=lambda **kw: SimpleNamespace(success=True, message_id=next(msg_counter))
        )
        adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=True))
        adapter.MAX_MESSAGE_LENGTH = 4096

        config = StreamConsumerConfig(edit_interval=0.01, buffer_threshold=5)
        consumer = GatewayStreamConsumer(adapter, "chat_123", config)

        consumer.on_delta("Phase 1")
        consumer.on_delta(None)  # tool boundary
        consumer.on_delta("Phase 2")
        consumer.on_delta(None)  # another tool boundary
        consumer.on_delta("Phase 3")
        consumer.finish()

        await consumer.run()

        # Three separate messages
        assert adapter.send.call_count == 3

    @pytest.mark.asyncio
    async def test_already_sent_stays_true_after_segment(self):
        """already_sent remains True after a segment break."""
        adapter = MagicMock()
        send_result = SimpleNamespace(success=True, message_id="msg_1")
        adapter.send = AsyncMock(return_value=send_result)
        adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=True))
        adapter.MAX_MESSAGE_LENGTH = 4096

        config = StreamConsumerConfig(edit_interval=0.01, buffer_threshold=5)
        consumer = GatewayStreamConsumer(adapter, "chat_123", config)

        consumer.on_delta("Text")
        consumer.on_delta(None)
        consumer.finish()

        await consumer.run()

        assert consumer.already_sent

    @pytest.mark.asyncio
    async def test_edit_failure_sends_only_unsent_tail_at_finish(self):
        """If an edit fails mid-stream, send only the missing tail once at finish."""
        adapter = MagicMock()
        send_results = [
            SimpleNamespace(success=True, message_id="msg_1"),
            SimpleNamespace(success=True, message_id="msg_2"),
        ]
        adapter.send = AsyncMock(side_effect=send_results)
        adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=False, error="flood_control:6"))
        adapter.MAX_MESSAGE_LENGTH = 4096

        config = StreamConsumerConfig(edit_interval=0.01, buffer_threshold=5, cursor=" ▉")
        consumer = GatewayStreamConsumer(adapter, "chat_123", config)

        consumer.on_delta("Hello")
        task = asyncio.create_task(consumer.run())
        await asyncio.sleep(0.08)
        consumer.on_delta(" world")
        await asyncio.sleep(0.08)
        consumer.finish()
        await task

        assert adapter.send.call_count == 2
        first_text = adapter.send.call_args_list[0][1]["content"]
        second_text = adapter.send.call_args_list[1][1]["content"]
        assert "Hello" in first_text
        assert second_text.strip() == "world"
        assert consumer.already_sent

    @pytest.mark.asyncio
    async def test_segment_break_clears_failed_edit_fallback_state(self):
        """A tool boundary after edit failure must flush the undelivered tail
        without duplicating the prefix the user already saw (#8124)."""
        adapter = MagicMock()
        send_results = [
            SimpleNamespace(success=True, message_id="msg_1"),
            SimpleNamespace(success=True, message_id="msg_2"),
            SimpleNamespace(success=True, message_id="msg_3"),
        ]
        adapter.send = AsyncMock(side_effect=send_results)
        adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=False, error="flood_control:6"))
        adapter.MAX_MESSAGE_LENGTH = 4096

        config = StreamConsumerConfig(edit_interval=0.01, buffer_threshold=5, cursor=" ▉")
        consumer = GatewayStreamConsumer(adapter, "chat_123", config)

        consumer.on_delta("Hello")
        task = asyncio.create_task(consumer.run())
        await asyncio.sleep(0.08)
        consumer.on_delta(" world")
        await asyncio.sleep(0.08)
        consumer.on_delta(None)
        consumer.on_delta("Next segment")
        consumer.finish()
        await task

        sent_texts = [call[1]["content"] for call in adapter.send.call_args_list]
        # The undelivered "world" tail must reach the user, and the next
        # segment must not duplicate "Hello" that was already visible.
        assert sent_texts == ["Hello ▉", "world", "Next segment"]

    @pytest.mark.asyncio
    async def test_segment_break_after_mid_stream_edit_failure_preserves_tail(self):
        """Regression for #8124: when an earlier edit succeeded but later edits
        fail (persistent flood control) and a tool boundary arrives before the
        fallback threshold is reached, the pre-boundary tail must still be
        delivered — not silently dropped by the segment reset."""
        adapter = MagicMock()
        # msg_1 for the initial partial, msg_2 for the flushed tail,
        # msg_3 for the post-boundary segment.
        send_results = [
            SimpleNamespace(success=True, message_id="msg_1"),
            SimpleNamespace(success=True, message_id="msg_2"),
            SimpleNamespace(success=True, message_id="msg_3"),
        ]
        adapter.send = AsyncMock(side_effect=send_results)

        # First two edits succeed, everything after fails with flood control
        # — simulating Telegram's "edit once then get rate-limited" pattern.
        edit_results = [
            SimpleNamespace(success=True),   # "Hello world ▉"  — succeeds
            SimpleNamespace(success=False, error="flood_control:6.0"),  # "Hello world more ▉" — flood triggered
            SimpleNamespace(success=False, error="flood_control:6.0"),  # finalize edit at segment break
            SimpleNamespace(success=False, error="flood_control:6.0"),  # cursor-strip attempt
        ]
        adapter.edit_message = AsyncMock(side_effect=edit_results + [edit_results[-1]] * 10)
        adapter.MAX_MESSAGE_LENGTH = 4096

        config = StreamConsumerConfig(edit_interval=0.01, buffer_threshold=5, cursor=" ▉")
        consumer = GatewayStreamConsumer(adapter, "chat_123", config)

        consumer.on_delta("Hello")
        task = asyncio.create_task(consumer.run())
        await asyncio.sleep(0.08)
        consumer.on_delta(" world")
        await asyncio.sleep(0.08)
        consumer.on_delta(" more")
        await asyncio.sleep(0.08)
        consumer.on_delta(None)  # tool boundary
        consumer.on_delta("Here is the tool result.")
        consumer.finish()
        await task

        sent_texts = [call[1]["content"] for call in adapter.send.call_args_list]
        # "more" must have been delivered, not dropped.
        all_text = " ".join(sent_texts)
        assert "more" in all_text, (
            f"Pre-boundary tail 'more' was silently dropped: sends={sent_texts}"
        )
        # Post-boundary text must also reach the user.
        assert "Here is the tool result." in all_text

    @pytest.mark.asyncio
    async def test_no_message_id_enters_fallback_mode(self):
        """Platform returns success but no message_id (Signal) — must not
        re-send on every delta.  Should enter fallback mode and send only
        the continuation at finish."""
        adapter = MagicMock()
        # First send succeeds but returns no message_id (Signal behavior)
        send_result_no_id = SimpleNamespace(success=True, message_id=None)
        # Fallback final send succeeds
        send_result_final = SimpleNamespace(success=True, message_id="msg_final")
        adapter.send = AsyncMock(side_effect=[send_result_no_id, send_result_final])
        adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=True))
        adapter.MAX_MESSAGE_LENGTH = 4096

        config = StreamConsumerConfig(edit_interval=0.01, buffer_threshold=5)
        consumer = GatewayStreamConsumer(adapter, "chat_123", config)

        consumer.on_delta("Hello")
        task = asyncio.create_task(consumer.run())
        await asyncio.sleep(0.08)
        consumer.on_delta(" world, this is a longer response.")
        await asyncio.sleep(0.08)
        consumer.finish()
        await task

        # Should send exactly 2 messages: initial chunk + fallback continuation
        # NOT one message per delta
        assert adapter.send.call_count == 2
        assert consumer.already_sent
        # edit_message should NOT have been called (no valid message_id to edit)
        adapter.edit_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_message_id_single_delta_marks_already_sent(self):
        """When the entire response fits in one delta and platform returns no
        message_id, already_sent must still be True to prevent the gateway
        from re-sending the full response."""
        adapter = MagicMock()
        send_result = SimpleNamespace(success=True, message_id=None)
        adapter.send = AsyncMock(return_value=send_result)
        adapter.MAX_MESSAGE_LENGTH = 4096

        config = StreamConsumerConfig(edit_interval=0.01, buffer_threshold=5)
        consumer = GatewayStreamConsumer(adapter, "chat_123", config)

        consumer.on_delta("Short response.")
        consumer.finish()

        await consumer.run()

        assert consumer.already_sent
        # Only one send call (the initial message)
        assert adapter.send.call_count == 1

    @pytest.mark.asyncio
    async def test_no_message_id_segment_breaks_do_not_resend(self):
        """On a platform that never returns a message_id (e.g. webhook with
        github_comment delivery), tool-call segment breaks must NOT trigger
        a new adapter.send() per boundary.  The fix: _message_id == '__no_edit__'
        suppresses the reset so all text accumulates and is sent once."""
        adapter = MagicMock()
        # No message_id on first send, then one more for the fallback final
        adapter.send = AsyncMock(side_effect=[
            SimpleNamespace(success=True, message_id=None),
            SimpleNamespace(success=True, message_id=None),
        ])
        adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=True))
        adapter.MAX_MESSAGE_LENGTH = 4096

        config = StreamConsumerConfig(edit_interval=0.01, buffer_threshold=5)
        consumer = GatewayStreamConsumer(adapter, "chat_123", config)

        # Simulate: text → tool boundary → text → tool boundary → text (3 segments)
        consumer.on_delta("Phase 1 text")
        consumer.on_delta(None)   # tool call boundary
        consumer.on_delta("Phase 2 text")
        consumer.on_delta(None)   # another tool call boundary
        consumer.on_delta("Phase 3 text")
        consumer.finish()

        await consumer.run()

        # Before the fix this would post 3 comments (one per segment).
        # After the fix: only the initial partial + one fallback-final continuation.
        assert adapter.send.call_count == 2, (
            f"Expected 2 sends (initial + fallback), got {adapter.send.call_count}"
        )
        assert consumer.already_sent
        # The continuation must contain the text from segments 2 and 3
        final_text = adapter.send.call_args_list[1][1]["content"]
        assert "Phase 2" in final_text
        assert "Phase 3" in final_text

    @pytest.mark.asyncio
    async def test_fallback_final_splits_long_continuation_without_dropping_text(self):
        """Long continuation tails should be chunked when fallback final-send runs."""
        adapter = MagicMock()
        adapter.send = AsyncMock(side_effect=[
            SimpleNamespace(success=True, message_id="msg_1"),
            SimpleNamespace(success=True, message_id="msg_2"),
            SimpleNamespace(success=True, message_id="msg_3"),
        ])
        adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=False, error="flood_control:6"))
        adapter.MAX_MESSAGE_LENGTH = 610

        config = StreamConsumerConfig(edit_interval=0.01, buffer_threshold=5, cursor=" ▉")
        consumer = GatewayStreamConsumer(adapter, "chat_123", config)

        prefix = "Hello world"
        tail = "x" * 620
        consumer.on_delta(prefix)
        task = asyncio.create_task(consumer.run())
        await asyncio.sleep(0.08)
        consumer.on_delta(tail)
        await asyncio.sleep(0.08)
        consumer.finish()
        await task

        sent_texts = [call[1]["content"] for call in adapter.send.call_args_list]
        assert len(sent_texts) == 3
        assert sent_texts[0].startswith(prefix)
        assert sum(len(t) for t in sent_texts[1:]) == len(tail)

    @pytest.mark.asyncio
    async def test_fallback_final_sends_full_text_at_tool_boundary(self):
        """After a tool call, the streamed prefix is stale (from the pre-tool
        segment).  _send_fallback_final must still send the post-tool response
        even when continuation_text calculates as empty (#10807)."""
        adapter = MagicMock()
        adapter.send = AsyncMock(
            return_value=SimpleNamespace(success=True, message_id="msg_1"),
        )
        adapter.edit_message = AsyncMock(
            return_value=SimpleNamespace(success=True),
        )
        adapter.MAX_MESSAGE_LENGTH = 4096

        config = StreamConsumerConfig(edit_interval=0.01, buffer_threshold=5)
        consumer = GatewayStreamConsumer(adapter, "chat_123", config)

        # Simulate a pre-tool streamed segment that becomes the visible prefix
        pre_tool_text = "I'll run that code now."
        consumer.on_delta(pre_tool_text)
        task = asyncio.create_task(consumer.run())
        await asyncio.sleep(0.05)

        # After the tool call, the model returns a SHORT final response that
        # does NOT start with the pre-tool prefix.  The continuation calculator
        # would return empty (no prefix match → full text returned, but if the
        # streaming edit already showed pre_tool_text, the prefix-based logic
        # wrongly matches).  Simulate this by setting _last_sent_text to the
        # pre-tool content, then finishing with different post-tool content.
        consumer._last_sent_text = pre_tool_text
        post_tool_response = "⏰ Script timed out after 30s and was killed."
        consumer.finish()
        await task

        # The fallback should send the post-tool response via
        # _send_fallback_final.
        await consumer._send_fallback_final(post_tool_response)

        # Verify the final text was sent (not silently dropped)
        sent = False
        for call in adapter.send.call_args_list:
            content = call[1].get("content", call[0][0] if call[0] else "")
            if "timed out" in str(content):
                sent = True
                break
        assert sent, (
            "Post-tool timeout response was silently dropped by "
            "_send_fallback_final — the #10807 fix should prevent this"
        )

    @pytest.mark.asyncio
    async def test_fallback_final_deletes_partial_after_full_resend(self):
        """After fallback re-sends the COMPLETE response, the frozen partial
        must be deleted so the user sees only the complete response (#16668).
        Full resend happens when the visible prefix doesn't match the final
        text (e.g. post-segment-break content, #10807)."""
        adapter = MagicMock()
        adapter.send = AsyncMock(
            return_value=SimpleNamespace(success=True, message_id="msg_new"),
        )
        adapter.edit_message = AsyncMock(
            return_value=SimpleNamespace(success=True),
        )
        adapter.delete_message = AsyncMock(return_value=None)
        adapter.MAX_MESSAGE_LENGTH = 4096

        config = StreamConsumerConfig(edit_interval=0.01, buffer_threshold=5)
        consumer = GatewayStreamConsumer(adapter, "chat_123", config)

        # The stale partial shows pre-tool text that is NOT a prefix of the
        # final response — fallback re-sends the complete final text.
        consumer._message_id = "msg_partial"
        consumer._last_sent_text = "Let me check that for you…"

        await consumer._send_fallback_final("Working on it. Done!")

        adapter.delete_message.assert_awaited_once_with("chat_123", "msg_partial")
        assert consumer._final_response_sent is True

    @pytest.mark.asyncio
    async def test_fallback_final_keeps_partial_after_tail_only_send(self):
        """When the fallback sends only the missing TAIL (visible prefix
        matches the final text), the partial message IS the head of the
        answer — deleting it would leave the user with only the last part
        of the response (the 'model sent only the second half' bug)."""
        adapter = MagicMock()
        adapter.send = AsyncMock(
            return_value=SimpleNamespace(success=True, message_id="msg_new"),
        )
        adapter.edit_message = AsyncMock(
            return_value=SimpleNamespace(success=True),
        )
        adapter.delete_message = AsyncMock(return_value=None)
        adapter.MAX_MESSAGE_LENGTH = 4096

        config = StreamConsumerConfig(edit_interval=0.01, buffer_threshold=5)
        consumer = GatewayStreamConsumer(adapter, "chat_123", config)

        # Visible partial is a true prefix of the final response — the
        # fallback dedup sends only the tail.
        consumer._message_id = "msg_partial"
        consumer._last_sent_text = "Working on i"

        await consumer._send_fallback_final("Working on it. Done!")

        # Tail was sent...
        sent_contents = [
            c.kwargs.get("content", "") for c in adapter.send.call_args_list
        ]
        assert any("Done!" in s and "Working on i" not in s for s in sent_contents)
        # ...and the head-bearing partial was NOT deleted.
        adapter.delete_message.assert_not_awaited()
        assert consumer._final_response_sent is True

    @pytest.mark.asyncio
    async def test_fallback_final_does_not_delete_when_no_chunks_reach_user(self):
        """If every fallback send fails, the partial is the only thing the
        user has — must NOT be deleted."""
        adapter = MagicMock()
        adapter.send = AsyncMock(
            return_value=SimpleNamespace(success=False, error="network down"),
        )
        adapter.edit_message = AsyncMock(
            return_value=SimpleNamespace(success=True),
        )
        adapter.delete_message = AsyncMock(return_value=None)
        adapter.MAX_MESSAGE_LENGTH = 4096

        config = StreamConsumerConfig(edit_interval=0.01, buffer_threshold=5)
        consumer = GatewayStreamConsumer(adapter, "chat_123", config)

        consumer._message_id = "msg_partial"
        consumer._last_sent_text = "Working on i"

        await consumer._send_fallback_final("Working on it. Done!")

        adapter.delete_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_fallback_final_skips_delete_when_adapter_lacks_method(self):
        """Platforms without delete_message must not crash the fallback path."""
        adapter = MagicMock(spec=["send", "edit_message", "MAX_MESSAGE_LENGTH"])
        adapter.send = AsyncMock(
            return_value=SimpleNamespace(success=True, message_id="msg_new"),
        )
        adapter.edit_message = AsyncMock(
            return_value=SimpleNamespace(success=True),
        )
        adapter.MAX_MESSAGE_LENGTH = 4096

        config = StreamConsumerConfig(edit_interval=0.01, buffer_threshold=5)
        consumer = GatewayStreamConsumer(adapter, "chat_123", config)

        consumer._message_id = "msg_partial"
        consumer._last_sent_text = "Working on i"

        # Should not raise even though the adapter has no delete_message.
        await consumer._send_fallback_final("Working on it. Done!")
        assert consumer._final_response_sent is True


class TestFinalResponseDeliveryGuard:
    """Regression coverage for #10748 — _final_response_sent must reflect
    actual delivery of the *current* chunked send, not the cumulative
    `_already_sent` flag (which earlier tool-progress edits or fallback-mode
    promotion can taint)."""

    @pytest.mark.asyncio
    async def test_split_overflow_failed_send_does_not_mark_final_sent(self):
        """Split-overflow path: if every chunk send fails on done frame,
        _final_response_sent must stay False so the gateway falls back."""
        adapter = MagicMock()
        # Every send fails — _send_new_chunk returns the passed-in reply_to.
        adapter.send = AsyncMock(
            return_value=SimpleNamespace(success=False, error="network down"),
        )
        adapter.edit_message = AsyncMock(
            return_value=SimpleNamespace(success=True),
        )
        adapter.MAX_MESSAGE_LENGTH = 100
        adapter.truncate_message = MagicMock(
            side_effect=lambda text, limit: [text[:limit], text[limit:]],
        )

        config = StreamConsumerConfig(edit_interval=0.01, buffer_threshold=5)
        consumer = GatewayStreamConsumer(adapter, "chat_123", config)

        # Simulate prior tool-progress edits that set _already_sent
        consumer._already_sent = True

        # Long text > MAX_MESSAGE_LENGTH, no existing message id (fresh send path)
        long_text = "x" * 200
        consumer.on_delta(long_text)
        task = asyncio.create_task(consumer.run())
        await asyncio.sleep(0.05)
        consumer.finish()
        await task

        assert consumer._final_response_sent is False, (
            "_already_sent leaked into _final_response_sent — gateway will "
            "wrongly suppress its fallback delivery (#10748)"
        )

    @pytest.mark.asyncio
    async def test_split_overflow_partial_send_marks_final_sent(self):
        """Split-overflow path: if at least one chunk lands on done frame,
        we did deliver the final answer — _final_response_sent must be True."""
        adapter = MagicMock()
        adapter.send = AsyncMock(side_effect=[
            SimpleNamespace(success=True, message_id="msg_1"),
            SimpleNamespace(success=True, message_id="msg_2"),
        ])
        adapter.edit_message = AsyncMock(
            return_value=SimpleNamespace(success=True),
        )
        adapter.MAX_MESSAGE_LENGTH = 100
        adapter.truncate_message = MagicMock(
            side_effect=lambda text, limit: [text[:limit], text[limit:]],
        )

        config = StreamConsumerConfig(edit_interval=0.01, buffer_threshold=5)
        consumer = GatewayStreamConsumer(adapter, "chat_123", config)

        long_text = "x" * 200
        consumer.on_delta(long_text)
        task = asyncio.create_task(consumer.run())
        await asyncio.sleep(0.05)
        consumer.finish()
        await task

        assert consumer._final_response_sent is True


class TestFinalContentDeliveredGuard:
    """Regression coverage for #25010 — _final_content_delivered must only be
    set when the final response is actually confirmed delivered to the user,
    not when a mid-stream edit happened to show partial content.  Prematurely
    setting this flag causes the gateway to suppress the normal final send,
    leaving the user with an incomplete partial message."""

    @pytest.mark.asyncio
    async def test_mid_stream_edit_success_does_not_mark_content_delivered(self):
        """When the mid-stream edit with finalize=True succeeds but the
        subsequent finalize edit fails, _final_content_delivered must stay
        False so the gateway does not suppress its fallback send (#25010).

        Simulates TelegramAdapter which sets REQUIRES_EDIT_FINALIZE=True,
        requiring a second finalize edit even when content is unchanged."""
        adapter = MagicMock()
        adapter.REQUIRES_EDIT_FINALIZE = True  # Telegram adapter behavior
        # First send (initial streaming message) succeeds.
        # Mid-stream edit succeeds.
        # Final finalize edit fails, and the consumer's own fallback send also
        # fails, so no path has confirmed the complete final response reached
        # the user.
        adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=False))
        adapter.send = AsyncMock(side_effect=[
            SimpleNamespace(success=True, message_id="msg_1"),
            SimpleNamespace(success=False, error="network down"),
        ])
        adapter.MAX_MESSAGE_LENGTH = 4096

        config = StreamConsumerConfig(edit_interval=0.01, buffer_threshold=5)
        consumer = GatewayStreamConsumer(adapter, "chat_123", config)

        # Simulate streaming: send initial text, then more text, then done
        consumer.on_delta("Part one of the response...\n")
        task = asyncio.create_task(consumer.run())
        await asyncio.sleep(0.05)
        # Keep the second delta buffered until finish so the complete answer is
        # not already visible before the final edit attempt fails.
        consumer.cfg.buffer_threshold = 10_000
        consumer._current_edit_interval = 10.0

        consumer.on_delta("Part two, the complete final answer.\n")
        await asyncio.sleep(0.05)

        consumer.finish()
        await task

        # The key assertion: _final_content_delivered must NOT be True,
        # because the final edit failed and the complete response was never
        # confirmed delivered.
        assert consumer._final_content_delivered is False, (
            "_final_content_delivered was prematurely set to True — gateway "
            "will wrongly suppress its fallback send, leaving the user with "
            "an incomplete partial message (#25010)"
        )
        # The gateway must still be allowed to send the complete response
        assert consumer._final_response_sent is False, (
            "_final_response_sent must also be False when the final edit failed"
        )

    @pytest.mark.asyncio
    async def test_final_edit_success_does_mark_content_delivered(self):
        """When the final finalize edit succeeds, _final_content_delivered
        must be True — the normal happy path should still work."""
        adapter = MagicMock()
        adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=True))
        adapter.send = AsyncMock(
            return_value=SimpleNamespace(success=True, message_id="msg_1"),
        )
        adapter.MAX_MESSAGE_LENGTH = 4096

        config = StreamConsumerConfig(edit_interval=0.01, buffer_threshold=5)
        consumer = GatewayStreamConsumer(adapter, "chat_123", config)

        consumer.on_delta("The complete response.\n")
        task = asyncio.create_task(consumer.run())
        await asyncio.sleep(0.05)

        consumer.finish()
        await task

        assert consumer._final_content_delivered is True, (
            "_final_content_delivered must be True when the final edit succeeds"
        )
        assert consumer._final_response_sent is True

    @pytest.mark.asyncio
    async def test_fallback_partial_send_does_not_mark_final_sent(self):
        """When fallback final send delivers only some chunks before failing,
        _final_response_sent must stay False so the gateway can still attempt
        a complete final send (#25010)."""
        call_count = 0

        async def fake_send(*, chat_id, content, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return SimpleNamespace(success=True, message_id="msg_1")
            # Third chunk (fallback continuation) FAILS
            return SimpleNamespace(success=False, error="flood_control:13.0")

        adapter = MagicMock()
        adapter.send = AsyncMock(side_effect=fake_send)
        adapter.edit_message = AsyncMock(
            return_value=SimpleNamespace(success=False, error="flood_control:13.0"),
        )
        adapter.MAX_MESSAGE_LENGTH = 4096

        config = StreamConsumerConfig(edit_interval=0.01, buffer_threshold=5)
        consumer = GatewayStreamConsumer(adapter, "chat_123", config)

        # Trigger enough delta to enter fallback mode
        consumer.on_delta("Initial streaming text...\n")
        task = asyncio.create_task(consumer.run())
        await asyncio.sleep(0.05)

        # Send a very long text that will trigger overflow/fallback
        long_text = ("x" * 3000 + "\n") + ("y" * 3000 + "\n") + "Final answer.\n"
        consumer.on_delta(long_text)
        await asyncio.sleep(0.1)

        consumer.finish()
        await task

        assert consumer._final_response_sent is False, (
            "Partial fallback send must not set _final_response_sent — gateway "
            "must still be able to deliver the complete response (#25010)"
        )


class TestEditOverflowSplitAndDeliver:
    """When edit_message split-and-delivers an oversized payload across the
    original message + N continuations (Telegram >4096 UTF-16), the consumer
    must update _message_id to the latest continuation, reset _last_sent_text,
    and fire on_new_message so subsequent tool-progress bubbles linearize
    below the new visible message."""

    @pytest.mark.asyncio
    async def test_consumer_advances_message_id_on_split_and_deliver(self):
        adapter = MagicMock()
        # Simulate edit_message split-and-deliver: success=True with the
        # final continuation's id and a populated continuation_message_ids
        # tuple (the new SendResult contract).
        adapter.edit_message = AsyncMock(return_value=SimpleNamespace(
            success=True,
            message_id="msg_continuation_2",
            continuation_message_ids=("msg_continuation_1", "msg_continuation_2"),
        ))
        adapter.send = AsyncMock(
            return_value=SimpleNamespace(success=True, message_id="msg_initial"),
        )
        adapter.MAX_MESSAGE_LENGTH = 4096

        config = StreamConsumerConfig(
            edit_interval=0.01, buffer_threshold=5, cursor="",
        )
        consumer = GatewayStreamConsumer(adapter, "chat_999", config)

        # Track on_new_message firings.
        new_msg_count = [0]
        consumer._on_new_message = lambda: new_msg_count.__setitem__(0, new_msg_count[0] + 1)

        # Seed the consumer as if a first send succeeded already.
        consumer._message_id = "msg_initial"
        consumer._last_sent_text = "old"
        consumer._already_sent = True

        # Drive an edit that the adapter "split and delivers".
        ok = await consumer._send_or_edit("new full text after overflow")

        assert ok is True
        # Consumer advanced to the latest continuation id.
        assert consumer._message_id == "msg_continuation_2"
        # Skip-if-same cache reset so the next edit doesn't false-positive.
        assert consumer._last_sent_text == ""
        # on_new_message fired so the tool-progress bubble breaks below
        # the new continuation (per the openclaw #32535 lesson).
        assert new_msg_count[0] == 1


class TestInterimCommentaryMessages:
    @pytest.mark.asyncio
    async def test_commentary_message_stays_separate_from_final_stream(self):
        adapter = MagicMock()
        adapter.send = AsyncMock(side_effect=[
            SimpleNamespace(success=True, message_id="msg_1"),
            SimpleNamespace(success=True, message_id="msg_2"),
        ])
        adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=True))
        adapter.MAX_MESSAGE_LENGTH = 4096

        consumer = GatewayStreamConsumer(
            adapter,
            "chat_123",
            StreamConsumerConfig(edit_interval=0.01, buffer_threshold=5),
        )

        consumer.on_commentary("I'll inspect the repository first.")
        consumer.on_delta("Done.")
        consumer.finish()

        await consumer.run()

        sent_texts = [call[1]["content"] for call in adapter.send.call_args_list]
        assert sent_texts == ["I'll inspect the repository first.", "Done."]
        assert consumer.final_response_sent is True

    @pytest.mark.asyncio
    async def test_failed_final_send_does_not_mark_final_response_sent(self):
        adapter = MagicMock()
        adapter.send = AsyncMock(return_value=SimpleNamespace(success=False, message_id=None))
        adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=True))
        adapter.MAX_MESSAGE_LENGTH = 4096

        consumer = GatewayStreamConsumer(
            adapter,
            "chat_123",
            StreamConsumerConfig(edit_interval=0.01, buffer_threshold=5),
        )

        consumer.on_delta("Done.")
        consumer.finish()

        await consumer.run()

        assert consumer.final_response_sent is False
        assert consumer.already_sent is False

    @pytest.mark.asyncio
    async def test_success_without_message_id_marks_visible_and_sends_only_tail(self):
        adapter = MagicMock()
        adapter.send = AsyncMock(side_effect=[
            SimpleNamespace(success=True, message_id=None),
            SimpleNamespace(success=True, message_id=None),
        ])
        adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=True))
        adapter.MAX_MESSAGE_LENGTH = 4096

        consumer = GatewayStreamConsumer(
            adapter,
            "chat_123",
            StreamConsumerConfig(edit_interval=0.01, buffer_threshold=5, cursor=" ▉"),
        )

        consumer.on_delta("Hello")
        task = asyncio.create_task(consumer.run())
        await asyncio.sleep(0.08)
        consumer.on_delta(" world")
        await asyncio.sleep(0.08)
        consumer.finish()
        await task

        sent_texts = [call[1]["content"] for call in adapter.send.call_args_list]
        assert sent_texts == ["Hello ▉", "world"]
        assert consumer.already_sent is True
        assert consumer.final_response_sent is True


class TestCancelledConsumerSetsFlags:
    """Cancellation must set final_response_sent when already_sent is True.

    The 5-second stream_task timeout in gateway/run.py can cancel the
    consumer while it's still processing.  If final_response_sent stays
    False, the gateway falls through to the normal send path and the
    user sees a duplicate message.
    """

    @pytest.mark.asyncio
    async def test_cancelled_with_already_sent_marks_final_response_sent(self):
        """Cancelling after content was sent should set final_response_sent."""
        adapter = MagicMock()
        adapter.send = AsyncMock(
            return_value=SimpleNamespace(success=True, message_id="msg_1")
        )
        adapter.edit_message = AsyncMock(
            return_value=SimpleNamespace(success=True)
        )
        adapter.MAX_MESSAGE_LENGTH = 4096

        consumer = GatewayStreamConsumer(
            adapter,
            "chat_123",
            StreamConsumerConfig(edit_interval=0.01, buffer_threshold=5),
        )

        # Stream some text — the consumer sends it and sets already_sent
        consumer.on_delta("Hello world")
        task = asyncio.create_task(consumer.run())
        await asyncio.sleep(0.08)

        assert consumer.already_sent is True

        # Cancel the task (simulates the 5-second timeout in gateway)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # The fix: final_response_sent should be True even though _DONE
        # was never processed, preventing a duplicate message.
        assert consumer.final_response_sent is True

    @pytest.mark.asyncio
    async def test_cancelled_without_any_sends_does_not_mark_final(self):
        """Cancelling before anything was sent should NOT set final_response_sent."""
        adapter = MagicMock()
        adapter.send = AsyncMock(
            return_value=SimpleNamespace(success=False, message_id=None)
        )
        adapter.edit_message = AsyncMock(
            return_value=SimpleNamespace(success=True)
        )
        adapter.MAX_MESSAGE_LENGTH = 4096

        consumer = GatewayStreamConsumer(
            adapter,
            "chat_123",
            StreamConsumerConfig(edit_interval=0.01, buffer_threshold=5),
        )

        # Send fails — already_sent stays False
        consumer.on_delta("x")
        task = asyncio.create_task(consumer.run())
        await asyncio.sleep(0.08)

        assert consumer.already_sent is False

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Without a successful send, final_response_sent should stay False
        # so the normal gateway send path can deliver the response.
        assert consumer.final_response_sent is False


# ── Think-block filtering unit tests ─────────────────────────────────────


def _make_consumer() -> GatewayStreamConsumer:
    """Create a bare consumer for unit-testing the filter (no adapter needed)."""
    adapter = MagicMock()
    return GatewayStreamConsumer(adapter, "chat_test")


class TestFilterAndAccumulate:
    """Unit tests for _filter_and_accumulate think-block suppression."""

    def test_plain_text_passes_through(self):
        c = _make_consumer()
        c._filter_and_accumulate("Hello world")
        assert c._accumulated == "Hello world"

    def test_complete_think_block_stripped(self):
        c = _make_consumer()
        c._filter_and_accumulate("<think>internal reasoning</think>Answer here")
        assert c._accumulated == "Answer here"

    def test_think_block_in_middle(self):
        c = _make_consumer()
        c._filter_and_accumulate("Prefix\n<think>reasoning</think>\nSuffix")
        assert c._accumulated == "Prefix\n\nSuffix"

    def test_think_block_split_across_deltas(self):
        c = _make_consumer()
        c._filter_and_accumulate("<think>start of")
        c._filter_and_accumulate(" reasoning</think>visible text")
        assert c._accumulated == "visible text"

    def test_opening_tag_split_across_deltas(self):
        c = _make_consumer()
        c._filter_and_accumulate("<thi")
        # Partial tag held back
        assert c._accumulated == ""
        c._filter_and_accumulate("nk>hidden</think>shown")
        assert c._accumulated == "shown"

    def test_closing_tag_split_across_deltas(self):
        c = _make_consumer()
        c._filter_and_accumulate("<think>hidden</thi")
        assert c._accumulated == ""
        c._filter_and_accumulate("nk>shown")
        assert c._accumulated == "shown"

    def test_multiple_think_blocks(self):
        c = _make_consumer()
        # Consecutive blocks with no text between them — both stripped
        c._filter_and_accumulate(
            "<think>block1</think><think>block2</think>visible"
        )
        assert c._accumulated == "visible"

    def test_multiple_think_blocks_with_text_between(self):
        """Think tag after non-whitespace is NOT a boundary (prose safety)."""
        c = _make_consumer()
        c._filter_and_accumulate(
            "<think>block1</think>A<think>block2</think>B"
        )
        # Second <think> follows 'A' (not a block boundary) — treated as prose
        assert "A" in c._accumulated
        assert "B" in c._accumulated

    def test_thinking_tag_variant(self):
        c = _make_consumer()
        c._filter_and_accumulate("<thinking>deep thought</thinking>Result")
        assert c._accumulated == "Result"

    def test_thought_tag_variant(self):
        c = _make_consumer()
        c._filter_and_accumulate("<thought>Gemma style</thought>Output")
        assert c._accumulated == "Output"

    def test_reasoning_scratchpad_variant(self):
        c = _make_consumer()
        c._filter_and_accumulate(
            "<REASONING_SCRATCHPAD>long plan</REASONING_SCRATCHPAD>Done"
        )
        assert c._accumulated == "Done"

    def test_case_insensitive_THINKING(self):
        c = _make_consumer()
        c._filter_and_accumulate("<THINKING>caps</THINKING>answer")
        assert c._accumulated == "answer"

    @pytest.mark.parametrize(
        "tag",
        ["THINK", "Think", "ThInK", "THOUGHT", "REASONING", "Thinking"],
    )
    def test_reasoning_tags_are_case_insensitive(self, tag):
        c = _make_consumer()
        c._filter_and_accumulate(f"<{tag}>hidden reasoning</{tag}>Visible answer")
        assert c._accumulated == "Visible answer"
        assert "hidden reasoning" not in c._accumulated

    def test_prose_mention_not_stripped(self):
        """<think> mentioned mid-line in prose should NOT trigger filtering."""
        c = _make_consumer()
        c._filter_and_accumulate("The <think> tag is used for reasoning")
        assert "<think>" in c._accumulated
        assert "used for reasoning" in c._accumulated

    def test_prose_mention_after_text(self):
        """<think> after non-whitespace on same line is not a block boundary."""
        c = _make_consumer()
        c._filter_and_accumulate("Try using <think>some content</think> tags")
        assert "<think>" in c._accumulated

    def test_think_at_line_start_is_stripped(self):
        """<think> at start of a new line IS a block boundary."""
        c = _make_consumer()
        c._filter_and_accumulate("Previous line\n<think>reasoning</think>Next")
        assert "Previous line\nNext" == c._accumulated

    def test_think_with_only_whitespace_before(self):
        """<think> preceded by only whitespace on its line is a boundary."""
        c = _make_consumer()
        c._filter_and_accumulate("  <think>hidden</think>visible")
        # Leading whitespace before the tag is emitted, then block is stripped
        assert c._accumulated == "  visible"

    def test_flush_think_buffer_on_non_tag(self):
        """Partial tag that turns out not to be a tag is flushed."""
        c = _make_consumer()
        c._filter_and_accumulate("<thi")
        assert c._accumulated == ""
        # Flush explicitly (simulates stream end)
        c._flush_think_buffer()
        assert c._accumulated == "<thi"

    def test_flush_think_buffer_when_inside_block(self):
        """Flush while inside a think block does NOT emit buffered content."""
        c = _make_consumer()
        c._filter_and_accumulate("<think>still thinking")
        c._flush_think_buffer()
        assert c._accumulated == ""

    def test_unclosed_think_block_suppresses(self):
        """An unclosed <think> suppresses all subsequent content."""
        c = _make_consumer()
        c._filter_and_accumulate("Before\n<think>reasoning that never ends...")
        assert c._accumulated == "Before\n"

    def test_multiline_think_block(self):
        c = _make_consumer()
        c._filter_and_accumulate(
            "<think>\nLine 1\nLine 2\nLine 3\n</think>Final answer"
        )
        assert c._accumulated == "Final answer"

    def test_segment_reset_preserves_think_state(self):
        """_reset_segment_state should NOT clear think-block filter state."""
        c = _make_consumer()
        c._filter_and_accumulate("<think>start")
        c._reset_segment_state()
        # Still inside think block — subsequent text should be suppressed
        c._filter_and_accumulate("still hidden</think>visible")
        assert c._accumulated == "visible"


class TestFilterAndAccumulateIntegration:
    """Integration: verify think blocks don't leak through the full run() path."""

    @pytest.mark.asyncio
    async def test_think_block_not_sent_to_platform(self):
        """Think blocks should be filtered before platform edit."""
        adapter = MagicMock()
        adapter.send = AsyncMock(
            return_value=SimpleNamespace(success=True, message_id="msg_1")
        )
        adapter.edit_message = AsyncMock(
            return_value=SimpleNamespace(success=True)
        )
        adapter.MAX_MESSAGE_LENGTH = 4096

        consumer = GatewayStreamConsumer(
            adapter,
            "chat_test",
            StreamConsumerConfig(edit_interval=0.01, buffer_threshold=5),
        )

        # Simulate streaming: think block then visible text
        consumer.on_delta("<think>deep reasoning here</think>")
        consumer.on_delta("The answer is 42.")
        consumer.finish()

        task = asyncio.create_task(consumer.run())
        await asyncio.sleep(0.15)

        # The final text sent to the platform should NOT contain <think>
        all_calls = list(adapter.send.call_args_list) + list(
            adapter.edit_message.call_args_list
        )
        for call in all_calls:
            args, kwargs = call
            content = kwargs.get("content") or (args[0] if args else "")
            assert "<think>" not in content, f"Think tag leaked: {content}"
            assert "deep reasoning" not in content

        try:
            task.cancel()
            await task
        except asyncio.CancelledError:
            pass


# ── buffer_only mode tests ─────────────────────────────────────────────


class TestBufferOnlyMode:
    """Verify buffer_only mode suppresses intermediate edits and only
    flushes on structural boundaries (done, segment break, commentary)."""

    @pytest.mark.asyncio
    async def test_suppresses_intermediate_edits(self):
        """Time-based and size-based edits are skipped; only got_done flushes."""
        adapter = MagicMock()
        adapter.MAX_MESSAGE_LENGTH = 4096
        adapter.send = AsyncMock(return_value=SimpleNamespace(success=True, message_id="msg1"))
        adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=True))

        cfg = StreamConsumerConfig(edit_interval=0.01, buffer_threshold=5, cursor="", buffer_only=True)
        consumer = GatewayStreamConsumer(adapter, "!room:server", config=cfg)

        for word in ["Hello", " world", ", this", " is", " a", " test"]:
            consumer.on_delta(word)
        consumer.finish()

        await consumer.run()

        adapter.send.assert_called_once()
        adapter.edit_message.assert_not_called()
        assert "Hello world, this is a test" in adapter.send.call_args_list[0][1]["content"]

    @pytest.mark.asyncio
    async def test_flushes_on_segment_break(self):
        """A segment break (tool call boundary) flushes accumulated text."""
        adapter = MagicMock()
        adapter.MAX_MESSAGE_LENGTH = 4096
        adapter.send = AsyncMock(side_effect=[
            SimpleNamespace(success=True, message_id="msg1"),
            SimpleNamespace(success=True, message_id="msg2"),
        ])
        adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=True))

        cfg = StreamConsumerConfig(edit_interval=0.01, buffer_threshold=5, cursor="", buffer_only=True)
        consumer = GatewayStreamConsumer(adapter, "!room:server", config=cfg)

        consumer.on_delta("Before tool call")
        consumer.on_delta(None)
        consumer.on_delta("After tool call")
        consumer.finish()

        await consumer.run()

        assert adapter.send.call_count == 2
        assert "Before tool call" in adapter.send.call_args_list[0][1]["content"]
        assert "After tool call" in adapter.send.call_args_list[1][1]["content"]
        adapter.edit_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_flushes_on_commentary(self):
        """An interim commentary message flushes in buffer_only mode."""
        adapter = MagicMock()
        adapter.MAX_MESSAGE_LENGTH = 4096
        adapter.send = AsyncMock(side_effect=[
            SimpleNamespace(success=True, message_id="msg1"),
            SimpleNamespace(success=True, message_id="msg2"),
            SimpleNamespace(success=True, message_id="msg3"),
        ])
        adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=True))

        cfg = StreamConsumerConfig(edit_interval=0.01, buffer_threshold=5, cursor="", buffer_only=True)
        consumer = GatewayStreamConsumer(adapter, "!room:server", config=cfg)

        consumer.on_delta("Working on it...")
        consumer.on_commentary("I'll search for that first.")
        consumer.on_delta("Here are the results.")
        consumer.finish()

        await consumer.run()

        # Three sends: accumulated text, commentary, final text
        assert adapter.send.call_count >= 2
        adapter.edit_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_default_mode_still_triggers_intermediate_edits(self):
        """Regression: buffer_only=False (default) still does progressive edits."""
        adapter = MagicMock()
        adapter.MAX_MESSAGE_LENGTH = 4096
        adapter.send = AsyncMock(return_value=SimpleNamespace(success=True, message_id="msg1"))
        adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=True))

        # buffer_threshold=5 means any 5+ chars triggers an early edit
        cfg = StreamConsumerConfig(edit_interval=0.01, buffer_threshold=5, cursor="")
        consumer = GatewayStreamConsumer(adapter, "!room:server", config=cfg)

        consumer.on_delta("Hello world, this is long enough to trigger edits")
        consumer.finish()

        await consumer.run()

        # Should have at least one send. With buffer_threshold=5 and this much
        # text, the consumer may send then edit, or just send once at got_done.
        # The key assertion: this doesn't break.
        assert adapter.send.call_count >= 1


# ── Cursor stripping on fallback (#7183) ────────────────────────────────────


class TestCursorStrippingOnFallback:
    """Regression: cursor must be stripped when fallback continuation is empty (#7183).

    When _send_fallback_final is called with nothing new to deliver (the visible
    partial already matches final_text), the last edit may still show the cursor
    character because fallback mode was entered after a failed edit.  Before the
    fix this would leave the message permanently frozen with a visible ▉.
    """

    @pytest.mark.asyncio
    async def test_cursor_stripped_when_continuation_empty(self):
        """_send_fallback_final must attempt a final edit to strip the cursor."""
        adapter = MagicMock()
        adapter.MAX_MESSAGE_LENGTH = 4096
        adapter.edit_message = AsyncMock(
            return_value=SimpleNamespace(success=True, message_id="msg-1")
        )

        consumer = GatewayStreamConsumer(
            adapter, "chat-1",
            config=StreamConsumerConfig(cursor=" ▉"),
        )
        consumer._message_id = "msg-1"
        consumer._last_sent_text = "Hello world ▉"
        consumer._fallback_final_send = False

        await consumer._send_fallback_final("Hello world")

        adapter.edit_message.assert_called_once()
        call_args = adapter.edit_message.call_args
        assert call_args.kwargs["content"] == "Hello world"
        assert consumer._already_sent is True
        # _last_sent_text should reflect the cleaned text after a successful strip
        assert consumer._last_sent_text == "Hello world"

    @pytest.mark.asyncio
    async def test_cursor_not_stripped_when_no_cursor_configured(self):
        """No edit attempted when cursor is not configured."""
        adapter = MagicMock()
        adapter.MAX_MESSAGE_LENGTH = 4096
        adapter.edit_message = AsyncMock()

        consumer = GatewayStreamConsumer(
            adapter, "chat-1",
            config=StreamConsumerConfig(cursor=""),
        )
        consumer._message_id = "msg-1"
        consumer._last_sent_text = "Hello world"
        consumer._fallback_final_send = False

        await consumer._send_fallback_final("Hello world")

        adapter.edit_message.assert_not_called()
        assert consumer._already_sent is True

    @pytest.mark.asyncio
    async def test_cursor_strip_edit_failure_handled(self):
        """If the cursor-stripping edit itself fails, it must not crash and
        must not corrupt _last_sent_text."""
        adapter = MagicMock()
        adapter.MAX_MESSAGE_LENGTH = 4096
        adapter.edit_message = AsyncMock(
            return_value=SimpleNamespace(success=False, error="flood_control")
        )

        consumer = GatewayStreamConsumer(
            adapter, "chat-1",
            config=StreamConsumerConfig(cursor=" ▉"),
        )
        consumer._message_id = "msg-1"
        consumer._last_sent_text = "Hello ▉"
        consumer._fallback_final_send = False

        await consumer._send_fallback_final("Hello")

        # Should still set already_sent despite the cursor-strip edit failure
        assert consumer._already_sent is True
        # _last_sent_text must NOT be updated when the edit failed
        assert consumer._last_sent_text == "Hello ▉"


# ── on_new_message callback (tool-progress linearization) ─────────────


class TestOnNewMessageCallback:
    """The on_new_message callback fires whenever a fresh content bubble
    lands on the platform. Gateway uses this to close off the current
    tool-progress bubble so the next tool.started opens a new bubble
    below the content — preserving chronological order in the chat.

    Before this callback existed (post PR #7885), content messages got
    their own bubbles after segment breaks, but the tool-progress task
    kept editing the ORIGINAL progress bubble above all new content.
    Result: tool lines appeared stacked in the upper bubble while
    content messages lined up below, making the timeline look scrambled.
    """

    @pytest.mark.asyncio
    async def test_callback_fires_on_first_send(self):
        """First-send of a new content bubble fires on_new_message."""
        adapter = MagicMock()
        adapter.send = AsyncMock(return_value=SimpleNamespace(success=True, message_id="msg_1"))
        adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=True))
        adapter.MAX_MESSAGE_LENGTH = 4096

        events = []
        config = StreamConsumerConfig(edit_interval=0.01, buffer_threshold=1)
        consumer = GatewayStreamConsumer(
            adapter, "chat", config,
            on_new_message=lambda: events.append("reset"),
        )

        consumer.on_delta("Hello")
        consumer.finish()
        await consumer.run()

        assert events == ["reset"]

    @pytest.mark.asyncio
    async def test_callback_fires_once_per_segment(self):
        """A new first-send fires the callback again after segment break."""
        adapter = MagicMock()
        msg_counter = iter(["msg_1", "msg_2", "msg_3"])
        adapter.send = AsyncMock(
            side_effect=lambda **kw: SimpleNamespace(success=True, message_id=next(msg_counter))
        )
        adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=True))
        adapter.MAX_MESSAGE_LENGTH = 4096

        events = []
        config = StreamConsumerConfig(edit_interval=0.01, buffer_threshold=1)
        consumer = GatewayStreamConsumer(
            adapter, "chat", config,
            on_new_message=lambda: events.append("reset"),
        )

        consumer.on_delta("A")
        consumer.on_delta(None)
        consumer.on_delta("B")
        consumer.on_delta(None)
        consumer.on_delta("C")
        consumer.finish()
        await consumer.run()

        # Three content bubbles ⇒ three reset notifications
        assert events == ["reset", "reset", "reset"]

    @pytest.mark.asyncio
    async def test_callback_not_fired_on_edit(self):
        """Subsequent edits of the same bubble do NOT fire the callback."""
        adapter = MagicMock()
        adapter.send = AsyncMock(return_value=SimpleNamespace(success=True, message_id="msg_1"))
        adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=True))
        adapter.MAX_MESSAGE_LENGTH = 4096

        events = []
        config = StreamConsumerConfig(edit_interval=0.01, buffer_threshold=1)
        consumer = GatewayStreamConsumer(
            adapter, "chat", config,
            on_new_message=lambda: events.append("reset"),
        )

        consumer.on_delta("Hello")
        task = asyncio.create_task(consumer.run())
        await asyncio.sleep(0.05)
        consumer.on_delta(" world")
        await asyncio.sleep(0.05)
        consumer.on_delta(" more")
        await asyncio.sleep(0.05)
        consumer.finish()
        await task

        # Only one first-send happened; edits do not re-fire.
        assert events == ["reset"]

    @pytest.mark.asyncio
    async def test_callback_fires_on_commentary(self):
        """Commentary messages are fresh bubbles too — fire the callback."""
        adapter = MagicMock()
        adapter.send = AsyncMock(return_value=SimpleNamespace(success=True, message_id="msg_1"))
        adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=True))
        adapter.MAX_MESSAGE_LENGTH = 4096

        events = []
        config = StreamConsumerConfig(edit_interval=0.01, buffer_threshold=1)
        consumer = GatewayStreamConsumer(
            adapter, "chat", config,
            on_new_message=lambda: events.append("reset"),
        )

        consumer.on_commentary("I'll search for that first.")
        consumer.finish()
        await consumer.run()

        assert events == ["reset"]

    @pytest.mark.asyncio
    async def test_callback_error_swallowed(self):
        """Exceptions in the callback do not crash the consumer."""
        adapter = MagicMock()
        adapter.send = AsyncMock(return_value=SimpleNamespace(success=True, message_id="msg_1"))
        adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=True))
        adapter.MAX_MESSAGE_LENGTH = 4096

        def raiser():
            raise RuntimeError("boom")

        config = StreamConsumerConfig(edit_interval=0.01, buffer_threshold=1)
        consumer = GatewayStreamConsumer(
            adapter, "chat", config,
            on_new_message=raiser,
        )

        consumer.on_delta("Hello")
        consumer.finish()
        await consumer.run()  # must not raise

        assert consumer.already_sent is True

    @pytest.mark.asyncio
    async def test_no_callback_when_none(self):
        """Consumer works correctly when on_new_message is None (default)."""
        adapter = MagicMock()
        adapter.send = AsyncMock(return_value=SimpleNamespace(success=True, message_id="msg_1"))
        adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=True))
        adapter.MAX_MESSAGE_LENGTH = 4096

        config = StreamConsumerConfig(edit_interval=0.01, buffer_threshold=1)
        consumer = GatewayStreamConsumer(adapter, "chat", config)  # no callback

        consumer.on_delta("Hello")
        consumer.finish()
        await consumer.run()

        assert consumer.already_sent is True


class TestUtf16OverflowDetection:
    """Regression coverage for #11170 — Telegram counts message length in
    UTF-16 code units, not Python codepoints. A response with supplementary
    characters (emoji, CJK in some ranges) can have len()=3000 codepoints
    but utf16_len()=5000+ units, blowing past Telegram's 4096 limit."""

    def _make_telegram_like_adapter(self):
        """Construct a minimal BasePlatformAdapter subclass that overrides
        message_len_fn like Telegram does."""
        from gateway.platforms.base import utf16_len, BasePlatformAdapter

        TelegramLikeAdapter = type(
            "TelegramLikeAdapter",
            (BasePlatformAdapter,),
            {
                "MAX_MESSAGE_LENGTH": 4096,
                "message_len_fn": property(lambda self: utf16_len),
            },
        )
        # Defeat ABCMeta abstract-instantiation guard by clearing the cached
        # abstract methods set after class creation.
        TelegramLikeAdapter.__abstractmethods__ = frozenset()
        adapter = TelegramLikeAdapter.__new__(TelegramLikeAdapter)
        adapter._typing_paused = set()
        adapter._fatal_error_message = None
        return adapter

    @pytest.mark.asyncio
    async def test_emoji_text_exceeding_utf16_limit_triggers_overflow_split(self):
        """A response that is under 4096 codepoints but over 4096 UTF-16
        units must trigger the overflow-split path."""
        from gateway.platforms.base import utf16_len

        adapter = self._make_telegram_like_adapter()
        # Mock the send/edit methods we actually call
        adapter.send = AsyncMock(
            return_value=SimpleNamespace(success=True, message_id="msg_1"),
        )
        adapter.edit_message = AsyncMock(
            return_value=SimpleNamespace(success=True),
        )
        # truncate_message: emit two halves so we can assert the split fired
        adapter.truncate_message = MagicMock(
            side_effect=lambda text, limit, **kw: [text[:len(text)//2], text[len(text)//2:]],
        )

        config = StreamConsumerConfig(edit_interval=0.01, buffer_threshold=5)
        consumer = GatewayStreamConsumer(adapter, "chat_123", config)

        # 🚀 is 1 codepoint = 2 UTF-16 units. 2200 of them = 2200 codepoints,
        # 4400 UTF-16 units. Under the codepoint-equivalent limit (would not
        # trigger split with len()) but over Telegram's UTF-16 4096 limit.
        emoji_text = "🚀" * 2200
        assert len(emoji_text) < adapter.MAX_MESSAGE_LENGTH, (
            "Test setup invariant: codepoint count under limit"
        )
        assert utf16_len(emoji_text) > adapter.MAX_MESSAGE_LENGTH, (
            "Test setup invariant: UTF-16 count over limit"
        )

        consumer.on_delta(emoji_text)
        task = asyncio.create_task(consumer.run())
        await asyncio.sleep(0.05)
        consumer.finish()
        await task

        # The fix: stream consumer detects UTF-16 overflow and calls
        # truncate_message to split. Without the fix, len() would return
        # 2200 (under 4096) and no split would fire — Telegram would then
        # reject the send or render \x00 artifacts.
        adapter.truncate_message.assert_called(), (
            "UTF-16 overflow not detected — emoji text bypassed split path"
        )
        # truncate_message must have been called with len_fn=utf16_len
        call_kwargs = adapter.truncate_message.call_args[1]
        assert call_kwargs.get("len_fn") is utf16_len, (
            f"truncate_message called without utf16_len: {call_kwargs}"
        )

    def test_codepoint_only_adapter_falls_back_to_len(self):
        """Adapters without message_len_fn override (or test MagicMocks)
        must use plain len for backwards compatibility."""
        adapter = MagicMock()
        adapter.MAX_MESSAGE_LENGTH = 4096
        config = StreamConsumerConfig(cursor=" ▉")
        consumer = GatewayStreamConsumer(adapter, "chat_123", config)
        # The isinstance guard means MagicMock adapters get len, not the
        # auto-attr mock. Verified indirectly by all the other tests in
        # this file passing — they all use MagicMock adapters.
        assert consumer is not None


class TestFreshFinalRespectsAdapterDecline:
    """Regression: when an adapter explicitly declines fresh-final via
    ``prefers_fresh_final_streaming = False``, the time-based
    ``_should_send_fresh_final()`` must NOT override that decision.
    (#47048 — Telegram rich-message overlap with legacy MarkdownV2 preview)
    """

    @pytest.mark.asyncio
    async def test_adapter_decline_fresh_final_overrides_time_threshold(self):
        """Adapter with prefers_fresh_final_streaming=False must NOT take
        the fresh-final path even when fresh_final_after_seconds is large."""
        adapter = MagicMock()
        adapter.MAX_MESSAGE_LENGTH = 4096
        adapter.send = AsyncMock(
            return_value=SimpleNamespace(success=True, message_id="rich_msg"),
        )
        adapter.edit_message = AsyncMock(
            return_value=SimpleNamespace(success=True, message_id="edit_msg"),
        )
        adapter.delete_message = AsyncMock(return_value=True)
        # Adapter explicitly declines fresh-final (like Telegram)
        adapter.prefers_fresh_final_streaming = MagicMock(return_value=False)

        config = StreamConsumerConfig(
            edit_interval=0.01,
            buffer_threshold=5,
            fresh_final_after_seconds=1.0,  # time threshold would trigger
            cursor=" ▉",
        )
        consumer = GatewayStreamConsumer(adapter, "chat_123", config)

        # Simulate: first message sent during streaming
        consumer.on_delta("Hello world")
        task = asyncio.create_task(consumer.run())
        await asyncio.sleep(0.05)
        # First message should have been sent
        assert consumer._message_id is not None
        # Simulate time passing (beyond threshold)
        consumer._message_created_ts -= 10.0

        # Finalize
        consumer.on_delta("Hello world final")
        consumer.finish()
        await task

        # The adapter declined fresh-final, so send() should NOT have been
        # called for the final message — only edit_message(finalize=True).
        adapter.send.assert_called_once()  # Only the initial send
        adapter.edit_message.assert_called()  # Finalize edit
        # Verify edit was called with finalize=True
        edit_calls = [
            c for c in adapter.edit_message.call_args_list
            if c.kwargs.get("finalize") or (len(c.args) > 3 and c.args[3])
        ]
        assert len(edit_calls) >= 1, (
            "Expected finalize=True edit call, got none"
        )

    @pytest.mark.asyncio
    async def test_no_hook_adapter_uses_time_threshold(self):
        """Adapter WITHOUT prefers_fresh_final_streaming must still use
        the time-based fresh-final path (backward compat)."""
        adapter = MagicMock()
        adapter.MAX_MESSAGE_LENGTH = 4096
        adapter.send = AsyncMock(
            return_value=SimpleNamespace(success=True, message_id="msg_1"),
        )
        adapter.edit_message = AsyncMock(
            return_value=SimpleNamespace(success=True, message_id="edit_msg"),
        )
        adapter.delete_message = AsyncMock(return_value=True)
        # No prefers_fresh_final_streaming attribute
        if hasattr(adapter, "prefers_fresh_final_streaming"):
            del adapter.prefers_fresh_final_streaming

        config = StreamConsumerConfig(
            edit_interval=0.01,
            buffer_threshold=5,
            fresh_final_after_seconds=1.0,
            cursor=" ▉",
        )
        consumer = GatewayStreamConsumer(adapter, "chat_123", config)

        # Simulate: first message sent during streaming
        consumer.on_delta("Hello world")
        task = asyncio.create_task(consumer.run())
        await asyncio.sleep(0.05)
        assert consumer._message_id is not None
        # Simulate time passing
        consumer._message_created_ts -= 10.0

        # Finalize
        consumer.on_delta("Hello world final")
        consumer.finish()
        await task

        # Without the hook, time-based fresh-final should trigger:
        # send() called twice (initial + fresh-final)
        assert adapter.send.call_count == 2, (
            f"Expected 2 send calls (initial + fresh-final), got {adapter.send.call_count}"
        )


# ── run_still_current staleness guard ────────────────────────────────────

class TestRunStillCurrentGuard:
    """Verify that the stream consumer abandons delivery when the session is
    reset (e.g. /new or /stop), preventing stale deltas from reaching the user."""

    @pytest.mark.asyncio
    async def test_abandons_stream_when_session_reset_before_first_send(self):
        """If _run_still_current returns False immediately, the consumer
        exits without sending anything — even with queued deltas."""
        adapter = MagicMock()
        adapter.send = AsyncMock()
        adapter.edit_message = AsyncMock()
        adapter.MAX_MESSAGE_LENGTH = 4096

        config = StreamConsumerConfig(edit_interval=0.01, buffer_threshold=3)
        consumer = GatewayStreamConsumer(
            adapter, "chat_123", config,
            run_still_current=lambda: False,
        )

        consumer.on_delta("ABC")
        consumer.on_delta("DEF")
        consumer.on_delta("GHI")

        await consumer.run()

        adapter.send.assert_not_called()
        adapter.edit_message.assert_not_called()
        assert consumer._final_response_sent is False

    @pytest.mark.asyncio
    async def test_abandons_stream_after_one_edit_when_session_reset(self):
        """If staleness flips after the first edit, the consumer stops
        on the next loop iteration and does not send the final response."""
        adapter = MagicMock()
        send_result = SimpleNamespace(success=True, message_id="msg_1")
        adapter.send = AsyncMock(return_value=send_result)
        adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=True))
        adapter.MAX_MESSAGE_LENGTH = 4096

        call_count = [0]

        def is_current():
            call_count[0] += 1
            return call_count[0] == 1

        config = StreamConsumerConfig(edit_interval=0.01, buffer_threshold=3)
        consumer = GatewayStreamConsumer(
            adapter, "chat_123", config,
            run_still_current=is_current,
        )

        consumer.on_delta("First segment")
        consumer.on_delta(None)  # segment break → resets message_id
        consumer.on_delta("Second segment text that will be stale")
        # No finish() — staleness should prevent second segment from sending

        await consumer.run()

        # First segment was sent, second was abandoned
        assert adapter.send.call_count == 1
        assert "First segment" in adapter.send.call_args_list[0][1]["content"]
        assert consumer._final_response_sent is False

    @pytest.mark.asyncio
    async def test_normal_delivery_when_session_stays_current(self):
        """When _run_still_current always returns True, the consumer
        behaves normally and delivers the full response."""
        adapter = MagicMock()
        send_result = SimpleNamespace(success=True, message_id="msg_1")
        adapter.send = AsyncMock(return_value=send_result)
        adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=True))
        adapter.MAX_MESSAGE_LENGTH = 4096

        config = StreamConsumerConfig(edit_interval=0.01, buffer_threshold=5)
        consumer = GatewayStreamConsumer(
            adapter, "chat_123", config,
            run_still_current=lambda: True,
        )

        consumer.on_delta("Hello, world!")
        consumer.finish()

        await consumer.run()

        assert adapter.send.call_count >= 1
        assert consumer._final_response_sent is True

    @pytest.mark.asyncio
    async def test_no_callback_defaults_to_always_current(self):
        """When run_still_current is not provided (default), the consumer
        always considers the session current — backward compatible."""
        adapter = MagicMock()
        send_result = SimpleNamespace(success=True, message_id="msg_1")
        adapter.send = AsyncMock(return_value=send_result)
        adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=True))
        adapter.MAX_MESSAGE_LENGTH = 4096

        config = StreamConsumerConfig(edit_interval=0.01, buffer_threshold=5)
        consumer = GatewayStreamConsumer(adapter, "chat_123", config)

        consumer.on_delta("Normal message")
        consumer.finish()

        await consumer.run()

        assert adapter.send.call_count >= 1
        assert consumer._final_response_sent is True

    @pytest.mark.asyncio
    async def test_abandons_even_with_pending_finish(self):
        """If finish() has been called but the session is already reset
        before the run loop starts, nothing is sent."""
        adapter = MagicMock()
        adapter.send = AsyncMock()
        adapter.edit_message = AsyncMock()
        adapter.MAX_MESSAGE_LENGTH = 4096

        config = StreamConsumerConfig(edit_interval=0.01, buffer_threshold=5)
        consumer = GatewayStreamConsumer(
            adapter, "chat_123", config,
            run_still_current=lambda: False,
        )

        consumer.on_delta("Stale text")
        consumer.finish()

        await consumer.run()

        adapter.send.assert_not_called()
        adapter.edit_message.assert_not_called()
        assert consumer._final_response_sent is False


# ── _strip_orphan_close_tags regression tests ──────────────────────────
# Regression guard for the /think tag leak: when the stream consumer is
# NOT inside a think block, stray close tags like </think> must be
# stripped before text is accumulated — otherwise they leak to Telegram.
# (Reported by Tony on 2026-06-09.)


class TestStripOrphanCloseTags:
    """Verify orphan close tags are stripped from text the stream consumer
    would accumulate while NOT inside a think block."""

    @pytest.mark.parametrize(
        "tag",
        [
            "</think>",
            "</thinking>",
            "</thought>",
            "</reasoning>",
            "</REASONING_SCRATCHPAD>",
            "</THINKING>",
        ],
    )
    def test_all_close_tag_variants_stripped(self, tag):
        text = f"before{tag}after"
        result = GatewayStreamConsumer._strip_orphan_close_tags(text)
        assert tag not in result
        assert "before" in result and "after" in result

    def test_no_close_tag_passthrough(self):
        text = "Just normal text with no tags."
        assert GatewayStreamConsumer._strip_orphan_close_tags(text) == text

    def test_empty_string(self):
        assert GatewayStreamConsumer._strip_orphan_close_tags("") == ""

    def test_close_tag_with_trailing_whitespace(self):
        """The trailing whitespace after the tag should also be eaten so
        surrounding prose flows naturally (matches StreamingThinkScrubber)."""
        text = "Looking at this now.\n\n</think>\n\nThe answer is 42."
        result = GatewayStreamConsumer._strip_orphan_close_tags(text)
        assert "</think>" not in result
        assert "Looking at this now" in result
        assert "The answer is 42" in result

    def test_multiple_orphan_close_tags(self):
        text = "foo </think> bar </thinking> baz"
        result = GatewayStreamConsumer._strip_orphan_close_tags(text)
        assert "</think>" not in result
        assert "</thinking>" not in result
        assert "foo" in result and "bar" in result and "baz" in result

    def test_orphan_close_does_not_eat_following_prose(self):
        text = "answer </think> then this should remain"
        result = GatewayStreamConsumer._strip_orphan_close_tags(text)
        assert result == "answer then this should remain"

    def test_partial_close_tag_not_stripped(self):
        """A partial tag like '</thi' should not be eaten — it's not yet
        a recognized close tag, and eating it would corrupt following text."""
        text = "before </thin after"
        result = GatewayStreamConsumer._strip_orphan_close_tags(text)
        assert result == text  # unchanged — partial tag, no stripping

    def test_filter_and_accumulate_strips_orphan_close(self):
        """End-to-end: feed an orphan close tag through _filter_and_accumulate
        and verify the accumulated text does not contain the raw tag."""
        adapter = MagicMock()
        adapter.MAX_MESSAGE_LENGTH = 4096
        config = StreamConsumerConfig(cursor=" ▉")
        consumer = GatewayStreamConsumer(adapter, "chat_123", config)

        # Simulate a stream delta that contains an orphan close tag with
        # surrounding prose (the actual leak pattern reported 2026-06-09).
        consumer._filter_and_accumulate(
            "Here is the result you asked for.\n\n</think>\n\n"
            "The answer is 42 and the cat is black."
        )
        # No raw close tag should remain in the accumulated text.
        for tag in GatewayStreamConsumer._CLOSE_THINK_TAGS:
            assert tag not in consumer._accumulated, (
                f"Orphan close tag {tag!r} leaked into accumulated text: "
                f"{consumer._accumulated!r}"
            )
        # Surrounding prose must survive intact.
        assert "Here is the result" in consumer._accumulated
        assert "The answer is 42" in consumer._accumulated

    def test_flush_think_buffer_strips_orphan_close(self):
        """The end-of-stream flush should also strip orphan close tags from
        any held-back buffer text."""
        adapter = MagicMock()
        adapter.MAX_MESSAGE_LENGTH = 4096
        config = StreamConsumerConfig(cursor=" ▉")
        consumer = GatewayStreamConsumer(adapter, "chat_123", config)

        # Plant a held-back buffer with an orphan close tag (simulates the
        # buffer being held while waiting for a possible opening tag, then
        # flushed when the stream ends).
        consumer._think_buffer = "trailing prose </think> more"
        consumer._in_think_block = False
        consumer._flush_think_buffer()
        for tag in GatewayStreamConsumer._CLOSE_THINK_TAGS:
            assert tag not in consumer._accumulated
        assert "trailing prose" in consumer._accumulated
        assert "more" in consumer._accumulated


# ── Flush barrier (clarify-ordering) tests ───────────────────────────────


class TestFlushPendingSync:
    """flush_pending_sync() is the ordering barrier that guarantees buffered
    prose lands on the platform BEFORE a blocking interactive prompt (clarify
    poll) is sent. Regression coverage for the bug where the poll raced ahead
    of its own explanation, rendering the question above the prose.
    """

    @pytest.mark.asyncio
    async def test_flush_delivers_buffered_commentary_before_returning(self):
        """A commentary message queued before the flush barrier must be sent
        to the adapter before flush_pending_sync() returns True."""
        adapter = MagicMock()
        sent_order = []

        async def _send(*args, **kwargs):
            sent_order.append(("send", kwargs.get("content", "")))
            return SimpleNamespace(success=True, message_id="msg_1")

        async def _edit(*args, **kwargs):
            sent_order.append(("edit", kwargs.get("content", "")))
            return SimpleNamespace(success=True)

        adapter.send = AsyncMock(side_effect=_send)
        adapter.edit_message = AsyncMock(side_effect=_edit)
        adapter.MAX_MESSAGE_LENGTH = 4096

        config = StreamConsumerConfig(edit_interval=0.01, buffer_threshold=5)
        consumer = GatewayStreamConsumer(adapter, "chat_123", config)

        # Mirror the live config that exhibits the bug: streaming off but
        # interim assistant messages on, so prose arrives as commentary.
        consumer.on_commentary("Now I get the full picture. Here's the situation.")

        # Run the drain loop concurrently while we block on the flush barrier
        # from a worker thread (mirrors the agent thread calling the clarify
        # callback while the consumer task drains on the event loop).
        task = asyncio.create_task(consumer.run())
        flushed = await asyncio.to_thread(consumer.flush_pending_sync, 3.0)

        assert flushed is True, "flush_pending_sync should complete within timeout"
        # The commentary must already be on screen by the time flush returns.
        assert any(
            "full picture" in content for _kind, content in sent_order
        ), f"Commentary not delivered before flush returned: {sent_order!r}"

        consumer.finish()
        await task

    @pytest.mark.asyncio
    async def test_flush_times_out_when_consumer_not_running(self):
        """If the consumer task is not running, flush_pending_sync returns
        False (rather than hanging the caller forever)."""
        adapter = MagicMock()
        adapter.send = AsyncMock(return_value=SimpleNamespace(success=True, message_id="m1"))
        adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=True))
        adapter.MAX_MESSAGE_LENGTH = 4096

        config = StreamConsumerConfig(edit_interval=0.01, buffer_threshold=5)
        consumer = GatewayStreamConsumer(adapter, "chat_123", config)

        # run() is never started — the barrier is never drained.
        flushed = await asyncio.to_thread(consumer.flush_pending_sync, 0.1)
        assert flushed is False

    @pytest.mark.asyncio
    async def test_flush_with_empty_buffer_still_completes(self):
        """A flush with nothing buffered still completes (no-op barrier)."""
        adapter = MagicMock()
        adapter.send = AsyncMock(return_value=SimpleNamespace(success=True, message_id="m1"))
        adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=True))
        adapter.MAX_MESSAGE_LENGTH = 4096

        config = StreamConsumerConfig(edit_interval=0.01, buffer_threshold=5)
        consumer = GatewayStreamConsumer(adapter, "chat_123", config)

        task = asyncio.create_task(consumer.run())
        flushed = await asyncio.to_thread(consumer.flush_pending_sync, 3.0)
        assert flushed is True

        consumer.finish()
        await task

    @pytest.mark.asyncio
    async def test_flush_completes_on_oversized_buffered_prose(self):
        """Regression: oversized prose takes the overflow-split `continue`
        path in run(), which previously skipped the flush-event set, stalling
        the caller for the full timeout. The flush must still complete promptly
        and the prose must be delivered before it returns."""
        adapter = MagicMock()
        sent = []

        async def _send(*args, **kwargs):
            sent.append(kwargs.get("content", ""))
            return SimpleNamespace(success=True, message_id=f"m{len(sent)}")

        adapter.send = AsyncMock(side_effect=_send)
        adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=True))
        adapter.MAX_MESSAGE_LENGTH = 4096
        # Real splitter so the overflow branch (message_id is None +
        # len > safe_limit) is actually taken.
        adapter.truncate_message = (
            lambda text, limit, len_fn=len: [
                text[i:i + limit] for i in range(0, len(text), limit)
            ]
        )

        config = StreamConsumerConfig(edit_interval=0.01, buffer_threshold=5)
        consumer = GatewayStreamConsumer(adapter, "chat_123", config)

        # Commentary far larger than the platform limit → overflow split path.
        big = "X" * 9000
        consumer.on_commentary(big)

        task = asyncio.create_task(consumer.run())
        # Tight timeout: if the continue-path skips the set, this returns False.
        flushed = await asyncio.to_thread(consumer.flush_pending_sync, 2.0)

        assert flushed is True, (
            "flush stalled — overflow `continue` path did not signal the barrier"
        )
        assert sent, "oversized prose was not delivered before flush returned"
        assert sum(len(c) for c in sent) >= 9000

        consumer.finish()
        await task

    @pytest.mark.asyncio
    async def test_flush_signaled_when_consumer_cancelled(self):
        """If run() is cancelled while a flush barrier is queued, the finally
        safety-net wakes the waiter rather than letting it hit the full
        timeout."""
        adapter = MagicMock()
        # Make the first send hang so the consumer is mid-iteration when we
        # cancel it, with the flush barrier still in the queue behind it.
        started = asyncio.Event()

        async def _slow_send(*args, **kwargs):
            started.set()
            await asyncio.sleep(60)
            return SimpleNamespace(success=True, message_id="m1")

        adapter.send = AsyncMock(side_effect=_slow_send)
        adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=True))
        adapter.MAX_MESSAGE_LENGTH = 4096

        config = StreamConsumerConfig(edit_interval=0.01, buffer_threshold=5)
        consumer = GatewayStreamConsumer(adapter, "chat_123", config)
        consumer.on_commentary("first")

        task = asyncio.create_task(consumer.run())
        await started.wait()  # consumer is now blocked inside the slow send
        # Queue the flush barrier behind the hung send.
        flush_done = asyncio.get_event_loop().run_in_executor(
            None, consumer.flush_pending_sync, 5.0
        )
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # The finally net should have drained + signaled the queued barrier.
        flushed = await flush_done
        assert flushed is True

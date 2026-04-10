"""Tests for GatewayStreamConsumer — media directive stripping in streaming."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig


# ── _sanitize_for_telegram unit tests ────────────────────────────────────────


class TestSanitizeForTelegram:
    """Verify terminal glyphs, box-drawing, braille, and zero-width chars are
    stripped, and block cursors are replaced with safe ASCII alternatives."""

    # Progress-bar glyphs ───────────────────────────────────────────────────

    def test_block_elements_stripped(self):
        """Full block: █ U+2588 and lighter variants ▁–▇ all removed."""
        text = "Loading: ██████████ 100%"
        result = GatewayStreamConsumer._sanitize_for_telegram(text)
        assert "█" not in result
        # Glyphs are stripped; double-spaces from that are collapsed to one space.
        assert "Loading: 100%" in result

    def test_light_block_elements(self):
        """Light shade ░ (U+2591) through medium ▒ (U+2592) stripped."""
        text = "[░░░░░▒▒▒▓▓▓███] 80%"
        result = GatewayStreamConsumer._sanitize_for_telegram(text)
        assert "░" not in result
        assert "▒" not in result
        assert "▓" not in result
        assert " 80%" in result

    def test_progress_bar_format_cleaned(self):
        """ASCII-style progress bar with block chars becomes clean text."""
        text = "████████░░ 80% complete"
        result = GatewayStreamConsumer._sanitize_for_telegram(text)
        assert "█" not in result
        assert "░" not in result
        assert "80% complete" in result

    # Block cursor ──────────────────────────────────────────────────────────

    def test_block_cursor_replaced_with_dots(self):
        """▉ block cursor becomes '...' (default safe fallback)."""
        text = "Thinking▉"
        result = GatewayStreamConsumer._sanitize_for_telegram(text)
        assert "▉" not in result
        assert "Thinking..." in result

    def test_block_cursor_custom_cursor(self):
        """When a custom non-block cursor is passed, it is preserved."""
        text = "Thinking▉"
        result = GatewayStreamConsumer._sanitize_for_telegram(text, cursor=" ✦")
        assert "▉" not in result
        assert "✦" in result

    def test_block_cursor_with_leading_space(self):
        """' ▉' (cursor with leading space) is also replaced."""
        text = "Thinking ▉"
        result = GatewayStreamConsumer._sanitize_for_telegram(text)
        assert "▉" not in result
        # The ▉ in text is replaced with "..." (safe fallback cursor)
        assert "Thinking ..." in result

    def test_non_block_cursor_preserved(self):
        """Safe custom cursors like '→' are left untouched."""
        text = "Thinking →"
        result = GatewayStreamConsumer._sanitize_for_telegram(text, cursor="→")
        assert "→" in result

    # Box-drawing characters ────────────────────────────────────────────────

    def test_box_drawing_stripped(self):
        """Horizontal line: ─ (U+2500) and vertical: │ (U+2502) removed."""
        text = "─── Header ───"
        result = GatewayStreamConsumer._sanitize_for_telegram(text)
        assert "─" not in result
        assert "Header" in result

    def test_box_drawing_corner_chars_stripped(self):
        """┌ ┐ └ ┘ and all box-drawing chars removed."""
        text = "┌────────┐\n│  Box   │\n└────────┘"
        result = GatewayStreamConsumer._sanitize_for_telegram(text)
        for char in "┌┐└┘│─":
            assert char not in result, f"{char} should have been stripped"
        assert "Box" in result

    def test_mixed_box_drawing(self):
        """Mixed box-drawing and text: chars stripped, text preserved."""
        text = "Step 1 ──── Step 2 │ Result"
        result = GatewayStreamConsumer._sanitize_for_telegram(text)
        assert "──" not in result
        assert "│" not in result
        assert "Step 1" in result
        assert "Step 2" in result
        assert "Result" in result

    # Zero-width and control characters ───────────────────────────────────

    def test_zero_width_space_stripped(self):
        """U+200B zero-width space removed."""
        text = "hello\u200bworld"
        result = GatewayStreamConsumer._sanitize_for_telegram(text)
        assert "\u200b" not in result
        assert "helloworld" in result

    def test_zero_width_joiner_stripped(self):
        """U+200D zero-width joiner removed."""
        text = "hi\u200dthere"
        result = GatewayStreamConsumer._sanitize_for_telegram(text)
        assert "\u200d" not in result

    def test_bom_stripped(self):
        """U+FEFF BOM removed."""
        text = "\ufeffHello"
        result = GatewayStreamConsumer._sanitize_for_telegram(text)
        assert "\ufeff" not in result
        assert "Hello" in result

    def test_non_breaking_space_replaced(self):
        """U+00A0 non-breaking space replaced with regular space."""
        text = "hello\u00a0world"
        result = GatewayStreamConsumer._sanitize_for_telegram(text)
        assert "\u00a0" not in result
        # Non-breaking space is replaced (not stripped) to preserve word boundary
        assert "hello world" in result

    # Braille patterns ─────────────────────────────────────────────────────

    def test_braille_patterns_stripped(self):
        """Unicode braille patterns U+2800–U+28FF removed."""
        # ⠀ = U+2800, ⠿ = U+28FF
        text = "before ⠀⠂⠃⠄ after"
        result = GatewayStreamConsumer._sanitize_for_telegram(text)
        assert "⠀" not in result
        assert "before" in result
        assert "after" in result

    # Combined / edge cases ────────────────────────────────────────────────

    def test_clean_text_unchanged(self):
        """Text with no problematic characters is returned as-is."""
        text = "Hello, this is a normal response with numbers 123 and punctuation!"
        result = GatewayStreamConsumer._sanitize_for_telegram(text)
        assert result == text

    def test_empty_string_unchanged(self):
        """Empty string returns empty string."""
        result = GatewayStreamConsumer._sanitize_for_telegram("")
        assert result == ""

    def test_only_glyphs_become_whitespace(self):
        """Text that becomes only glyphs returns empty-ish string."""
        text = "████ ░░░ ▓▓"
        result = GatewayStreamConsumer._sanitize_for_telegram(text)
        assert result.strip() == ""

    def test_double_spaces_collapsed(self):
        """Stripping glyphs doesn't leave double spaces."""
        text = "Step 1 ──── Step 2"
        result = GatewayStreamConsumer._sanitize_for_telegram(text)
        assert "  " not in result
        assert "Step 1 Step 2" in result

    def test_newlines_preserved(self):
        """Newlines are preserved through sanitization."""
        text = "Line 1\nLine 2\nLine 3"
        result = GatewayStreamConsumer._sanitize_for_telegram(text)
        assert result == text

    def test_unicode_emoji_preserved(self):
        """Emoji outside the stripped ranges (👍 🚀) are preserved."""
        text = "Great work! 👍 🚀"
        result = GatewayStreamConsumer._sanitize_for_telegram(text)
        assert "👍" in result
        assert "🚀" in result

    def test_telegram_code_block_preserved(self):
        """Code fences with backticks are preserved (common in LLM output)."""
        text = "```python\nprint('hello')\n```"
        result = GatewayStreamConsumer._sanitize_for_telegram(text)
        assert "```" in result
        assert "print" in result

    def test_markdown_bold_italic_preserved(self):
        """Bold (**) and italic (*) markdown are preserved."""
        text = "This is **bold** and *italic* text."
        result = GatewayStreamConsumer._sanitize_for_telegram(text)
        assert "**bold**" in result
        assert "*italic*" in result

    def test_fast_path_for_clean_text(self):
        """When text is already clean the fast path returns it unchanged."""
        text = "No glyphs here — just normal ASCII text."
        # Fast path is triggered when text has no suspicious chars AND
        # cursor is one of the known-safe options
        result = GatewayStreamConsumer._sanitize_for_telegram(text)
        assert result == text
        # Also verify it is the exact same object (fast path identity)
        assert result is text

    def test_realistic_stream_with_glyphs(self):
        """Simulate a real LLM stream that includes progress bars."""
        text = (
            "Let me analyze this...\n"
            "███████████████████ 100%\n"
            "Found 42 items matching your query.\n"
            "▓▓▓▓▓▓░░░░ Loading more...\n"
            "Done!"
        )
        result = GatewayStreamConsumer._sanitize_for_telegram(text)
        assert "█" not in result
        assert "▓" not in result
        assert "░" not in result
        assert "Let me analyze this" in result
        assert "Found 42 items" in result
        assert "Done" in result


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
        """A tool boundary after edit failure must not duplicate the next segment.

        Architecture note: segment breaks are processed inside run()'s queue-drain loop.
        The sequence with cursor=" ▉" and edit_interval=0.08s is:

        1. run() starts; "Hello" buffered; edit_interval not elapsed → no send yet.
        2. "world" arrives; accumulated="Helloworld"; elapsed≈0.08 → sends "world".
           (first send = "Helloworld", not "Hello" alone — edit_interval was too short.)
        3. on_delta(None) queues _NEW_SEGMENT; run() processes it, should_edit=True
           → sends accumulated with cursor: "Helloworld▉".  Message ID reset.
        4. "Next segment" queued; finish() sends it as a new message.

        The key invariant this test verifies: no content from segment 1 ("Helloworld")
        appears in segment 2 ("Next segment").  The cursor gets sanitized to "..."
        by _sanitize_for_telegram, so the segment-break send becomes "Helloworld...".
        """
        adapter = MagicMock()
        send_results = [
            SimpleNamespace(success=True, message_id="msg_1"),
            SimpleNamespace(success=True, message_id="msg_2"),
            SimpleNamespace(success=True, message_id="msg_3"),
        ]
        adapter.send = AsyncMock(side_effect=send_results)
        adapter.edit_message = AsyncMock(
            return_value=SimpleNamespace(success=False, error="flood_control:6")
        )
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
        # Segment 1 content ("Helloworld...") must not reappear in segment 2.
        # The second send starts a fresh message — no duplication of segment 1.
        assert "Next segment" in sent_texts[-1]
        assert "Hello" not in sent_texts[-1]
        assert "world" not in sent_texts[-1]

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

        prefix = "abc"
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

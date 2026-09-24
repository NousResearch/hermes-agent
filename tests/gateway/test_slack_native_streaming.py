"""Tests: SlackAdapter native streaming (chat.startStream/appendStream/stopStream).

Behaviour contract:
  * supports_draft_streaming: True when connected with default unfurl behavior;
    False after a cached feature-gate failure, when disconnected, or when an
    explicit unfurl control requires the chat.postMessage fallback.
  * send_draft first frame: chat_startStream with thread_ts + initial text;
    returns the stream ts as message_id.
  * send_draft subsequent frames: chat_appendStream with only the delta;
    trailing cursor glyph stripped before delta computation.
  * identical frame: no API call, success.
  * prefix mismatch: stream sealed, frame fails (consumer falls back to edits).
  * send() finalization: active stream sealed via chat_stopStream with the
    remaining delta instead of chat_postMessage (no duplicate message).
  * send() with unrelated content: stream left open, normal post proceeds.
  * startStream feature-gate error: caches _native_stream_unsupported so
    future supports_draft_streaming() returns False.
  * disconnect(): dangling streams sealed.

Duplicate-reply invariant:
  * A successfully streamed answer is NEVER posted a second time as a fresh
    message — not when the agent's final differs from the streamed frames
    only by surrounding whitespace (``final_response.strip()`` /
    ``rstrip() + footer``), and not when chat.stopStream fails after the
    whole answer is already visible (the final is then committed in place
    via chat.update).
  * A genuinely uncommittable stream (stopStream AND chat.update fail) still
    falls back to a fresh post so the answer is not lost.
  * Interim sends (``_interim_send`` / ``expect_edits``) never seal a stream.
  * Streams are keyed per (team, channel, thread): two threads in one channel
    never seal each other's stream.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
from plugins.platforms.slack.adapter import SlackAdapter


def _make_adapter(extra=None):
    config = PlatformConfig(enabled=True, token="xoxb-fake", extra=extra or {})
    a = SlackAdapter(config)
    a._app = MagicMock()
    client = AsyncMock()
    client.chat_postMessage = AsyncMock(return_value={"ts": "999.111"})
    client.chat_update = AsyncMock(return_value={"ts": "999.111"})
    client.chat_startStream = AsyncMock(return_value={"ok": True, "ts": "123.456"})
    client.chat_appendStream = AsyncMock(return_value={"ok": True})
    client.chat_stopStream = AsyncMock(return_value={"ok": True})
    a._get_client = MagicMock(return_value=client)
    a.stop_typing = AsyncMock()
    a._running = True
    return a, client


def _open_streams(adapter, chat_id="D1"):
    """Stream entries currently open for ``chat_id`` (any thread/team)."""
    return [s for k, s in adapter._active_streams.items() if k[1] == chat_id]


META = {"thread_id": "111.000", "user_id": "U123"}
META_B = {"thread_id": "222.000", "user_id": "U123"}


class TestSupportsDraftStreaming:
    def test_supported_when_connected(self):
        adapter, _ = _make_adapter()
        assert adapter.supports_draft_streaming(chat_type="dm") is True

    @pytest.mark.parametrize(
        ("unfurl_key", "configured_value"),
        [
            ("unfurl_links", False),
            ("unfurl_links", True),
            ("unfurl_media", False),
            ("unfurl_media", True),
        ],
    )
    def test_explicit_unfurl_control_disables_native_streaming(
        self, unfurl_key, configured_value
    ):
        adapter, _ = _make_adapter({unfurl_key: configured_value})

        assert adapter.supports_draft_streaming(chat_type="dm") is False

    def test_unsupported_when_disconnected(self):
        adapter, _ = _make_adapter()
        adapter._app = None
        assert adapter.supports_draft_streaming() is False



class TestSendDraft:
    @pytest.mark.asyncio
    async def test_first_frame_starts_stream(self):
        adapter, client = _make_adapter()
        result = await adapter.send_draft("D1", 7, "Hello wo", metadata=META)
        assert result.success
        assert result.message_id == "123.456"
        kwargs = client.chat_startStream.await_args.kwargs
        assert kwargs["channel"] == "D1"
        assert kwargs["thread_ts"] == "111.000"
        assert kwargs["markdown_text"] == "Hello wo"
        assert kwargs["recipient_user_id"] == "U123"
        client.chat_appendStream.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_subsequent_frame_appends_delta_only(self):
        adapter, client = _make_adapter()
        await adapter.send_draft("D1", 7, "Hello wo", metadata=META)
        result = await adapter.send_draft("D1", 7, "Hello world!", metadata=META)
        assert result.success
        kwargs = client.chat_appendStream.await_args.kwargs
        assert kwargs["markdown_text"] == "rld!"
        assert kwargs["ts"] == "123.456"

    @pytest.mark.asyncio
    async def test_cursor_glyph_stripped(self):
        adapter, client = _make_adapter()
        await adapter.send_draft("D1", 7, "Hello ▉", metadata=META)
        assert client.chat_startStream.await_args.kwargs["markdown_text"] == "Hello"
        await adapter.send_draft("D1", 7, "Hello world ▉", metadata=META)
        assert client.chat_appendStream.await_args.kwargs["markdown_text"] == " world"

    @pytest.mark.asyncio
    async def test_identical_frame_is_noop(self):
        adapter, client = _make_adapter()
        await adapter.send_draft("D1", 7, "Hello", metadata=META)
        result = await adapter.send_draft("D1", 7, "Hello ▉", metadata=META)
        assert result.success
        client.chat_appendStream.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_prefix_mismatch_seals_and_fails(self):
        adapter, client = _make_adapter()
        await adapter.send_draft("D1", 7, "Hello", metadata=META)
        result = await adapter.send_draft("D1", 7, "Rewritten text", metadata=META)
        assert not result.success
        client.chat_stopStream.assert_awaited()
        assert not _open_streams(adapter)

    @pytest.mark.asyncio
    async def test_no_thread_ts_fails_cleanly(self):
        adapter, client = _make_adapter()
        result = await adapter.send_draft("D1", 7, "Hello", metadata={})
        assert not result.success
        client.chat_startStream.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_new_draft_id_seals_prior_stream(self):
        adapter, client = _make_adapter()
        await adapter.send_draft("D1", 7, "Segment one", metadata=META)
        client.chat_startStream.return_value = {"ok": True, "ts": "124.000"}
        result = await adapter.send_draft("D1", 8, "Segment two", metadata=META)
        assert result.success
        client.chat_stopStream.assert_awaited()  # sealed segment one
        (stream,) = _open_streams(adapter)
        assert stream["ts"] == "124.000"

    @pytest.mark.asyncio
    async def test_streams_are_keyed_per_thread(self):
        """Two threads in one channel: frames for B must not seal A."""
        adapter, client = _make_adapter()
        await adapter.send_draft("C1", 7, "Thread A answer", metadata=META)
        client.chat_startStream.return_value = {"ok": True, "ts": "456.000"}
        await adapter.send_draft("C1", 8, "Thread B answer", metadata=META_B)
        client.chat_stopStream.assert_not_awaited()
        assert {s["ts"] for s in _open_streams(adapter, "C1")} == {"123.456", "456.000"}


class TestFeatureGateFallback:
    @pytest.mark.asyncio
    async def test_not_allowed_caches_unsupported(self):
        adapter, client = _make_adapter()
        client.chat_startStream = AsyncMock(
            side_effect=Exception("The request to the Slack API failed. (not_allowed)")
        )
        result = await adapter.send_draft("D1", 7, "Hello", metadata=META)
        assert not result.success
        assert adapter._native_stream_unsupported is True
        assert adapter.supports_draft_streaming() is False


class TestStreamRelation:
    """Pure classification of the turn-final against the streamed text."""

    def test_exact_prefix_is_raw_delta(self):
        assert SlackAdapter._stream_relation("Hello", "Hello world") == ("extends", " world")
        assert SlackAdapter._stream_relation("Hello", "Hello") == ("equal", "")

    def test_surrounding_whitespace_is_tolerated(self):
        assert SlackAdapter._stream_relation("\n\nHello world\n", "Hello world") == ("equal", "")
        assert SlackAdapter._stream_relation("Hello world", "\n\nHello world\n")[0] == "equal"

    def test_delta_is_sliced_from_raw_final(self):
        # Streamed frame ended with a trailing space+newline; the agent
        # rstrip()s before appending the footer. The delta must be the exact
        # raw tail of the final, whitespace preserved.
        kind, delta = SlackAdapter._stream_relation("Answer \n", "Answer\n\n-# footer")
        assert kind == "extends"
        assert delta == "\n\n-# footer"

    def test_crlf_and_fence_closure_deltas_are_exact(self):
        kind, delta = SlackAdapter._stream_relation("line1\r\n", "line1\r\nline2")
        assert (kind, delta) == ("extends", "line2")
        kind, delta = SlackAdapter._stream_relation("```py\nx = 1", "```py\nx = 1\n```")
        assert (kind, delta) == ("extends", "\n```")

    def test_unrelated_and_empty_stream(self):
        assert SlackAdapter._stream_relation("Streaming text", "Unrelated notice") == ("unrelated", "")
        assert SlackAdapter._stream_relation("   ", "Hello")[0] == "unrelated"
        assert SlackAdapter._stream_relation("", "Hello")[0] == "unrelated"


class TestSendFinalization:
    @pytest.mark.asyncio
    async def test_final_send_seals_stream_no_duplicate_post(self):
        adapter, client = _make_adapter()
        await adapter.send_draft("D1", 7, "Hello wo", metadata=META)
        result = await adapter.send("D1", "Hello world, done.", metadata=META)
        assert result.success
        assert result.message_id == "123.456"
        kwargs = client.chat_stopStream.await_args.kwargs
        assert kwargs["markdown_text"] == "rld, done."
        client.chat_postMessage.assert_not_awaited()
        assert not _open_streams(adapter)

    @pytest.mark.asyncio
    async def test_final_send_equal_content_seals_without_delta(self):
        """A: streamed == final → one Slack message only."""
        adapter, client = _make_adapter()
        await adapter.send_draft("D1", 7, "Hello world", metadata=META)
        result = await adapter.send("D1", "Hello world", metadata=META)
        assert result.success
        kwargs = client.chat_stopStream.await_args.kwargs
        assert "markdown_text" not in kwargs
        client.chat_postMessage.assert_not_awaited()
        client.chat_update.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_whitespace_only_difference_does_not_duplicate(self):
        """B: the agent strips final_response; the streamed frames were not."""
        adapter, client = _make_adapter()
        await adapter.send_draft("D1", 7, "\n\nHello world\n", metadata=META)
        result = await adapter.send("D1", "Hello world", metadata=META)
        assert result.success
        assert result.message_id == "123.456"
        assert client.chat_stopStream.await_count == 1
        assert "markdown_text" not in client.chat_stopStream.await_args.kwargs
        client.chat_postMessage.assert_not_awaited()
        assert not _open_streams(adapter)

    @pytest.mark.asyncio
    async def test_footer_after_rstrip_appends_exact_tail(self):
        """B: ``final.rstrip() + "\\n\\n" + footer`` vs a streamed trailing space."""
        adapter, client = _make_adapter()
        await adapter.send_draft("D1", 7, "Answer \n", metadata=META)
        result = await adapter.send("D1", "Answer\n\n-# footer", metadata=META)
        assert result.success
        assert client.chat_stopStream.await_args.kwargs["markdown_text"] == "\n\n-# footer"
        client.chat_postMessage.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_unrelated_send_passes_through(self):
        adapter, client = _make_adapter()
        await adapter.send_draft("D1", 7, "Streaming text here", metadata=META)
        result = await adapter.send("D1", "Unrelated notice", metadata=META)
        assert result.success
        client.chat_postMessage.assert_awaited()
        # Stream stays open for its own finalization.
        assert _open_streams(adapter)

    @pytest.mark.asyncio
    async def test_stop_stream_failure_with_full_answer_visible_commits_in_place(self):
        """C: stopStream fails after the whole answer streamed → no fresh post."""
        adapter, client = _make_adapter()
        await adapter.send_draft("D1", 7, "Hello world", metadata=META)
        client.chat_stopStream = AsyncMock(side_effect=Exception("boom"))
        result = await adapter.send("D1", "Hello world", metadata=META)
        assert result.success
        assert result.message_id == "123.456"
        assert client.chat_stopStream.await_count == 2  # one bounded retry
        client.chat_update.assert_awaited_once()
        assert client.chat_update.await_args.kwargs["ts"] == "123.456"
        client.chat_postMessage.assert_not_awaited()
        assert not _open_streams(adapter)

    @pytest.mark.asyncio
    async def test_stop_and_update_both_fail_falls_back_to_fresh_post(self):
        """C2/D: an uncommittable stream still delivers the answer (loss-safe)."""
        adapter, client = _make_adapter()
        await adapter.send_draft("D1", 7, "Hello", metadata=META)
        client.chat_stopStream = AsyncMock(side_effect=Exception("boom"))
        client.chat_update = AsyncMock(side_effect=Exception("update boom"))
        result = await adapter.send("D1", "Hello world", metadata=META)
        assert result.success
        client.chat_postMessage.assert_awaited_once()
        assert client.chat_postMessage.await_args.kwargs["text"] == "Hello world"
        # A stop that carries a tail is never retried: ``markdown_text`` APPENDS.
        assert client.chat_stopStream.await_count == 1
        assert client.chat_update.await_count == 1
        assert not _open_streams(adapter)

    @pytest.mark.asyncio
    async def test_an_ambiguous_stop_failure_never_appends_the_tail_twice(self):
        """stopStream landed server-side but raised here (timeout on the response): a retry with
        the same ``markdown_text`` would show "Hello world world". The final is committed by the
        idempotent in-place update instead."""
        adapter, client = _make_adapter()
        await adapter.send_draft("D1", 7, "Hello", metadata=META)
        client.chat_stopStream = AsyncMock(side_effect=Exception("read timeout"))
        result = await adapter.send("D1", "Hello world", metadata=META)
        assert result.success
        tails = [c.kwargs.get("markdown_text") for c in client.chat_stopStream.await_args_list]
        assert tails == [" world"]
        assert client.chat_update.await_args.kwargs["text"] == "Hello world"
        client.chat_postMessage.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_partial_stream_stop_failure_commits_full_final_in_place(self):
        """D: partial stream, stopStream fails → final committed via update."""
        adapter, client = _make_adapter()
        await adapter.send_draft("D1", 7, "Hello", metadata=META)
        client.chat_stopStream = AsyncMock(side_effect=Exception("boom"))
        result = await adapter.send("D1", "Hello world", metadata=META)
        assert result.success
        assert client.chat_update.await_args.kwargs["text"] == "Hello world"
        client.chat_postMessage.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_persistent_stop_failure_is_bounded(self):
        """H: stopStream raising on every call → ≤2 stop calls, one output."""
        adapter, client = _make_adapter()
        await adapter.send_draft("D1", 7, "Hello world", metadata=META)
        client.chat_stopStream = AsyncMock(side_effect=Exception("always"))
        client.chat_update = AsyncMock(side_effect=Exception("always"))
        await adapter.send("D1", "Hello world", metadata=META)
        assert client.chat_stopStream.await_count == 2
        assert client.chat_update.await_count == 1
        assert client.chat_postMessage.await_count == 1

    @pytest.mark.asyncio
    async def test_whitespace_only_stream_is_unusable(self):
        """E: a stream with no substance never claims the final."""
        adapter, client = _make_adapter()
        await adapter.send_draft("D1", 7, "   ", metadata=META)
        result = await adapter.send("D1", "Hello world", metadata=META)
        assert result.success
        client.chat_postMessage.assert_awaited_once()
        client.chat_stopStream.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_non_streaming_send_unchanged(self):
        """F: no active stream → plain post, no stream API calls."""
        adapter, client = _make_adapter()
        result = await adapter.send("D1", "Hello world", metadata=META)
        assert result.success
        client.chat_postMessage.assert_awaited_once()
        client.chat_stopStream.assert_not_awaited()
        client.chat_update.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_interim_send_never_seals_even_when_equal(self):
        adapter, client = _make_adapter()
        await adapter.send_draft("D1", 7, "Hello world", metadata=META)
        interim = dict(META, _interim_send=True)
        result = await adapter.send("D1", "Hello world", metadata=interim)
        assert result.success
        client.chat_stopStream.assert_not_awaited()
        client.chat_postMessage.assert_awaited_once()
        assert _open_streams(adapter)
        # The real final still seals exactly once, with no extra post.
        result = await adapter.send("D1", "Hello world", metadata=META)
        assert result.message_id == "123.456"
        assert client.chat_stopStream.await_count == 1
        assert client.chat_postMessage.await_count == 1

    @pytest.mark.asyncio
    async def test_expect_edits_preview_never_seals(self):
        adapter, client = _make_adapter()
        await adapter.send_draft("D1", 7, "Hello world", metadata=META)
        preview = dict(META, expect_edits=True)
        await adapter.send("D1", "Hello world", metadata=preview)
        client.chat_stopStream.assert_not_awaited()
        assert _open_streams(adapter)

    @pytest.mark.asyncio
    async def test_two_threads_finalize_their_own_streams(self):
        adapter, client = _make_adapter()
        await adapter.send_draft("C1", 7, "Thread A answer", metadata=META)
        client.chat_startStream.return_value = {"ok": True, "ts": "456.000"}
        await adapter.send_draft("C1", 8, "Thread B answer", metadata=META_B)
        rb = await adapter.send("C1", "Thread B answer", metadata=META_B)
        assert rb.message_id == "456.000"
        assert client.chat_stopStream.await_args.kwargs["ts"] == "456.000"
        assert [s["ts"] for s in _open_streams(adapter, "C1")] == ["123.456"]
        ra = await adapter.send("C1", "Thread A answer", metadata=META)
        assert ra.message_id == "123.456"
        assert client.chat_stopStream.await_count == 2
        client.chat_postMessage.assert_not_awaited()
        assert not _open_streams(adapter, "C1")

    @pytest.mark.asyncio
    async def test_team_id_routes_stream_calls(self):
        adapter, client = _make_adapter()
        meta = dict(META, slack_team_id="T999")
        await adapter.send_draft("C1", 7, "Hello", metadata=meta)
        client.chat_stopStream = AsyncMock(side_effect=Exception("boom"))
        await adapter.send("C1", "Hello world", metadata=meta)
        teams = {c.kwargs.get("team_id") for c in adapter._get_client.call_args_list}
        assert teams == {"T999"}
        client.chat_postMessage.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_oversized_tail_uses_normal_split_path(self):
        adapter, client = _make_adapter()
        await adapter.send_draft("D1", 7, "Intro", metadata=META)
        tail = "x" * (adapter.MAX_MESSAGE_LENGTH + 10)
        result = await adapter.send("D1", "Intro" + tail, metadata=META)
        assert result.success
        # Stream closed on what was visible (no oversized append), then the
        # normal split path delivers the full final; nothing dangles.
        client.chat_stopStream.assert_awaited_once()
        assert "markdown_text" not in client.chat_stopStream.await_args.kwargs
        assert client.chat_postMessage.await_count >= 1
        assert not _open_streams(adapter)

    @pytest.mark.asyncio
    async def test_rewritten_turn_final_replaces_stream_in_place(self):
        """A mrkdwn-rewritten turn-final (notify=True) replaces the sealed stream; no 2nd post."""
        adapter, client = _make_adapter()
        await adapter.send_draft("D1", 7, "*Done:* all good", metadata=META)
        result = await adapter.send("D1", "_Done:_ all good", metadata=dict(META, notify=True))
        assert result.success
        client.chat_stopStream.assert_awaited_once()
        client.chat_update.assert_awaited()
        assert client.chat_update.await_args.kwargs["ts"] == result.message_id
        client.chat_postMessage.assert_not_awaited()
        assert not _open_streams(adapter)

    @pytest.mark.asyncio
    async def test_rewritten_turn_final_posts_when_update_fails(self):
        adapter, client = _make_adapter()
        await adapter.send_draft("D1", 7, "Draft answer that got rewritten", metadata=META)
        client.chat_update = AsyncMock(side_effect=Exception("update failed"))
        result = await adapter.send("D1", "Completely new answer", metadata=dict(META, notify=True))
        assert result.success
        client.chat_postMessage.assert_awaited_once()
        assert not _open_streams(adapter)

    @pytest.mark.asyncio
    async def test_explicit_team_metadata_fills_recipient_team(self):
        adapter, client = _make_adapter()
        await adapter.send_draft("C1", 7, "Hello", metadata=dict(META, slack_team_id="T999"))
        assert client.chat_startStream.await_args.kwargs["recipient_team_id"] == "T999"

    @pytest.mark.asyncio
    async def test_thread_status_cleared_with_metadata(self):
        adapter, client = _make_adapter()
        await adapter.send_draft("D1", 7, "Hello", metadata=META)
        await adapter.send("D1", "Hello world", metadata=META)
        adapter.stop_typing.assert_awaited_once_with("D1", META)


RICH_MD = "# Title\n\nbody text with **bold**\n\n| a | b |\n|---|---|\n| 1 | 2 |"


class TestRichBlocksAfterSeal:
    """G: with ``rich_blocks`` the sealed stream gets its layout via chat.update, once."""

    @pytest.mark.asyncio
    async def test_rich_final_seals_then_renders_blocks_once(self):
        adapter, client = _make_adapter({"rich_blocks": True})
        body = f"Intro.\n\n{RICH_MD}\n"
        await adapter.send_draft("D1", 7, body, metadata=META)
        result = await adapter.send("D1", body.strip(), metadata=META)
        assert result.success and result.message_id == "123.456"
        assert client.chat_stopStream.await_count == 1
        client.chat_postMessage.assert_not_awaited()
        client.chat_update.assert_awaited_once()
        kwargs = client.chat_update.await_args.kwargs
        assert kwargs["ts"] == "123.456"
        assert kwargs["blocks"]
        assert "header" in [b["type"] for b in kwargs["blocks"]]

    @pytest.mark.asyncio
    async def test_block_rejection_retries_without_blocks_no_post(self):
        adapter, client = _make_adapter({"rich_blocks": True})
        await adapter.send_draft("D1", 7, RICH_MD, metadata=META)
        client.chat_update = AsyncMock(
            side_effect=[Exception("invalid_blocks"), {"ok": True}]
        )
        result = await adapter.send("D1", RICH_MD, metadata=META)
        assert result.success
        assert client.chat_update.await_count == 2
        assert client.chat_update.await_args.kwargs["blocks"] == []
        client.chat_postMessage.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_block_update_failure_after_seal_keeps_markdown(self):
        adapter, client = _make_adapter({"rich_blocks": True})
        await adapter.send_draft("D1", 7, RICH_MD, metadata=META)
        client.chat_update = AsyncMock(side_effect=Exception("ratelimited"))
        result = await adapter.send("D1", RICH_MD, metadata=META)
        assert result.success and result.message_id == "123.456"
        client.chat_postMessage.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_rich_blocks_applied_after_seal(self):
        adapter, client = _make_adapter({"rich_blocks": True})
        rich = "# Title\n\nbody text"
        await adapter.send_draft("D1", 7, rich[:5], metadata=META)
        result = await adapter.send("D1", rich, metadata=META)
        assert result.success
        client.chat_update.assert_awaited()
        assert client.chat_update.await_args.kwargs["blocks"]

    @pytest.mark.asyncio
    async def test_plain_final_without_rich_blocks_never_updates(self):
        adapter, client = _make_adapter()
        await adapter.send_draft("D1", 7, RICH_MD[:8], metadata=META)
        result = await adapter.send("D1", RICH_MD, metadata=META)
        assert result.success and result.message_id == "123.456"
        client.chat_update.assert_not_awaited()
        client.chat_postMessage.assert_not_awaited()


class TestEndToEndConsumer:
    """GatewayStreamConsumer → real SlackAdapter (mocked Slack client)."""

    async def _run(self, adapter, deltas, final, *, segment_break_after=None):
        cfg = StreamConsumerConfig(
            transport="auto", chat_type="dm", edit_interval=0.01, buffer_threshold=1,
        )
        consumer = GatewayStreamConsumer(adapter, "D1", cfg, metadata=dict(META))
        task = asyncio.create_task(consumer.run())
        for i, d in enumerate(deltas):
            consumer.on_delta(d)
            await asyncio.sleep(0.05)
            if segment_break_after is not None and i == segment_break_after:
                consumer.on_segment_break()
                await asyncio.sleep(0.05)
        consumer.finish(final)
        await task
        return consumer

    @pytest.mark.asyncio
    async def test_stripped_final_is_one_message(self):
        adapter, client = _make_adapter()
        consumer = await self._run(adapter, ["\n\nHello ", "world\n"], "Hello world")
        assert client.chat_startStream.await_count == 1
        assert client.chat_stopStream.await_count == 1
        client.chat_postMessage.assert_not_awaited()
        assert not _open_streams(adapter)
        assert consumer.final_response_sent
        assert consumer.delivered_final_matches("Hello world") is True

    @pytest.mark.asyncio
    async def test_tool_boundary_yields_two_sealed_streams_no_post(self):
        adapter, client = _make_adapter()
        client.chat_startStream = AsyncMock(
            side_effect=[{"ok": True, "ts": "1.0"}, {"ok": True, "ts": "2.0"}]
        )
        await self._run(
            adapter, ["First segment ", "here.", "Second segment."],
            "Second segment.", segment_break_after=1,
        )
        assert client.chat_startStream.await_count == 2
        assert client.chat_stopStream.await_count == 2
        client.chat_postMessage.assert_not_awaited()
        assert not _open_streams(adapter)


class TestDisconnectCleanup:
    @pytest.mark.asyncio
    async def test_disconnect_seals_dangling_streams(self):
        adapter, client = _make_adapter()
        await adapter.send_draft("D1", 7, "Dangling", metadata=META)
        adapter._stop_socket_mode_handler = AsyncMock()
        adapter._release_platform_lock = MagicMock()
        await adapter.disconnect()
        client.chat_stopStream.assert_awaited()
        assert client.chat_stopStream.await_args.kwargs["channel"] == "D1"
        assert not adapter._active_streams

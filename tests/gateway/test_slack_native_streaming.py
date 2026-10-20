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


class _StreamExpiredError(Exception):
    """slack_sdk.SlackApiError's shape (``exc.response["error"]``) without importing the SDK,
    which CI stubs as a bare module. The adapter only reads the response mapping."""

    def __init__(self, message, response):
        super().__init__(message)
        self.response = response


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
        await adapter.send_draft("D1", 7, "Hello \u2589", metadata=META)
        assert client.chat_startStream.await_args.kwargs["markdown_text"] == "Hello"
        await adapter.send_draft("D1", 7, "Hello world \u2589", metadata=META)
        assert client.chat_appendStream.await_args.kwargs["markdown_text"] == " world"

    @pytest.mark.asyncio
    async def test_identical_frame_is_noop(self):
        adapter, client = _make_adapter()
        await adapter.send_draft("D1", 7, "Hello", metadata=META)
        result = await adapter.send_draft("D1", 7, "Hello \u2589", metadata=META)
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
    async def test_expired_stream_reopens_seeded_with_only_the_unsent_tail(self):
        """Slack seals a native draft stream server-side after a few minutes of a
        long turn — the same seal the native task-card stream hits (see
        _slack_error_is's other caller). The next chat.appendStream fails with
        message_not_in_streaming_state; the lane must not permanently disable
        draft streaming for the run (#_send_draft_frame's "any failure
        permanently disables drafts"): drop the dead ts and start a fresh stream
        in the same thread seeded with ONLY the text past the sealed message (the
        prefix is already visible there), while later deltas still resume correctly."""
        adapter, client = _make_adapter()
        await adapter.send_draft("D1", 7, "Hello wo", metadata=META)

        client.chat_appendStream = AsyncMock(
            side_effect=_StreamExpiredError("expired", {"ok": False, "error": "message_not_in_streaming_state"})
        )
        client.chat_startStream = AsyncMock(return_value={"ok": True, "ts": "124.000"})

        result = await adapter.send_draft("D1", 7, "Hello world!", metadata=META)

        assert result.success
        assert result.message_id == "124.000"
        kwargs = client.chat_startStream.await_args.kwargs
        assert kwargs["markdown_text"] == "rld!"  # sealed message already shows "Hello wo"
        (reopened,) = _open_streams(adapter)  # same per-thread key, dead ts replaced
        assert reopened["ts"] == "124.000"
        assert reopened["sent"] == "Hello world!"  # full segment: deltas diff against it
        assert reopened["base"] == len("Hello wo")
        assert kwargs["thread_ts"] == META["thread_id"]
        assert adapter._native_stream_unsupported is False  # not the feature-gate path

        # A later frame resumes as a normal delta against the reopened stream.
        client.chat_appendStream = AsyncMock(return_value={"ok": True})
        result2 = await adapter.send_draft("D1", 7, "Hello world! More.", metadata=META)
        assert result2.success
        assert client.chat_appendStream.await_args.kwargs["markdown_text"] == " More."


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
    async def test_unrelated_send_passes_through(self):
        adapter, client = _make_adapter()
        await adapter.send_draft("D1", 7, "Streaming text here", metadata=META)
        result = await adapter.send("D1", "Unrelated notice", metadata=META)
        assert result.success
        client.chat_postMessage.assert_awaited()
        # Stream stays open for its own finalization.
        assert _open_streams(adapter)

    @pytest.mark.asyncio
    async def test_mid_turn_notify_reply_leaves_stream_open_and_posts_fresh(self):
        """/status, /approve and clarify answers are notify sends in the SAME thread; they
        must not overwrite the half-streamed answer in place."""
        adapter, client = _make_adapter()
        await adapter.send_draft("D1", 7, "Partial answer being streamed", metadata=META)
        result = await adapter.send("D1", "Status: running, 3 tools", metadata={**META, "notify": True})
        assert result.message_id == "999.111"
        client.chat_postMessage.assert_awaited_once()
        client.chat_stopStream.assert_not_awaited()
        client.chat_update.assert_not_awaited()
        (stream,) = _open_streams(adapter)
        assert stream["sent"] == "Partial answer being streamed"


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
        await adapter.send_draft("D1", 7, "*Draft:* answer that got restyled", metadata=META)
        client.chat_update = AsyncMock(side_effect=Exception("update failed"))
        result = await adapter.send("D1", "_Draft:_ answer that got restyled", metadata=dict(META, notify=True))
        assert result.success
        client.chat_postMessage.assert_awaited_once()
        assert not _open_streams(adapter)


RICH_MD = "# Title\n\nbody text with **bold**\n\n| a | b |\n|---|---|\n| 1 | 2 |"


class TestRichBlocksAfterSeal:
    """G: with ``rich_blocks`` the sealed stream gets its layout via chat.update, once."""


    @pytest.mark.asyncio
    async def test_rich_blocks_applied_after_seal(self):
        adapter, client = _make_adapter({"rich_blocks": True})
        rich = "# Title\n\nbody text"
        await adapter.send_draft("D1", 7, rich[:5], metadata=META)
        result = await adapter.send("D1", rich, metadata=META)
        assert result.success
        client.chat_update.assert_awaited()
        assert client.chat_update.await_args.kwargs["blocks"]


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

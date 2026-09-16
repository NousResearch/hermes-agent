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
"""

import asyncio
import copy
import json
import queue
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.run_turn_runner import TurnRunner
from gateway.session import SessionSource
from gateway.turn_context import TurnContext
from plugins.platforms.slack.adapter import SlackAdapter
from tests.gateway.test_run_progress_topics import _make_runner


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


META = {"thread_id": "111.000", "user_id": "U123"}


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

    def test_unsupported_after_feature_gate_failure(self):
        adapter, _ = _make_adapter()
        adapter._native_stream_unsupported = True
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
        assert "D1" not in adapter._active_streams

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
        assert adapter._active_streams["D1"]["ts"] == "124.000"


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

    @pytest.mark.asyncio
    async def test_transient_error_does_not_cache(self):
        adapter, client = _make_adapter()
        client.chat_startStream = AsyncMock(side_effect=Exception("timeout"))
        result = await adapter.send_draft("D1", 7, "Hello", metadata=META)
        assert not result.success
        assert adapter._native_stream_unsupported is False


class TestSendFinalization:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("oversized", [False, True], ids=["small", "oversized"])
    async def test_full_progress_leaves_answer_stream_open(self, oversized):
        adapter, client = _make_adapter()
        metadata = {**META, "slack_team_id": "T123", "caller_value": {"keep": True}}
        original_metadata = copy.deepcopy(metadata)
        progress_attempted = asyncio.Event()
        post_ids = []

        def post_message(**kwargs):
            post_ids.append(f"999.{len(post_ids) + 1:03d}")
            progress_attempted.set()
            return {"ok": True, "ts": post_ids[-1]}

        def stop_stream(**kwargs):
            progress_attempted.set()  # Wake RED on the erroneous early seal too.
            return {"ok": True}

        client.chat_postMessage.side_effect = post_message
        client.chat_stopStream.side_effect = stop_stream
        ctx = TurnContext(
            source=SessionSource(platform=Platform.SLACK, chat_id="D1", chat_type="dm"),
            _run_still_current=lambda: True,
            progress_mode="full", tool_progress_enabled=True,
            progress_queue=queue.Queue(), _progress_metadata=metadata,
            _cleanup_progress=True,
        )
        runner = TurnRunner(_make_runner(adapter), ctx)
        # An acknowledged, nonempty prefix also matches the unknown tool's emoji.
        draft = await adapter.send_draft("D1", 7, "⚙️", metadata=metadata)
        assert draft.success and draft.message_id == "123.456"
        client.chat_startStream.assert_awaited_once()
        arguments = {"value": "x" * (adapter.MAX_MESSAGE_LENGTH + 1000 if oversized else 10)}
        runner.progress_callback("tool.started", "full_native_slack", args=arguments)
        sender = asyncio.create_task(runner.send_progress_messages())
        try:
            await asyncio.wait_for(progress_attempted.wait(), timeout=2)
        finally:
            sender.cancel()
            await asyncio.wait_for(sender, timeout=2)

        client.chat_stopStream.assert_not_awaited()
        posts = [call.kwargs for call in client.chat_postMessage.await_args_list]
        assert len(posts) > 1 if oversized else len(posts) == 1
        header, _, body = "".join(post["text"] for post in posts).partition("\n")
        assert header == "⚙️ full_native_slack"
        assert json.loads(body) == arguments
        assert all(post["channel"] == "D1" and post["thread_ts"] == META["thread_id"] for post in posts)
        assert ctx._cleanup_msg_ids == post_ids
        assert draft.message_id not in ctx._cleanup_msg_ids
        assert ctx._progress_metadata is metadata
        assert metadata == original_metadata
        client.chat_update.assert_not_awaited()

        extended = await adapter.send_draft("D1", 7, "⚙️ Answer continues", metadata=metadata)
        assert extended.success and extended.message_id == draft.message_id
        client.chat_startStream.assert_awaited_once()
        client.chat_appendStream.assert_awaited_once_with(
            channel="D1", ts=draft.message_id, markdown_text=" Answer continues",
        )
        client.chat_stopStream.assert_not_awaited()
        final = await adapter.send("D1", "⚙️ Answer continues. Done.", metadata=metadata)
        assert final.success and final.message_id == draft.message_id
        client.chat_stopStream.assert_awaited_once_with(
            channel="D1", ts=draft.message_id, markdown_text=". Done.",
        )
        assert client.chat_postMessage.await_count == len(posts)
        assert "D1" not in adapter._active_streams
        assert metadata == original_metadata
        assert all("_interim_send" not in call.kwargs for call in client.mock_calls)

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
        assert "D1" not in adapter._active_streams

    @pytest.mark.asyncio
    async def test_final_send_equal_content_seals_without_delta(self):
        adapter, client = _make_adapter()
        await adapter.send_draft("D1", 7, "Hello world", metadata=META)
        result = await adapter.send("D1", "Hello world", metadata=META)
        assert result.success
        kwargs = client.chat_stopStream.await_args.kwargs
        assert "markdown_text" not in kwargs
        client.chat_postMessage.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "metadata",
        [None, {}, META, {**META, "_interim_send": True}, {**META, "_interim_send": False}],
    )
    async def test_unrelated_send_passes_through(self, metadata):
        adapter, client = _make_adapter()
        original_metadata = copy.deepcopy(metadata)
        await adapter.send_draft("D1", 7, "Streaming text here", metadata=META)
        result = await adapter.send("D1", "Unrelated notice", metadata=metadata)
        assert result.success
        client.chat_postMessage.assert_awaited()
        assert metadata == original_metadata
        assert "_interim_send" not in client.chat_postMessage.await_args.kwargs
        # Stream stays open for its own finalization.
        assert "D1" in adapter._active_streams

    @pytest.mark.asyncio
    async def test_stop_stream_failure_falls_back_to_post(self):
        adapter, client = _make_adapter()
        await adapter.send_draft("D1", 7, "Hello", metadata=META)
        client.chat_stopStream = AsyncMock(side_effect=Exception("boom"))
        result = await adapter.send("D1", "Hello world", metadata=META)
        assert result.success
        client.chat_postMessage.assert_awaited()

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
        assert not adapter._active_streams

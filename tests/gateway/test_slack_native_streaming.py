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

from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
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
    async def test_unrelated_send_passes_through(self):
        adapter, client = _make_adapter()
        await adapter.send_draft("D1", 7, "Streaming text here", metadata=META)
        result = await adapter.send("D1", "Unrelated notice", metadata=META)
        assert result.success
        client.chat_postMessage.assert_awaited()
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("chat_id", "thread_id", "dm_threads", "expected_thread"),
    [
        ("D1", None, False, None),
        ("D1", "111.000", False, "111.000"),
        ("D1", "222.000", True, "222.000"),
        ("C1", "111.000", False, "111.000"),
        ("C1", "222.000", False, "222.000"),
    ],
)
async def test_flat_dm_session_setting_controls_progress_and_final_delivery(
    chat_id, thread_id, dm_threads, expected_thread,
):
    from gateway.config import Platform
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource

    adapter, client = _make_adapter({
        "dm_top_level_threads_as_sessions": dm_threads, "reply_in_thread": True,
    })
    runner = object.__new__(GatewayRunner)
    runner._adapter_for_source = lambda source: adapter
    source = SessionSource(
        platform=Platform.SLACK, chat_id=chat_id,
        chat_type="dm" if chat_id.startswith("D") else "group",
        thread_id=thread_id, user_id="U123",
    )
    progress_metadata, progress_reply_to, status_metadata = runner._run_agent_progress_threading(
        source, "222.000", False,
    )
    assert (status_metadata or {}).get("thread_id") == expected_thread
    progress = await adapter.send(chat_id, "Working", progress_reply_to, progress_metadata)
    final = await adapter.send(
        chat_id, "Done", "222.000", runner._thread_metadata_for_source(source, "222.000"),
    )
    assert progress.success and final.success
    assert [call.kwargs.get("thread_ts") for call in client.chat_postMessage.await_args_list] == [
        expected_thread, expected_thread,
    ]
    draft = await adapter.send_draft(chat_id, 42, "Draft", metadata=status_metadata)
    if expected_thread is None:
        assert not draft.success  # The consumer uses flat post/edit streaming instead.
        client.chat_startStream.assert_not_awaited()
    else:
        assert draft.success
        assert client.chat_startStream.await_args.kwargs["thread_ts"] == expected_thread

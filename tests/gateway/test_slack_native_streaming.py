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


class TestFullProgressLiteral:
    """Real callback, queue, Slack adapter and SDK; only HTTP transport is mocked."""

    @staticmethod
    def make_native(extra=None, metadata=None):
        from types import SimpleNamespace
        from slack_sdk.web.async_client import AsyncWebClient

        adapter = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-fake", extra=extra or {}))
        client = AsyncWebClient(token="xoxb-fake")
        adapter._app = SimpleNamespace(client=client)
        adapter._team_clients["T123"] = client
        adapter._running = True
        calls = []
        changed = asyncio.Event()

        async def transport(method, **kwargs):
            payload = copy.deepcopy(kwargs.get("json") or kwargs.get("params") or {})
            calls.append((method, payload))
            changed.set()
            posts = sum(name == "chat.postMessage" for name, _ in calls)
            return {"ok": True, "ts": "123.456" if method == "chat.startStream" else f"999.{110 + posts}"}

        client.api_call = AsyncMock(side_effect=transport)
        ctx = TurnContext(
            source=SessionSource(platform=Platform.SLACK, chat_id="D1", chat_type="dm"),
            _run_still_current=lambda: True,
            progress_mode="full", tool_progress_enabled=True,
            progress_queue=queue.Queue(), _progress_metadata=metadata,
            _progress_reply_to=META["thread_id"], _cleanup_progress=True,
        )
        return adapter, TurnRunner(_make_runner(adapter), ctx), ctx, calls, changed

    @staticmethod
    async def wait_for_call(calls, changed, method, count=1):
        async def wait():
            while len([item for item in calls if item[0] == method]) < count:
                changed.clear()
                await changed.wait()
        await asyncio.wait_for(wait(), timeout=4)


    @pytest.mark.asyncio
    @pytest.mark.parametrize("delta", [-1, 0, 1, 42000])
    @pytest.mark.parametrize("seed", [False, True], ids=["send", "edit-rollover"])
    async def test_literal_payload_size_and_lossless_split(self, delta, seed):
        from html import escape, unescape

        adapter, runner, ctx, calls, changed = self.make_native(metadata=None)
        limit = adapter.MAX_MESSAGE_LENGTH - 64  # Preserve the native message cap and gateway reserve.
        arguments = {"body": "[Guide](https://example.test/guide) **b** _u_ `x` ``` & <@U123> 😀", "pad": ""}
        prefix = "⚙️ boundary\n"
        seed_text = "⚙️ seed\n{}\n" if seed else ""
        overhead = len(escape(seed_text + prefix + json.dumps(arguments, ensure_ascii=False), quote=False))
        budget = limit + delta - overhead
        arguments["pad"] = "&" * (budget // 5) + "x" * (budget % 5)
        original = copy.deepcopy(arguments)
        entry = prefix + json.dumps(arguments, ensure_ascii=False)
        runner.progress_callback("tool.started", "seed" if seed else "boundary", args={} if seed else arguments)
        sender = asyncio.create_task(runner.send_progress_messages())
        try:
            await self.wait_for_call(calls, changed, "chat.postMessage")
            if seed:
                runner.progress_callback("tool.started", "boundary", args=arguments)
                # Let the existing queue completion drain handle the final entry.
        finally:
            sender.cancel()
            await asyncio.wait_for(sender, timeout=4)

        latest = {}
        order = []
        for method, payload in calls:
            if method not in {"chat.postMessage", "chat.update"}:
                continue
            self.assert_plain_payload(payload)
            assert len(payload["text"]) <= limit
            mid = payload["ts"] if method == "chat.update" else f"999.{111 + len(order)}"
            if method == "chat.postMessage":
                order.append(mid)
            latest[mid] = unescape(payload["text"])
        delivered = [latest[mid] for mid in order]
        if seed and delta <= 0:
            assert delivered == [seed_text + entry]
            assert any(method == "chat.update" for method, _ in calls)
        else:
            if seed:
                assert delivered.pop(0) == seed_text.rstrip("\n")
            # Independent per-character oracle: no trimming, labels or fence repair.
            expected = []
            current = []
            size = 0
            for char in entry:
                width = len(escape(char, quote=False))
                if size + width > limit:
                    expected.append("".join(current))
                    current, size = [], 0
                current.append(char)
                size += width
            expected.append("".join(current))
            assert delivered == expected
        assert arguments == original

    @pytest.mark.asyncio
    @pytest.mark.parametrize("operation", ["send", "edit"])
    async def test_literal_block_rejection_never_falls_back_to_markdown(self, operation):
        adapter, _, _, calls, _ = self.make_native({"rich_blocks": True})
        assert adapter._app is not None
        client = adapter._app.client
        client.api_call.side_effect = Exception("invalid_blocks")
        metadata = {**META, "_literal_text": True, "_interim_send": True}
        if operation == "send":
            result = await adapter.send("D1", "**data** <@U123>", metadata=metadata)
        else:
            result = await adapter.edit_message("D1", "999.111", "**data** <@U123>", finalize=True, metadata=metadata)
        assert not result.success
        attempts = [call for call in client.api_call.await_args_list if call.args[0] in {"chat.postMessage", "chat.update"}]
        assert len(attempts) == 1
        assert attempts[0].kwargs["json"]["blocks"][0]["text"]["type"] == "plain_text"

    @staticmethod
    def arguments():
        return {
            "[key](https://example.test/key) **bold** _under_ `code` <tag>":
                "[Guide](https://example.test/guide) **Important** _italic_ `inline` ```fence```",
            "angles": "<@U123> <!channel> <!here> <!everyone> <#C123> <https://example.test|label>",
            "entities": "<tag> & &lt; &amp;lt; \\\\literal\\n 😀",
            "image": "![photo](https://example.test/photo.png)",
            "api_key": "full-literal-secret-sentinel",
            "nested": [{"password": "nested-secret-sentinel", "keep": "**readable**"}],
        }

    @staticmethod
    def assert_plain_payload(payload):
        # Slack documents plain_text for both postMessage and chat.update.
        # Its top-level mrkdwn flag is documented only for postMessage.
        assert payload["parse"] == "none"
        assert payload["link_names"] in (False, 0)
        from html import unescape

        blocks = payload["blocks"]
        assert 1 <= len(blocks) <= 50
        assert "".join(unescape(block["text"]["text"]) for block in blocks) == unescape(payload["text"])
        assert "".join(block["text"]["text"] for block in blocks) == payload["text"]
        for block in blocks:
            assert block["type"] == "section"
            assert block["text"]["type"] == "plain_text"
            assert block["text"]["emoji"] is False
            assert len(block["text"]["text"]) <= 3000
        assert "<" not in payload["text"] and ">" not in payload["text"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("operation", ["send", "edit"])
    @pytest.mark.parametrize("extra", [{}, {"rich_blocks": True, "unfurl_links": True, "unfurl_media": True}])
    async def test_markdown_json_round_trips(self, operation, extra):
        from html import unescape

        metadata = {**META, "slack_team_id": "T123", "caller_value": {"keep": True}}
        original_metadata = copy.deepcopy(metadata)
        adapter, runner, ctx, calls, changed = self.make_native(extra, metadata)
        arguments = self.arguments()
        original = copy.deepcopy(arguments)
        expected = copy.deepcopy(arguments)
        expected["api_key"] = "***"
        expected["nested"][0]["password"] = "***"
        if operation == "edit":
            runner.progress_callback("tool.started", "seed", args={})
        else:
            runner.progress_callback("tool.started", "literal_probe", args=arguments)
        sender = asyncio.create_task(runner.send_progress_messages())
        try:
            await self.wait_for_call(calls, changed, "chat.postMessage")
            if operation == "edit":
                runner.progress_callback("tool.started", "literal_probe", args=arguments)
                await self.wait_for_call(calls, changed, "chat.update")
        finally:
            sender.cancel()
            await asyncio.wait_for(sender, timeout=4)

        method = "chat.postMessage" if operation == "send" else "chat.update"
        payload = [payload for name, payload in calls if name == method][-1]
        body = unescape(payload["text"]).split("⚙️ literal_probe\n", 1)[1]
        assert json.loads(body) == expected
        self.assert_plain_payload(payload)
        assert "full-literal-secret-sentinel" not in str(calls)
        assert "nested-secret-sentinel" not in str(calls)
        assert arguments == original
        assert ctx._progress_metadata is metadata and metadata == original_metadata
        assert ctx._cleanup_msg_ids == ["999.111"]
        for name, posted in calls:
            assert "_literal_text" not in posted and "_interim_send" not in posted
            if name == "chat.postMessage":
                assert posted["mrkdwn"] is False
                assert posted["unfurl_links"] is False and posted["unfurl_media"] is False
                assert posted["thread_ts"] == META["thread_id"]

    @pytest.mark.asyncio
    async def test_full_progress_preserves_answer_stream(self):
        adapter, runner, ctx, calls, changed = self.make_native(metadata=META)
        draft = await adapter.send_draft("D1", 7, "⚙️", metadata=META)
        assert draft.success
        runner.progress_callback("tool.started", "literal_probe", args={"body": "**literal**"})
        sender = asyncio.create_task(runner.send_progress_messages())
        try:
            # Wait for either native outcome, so an erroneous seal is a behavioral failure.
            async def wait():
                while not any(name in {"chat.postMessage", "chat.stopStream"} for name, _ in calls):
                    changed.clear()
                    await changed.wait()
            await asyncio.wait_for(wait(), timeout=4)
        finally:
            sender.cancel()
            await asyncio.wait_for(sender, timeout=4)
        assert not any(name == "chat.stopStream" for name, _ in calls)
        assert draft.message_id not in ctx._cleanup_msg_ids
        assert (await adapter.send_draft("D1", 7, "⚙️ Answer", metadata=META)).success
        assert (await adapter.send("D1", "⚙️ Answer done", metadata=META)).success
        assert [p["markdown_text"] for n, p in calls if n == "chat.appendStream"] == [" Answer"]
        assert [p["markdown_text"] for n, p in calls if n == "chat.stopStream"] == [" done"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("metadata", [None, META, {**META, "_interim_send": True}])
    async def test_ordinary_send_edit_keep_markdown(self, metadata):
        adapter, _, _, calls, _ = self.make_native()
        original = copy.deepcopy(metadata)
        content = "[Guide](https://example.test/guide) **Important**"
        assert (await adapter.send("D1", content, metadata=metadata)).success
        assert (await adapter.edit_message("D1", "999.111", content, metadata=metadata)).success
        post, update = [p for n, p in calls if n in {"chat.postMessage", "chat.update"}]
        assert post["text"] == update["text"] == "<https://example.test/guide|Guide> *Important*"
        assert post["mrkdwn"] is True
        assert not post.get("blocks") and not update.get("blocks")
        assert metadata == original


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

"""Tests for the Weixin platform adapter."""

import asyncio
import base64
import json
import os
from unittest.mock import AsyncMock, Mock, patch

import pytest

from gateway.config import PlatformConfig
from gateway.config import GatewayConfig, HomeChannel, Platform, _apply_env_overrides
from gateway.platforms.base import SendResult
from gateway.platforms.base import MessageEvent, MessageType
from gateway.platforms import weixin
from gateway.platforms.weixin import ContextTokenStore, WeixinAdapter
from tools.send_message_tool import _parse_target_ref, _send_to_platform


def _make_adapter() -> WeixinAdapter:
    return WeixinAdapter(
        PlatformConfig(
            enabled=True,
            token="test-token",
            extra={"account_id": "test-account"},
        )
    )


class TestWeixinInboundVoiceTranscript:
    def test_voice_transcript_keeps_voice_origin_marker(self):
        item_list = [
            {
                "type": weixin.ITEM_VOICE,
                "voice_item": {"text": "帮我查一下今天天气"},
            }
        ]

        assert weixin._extract_text(item_list) == (
            "[Voice transcription provided by Weixin]\n"
            "帮我查一下今天天气"
        )


class TestWeixinFormatting:

    def test_format_message_preserves_markdown_tables(self):
        adapter = _make_adapter()

        content = (
            "| Setting | Value |\n"
            "| --- | --- |\n"
            "| Timeout | 30s |\n"
            "| Retries | 3 |\n"
        )

        assert adapter.format_message(content) == content.strip()


    def test_format_message_wraps_long_plain_lines_for_copying(self):
        adapter = _make_adapter()

        content = (
            "Here is a long issue template line with many copyable fields "
            + " ".join(f"field_{idx}=value_{idx}" for idx in range(24))
        )

        formatted = adapter.format_message(content)

        assert "\n" in formatted
        assert all(len(line) <= weixin.WEIXIN_COPY_LINE_WIDTH for line in formatted.splitlines())
        assert " ".join(formatted.split()) == " ".join(content.split())


class TestWeixinChunking:


    def test_split_text_keeps_four_line_structured_blocks_together(self):
        adapter = _make_adapter()

        content = adapter.format_message(
            "今天结论：\n"
            "- 留存下降 3%\n"
            "- 转化上涨 8%\n"
            "- 主要问题在首日激活"
        )
        chunks = adapter._split_text(content)

        assert chunks == ["今天结论：\n- 留存下降 3%\n- 转化上涨 8%\n- 主要问题在首日激活"]


    def test_split_text_keeps_complete_code_block_together_when_possible(self):
        adapter = _make_adapter()
        adapter.MAX_MESSAGE_LENGTH = 80

        content = adapter.format_message(
            "## Intro\n\nShort paragraph.\n\n```python\nprint('hello world')\nprint('again')\n```\n\nTail paragraph."
        )
        chunks = adapter._split_text(content)

        assert len(chunks) >= 2
        assert any(
            "```python\nprint('hello world')\nprint('again')\n```" in chunk
            for chunk in chunks
        )
        assert all(chunk.count("```") % 2 == 0 for chunk in chunks)


    def test_split_text_can_restore_legacy_multiline_splitting_via_config(self):
        adapter = WeixinAdapter(
            PlatformConfig(
                enabled=True,
                extra={
                    "account_id": "acct",
                    "token": "***",
                    "split_multiline_messages": True,
                },
            )
        )

        content = adapter.format_message("第一行\n第二行\n第三行")
        chunks = adapter._split_text(content)

        assert chunks == ["第一行", "第二行", "第三行"]


class TestWeixinConfig:

    def test_get_connected_platforms_includes_weixin_with_token(self):
        config = GatewayConfig(
            platforms={
                Platform.WEIXIN: PlatformConfig(
                    enabled=True,
                    token="bot-token",
                    extra={"account_id": "bot-account"},
                )
            }
        )

        assert config.get_connected_platforms() == [Platform.WEIXIN]


class TestWeixinStatePersistence:
    def test_save_weixin_account_preserves_existing_file_on_replace_failure(self, tmp_path, monkeypatch):
        account_path = tmp_path / "weixin" / "accounts" / "acct.json"
        account_path.parent.mkdir(parents=True, exist_ok=True)
        original = {"token": "old-token", "base_url": "https://old.example.com"}
        account_path.write_text(json.dumps(original), encoding="utf-8")

        def _boom(_src, _dst):
            raise OSError("disk full")

        monkeypatch.setattr("utils.os.replace", _boom)

        try:
            weixin.save_weixin_account(
                str(tmp_path),
                account_id="acct",
                token="new-token",
                base_url="https://new.example.com",
                user_id="wxid_new",
            )
        except OSError:
            pass
        else:
            raise AssertionError("expected save_weixin_account to propagate replace failure")

        assert json.loads(account_path.read_text(encoding="utf-8")) == original

    @pytest.mark.asyncio
    async def test_context_token_persist_runs_off_event_loop_thread(self, tmp_path):
        """atomic_json_write() calls os.fsync(), which blocks until the write
        reaches stable storage. ContextTokenStore.set() runs on the event
        loop for every inbound message carrying a context_token
        (_process_message), so the persist step must be offloaded to a
        thread — mirrors test_directory_write_runs_off_event_loop_thread in
        test_channel_directory.py for the same #83906 bug class."""
        import threading

        store = ContextTokenStore(str(tmp_path))
        loop_thread = threading.get_ident()
        write_threads = []

        def fake_write(path, data, *args, **kwargs):
            write_threads.append(threading.get_ident())

        with patch("gateway.platforms.weixin.atomic_json_write", side_effect=fake_write):
            await store.set("acct-1", "user-1", "ctx-token-abc")

        assert store.get("acct-1", "user-1") == "ctx-token-abc"
        assert write_threads
        assert all(tid != loop_thread for tid in write_threads)


class TestWeixinQrLogin:
    @pytest.mark.asyncio
    async def test_qr_login_timeout_uses_monotonic_clock(self, tmp_path):
        first_qr = {
            "qrcode": "qr-1",
            "qrcode_img_content": "https://example.com/qr-1",
        }
        pending = {"status": "wait"}

        with patch("gateway.platforms.weixin._api_get", new_callable=AsyncMock) as api_get_mock, \
             patch("gateway.platforms.weixin.time") as mock_time, \
             patch("gateway.platforms.weixin.AIOHTTP_AVAILABLE", True), \
             patch("gateway.platforms.weixin.aiohttp.ClientSession", create=True) as session_cls, \
             patch("builtins.print"):
            api_get_mock.side_effect = [first_qr, pending]
            mock_time.monotonic.side_effect = [1000, 1000.2, 1001.1]
            mock_time.time.side_effect = [1000, 900, 901, 902]

            session = AsyncMock()
            session.__aenter__.return_value = session
            session.__aexit__.return_value = False
            session_cls.return_value = session

            result = await weixin.qr_login(str(tmp_path), timeout_seconds=1)

        assert result is None
        assert api_get_mock.await_count == 2


class TestWeixinSendMessageIntegration:
    def test_parse_target_ref_accepts_weixin_ids(self):
        assert _parse_target_ref("weixin", "wxid_test123") == ("wxid_test123", None, True)
        assert _parse_target_ref("weixin", "filehelper") == ("filehelper", None, True)
        assert _parse_target_ref("weixin", "group@chatroom") == ("group@chatroom", None, True)


class TestWeixinChunkDelivery:
    def _connected_adapter(self) -> WeixinAdapter:
        adapter = _make_adapter()
        adapter._session = object()
        adapter._send_session = adapter._session
        adapter._token = "test-token"
        adapter._base_url = "https://weixin.example.com"
        adapter._token_store.get = lambda account_id, chat_id: "ctx-token"
        return adapter


    @patch("gateway.platforms.weixin.asyncio.sleep", new_callable=AsyncMock)
    @patch("gateway.platforms.weixin._send_message", new_callable=AsyncMock)
    def test_send_retries_failed_chunk_before_continuing(self, send_message_mock, sleep_mock):
        adapter = self._connected_adapter()
        adapter.MAX_MESSAGE_LENGTH = 12
        calls = {"count": 0}

        async def flaky_send(*args, **kwargs):
            calls["count"] += 1
            if calls["count"] == 2:
                raise RuntimeError("temporary iLink failure")

        send_message_mock.side_effect = flaky_send

        # Use double newlines so _pack_markdown_blocks splits into 3 blocks
        result = asyncio.run(adapter.send("wxid_test123", "first\n\nsecond\n\nthird"))

        assert result.success is True
        # 3 chunks, but chunk 2 fails once and retries → 4 _send_message calls total
        assert send_message_mock.await_count == 4
        # The retried chunk should reuse the same client_id for deduplication
        first_try = send_message_mock.await_args_list[1].kwargs
        retry = send_message_mock.await_args_list[2].kwargs
        assert first_try["text"] == retry["text"]
        assert first_try["client_id"] == retry["client_id"]

    @patch("gateway.platforms.weixin.asyncio.sleep", new_callable=AsyncMock)
    @patch("gateway.platforms.weixin._send_message", new_callable=AsyncMock)
    def test_repeated_rate_limits_open_circuit_for_followup_sends(self, send_message_mock, sleep_mock):
        adapter = self._connected_adapter()
        adapter._send_chunk_retries = 3
        adapter._send_chunk_retry_delay_seconds = 0
        adapter._rate_limit_circuit_threshold = 2
        adapter._rate_limit_circuit_window_seconds = 60
        adapter._rate_limit_circuit_open_seconds = 60

        send_message_mock.return_value = {
            "ret": weixin.RATE_LIMIT_ERRCODE,
            "errcode": weixin.RATE_LIMIT_ERRCODE,
            "errmsg": "frequency limit",
        }

        first = asyncio.run(adapter.send("wxid_test123", "first"))
        second = asyncio.run(adapter.send("wxid_test123", "second"))

        assert first.success is False
        assert "cooldown" in (first.error or "")
        assert second.success is False
        assert "cooldown" in (second.error or "")
        # The first rate-limit response is retried once. The second response
        # crosses the sliding-window threshold, opens the breaker, and both the
        # rest of the current chunk and follow-up sends fail fast.
        assert send_message_mock.await_count == 2
        assert sleep_mock.await_count == 1


class TestWeixinOutboundMedia:


    def test_send_file_uses_post_for_upload_full_url_and_hex_encoded_aes_key(self, tmp_path):
        class _UploadResponse:
            def __init__(self):
                self.status = 200
                self.headers = {"x-encrypted-param": "enc-param"}

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def read(self):
                return b""

            async def text(self):
                return ""

        class _RecordingSession:
            def __init__(self):
                self.post_calls = []

            def post(self, url, **kwargs):
                self.post_calls.append((url, kwargs))
                return _UploadResponse()

            def put(self, *_args, **_kwargs):
                raise AssertionError("upload_full_url branch should use POST")

        image_path = tmp_path / "demo.png"
        image_path.write_bytes(b"fake-png-bytes")

        adapter = _make_adapter()
        session = _RecordingSession()
        adapter._session = session
        adapter._send_session = session
        adapter._token = "test-token"
        adapter._base_url = "https://weixin.example.com"
        adapter._cdn_base_url = "https://cdn.example.com/c2c"
        adapter._token_store.get = lambda account_id, chat_id: None

        aes_key = bytes(range(16))
        expected_aes_key = base64.b64encode(aes_key.hex().encode("ascii")).decode("ascii")

        with patch("gateway.platforms.weixin._get_upload_url", new=AsyncMock(return_value={"upload_full_url": "https://upload.example.com/media"})), \
             patch("gateway.platforms.weixin._api_post", new_callable=AsyncMock) as api_post_mock, \
             patch("gateway.platforms.weixin.secrets.token_hex", return_value="filekey-123"), \
             patch("gateway.platforms.weixin.secrets.token_bytes", return_value=aes_key):
            message_id = asyncio.run(adapter._send_file("wxid_test123", str(image_path), ""))

        assert message_id.startswith("hermes-weixin-")
        assert len(session.post_calls) == 1
        upload_url, upload_kwargs = session.post_calls[0]
        assert upload_url == "https://upload.example.com/media"
        assert upload_kwargs["headers"] == {"Content-Type": "application/octet-stream"}
        assert upload_kwargs["data"]
        # Timeout is now enforced externally via asyncio.wait_for() rather than
        # aiohttp.ClientTimeout, so it no longer appears as a post() kwarg.
        assert "timeout" not in upload_kwargs
        payload = api_post_mock.await_args.kwargs["payload"]
        media = payload["msg"]["item_list"][0]["image_item"]["media"]
        assert media["encrypt_query_param"] == "enc-param"
        assert media["aes_key"] == expected_aes_key


class TestWeixinRemoteMediaSafety:
    def test_download_remote_media_blocks_unsafe_urls(self):
        adapter = _make_adapter()

        with patch("tools.url_safety.is_safe_url", return_value=False):
            try:
                asyncio.run(adapter._download_remote_media("http://127.0.0.1/private.png"))
            except ValueError as exc:
                assert "Blocked unsafe URL" in str(exc)
            else:
                raise AssertionError("expected ValueError for unsafe URL")


class TestWeixinMarkdownLinks:
    """Markdown links should be preserved so WeChat can render them natively."""


    def test_format_message_preserves_links_inside_code_blocks(self):
        adapter = _make_adapter()

        content = "See below:\n\n```\n[link](https://example.com)\n```\n\nDone."
        result = adapter.format_message(content)
        assert "[link](https://example.com)" in result


class TestWeixinBlankMessagePrevention:
    """Regression tests for the blank-bubble bugs.

    Three separate guards now prevent a blank WeChat message from ever being
    dispatched:

    1. ``_split_text_for_weixin_delivery("")`` returns ``[]`` — not ``[""]``.
    2. ``send()`` filters out empty/whitespace-only chunks before calling
       ``_send_text_chunk``.
    3. ``_send_message()`` raises ``ValueError`` for empty text as a last-resort
       safety net.
    """


    def test_split_text_returns_empty_list_for_empty_string_split_per_line(self):
        adapter = WeixinAdapter(
            PlatformConfig(
                enabled=True,
                extra={
                    "account_id": "acct",
                    "token": "test-tok",
                    "split_multiline_messages": True,
                },
            )
        )
        assert adapter._split_text("") == []


class TestWeixinStreamingCursorSuppression:
    """WeChat doesn't support message editing — cursor must be suppressed."""

    def test_supports_message_editing_is_false(self):
        adapter = _make_adapter()
        assert adapter.SUPPORTS_MESSAGE_EDITING is False


class TestWeixinMediaBuilder:
    """Media builder uses base64(hex_key), not base64(raw_bytes) for aes_key."""


    def test_voice_builder_for_audio_files_uses_file_attachment_type(self):
        adapter = _make_adapter()
        media_type, builder = adapter._outbound_media_builder("note.mp3")
        assert media_type == weixin.MEDIA_FILE

        item = builder(
            encrypt_query_param="eq",
            aes_key_for_api="fakekey",
            ciphertext_size=512,
            plaintext_size=500,
            filename="note.mp3",
            rawfilemd5="abc",
        )
        assert item["type"] == weixin.ITEM_FILE
        assert item["file_item"]["file_name"] == "note.mp3"


class TestWeixinSendImageFileParameterName:
    """Regression test for send_image_file parameter name mismatch.

    The gateway calls send_image_file(chat_id=..., image_path=...) but the
    WeixinAdapter previously used 'path' as the parameter name, causing
    image sending to fail. This test ensures the interface stays correct.
    """

    @patch.object(WeixinAdapter, "send_document", new_callable=AsyncMock)
    def test_send_image_file_uses_image_path_parameter(self, send_document_mock):
        """Verify send_image_file accepts image_path and forwards to send_document."""
        adapter = _make_adapter()
        adapter._session = object()
        adapter._send_session = adapter._session
        adapter._token = "test-token"

        send_document_mock.return_value = weixin.SendResult(success=True, message_id="test-id")

        # This is the call pattern used by gateway/run.py extract_media
        result = asyncio.run(
            adapter.send_image_file(
                chat_id="wxid_test123",
                image_path="/tmp/test_image.png",
                caption="Test caption",
                metadata={"thread_id": "thread-123"},
            )
        )

        assert result.success is True
        send_document_mock.assert_awaited_once_with(
            chat_id="wxid_test123",
            file_path="/tmp/test_image.png",
            caption="Test caption",
            metadata={"thread_id": "thread-123"},
        )


class TestWeixinVoiceSending:
    def _connected_adapter(self) -> WeixinAdapter:
        adapter = _make_adapter()
        adapter._session = object()
        adapter._send_session = adapter._session
        adapter._token = "test-token"
        adapter._base_url = "https://weixin.example.com"
        adapter._token_store.get = lambda account_id, chat_id: "ctx-token"
        return adapter


    @patch.object(weixin, "_api_post", new_callable=AsyncMock)
    @patch.object(weixin, "_upload_ciphertext", new_callable=AsyncMock)
    @patch.object(weixin, "_get_upload_url", new_callable=AsyncMock)
    def test_send_file_sets_voice_metadata_for_silk_payload(
        self,
        get_upload_url_mock,
        upload_ciphertext_mock,
        api_post_mock,
        tmp_path,
    ):
        adapter = self._connected_adapter()
        silk = tmp_path / "voice.silk"
        silk.write_bytes(b"\x02#!SILK_V3\x01\x00")
        get_upload_url_mock.return_value = {"upload_full_url": "https://cdn.example.com/upload"}
        upload_ciphertext_mock.return_value = "enc-q"
        api_post_mock.return_value = {"success": True}

        asyncio.run(adapter._send_file("wxid_test123", str(silk), ""))

        payload = api_post_mock.await_args.kwargs["payload"]
        voice_item = payload["msg"]["item_list"][0]["voice_item"]
        assert voice_item.get("playtime", 0) == 0
        assert voice_item["encode_type"] == 6
        assert voice_item["sample_rate"] == 24000
        assert voice_item["bits_per_sample"] == 16


class TestIsStaleSessionRet:
    """Regression test for #17228: distinguish stale-session ret=-2 from rate-limit ret=-2."""


    def test_ret_minus_2_with_freq_limit_is_not_stale(self):
        # Genuine rate limit — must NOT be treated as stale session.
        assert weixin._is_stale_session_ret(-2, None, "freq limit") is False


    def test_errcode_minus_14_is_not_matched_here(self):
        # -14 is handled by the separate SESSION_EXPIRED_ERRCODE path; the
        # helper only disambiguates -2 from a genuine rate limit.
        assert weixin._is_stale_session_ret(-14, None, "session expired") is False


class TestWeixinContentDedup:
    """Regression tests for Issue #16182 — upstream API sends duplicate content
    with different message_ids, bypassing message_id deduplication.
    """

    def test_duplicate_content_with_different_message_ids_is_dropped(self):
        adapter = _make_adapter()
        adapter._poll_session = object()
        adapter.handle_message = AsyncMock()
        # Tighten the text-debounce delay so the flush completes quickly.
        adapter._text_batch_delay_seconds = 0.05
        adapter._text_batch_split_delay_seconds = 0.05

        base_msg = {
            "from_user_id": "wxid_user1",
            "item_list": [{"type": 1, "text_item": {"text": "hello world"}}],
        }

        async def _drive():
            # Both inbound messages share the same event loop so the debounce
            # task created by the first one survives to be flushed.
            await adapter._process_message({**base_msg, "message_id": "msg-1"})
            await adapter._process_message({**base_msg, "message_id": "msg-2"})
            # Wait out the quiet period so the buffered text batch flushes.
            await asyncio.sleep(0.2)

        asyncio.run(_drive())

        # Content-dedup drops the second (duplicate) message before it is even
        # enqueued, so only one combined dispatch reaches handle_message.
        assert adapter.handle_message.await_count == 1
        event = adapter.handle_message.await_args[0][0]
        assert event.text == "hello world"


class TestWeixinTextDebounce:
    """Text-debounce batching for rapid multi-message bursts (issue #35301).

    Delays are read from ``config.extra`` (config.yaml), not env vars.
    """


    def test_batch_delays_overridden_via_config_extra(self):
        adapter = WeixinAdapter(
            PlatformConfig(
                enabled=True,
                token="test-token",
                extra={
                    "account_id": "test-account",
                    "text_batch_delay_seconds": "0.5",
                    "text_batch_split_delay_seconds": 1.5,
                },
            )
        )
        assert adapter._text_batch_delay_seconds == 0.5
        assert adapter._text_batch_split_delay_seconds == 1.5


class _StubResponse:
    def __init__(self, *, status=200, body="{}", delay=0.0):
        self.status = status
        self.ok = 200 <= status < 300
        self._body = body
        self._delay = delay

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    async def text(self):
        if self._delay:
            await asyncio.sleep(self._delay)
        return self._body


class _StubSession:
    """Records request kwargs and returns a configurable async-CM response.

    Unlike aiohttp.ClientSession it installs no TimerContext, so it cannot
    reproduce aiohttp's cross-loop crash directly; these tests instead pin the
    observable contract of the asyncio.wait_for migration.
    """

    def __init__(self, response):
        self._response = response
        self.post_calls = []
        self.get_calls = []

    def post(self, url, **kwargs):
        self.post_calls.append((url, kwargs))
        return self._response

    def get(self, url, **kwargs):
        self.get_calls.append((url, kwargs))
        return self._response


class TestWeixinApiTimeout:
    def test_api_post_does_not_pass_aiohttp_timeout_kwarg(self):
        session = _StubSession(_StubResponse(body='{"ret": 0}'))
        result = asyncio.run(
            weixin._api_post(
                session,
                base_url="https://weixin.example.com",
                endpoint="ep",
                payload={"k": "v"},
                token="tok",
                timeout_ms=5000,
            )
        )
        assert result == {"ret": 0}
        # The fix enforces the timeout via asyncio.wait_for, so ClientTimeout is
        # gone and `timeout` is no longer forwarded to session.post().
        [(_url, kwargs)] = session.post_calls
        assert "timeout" not in kwargs


    def test_get_updates_returns_empty_sentinel_on_timeout(self):
        # wait_for raises asyncio.TimeoutError, which _get_updates swallows into
        # an empty long-poll batch rather than propagating.
        session = _StubSession(_StubResponse(delay=1.0))
        result = asyncio.run(
            weixin._get_updates(
                session,
                base_url="https://weixin.example.com",
                token="tok",
                sync_buf="buf-123",
                timeout_ms=1,
            )
        )
        assert result == {"ret": 0, "msgs": [], "get_updates_buf": "buf-123"}


class TestWeixinVoiceAlwaysDownloaded:
    """Regression tests for #27300: when WeChat (Weixin) returns a
    ``voice_item.text`` (Tencent Cloud's STT) we must still download
    the raw audio and route it through Hermes' own STT pipeline.

    Non-Chinese users currently see garbled transcriptions because the
    existing code short-circuits in two places: ``_download_voice``
    returns ``None`` whenever Tencent provided *any* text (even
    incorrect), and ``_extract_text`` returns that text as the message
    body. The fix is to always download and never return Tencent's
    text — the central STT pipeline in ``gateway/run.py`` produces
    the actual body from the downloaded audio.
    """

    def _make_voice_item(self, text: str = "") -> dict:
        """Build a minimal voice item with media + optional Tencent text."""
        return {
            "type": weixin.ITEM_VOICE,
            "voice_item": {
                "text": text,
                "media": {
                    "encrypt_query_param": "q",
                    "aes_key": "a" * 32,
                    "full_url": "https://example.invalid/voice.silk",
                },
            },
        }

    @pytest.mark.asyncio
    async def test_download_voice_returns_path_when_tencent_text_set(self, tmp_path, monkeypatch):
        """#27300 PRIMARY: ``_download_voice`` must not short-circuit on
        ``voice_item.text``. The audio is needed so Hermes' own STT can
        re-transcribe when Tencent's text is in the wrong language.
        """
        adapter = _make_adapter()
        adapter._cdn_base_url = "https://example.invalid"
        adapter._poll_session = Mock()

        fake_audio_bytes = b"\\x00\\x01\\x02FAKE_SILK"
        monkeypatch.setattr(weixin, "cache_audio_from_bytes_async",
                            AsyncMock(side_effect=lambda data, ext: str(tmp_path / f"voice.{ext.lstrip('.')}")))

        async def _fake_download(session, *, cdn_base_url, encrypted_query_param,
                                 aes_key_b64, full_url, timeout_seconds):
            return fake_audio_bytes

        monkeypatch.setattr(weixin, "_download_and_decrypt_media", _fake_download)

        item = self._make_voice_item(text="garbled-tencent-transcript")
        result = await adapter._download_voice(item)

        # Currently broken: returns None when voice_item.text is set.
        # After fix: returns a local path so the central STT pipeline
        # can pick it up and re-transcribe.
        assert result is not None, (
            "_download_voice returned None even though raw audio is "
            "available — Hermes' STT pipeline needs the audio to handle "
            "non-Chinese voice messages (#27300)."
        )
        assert result.endswith(".silk")

    def test_extract_text_does_not_return_tencent_voice_text(self):
        """#27300 SECONDARY: ``_extract_text`` must not return
        ``voice_item.text`` verbatim. That text is Tencent Cloud's
        STT output, which is wrong for non-Chinese audio and the
        whole reason #27300 was filed. Returning empty forces the
        central STT pipeline's transcript to become the body.
        """
        item_list = [self._make_voice_item(text="garbled-tencent-transcript")]
        result = weixin._extract_text(item_list)
        # Currently broken: returns "garbled-tencent-transcript".
        # After fix: returns "" (empty string) so the central pipeline
        # transcript replaces it as the user-visible body.
        assert result != "garbled-tencent-transcript", (
            "_extract_text returned Tencent's text directly — for "
            "non-Chinese audio this is garbage; the central STT "
            "pipeline's transcript should be the body (#27300)."
        )


    @pytest.mark.asyncio
    async def test_collect_media_includes_voice_when_tencent_text_set(self, tmp_path, monkeypatch):
        """#27300 INTEGRATION: ``_collect_media`` should add a ``.silk``
        path to ``media_paths`` even when Tencent returned text, so the
        central STT pipeline can re-transcribe. Currently the
        short-circuit in ``_download_voice`` means the audio is never
        downloaded, and the message body is whatever Tencent wrote
        (garbled for non-Chinese audio).
        """
        adapter = _make_adapter()
        adapter._cdn_base_url = "https://example.invalid"
        adapter._poll_session = Mock()

        monkeypatch.setattr(weixin, "cache_audio_from_bytes_async",
                            AsyncMock(side_effect=lambda data, ext: str(tmp_path / f"voice.{ext.lstrip('.')}")))

        async def _fake_download(session, *, cdn_base_url, encrypted_query_param,
                                 aes_key_b64, full_url, timeout_seconds):
            return b"\\x00FAKE"

        monkeypatch.setattr(weixin, "_download_and_decrypt_media", _fake_download)

        media_paths: list = []
        media_types: list = []
        item = self._make_voice_item(text="какой-то текст")
        await adapter._collect_media(item, media_paths, media_types)

        assert len(media_paths) == 1, (
            "_collect_media dropped the voice attachment because "
            "voice_item.text was set — Hermes' STT never gets a "
            "chance to re-transcribe (#27300)."
        )
        assert media_types == ["audio/silk"]


class TestWeixinVoiceGatewayHandoff:
    """#27300 integration-level regression: the routing fix must not only
    download the audio and drop Tencent's text at the adapter level — the
    inbound voice item must surface as a VOICE ``MessageEvent`` carrying the
    ``audio/silk`` media, and that event must reach the runner's central STT
    pipeline (``_enrich_message_with_transcription``) instead of being trusted
    as already-transcribed text. This covers the gateway-runner handoff that the
    adapter-only tests above do not exercise.
    """

    def _inbound_voice_message(self, text: str) -> dict:
        return {
            "from_user_id": "user-123",
            "to_user_id": "test-account",
            "message_id": "msg-voice-1",
            "msg_type": 1,
            "item_list": [
                {
                    "type": weixin.ITEM_VOICE,
                    "voice_item": {
                        "text": text,
                        "media": {
                            "encrypt_query_param": "q",
                            "aes_key": "a" * 32,
                            "full_url": "https://example.invalid/voice.silk",
                        },
                    },
                }
            ],
        }


    @pytest.mark.asyncio
    async def test_voice_event_body_is_not_tencent_text(self, tmp_path, monkeypatch):
        """The VOICE event handed to the runner must NOT carry Tencent's STT
        text as its body — the central pipeline's transcript replaces it.
        """
        adapter = _make_adapter()
        adapter._poll_session = Mock()
        adapter._token = None
        adapter._cdn_base_url = "https://example.invalid"

        monkeypatch.setattr(weixin, "cache_audio_from_bytes_async",
                            AsyncMock(side_effect=lambda data, ext: str(tmp_path / f"voice.{ext.lstrip('.')}")))
        async def _fake_download(*a, **k):
            return b"\x00\x01FAKE_SILK"
        monkeypatch.setattr(weixin, "_download_and_decrypt_media", _fake_download)

        captured = {}

        async def _capture(event):
            captured["event"] = event

        adapter.handle_message = _capture

        tencent_text = "garbled English phonemes for a Russian voice"
        await adapter._process_message(self._inbound_voice_message(tencent_text))

        assert "event" in captured
        event = captured["event"]
        # The text field must be empty (Tencent text dropped) so the runner
        # has no pre-filled body and routes the audio to STT.
        assert event.text != tencent_text, (
            "VOICE event body leaked Tencent's STT text — runner would trust "
            "the wrong transcript instead of re-transcribing (#27300)."
        )


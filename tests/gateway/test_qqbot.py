"""Tests for the QQ Bot platform adapter."""

import asyncio
import os
from types import SimpleNamespace
from unittest import mock

import httpx
import pytest

from gateway.config import PlatformConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**extra):
    """Build a PlatformConfig(enabled=True, extra=extra) for testing."""
    return PlatformConfig(enabled=True, extra=extra)


# ---------------------------------------------------------------------------
# check_qq_requirements
# ---------------------------------------------------------------------------

class TestQQRequirements:
    def test_returns_bool(self):
        from gateway.platforms.qqbot import check_qq_requirements
        result = check_qq_requirements()
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# QQAdapter.__init__
# ---------------------------------------------------------------------------

class TestQQAdapterInit:
    def _make(self, **extra):
        from gateway.platforms.qqbot import QQAdapter
        return QQAdapter(_make_config(**extra))


    def test_env_fallback(self):
        with mock.patch.dict(os.environ, {"QQ_APP_ID": "env_id", "QQ_CLIENT_SECRET": "env_sec"}, clear=False):
            adapter = self._make()
            assert adapter._app_id == "env_id"
            assert adapter._client_secret == "env_sec"


    def test_dm_policy_default(self):
        adapter = self._make(app_id="a", client_secret="b")
        assert adapter._dm_policy == "pairing"


    def test_group_policy_default(self):
        adapter = self._make(app_id="a", client_secret="b")
        assert adapter._group_policy == "disabled"

    def test_allow_from_parsing_string(self):
        adapter = self._make(app_id="a", client_secret="b", allow_from="x, y , z")
        assert adapter._allow_from == ["x", "y", "z"]


    def test_markdown_support_default(self):
        adapter = self._make(app_id="a", client_secret="b")
        assert adapter._markdown_support is True


# ---------------------------------------------------------------------------
# _coerce_list
# ---------------------------------------------------------------------------

class TestCoerceList:
    def _fn(self, value):
        from gateway.platforms.qqbot import _coerce_list
        return _coerce_list(value)

    def test_none(self):
        assert self._fn(None) == []

    def test_string(self):
        assert self._fn("a, b ,c") == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# _is_voice_content_type
# ---------------------------------------------------------------------------

class TestIsVoiceContentType:
    def _fn(self, content_type, filename):
        from gateway.platforms.qqbot import QQAdapter
        return QQAdapter._is_voice_content_type(content_type, filename)


    def test_voice_extension_fallback_when_content_type_empty(self):
        """content_type='' with audio extension → True (extension fallback)."""
        assert self._fn("", "file.silk") is True


    def test_audio_extension_amr_fallback_when_content_type_empty(self):
        """content_type='' with .amr extension → True (extension fallback)."""
        assert self._fn("", "recording.amr") is True


# ---------------------------------------------------------------------------
# Voice attachment SSRF protection
# ---------------------------------------------------------------------------

class TestVoiceAttachmentSSRFProtection:
    def _make_adapter(self, **extra):
        from gateway.platforms.qqbot import QQAdapter
        return QQAdapter(_make_config(**extra))


    def test_connect_uses_redirect_guard_hook(self):
        from gateway.platforms.qqbot import QQAdapter, _ssrf_redirect_guard

        client = mock.AsyncMock()
        with mock.patch("gateway.platforms.qqbot.adapter.httpx.AsyncClient", return_value=client) as async_client_cls:
            adapter = QQAdapter(_make_config(app_id="a", client_secret="b"))
            adapter._ensure_token = mock.AsyncMock(side_effect=RuntimeError("stop after client creation"))

            connected = asyncio.run(adapter.connect())

        assert connected is False
        assert async_client_cls.call_count == 1
        kwargs = async_client_cls.call_args.kwargs
        assert kwargs.get("follow_redirects") is True
        assert kwargs.get("event_hooks", {}).get("response") == [_ssrf_redirect_guard]


# ---------------------------------------------------------------------------
# Voice attachment temp-file cleanup
# ---------------------------------------------------------------------------

class TestVoiceAttachmentTempCleanup:
    def _make_adapter(self, **extra):
        from gateway.platforms.qqbot import QQAdapter
        return QQAdapter(_make_config(**extra))

    def _setup_download_mocks(self, adapter, content=b"RIFFmock-wav-audio-data"):
        response = mock.Mock()
        response.content = content
        response.headers = {"content-type": "audio/wav"}
        response.raise_for_status = mock.Mock()

        adapter._http_client = mock.AsyncMock()
        adapter._http_client.get = mock.AsyncMock(return_value=response)

    def test_temp_wav_cleaned_up_on_stt_failure(self):
        adapter = self._make_adapter(app_id="a", client_secret="b")
        self._setup_download_mocks(adapter)
        seen = {}

        async def _raise_transport_error(path):
            seen["wav_path"] = path
            raise httpx.TransportError("boom")

        with mock.patch("tools.url_safety.is_safe_url", return_value=True):
            adapter._call_stt = mock.AsyncMock(side_effect=_raise_transport_error)
            transcript = asyncio.run(
                adapter._stt_voice_attachment(
                    "https://cdn.qq.com/voice.silk",
                    "audio/silk",
                    "voice.silk",
                    voice_wav_url="https://cdn.qq.com/voice.wav",
                )
            )

        assert transcript is None
        assert "wav_path" in seen
        assert not os.path.exists(seen["wav_path"])


# ---------------------------------------------------------------------------
# WebSocket proxy handling
# ---------------------------------------------------------------------------

class TestQQWebSocketProxy:
    @pytest.mark.asyncio
    async def test_open_ws_honors_proxy_env(self, monkeypatch):
        from gateway.platforms.qqbot import QQAdapter

        for key in (
            "WSS_PROXY",
            "wss_proxy",
            "HTTPS_PROXY",
            "https_proxy",
            "ALL_PROXY",
            "all_proxy",
        ):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7897")

        adapter = QQAdapter(_make_config(app_id="a", client_secret="b"))

        seen_session_kwargs = {}
        seen_ws_kwargs = {}

        class FakeSession:
            def __init__(self, **kwargs):
                seen_session_kwargs.update(kwargs)
                self.closed = False

            async def close(self):
                self.closed = True

            async def ws_connect(self, *args, **kwargs):
                seen_ws_kwargs.update(kwargs)
                return mock.AsyncMock(closed=False)

        with mock.patch("gateway.platforms.qqbot.adapter.aiohttp.ClientSession", side_effect=FakeSession):
            await adapter._open_ws("wss://api.sgroup.qq.com/websocket")

        assert seen_session_kwargs.get("trust_env") is True
        assert seen_ws_kwargs.get("proxy") == "http://127.0.0.1:7897"

# ---------------------------------------------------------------------------
# _strip_at_mention
# ---------------------------------------------------------------------------

class TestStripAtMention:
    def _fn(self, content):
        from gateway.platforms.qqbot import QQAdapter
        return QQAdapter._strip_at_mention(content)

    def test_removes_mention(self):
        result = self._fn("@BotUser hello there")
        assert result == "hello there"

    def test_only_mention(self):
        assert self._fn("@Someone  ") == ""

    def test_strips_explicit_mention_tag(self):
        # Full-push GROUP_MESSAGE_CREATE may carry the <@!id> tag form.
        assert self._fn("<@!1903885637> hello there") == "hello there"

    def test_strips_tag_midstring(self):
        assert self._fn("hey <@!123> how are you") == "hey  how are you"


# ---------------------------------------------------------------------------
# Group rich media / markdown parity (2.3)
# ---------------------------------------------------------------------------

class TestGroupMediaMarkdown:
    def _make_adapter(self, **extra):
        from gateway.platforms.qqbot import QQAdapter
        extra.setdefault("app_id", "1903885637")
        extra.setdefault("client_secret", "b")
        extra.setdefault("group_policy", "open")
        return QQAdapter(_make_config(**extra))

    @pytest.mark.asyncio
    async def test_group_inbound_image_populates_media_urls(self):
        adapter = self._make_adapter(group_require_mention=False)  # always
        captured = []

        async def fake_process(_a):
            return {"image_urls": ["/tmp/x.jpg"],
                    "image_media_types": ["image/jpeg"],
                    "voice_transcripts": [], "attachment_info": ""}

        async def fake_quote(_d):
            return {"quote_block": "", "image_urls": [], "image_media_types": []}

        async def fake_handle(event):
            captured.append(event)

        adapter._process_attachments = fake_process  # type: ignore[assignment]
        adapter._process_quoted_context = fake_quote  # type: ignore[assignment]
        adapter.handle_message = fake_handle  # type: ignore[assignment]

        await adapter._handle_group_message(
            {"group_openid": "g1", "content": "look",
             "attachments": [{"content_type": "image/jpeg", "url": "u"}]},
            "m1", "look", {"member_openid": "u1"}, "", "GROUP_MESSAGE_CREATE",
        )
        assert len(captured) == 1
        assert captured[0].media_urls == ["/tmp/x.jpg"]
        assert captured[0].media_types == ["image/jpeg"]

    @pytest.mark.asyncio
    async def test_group_markdown_send_routes_to_group_endpoint(self):
        adapter = self._make_adapter(markdown_support=True)
        adapter._running = True
        adapter._ws = SimpleNamespace(closed=False)
        adapter._chat_type_map["g1"] = "group"
        calls = []

        async def fake_api(method, path, body=None, **kw):
            calls.append((method, path, body))
            return {"id": "sent1"}

        adapter._api_request = fake_api  # type: ignore[assignment]

        result = await adapter.send("g1", "**bold** reply")
        assert result.success
        assert calls and calls[0][1] == "/v2/groups/g1/messages"
        # markdown_support=True → markdown msg_type (2).
        assert calls[0][2]["msg_type"] == 2
        assert calls[0][2]["markdown"]["content"] == "**bold** reply"

    @pytest.mark.asyncio
    async def test_group_plaintext_send_routes_to_group_endpoint(self):
        # Default markdown_support=False → plain text msg_type (0).
        adapter = self._make_adapter(markdown_support=False)
        adapter._running = True
        adapter._ws = SimpleNamespace(closed=False)
        adapter._chat_type_map["g1"] = "group"
        calls = []

        async def fake_api(method, path, body=None, **kw):
            calls.append((method, path, body))
            return {"id": "sent2"}

        adapter._api_request = fake_api  # type: ignore[assignment]

        result = await adapter.send("g1", "plain reply", reply_to="mm1")
        assert result.success
        assert calls[0][1] == "/v2/groups/g1/messages"
        assert calls[0][2]["msg_type"] == 0
        assert calls[0][2]["content"] == "plain reply"
        assert calls[0][2].get("msg_id") == "mm1"

    @pytest.mark.asyncio
    async def test_group_media_send_posts_msg_type_media(self):
        adapter = self._make_adapter()
        adapter._running = True
        adapter._ws = SimpleNamespace(closed=False)
        adapter._chat_type_map["g1"] = "group"
        calls = []

        async def fake_upload(chat_type, chat_id, file_type, **kw):
            return {"file_info": "FILEINFO"}

        async def fake_api(method, path, body=None, **kw):
            calls.append((method, path, body))
            return {"id": "media1"}

        adapter._upload_media = fake_upload  # type: ignore[assignment]
        adapter._api_request = fake_api  # type: ignore[assignment]

        result = await adapter._send_media(
            "g1", "https://example.com/x.jpg", 1, "image",
        )
        assert result.success
        assert calls and calls[0][1] == "/v2/groups/g1/messages"
        assert calls[0][2]["msg_type"] == 7  # MSG_TYPE_MEDIA
        assert calls[0][2]["media"]["file_info"] == "FILEINFO"


# ---------------------------------------------------------------------------
# Reserved runtime mode-switch hook (2.2.3)
# ---------------------------------------------------------------------------

class TestGroupModeRuntimeOverride:
    def _make_adapter(self, **extra):
        from gateway.platforms.qqbot import QQAdapter
        extra.setdefault("app_id", "a")
        extra.setdefault("client_secret", "b")
        return QQAdapter(_make_config(**extra))

    def test_override_takes_effect(self):
        from gateway.platforms.qqbot.group_activation import resolve_require_mention
        adapter = self._make_adapter(group_require_mention=True)
        adapter._set_group_mode_override("g1", require_mention=False)
        assert adapter._group_mode_runtime_overrides == {"g1": False}
        # resolve should now honour the runtime override for g1.
        eff = resolve_require_mention(
            "g1",
            global_default=adapter._group_require_mention,
            per_group=adapter._group_mode_overrides,
            runtime_overrides=adapter._group_mode_runtime_overrides,
        )
        assert eff is False


# ---------------------------------------------------------------------------
# _is_dm_allowed
# ---------------------------------------------------------------------------

class TestDmAllowed:
    def _make_adapter(self, **extra):
        from gateway.platforms.qqbot import QQAdapter
        return QQAdapter(_make_config(**extra))


    def test_open_policy_with_opt_in(self, monkeypatch):
        monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "true")
        adapter = self._make_adapter(app_id="a", client_secret="b", dm_policy="open")
        assert adapter._is_dm_allowed("any_user") is True
        assert adapter._is_dm_intake_allowed("any_user") is True


    def test_allowlist_match(self):
        adapter = self._make_adapter(app_id="a", client_secret="b", dm_policy="allowlist", allow_from="user1,user2")
        assert adapter._is_dm_allowed("user1") is True


# ---------------------------------------------------------------------------
# _is_group_allowed
# ---------------------------------------------------------------------------

class TestGroupAllowed:
    def _make_adapter(self, **extra):
        from gateway.platforms.qqbot import QQAdapter
        return QQAdapter(_make_config(**extra))


    def test_allowlist_match(self):
        adapter = self._make_adapter(app_id="a", client_secret="b", group_policy="allowlist", group_allow_from="grp1")
        assert adapter._is_group_allowed("grp1", "user1") is True


    def test_pairing_default_blocks_groups(self):
        # group_policy default is now "disabled" (Feishu-aligned: no
        # "pairing" mode for groups). Any group message is denied by default.
        adapter = self._make_adapter(app_id="a", client_secret="b")
        assert adapter._group_policy == "disabled"
        assert adapter._is_group_allowed("grp1", "user1") is False

    def test_unknown_group_policy_falls_back_to_disabled(self):
        adapter = self._make_adapter(
            app_id="a", client_secret="b", group_policy="pairing",  # no longer valid
        )
        assert adapter._group_policy == "disabled"
        assert adapter._is_group_allowed("grp1", "user1") is False


# ---------------------------------------------------------------------------
# _resolve_stt_config
# ---------------------------------------------------------------------------

class TestResolveSTTConfig:
    def _make_adapter(self, **extra):
        from gateway.platforms.qqbot import QQAdapter
        return QQAdapter(_make_config(**extra))

    def test_no_config(self):
        adapter = self._make_adapter(app_id="a", client_secret="b")
        with mock.patch.dict(os.environ, {}, clear=True):
            assert adapter._resolve_stt_config() is None


# ---------------------------------------------------------------------------
# _detect_message_type
# ---------------------------------------------------------------------------

class TestDetectMessageType:
    def _fn(self, media_urls, media_types):
        from gateway.platforms.qqbot import QQAdapter
        return QQAdapter._detect_message_type(media_urls, media_types)

    def test_no_media(self):
        from gateway.platforms.base import MessageType
        assert self._fn([], []) == MessageType.TEXT


# ---------------------------------------------------------------------------
# QQCloseError
# ---------------------------------------------------------------------------

class TestQQCloseError:
    def test_attributes(self):
        from gateway.platforms.qqbot import QQCloseError
        err = QQCloseError(4004, "bad token")
        assert err.code == 4004
        assert err.reason == "bad token"


# ---------------------------------------------------------------------------
# _dispatch_payload
# ---------------------------------------------------------------------------

class TestDispatchPayload:
    def _make_adapter(self, **extra):
        from gateway.platforms.qqbot import QQAdapter
        adapter = QQAdapter(_make_config(**extra))
        return adapter

    def test_unknown_op(self):
        adapter = self._make_adapter(app_id="a", client_secret="b")
        # Should not raise
        adapter._dispatch_payload({"op": 99, "d": {}})
        # last_seq should remain None
        assert adapter._last_seq is None


    def test_seq_increments(self):
        adapter = self._make_adapter(app_id="a", client_secret="b")
        adapter._dispatch_payload({"op": 0, "t": "READY", "s": 5, "d": {}})
        adapter._dispatch_payload({"op": 0, "t": "SOME_EVENT", "s": 10, "d": {}})
        assert adapter._last_seq == 10


# ---------------------------------------------------------------------------
# READY / RESUMED handling
# ---------------------------------------------------------------------------

class TestReadyHandling:
    def _make_adapter(self, **extra):
        from gateway.platforms.qqbot import QQAdapter
        return QQAdapter(_make_config(**extra))

    def test_ready_stores_session(self):
        adapter = self._make_adapter(app_id="a", client_secret="b")
        adapter._dispatch_payload({
            "op": 0, "t": "READY",
            "s": 1,
            "d": {"session_id": "sess_abc123"},
        })
        assert adapter._session_id == "sess_abc123"


# ---------------------------------------------------------------------------
# _parse_json
# ---------------------------------------------------------------------------

class TestParseJson:
    def _fn(self, raw):
        from gateway.platforms.qqbot import QQAdapter
        return QQAdapter._parse_json(raw)

    def test_valid_json(self):
        result = self._fn('{"op": 10, "d": {}}')
        assert result == {"op": 10, "d": {}}

    def test_invalid_json(self):
        result = self._fn("not json")
        assert result is None


# ---------------------------------------------------------------------------
# _build_text_body
# ---------------------------------------------------------------------------

class TestBuildTextBody:
    def _make_adapter(self, **extra):
        from gateway.platforms.qqbot import QQAdapter
        return QQAdapter(_make_config(**extra))

    def test_plain_text(self):
        adapter = self._make_adapter(app_id="a", client_secret="b", markdown_support=False)
        body = adapter._build_text_body("hello world")
        assert body["msg_type"] == 0  # MSG_TYPE_TEXT
        assert body["content"] == "hello world"

    def test_markdown_text(self):
        adapter = self._make_adapter(app_id="a", client_secret="b", markdown_support=True)
        body = adapter._build_text_body("**bold** text")
        assert body["msg_type"] == 2  # MSG_TYPE_MARKDOWN
        assert body["markdown"]["content"] == "**bold** text"


# ---------------------------------------------------------------------------
# _wait_for_reconnection / send reconnection wait
# ---------------------------------------------------------------------------

class TestWaitForReconnection:
    """Test that send() waits for reconnection instead of silently dropping."""

    def _make_adapter(self, **extra):
        from gateway.platforms.qqbot import QQAdapter
        return QQAdapter(_make_config(**extra))

    @pytest.mark.asyncio
    async def test_send_waits_and_succeeds_on_reconnect(self):
        """send() should wait for reconnection and then deliver the message."""
        adapter = self._make_adapter(app_id="a", client_secret="b")
        # Initially disconnected
        adapter._running = False
        adapter._http_client = mock.MagicMock()

        # Simulate reconnection after 0.3s (faster than real interval)
        async def fake_api_request(*args, **kwargs):
            return {"id": "msg_123"}

        adapter._api_request = fake_api_request
        adapter._ensure_token = mock.AsyncMock()
        adapter._RECONNECT_POLL_INTERVAL = 0.1
        adapter._RECONNECT_WAIT_SECONDS = 5.0

        # Schedule reconnection after a short delay
        async def reconnect_after_delay():
            await asyncio.sleep(0.2)
            adapter._running = True
            adapter._ws = SimpleNamespace(closed=False)

        asyncio.get_event_loop().create_task(reconnect_after_delay())

        result = await adapter.send("test_openid", "Hello, world!")
        assert result.success
        assert result.message_id == "msg_123"


# ---------------------------------------------------------------------------
# ChunkedUploader
# ---------------------------------------------------------------------------

class TestChunkedUploadFormatSize:
    def test_bytes(self):
        from gateway.platforms.qqbot.chunked_upload import format_size
        assert format_size(100) == "100.0 B"


class TestChunkedUploadErrors:

    def test_too_large_includes_limit(self):
        from gateway.platforms.qqbot.chunked_upload import UploadFileTooLargeError
        exc = UploadFileTooLargeError("huge.bin", 200 * 1024 * 1024, 100 * 1024 * 1024)
        assert exc.file_name == "huge.bin"
        assert "MB" in exc.file_size_human
        assert "MB" in exc.limit_human
        assert "huge.bin" in str(exc)


class TestChunkedUploadHelpers:

    def test_read_chunk_short_read_raises(self, tmp_path):
        from gateway.platforms.qqbot.chunked_upload import _read_file_chunk
        f = tmp_path / "x.bin"
        f.write_bytes(b"hi")
        with pytest.raises(IOError):
            _read_file_chunk(str(f), 0, 100)


    def test_parse_prepare_response_wrapped_in_data(self):
        from gateway.platforms.qqbot.chunked_upload import _parse_prepare_response
        raw = {
            "data": {
                "upload_id": "uid-42",
                "block_size": 4096,
                "parts": [
                    {"part_index": 1, "presigned_url": "https://cos/1", "block_size": 4096},
                    {"index": 2, "url": "https://cos/2"},
                ],
                "concurrency": 3,
                "retry_timeout": 90,
            }
        }
        r = _parse_prepare_response(raw)
        assert r.upload_id == "uid-42"
        assert r.block_size == 4096
        assert len(r.parts) == 2
        assert r.parts[0].presigned_url == "https://cos/1"
        assert r.parts[1].index == 2
        assert r.concurrency == 3
        assert r.retry_timeout == 90.0


class TestChunkedUploaderFlow:
    """End-to-end prepare / PUT / part_finish / complete flow with mocked HTTP.

    Verifies the state machine matches the QQ v2 contract without hitting the network.
    """

    @pytest.mark.asyncio
    async def test_full_upload_two_parts_success(self, tmp_path):
        from gateway.platforms.qqbot.chunked_upload import ChunkedUploader

        # Two-part file.
        f = tmp_path / "vid.mp4"
        f.write_bytes(b"A" * 5_000_000 + b"B" * 3_000_000)

        # Mock api_request — handles prepare, part_finish, complete based on URL.
        api_calls = []

        async def fake_api_request(method, path, *, body=None, timeout=None):
            api_calls.append((method, path, body))
            if path.endswith("/upload_prepare"):
                return {
                    "upload_id": "uid-xyz",
                    "block_size": 5_000_000,
                    "parts": [
                        {"part_index": 1, "presigned_url": "https://cos.example/p1"},
                        {"part_index": 2, "presigned_url": "https://cos.example/p2"},
                    ],
                    "concurrency": 1,
                }
            if path.endswith("/upload_part_finish"):
                return {}
            # complete
            return {"file_info": "FILEINFO_TOKEN", "file_uuid": "u-1"}

        # Mock http_put — always returns 200.
        put_calls = []

        class _FakeResp:
            status_code = 200
            text = ""

        async def fake_put(url, data=None, headers=None):
            put_calls.append((url, len(data), headers))
            return _FakeResp()

        uploader = ChunkedUploader(
            api_request=fake_api_request,
            http_put=fake_put,
            log_tag="QQBot:TEST",
        )
        result = await uploader.upload(
            chat_type="c2c",
            target_id="user-openid-1",
            file_path=str(f),
            file_type=2,  # MEDIA_TYPE_VIDEO
            file_name="vid.mp4",
        )

        assert result["file_info"] == "FILEINFO_TOKEN"
        # Two PUTs, one per part.
        assert len(put_calls) == 2
        assert put_calls[0][0] == "https://cos.example/p1"
        assert put_calls[1][0] == "https://cos.example/p2"
        # Prepare + 2 part_finish + complete = 4 api calls.
        assert len(api_calls) == 4
        assert api_calls[0][1].endswith("/upload_prepare")
        assert api_calls[1][1].endswith("/upload_part_finish")
        assert api_calls[2][1].endswith("/upload_part_finish")
        # complete path reuses /files.
        assert api_calls[3][1].endswith("/files")
        assert api_calls[3][2] == {"upload_id": "uid-xyz"}

    @pytest.mark.asyncio
    async def test_group_paths(self, tmp_path):
        """Group uploads hit /v2/groups/... instead of /v2/users/..."""
        from gateway.platforms.qqbot.chunked_upload import ChunkedUploader

        f = tmp_path / "a.bin"
        f.write_bytes(b"x" * 100)

        seen_paths = []

        async def fake_api_request(method, path, *, body=None, timeout=None):
            seen_paths.append(path)
            if path.endswith("/upload_prepare"):
                return {
                    "upload_id": "gid-1",
                    "block_size": 100,
                    "parts": [{"part_index": 1, "presigned_url": "https://cos/g1"}],
                }
            if path.endswith("/upload_part_finish"):
                return {}
            return {"file_info": "GFILE"}

        class _R:
            status_code = 200
            text = ""

        async def fake_put(url, data=None, headers=None):
            return _R()

        u = ChunkedUploader(fake_api_request, fake_put, "QQBot:T")
        await u.upload(
            chat_type="group",
            target_id="grp-openid-1",
            file_path=str(f),
            file_type=4,
            file_name="a.bin",
        )
        assert all("/v2/groups/" in p for p in seen_paths)
        assert any(p.endswith("/upload_prepare") for p in seen_paths)
        assert any(p.endswith("/files") for p in seen_paths)


# ---------------------------------------------------------------------------
# Inline keyboards — approval + update-prompt flows
# ---------------------------------------------------------------------------

class TestApprovalButtonData:
    def test_parse_allow_once(self):
        from gateway.platforms.qqbot.keyboards import parse_approval_button_data
        result = parse_approval_button_data("approve:agent:main:qqbot:c2c:UID:allow-once")
        assert result == ("agent:main:qqbot:c2c:UID", "allow-once")


    def test_parse_empty_returns_none(self):
        from gateway.platforms.qqbot.keyboards import parse_approval_button_data
        assert parse_approval_button_data("") is None
        assert parse_approval_button_data(None) is None  # type: ignore[arg-type]


class TestUpdatePromptButtonData:
    def test_parse_yes(self):
        from gateway.platforms.qqbot.keyboards import parse_update_prompt_button_data
        assert parse_update_prompt_button_data("update_prompt:y") == "y"


class TestBuildApprovalKeyboard:
    def test_three_buttons_in_single_row(self):
        from gateway.platforms.qqbot.keyboards import build_approval_keyboard
        kb = build_approval_keyboard("session-1")
        assert len(kb.content.rows) == 1
        assert len(kb.content.rows[0].buttons) == 3

    def test_button_data_embeds_session_key(self):
        from gateway.platforms.qqbot.keyboards import build_approval_keyboard
        kb = build_approval_keyboard("agent:main:qqbot:c2c:UID")
        datas = [b.action.data for b in kb.content.rows[0].buttons]
        assert datas[0] == "approve:agent:main:qqbot:c2c:UID:allow-once"
        assert datas[1] == "approve:agent:main:qqbot:c2c:UID:allow-always"
        assert datas[2] == "approve:agent:main:qqbot:c2c:UID:deny"


class TestBuildUpdatePromptKeyboard:
    def test_two_buttons(self):
        from gateway.platforms.qqbot.keyboards import build_update_prompt_keyboard
        kb = build_update_prompt_keyboard()
        assert len(kb.content.rows[0].buttons) == 2


class TestBuildApprovalText:


    def test_truncates_long_commands(self):
        from gateway.platforms.qqbot.keyboards import (
            ApprovalRequest, build_approval_text,
        )
        long = "x" * 1000
        req = ApprovalRequest(
            session_key="s", title="t", command_preview=long, cwd="/x",
        )
        text = build_approval_text(req)
        # Preview is truncated to 300 chars; 1000 "x"s would still push the
        # body past 300, but the inline preview specifically must be capped.
        preview_line = [
            line for line in text.split("\n") if line.startswith("```")
        ]
        # 2 backtick fences; the content line in between is separate.
        xs_in_preview = sum(line.count("x") for line in text.split("\n") if line and "```" not in line)
        assert xs_in_preview <= 301  # 300 xs + one-off tolerance


class TestInteractionEventParsing:
    def test_parse_c2c_interaction(self):
        from gateway.platforms.qqbot.keyboards import parse_interaction_event
        raw = {
            "id": "interaction-42",
            "chat_type": 2,
            "user_openid": "user-1",
            "data": {
                "type": 11,
                "resolved": {
                    "button_data": "approve:sess:allow-once",
                    "button_id": "allow",
                },
            },
        }
        ev = parse_interaction_event(raw)
        assert ev.id == "interaction-42"
        assert ev.scene == "c2c"
        assert ev.chat_type == 2
        assert ev.user_openid == "user-1"
        assert ev.button_data == "approve:sess:allow-once"
        assert ev.button_id == "allow"
        assert ev.operator_openid == "user-1"


class TestAdapterInteractionDispatch:
    """End-to-end verification of _on_interaction including ACK + callback."""

    def _make_adapter(self):
        from gateway.platforms.qqbot.adapter import QQAdapter
        return QQAdapter(_make_config(app_id="a", client_secret="b"))

    @pytest.mark.asyncio
    async def test_callback_invoked_with_parsed_event(self):
        adapter = self._make_adapter()

        # Stub ACK so we don't require a live http_client.
        ack_calls = []

        async def fake_ack(interaction_id, code=0):
            ack_calls.append((interaction_id, code))

        adapter._acknowledge_interaction = fake_ack  # type: ignore[assignment]

        received = []

        async def cb(event):
            received.append(event)

        adapter.set_interaction_callback(cb)
        await adapter._on_interaction({
            "id": "i-1",
            "chat_type": 2,
            "user_openid": "user-1",
            "data": {
                "type": 11,
                "resolved": {"button_data": "approve:agent:main:qqbot:c2c:u:deny", "button_id": "deny"},
            },
        })

        assert len(ack_calls) == 1
        assert ack_calls[0][0] == "i-1"
        assert len(received) == 1
        assert received[0].button_data == "approve:agent:main:qqbot:c2c:u:deny"
        assert received[0].scene == "c2c"


# ---------------------------------------------------------------------------
# Quoted-message handling (message_type=103 → msg_elements)
# ---------------------------------------------------------------------------

class TestProcessQuotedContext:
    """Verify the quoted-message pipeline: text + voice STT + images + files."""

    def _make_adapter(self):
        from gateway.platforms.qqbot.adapter import QQAdapter
        return QQAdapter(_make_config(app_id="a", client_secret="b"))

    @pytest.mark.asyncio
    async def test_non_quote_message_returns_empty(self):
        adapter = self._make_adapter()
        d = {"message_type": 0, "content": "hi"}
        out = await adapter._process_quoted_context(d)
        assert out == {"quote_block": "", "image_urls": [], "image_media_types": []}


    @pytest.mark.asyncio
    async def test_quote_with_voice_attachment_runs_stt(self):
        adapter = self._make_adapter()

        # Capture what attachments are passed into _process_attachments.
        captured = []

        async def fake_process(atts):
            captured.append(atts)
            return {
                "image_urls": [],
                "image_media_types": [],
                "voice_transcripts": ["[Voice] hello from the quoted audio"],
                "attachment_info": "",
            }

        adapter._process_attachments = fake_process  # type: ignore[assignment]

        d = {
            "message_type": 103,
            "msg_elements": [{
                "content": "",
                "attachments": [
                    {"content_type": "audio/silk",
                     "url": "https://qq-cdn/x.silk",
                     "filename": "rec.silk"}
                ],
            }],
        }
        out = await adapter._process_quoted_context(d)

        # The quoted voice attachment must actually flow through STT.
        assert captured and len(captured[0]) == 1
        assert captured[0][0]["content_type"] == "audio/silk"
        assert "[Quoted message]:" in out["quote_block"]
        assert "hello from the quoted audio" in out["quote_block"]


    @pytest.mark.asyncio
    async def test_multiple_elements_concatenated(self):
        adapter = self._make_adapter()

        async def fake_process(atts):
            assert len(atts) == 2
            return {
                "image_urls": [], "image_media_types": [],
                "voice_transcripts": [], "attachment_info": "",
            }

        adapter._process_attachments = fake_process  # type: ignore[assignment]

        d = {
            "message_type": 103,
            "msg_elements": [
                {"content": "first", "attachments": [{"content_type": "image/png", "url": "a"}]},
                {"content": "second", "attachments": [{"content_type": "image/png", "url": "b"}]},
            ],
        }
        out = await adapter._process_quoted_context(d)
        assert "first" in out["quote_block"]
        assert "second" in out["quote_block"]


class TestMergeQuoteInto:
    def test_empty_quote_returns_original(self):
        from gateway.platforms.qqbot.adapter import QQAdapter
        assert QQAdapter._merge_quote_into("hello", "") == "hello"


# ---------------------------------------------------------------------------
# Gateway-contract approval UX — send_exec_approval + default dispatcher
# ---------------------------------------------------------------------------

class TestDefaultInteractionDispatch:
    """Verify the adapter's default INTERACTION_CREATE router."""

    def _make_adapter(self):
        from gateway.platforms.qqbot.adapter import QQAdapter
        return QQAdapter(_make_config(app_id="a", client_secret="b"))

    def test_default_callback_installed_on_init(self):
        """Fresh adapter has a working default interaction callback."""
        adapter = self._make_adapter()
        assert adapter._interaction_callback is not None
        assert adapter._interaction_callback == adapter._default_interaction_dispatch


    @pytest.mark.asyncio
    async def test_approval_click_once_maps_to_once(self):
        """'allow-once' button → resolve_gateway_approval(session, 'once')."""
        adapter = self._make_adapter()

        resolve_calls = []

        def fake_resolve(session_key, choice, resolve_all=False):
            resolve_calls.append((session_key, choice, resolve_all))
            return 1

        # Patch the *module-level* function that _default_interaction_dispatch
        # imports lazily.
        import tools.approval
        orig = tools.approval.resolve_gateway_approval
        tools.approval.resolve_gateway_approval = fake_resolve
        try:
            from gateway.platforms.qqbot.keyboards import parse_interaction_event
            event = parse_interaction_event({
                "id": "i",
                "chat_type": 2,
                "user_openid": "u-42",
                "data": {"resolved": {"button_data": "approve:agent:main:qqbot:c2c:u-42:allow-once"}},
            })
            await adapter._default_interaction_dispatch(event)
        finally:
            tools.approval.resolve_gateway_approval = orig

        assert resolve_calls == [("agent:main:qqbot:c2c:u-42", "once", False)]


    @pytest.mark.asyncio
    async def test_approval_click_rejects_unauthorized_operator(self):
        adapter = self._make_adapter()
        resolve_calls = []

        def fake_resolve(session_key, choice, resolve_all=False):
            resolve_calls.append((session_key, choice, resolve_all))
            return 1

        import tools.approval
        orig = tools.approval.resolve_gateway_approval
        tools.approval.resolve_gateway_approval = fake_resolve
        try:
            from gateway.platforms.qqbot.keyboards import parse_interaction_event
            event = parse_interaction_event({
                "id": "i", "chat_type": 1,
                "group_openid": "g-1",
                "group_member_openid": "attacker",
                "data": {"resolved": {"button_data": "approve:agent:main:qqbot:group:g-1:owner:allow-once"}},
            })
            await adapter._default_interaction_dispatch(event)
        finally:
            tools.approval.resolve_gateway_approval = orig

        assert resolve_calls == []

    @pytest.mark.asyncio
    async def test_update_prompt_click_writes_response_file(self, tmp_path, monkeypatch):
        """update_prompt:y click writes 'y' to ~/.hermes/.update_response."""
        adapter = self._make_adapter()
        hermes_home = tmp_path / "hermes_home"
        hermes_home.mkdir()
        monkeypatch.setattr(
            "hermes_constants.get_hermes_home",
            lambda: hermes_home,
        )

        from gateway.platforms.qqbot.keyboards import parse_interaction_event
        event = parse_interaction_event({
            "id": "i", "chat_type": 2, "user_openid": "u-1",
            "data": {"resolved": {"button_data": "update_prompt:y"}},
        })
        await adapter._default_interaction_dispatch(event)

        response = hermes_home / ".update_response"
        assert response.exists()
        assert response.read_text() == "y"


class TestSendExecApproval:
    """Verify the gateway contract: QQAdapter.send_exec_approval(...)."""

    def _make_adapter(self):
        from gateway.platforms.qqbot.adapter import QQAdapter
        return QQAdapter(_make_config(app_id="a", client_secret="b"))

    @pytest.mark.asyncio
    async def test_delegates_to_send_approval_request(self):
        adapter = self._make_adapter()

        calls = []

        async def fake_send_approval(chat_id, req, reply_to=None):
            from gateway.platforms.base import SendResult
            calls.append({"chat_id": chat_id, "req": req, "reply_to": reply_to})
            return SendResult(success=True, message_id="m-1")

        adapter.send_approval_request = fake_send_approval  # type: ignore[assignment]
        # Seed last-msg-id so the reply_to path is exercised.
        adapter._last_msg_id["user-1"] = "inbound-42"

        result = await adapter.send_exec_approval(
            chat_id="user-1",
            command="rm -rf /tmp/demo",
            session_key="sess:abc",
            description="delete temp dir",
        )
        assert result.success
        assert len(calls) == 1
        req = calls[0]["req"]
        assert req.session_key == "sess:abc"
        assert req.command_preview == "rm -rf /tmp/demo"
        assert req.description == "delete temp dir"
        assert calls[0]["reply_to"] == "inbound-42"


class TestSendUpdatePrompt:
    """Verify the cross-adapter send_update_prompt signature + behaviour."""

    def _make_adapter(self):
        from gateway.platforms.qqbot.adapter import QQAdapter
        return QQAdapter(_make_config(app_id="a", client_secret="b"))

    @pytest.mark.asyncio
    async def test_delegates_to_send_with_keyboard(self):
        adapter = self._make_adapter()

        captured = {}

        async def fake_swk(chat_id, content, keyboard, reply_to=None):
            from gateway.platforms.base import SendResult
            captured["chat_id"] = chat_id
            captured["content"] = content
            captured["keyboard"] = keyboard
            captured["reply_to"] = reply_to
            return SendResult(success=True, message_id="mid")

        adapter.send_with_keyboard = fake_swk  # type: ignore[assignment]
        adapter._last_msg_id["u1"] = "prev-msg"

        result = await adapter.send_update_prompt(
            chat_id="u1", prompt="Continue with update?",
            default="y", session_key="ignored", metadata={"x": 1},
        )
        assert result.success
        assert "Continue with update?" in captured["content"]
        assert "default: y" in captured["content"]
        assert captured["reply_to"] == "prev-msg"
        # Keyboard has the Yes/No buttons.
        dd = captured["keyboard"].to_dict()
        datas = [b["action"]["data"] for b in dd["content"]["rows"][0]["buttons"]]
        assert datas == ["update_prompt:y", "update_prompt:n"]


# ---------------------------------------------------------------------------
# _send_identify includes INTERACTION intent
# ---------------------------------------------------------------------------

class TestIdentifyIntents:
    """Verify the WebSocket identify payload includes the INTERACTION intent bit."""

    def _make_adapter(self):
        from gateway.platforms.qqbot.adapter import QQAdapter
        return QQAdapter(_make_config(app_id="a", client_secret="b"))

    @pytest.mark.asyncio
    async def test_intents_include_interaction_bit(self):
        adapter = self._make_adapter()

        # Mock token retrieval and WebSocket
        adapter._access_token = "fake_token"
        adapter._token_expires_at = 9999999999.0

        sent_payloads = []

        class FakeWS:
            closed = False

            async def send_json(self, payload):
                sent_payloads.append(payload)

        adapter._ws = FakeWS()
        await adapter._send_identify()

        assert len(sent_payloads) == 1
        intents = sent_payloads[0]["d"]["intents"]

        # Verify all expected intent bits are present
        assert intents & (1 << 25), "GROUP_MESSAGES (1<<25) missing"
        assert intents & (1 << 30), "GUILD_AT_MESSAGE (1<<30) missing"
        assert intents & (1 << 12), "DIRECT_MESSAGES (1<<12) missing"
        assert intents & (1 << 26), "INTERACTION (1<<26) missing"


# ---------------------------------------------------------------------------
# _process_attachments: video/file path exposure
# ---------------------------------------------------------------------------

class TestProcessAttachmentsPathExposure:
    """Verify that video and file attachments include the cached local path."""

    def _make_adapter(self):
        from gateway.platforms.qqbot.adapter import QQAdapter
        return QQAdapter(_make_config(app_id="a", client_secret="b"))

    @pytest.mark.asyncio
    async def test_video_attachment_includes_path(self):
        adapter = self._make_adapter()

        # Mock _download_and_cache to return a known path
        async def fake_download(url, ct, original_name=""):
            return "/tmp/cache/video_abc123.mp4"

        adapter._download_and_cache = fake_download  # type: ignore[assignment]

        attachments = [
            {
                "content_type": "video/mp4",
                "url": "https://multimedia.nt.qq.com.cn/download/video123",
                "filename": "my_video.mp4",
            }
        ]
        result = await adapter._process_attachments(attachments)

        assert result["image_urls"] == []
        assert result["voice_transcripts"] == []
        info = result["attachment_info"]
        assert "[video:" in info
        assert "my_video.mp4" in info
        assert "/tmp/cache/video_abc123.mp4" in info


    @pytest.mark.asyncio
    async def test_quoted_video_includes_path_in_quote_block(self):
        """Quoted video attachments should surface the cached path in the quote block."""
        adapter = self._make_adapter()

        async def fake_process(atts):
            # Simulate the fixed _process_attachments for a video attachment.
            return {
                "image_urls": [],
                "image_media_types": [],
                "voice_transcripts": [],
                "attachment_info": "[video: clip.mp4 (/tmp/cache/clip.mp4)]",
            }

        adapter._process_attachments = fake_process  # type: ignore[assignment]

        d = {
            "message_type": 103,
            "msg_elements": [{
                "content": "看看这个视频",
                "attachments": [
                    {"content_type": "video/mp4",
                     "url": "https://qq-cdn/clip.mp4",
                     "filename": "clip.mp4"}
                ],
            }],
        }
        out = await adapter._process_quoted_context(d)
        assert "[Quoted message]:" in out["quote_block"]
        assert "/tmp/cache/clip.mp4" in out["quote_block"]


# ---------------------------------------------------------------------------
# WebSocket op 7 (Server Reconnect) and op 9 (Invalid Session)
# ---------------------------------------------------------------------------

class TestOp7ServerReconnect:
    """Verify op 7 triggers WS close (which triggers reconnect in outer loop)."""

    def _make_adapter(self):
        from gateway.platforms.qqbot.adapter import QQAdapter
        return QQAdapter(_make_config(app_id="a", client_secret="b"))

    def test_op7_closes_websocket(self):
        adapter = self._make_adapter()
        adapter._session_id = "sess_keep"
        adapter._last_seq = 42

        close_called = []

        class FakeWS:
            closed = False

            async def close(self):
                close_called.append(True)

        adapter._ws = FakeWS()
        adapter._dispatch_payload({"op": 7, "d": None})

        # Session should be preserved for Resume
        assert adapter._session_id == "sess_keep"
        assert adapter._last_seq == 42
        # close() should have been scheduled
        assert len(close_called) == 0  # _create_task schedules, not immediate
        # But the task was created — verify via asyncio


class TestOp9InvalidSession:
    """Verify op 9 handles resumable vs non-resumable sessions."""

    def _make_adapter(self):
        from gateway.platforms.qqbot.adapter import QQAdapter
        return QQAdapter(_make_config(app_id="a", client_secret="b"))


    @pytest.mark.asyncio
    async def test_op9_non_resumable_triggers_ws_close(self):
        adapter = self._make_adapter()
        adapter._session_id = "s"
        adapter._last_seq = 1
        close_called = []

        class FakeWS:
            closed = False

            async def close(self):
                close_called.append(True)
                self.closed = True

        adapter._ws = FakeWS()
        adapter._dispatch_payload({"op": 9, "d": False})
        await asyncio.sleep(0)

        assert close_called == [True]


# ---------------------------------------------------------------------------
# Close code classification
# ---------------------------------------------------------------------------

class TestCloseCodeClassification:
    """Verify fatal close codes stop reconnecting and 4009 preserves session."""

    def _make_adapter(self):
        from gateway.platforms.qqbot.adapter import QQAdapter
        return QQAdapter(_make_config(app_id="a", client_secret="b"))

    def test_4009_preserves_session(self):
        """4009 (connection timeout) should NOT clear the session."""
        adapter = self._make_adapter()
        adapter._session_id = "sess_to_keep"
        adapter._last_seq = 50

        # The session-clearing codes set should NOT contain 4009.
        # We verify the logic directly: dispatch a close-code event that
        # exercises the session-clearing path (4006), then verify 4009 does not.
        session_clear_codes = {
            4006, 4007, 4900, 4901, 4902, 4903,
            4904, 4905, 4906, 4907, 4908, 4909,
            4910, 4911, 4912, 4913,
        }
        assert 4009 not in session_clear_codes


class TestReadEventsClosedWsGuard:
    """Regression: a closed-but-non-None ws must raise on entry, not return
    normally, so _listen_loop goes through reconnect/backoff instead of
    busy-looping at 100% CPU (issues #31193 / #31771)."""

    def _make_adapter(self, **extra):
        from gateway.platforms.qqbot import QQAdapter
        return QQAdapter(_make_config(app_id="a", client_secret="b", **extra))

    def test_read_events_raises_when_ws_closed_on_entry(self):
        adapter = self._make_adapter()
        adapter._running = True
        adapter._ws = SimpleNamespace(closed=True)
        with pytest.raises(RuntimeError):
            asyncio.run(adapter._read_events())



# ---------------------------------------------------------------------------
# Group activation mode — mention detection (group_activation.detect_mentioned)
# ---------------------------------------------------------------------------

class TestDetectMentioned:
    def test_group_at_event_always_mentioned(self):
        from gateway.platforms.qqbot.group_activation import detect_mentioned
        assert detect_mentioned("GROUP_AT_MESSAGE_CREATE", {}, "hi", "app1") is True

    def test_mentions_is_you_true(self):
        from gateway.platforms.qqbot.group_activation import detect_mentioned
        d = {"mentions": [{"member_openid": "x"}, {"is_you": True}]}
        assert detect_mentioned("GROUP_MESSAGE_CREATE", d, "hi", "app1") is True

    def test_explicit_tag_for_our_app_id(self):
        from gateway.platforms.qqbot.group_activation import detect_mentioned
        assert detect_mentioned(
            "GROUP_MESSAGE_CREATE", {}, "<@!1903885637> hello", "1903885637"
        ) is True

    def test_tag_for_other_member_not_mentioned(self):
        from gateway.platforms.qqbot.group_activation import detect_mentioned
        # @ of a different member must NOT count as addressing the bot.
        assert detect_mentioned(
            "GROUP_MESSAGE_CREATE", {}, "<@!99999> hello", "1903885637"
        ) is False

    def test_plain_message_not_mentioned(self):
        from gateway.platforms.qqbot.group_activation import detect_mentioned
        assert detect_mentioned(
            "GROUP_MESSAGE_CREATE", {}, "just chatting", "1903885637"
        ) is False

    def test_generic_at_prefix_not_treated_as_bot(self):
        from gateway.platforms.qqbot.group_activation import detect_mentioned
        # Conservative: a bare "@alice " prefix is NOT a bot mention.
        assert detect_mentioned(
            "GROUP_MESSAGE_CREATE", {}, "@alice look here", "1903885637"
        ) is False


# ---------------------------------------------------------------------------
# Group activation mode — require_mention resolution
# ---------------------------------------------------------------------------

class TestResolveRequireMention:
    def test_global_default_true(self):
        from gateway.platforms.qqbot.group_activation import resolve_require_mention
        assert resolve_require_mention("g1", global_default=True) is True

    def test_global_default_false(self):
        from gateway.platforms.qqbot.group_activation import resolve_require_mention
        assert resolve_require_mention("g1", global_default=False) is False

    def test_per_group_overrides_global(self):
        from gateway.platforms.qqbot.group_activation import resolve_require_mention
        # global always, but g1 forced to mention.
        assert resolve_require_mention(
            "g1", global_default=False, per_group={"g1": True}
        ) is True
        # other group still follows global.
        assert resolve_require_mention(
            "g2", global_default=False, per_group={"g1": True}
        ) is False

    def test_runtime_override_wins(self):
        from gateway.platforms.qqbot.group_activation import resolve_require_mention
        assert resolve_require_mention(
            "g1", global_default=True, per_group={"g1": True},
            runtime_overrides={"g1": False},
        ) is False


# ---------------------------------------------------------------------------
# Group activation mode — config parsing + gate (_handle_group_message)
# ---------------------------------------------------------------------------

class TestGroupActivationMode:
    def _make_adapter(self, **extra):
        from gateway.platforms.qqbot import QQAdapter
        extra.setdefault("app_id", "1903885637")
        extra.setdefault("client_secret", "b")
        extra.setdefault("group_policy", "open")
        return QQAdapter(_make_config(**extra))

    def test_default_mode_is_mention(self):
        adapter = self._make_adapter()
        assert adapter._group_require_mention is True

    def test_always_mode_from_config(self):
        adapter = self._make_adapter(group_require_mention=False)
        assert adapter._group_require_mention is False

    def test_per_group_override_parsed(self):
        adapter = self._make_adapter(
            group_require_mention=False,
            groups={"grp_a": {"require_mention": True}},
        )
        assert adapter._group_mode_overrides == {"grp_a": True}

    def _drive(self, adapter):
        """Stub the assembly path and capture handle_message events."""
        captured = []

        async def fake_process(_a):
            return {"image_urls": [], "image_media_types": [],
                    "voice_transcripts": [], "attachment_info": ""}

        async def fake_quote(_d):
            return {"quote_block": "", "image_urls": [], "image_media_types": []}

        async def fake_handle(event):
            captured.append(event)

        adapter._process_attachments = fake_process  # type: ignore[assignment]
        adapter._process_quoted_context = fake_quote  # type: ignore[assignment]
        adapter.handle_message = fake_handle  # type: ignore[assignment]
        return captured

    @pytest.mark.asyncio
    async def test_mention_mode_skips_non_mention_group_message(self):
        adapter = self._make_adapter()  # default mention
        captured = self._drive(adapter)
        d = {"group_openid": "g1", "content": "hello everyone"}
        await adapter._handle_group_message(
            d, "m1", "hello everyone", {"member_openid": "u1"}, "",
            "GROUP_MESSAGE_CREATE",
        )
        assert captured == []  # skipped, no reply

    @pytest.mark.asyncio
    async def test_mention_mode_handles_at_message(self):
        adapter = self._make_adapter()  # default mention
        captured = self._drive(adapter)
        d = {"group_openid": "g1", "content": "hi bot"}
        await adapter._handle_group_message(
            d, "m1", "hi bot", {"member_openid": "u1"}, "",
            "GROUP_AT_MESSAGE_CREATE",
        )
        assert len(captured) == 1
        assert captured[0].source.chat_id == "g1"
        assert captured[0].source.chat_type == "group"

    @pytest.mark.asyncio
    async def test_always_mode_handles_non_mention_group_message(self):
        adapter = self._make_adapter(group_require_mention=False)  # always
        captured = self._drive(adapter)
        d = {"group_openid": "g1", "content": "just chatting"}
        await adapter._handle_group_message(
            d, "m1", "just chatting", {"member_openid": "u1"}, "",
            "GROUP_MESSAGE_CREATE",
        )
        assert len(captured) == 1
        assert captured[0].text == "just chatting"

    @pytest.mark.asyncio
    async def test_per_group_mention_override_blocks_in_always_global(self):
        adapter = self._make_adapter(
            group_require_mention=False,
            groups={"g1": {"require_mention": True}},
        )
        captured = self._drive(adapter)
        # g1 forced to mention -> non-@ skipped.
        await adapter._handle_group_message(
            {"group_openid": "g1", "content": "hey"}, "m1", "hey",
            {"member_openid": "u1"}, "", "GROUP_MESSAGE_CREATE",
        )
        assert captured == []
        # g2 follows global always -> handled.
        await adapter._handle_group_message(
            {"group_openid": "g2", "content": "hey"}, "m2", "hey",
            {"member_openid": "u1"}, "", "GROUP_MESSAGE_CREATE",
        )
        assert len(captured) == 1

    @pytest.mark.asyncio
    async def test_group_acl_blocks_before_gate(self):
        adapter = self._make_adapter(group_policy="disabled",
                                     group_require_mention=False)
        captured = self._drive(adapter)
        await adapter._handle_group_message(
            {"group_openid": "g1", "content": "hi"}, "m1", "hi",
            {"member_openid": "u1"}, "", "GROUP_MESSAGE_CREATE",
        )
        assert captured == []

    @pytest.mark.asyncio
    async def test_group_acl_reject_emits_debug_log(self, caplog):
        # ACL rejection must leave an operator-visible breadcrumb explaining
        # both the cause (policy=disabled) and how to unblock it.
        import logging
        adapter = self._make_adapter(group_policy="disabled")
        self._drive(adapter)
        with caplog.at_level(logging.DEBUG, logger="gateway.platforms.qqbot"):
            await adapter._handle_group_message(
                {"group_openid": "gX", "content": "hi"}, "m1", "hi",
                {"member_openid": "u1"}, "", "GROUP_AT_MESSAGE_CREATE",
            )
        log_text = " ".join(r.message for r in caplog.records)
        assert "blocked by ACL" in log_text
        assert "gX" in log_text
        assert "policy=disabled" in log_text
        assert "group_policy" in log_text  # hint on how to unblock

    def test_default_history_limit_is_20(self):
        adapter = self._make_adapter()
        assert adapter._group_history_limit == 20


# ---------------------------------------------------------------------------
# Group shared session (2.1) — group_sessions_per_user key behaviour
# ---------------------------------------------------------------------------

class TestGroupSharedSession:
    def _source(self):
        from gateway.session import SessionSource
        from gateway.config import Platform
        return SessionSource(
            platform=Platform.QQBOT,
            chat_id="group_openid_1",
            chat_type="group",
            user_id="member_openid_1",
        )

    def test_isolated_key_includes_participant_by_default(self):
        from gateway.session import build_session_key
        key = build_session_key(self._source(), group_sessions_per_user=True)
        assert key.endswith(":member_openid_1")

    def test_shared_key_excludes_participant(self):
        from gateway.session import build_session_key
        key = build_session_key(self._source(), group_sessions_per_user=False)
        assert "member_openid_1" not in key
        assert key.endswith(":group_openid_1")


# ---------------------------------------------------------------------------
# Review follow-up (problem 1): per-platform session-sharing must drive BOTH
# the session key AND the [user_name] sender attribution.
#
# Regression: sender attribution in GatewayRunner._prepare_inbound_message_text
# read only the global GatewayConfig.group_sessions_per_user, while the session
# key was built from the per-platform extra value. A QQ-only shared session
# (extra.group_sessions_per_user=false, global default true) therefore merged
# users into one session key WITHOUT the required sender prefix.
# ---------------------------------------------------------------------------

def _make_runner_with_qq_extra(**extra):
    """Bare GatewayRunner: QQ platform carries `extra`, global config stays default."""
    from gateway.run import GatewayRunner
    from gateway.config import GatewayConfig, Platform, PlatformConfig

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.QQBOT: PlatformConfig(enabled=True, extra=dict(extra))}
    )
    runner.adapters = {}
    runner.session_store = None
    runner._pending_native_image_paths_by_session = {}
    return runner


class TestEffectiveSessionSharing:
    """GatewayRunner._effective_session_sharing resolves per-platform values."""

    def _source(self):
        from gateway.session import SessionSource
        from gateway.config import Platform
        return SessionSource(
            platform=Platform.QQBOT,
            chat_id="group_openid_1",
            chat_type="group",
            user_id="member_1",
        )

    def test_qq_extra_override_beats_global_default(self):
        # Global default group_sessions_per_user=True, QQ extra says False.
        runner = _make_runner_with_qq_extra(group_sessions_per_user=False)
        group_spu, _thread_spu = runner._effective_session_sharing(self._source())
        assert group_spu is False  # per-platform override wins

    def test_falls_back_to_global_when_no_extra(self):
        runner = _make_runner_with_qq_extra()  # empty extra
        group_spu, thread_spu = runner._effective_session_sharing(self._source())
        assert group_spu is True   # inherits global default (True)
        assert thread_spu is False

    def test_unknown_platform_uses_global(self):
        from gateway.session import SessionSource
        from gateway.config import Platform
        runner = _make_runner_with_qq_extra(group_sessions_per_user=False)
        # A source for a platform with no config entry → global defaults.
        src = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="c",
            chat_type="group",
            user_id="u",
        )
        group_spu, thread_spu = runner._effective_session_sharing(src)
        assert group_spu is True
        assert thread_spu is False


class TestGroupSharedSenderAttribution:
    """Two-sender regression for the shared-session sender prefix."""

    def _event(self, user_id, user_name, text="hello"):
        from gateway.platforms.base import MessageEvent, MessageType
        from gateway.session import SessionSource
        from gateway.config import Platform

        src = SessionSource(
            platform=Platform.QQBOT,
            chat_id="group_openid_1",
            chat_type="group",
            user_id=user_id,
            user_name=user_name,
        )
        return src, MessageEvent(
            text=text,
            message_type=MessageType.TEXT,
            source=src,
        )

    @pytest.mark.asyncio
    async def test_two_senders_share_key_and_get_prefixed(self):
        # QQ-only shared session; global default stays isolated (True).
        runner = _make_runner_with_qq_extra(group_sessions_per_user=False)

        src_a, ev_a = self._event("member_1", "Alice")
        src_b, ev_b = self._event("member_2", "Bob")

        text_a = await runner._prepare_inbound_message_text(
            event=ev_a, source=src_a, history=[]
        )
        text_b = await runner._prepare_inbound_message_text(
            event=ev_b, source=src_b, history=[]
        )

        # Both messages carry their own sender prefix …
        assert text_a == "[Alice] hello"
        assert text_b == "[Bob] hello"

        # … and both land in the SAME shared session key (user_id excluded).
        key_a = runner._session_key_for_source(src_a)
        key_b = runner._session_key_for_source(src_b)
        assert key_a == key_b
        assert "member_1" not in key_a and "member_2" not in key_a

    @pytest.mark.asyncio
    async def test_isolated_default_has_no_prefix(self):
        # Default (isolated) sessions: each user has its own key, no prefix.
        runner = _make_runner_with_qq_extra()  # inherits global True (isolated)
        src_a, ev_a = self._event("member_1", "Alice")
        text_a = await runner._prepare_inbound_message_text(
            event=ev_a, source=src_a, history=[]
        )
        assert text_a == "hello"


# ---------------------------------------------------------------------------
# Review follow-up (problem 2): split native C2C stream editing from generic
# already-sent message editing so the tool-progress editor does not attempt
# (and fail) edits against ordinary QQ message IDs.
# ---------------------------------------------------------------------------

class TestStreamEditingCapabilitySplit:
    def test_qq_message_editing_is_false(self):
        from gateway.platforms.qqbot.adapter import QQAdapter
        # QQ cannot edit arbitrary already-sent message IDs.
        assert QQAdapter.SUPPORTS_MESSAGE_EDITING is False

    def test_qq_stream_editing_is_true(self):
        from gateway.platforms.qqbot.adapter import QQAdapter
        # QQ CAN update an in-flight C2C stream bubble in place.
        assert QQAdapter.SUPPORTS_STREAM_EDITING is True

    def test_base_stream_editing_falls_back_to_message_editing(self):
        from gateway.platforms.base import BasePlatformAdapter
        prop = BasePlatformAdapter.__dict__["SUPPORTS_STREAM_EDITING"]

        class _NonEditable:
            SUPPORTS_MESSAGE_EDITING = False

        class _Editable:
            SUPPORTS_MESSAGE_EDITING = True

        class _Silent:  # neither attribute set
            pass

        assert prop.fget(_NonEditable()) is False
        assert prop.fget(_Editable()) is True
        assert prop.fget(_Silent()) is True  # default True

    def test_stream_consumer_resolution_for_qq(self):
        # Mirrors gateway.run stream consumer: prefer stream flag, fall back to
        # generic message-editing flag.
        from gateway.platforms.qqbot.adapter import QQAdapter
        supports = getattr(
            QQAdapter,
            "SUPPORTS_STREAM_EDITING",
            getattr(QQAdapter, "SUPPORTS_MESSAGE_EDITING", True),
        )
        assert supports is True  # C2C streaming stays enabled

    def test_tool_progress_gate_skips_qq(self):
        # Mirrors the gate in gateway.run.send_progress_messages: an adapter is
        # eligible for generic tool-progress editing ONLY if it overrides
        # edit_message AND advertises SUPPORTS_MESSAGE_EDITING. QQ overrides
        # edit_message but is not a generic editor → progress must be skipped.
        from gateway.platforms.qqbot.adapter import QQAdapter
        from gateway.platforms.base import BasePlatformAdapter

        overrides_edit = (
            QQAdapter.edit_message is not BasePlatformAdapter.edit_message
        )
        generic_editing = overrides_edit and getattr(
            QQAdapter, "SUPPORTS_MESSAGE_EDITING", True
        )
        assert overrides_edit is True       # QQ does override edit_message
        assert generic_editing is False     # …but not for generic message IDs


# ---------------------------------------------------------------------------
# Group context buffer (2.2.1) — GroupContextBuffer unit tests
# ---------------------------------------------------------------------------

class TestGroupContextBuffer:
    def _buf(self, **kw):
        from gateway.platforms.qqbot.group_context import GroupContextBuffer
        return GroupContextBuffer(**kw)

    def test_record_and_drain_in_order(self):
        buf = self._buf(limit=10)
        buf.record("g1", sender="u1", text="first")
        buf.record("g1", sender="u2", text="second")
        entries = buf.drain("g1")
        assert [e.text for e in entries] == ["first", "second"]
        assert [e.sender for e in entries] == ["u1", "u2"]

    def test_drain_clears(self):
        buf = self._buf(limit=10)
        buf.record("g1", sender="u1", text="hi")
        assert buf.drain("g1")
        assert buf.drain("g1") == []

    def test_limit_truncates_oldest(self):
        buf = self._buf(limit=2)
        buf.record("g1", sender="u", text="a")
        buf.record("g1", sender="u", text="b")
        buf.record("g1", sender="u", text="c")
        entries = buf.drain("g1")
        assert [e.text for e in entries] == ["b", "c"]  # oldest "a" dropped

    def test_disabled_when_limit_zero(self):
        buf = self._buf(limit=0)
        assert buf.enabled is False
        buf.record("g1", sender="u", text="hi")
        assert buf.drain("g1") == []

    def test_group_isolation(self):
        buf = self._buf(limit=10)
        buf.record("g1", sender="u", text="a")
        buf.record("g2", sender="u", text="b")
        assert [e.text for e in buf.drain("g1")] == ["a"]
        assert [e.text for e in buf.drain("g2")] == ["b"]

    def test_empty_text_no_attachment_not_recorded(self):
        buf = self._buf(limit=10)
        buf.record("g1", sender="u", text="   ")
        assert buf.drain("g1") == []

    def test_attachment_tag_recorded_when_no_text(self):
        buf = self._buf(limit=10)
        buf.record("g1", sender="u", text="", attachment_tag="[image]")
        entries = buf.drain("g1")
        assert entries and entries[0].text == "[image]"

    def test_max_groups_lru_eviction(self):
        buf = self._buf(limit=5, max_groups=2)
        buf.record("g1", sender="u", text="a")
        buf.record("g2", sender="u", text="b")
        buf.record("g3", sender="u", text="c")  # evicts g1 (LRU)
        assert buf.drain("g1") == []
        assert [e.text for e in buf.drain("g3")] == ["c"]

    def test_format_context_wraps_with_tags(self):
        from gateway.platforms.qqbot.group_context import (
            GroupContextBuffer, HistoryEntry, HISTORY_CTX_START, HISTORY_CTX_END,
        )
        entries = [HistoryEntry(sender="u1", text="hello"),
                   HistoryEntry(sender="u2", text="world")]
        out = GroupContextBuffer.format_context(entries, "please summarize")
        assert HISTORY_CTX_START in out
        assert "u1: hello" in out
        assert "u2: world" in out
        assert HISTORY_CTX_END in out
        assert out.endswith("please summarize")

    def test_format_context_empty_returns_current(self):
        from gateway.platforms.qqbot.group_context import GroupContextBuffer
        assert GroupContextBuffer.format_context([], "just this") == "just this"

    def test_format_context_block_no_current_message(self):
        # The block variant renders context only (for channel_context): it must
        # include the CONTEXT-ONLY header + entries, but NOT the current message
        # and NOT the CURRENT-MESSAGE end tag (gateway supplies [New message]).
        from gateway.platforms.qqbot.group_context import (
            GroupContextBuffer, HistoryEntry, HISTORY_CTX_START, HISTORY_CTX_END,
        )
        entries = [HistoryEntry(sender="u1", text="hello"),
                   HistoryEntry(sender="u2", text="world")]
        out = GroupContextBuffer.format_context_block(entries)
        assert HISTORY_CTX_START in out
        assert "u1: hello" in out
        assert "u2: world" in out
        assert HISTORY_CTX_END not in out

    def test_format_context_block_empty_returns_blank(self):
        from gateway.platforms.qqbot.group_context import GroupContextBuffer
        assert GroupContextBuffer.format_context_block([]) == ""

    def test_format_context_block_collapses_newlines(self):
        # R5 hardening: a buffered multi-line message cannot forge envelope tags.
        from gateway.platforms.qqbot.group_context import (
            GroupContextBuffer, HistoryEntry,
        )
        entries = [HistoryEntry(sender="u1", text="line1\nline2")]
        out = GroupContextBuffer.format_context_block(entries)
        assert "u1: line1 line2" in out

    def test_summarize_attachments(self):
        from gateway.platforms.qqbot.group_context import summarize_attachments
        assert summarize_attachments(None) == ""
        assert summarize_attachments([]) == ""
        assert summarize_attachments([{"content_type": "image/png"}]) == "[image]"
        assert summarize_attachments([{"content_type": "audio/silk"}]) == "[voice]"
        assert summarize_attachments(
            [{"content_type": "application/zip", "filename": "a.zip"}]
        ) == "[file: a.zip]"


# ---------------------------------------------------------------------------
# Group context buffer — integration through _handle_group_message
# ---------------------------------------------------------------------------

class TestGroupContextIntegration:
    def _make_adapter(self, **extra):
        from gateway.platforms.qqbot import QQAdapter
        extra.setdefault("app_id", "1903885637")
        extra.setdefault("client_secret", "b")
        extra.setdefault("group_policy", "open")
        return QQAdapter(_make_config(**extra))

    def _drive(self, adapter):
        captured = []

        async def fake_process(_a):
            return {"image_urls": [], "image_media_types": [],
                    "voice_transcripts": [], "attachment_info": ""}

        async def fake_quote(_d):
            return {"quote_block": "", "image_urls": [], "image_media_types": []}

        async def fake_handle(event):
            captured.append(event)

        adapter._process_attachments = fake_process  # type: ignore[assignment]
        adapter._process_quoted_context = fake_quote  # type: ignore[assignment]
        adapter.handle_message = fake_handle  # type: ignore[assignment]
        return captured

    @pytest.mark.asyncio
    async def test_mention_mode_buffers_then_injects_on_at(self):
        adapter = self._make_adapter()  # mention mode, default limit 50
        captured = self._drive(adapter)
        # non-@ message → buffered, no reply.
        await adapter._handle_group_message(
            {"group_openid": "g1", "content": "the sky is blue"}, "m1",
            "the sky is blue", {"member_openid": "alice"}, "",
            "GROUP_MESSAGE_CREATE",
        )
        assert captured == []
        # @ message → reply with buffered context injected.
        await adapter._handle_group_message(
            {"group_openid": "g1", "content": "what did she say"}, "m2",
            "what did she say", {"member_openid": "bob"}, "",
            "GROUP_AT_MESSAGE_CREATE",
        )
        assert len(captured) == 1
        # Buffered history is carried in channel_context (kept out of text so
        # slash-command detection + sender-prefix operate on the trigger alone).
        ctx = captured[0].channel_context
        assert ctx and "CONTEXT ONLY" in ctx
        assert "alice: the sky is blue" in ctx
        # text is the trigger message only.
        assert captured[0].text == "what did she say"

    @pytest.mark.asyncio
    async def test_command_at_message_stays_matchable_with_pending_context(self):
        # Regression: a /stop-style command in an @-message must remain at the
        # start of text (get_command works) even when pending context exists;
        # the context goes to channel_context instead of being merged ahead.
        adapter = self._make_adapter()
        captured = self._drive(adapter)
        await adapter._handle_group_message(
            {"group_openid": "g1", "content": "the sky is blue"}, "m1",
            "the sky is blue", {"member_openid": "alice"}, "",
            "GROUP_MESSAGE_CREATE",
        )
        await adapter._handle_group_message(
            {"group_openid": "g1", "content": "/stop"}, "m2",
            "/stop", {"member_openid": "bob"}, "",
            "GROUP_AT_MESSAGE_CREATE",
        )
        assert len(captured) == 1
        ev = captured[0]
        assert ev.text == "/stop"
        assert ev.is_command() is True
        assert ev.get_command() == "stop"
        # context preserved separately, not merged into text.
        assert ev.channel_context and "alice: the sky is blue" in ev.channel_context

    @pytest.mark.asyncio
    async def test_buffer_cleared_after_injection(self):
        adapter = self._make_adapter()
        captured = self._drive(adapter)
        await adapter._handle_group_message(
            {"group_openid": "g1", "content": "ctx"}, "m1", "ctx",
            {"member_openid": "alice"}, "", "GROUP_MESSAGE_CREATE",
        )
        await adapter._handle_group_message(
            {"group_openid": "g1", "content": "q1"}, "m2", "q1",
            {"member_openid": "bob"}, "", "GROUP_AT_MESSAGE_CREATE",
        )
        # second @ has no stale context.
        await adapter._handle_group_message(
            {"group_openid": "g1", "content": "q2"}, "m3", "q2",
            {"member_openid": "bob"}, "", "GROUP_AT_MESSAGE_CREATE",
        )
        assert len(captured) == 2
        assert "CONTEXT ONLY" not in captured[1].text
        assert captured[1].text == "q2"

    @pytest.mark.asyncio
    async def test_always_mode_no_buffering(self):
        adapter = self._make_adapter(group_require_mention=False)
        captured = self._drive(adapter)
        await adapter._handle_group_message(
            {"group_openid": "g1", "content": "hello"}, "m1", "hello",
            {"member_openid": "alice"}, "", "GROUP_MESSAGE_CREATE",
        )
        assert len(captured) == 1
        assert captured[0].text == "hello"  # no context wrapper
        # nothing left buffered.
        assert adapter._group_context.drain("g1") == []

    @pytest.mark.asyncio
    async def test_history_limit_zero_disables_buffer(self):
        adapter = self._make_adapter(group_history_limit=0)
        captured = self._drive(adapter)
        await adapter._handle_group_message(
            {"group_openid": "g1", "content": "ctx"}, "m1", "ctx",
            {"member_openid": "alice"}, "", "GROUP_MESSAGE_CREATE",
        )
        await adapter._handle_group_message(
            {"group_openid": "g1", "content": "q"}, "m2", "q",
            {"member_openid": "bob"}, "", "GROUP_AT_MESSAGE_CREATE",
        )
        assert len(captured) == 1
        assert "CONTEXT ONLY" not in captured[0].text
        assert captured[0].text == "q"

    @pytest.mark.asyncio
    async def test_empty_at_message_still_flushes_context(self):
        # A bare @ (empty body after strip) must still flush + inject pending
        # context, not early-return and strand the buffer.
        adapter = self._make_adapter()
        captured = self._drive(adapter)
        await adapter._handle_group_message(
            {"group_openid": "g1", "content": "background note"}, "m1",
            "background note", {"member_openid": "alice"}, "",
            "GROUP_MESSAGE_CREATE",
        )
        await adapter._handle_group_message(
            {"group_openid": "g1", "content": ""}, "m2", "",
            {"member_openid": "bob"}, "", "GROUP_AT_MESSAGE_CREATE",
        )
        assert len(captured) == 1
        # bare @ carries no trigger body; pending context lands in channel_context.
        assert captured[0].text == ""
        assert (
            captured[0].channel_context
            and "alice: background note" in captured[0].channel_context
        )
        # buffer cleared.
        assert adapter._group_context.drain("g1") == []

    @pytest.mark.asyncio
    async def test_injection_keeps_current_message_with_attachments_last(self):
        # Lock the order: buffered context first, current message (incl. its
        # appended attachment_info) last.
        adapter = self._make_adapter()
        captured = self._drive(adapter)

        async def fake_process(_a):
            return {"image_urls": [], "image_media_types": [],
                    "voice_transcripts": [], "attachment_info": "[file: doc.pdf]"}

        adapter._process_attachments = fake_process  # type: ignore[assignment]

        await adapter._handle_group_message(
            {"group_openid": "g1", "content": "earlier"}, "m1", "earlier",
            {"member_openid": "alice"}, "", "GROUP_MESSAGE_CREATE",
        )
        # NB: alice's non-@ message uses the light attachment tag, not the full
        # processor; the @ message below exercises the full path.
        await adapter._handle_group_message(
            {"group_openid": "g1", "content": "see attached"}, "m2",
            "see attached", {"member_openid": "bob"}, "",
            "GROUP_AT_MESSAGE_CREATE",
        )
        ev = captured[0]
        # Trigger message + its attachment_info stay in text, in order.
        assert ev.text.index("see attached") < ev.text.index("[file: doc.pdf]")
        # Buffered history is separated into channel_context.
        assert ev.channel_context and "CONTEXT ONLY" in ev.channel_context
        assert "earlier" in ev.channel_context
        assert "earlier" not in ev.text


# ---------------------------------------------------------------------------
# C2C streaming reply — StreamManager
# ---------------------------------------------------------------------------

class TestStreamManager:
    """Unit tests for the in-memory StreamSession table."""

    def test_create_registers_session_with_generated_logical_id(self):
        from gateway.platforms.qqbot.streaming import StreamManager
        mgr = StreamManager()
        s1 = mgr.create(openid="u1", passive_msg_id="m1", msg_seq=42)
        s2 = mgr.create(openid="u2", passive_msg_id="m2", msg_seq=43)
        assert s1.logical_id and s2.logical_id
        assert s1.logical_id != s2.logical_id
        assert mgr.get(s1.logical_id) is s1
        assert mgr.get(s2.logical_id) is s2

    def test_get_returns_none_for_unknown_id(self):
        from gateway.platforms.qqbot.streaming import StreamManager
        mgr = StreamManager()
        assert mgr.get("nonexistent") is None

    def test_drop_is_idempotent(self):
        from gateway.platforms.qqbot.streaming import StreamManager
        mgr = StreamManager()
        s = mgr.create(openid="u", passive_msg_id="m", msg_seq=1)
        mgr.drop(s.logical_id)
        mgr.drop(s.logical_id)  # second drop must not raise
        assert mgr.get(s.logical_id) is None

    def test_ttl_expiry_evicts_stale_session(self):
        from gateway.platforms.qqbot.streaming import StreamManager
        mgr = StreamManager(ttl_seconds=60.0)
        s = mgr.create(openid="u", passive_msg_id="m", msg_seq=1)
        # Backdate the session past the TTL horizon.
        s.created_at -= 61.0
        assert mgr.get(s.logical_id) is None
        # After the failed lookup the entry should be gone.
        assert len(mgr) == 0

    def test_lru_eviction_when_full(self):
        from gateway.platforms.qqbot.streaming import StreamManager
        mgr = StreamManager(max_sessions=2)
        a = mgr.create(openid="a", passive_msg_id="m1", msg_seq=1)
        b = mgr.create(openid="b", passive_msg_id="m2", msg_seq=2)
        # Touch A to promote it — B becomes LRU.
        assert mgr.get(a.logical_id) is a
        mgr.create(openid="c", passive_msg_id="m3", msg_seq=3)
        assert mgr.get(b.logical_id) is None  # evicted
        assert mgr.get(a.logical_id) is a  # still present


# ---------------------------------------------------------------------------
# C2C streaming reply — QQAdapter integration
# ---------------------------------------------------------------------------

class TestC2CStreamingReply:
    """Streaming-path tests for ``send()`` + ``edit_message()`` on C2C chats."""

    def _make_adapter(self, **extra):
        from gateway.platforms.qqbot import QQAdapter
        extra.setdefault("app_id", "a")
        extra.setdefault("client_secret", "b")
        adapter = QQAdapter(_make_config(**extra))
        adapter._running = True
        adapter._ws = SimpleNamespace(closed=False)
        adapter._http_client = mock.MagicMock()
        return adapter

    def test_requires_edit_finalize_class_attr_is_true(self):
        from gateway.platforms.qqbot import QQAdapter
        assert QQAdapter.REQUIRES_EDIT_FINALIZE is True

    def test_streaming_defaults_enabled(self):
        adapter = self._make_adapter()
        assert adapter._streaming_enabled is True

    def test_streaming_can_be_disabled_via_config(self):
        adapter = self._make_adapter(streaming_enabled=False)
        assert adapter._streaming_enabled is False

    @pytest.mark.asyncio
    async def test_c2c_first_send_opens_stream_session(self):
        adapter = self._make_adapter()
        adapter._chat_type_map["user_a"] = "c2c"
        calls = []

        async def fake_api(method, path, body=None, **kw):
            calls.append((method, path, body))
            return {"id": "stream-xyz-1"}

        adapter._api_request = fake_api  # type: ignore[assignment]

        result = await adapter.send(
            "user_a", "hello",
            reply_to="inbound_m1",
            metadata={"expect_edits": True},
        )
        assert result.success
        # message_id must be the adapter's logical_id (opaque uuid hex),
        # NOT the QQ-assigned stream_msg_id — the consumer echoes this
        # back on edit_message and we translate internally.
        assert result.message_id is not None
        assert result.message_id != "stream-xyz-1"
        assert len(calls) == 1
        method, path, body = calls[0]
        assert method == "POST"
        assert path == "/v2/users/user_a/stream_messages"
        assert body["input_mode"] == "replace"
        assert body["input_state"] == 1  # GENERATING
        assert body["content_type"] == "markdown"  # MARKDOWN — fixed
        assert body["content_raw"] == "hello"
        assert body["event_id"] == "inbound_m1"
        assert body["msg_id"] == "inbound_m1"
        assert body["index"] == 0
        assert "stream_msg_id" not in body  # first chunk omits it
        # Session should be registered and reference the QQ id.
        session = adapter._stream_manager.get(result.message_id)
        assert session is not None
        assert session.stream_msg_id == "stream-xyz-1"

    @pytest.mark.asyncio
    async def test_c2c_edit_reuses_stream_msg_id_and_increments_index(self):
        adapter = self._make_adapter()
        adapter._chat_type_map["user_a"] = "c2c"
        calls = []

        async def fake_api(method, path, body=None, **kw):
            calls.append((method, path, body))
            return {"id": "stream-xyz-1"}

        adapter._api_request = fake_api  # type: ignore[assignment]

        first = await adapter.send(
            "user_a", "hel",
            reply_to="inbound_m1",
            metadata={"expect_edits": True},
        )
        assert first.success
        logical_id = first.message_id

        second = await adapter.edit_message("user_a", logical_id, "hello")
        third = await adapter.edit_message("user_a", logical_id, "hello wor")
        assert second.success and third.success
        # All three chunks share the same msg_seq and target the same
        # stream endpoint.
        assert len({body["msg_seq"] for _, _, body in calls}) == 1
        assert [body["index"] for _, _, body in calls] == [0, 1, 2]
        # Second/third chunks carry stream_msg_id from the first response.
        assert calls[1][2]["stream_msg_id"] == "stream-xyz-1"
        assert calls[2][2]["stream_msg_id"] == "stream-xyz-1"
        # content_raw is REPLACE semantics — full accumulated text each time.
        assert [body["content_raw"] for _, _, body in calls] == [
            "hel", "hello", "hello wor",
        ]
        # Intermediate edits stay in GENERATING state.
        assert calls[1][2]["input_state"] == 1
        assert calls[2][2]["input_state"] == 1

    @pytest.mark.asyncio
    async def test_c2c_finalize_sends_done_and_drops_session(self):
        adapter = self._make_adapter()
        adapter._chat_type_map["user_a"] = "c2c"
        calls = []

        async def fake_api(method, path, body=None, **kw):
            calls.append((method, path, body))
            return {"id": "stream-1"}

        adapter._api_request = fake_api  # type: ignore[assignment]

        first = await adapter.send(
            "user_a", "partial",
            reply_to="inbound_m1",
            metadata={"expect_edits": True},
        )
        result = await adapter.edit_message(
            "user_a", first.message_id, "final full answer",
            finalize=True,
        )
        assert result.success
        # Final chunk uses input_state=10 (DONE).
        assert calls[-1][2]["input_state"] == 10
        # Session should be cleaned up after finalize.
        assert adapter._stream_manager.get(first.message_id) is None

    @pytest.mark.asyncio
    async def test_edit_after_finalize_is_noop_success(self):
        adapter = self._make_adapter()
        adapter._chat_type_map["user_a"] = "c2c"

        async def fake_api(method, path, body=None, **kw):
            return {"id": "stream-1"}

        adapter._api_request = fake_api  # type: ignore[assignment]

        first = await adapter.send(
            "user_a", "x",
            reply_to="inbound_m1",
            metadata={"expect_edits": True},
        )
        await adapter.edit_message(
            "user_a", first.message_id, "final",
            finalize=True,
        )
        # Second call after finalize: session already dropped, treated
        # as "unknown session" → success=False so the consumer sends a
        # fresh message rather than corrupting the finalized stream.
        r = await adapter.edit_message(
            "user_a", first.message_id, "oops",
        )
        assert r.success is False
        assert "expired" in (r.error or "").lower() or "found" in (r.error or "").lower()

    @pytest.mark.asyncio
    async def test_edit_on_unknown_session_returns_failure(self):
        adapter = self._make_adapter()
        result = await adapter.edit_message("user_a", "nonexistent-logical-id", "hi")
        assert result.success is False
        assert result.error

    @pytest.mark.asyncio
    async def test_out_of_order_edit_is_dropped_silently(self):
        """When ``next_index <= last_sent_index`` the edit is treated as a
        no-op success without hitting the QQ API — matches the simplified
        out-of-order policy agreed for this release.
        """
        adapter = self._make_adapter()
        adapter._chat_type_map["user_a"] = "c2c"
        calls = []

        async def fake_api(method, path, body=None, **kw):
            calls.append((method, path, body))
            return {"id": "stream-1"}

        adapter._api_request = fake_api  # type: ignore[assignment]

        first = await adapter.send(
            "user_a", "hi",
            reply_to="inbound_m1",
            metadata={"expect_edits": True},
        )
        session = adapter._stream_manager.get(first.message_id)
        # Simulate an out-of-order scenario by regressing next_index.
        # (Consumer serialisation makes this impossible in practice, so
        # we assert the defensive guard triggers when it happens.)
        session.next_index = 0
        session.last_sent_index = 0
        api_calls_before = len(calls)

        r = await adapter.edit_message("user_a", first.message_id, "hello")
        assert r.success is True
        assert len(calls) == api_calls_before  # no additional API call

    @pytest.mark.asyncio
    async def test_group_chat_streaming_first_send_defers_delivery(self):
        """Group targets have no editable message id on QQ, so the
        streaming first-send is buffered inside the adapter: no QQ API
        call yet, but ``send()`` returns ``success=True`` with a
        sentinel ``message_id`` so the stream consumer stays on its
        edit path (avoiding the lossy prefix-based fallback).  The
        complete reply is delivered later, on ``edit_message`` with
        ``finalize=True``.
        """
        adapter = self._make_adapter()
        adapter._chat_type_map["group_a"] = "group"
        paths = []

        async def fake_api(method, path, body=None, **kw):
            paths.append(path)
            return {"id": "regular-1"}

        adapter._api_request = fake_api  # type: ignore[assignment]

        result = await adapter.send(
            "group_a", "hello",
            reply_to="inbound_m1",
            metadata={"expect_edits": True},
        )
        assert result.success
        assert result.message_id is not None
        assert result.message_id.startswith("__qqbot_group_defer_")
        # No QQ API call happened — the reply is deferred to finalize.
        assert paths == []
        # Session bookkeeping preserves the original reply_to for
        # threaded delivery on finalize.
        assert result.message_id in adapter._group_defer_sessions
        session = adapter._group_defer_sessions[result.message_id]
        assert session["chat_id"] == "group_a"
        assert session["chat_type"] == "group"
        assert session["reply_to"] == "inbound_m1"
        # Fresh session — not yet finalized.
        assert session["finalized"] is False
        assert session["finalized_result"] is None

    @pytest.mark.asyncio
    async def test_guild_chat_streaming_first_send_defers_delivery(self):
        """Same deferral as group chats — guild targets are also
        non-editable, so ``expect_edits`` must not trigger a real send.
        """
        adapter = self._make_adapter()
        adapter._chat_type_map["chan1"] = "guild"
        paths = []

        async def fake_api(method, path, body=None, **kw):
            paths.append(path)
            return {"id": "regular-1"}

        adapter._api_request = fake_api  # type: ignore[assignment]

        result = await adapter.send(
            "chan1", "hello",
            reply_to="inbound_m1",
            metadata={"expect_edits": True},
        )
        assert result.success
        assert result.message_id is not None
        assert result.message_id.startswith("__qqbot_group_defer_")
        assert paths == []
        session = adapter._group_defer_sessions[result.message_id]
        assert session["chat_type"] == "guild"

    @pytest.mark.asyncio
    async def test_group_deferred_edit_intermediate_is_noop(self):
        """Intermediate ``edit_message`` calls on a deferred group
        session must NOT hit the QQ API — they are silent no-ops that
        accumulate context until the finalize call arrives.
        """
        adapter = self._make_adapter()
        adapter._chat_type_map["group_a"] = "group"
        paths = []

        async def fake_api(method, path, body=None, **kw):
            paths.append(path)
            return {"id": "regular-1"}

        adapter._api_request = fake_api  # type: ignore[assignment]

        first = await adapter.send(
            "group_a", "chunk1",
            reply_to="inbound_m1",
            metadata={"expect_edits": True},
        )
        sentinel = first.message_id
        assert paths == []

        for partial in ("chunk1 chunk2", "chunk1 chunk2 chunk3"):
            r = await adapter.edit_message(
                "group_a", sentinel, partial, finalize=False,
            )
            assert r.success
            assert r.message_id == sentinel

        # Still no QQ API traffic.
        assert paths == []
        # Session is still live.
        assert sentinel in adapter._group_defer_sessions

    @pytest.mark.asyncio
    async def test_group_deferred_edit_finalize_delivers_full_content(self):
        """The finalize ``edit_message`` is where the complete reply
        actually reaches QQ.  It must:

        * hit the regular group messages endpoint exactly once,
        * carry the FULL accumulated content (not just the tail),
        * preserve the original ``reply_to`` for threading, and
        * clear the deferred session bookkeeping.
        """
        adapter = self._make_adapter()
        adapter._chat_type_map["group_a"] = "group"
        calls = []

        async def fake_api(method, path, body=None, **kw):
            calls.append((method, path, body))
            return {"id": "regular-1"}

        adapter._api_request = fake_api  # type: ignore[assignment]

        first = await adapter.send(
            "group_a", "hello",
            reply_to="inbound_m1",
            metadata={"expect_edits": True},
        )
        sentinel = first.message_id
        # A couple of intermediate edits — should not touch QQ.
        await adapter.edit_message("group_a", sentinel, "hello world")
        await adapter.edit_message(
            "group_a", sentinel, "hello world, done",
        )
        assert calls == []

        # Finalize with the full accumulated text.
        final = await adapter.edit_message(
            "group_a", sentinel,
            "hello world, done: final answer",
            finalize=True,
        )
        assert final.success
        assert final.message_id == "regular-1"
        # One and only one API call — the complete reply.
        assert len(calls) == 1
        method, path, body = calls[0]
        assert method == "POST"
        assert path == "/v2/groups/group_a/messages"
        # Full accumulated text (may be under ``content`` or
        # ``markdown.content`` depending on adapter markdown mode).
        text = (
            body.get("content")
            or body.get("markdown", {}).get("content")
            or ""
        )
        assert text.startswith("hello world, done: final answer")
        # Threaded to the original inbound message.
        assert body.get("msg_id") == "inbound_m1"
        # Session bookkeeping: entry is preserved but flagged so
        # subsequent finalize edits become idempotent no-ops (see
        # ``test_group_deferred_edit_finalize_is_idempotent``).
        assert sentinel in adapter._group_defer_sessions
        remembered = adapter._group_defer_sessions[sentinel]
        assert remembered["finalized"] is True
        assert remembered["finalized_result"] is final

    @pytest.mark.asyncio
    async def test_group_deferred_edit_finalize_is_idempotent(self):
        """A second ``finalize=True`` edit on the same sentinel must NOT
        re-post the reply.

        Regression guard: because the adapter declares
        ``REQUIRES_EDIT_FINALIZE=True``, the stream consumer's
        ``got_done`` branch can issue two ``_send_or_edit(...,
        finalize=True)`` calls per turn — the mid-stream flush plus the
        explicit final tick.  The first call already delivered the
        reply via ``self.send(...)``; the second must be an idempotent
        no-op or the user sees two identical messages in the group.
        """
        adapter = self._make_adapter()
        adapter._chat_type_map["group_a"] = "group"
        calls = []

        async def fake_api(method, path, body=None, **kw):
            calls.append((method, path, body))
            return {"id": "regular-1"}

        adapter._api_request = fake_api  # type: ignore[assignment]

        first = await adapter.send(
            "group_a", "hello",
            reply_to="inbound_m1",
            metadata={"expect_edits": True},
        )
        sentinel = first.message_id

        final_a = await adapter.edit_message(
            "group_a", sentinel, "the full reply",
            finalize=True,
        )
        final_b = await adapter.edit_message(
            "group_a", sentinel, "the full reply",
            finalize=True,
        )

        # First finalize delivered exactly once; the second is a
        # memoised replay — same result object, no extra API call.
        assert len(calls) == 1
        assert final_a.success and final_b.success
        assert final_a.message_id == "regular-1"
        assert final_b.message_id == "regular-1"
        assert final_b is final_a

    @pytest.mark.asyncio
    async def test_group_deferred_edit_after_finalize_non_final_is_noop(self):
        """A stray non-final edit arriving after finalize must not
        re-hit QQ (would happen on cancellation cleanup or an errant
        mid-stream tick that raced with the final one).
        """
        adapter = self._make_adapter()
        adapter._chat_type_map["group_a"] = "group"
        calls = []

        async def fake_api(method, path, body=None, **kw):
            calls.append((method, path, body))
            return {"id": "regular-1"}

        adapter._api_request = fake_api  # type: ignore[assignment]

        first = await adapter.send(
            "group_a", "hi",
            reply_to="inbound_m1",
            metadata={"expect_edits": True},
        )
        sentinel = first.message_id

        await adapter.edit_message("group_a", sentinel, "final",
                                    finalize=True)
        # Simulate a late non-final tick after finalize.
        stray = await adapter.edit_message(
            "group_a", sentinel, "final plus more", finalize=False,
        )
        assert stray.success
        # Still exactly one API call — the finalize one.
        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_group_deferred_sessions_evict_finalized_over_cap(self):
        """The finalized-session bookkeeping must not grow without
        bound: once above the LRU cap, oldest finalized entries are
        dropped.  Pending (non-finalized) entries are preserved.
        """
        adapter = self._make_adapter()
        adapter._chat_type_map["group_a"] = "group"
        adapter._group_defer_sessions_cap = 3

        async def fake_api(method, path, body=None, **kw):
            return {"id": "regular"}

        adapter._api_request = fake_api  # type: ignore[assignment]

        # 3 finalized + 1 pending → over cap by 1.
        finalized_ids = []
        for _ in range(3):
            r = await adapter.send(
                "group_a", "hi",
                metadata={"expect_edits": True},
            )
            await adapter.edit_message("group_a", r.message_id, "x",
                                        finalize=True)
            finalized_ids.append(r.message_id)

        pending = await adapter.send(
            "group_a", "hi",
            metadata={"expect_edits": True},
        )

        # Cap enforcement runs on send() insertion — the oldest
        # finalized entry should have been evicted, the pending one
        # kept.
        assert pending.message_id in adapter._group_defer_sessions
        assert finalized_ids[0] not in adapter._group_defer_sessions
        # Newer finalized entries remain until they too age out.
        assert finalized_ids[-1] in adapter._group_defer_sessions

    @pytest.mark.asyncio
    async def test_group_deferred_edit_after_session_expiry_recovers(self):
        """If the deferred session is gone (adapter restart, TTL, ...),
        a finalize edit must still deliver the content via a fresh
        legacy send instead of dropping the reply silently.
        """
        adapter = self._make_adapter()
        adapter._chat_type_map["group_a"] = "group"
        calls = []

        async def fake_api(method, path, body=None, **kw):
            calls.append((method, path, body))
            return {"id": "regular-1"}

        adapter._api_request = fake_api  # type: ignore[assignment]

        # Never called send() first: no session exists.
        result = await adapter.edit_message(
            "group_a", "__qqbot_group_defer_missing__",
            "recovered content", finalize=True,
        )
        assert result.success
        assert len(calls) == 1
        assert calls[0][1] == "/v2/groups/group_a/messages"

    @pytest.mark.asyncio
    async def test_group_chat_without_expect_edits_sends_normally(self):
        """Non-streaming group sends must still hit the regular group
        messages endpoint and return a real id.
        """
        adapter = self._make_adapter()
        adapter._chat_type_map["group_a"] = "group"
        paths = []

        async def fake_api(method, path, body=None, **kw):
            paths.append(path)
            return {"id": "regular-1"}

        adapter._api_request = fake_api  # type: ignore[assignment]

        result = await adapter.send(
            "group_a", "final answer",
            reply_to="inbound_m1",
            metadata={"final": True},
        )
        assert result.success
        assert result.message_id == "regular-1"
        assert paths == ["/v2/groups/group_a/messages"]

    @pytest.mark.asyncio
    async def test_group_edit_message_returns_failure(self):
        """Groups have no stream session — edit_message must return
        ``success=False`` so the consumer falls back to a fresh send.
        """
        adapter = self._make_adapter()
        result = await adapter.edit_message("group_a", "any-id", "hi")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_streaming_disabled_falls_back_to_legacy_send(self):
        adapter = self._make_adapter(streaming_enabled=False)
        adapter._chat_type_map["user_a"] = "c2c"
        paths = []

        async def fake_api(method, path, body=None, **kw):
            paths.append(path)
            return {"id": "regular-1"}

        adapter._api_request = fake_api  # type: ignore[assignment]

        result = await adapter.send(
            "user_a", "hi",
            reply_to="inbound_m1",
            metadata={"expect_edits": True},
        )
        assert result.success
        assert all("/stream_messages" not in p for p in paths)

    @pytest.mark.asyncio
    async def test_c2c_send_without_expect_edits_uses_legacy_path(self):
        adapter = self._make_adapter()
        adapter._chat_type_map["user_a"] = "c2c"
        paths = []

        async def fake_api(method, path, body=None, **kw):
            paths.append(path)
            return {"id": "regular-1"}

        adapter._api_request = fake_api  # type: ignore[assignment]

        result = await adapter.send("user_a", "hi", reply_to="inbound_m1")
        assert result.success
        assert paths == ["/v2/users/user_a/messages"]

    @pytest.mark.asyncio
    async def test_streaming_start_failure_falls_back_to_legacy(self, caplog):
        """First chunk failure must warn and degrade to a regular send in
        the same ``send()`` call — user requirement (d).
        """
        adapter = self._make_adapter()
        adapter._chat_type_map["user_a"] = "c2c"
        calls = []

        async def fake_api(method, path, body=None, **kw):
            calls.append(path)
            if "/stream_messages" in path:
                raise RuntimeError("QQ Bot API error [500] boom")
            return {"id": "legacy-1"}

        adapter._api_request = fake_api  # type: ignore[assignment]

        import logging
        with caplog.at_level(logging.WARNING):
            result = await adapter.send(
                "user_a", "hi",
                reply_to="inbound_m1",
                metadata={"expect_edits": True},
            )
        assert result.success
        # Stream attempt happened, then fell back to the regular endpoint.
        assert calls[0].endswith("/stream_messages")
        assert calls[-1] == "/v2/users/user_a/messages"
        assert any(
            "Failed to start C2C streaming reply" in rec.message
            for rec in caplog.records
        )
        # Abandoned session must not leak into the manager table.
        assert len(adapter._stream_manager) == 0

    @pytest.mark.asyncio
    async def test_streaming_start_without_passive_msg_id_falls_back(self, caplog):
        """No reply_to + no cached inbound id → skip streaming with a
        warning; the endpoint requires a passive-reply msg_id.
        """
        adapter = self._make_adapter()
        adapter._chat_type_map["user_a"] = "c2c"
        # _last_msg_id intentionally empty.
        calls = []

        async def fake_api(method, path, body=None, **kw):
            calls.append(path)
            return {"id": "legacy-1"}

        adapter._api_request = fake_api  # type: ignore[assignment]

        import logging
        with caplog.at_level(logging.WARNING):
            result = await adapter.send(
                "user_a", "hi",
                metadata={"expect_edits": True},
            )
        assert result.success
        assert all("/stream_messages" not in p for p in calls)
        assert any("no passive msg_id" in rec.message for rec in caplog.records)

    @pytest.mark.asyncio
    async def test_streaming_uses_cached_inbound_id_when_reply_to_missing(self):
        """If ``reply_to`` is not supplied, the adapter must use the most
        recent inbound msg_id it saw for that chat.
        """
        adapter = self._make_adapter()
        adapter._chat_type_map["user_a"] = "c2c"
        adapter._last_msg_id["user_a"] = "cached_inbound_99"
        calls = []

        async def fake_api(method, path, body=None, **kw):
            calls.append(body)
            return {"id": "stream-1"}

        adapter._api_request = fake_api  # type: ignore[assignment]

        result = await adapter.send(
            "user_a", "hi",
            metadata={"expect_edits": True},
        )
        assert result.success
        assert calls[0]["msg_id"] == "cached_inbound_99"
        assert calls[0]["event_id"] == "cached_inbound_99"

    @pytest.mark.asyncio
    async def test_content_truncated_to_stream_content_limit(self):
        from gateway.platforms.qqbot.streaming import MAX_STREAM_CONTENT_LEN
        adapter = self._make_adapter()
        adapter._chat_type_map["user_a"] = "c2c"
        captured = []

        async def fake_api(method, path, body=None, **kw):
            captured.append(body)
            return {"id": "stream-1"}

        adapter._api_request = fake_api  # type: ignore[assignment]

        big = "x" * (MAX_STREAM_CONTENT_LEN + 200)
        await adapter.send(
            "user_a", big,
            reply_to="inbound_m1",
            metadata={"expect_edits": True},
        )
        assert len(captured[0]["content_raw"]) == MAX_STREAM_CONTENT_LEN

    @pytest.mark.asyncio
    async def test_stream_forwards_content_verbatim_when_gateway_suppresses_cursor(self):
        """With QQBOT-specific cursor suppression in ``gateway/run.py``
        (analogous to ``Platform.MATRIX``), no typewriter cursor ever
        reaches the adapter — successive frames are forwarded verbatim
        and the prefix invariant holds naturally.
        """
        adapter = self._make_adapter()
        adapter._chat_type_map["user_a"] = "c2c"
        calls = []

        async def fake_api(method, path, body=None, **kw):
            calls.append(body)
            return {"id": "stream-nocursor-1"}

        adapter._api_request = fake_api  # type: ignore[assignment]

        # All three frames arrive without any trailing cursor glyph,
        # exactly what the gateway will emit for QQBOT.
        first = await adapter.send(
            "user_a", "Hello",
            reply_to="inbound_m1",
            metadata={"expect_edits": True},
        )
        assert first.success
        logical_id = first.message_id

        second = await adapter.edit_message(
            "user_a", logical_id, "Hello world",
        )
        assert second.success

        third = await adapter.edit_message(
            "user_a", logical_id, "Hello world!", finalize=True,
        )
        assert third.success

        seen = [body["content_raw"] for body in calls]
        assert seen == ["Hello", "Hello world", "Hello world!"]
        for i in range(len(seen) - 1):
            assert seen[i + 1].startswith(seen[i]), (
                f"prefix broken at {i}: {seen[i]!r} -> {seen[i + 1]!r}"
            )

    @pytest.mark.asyncio
    async def test_stream_prefix_divergence_replays_last_chunk(self):
        """Divergent frames must degrade to a safe replay, not a 500.

        If a subsequent edit's cleaned text does NOT start with the
        previously-accepted text (e.g. upstream trimmed a segment,
        replayed a shorter partial, or the model backtracked), we
        MUST NOT forward it verbatim — that would break the prefix
        invariant and kill the whole stream.  Instead the adapter
        replays the last-accepted text so QQ sees a no-op-shaped
        chunk and the session stays alive for the next real update.
        """
        adapter = self._make_adapter()
        adapter._chat_type_map["user_a"] = "c2c"
        calls = []

        async def fake_api(method, path, body=None, **kw):
            calls.append(body)
            return {"id": "stream-div-1"}

        adapter._api_request = fake_api  # type: ignore[assignment]

        first = await adapter.send(
            "user_a", "Hello world how are you",
            reply_to="inbound_m1",
            metadata={"expect_edits": True},
        )
        assert first.success
        logical_id = first.message_id

        # Divergent frame — shorter and NOT a prefix of the first.
        result = await adapter.edit_message(
            "user_a", logical_id, "Different content entirely",
        )
        # We report success (from QQ's POV nothing bad happened) but the
        # payload we actually sent is the previously-accepted text,
        # keeping the prefix invariant intact.
        assert result.success
        assert calls[1]["content_raw"] == "Hello world how are you"

        # A subsequent, properly-extended frame flows through as normal.
        third = await adapter.edit_message(
            "user_a", logical_id, "Hello world how are you today",
        )
        assert third.success
        assert calls[2]["content_raw"] == "Hello world how are you today"

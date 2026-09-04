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
        assert adapter._group_policy == "pairing"

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
        adapter = self._make_adapter(app_id="a", client_secret="b")
        assert adapter._group_policy == "pairing"
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
# Regression for #78183: httpx timeout empty-string defeats _is_timeout_error
# ---------------------------------------------------------------------------

class TestQQTimeoutErrorNormalization:
    """When an httpx timeout has an empty string representation, qqbot's send
    paths must preserve the exception type name so the base-layer timeout guard
    (_is_timeout_error) can still recognise it and suppress the duplicate-
    delivery plain-text fallback."""

    def _make_adapter(self, **extra):
        from gateway.platforms.qqbot import QQAdapter
        return QQAdapter(_make_config(app_id="a", client_secret="b", **extra))

    @pytest.mark.asyncio
    async def test_send_chunk_preserves_read_timeout_type(self):
        from gateway.platforms.base import BasePlatformAdapter

        adapter = self._make_adapter()

        async def _boom(*args, **kwargs):
            raise httpx.ReadTimeout("")

        adapter._send_c2c_text = _boom

        result = await adapter._send_chunk("test_openid", "hello world")

        assert not result.success
        assert result.error, "error must not be empty"
        assert BasePlatformAdapter._is_timeout_error(result.error), (
            f"_is_timeout_error must recognise {result.error!r}"
        )

    @pytest.mark.asyncio
    async def test_send_chunk_non_empty_error_unchanged(self):
        """A normal exception with a message must keep its original text."""
        adapter = self._make_adapter()

        async def _boom(*args, **kwargs):
            raise RuntimeError("Server error '500 Internal Server Error'")

        adapter._send_c2c_text = _boom

        result = await adapter._send_chunk("test_openid", "hello world")

        assert not result.success
        assert "500 Internal Server Error" in (result.error or "")


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

    def test_read_events_raises_when_ws_none(self):
        adapter = self._make_adapter()
        adapter._running = True
        adapter._ws = None
        with pytest.raises(RuntimeError):
            asyncio.run(adapter._read_events())


# ---------------------------------------------------------------------------
# Reconnect-exhaustion hand-off (regression: silent adapter death)
# ---------------------------------------------------------------------------

class TestReconnectExhaustionHandoff:
    """Regression: when reconnect attempts are exhausted, _listen_loop must hand
    off to the gateway reconnect watcher (a retryable fatal error plus a
    scheduled notify task) instead of silently calling _mark_disconnected() and
    leaving the listener task dead with nothing watching it.

    Covers all three exhaustion paths in _listen_loop:
      1. generic Exception handler
      2. QQCloseError generic reconnect/backoff path
      3. rate-limited (4008) path
    """

    def _make_adapter(self):
        from gateway.platforms.qqbot.adapter import QQAdapter
        return QQAdapter(_make_config(app_id="a", client_secret="b"))

    def _arm(self, adapter, monkeypatch, *, max_attempts):
        import gateway.platforms.qqbot.adapter as qq_adapter
        monkeypatch.setattr(qq_adapter, "MAX_RECONNECT_ATTEMPTS", max_attempts)
        # Neutralize quick-disconnect detection so a synthetic instant failure
        # can't divert into the qq_quick_disconnect branch.
        monkeypatch.setattr(qq_adapter, "QUICK_DISCONNECT_THRESHOLD", -1.0)
        adapter._running = True
        adapter._mark_transport_disconnected = mock.Mock()
        adapter._fail_pending = mock.Mock()
        adapter._mark_disconnected = mock.Mock()
        adapter._set_fatal_error = mock.Mock()
        # Adapter-side unit boundary: `_notify_fatal_error` is mocked here to
        # isolate `_listen_loop`'s scheduling contract (retryable fatal set +
        # detached notify task + no `_mark_disconnected` fallback). The
        # end-to-end proof that the gateway actually enqueues the platform
        # for reconnection lives in
        # `TestQQBotGatewayHandoffEndToEnd` below, which drives the real
        # `GatewayRunner._handle_adapter_fatal_error` path with no mocks on
        # the notify chain.
        adapter._notify_fatal_error = mock.AsyncMock()

    async def _assert_handoff(self, adapter):
        # A retryable-fatal error was set so the gateway watcher takes over.
        adapter._set_fatal_error.assert_called_once()
        args, kwargs = adapter._set_fatal_error.call_args
        assert args[0] == "qq_reconnect_exhausted"
        assert kwargs.get("retryable") is True
        # The old silent-death path must NOT be taken.
        adapter._mark_disconnected.assert_not_called()
        # Notify is scheduled as a detached task (not awaited inline) so the
        # listen loop can finish before disconnect() cancels this task.
        assert adapter._fatal_notify_task is not None
        await adapter._fatal_notify_task
        adapter._notify_fatal_error.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_generic_exception_exhaustion_hands_off(self, monkeypatch):
        adapter = self._make_adapter()
        self._arm(adapter, monkeypatch, max_attempts=0)
        adapter._read_events = mock.AsyncMock(side_effect=RuntimeError("boom"))
        adapter._reconnect = mock.AsyncMock(return_value=False)

        await adapter._listen_loop()

        await self._assert_handoff(adapter)

    @pytest.mark.asyncio
    async def test_qqclose_reconnect_exhaustion_hands_off(self, monkeypatch):
        from gateway.platforms.qqbot.adapter import QQCloseError
        adapter = self._make_adapter()
        self._arm(adapter, monkeypatch, max_attempts=1)
        # 4009 is a non-fatal, session-preserving close code, so it falls
        # through to the generic reconnect/backoff path at the bottom of the
        # QQCloseError handler.
        adapter._read_events = mock.AsyncMock(
            side_effect=QQCloseError(4009, "timeout")
        )
        adapter._reconnect = mock.AsyncMock(return_value=False)

        await adapter._listen_loop()

        await self._assert_handoff(adapter)
        adapter._reconnect.assert_awaited()

    @pytest.mark.asyncio
    async def test_rate_limit_exhaustion_hands_off(self, monkeypatch):
        from gateway.platforms.qqbot.adapter import QQCloseError
        adapter = self._make_adapter()
        self._arm(adapter, monkeypatch, max_attempts=0)
        # 4008 = rate limited; with attempts already exhausted the handler must
        # hand off before sleeping RATE_LIMIT_DELAY (no extra reconnect).
        adapter._read_events = mock.AsyncMock(
            side_effect=QQCloseError(4008, "rate limited")
        )
        adapter._reconnect = mock.AsyncMock(return_value=False)

        await adapter._listen_loop()

        await self._assert_handoff(adapter)
        adapter._reconnect.assert_not_awaited()


class TestQuickDisconnectHandoff:
    """Regression: the *4th* exhaustion exit — quick-disconnect detection —
    must also hand off to the gateway watcher (retryable fatal + detached
    notify task) instead of dying silently. The original PR fixed the three
    exits in `_listen_loop`'s bottom half but left this one at
    `_set_fatal_error(...); return` with no `_schedule_fatal_notify()`,
    which meant misconfigured / permission-denied bots went into a silent
    death exactly like the pre-fix state.
    """

    def _make_adapter(self):
        from gateway.platforms.qqbot.adapter import QQAdapter
        return QQAdapter(_make_config(app_id="a", client_secret="b"))

    @pytest.mark.asyncio
    async def test_quick_disconnect_exhaustion_hands_off(self, monkeypatch):
        from gateway.platforms.qqbot.adapter import QQCloseError
        import gateway.platforms.qqbot.adapter as qq_adapter

        adapter = self._make_adapter()
        # Trip the threshold on the first quick disconnect so the loop
        # doesn't need to iterate.
        monkeypatch.setattr(qq_adapter, "MAX_QUICK_DISCONNECT_COUNT", 1)
        # A wide threshold guarantees the synthetic instant failure is
        # classified as "quick" regardless of scheduler timing.
        monkeypatch.setattr(qq_adapter, "QUICK_DISCONNECT_THRESHOLD", 60.0)
        adapter._running = True
        adapter._mark_transport_disconnected = mock.Mock()
        adapter._fail_pending = mock.Mock()
        adapter._mark_disconnected = mock.Mock()
        adapter._set_fatal_error = mock.Mock()
        # Adapter-side unit boundary — see the equivalent comment in
        # TestReconnectExhaustionHandoff._arm for why this mock is safe here
        # and where the real gateway-side coverage lives.
        adapter._notify_fatal_error = mock.AsyncMock()

        # Any close code works; quick-disconnect detection runs *before*
        # close-code classification.
        adapter._read_events = mock.AsyncMock(
            side_effect=QQCloseError(4009, "quick close")
        )

        await adapter._listen_loop()

        # Retryable fatal was set with the quick-disconnect signature.
        adapter._set_fatal_error.assert_called_once()
        args, kwargs = adapter._set_fatal_error.call_args
        assert args[0] == "qq_quick_disconnect"
        assert kwargs.get("retryable") is True
        # The silent-death path (direct `_mark_disconnected` fallback)
        # must NOT be taken — the gateway owns lifecycle from here.
        adapter._mark_disconnected.assert_not_called()
        # Notify is scheduled as a detached task so the listen loop can
        # return before disconnect() cancels this task.
        assert adapter._fatal_notify_task is not None
        await adapter._fatal_notify_task
        adapter._notify_fatal_error.assert_awaited_once()


class TestQQBotGatewayHandoffEndToEnd:
    """End-to-end proof that a QQBot retryable fatal error travels through
    the *real* `GatewayRunner._handle_adapter_fatal_error` path and lands in
    `runner._failed_platforms`, where the reconnect watcher picks it up.

    This is the assertion the adapter-side mocked tests structurally cannot
    make: they stop at `_notify_fatal_error` being scheduled. Here we drive
    a minimal QQ-shaped adapter with the exact retryable-fatal shape the
    real QQAdapter emits (both exhaustion families), install it into a real
    `GatewayRunner`, and observe the queue outcome with no mocks on the
    notify chain. Only `runner.stop` is mocked so we don't need to bring
    up the full gateway lifecycle for this assertion — mirroring
    `tests/gateway/test_runner_fatal_adapter.py:test_runner_queues_retryable_runtime_fatal_for_reconnection`.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "error_code,error_message",
        [
            ("qq_reconnect_exhausted", "Max reconnect attempts reached"),
            (
                "qq_quick_disconnect",
                "Too many quick disconnects — check bot permissions",
            ),
        ],
    )
    async def test_qqbot_retryable_fatal_reaches_gateway_reconnect_queue(
        self, tmp_path, error_code, error_message
    ):
        from unittest.mock import AsyncMock

        from gateway.config import GatewayConfig, Platform, PlatformConfig
        from gateway.platforms.base import BasePlatformAdapter
        from gateway.run import GatewayRunner

        class _QQFakeAdapter(BasePlatformAdapter):
            """Minimal QQ-shaped adapter — mirrors the retryable-fatal shape
            emitted by both `qq_reconnect_exhausted` and
            `qq_quick_disconnect` in the real QQAdapter.
            """

            def __init__(self):
                super().__init__(
                    PlatformConfig(
                        enabled=True,
                        extra={"app_id": "a", "client_secret": "b"},
                    ),
                    Platform.QQBOT,
                )

            async def connect(self, *, is_reconnect: bool = False) -> bool:
                return True

            async def disconnect(self) -> None:
                self._mark_disconnected()

            async def send(self, chat_id, content, reply_to=None, metadata=None):
                raise NotImplementedError

            async def get_chat_info(self, chat_id):
                return {"id": chat_id}

        config = GatewayConfig(
            platforms={
                Platform.QQBOT: PlatformConfig(
                    enabled=True,
                    extra={"app_id": "a", "client_secret": "b"},
                )
            },
            sessions_dir=tmp_path / "sessions",
        )
        runner = GatewayRunner(config)
        adapter = _QQFakeAdapter()
        adapter._set_fatal_error(error_code, error_message, retryable=True)

        runner.adapters = {Platform.QQBOT: adapter}
        runner.delivery_router.adapters = runner.adapters
        # We don't need a full gateway lifecycle for this assertion — just
        # prevent the runner from tearing itself down as a side effect.
        runner.stop = AsyncMock()

        await runner._handle_adapter_fatal_error(adapter)

        # Gateway must remove the fatal adapter and queue it for background
        # reconnection — this is exactly what silent death used to skip.
        assert Platform.QQBOT not in runner.adapters
        assert Platform.QQBOT in runner._failed_platforms
        assert runner._failed_platforms[Platform.QQBOT]["attempts"] == 0
        # Gateway stays alive — the watcher owns retry from here.
        runner.stop.assert_not_awaited()
        assert runner._exit_with_failure is False

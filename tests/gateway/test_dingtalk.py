"""Tests for DingTalk platform adapter."""
import asyncio
import json
import os
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform, PlatformConfig


class _FakeDingTalkModel:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _FakeChatbotMessage(SimpleNamespace):
    @classmethod
    def from_dict(cls, data):
        data = data or {}
        return cls(
            message_id=data.get("msgId") or data.get("messageId") or data.get("message_id") or "",
            conversation_id=data.get("conversationId") or data.get("conversation_id") or "",
            conversation_type=str(data.get("conversationType") or data.get("conversation_type") or "1"),
            sender_id=data.get("senderId") or data.get("sender_id") or "",
            sender_staff_id=data.get("senderStaffId") or data.get("sender_staff_id") or data.get("senderId") or "",
            sender_nick=data.get("senderNick") or data.get("sender_nick") or "",
            text=data.get("text") or "",
            rich_text=data.get("richText") or data.get("rich_text"),
            rich_text_content=data.get("richTextContent") or data.get("rich_text_content"),
            session_webhook=data.get("sessionWebhook") or data.get("session_webhook") or "",
            session_webhook_expired_time=data.get("sessionWebhookExpiredTime") or data.get("session_webhook_expired_time") or 0,
            create_at=data.get("createAt") or data.get("create_at") or 0,
            at_users=data.get("atUsers") or data.get("at_users") or [],
            is_in_at_list=bool(data.get("isInAtList") or data.get("is_in_at_list")),
        )


@pytest.fixture(autouse=True)
def _fake_dingtalk_optional_sdks(monkeypatch):
    """Keep DingTalk adapter tests hermetic when optional SDKs are absent."""
    import plugins.platforms.dingtalk.adapter as dt

    card_models = SimpleNamespace(**{
        name: _FakeDingTalkModel
        for name in (
            "CreateCardRequest",
            "CreateCardRequestCardData",
            "CreateCardRequestImGroupOpenSpaceModel",
            "CreateCardRequestImRobotOpenSpaceModel",
            "CreateCardHeaders",
            "DeliverCardRequest",
            "DeliverCardRequestImGroupOpenDeliverModel",
            "DeliverCardRequestImRobotOpenDeliverModel",
            "DeliverCardHeaders",
            "StreamingUpdateRequest",
            "StreamingUpdateHeaders",
        )
    })
    robot_models = SimpleNamespace(**{
        name: _FakeDingTalkModel
        for name in (
            "RobotReplyEmotionRequestTextEmotion",
            "RobotReplyEmotionRequest",
            "RobotReplyEmotionHeaders",
            "RobotRecallEmotionRequestTextEmotion",
            "RobotRecallEmotionRequest",
            "RobotRecallEmotionHeaders",
            "RobotMessageFileDownloadRequest",
            "RobotMessageFileDownloadHeaders",
        )
    })

    monkeypatch.setattr(dt, "ChatbotMessage", _FakeChatbotMessage, raising=False)
    monkeypatch.setattr(
        dt,
        "AckMessage",
        SimpleNamespace(STATUS_OK=200, STATUS_SYSTEM_EXCEPTION=500),
        raising=False,
    )
    monkeypatch.setattr(dt, "tea_util_models", SimpleNamespace(RuntimeOptions=_FakeDingTalkModel), raising=False)
    monkeypatch.setattr(dt, "dingtalk_card_models", card_models, raising=False)
    monkeypatch.setattr(dt, "dingtalk_robot_models", robot_models, raising=False)


# ---------------------------------------------------------------------------
# Requirements check
# ---------------------------------------------------------------------------


class TestDingTalkRequirements:


    def test_returns_false_when_env_vars_missing(self, monkeypatch):
        monkeypatch.setattr(
            "plugins.platforms.dingtalk.adapter.DINGTALK_STREAM_AVAILABLE", True
        )
        monkeypatch.setattr("plugins.platforms.dingtalk.adapter.HTTPX_AVAILABLE", True)
        monkeypatch.delenv("DINGTALK_CLIENT_ID", raising=False)
        monkeypatch.delenv("DINGTALK_CLIENT_SECRET", raising=False)
        from plugins.platforms.dingtalk.adapter import check_dingtalk_requirements
        assert check_dingtalk_requirements() is False


# ---------------------------------------------------------------------------
# Adapter construction
# ---------------------------------------------------------------------------


class TestDingTalkAdapterInit:

    def test_reads_config_from_extra(self):
        from plugins.platforms.dingtalk.adapter import DingTalkAdapter
        config = PlatformConfig(
            enabled=True,
            extra={"client_id": "cfg-id", "client_secret": "cfg-secret"},
        )
        adapter = DingTalkAdapter(config)
        assert adapter._client_id == "cfg-id"
        assert adapter._client_secret == "cfg-secret"
        assert adapter.name == "Dingtalk"  # base class uses .title()


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


class TestDeduplication:

    def test_first_message_not_duplicate(self):
        from plugins.platforms.dingtalk.adapter import DingTalkAdapter
        adapter = DingTalkAdapter(PlatformConfig(enabled=True))
        assert adapter._dedup.is_duplicate("msg-1") is False


# ---------------------------------------------------------------------------
# Send
# ---------------------------------------------------------------------------


class TestSend:

    @pytest.mark.asyncio
    async def test_send_posts_to_webhook(self):
        from plugins.platforms.dingtalk.adapter import DingTalkAdapter
        adapter = DingTalkAdapter(PlatformConfig(enabled=True))

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "OK"

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        adapter._http_client = mock_client

        result = await adapter.send(
            "chat-123", "Hello!",
            metadata={"session_webhook": "https://dingtalk.example/webhook"}
        )
        assert result.success is True
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert call_args[0][0] == "https://dingtalk.example/webhook"
        payload = call_args[1]["json"]
        assert payload["msgtype"] == "markdown"
        assert payload["markdown"]["title"] == "Hermes"
        assert payload["markdown"]["text"] == "Hello!"


    @pytest.mark.asyncio
    async def test_send_image_renders_markdown_image(self):
        from plugins.platforms.dingtalk.adapter import DingTalkAdapter
        adapter = DingTalkAdapter(PlatformConfig(enabled=True))

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "OK"

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        adapter._http_client = mock_client

        result = await adapter.send_image(
            "chat-123",
            "https://example.com/demo.png",
            caption="Screenshot",
            metadata={"session_webhook": "https://dingtalk.example/webhook"},
        )

        assert result.success is True
        payload = mock_client.post.call_args.kwargs["json"]
        assert payload["msgtype"] == "markdown"
        assert payload["markdown"]["text"] == "Screenshot\n\n![image](https://example.com/demo.png)"


# ---------------------------------------------------------------------------
# Connect / disconnect
# ---------------------------------------------------------------------------


class TestNativeVoice:
    @pytest.fixture
    def adapter(self):
        from plugins.platforms.dingtalk.adapter import DingTalkAdapter
        adapter = DingTalkAdapter(PlatformConfig(enabled=True, extra={
            "client_id": "robot-test", "client_secret": "secret-test",
            "allowed_users": ["staff-test"],
        }))
        adapter._mark_connected()
        adapter._get_access_token = AsyncMock(return_value="token-sentinel")
        adapter._resolve_media_codes = AsyncMock()
        adapter.handle_message = AsyncMock()
        return adapter

    async def inbound(self, adapter, conversation_type="2"):
        message = _FakeChatbotMessage.from_dict({
            "msgId": "message-test", "conversationId": "chat-test",
            "conversationType": conversation_type, "senderId": "encrypted-test",
            "senderStaffId": "staff-test", "text": {"content": "hello"},
            "sessionWebhook": "https://oapi.dingtalk.com/robot/send?access_token=webhook-test",
            "sessionWebhookExpiredTime": int(datetime.now(timezone.utc).timestamp() * 1000) + 3600000,
        })
        await adapter._on_message(message)
        assert adapter.handle_message.await_count == 1
        return message

    @pytest.mark.asyncio
    @pytest.mark.parametrize("conversation_type", ["1", "2"])
    async def test_inbound_route_upload_and_caption_contract(self, adapter, monkeypatch, conversation_type):
        import httpx
        from plugins.platforms.dingtalk import voice
        message = await self.inbound(adapter, conversation_type)
        media = b"#!AMR\nfixture-only"
        monkeypatch.setattr(voice, "prepare_voice", AsyncMock(return_value=(media, 1234)))
        requests = []

        def transport(request):
            requests.append(request)
            if request.url.path == "/media/upload":
                assert request.url.params["access_token"] == "token-sentinel"
                assert b'name="type"\r\n\r\nvoice' in request.content
                assert b"Content-Type: audio/amr" in request.content
                assert media in request.content
                return httpx.Response(200, json={"errcode": 0, "media_id": "media-test"})
            if request.url.path == "/robot/send":
                assert json.loads(request.content)["markdown"]["text"] == "caption-test"
                return httpx.Response(200, json={"errcode": 0})
            payload = json.loads(request.content)
            assert request.headers["x-acs-dingtalk-access-token"] == "token-sentinel"
            assert payload["msgKey"] == "sampleAudio"
            assert payload["robotCode"] == "robot-test"
            assert json.loads(payload["msgParam"]) == {"mediaId": "media-test", "duration": "1234"}
            if conversation_type == "2":
                assert request.url.path.endswith("/groupMessages/send")
                assert payload["openConversationId"] == message.conversation_id
                assert "userIds" not in payload
            else:
                assert request.url.path.endswith("/oToMessages/batchSend")
                assert payload["userIds"] == [message.sender_staff_id]
                assert "openConversationId" not in payload
            return httpx.Response(200, json={"processQueryKey": "task-test"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
            adapter._http_client = client
            result = await adapter.send_voice("chat-test", "fixture.amr", caption="caption-test",
                                              metadata={"session_webhook": "https://untrusted.invalid", "userIds": ["wrong"]})
        assert result.success and result.message_id == "task-test"
        assert len(requests) == 3
        assert requests[-1].url == message.session_webhook

    @pytest.mark.asyncio
    @pytest.mark.parametrize("case", ["missing", "expired", "mismatch", "unknown_type", "missing_staff", "denied", "disconnected"])
    async def test_route_fails_closed_before_local_or_network_io(self, adapter, monkeypatch, case):
        from plugins.platforms.dingtalk import voice
        message = await self.inbound(adapter, "1")
        adapter._http_client = AsyncMock()
        prepare = AsyncMock()
        monkeypatch.setattr(voice, "prepare_voice", prepare)
        if case == "missing":
            adapter._message_contexts.clear()
        elif case == "expired":
            adapter._session_webhooks["chat-test"] = (message.session_webhook, 1)
        elif case == "mismatch":
            message.conversation_id = "another-chat"
        elif case == "unknown_type":
            message.conversation_type = "3"
        elif case == "missing_staff":
            message.sender_staff_id = ""
        elif case == "denied":
            adapter._allowed_users = {"someone-else"}
        else:
            adapter._mark_disconnected()
        result = await adapter.send_voice("chat-test", "unused.amr")
        assert not result.success
        prepare.assert_not_awaited()
        adapter._get_access_token.assert_not_awaited()
        adapter._http_client.post.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("stage,status,data", [
        ("upload", 403, {"message": "token-sentinel"}),
        ("upload", 200, {"errcode": 40014, "media_id": "not-accepted"}),
        ("upload", 200, {}), ("upload", 200, []),
        ("send", 503, {}), ("send", 200, {"code": "Forbidden", "processQueryKey": "not-accepted"}),
        ("send", 200, {"code": "", "errcode": 1, "processQueryKey": "not-accepted"}),
        ("send", 200, {}), ("send", 200, {"processQueryKey": {"invalid": "type"}}),
        ("send", 200, {"processQueryKey": "not-accepted", "invalidStaffIdList": ["staff-test"]}),
        ("send", 200, {"processQueryKey": "not-accepted", "flowControlledStaffIdList": ["staff-test"]}),
    ])
    async def test_provider_acceptance_is_required(self, adapter, monkeypatch, stage, status, data):
        import httpx
        from plugins.platforms.dingtalk import voice
        await self.inbound(adapter, "1")
        monkeypatch.setattr(voice, "prepare_voice", AsyncMock(return_value=(b"#!AMR\nfixture", 800)))
        requests = []
        def transport(request):
            requests.append(request)
            if stage == "send" and len(requests) == 1:
                return httpx.Response(200, json={"media_id": "media-test"})
            return httpx.Response(status, json=data)
        async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
            adapter._http_client = client
            result = await adapter.send_voice("chat-test", "unused.amr", caption="must-not-send")
        assert not result.success and result.message_id is None
        assert "token-sentinel" not in result.error and "not-accepted" not in result.error
        assert len(requests) == (1 if stage == "upload" else 2)

    @pytest.mark.asyncio
    async def test_token_and_transport_errors_are_redacted(self, adapter, monkeypatch, caplog):
        from plugins.platforms.dingtalk import voice
        from plugins.platforms.dingtalk.adapter import DingTalkAdapter
        await self.inbound(adapter)
        monkeypatch.setattr(voice, "prepare_voice", AsyncMock(return_value=(b"#!AMR\nfixture", 800)))
        adapter._http_client = AsyncMock()
        adapter._http_client.post.side_effect = RuntimeError("token-sentinel private-path")
        result = await adapter.send_voice("chat-test", "unused.amr")
        assert not result.success
        assert "token-sentinel" not in result.error + caplog.text
        adapter._stream_client = MagicMock()
        adapter._stream_client.get_access_token.side_effect = RuntimeError("secret-sentinel")
        assert await DingTalkAdapter._get_access_token(adapter) is None
        assert "secret-sentinel" not in caplog.text

    @pytest.mark.asyncio
    async def test_caption_partial_failure_preserves_voice_receipt(self, adapter, monkeypatch):
        import httpx
        from plugins.platforms.dingtalk import voice
        from gateway.platforms.base import SendResult
        await self.inbound(adapter)
        monkeypatch.setattr(voice, "prepare_voice", AsyncMock(return_value=(b"#!AMR\nfixture", 800)))
        adapter.send = AsyncMock(return_value=SendResult(success=False, error="private-provider-detail"))
        def transport(request):
            return httpx.Response(200, json={"media_id": "media-test", "processQueryKey": "task-test"})
        async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
            adapter._http_client = client
            result = await adapter.send_voice("chat-test", "unused.amr", caption="caption-test")
        assert result.success and result.message_id == "task-test"
        assert result.raw_response == {"voice_sent": True, "caption_sent": False}
        assert result.error is None


    @pytest.mark.asyncio
    async def test_local_wav_to_native_voice_smoke(self, adapter, tmp_path):
        import httpx
        import shutil
        import wave
        from plugins.platforms.dingtalk.voice import prepare_voice, run_audio_tool
        if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            pytest.skip("ffmpeg and ffprobe are required for the local audio smoke")
        encoders = await run_audio_tool("ffmpeg", "-hide_banner", "-encoders", timeout=10)
        if b"libopencore_amrnb" not in encoders:
            pytest.skip("ffmpeg has no libopencore_amrnb encoder")
        source = tmp_path / "synthetic.wav"
        with wave.open(str(source), "wb") as audio:
            audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            audio.writeframes(b"\0\0" * 16000)
        original = source.read_bytes()
        await self.inbound(adapter)
        requests = []
        def transport(request):
            requests.append(request)
            if request.url.path == "/media/upload":
                assert b"#!AMR\n" in request.content
                return httpx.Response(200, json={"errcode": 0, "media_id": "local-smoke-media"})
            duration = int(json.loads(json.loads(request.content)["msgParam"])["duration"])
            # AMR framing and ffprobe's bitrate estimate add small padding.
            assert 1000 <= duration <= 1100
            return httpx.Response(200, json={"processQueryKey": "local-smoke-task"})
        async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
            adapter._http_client = client
            result = await adapter.play_tts("chat-test", str(source))
        assert result.success and result.message_id == "local-smoke-task"
        assert len(requests) == 2 and source.read_bytes() == original
        media, duration = await prepare_voice(str(source))
        amr = tmp_path / "existing.amr"
        amr.write_bytes(media)
        assert await prepare_voice(str(amr)) == (media, duration)
        print(f"LOCAL AUDIO SMOKE: WAV -> AMR-NB, 8000 Hz, mono; {len(media)} bytes; {duration} ms; mock upload/send accepted")


class TestVoicePreparation:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("probe", [
        {}, {"streams": []},
        {"streams": [{"codec_name": "amr_wb", "sample_rate": "8000", "channels": 1}], "format": {"duration": "1"}},
        {"streams": [{"codec_name": "amr_nb", "sample_rate": "16000", "channels": 1}], "format": {"duration": "1"}},
        {"streams": [{"codec_name": "amr_nb", "sample_rate": "8000", "channels": 2}], "format": {"duration": "1"}},
        *[{"streams": [{"codec_name": "amr_nb", "sample_rate": "8000", "channels": 1}], "format": {"duration": d}}
          for d in ("nan", "inf", "0", "-1", "invalid")],
    ])
    async def test_codec_and_duration_not_extension_determine_validity(self, tmp_path, monkeypatch, probe):
        from plugins.platforms.dingtalk import voice
        source = tmp_path / "voice.amr"
        source.write_bytes(b"#!AMR\nfixture")
        tool = AsyncMock(return_value=json.dumps(probe).encode())
        monkeypatch.setattr(voice, "run_audio_tool", tool)
        with pytest.raises(ValueError, match="AMR-NB, 8 kHz, mono"):
            await voice.prepare_voice(str(source))
        snapshot = tool.await_args.args[-1]
        assert snapshot != str(source) and not os.path.exists(snapshot)
        assert source.exists()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("data,error", [(b"RIFF-not-amr", "AMR-NB file"), (b"#!AMR\n" + b"x" * (2 * 1024 * 1024), "2 MB")], ids=["invalid-header", "oversized"])
    async def test_invalid_or_oversized_media_never_reaches_probe(self, tmp_path, monkeypatch, data, error):
        from plugins.platforms.dingtalk import voice
        source = tmp_path / "voice.amr"
        source.write_bytes(data)
        tool = AsyncMock()
        monkeypatch.setattr(voice, "run_audio_tool", tool)
        with pytest.raises(ValueError, match=error):
            await voice.prepare_voice(str(source))
        tool.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("tool", ["ffmpeg", "ffprobe"])
    async def test_missing_binary_errors_are_actionable(self, monkeypatch, tool):
        from plugins.platforms.dingtalk import voice
        monkeypatch.setattr(voice.asyncio, "create_subprocess_exec", AsyncMock(side_effect=FileNotFoundError("private-path")))
        with pytest.raises(ValueError, match=tool + " is required") as exc:
            await voice.run_audio_tool(tool, timeout=1)
        assert "private-path" not in str(exc.value)

    @pytest.mark.asyncio
    async def test_missing_encoder_error_is_actionable(self, monkeypatch):
        from plugins.platforms.dingtalk import voice
        process = MagicMock(returncode=1)
        process.communicate = AsyncMock(return_value=(b"", b"Unknown encoder 'libopencore_amrnb' private-path"))
        monkeypatch.setattr(voice.asyncio, "create_subprocess_exec", AsyncMock(return_value=process))
        with pytest.raises(ValueError, match="needs the libopencore_amrnb encoder") as exc:
            await voice.run_audio_tool("ffmpeg", timeout=1)
        assert "private-path" not in str(exc.value)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("cancel", [False, True])
    async def test_subprocess_timeout_and_cancellation_reap_child(self, tmp_path, monkeypatch, cancel):
        from pathlib import Path
        from plugins.platforms.dingtalk import voice
        source = tmp_path / "voice.wav"
        source.write_bytes(b"fixture")
        started = asyncio.Event()
        process = MagicMock(returncode=None)
        paths = []
        async def communicate():
            started.set()
            if process.returncode is None:
                await asyncio.Future()
            return b"", b""
        def kill():
            process.returncode = -9
        async def spawn(*args, **kwargs):
            paths.append(args[-1])
            return process
        process.kill.side_effect = kill
        process.communicate = AsyncMock(side_effect=communicate)
        monkeypatch.setattr(voice.asyncio, "create_subprocess_exec", spawn)
        if cancel:
            task = asyncio.create_task(voice.prepare_voice(str(source)))
            await started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert not Path(paths[0]).exists()
        else:
            with pytest.raises(ValueError, match="timed out"):
                await voice.run_audio_tool("ffmpeg", timeout=0.01)
        process.kill.assert_called_once()
        assert process.communicate.await_count == 2
        assert source.exists()


class TestConnect:


    @pytest.mark.asyncio
    async def test_connect_fails_without_sdk(self, monkeypatch):
        monkeypatch.setattr(
            "plugins.platforms.dingtalk.adapter.DINGTALK_STREAM_AVAILABLE", False
        )
        from plugins.platforms.dingtalk.adapter import DingTalkAdapter
        adapter = DingTalkAdapter(PlatformConfig(enabled=True))
        result = await adapter.connect()
        assert result is False


    @pytest.mark.asyncio
    async def test_disconnect_finalizes_open_streaming_cards(self):
        """Streaming cards must be finalized before HTTP client closes."""
        from unittest.mock import AsyncMock, patch
        from plugins.platforms.dingtalk.adapter import DingTalkAdapter
        adapter = DingTalkAdapter(PlatformConfig(enabled=True))
        adapter._http_client = AsyncMock()
        adapter._stream_task = None
        adapter._streaming_cards = {
            "chat-1": {"track-a": "last content"},
            "chat-2": {"track-b": "other"},
        }

        close_calls = []

        async def fake_close_siblings(chat_id):
            # HTTP client must still be alive at call time.
            assert adapter._http_client is not None, (
                "HTTP client was already closed before card finalization"
            )
            close_calls.append(chat_id)
            adapter._streaming_cards.pop(chat_id, None)

        with patch.object(adapter, "_close_streaming_siblings", side_effect=fake_close_siblings):
            await adapter.disconnect()

        assert set(close_calls) == {"chat-1", "chat-2"}
        assert adapter._streaming_cards == {}
        assert adapter._http_client is None


# ---------------------------------------------------------------------------
# Platform enum
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# SDK compatibility regression tests (dingtalk-stream >= 0.20 / 0.24)
# ---------------------------------------------------------------------------


class TestWebhookDomainAllowlist:
    """Guard the webhook origin allowlist against regression.

    The SDK started returning reply webhooks on ``oapi.dingtalk.com`` in
    addition to ``api.dingtalk.com``. Both must be accepted, and hostile
    lookalikes must still be rejected (SSRF defence-in-depth).
    """

    def test_api_domain_accepted(self):
        from plugins.platforms.dingtalk.adapter import _DINGTALK_WEBHOOK_RE
        assert _DINGTALK_WEBHOOK_RE.match(
            "https://api.dingtalk.com/robot/send?access_token=x"
        )

    def test_oapi_domain_accepted(self):
        from plugins.platforms.dingtalk.adapter import _DINGTALK_WEBHOOK_RE
        assert _DINGTALK_WEBHOOK_RE.match(
            "https://oapi.dingtalk.com/robot/send?access_token=x"
        )


class TestHandlerProcessIsAsync:
    """dingtalk-stream >= 0.20 requires ``process`` to be a coroutine."""

    def test_process_is_coroutine_function(self):
        from plugins.platforms.dingtalk.adapter import _IncomingHandler
        assert asyncio.iscoroutinefunction(_IncomingHandler.process)


class TestExtractText:
    """_extract_text must handle both legacy and current SDK payload shapes.

    Before SDK 0.20 ``message.text`` was a ``dict`` with a ``content`` key.
    From 0.20 onward it is a ``TextContent`` dataclass whose ``__str__``
    returns ``"TextContent(content=...)"`` — falling back to ``str(text)``
    leaks that repr into the agent's input.
    """


    def test_text_as_textcontent_object(self):
        """SDK >= 0.20 shape: object with ``.content`` attribute."""
        from plugins.platforms.dingtalk.adapter import DingTalkAdapter

        class FakeTextContent:
            content = "hello from new sdk"

            def __str__(self):  # mimic real SDK repr
                return f"TextContent(content={self.content})"

        msg = MagicMock()
        msg.text = FakeTextContent()
        msg.rich_text_content = None
        msg.rich_text = None
        result = DingTalkAdapter._extract_text(msg)
        assert result == "hello from new sdk"
        assert "TextContent(" not in result


    def test_rich_text_content_new_shape(self):
        """SDK >= 0.20 exposes rich text as ``message.rich_text_content.rich_text_list``."""
        from plugins.platforms.dingtalk.adapter import DingTalkAdapter

        class FakeRichText:
            rich_text_list = [{"text": "hello "}, {"text": "world"}]

        msg = MagicMock()
        msg.text = None
        msg.rich_text_content = FakeRichText()
        msg.rich_text = None
        result = DingTalkAdapter._extract_text(msg)
        assert "hello" in result and "world" in result


    def test_empty_message(self):
        from plugins.platforms.dingtalk.adapter import DingTalkAdapter
        msg = MagicMock()
        msg.text = None
        msg.rich_text_content = None
        msg.rich_text = None
        assert DingTalkAdapter._extract_text(msg) == ""

    # --- Card / interactiveCard message handling (文档分享卡片) ---

    def test_card_with_dict_content_and_url(self):
        """card msgtype with extensions.card.content as dict with url key."""
        from plugins.platforms.dingtalk.adapter import DingTalkAdapter
        msg = MagicMock()
        msg.text = None
        msg.rich_text = None
        msg.message_type = "card"
        msg.extensions = {
            "card": {
                "title": "Q3经营分析报告",
                "content": {"url": "https://dingtalk.com/doc/abc123"},
            }
        }
        assert DingTalkAdapter._extract_text(msg) == "[文档] Q3经营分析报告 https://dingtalk.com/doc/abc123"

    def test_card_with_dict_content_docurl(self):
        """card msgtype with extensions.card.content as dict with docUrl key."""
        from plugins.platforms.dingtalk.adapter import DingTalkAdapter
        msg = MagicMock()
        msg.text = None
        msg.rich_text = None
        msg.message_type = "card"
        msg.extensions = {
            "card": {
                "title": "周报模板",
                "content": {"docUrl": "https://docs.dingtalk.com/xyz"},
            }
        }
        assert DingTalkAdapter._extract_text(msg) == "[文档] 周报模板 https://docs.dingtalk.com/xyz"


    def test_interactive_card_with_title_and_url(self):
        """interactiveCard msgtype with both title and biz_custom_action_url."""
        from plugins.platforms.dingtalk.adapter import DingTalkAdapter
        msg = MagicMock()
        msg.text = None
        msg.rich_text = None
        msg.message_type = "interactiveCard"
        msg.extensions = {
            "content": {
                "title": "项目看板",
                "biz_custom_action_url": "https://dingtalk.com/doc/kanban",
            }
        }
        assert DingTalkAdapter._extract_text(msg) == "[文档卡片] 项目看板 https://dingtalk.com/doc/kanban"


class TestExtractMedia:
    """_extract_media must split native voice rich-text items (auto-STT)
    from generic audio file uploads (kept as attachments, no STT)."""

    def _msg_with_rich_text(self, items):
        msg = MagicMock()
        msg.text = None
        msg.image_content = None
        msg.rich_text_content = None
        msg.rich_text = items
        return msg


    def test_richtext_reset_does_not_clobber_voice(self):
        """A richText envelope containing a native voice item must stay
        VOICE — the ``msg_type_str == "richText"`` re-derivation used to
        reset it to TEXT, dropping the voice note from the STT path
        (#38211, #38219)."""
        from plugins.platforms.dingtalk.adapter import DingTalkAdapter
        from gateway.platforms.event import MessageType

        msg = self._msg_with_rich_text(
            [{"type": "voice", "downloadCode": "dl_voice_rt"}]
        )
        msg.message_type = "richText"
        msg_type, urls, mtypes = DingTalkAdapter._extract_media(
            DingTalkAdapter, msg
        )
        assert msg_type == MessageType.VOICE
        assert urls == ["dl_voice_rt"]
        assert mtypes == ["audio"]


    def test_image_no_filename_still_photo(self):
        """msgtype='image' without fileName → still PHOTO (MIME heuristic)."""
        from plugins.platforms.dingtalk.adapter import DingTalkAdapter
        from gateway.platforms.event import MessageType

        msg = MagicMock()
        msg.text = None
        msg.image_content = None
        msg.rich_text_content = None
        msg.rich_text = None
        msg.message_type = "image"
        msg.extensions = {"content": {"downloadCode": "dl_img_noext"}}
        msg_type, urls, mtypes = DingTalkAdapter._extract_media(
            DingTalkAdapter, msg
        )
        assert msg_type == MessageType.PHOTO
        assert urls == ["dl_img_noext"]
        # Without fileName, mime defaults to octet-stream but msg_type_str=="image" still wins
        assert mtypes == ["application/octet-stream"]


# ---------------------------------------------------------------------------
# Group gating — require_mention + allowed_users (parity with other platforms)
# ---------------------------------------------------------------------------


def _make_gating_adapter(monkeypatch, *, extra=None, env=None):
    """Build a DingTalkAdapter with only the gating fields populated.

    Clears every DINGTALK_* gating env var before applying the caller's
    overrides so individual tests stay isolated.
    """
    for key in (
        "DINGTALK_REQUIRE_MENTION",
        "DINGTALK_MENTION_PATTERNS",
        "DINGTALK_FREE_RESPONSE_CHATS",
        "DINGTALK_ALLOWED_USERS",
    ):
        monkeypatch.delenv(key, raising=False)
    for key, value in (env or {}).items():
        monkeypatch.setenv(key, value)
    from plugins.platforms.dingtalk.adapter import DingTalkAdapter
    return DingTalkAdapter(PlatformConfig(enabled=True, extra=extra or {}))


class TestAllowedUsersGate:

    def test_empty_allowlist_allows_everyone(self, monkeypatch):
        adapter = _make_gating_adapter(monkeypatch)
        assert adapter._is_user_allowed("anyone", "any-staff") is True


    def test_matches_sender_id_case_insensitive(self, monkeypatch):
        adapter = _make_gating_adapter(
            monkeypatch, extra={"allowed_users": ["SenderABC"]}
        )
        assert adapter._is_user_allowed("senderabc", "") is True


class TestMentionPatterns:


    def test_pattern_matches_text(self, monkeypatch):
        adapter = _make_gating_adapter(
            monkeypatch, extra={"mention_patterns": ["^hermes"]}
        )
        assert adapter._message_matches_mention_patterns("hermes please help") is True
        assert adapter._message_matches_mention_patterns("please hermes help") is False


    def test_env_var_json_populates_patterns(self, monkeypatch):
        adapter = _make_gating_adapter(
            monkeypatch,
            env={"DINGTALK_MENTION_PATTERNS": '["^bot", "^assistant"]'},
        )
        assert len(adapter._mention_patterns) == 2
        assert adapter._message_matches_mention_patterns("bot ping") is True


class TestShouldProcessMessage:

    def test_dm_always_accepted(self, monkeypatch):
        adapter = _make_gating_adapter(
            monkeypatch, extra={"require_mention": True}
        )
        msg = MagicMock(is_in_at_list=False)
        assert adapter._should_process_message(msg, "hi", is_group=False, chat_id="dm1") is True


    def test_group_accepted_when_chat_in_free_response_list(self, monkeypatch):
        adapter = _make_gating_adapter(
            monkeypatch,
            extra={"require_mention": True, "free_response_chats": ["grp1"]},
        )
        msg = MagicMock(is_in_at_list=False)
        assert adapter._should_process_message(msg, "hi", is_group=True, chat_id="grp1") is True
        # Different group still blocked
        assert adapter._should_process_message(msg, "hi", is_group=True, chat_id="grp2") is False


# ---------------------------------------------------------------------------
# _IncomingHandler.process — session_webhook extraction & fire-and-forget
# ---------------------------------------------------------------------------


class TestIncomingHandlerProcess:
    """Verify that _IncomingHandler.process correctly converts callback data
    and dispatches message processing as a background task (fire-and-forget)
    so the SDK ACK is returned immediately."""


    @pytest.mark.asyncio
    async def test_process_returns_ack_immediately(self):
        """process() must not block on _on_message — it should return
        the ACK tuple before the message is fully processed."""
        from plugins.platforms.dingtalk.adapter import _IncomingHandler, DingTalkAdapter

        processing_started = asyncio.Event()
        processing_gate = asyncio.Event()

        async def slow_on_message(msg):
            processing_started.set()
            await processing_gate.wait()  # Block until we release

        adapter = DingTalkAdapter(PlatformConfig(enabled=True))
        adapter._on_message = slow_on_message
        handler = _IncomingHandler(adapter, asyncio.get_running_loop())

        callback = MagicMock()
        callback.data = {
            "msgtype": "text",
            "text": {"content": "test"},
            "senderId": "u",
            "conversationId": "c",
            "sessionWebhook": "https://oapi.dingtalk.com/x",
            "msgId": "m",
        }

        # process() should return immediately even though _on_message blocks
        result = await handler.process(callback)
        assert result[0] == 200

        # Clean up: release the gate so the background task finishes
        processing_gate.set()
        await asyncio.sleep(0.05)


# ---------------------------------------------------------------------------
# Text extraction — mention preservation + platform sanity
# ---------------------------------------------------------------------------

class TestExtractTextMentions:

    def test_preserves_at_mentions_in_text(self):
        """@mentions are routing signals (via isInAtList), not text to strip.

        Stripping all @handles collateral-damages emails, SSH URLs, and
        literal references the user wrote.
        """
        from plugins.platforms.dingtalk.adapter import DingTalkAdapter
        cases = [
            ("@bot hello", "@bot hello"),
            ("contact alice@example.com", "contact alice@example.com"),
            ("git@github.com:foo/bar.git", "git@github.com:foo/bar.git"),
            ("what does @openai think", "what does @openai think"),
            ("@机器人 转发给 @老王", "@机器人 转发给 @老王"),
        ]
        for text, expected in cases:
            msg = MagicMock()
            msg.text = text
            msg.rich_text = None
            msg.rich_text_content = None
            assert DingTalkAdapter._extract_text(msg) == expected, (
                f"mangled: {text!r} -> {DingTalkAdapter._extract_text(msg)!r}"
            )


# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Concurrency — chat-scoped message context
# ---------------------------------------------------------------------------


class TestMessageContextIsolation:

    def test_contexts_keyed_by_chat_id(self):
        """Two concurrent chats must not clobber each other's context."""
        from plugins.platforms.dingtalk.adapter import DingTalkAdapter
        adapter = DingTalkAdapter(PlatformConfig(enabled=True))

        msg_a = MagicMock(conversation_id="chat-A", sender_staff_id="user-A")
        msg_b = MagicMock(conversation_id="chat-B", sender_staff_id="user-B")
        adapter._message_contexts["chat-A"] = msg_a
        adapter._message_contexts["chat-B"] = msg_b

        assert adapter._message_contexts["chat-A"] is msg_a
        assert adapter._message_contexts["chat-B"] is msg_b


# ---------------------------------------------------------------------------
# Card lifecycle: finalize via metadata["streaming"]
# ---------------------------------------------------------------------------


class TestCardLifecycle:

    @pytest.fixture
    def adapter_with_card(self):
        from plugins.platforms.dingtalk.adapter import DingTalkAdapter
        a = DingTalkAdapter(PlatformConfig(
            enabled=True,
            extra={"card_template_id": "tmpl-1"},
        ))
        a._card_sdk = MagicMock()
        a._card_sdk.create_card_with_options_async = AsyncMock()
        a._card_sdk.deliver_card_with_options_async = AsyncMock()
        a._card_sdk.streaming_update_with_options_async = AsyncMock()
        a._http_client = AsyncMock()
        a._get_access_token = AsyncMock(return_value="token")
        # Minimal message context
        msg = MagicMock(
            conversation_id="chat-1",
            conversation_type="1",
            sender_staff_id="staff-1",
            message_id="user-msg-1",
        )
        a._message_contexts["chat-1"] = msg
        a._session_webhooks["chat-1"] = (
            "https://api.dingtalk.com/x", 9999999999999,
        )
        return a

    @pytest.mark.asyncio
    async def test_final_reply_finalizes_card(self, adapter_with_card):
        """send(reply_to=...) creates a closed card (final response path)."""
        a = adapter_with_card
        result = await a.send("chat-1", "Hello", reply_to="user-msg-1")
        assert result.success
        call = a._card_sdk.streaming_update_with_options_async.call_args
        assert call[0][0].is_finalize is True
        # Not tracked as streaming — it's already closed.
        assert "chat-1" not in a._streaming_cards

    @pytest.mark.asyncio
    async def test_intermediate_send_stays_streaming(self, adapter_with_card):
        """send() without reply_to creates an OPEN card (tool progress /
        commentary / streaming first chunk).  No flicker closed→streaming
        when edit_message follows."""
        a = adapter_with_card
        result = await a.send("chat-1", "💻 terminal: ls")
        assert result.success
        call = a._card_sdk.streaming_update_with_options_async.call_args
        assert call[0][0].is_finalize is False
        # Tracked for sibling cleanup.
        assert result.message_id in a._streaming_cards.get("chat-1", {})


    @pytest.mark.asyncio
    async def test_edit_message_finalize_fires_done(self, adapter_with_card):
        """Stream consumer's final edit_message(finalize=True) fires Done."""
        a = adapter_with_card
        fired: list[str] = []
        a._fire_done_reaction = lambda cid: fired.append(cid)

        await a.send("chat-1", "initial")
        # Reopen via edit_message(finalize=False) then close.
        await a.edit_message(
            chat_id="chat-1", message_id="track-X",
            content="streaming...", finalize=False,
        )
        await a.edit_message(
            chat_id="chat-1", message_id="track-X",
            content="final", finalize=True,
        )
        assert "chat-1" in fired


# ---------------------------------------------------------------------------
# AI Card Tests
# ---------------------------------------------------------------------------

class TestDingTalkAdapterAICards:
    @pytest.fixture
    def config(self):
        return PlatformConfig(
            enabled=True,
            extra={
                "client_id": "test_id",
                "client_secret": "test_secret",
                "card_template_id": "test_card_template",
            },
        )

    @pytest.fixture
    def mock_stream_client(self):
        client = MagicMock()
        client.get_access_token = MagicMock(return_value="test_token")
        return client

    @pytest.fixture
    def mock_http_client(self):
        return AsyncMock()

    @pytest.fixture
    def mock_message(self):
        msg = MagicMock()
        msg.message_id = "test_msg_id"
        msg.conversation_id = "test_conv_id"
        msg.conversation_type = "1"
        msg.sender_id = "sender1"
        msg.sender_nick = "Test User"
        msg.sender_staff_id = "staff1"
        msg.text = MagicMock(content="Hello")
        msg.session_webhook = "https://api.dingtalk.com/robot/sendBySession?session=test"
        msg.session_webhook_expired_time = 999999999999
        msg.create_at = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
        msg.at_users = []
        return msg

    @pytest.mark.asyncio
    async def test_send_uses_ai_card_if_configured(self, config, mock_stream_client, mock_http_client, mock_message):
        from plugins.platforms.dingtalk.adapter import DingTalkAdapter

        adapter = DingTalkAdapter(config)
        adapter._stream_client = mock_stream_client
        adapter._http_client = mock_http_client
        adapter._message_contexts["test_conv_id"] = mock_message
        adapter._session_webhooks = {"test_conv_id": ("https://api.dingtalk.com/robot/sendBySession?session=test", 9999999999999)}
        adapter._card_template_id = "test_card_template"

        # Mock the card SDK with proper async methods
        mock_card_sdk = MagicMock()
        mock_card_sdk.create_card_with_options_async = AsyncMock()
        mock_card_sdk.deliver_card_with_options_async = AsyncMock()
        mock_card_sdk.streaming_update_with_options_async = AsyncMock()
        adapter._card_sdk = mock_card_sdk

        # Mock access token
        adapter._get_access_token = AsyncMock(return_value="test_token")

        result = await adapter.send("test_conv_id", "Hello World")

        mock_card_sdk.create_card_with_options_async.assert_called_once()
        mock_card_sdk.deliver_card_with_options_async.assert_called_once()
        mock_card_sdk.streaming_update_with_options_async.assert_called_once()
        assert result.success is True

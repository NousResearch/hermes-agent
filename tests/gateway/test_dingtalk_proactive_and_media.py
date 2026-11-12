"""DingTalk: proactive delivery (no session webhook) and local media sending.

Both paths are exercised with fake HTTP clients — no network, no DingTalk account.
"""
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult

CHAT = "cid-test/abc="
STAFF = "275036181138007396"


class _FakeResponse:
    status_code = 200
    content = b'{"invalidStaffIdList":[],"filteredStaffIdList":[]}'
    text = '{"invalidStaffIdList":[],"filteredStaffIdList":[]}'

    def json(self):
        return {"invalidStaffIdList": [], "filteredStaffIdList": []}


class _RecordingClient:
    def __init__(self):
        self.calls = []

    async def post(self, url, **kwargs):
        self.calls.append((url, kwargs.get("json"), kwargs.get("params")))
        return _FakeResponse()


def _make_adapter(extra=None, staff_ids_path=None):
    from plugins.platforms.dingtalk.adapter import DingTalkAdapter

    adapter = DingTalkAdapter(PlatformConfig(enabled=True, extra=extra or {}))
    adapter._http_client = _RecordingClient()
    adapter._get_access_token = AsyncMock(return_value="tok")
    if staff_ids_path is not None:
        adapter._staff_ids.clear()
        adapter._staff_ids_path = lambda: staff_ids_path
    return adapter


# --------------------------------------------------------------------------- send() fallback


class TestSendFallsBackToProactivePush:

    @pytest.mark.asyncio
    async def test_no_webhook_uses_proactive_push(self):
        adapter = _make_adapter()
        adapter._proactive_push = AsyncMock(return_value=SendResult(success=True, message_id="p1"))

        result = await adapter.send(CHAT, "hello")

        assert result.success is True
        assert result.message_id == "p1"
        adapter._proactive_push.assert_awaited_once()
        assert adapter._http_client.calls == []  # never reached the webhook branch

    @pytest.mark.asyncio
    async def test_no_webhook_and_no_push_still_reports_the_old_error(self):
        adapter = _make_adapter()
        adapter._proactive_push = AsyncMock(return_value=None)

        result = await adapter.send(CHAT, "hello")

        assert result.success is False
        assert "session_webhook" in (result.error or "")

    @pytest.mark.asyncio
    async def test_webhook_path_unchanged(self):
        adapter = _make_adapter()
        adapter._proactive_push = AsyncMock(return_value=SendResult(success=True))

        result = await adapter.send(
            CHAT, "hello", metadata={"session_webhook": "https://dingtalk.example/hook"}
        )

        assert result.success is True
        adapter._proactive_push.assert_not_awaited()
        assert adapter._http_client.calls[0][0] == "https://dingtalk.example/hook"


# --------------------------------------------------------------------------- staffId learning


class TestStaffIdResolution:

    def test_learned_staff_id_is_persisted_and_reloaded(self, tmp_path):
        path = tmp_path / "cache" / "dingtalk_staff_ids.json"
        adapter = _make_adapter(staff_ids_path=path)

        adapter._remember_staff_id(CHAT, STAFF)

        assert json.loads(path.read_text(encoding="utf-8")) == {CHAT: STAFF}
        # a fresh adapter (i.e. after a gateway restart) picks it up from disk
        restarted = _make_adapter(staff_ids_path=path)
        assert restarted._staff_id_for(CHAT) == STAFF

    def test_inbound_context_wins_over_config(self):
        adapter = _make_adapter(extra={"staff_id": "from-config"})
        adapter._message_contexts[CHAT] = SimpleNamespace(sender_staff_id=STAFF)

        assert adapter._staff_id_for(CHAT) == STAFF

    def test_config_extra_is_the_cold_start_fallback(self):
        adapter = _make_adapter(extra={"staff_id": "from-config"})

        assert adapter._staff_id_for(CHAT) == "from-config"


# --------------------------------------------------------------------------- proactive routing


class TestProactivePushRouting:

    @pytest.mark.asyncio
    async def test_one_to_one_uses_batch_send_with_staff_id(self):
        adapter = _make_adapter(extra={"staff_id": STAFF})

        result = await adapter._proactive_push(CHAT, "**done**")

        assert result.success is True
        url, body, _ = adapter._http_client.calls[0]
        assert url.endswith("/v1.0/robot/oToMessages/batchSend")
        assert body["userIds"] == [STAFF]
        assert body["msgKey"] == "sampleMarkdown"

    @pytest.mark.asyncio
    async def test_group_uses_group_send_with_conversation_id(self):
        adapter = _make_adapter(extra={"staff_id": STAFF})
        adapter._message_contexts[CHAT] = SimpleNamespace(conversation_type="2")

        result = await adapter._proactive_push(CHAT, "**done**")

        assert result.success is True
        url, body, _ = adapter._http_client.calls[0]
        assert url.endswith("/v1.0/robot/groupMessages/send")
        assert body["openConversationId"] == CHAT

    @pytest.mark.asyncio
    async def test_no_staff_id_means_no_attempt(self, monkeypatch):
        adapter = _make_adapter()
        monkeypatch.setattr(adapter, "_staff_id_for", lambda chat_id: "")

        assert await adapter._proactive_push(CHAT, "hi") is None
        assert adapter._http_client.calls == []


# --------------------------------------------------------------------------- media


class TestMediaSending:

    @pytest.mark.asyncio
    async def test_image_uses_media_id_as_photo_url(self, tmp_path):
        image = tmp_path / "a.png"
        image.write_bytes(b"png")
        adapter = _make_adapter(extra={"staff_id": STAFF})
        adapter._upload_media = AsyncMock(return_value=("@media_id", None))

        result = await adapter.send_image_file(CHAT, str(image))

        assert result.success is True
        url, body, _ = adapter._http_client.calls[0]
        assert url.endswith("/oToMessages/batchSend")
        assert body["msgKey"] == "sampleImageMsg"
        assert json.loads(body["msgParam"]) == {"photoURL": "@media_id"}

    @pytest.mark.asyncio
    async def test_document_rejects_extension_outside_the_whitelist(self, tmp_path):
        doc = tmp_path / "notes.md"
        doc.write_text("x", encoding="utf-8")
        adapter = _make_adapter(extra={"staff_id": STAFF})
        adapter._upload_media = AsyncMock()

        result = await adapter.send_document(CHAT, str(doc))

        assert result.success is False
        assert "xlsx/pdf/zip/rar/doc/docx" in (result.error or "")
        adapter._upload_media.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_document_in_whitelist_is_sent(self, tmp_path):
        doc = tmp_path / "report.pdf"
        doc.write_bytes(b"%PDF")
        adapter = _make_adapter(extra={"staff_id": STAFF})
        adapter._upload_media = AsyncMock(return_value=("@file_id", None))

        result = await adapter.send_document(CHAT, str(doc))

        assert result.success is True
        body = adapter._http_client.calls[0][1]
        assert body["msgKey"] == "sampleFile"
        assert json.loads(body["msgParam"]) == {
            "mediaId": "@file_id", "fileName": "report.pdf", "fileType": "pdf"}

    @pytest.mark.asyncio
    async def test_oversized_media_is_rejected_before_upload(self, tmp_path, monkeypatch):
        big = tmp_path / "big.mp4"
        big.write_bytes(b"x")
        adapter = _make_adapter(extra={"staff_id": STAFF})
        adapter._upload_media = AsyncMock()
        monkeypatch.setattr(adapter, "MEDIA_LIMIT_BYTES", 0)

        result = await adapter.send_video(CHAT, str(big))

        assert result.success is False
        assert "20MB" in (result.error or "")
        adapter._upload_media.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_video_carries_millisecond_duration_and_cover(self, tmp_path):
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"x")
        cover = tmp_path / "cover.jpg"
        cover.write_bytes(b"jpg")
        adapter = _make_adapter(extra={"staff_id": STAFF})
        adapter._upload_media = AsyncMock(side_effect=[("@video_id", None), ("@cover_id", None)])
        adapter._extract_cover = AsyncMock(return_value=str(cover))
        adapter._probe_video = AsyncMock(
            return_value={"duration": "24532", "width": "1280", "height": "704"})

        result = await adapter.send_video(CHAT, str(video))

        assert result.success is True
        body = adapter._http_client.calls[0][1]
        assert body["msgKey"] == "sampleVideo"
        param = json.loads(body["msgParam"])
        # duration must be MILLISECONDS: a seconds value makes the client drop the cover
        assert param["duration"] == "24532"
        assert param["picMediaId"] == "@cover_id"
        assert param["videoMediaId"] == "@video_id"

    @pytest.mark.asyncio
    async def test_video_without_cover_fails_instead_of_sending_a_blank_card(self, tmp_path):
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"x")
        adapter = _make_adapter(extra={"staff_id": STAFF})
        adapter._upload_media = AsyncMock(return_value=("@video_id", None))
        adapter._extract_cover = AsyncMock(return_value=None)

        result = await adapter.send_video(CHAT, str(video))

        assert result.success is False
        assert "封面" in (result.error or "")

"""Tests for QQ Bot non-image file attachment handling."""

import asyncio
import io
import json
import zipfile
from pathlib import Path
from unittest import mock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.qqbot import QQAdapter


def _make_adapter() -> QQAdapter:
    adapter = QQAdapter(PlatformConfig(enabled=True, extra={"app_id": "a", "client_secret": "b"}))
    adapter._http_client = mock.AsyncMock()
    return adapter


class _FakeStreamResponse:
    def __init__(self, payload: bytes, *, content_length: int | None = None, chunks=None):
        self.headers = {}
        if content_length is not None:
            self.headers["content-length"] = str(content_length)
        self._payload = payload
        self._chunks = chunks

    def raise_for_status(self):
        return None

    async def aiter_bytes(self):
        for chunk in self._chunks if self._chunks is not None else [self._payload]:
            yield chunk


class _FakeStreamContext:
    def __init__(self, response: _FakeStreamResponse):
        self.response = response

    async def __aenter__(self):
        return self.response

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _RaisingStreamContext:
    async def __aenter__(self):
        raise ValueError("blocked unsafe media URL")

    async def __aexit__(self, exc_type, exc, tb):
        return False


def _set_stream_response(
    adapter: QQAdapter,
    payload: bytes,
    *,
    content_length: int | None = None,
    chunks=None,
) -> None:
    response = _FakeStreamResponse(payload, content_length=content_length, chunks=chunks)
    adapter._http_client.stream = mock.Mock(return_value=_FakeStreamContext(response))


class TestQQAttachmentPolicy:
    def test_txt_html_pdf_zip_are_allowed(self):
        from gateway.platforms.qqbot.adapter import QQ_INBOUND_ATTACHMENT_POLICY

        for filename in ("note.txt", "page.html", "report.pdf", "src.zip"):
            assert QQ_INBOUND_ATTACHMENT_POLICY.classify(filename, "") is not None

    def test_executable_and_secret_key_files_are_rejected(self):
        from gateway.platforms.qqbot.adapter import QQ_INBOUND_ATTACHMENT_POLICY

        for filename in ("run.exe", "lib.so", ".env", "id_rsa.key", "cert.pem"):
            assert QQ_INBOUND_ATTACHMENT_POLICY.classify(filename, "") is None

    def test_archive_limits_are_larger_than_regular_files(self):
        from gateway.platforms.qqbot.adapter import QQ_INBOUND_ATTACHMENT_POLICY

        regular = QQ_INBOUND_ATTACHMENT_POLICY.classify("note.txt", "text/plain")
        archive = QQ_INBOUND_ATTACHMENT_POLICY.classify("project.zip", "application/zip")

        assert regular.max_bytes == 20 * 1024 * 1024
        assert archive.max_bytes == 200 * 1024 * 1024
        assert archive.extract_max_total_bytes == 1024 * 1024 * 1024
        assert archive.extract_max_files == 10000


class TestQQDownloadedFileDescriptions:
    @pytest.mark.asyncio
    async def test_text_file_is_cached_and_previewed(self, tmp_path, monkeypatch):
        adapter = _make_adapter()
        payload = b"hello\nworld\n"
        _set_stream_response(adapter, payload, content_length=len(payload))
        monkeypatch.setattr("gateway.platforms.qqbot.adapter.cache_document_from_bytes", lambda data, name: str(tmp_path / name))
        monkeypatch.setattr("tools.url_safety.is_safe_url", lambda url: True)
        monkeypatch.setattr(Path, "write_bytes", lambda self, data: len(data), raising=False)

        result = await adapter._process_attachments([
            {"content_type": "text/plain", "url": "https://qq.example/note.txt", "filename": "note.txt"}
        ])

        assert "note.txt" in result["attachment_info"]
        assert "本地路径" in result["attachment_info"]
        assert "hello" in result["attachment_info"]
        assert "world" in result["attachment_info"]

    @pytest.mark.asyncio
    async def test_html_file_gets_text_preview_without_raw_tags(self, tmp_path, monkeypatch):
        adapter = _make_adapter()
        payload = b"<html><body><h1>Title</h1><script>bad()</script><p>Hello HTML</p></body></html>"
        _set_stream_response(adapter, payload, content_length=len(payload))
        monkeypatch.setattr("gateway.platforms.qqbot.adapter.cache_document_from_bytes", lambda data, name: str(tmp_path / name))
        monkeypatch.setattr("tools.url_safety.is_safe_url", lambda url: True)
        monkeypatch.setattr(Path, "write_bytes", lambda self, data: len(data), raising=False)

        result = await adapter._process_attachments([
            {"content_type": "text/html", "url": "https://qq.example/page.html", "filename": "page.html"}
        ])

        info = result["attachment_info"]
        assert "page.html" in info
        assert "Title" in info
        assert "Hello HTML" in info
        assert "<h1>" not in info
        assert "bad()" not in info

    @pytest.mark.asyncio
    async def test_zip_file_is_cached_but_not_auto_extracted(self, tmp_path, monkeypatch):
        adapter = _make_adapter()
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("README.md", "hello")
        payload = buf.getvalue()
        _set_stream_response(adapter, payload, content_length=len(payload))
        monkeypatch.setattr("gateway.platforms.qqbot.adapter.cache_document_from_bytes", lambda data, name: str(tmp_path / name))
        monkeypatch.setattr("tools.url_safety.is_safe_url", lambda url: True)
        monkeypatch.setattr(Path, "write_bytes", lambda self, data: len(data), raising=False)

        result = await adapter._process_attachments([
            {"content_type": "application/zip", "url": "https://qq.example/project.zip", "filename": "project.zip"}
        ])

        info = result["attachment_info"]
        assert "project.zip" in info
        assert "未自动解压" in info
        assert "文件数: 1" in info

    @pytest.mark.asyncio
    async def test_oversized_archive_is_rejected_before_cache(self, monkeypatch):
        adapter = _make_adapter()
        monkeypatch.setattr("tools.url_safety.is_safe_url", lambda url: True)
        too_large = 200 * 1024 * 1024 + 1
        _set_stream_response(adapter, b"x", content_length=too_large)

        with mock.patch("gateway.platforms.qqbot.adapter.cache_document_from_bytes") as cache_doc:
            result = await adapter._process_attachments([
                {"content_type": "application/zip", "url": "https://qq.example/big.zip", "filename": "big.zip"}
            ])

        cache_doc.assert_not_called()
        assert "超过大小限制" in result["attachment_info"]

    @pytest.mark.asyncio
    async def test_blocked_extension_is_reported_not_downloaded(self):
        adapter = _make_adapter()

        result = await adapter._process_attachments([
            {"content_type": "application/octet-stream", "url": "https://qq.example/run.exe", "filename": "run.exe"}
        ])

        adapter._http_client.get.assert_not_called()
        adapter._http_client.stream.assert_not_called()
        assert "不支持或敏感类型" in result["attachment_info"]

    @pytest.mark.asyncio
    async def test_streaming_download_aborts_when_chunks_exceed_limit(self, monkeypatch):
        adapter = _make_adapter()
        monkeypatch.setattr("tools.url_safety.is_safe_url", lambda url: True)
        _set_stream_response(adapter, b"", chunks=[b"123", b"456"])

        data = await adapter._download_attachment_bytes("https://qq.example/no-length.bin", 5)

        assert data is None
        adapter._http_client.stream.assert_called_once()

    @pytest.mark.asyncio
    async def test_oversized_image_is_rejected_before_cache(self, monkeypatch):
        adapter = _make_adapter()
        monkeypatch.setattr("tools.url_safety.is_safe_url", lambda url: True)
        too_large = 25 * 1024 * 1024 + 1
        _set_stream_response(adapter, b"\x89PNG\r\n\x1a\nsmall", content_length=too_large)

        with mock.patch("gateway.platforms.qqbot.adapter.cache_image_from_bytes") as cache_img:
            result = await adapter._process_attachments([
                {"content_type": "image/png", "url": "https://qq.example/big.png", "filename": "big.png"}
            ], message_id="msg-img")

        cache_img.assert_not_called()
        assert result["image_urls"] == []
        assert "图片" in result["attachment_info"]
        assert "超过大小限制" in result["attachment_info"]

    @pytest.mark.asyncio
    async def test_oversized_voice_is_rejected_before_stt(self, tmp_path, monkeypatch):
        adapter = _make_adapter()
        monkeypatch.setattr("tools.url_safety.is_safe_url", lambda url: True)
        too_large = 100 * 1024 * 1024 + 1
        _set_stream_response(adapter, b"0123456789abcdef", content_length=too_large)
        wav = tmp_path / "voice.wav"
        wav.write_bytes(b"RIFFxxxx")
        adapter._convert_audio_to_wav_file = mock.AsyncMock(return_value=str(wav))
        adapter._call_stt = mock.AsyncMock(return_value="should-not-run")

        result = await adapter._process_attachments([
            {"content_type": "voice", "url": "https://qq.example/voice.silk", "filename": "voice.silk"}
        ], message_id="msg-voice")

        adapter._call_stt.assert_not_called()
        assert result["voice_transcripts"] == ["[Voice] [语音识别失败]"]
        assert "语音超过大小限制" in result["attachment_info"]

    @pytest.mark.asyncio
    async def test_voice_download_error_is_reported_without_raising(self, monkeypatch):
        adapter = _make_adapter()
        monkeypatch.setattr("tools.url_safety.is_safe_url", lambda url: True)
        adapter._http_client.stream = mock.Mock(return_value=_RaisingStreamContext())
        adapter._call_stt = mock.AsyncMock(return_value="should-not-run")

        result = await adapter._process_attachments([
            {"content_type": "voice", "url": "https://qq.example/voice.silk", "filename": "voice.silk"}
        ], message_id="msg-voice-error")

        adapter._call_stt.assert_not_called()
        assert result["voice_transcripts"] == ["[Voice] [语音识别失败]"]
        assert "语音下载或识别失败" in result["attachment_info"]

    @pytest.mark.asyncio
    async def test_regular_audio_file_is_structured_media_not_voice_stt(self, tmp_path, monkeypatch):
        adapter = _make_adapter()
        payload = b"ID3 audio payload"
        _set_stream_response(adapter, payload, content_length=len(payload))
        monkeypatch.setattr("tools.url_safety.is_safe_url", lambda url: True)
        monkeypatch.setattr("gateway.platforms.qqbot.adapter.get_document_cache_dir", lambda: tmp_path)
        adapter._stt_voice_attachment = mock.AsyncMock(return_value="wrong")

        result = await adapter._process_attachments([
            {"content_type": "audio/mpeg", "url": "https://qq.example/song.mp3", "filename": "song.mp3"}
        ], message_id="msg-audio")

        adapter._stt_voice_attachment.assert_not_called()
        assert result["voice_transcripts"] == []
        assert "song.mp3" in result["attachment_info"]
        assert result["attachments"][0].kind == "media"
        assert result["attachments"][0].filename == "song.mp3"

    @pytest.mark.asyncio
    async def test_inbound_cache_uses_message_id_directory_and_meta_json(self, tmp_path, monkeypatch):
        adapter = _make_adapter()
        payload = b"hello\n"
        _set_stream_response(adapter, payload, content_length=len(payload))
        monkeypatch.setattr("tools.url_safety.is_safe_url", lambda url: True)
        monkeypatch.setattr("gateway.platforms.qqbot.adapter.get_document_cache_dir", lambda: tmp_path)

        result = await adapter._process_attachments([
            {"content_type": "text/plain", "url": "https://qq.example/note.txt", "filename": "note.txt"}
        ], message_id="msg-123")

        attachment = result["attachments"][0]
        local_path = Path(attachment.local_path)
        assert local_path.parent == tmp_path / "qqbot" / "msg-123"
        assert local_path.read_bytes() == payload
        meta = json.loads((local_path.parent / "meta.json").read_text(encoding="utf-8"))
        assert meta["platform"] == "qqbot"
        assert meta["message_id"] == "msg-123"
        assert meta["attachments"][0]["filename"] == "note.txt"
        assert meta["attachments"][0]["local_path"] == str(local_path)

    def test_zip_extracts_to_isolated_directory_and_blocks_traversal(self, tmp_path, monkeypatch):
        monkeypatch.setattr("gateway.platforms.qqbot.adapter.get_document_cache_dir", lambda: tmp_path)
        archive = tmp_path / "qqbot" / "msg-zip" / "archive.zip"
        archive.parent.mkdir(parents=True)
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("dir/file.txt", "ok")

        extracted = QQAdapter.extract_archive_to_isolated_dir(str(archive), message_id="msg-zip")

        assert extracted.parent == archive.parent
        assert (extracted / "dir" / "file.txt").read_text(encoding="utf-8") == "ok"

        unsafe = archive.parent / "unsafe.zip"
        with zipfile.ZipFile(unsafe, "w") as zf:
            zf.writestr("../escape.txt", "no")

        with pytest.raises(ValueError, match="Unsafe archive path"):
            QQAdapter.extract_archive_to_isolated_dir(str(unsafe), message_id="msg-zip")

    @pytest.mark.asyncio
    async def test_c2c_message_event_includes_structured_non_image_attachment(self, tmp_path, monkeypatch):
        adapter = _make_adapter()
        payload = b"hello from file\n"
        _set_stream_response(adapter, payload, content_length=len(payload))
        adapter.handle_message = mock.AsyncMock()
        monkeypatch.setattr("tools.url_safety.is_safe_url", lambda url: True)
        monkeypatch.setattr("gateway.platforms.qqbot.adapter.get_document_cache_dir", lambda: tmp_path)

        await adapter._handle_c2c_message(
            {
                "attachments": [
                    {
                        "content_type": "text/plain",
                        "url": "https://qq.example/note.txt",
                        "filename": "note.txt",
                    }
                ]
            },
            "msg-event",
            "see attached",
            {"user_openid": "user-1"},
            "",
        )

        adapter.handle_message.assert_awaited_once()
        event = adapter.handle_message.await_args.args[0]
        assert "本地路径" in event.text
        assert len(event.attachments) == 1
        assert event.attachments[0].filename == "note.txt"
        assert event.attachments[0].message_id == "msg-event"
        assert Path(event.attachments[0].local_path).parent == tmp_path / "qqbot" / "msg-event"

    @pytest.mark.asyncio
    async def test_send_document_blocks_dangerous_extensions_locally(self):
        adapter = _make_adapter()
        adapter._send_media = mock.AsyncMock()

        result = await adapter.send_document("chat", "/tmp/run.exe")

        adapter._send_media.assert_not_called()
        assert result.success is False
        assert result.retryable is False
        assert "危险扩展" in result.error

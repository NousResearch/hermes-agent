"""Tests for media-type arbitration (classify_media) and the download paths using it.

A platform-reported content_type can be wrong — a .dxf CAD file labelled image/*
once walked the image path, was rejected by the magic-byte check and silently
dropped. classify_media arbitrates extension → magic bytes → reported type, and the
platform download paths consult it before treating bytes as an image.
"""
import asyncio
import io
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
DXF = b"0\nSECTION\n2\nHEADER\n0\nENDSEC\n0\nEOF\n"


def _classify(content_type, filename="", raw=b""):
    from gateway.platforms.base import classify_media  # deferred: absent on the pre-fix tree

    return classify_media(content_type, filename, raw)


class TestClassifyMedia:
    def test_extension_wins(self):
        assert _classify("image/jpeg", "plan.dxf") == "document"
        assert _classify("application/octet-stream", "photo.png") == "image"
        assert _classify("", "report.pdf") == "document"
        assert _classify("", "song.mp3") == "audio"
        assert _classify("", "clip.mp4") == "video"

    def test_magic_bytes_beat_a_wrong_label(self):
        # The .dxf case: an image/* label with non-image bytes is a document, never an image.
        assert _classify("image/x-dxf", "", DXF) == "document"
        assert _classify("image/png", "", DXF) == "document"

    def test_magic_bytes_confirm_an_unlabeled_image(self):
        assert _classify("application/octet-stream", "", PNG) == "image"

    def test_reported_type_is_the_last_resort(self):
        assert _classify("image/jpeg", "") == "image"
        assert _classify("audio/ogg", "voice.bin") == "audio"
        assert _classify("", "") == "document"


class TestQqbotUsesClassify:
    def _make_adapter(self):
        from gateway.platforms.qqbot.adapter import QQAdapter

        return object.__new__(QQAdapter)

    def test_mislabeled_dxf_is_not_collected_as_image(self):
        adapter = self._make_adapter()
        adapter._download_and_cache = AsyncMock(return_value="/tmp/plan.dxf")
        attachments = [{"content_type": "image/x-dxf", "url": "https://cdn.example/plan.dxf", "filename": "plan.dxf"}]

        result = asyncio.run(adapter._process_attachments(attachments))

        assert result["image_urls"] == []
        assert "plan.dxf" in result["attachment_info"]

    def test_download_and_cache_stores_mislabeled_bytes_as_document(self):
        import gateway.platforms.qqbot.adapter as qa

        adapter = self._make_adapter()
        resp = SimpleNamespace(content=DXF, raise_for_status=lambda: None)
        adapter._http_client = SimpleNamespace(get=AsyncMock(return_value=resp))
        adapter._qq_media_headers = lambda: {}
        doc_cache = AsyncMock(return_value="/tmp/doc")
        img_cache = AsyncMock(return_value="/tmp/img")
        with patch("tools.url_safety.is_safe_url", return_value=True), \
             patch.object(qa, "cache_document_from_bytes_async", new=doc_cache), \
             patch.object(qa, "cache_image_from_bytes_async", new=img_cache):
            path = asyncio.run(adapter._download_and_cache(
                "https://cdn.example/download", "image/x-dxf", "plan.dxf"))

        assert path == "/tmp/doc"
        doc_cache.assert_awaited_once()
        assert img_cache.await_count == 0


class TestFeishuUsesClassify:
    def _make_adapter(self):
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = object.__new__(FeishuAdapter)
        adapter._client = SimpleNamespace()
        return adapter

    @staticmethod
    def _response(raw: bytes, content_type: str, filename: str = ""):
        return SimpleNamespace(
            success=lambda: True, file=io.BytesIO(raw), file_name=filename,
            raw=SimpleNamespace(headers={"Content-Type": content_type}),
        )

    def test_image_resource_with_non_image_bytes_becomes_document(self):
        import plugins.platforms.feishu.adapter as fa

        adapter = self._make_adapter()
        adapter._fetch_message_resource = AsyncMock(
            return_value=self._response(DXF, "image/x-dxf", filename="plan.dxf"))
        doc_cache = AsyncMock(return_value="/tmp/plan-doc.dxf")
        with patch.object(fa, "cache_document_from_bytes_async", new=doc_cache):
            path, media_type = asyncio.run(adapter._download_feishu_image(message_id="om_1", image_key="img_1"))

        assert path == "/tmp/plan-doc.dxf"
        doc_cache.assert_awaited_once()
        assert not media_type.startswith("image/")

    def test_message_resource_with_non_image_bytes_becomes_document(self):
        import plugins.platforms.feishu.adapter as fa

        adapter = self._make_adapter()
        adapter._fetch_message_resource = AsyncMock(
            return_value=self._response(DXF, "image/x-dxf"))
        doc_cache = AsyncMock(return_value="/tmp/plan-doc.dxf")
        img_cache = AsyncMock(return_value="/tmp/plan-img.jpg")
        with patch.object(fa, "cache_document_from_bytes_async", new=doc_cache), \
             patch.object(fa, "cache_image_from_bytes_async", new=img_cache):
            path, _media_type = asyncio.run(adapter._download_feishu_message_resource(
                message_id="om_1", file_key="fk_1", resource_type="file", fallback_filename="plan.dxf"))

        assert path == "/tmp/plan-doc.dxf"
        doc_cache.assert_awaited_once()
        assert img_cache.await_count == 0


class TestWeixinUsesClassifyFallback:
    def _make_adapter(self):
        from gateway.platforms.weixin import WeixinAdapter

        adapter = object.__new__(WeixinAdapter)
        adapter.platform = SimpleNamespace(value="weixin")  # backs the `name` property
        adapter._poll_session = object()
        adapter._cdn_base_url = "https://cdn.example"
        return adapter

    def test_image_cache_rejection_falls_back_to_document(self):
        import gateway.platforms.weixin as wx

        adapter = self._make_adapter()
        item = {"type": wx.ITEM_IMAGE, "image_item": {"file_name": "plan.dxf", "media": {}}}
        doc_cache = AsyncMock(return_value="/tmp/plan-doc.dxf")
        with patch.object(wx, "_download_and_decrypt_media", new=AsyncMock(return_value=DXF)), \
             patch.object(wx, "cache_image_from_bytes_async", new=AsyncMock(side_effect=ValueError("not an image"))), \
             patch.object(wx, "cache_document_from_bytes_async", new=doc_cache):
            path, _mime = asyncio.run(adapter._download_media(item, wx._INBOUND_MEDIA[wx.ITEM_IMAGE]))

        assert path == "/tmp/plan-doc.dxf"
        doc_cache.assert_awaited_once()

    def test_valid_image_still_caches_as_image(self):
        import gateway.platforms.weixin as wx

        adapter = self._make_adapter()
        item = {"type": wx.ITEM_IMAGE, "image_item": {"file_name": "photo.jpg", "media": {}}}
        with patch.object(wx, "_download_and_decrypt_media", new=AsyncMock(return_value=PNG)), \
             patch.object(wx, "cache_image_from_bytes_async", new=AsyncMock(return_value="/tmp/photo.jpg")) as img_cache:
            path, _mime = asyncio.run(adapter._download_media(item, wx._INBOUND_MEDIA[wx.ITEM_IMAGE]))

        assert path == "/tmp/photo.jpg"
        img_cache.assert_awaited_once()

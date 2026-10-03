"""Cross-platform tests: attachment download failures must reach the agent.

A failed media download used to vanish silently — the agent never learned a
file was sent. Each platform now folds an ``[Attachment download failed: ...]``
note into the text the agent receives (Feishu: message text; QQ: attachment
info block; Weixin: message text).
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock


# ---------------------------------------------------------------------------
# Feishu
# ---------------------------------------------------------------------------

class TestFeishuDownloadFailures:
    def _make_adapter(self):
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = object.__new__(FeishuAdapter)
        adapter._client = SimpleNamespace(request=object())
        return adapter

    def test_resource_downloads_collect_failure_labels(self):
        adapter = self._make_adapter()
        adapter._download_feishu_image = AsyncMock(return_value=("", ""))
        adapter._download_feishu_message_resource = AsyncMock(return_value=("", ""))
        normalized = SimpleNamespace(
            image_keys=["img_key_1"],
            media_refs=[SimpleNamespace(file_key="fk_1", resource_type="file", file_name="report.pdf")],
        )

        failures = []
        urls, types = asyncio.run(adapter._download_feishu_message_resources(
            message_id="om_1", normalized=normalized, collect_failures=failures,
        ))

        assert urls == [] and types == []
        assert failures == ["image", "report.pdf"]

    def test_extract_message_content_notes_failed_downloads(self):
        adapter = self._make_adapter()
        adapter._normalize = lambda *a, **k: SimpleNamespace(
            text_content="hello", image_keys=["img_key_1"], media_refs=[],
            preferred_message_type="photo", mentions=[],
        )
        adapter._download_feishu_image = AsyncMock(return_value=("", ""))
        message = SimpleNamespace(content="", message_type="photo", message_id="om_1", mentions=None)

        text, _inbound_type, media_urls, _media_types, _inlined, _mentions = asyncio.run(
            adapter._extract_message_content(message))

        assert media_urls == []
        assert text == "hello\n\n[Attachment download failed: image]"


# ---------------------------------------------------------------------------
# QQ Bot
# ---------------------------------------------------------------------------

class TestQqbotDownloadFailures:
    def _make_adapter(self):
        from gateway.platforms.qqbot.adapter import QQAdapter

        return object.__new__(QQAdapter)  # _log_tag resolves via the class property

    def test_failed_download_is_reported_in_attachment_info(self):
        adapter = self._make_adapter()
        adapter._download_and_cache = AsyncMock(side_effect=RuntimeError("boom"))
        attachments = [{"content_type": "file", "url": "https://cdn.example/doc.zip", "filename": "doc.zip"}]

        result = asyncio.run(adapter._process_attachments(attachments))

        assert result["attachment_info"] == "[Attachment download failed: doc.zip]"

    def test_missing_cached_path_is_reported(self):
        adapter = self._make_adapter()
        adapter._download_and_cache = AsyncMock(return_value=None)
        attachments = [{"content_type": "image/jpeg", "url": "https://cdn.example/p.jpg", "filename": "p.jpg"}]

        result = asyncio.run(adapter._process_attachments(attachments))

        assert result["attachment_info"] == "[Attachment download failed: p.jpg]"

    def test_successful_image_download_has_no_failure_note(self, tmp_path):
        adapter = self._make_adapter()
        img = tmp_path / "pic.jpg"
        img.write_bytes(b"x")
        adapter._download_and_cache = AsyncMock(return_value=str(img))
        attachments = [{"content_type": "image/jpeg", "url": "https://cdn.example/pic.jpg", "filename": "pic.jpg"}]

        result = asyncio.run(adapter._process_attachments(attachments))

        assert result["image_urls"] == [str(img)]
        assert result["attachment_info"] == ""


# ---------------------------------------------------------------------------
# Weixin
# ---------------------------------------------------------------------------

class TestWeixinDownloadFailures:
    def _make_adapter(self):
        from gateway.platforms.weixin import WeixinAdapter

        return object.__new__(WeixinAdapter)

    def test_collect_media_returns_failure_label(self):
        from gateway.platforms import weixin

        adapter = self._make_adapter()
        adapter._download_media = AsyncMock(return_value=(None, ""))

        failures = asyncio.run(adapter._collect_media({"type": weixin.ITEM_IMAGE}, [], []))

        assert failures == ["image"]

    def test_collect_media_success_returns_empty(self):
        from gateway.platforms import weixin

        adapter = self._make_adapter()
        adapter._download_media = AsyncMock(return_value=("/tmp/pic.jpg", "image/jpeg"))
        media_paths, media_types = [], []

        failures = asyncio.run(adapter._collect_media({"type": weixin.ITEM_IMAGE}, media_paths, media_types))

        assert failures == []
        assert media_paths == ["/tmp/pic.jpg"]
        assert media_types == ["image/jpeg"]

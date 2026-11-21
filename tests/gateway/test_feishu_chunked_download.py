"""Tests for the Feishu large-resource chunked download fallback.

Oversized uploads fail the single-shot resource download with 234037
"Downloaded file size exceeds limit"; the adapter retries the resource via
ranged reads and stitches the file together from 4MB chunks. Only that error
code is worth a ranged retry, and the ranged response's Content-Type must ride
along so the media is labelled correctly when the filename is missing.
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from plugins.platforms.feishu import adapter as feishu_adapter

CHUNK = 4 * 1024 * 1024


def _make_adapter():
    from plugins.platforms.feishu.adapter import FeishuAdapter

    adapter = object.__new__(FeishuAdapter)
    adapter._client = SimpleNamespace(request=object())  # non-None is all the guard checks
    return adapter


class TestChunkedResourceFallback:
    def test_stitches_all_ranges(self):
        adapter = _make_adapter()
        calls = []
        total = 10 * 1024 * 1024  # 10MB → 4MB + 4MB + 2MB

        async def fake_range(*, uri, resource_type, start, end):
            calls.append((start, end))
            size = min(end, total - 1) - start + 1
            return b"x" * size, total, "application/octet-stream"

        adapter._feishu_resource_range = fake_range
        out, content_type = asyncio.run(adapter._download_feishu_resource_chunked(
            message_id="om_1", file_key="fk_1", resource_type="file",
        ))

        assert len(out) == total
        assert content_type == "application/octet-stream"
        assert calls == [
            (0, CHUNK - 1),
            (CHUNK, 2 * CHUNK - 1),
            (2 * CHUNK, 3 * CHUNK - 1),
        ]

    def test_gives_up_when_a_piece_fails(self):
        adapter = _make_adapter()
        total = 10 * 1024 * 1024
        seen = {"n": 0}

        async def fake_range(*, uri, resource_type, start, end):
            seen["n"] += 1
            if seen["n"] == 1:
                return b"x" * CHUNK, total, "image/png"
            return b"", 0, ""

        adapter._feishu_resource_range = fake_range
        out, content_type = asyncio.run(adapter._download_feishu_resource_chunked(
            message_id="om_1", file_key="fk_1", resource_type="file",
        ))

        assert (out, content_type) == (b"", "")

    def test_whole_body_when_server_ignores_range(self):
        adapter = _make_adapter()
        body = b"whole-file-bytes"

        async def fake_range(*, uri, resource_type, start, end):
            return body, 0, "video/mp4"  # no Content-Range: the server answered 200 with the full body

        adapter._feishu_resource_range = fake_range
        out, content_type = asyncio.run(adapter._download_feishu_resource_chunked(
            message_id="om_1", file_key="fk_1", resource_type="file",
        ))

        assert out == body
        assert content_type == "video/mp4"

    def test_small_file_within_one_chunk(self):
        adapter = _make_adapter()

        async def fake_range(*, uri, resource_type, start, end):
            return b"tiny", 4, "text/plain"  # len(body) == total → done after the first read

        adapter._feishu_resource_range = fake_range
        out, content_type = asyncio.run(adapter._download_feishu_resource_chunked(
            message_id="om_1", file_key="fk_1", resource_type="file",
        ))

        assert out == b"tiny"
        assert content_type == "text/plain"

    def test_range_request_shape(self, monkeypatch):
        adapter = _make_adapter()
        sent = {}

        class FakeBuilder:
            def http_method(self, value):
                sent["method"] = value
                return self

            def uri(self, value):
                sent["uri"] = value
                return self

            def queries(self, value):
                sent["queries"] = value
                return self

            def headers(self, value):
                sent["headers"] = value
                return self

            def token_types(self, value):
                sent["token_types"] = value
                return self

            def build(self):
                return sent

        class FakeBaseRequest:
            @staticmethod
            def builder():
                return FakeBuilder()

        monkeypatch.setattr(feishu_adapter, "BaseRequest", FakeBaseRequest)
        monkeypatch.setattr(feishu_adapter, "HttpMethod", SimpleNamespace(GET="GET"))
        monkeypatch.setattr(feishu_adapter, "AccessTokenType", SimpleNamespace(TENANT="TENANT"))

        response = SimpleNamespace(
            raw=SimpleNamespace(
                content=b"x" * 100,
                headers={"Content-Range": "bytes 0-99/1000", "Content-Type": "application/pdf; charset=utf-8"},
            ),
        )
        adapter._run_blocking = AsyncMock(return_value=response)

        body, total, content_type = asyncio.run(adapter._feishu_resource_range(
            uri="/open-apis/im/v1/messages/om_1/resources/fk_1",
            resource_type="file", start=0, end=99,
        ))

        assert (body, total) == (b"x" * 100, 1000)
        assert content_type == "application/pdf"  # normalised: lowercased, parameters dropped
        assert sent["uri"] == "/open-apis/im/v1/messages/om_1/resources/fk_1"
        assert sent["queries"] == [("type", "file")]
        assert sent["headers"] == {"Range": "bytes=0-99"}
        assert sent["method"] == "GET"

    def test_download_uses_chunked_fallback_when_single_shot_fails(self):
        adapter = _make_adapter()
        failed = SimpleNamespace(success=lambda: False, code=234037, msg="Downloaded file size exceeds limit")
        adapter._fetch_message_resource = AsyncMock(return_value=failed)
        file_bytes = b"%PDF-1.4 " + b"x" * 1000
        adapter._download_feishu_resource_chunked = AsyncMock(return_value=(file_bytes, "application/pdf"))

        cached = {}

        async def fake_cache(body, filename):
            cached["body"] = body
            cached["filename"] = filename
            return "/tmp/cached_report.pdf"

        with patch.object(feishu_adapter, "cache_document_from_bytes_async", new=fake_cache):
            path, media_type = asyncio.run(adapter._download_feishu_message_resource(
                message_id="om_1", file_key="fk_1", resource_type="file", fallback_filename="report.pdf",
            ))

        assert path == "/tmp/cached_report.pdf"
        assert media_type == "application/pdf"
        assert cached["body"] == file_bytes
        assert cached["filename"] == "report.pdf"
        adapter._download_feishu_resource_chunked.assert_awaited_once_with(
            message_id="om_1", file_key="fk_1", resource_type="file",
        )

    def test_ranged_content_type_labels_the_media(self):
        # ref.file_name is often absent; without the ranged Content-Type an oversized
        # video would surface as application/octet-stream (and an image as a document).
        adapter = _make_adapter()
        failed = SimpleNamespace(success=lambda: False, code=234037, msg="Downloaded file size exceeds limit")
        adapter._fetch_message_resource = AsyncMock(return_value=failed)
        file_bytes = b"\x00\x00\x00\x18ftypmp42" + b"x" * 100
        adapter._download_feishu_resource_chunked = AsyncMock(return_value=(file_bytes, "video/mp4"))
        cache_mock = AsyncMock(return_value="/tmp/cached.mp4")

        with patch.object(feishu_adapter, "cache_document_from_bytes_async", new=cache_mock):
            path, media_type = asyncio.run(adapter._download_feishu_message_resource(
                message_id="om_1", file_key="fk_1", resource_type="media", fallback_filename="",
            ))

        assert path == "/tmp/cached.mp4"
        assert media_type == "video/mp4"
        assert cache_mock.await_args[0][1] == "media_fk_1.mp4"

    def test_non_oversize_failure_skips_the_ranged_retry(self):
        # An expired token or a revoked permission cannot be fixed by Range: the code
        # must not pay an extra round trip for failures that are not 234037.
        adapter = _make_adapter()
        failed = SimpleNamespace(success=lambda: False, code=99991400, msg="token invalid")
        adapter._fetch_message_resource = AsyncMock(return_value=failed)
        adapter._download_feishu_resource_chunked = AsyncMock(return_value=(b"x", "image/png"))

        path, media_type = asyncio.run(adapter._download_feishu_message_resource(
            message_id="om_1", file_key="fk_1", resource_type="file", fallback_filename="a.pdf",
        ))

        assert (path, media_type) == ("", "")
        adapter._download_feishu_resource_chunked.assert_not_awaited()

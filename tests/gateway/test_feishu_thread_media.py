"""Feishu topic media and text sends use the newest message as a reply anchor."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.feishu.adapter import FeishuAdapter


def _response(*, success: bool, message_id: str = "om_sent", file_key: str | None = None):
    data = SimpleNamespace(message_id=message_id)
    if file_key is not None:
        data.file_key = file_key
    return SimpleNamespace(success=lambda: success, data=data, code=99992402 if not success else 0)


class _MessageAPI:
    def __init__(self, *, reply_response=None, create_response=None):
        self.reply_calls = []
        self.create_calls = []
        self.reply_response = reply_response or _response(success=True)
        self.create_response = create_response or _response(success=True)

    def reply(self, request):
        self.reply_calls.append(request)
        if isinstance(self.reply_response, BaseException):
            raise self.reply_response
        return self.reply_response

    def create(self, request):
        self.create_calls.append(request)
        return self.create_response


class _FileAPI:
    def create(self, _request):
        return _response(success=True, file_key="file_key")


class _ImageAPI:
    def create(self, _request):
        return SimpleNamespace(success=lambda: True, data=SimpleNamespace(image_key="image_key"))


def _adapter(message_api: _MessageAPI) -> FeishuAdapter:
    adapter = FeishuAdapter(PlatformConfig())
    adapter._client = SimpleNamespace(
        im=SimpleNamespace(v1=SimpleNamespace(message=message_api, file=_FileAPI(), image=_ImageAPI()))
    )

    async def run_blocking(func, *args, **kwargs):
        return func(*args, **kwargs)

    adapter._run_blocking = run_blocking
    return adapter


def _request_body(request):
    return getattr(request, "request_body", None) or getattr(request, "body", None)


@pytest.mark.parametrize("kind", ["text", "file", "image", "captioned_file"])
def test_thread_sends_reply_to_fetched_anchor_without_thread_receive_id(kind, tmp_path):
    async def run():
        message_api = _MessageAPI()
        adapter = _adapter(message_api)
        adapter._fetch_last_message_in_thread = AsyncMock(return_value="om_anchor")

        if kind == "text":
            result = await adapter.send("oc_chat", "hello", metadata={"thread_id": "omt_thread"})
        elif kind == "file":
            path = tmp_path / "document.pdf"
            path.write_bytes(b"pdf")
            result = await adapter.send_document("oc_chat", str(path), metadata={"thread_id": "omt_thread"})
        elif kind == "captioned_file":
            path = tmp_path / "document.pdf"
            path.write_bytes(b"pdf")
            result = await adapter.send_document(
                "oc_chat", str(path), caption="caption", metadata={"thread_id": "omt_thread"}
            )
        else:
            path = tmp_path / "image.png"
            path.write_bytes(b"png")
            result = await adapter.send_image_file("oc_chat", str(path), metadata={"thread_id": "omt_thread"})

        return result, message_api

    result, message_api = asyncio.run(run())
    assert result.success
    assert len(message_api.reply_calls) == 1
    assert message_api.create_calls == []
    body = _request_body(message_api.reply_calls[0])
    assert body.reply_in_thread is True
    if kind == "captioned_file":
        assert body.msg_type == "post"
    assert message_api.reply_calls[0].message_id == "om_anchor"


def test_missing_thread_anchor_falls_back_to_chat_create():
    async def run():
        message_api = _MessageAPI()
        adapter = _adapter(message_api)
        adapter._fetch_last_message_in_thread = AsyncMock(return_value=None)
        result = await adapter.send("oc_chat", "hello", metadata={"thread_id": "omt_thread"})
        return result, message_api

    result, message_api = asyncio.run(run())
    assert result.success
    assert message_api.reply_calls == []
    assert len(message_api.create_calls) == 1
    assert message_api.create_calls[0].receive_id_type == "chat_id"
    assert _request_body(message_api.create_calls[0]).receive_id == "oc_chat"


@pytest.mark.parametrize("reply_response", [_response(success=False), RuntimeError("reply unavailable")])
def test_failed_thread_reply_falls_back_to_chat_create(reply_response):
    async def run():
        message_api = _MessageAPI(reply_response=reply_response)
        adapter = _adapter(message_api)
        adapter._fetch_last_message_in_thread = AsyncMock(return_value="om_anchor")
        result = await adapter.send("oc_chat", "hello", metadata={"thread_id": "omt_thread"})
        return result, message_api

    result, message_api = asyncio.run(run())
    assert result.success
    assert len(message_api.reply_calls) == 1
    assert len(message_api.create_calls) == 1
    assert message_api.create_calls[0].receive_id_type == "chat_id"
    assert _request_body(message_api.create_calls[0]).receive_id == "oc_chat"


def test_fetch_last_message_sorts_newest_first():
    async def run():
        requests = []

        class _ListingAPI:
            def list(self, request):
                requests.append(request)
                return SimpleNamespace(
                    success=lambda: True,
                    data=SimpleNamespace(items=[SimpleNamespace(message_id="om_newest")]),
                )

        adapter = FeishuAdapter(PlatformConfig())
        adapter._client = SimpleNamespace(im=SimpleNamespace(v1=SimpleNamespace(message=_ListingAPI())))

        async def run_blocking(func, *args, **kwargs):
            return func(*args, **kwargs)

        adapter._run_blocking = run_blocking
        return await adapter._fetch_last_message_in_thread("omt_thread"), requests

    result, requests = asyncio.run(run())
    assert result == "om_newest"
    assert len(requests) == 1
    assert requests[0].sort_type == "ByCreateTimeDesc"

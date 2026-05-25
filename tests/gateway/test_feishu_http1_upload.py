"""Public media delivery must recover the reported SDK HTTP/2 stream reset."""
import json
from types import SimpleNamespace

import httpx
import pytest

from gateway.config import PlatformConfig
from plugins.platforms.feishu.adapter import FeishuAdapter


@pytest.mark.asyncio
@pytest.mark.parametrize("kind,domain", [("image", "feishu"), ("file", "lark")])
@pytest.mark.parametrize("mode", ["reset", "normal", "dns", "empty", "token-error", "upload-error"])
async def test_stream_reset_recovers_complete_media_delivery(tmp_path, monkeypatch, kind, domain, mode):
    adapter = FeishuAdapter(PlatformConfig(extra={
        "app_id": "fixture-app", "app_secret": "fixture-secret", "domain": domain,
    }))
    payload = b"local fixture media bytes"
    media = tmp_path / ("image.png" if kind == "image" else "report.txt")
    media.write_bytes(payload)
    sent = []
    requests = []

    def upload(request):
        stream = getattr(request.request_body, kind)
        assert stream.read() == payload
        if mode == "normal":
            return SimpleNamespace(success=lambda: True, data=SimpleNamespace(**{kind + "_key": "fixture-key"}))
        if mode == "empty":
            return SimpleNamespace(success=lambda: True, data=None)
        if mode == "dns":
            raise ConnectionError("DNS resolution failed")
        raise ConnectionError("Stream 1 was reset by remote peer. Reason: 0x1.")

    def send(request):
        sent.append(request)
        return SimpleNamespace(success=lambda: True, data=SimpleNamespace(message_id="om_fixture"))

    adapter._client = SimpleNamespace(im=SimpleNamespace(v1=SimpleNamespace(
        **{kind: SimpleNamespace(create=upload)}, message=SimpleNamespace(create=send, reply=send),
    )))

    def transport(request):
        requests.append(request)
        if request.url.path.endswith("/tenant_access_token/internal"):
            assert json.loads(request.content) == {"app_id": "fixture-app", "app_secret": "fixture-secret"}
            if mode == "token-error":
                return httpx.Response(200, text="not JSON")
            return httpx.Response(200, json={"code": 0, "tenant_access_token": "fixture-token"})
        if mode == "upload-error":
            return httpx.Response(503, text="unavailable")
        assert payload in request.content
        assert request.headers["authorization"] == "Bearer fixture-token"
        return httpx.Response(200, json={"code": 0, "data": {kind + "_key": "fixture-key"}})

    original_client = httpx.AsyncClient
    def client(**kwargs):
        assert kwargs["http2"] is False
        return original_client(**kwargs, transport=httpx.MockTransport(transport))
    monkeypatch.setattr(httpx, "AsyncClient", client)
    try:
        if kind == "image":
            result = await adapter.send_image_file("oc_fixture", str(media), reply_to="om_parent", caption="caption")
        else:
            result = await adapter.send_document("oc_fixture", str(media), reply_to="om_parent", caption="caption", file_name="custom.txt")
    finally:
        if adapter._sdk_executor:
            adapter._sdk_executor.shutdown(wait=True)
    if mode in {"dns", "empty", "token-error", "upload-error"}:
        assert not result.success
        assert not sent
        assert len(requests) == {"dns": 0, "empty": 0, "token-error": 1, "upload-error": 2}[mode]
        return
    assert result.success, result.error
    assert result.message_id == "om_fixture"
    assert len(sent) == 1
    assert "fixture-key" in sent[0].request_body.content
    assert "caption" in sent[0].request_body.content
    if mode == "normal":
        assert not requests
        return
    host = "open.feishu.cn" if domain == "feishu" else "open.larksuite.com"
    assert [r.url.host for r in requests] == [host, host]
    assert requests[-1].url.path == f"/open-apis/im/v1/{kind}s"
    if kind == "file":
        assert b"custom.txt" in requests[-1].content

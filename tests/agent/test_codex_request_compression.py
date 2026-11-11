"""Official Codex requests use lossless zstd up to 256 KiB and preserve larger JSON bodies."""

import json

import httpx
import pytest

from agent.process_bootstrap import build_keepalive_http_client

zstd = pytest.importorskip("compression.zstd")

OFFICIAL = "https://chatgpt.com/backend-api/codex"


def _send_through_hooks(client, url: str, payload: dict):
    request = client.build_request("POST", url, json=payload)
    for hook in client.event_hooks["request"]:
        hook(request)
    return request


def test_official_codex_responses_body_is_zstd_and_lossless():
    payload = {"model": "gpt-6.1-sol", "input": [{"role": "user", "content": "hi " * 500}]}
    client = build_keepalive_http_client(OFFICIAL)
    try:
        request = _send_through_hooks(client, f"{OFFICIAL}/responses", payload)
    finally:
        client.close()
    assert request.headers["Content-Encoding"] == "zstd"
    assert int(request.headers["Content-Length"]) == len(request.content)
    assert json.loads(zstd.decompress(request.content)) == payload


@pytest.mark.parametrize("base_url, path", [
    ("https://my-proxy.example.com/backend-api/codex", "/responses"),  # custom proxy: never compressed
    ("https://api.openai.com/v1", "/responses"),
    (OFFICIAL, "/models"),  # official host, non-Responses endpoint
])
def test_other_destinations_stay_raw_json(base_url, path):
    payload = {"model": "x", "input": "hello"}
    client = build_keepalive_http_client(base_url)
    try:
        request = _send_through_hooks(client, f"{base_url}{path}", payload)
    finally:
        client.close()
    assert "Content-Encoding" not in request.headers
    assert json.loads(request.content) == payload


async def _send_through_transport(async_mode, body: bytes, headers: dict):
    configured = build_keepalive_http_client(OFFICIAL, async_mode=async_mode)
    sent = []

    def receive(request):
        # The transport consumes the stream, which can differ from request.content's cache.
        sent.append((dict(request.headers), b"".join(request.stream)))
        return httpx.Response(200)

    transport = httpx.MockTransport(receive)
    try:
        if async_mode:
            async with httpx.AsyncClient(transport=transport, event_hooks=configured.event_hooks) as client:
                request = client.build_request("POST", f"{OFFICIAL}/responses", content=body, headers=headers)
                original_headers = dict(request.headers)
                await client.send(request)
        else:
            with httpx.Client(transport=transport, event_hooks=configured.event_hooks) as client:
                request = client.build_request("POST", f"{OFFICIAL}/responses", content=body, headers=headers)
                original_headers = dict(request.headers)
                client.send(request)
    finally:
        if async_mode:
            await configured.aclose()
        else:
            configured.close()
    return original_headers, *sent[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("body_size, expected_encoding", [
    (262143, "zstd"), (262144, "zstd"), (262145, None),
])
async def test_compression_preserves_json_and_stops_above_256_kib(async_mode, body_size, expected_encoding):
    # A multibyte character makes the boundary about encoded bytes, not string length.
    payload = {"model": "gpt-6.1-sol", "input": "é"}
    overhead = len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode())
    payload["input"] += "x" * (body_size - overhead)
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    assert len(body) == body_size
    original_headers, sent_headers, sent_body = await _send_through_transport(
        async_mode, body, {"Content-Type": "application/json", "session_id": "test-session"},
    )
    assert sent_headers.get("content-encoding") == expected_encoding
    assert int(sent_headers["content-length"]) == len(sent_body)
    if expected_encoding is None:
        assert sent_body == body
        assert sent_headers == original_headers
    else:
        assert zstd.decompress(sent_body) == body
        assert sent_headers["content-type"] == "application/json"
        assert sent_headers["session_id"] == "test-session"


@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
async def test_existing_content_encoding_is_preserved(async_mode):
    body = zstd.compress(b'{"input":"already encoded"}')
    original_headers, sent_headers, sent_body = await _send_through_transport(
        async_mode, body, {"Content-Type": "application/json", "Content-Encoding": "zstd"},
    )
    assert sent_body == body
    assert sent_headers == original_headers

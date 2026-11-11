"""Codex Responses bodies go out zstd-compressed to the official backend only (official-client parity)."""

import json

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


@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("body_size", [262143, 262144, 262145])
async def test_compression_preserves_json_and_stops_above_256_kib(async_mode, body_size):
    # A multibyte character makes the boundary about encoded bytes, not string length.
    payload = {"model": "gpt-6.1-sol", "input": "é"}
    overhead = len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode())
    payload["input"] += "x" * (body_size - overhead)
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    assert len(body) == body_size
    client = build_keepalive_http_client(OFFICIAL, async_mode=async_mode)
    try:
        request = client.build_request("POST", f"{OFFICIAL}/responses", content=body,
                                       headers={"Content-Type": "application/json", "session_id": "test-session"})
        original_headers = dict(request.headers)
        for hook in client.event_hooks["request"]:
            if async_mode:
                await hook(request)
            else:
                hook(request)
        if body_size > 262144:
            assert request.content == body
            assert dict(request.headers) == original_headers
        else:
            assert request.headers["Content-Encoding"] == "zstd"
            assert int(request.headers["Content-Length"]) == len(request.content)
            assert zstd.decompress(request.content) == body
            assert request.headers["Content-Type"] == "application/json"
            assert request.headers["session_id"] == "test-session"
    finally:
        if async_mode:
            await client.aclose()
        else:
            client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
async def test_existing_content_encoding_is_preserved(async_mode):
    body = zstd.compress(b'{"input":"already encoded"}')
    client = build_keepalive_http_client(OFFICIAL, async_mode=async_mode)
    try:
        request = client.build_request("POST", f"{OFFICIAL}/responses", content=body,
                                       headers={"Content-Type": "application/json", "Content-Encoding": "zstd"})
        original_headers = dict(request.headers)
        for hook in client.event_hooks["request"]:
            if async_mode:
                await hook(request)
            else:
                hook(request)
        assert request.content == body
        assert dict(request.headers) == original_headers
    finally:
        if async_mode:
            await client.aclose()
        else:
            client.close()

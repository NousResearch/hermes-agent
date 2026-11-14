"""Codex Responses bodies go out zstd-compressed to the official backend only (official-client parity)."""

import asyncio
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


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("body_size", [131072, 131073, 400000])
def test_multiblock_codex_bodies_stay_raw_without_losing_bytes(async_mode, body_size):
    """Regress the live 128 KiB / 128 KiB+1 text-only encoding A/B boundary.

    Some valid multi-block frames get a generic 400; streaming zstd is not a
    general cure. Keep single-block compression, otherwise send the full JSON.
    """
    payload = {"model": "gpt-6.1-sol", "input": [
        {"role": "user", "content": "protocol fixture: příliš žluťoučký\n"},
    ]}
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    raw = raw[:-4] + b"x" * (body_size - len(raw)) + raw[-4:]
    assert len(raw) == body_size
    json.loads(raw)
    client = build_keepalive_http_client(OFFICIAL, async_mode=async_mode)
    assert client is not None
    request = client.build_request("POST", f"{OFFICIAL}/responses", content=raw,
                                   headers={"Content-Type": "application/json"})

    async def apply_async():
        try:
            for hook in client.event_hooks["request"]:
                await hook(request)
                encoded = request.content
                await hook(request)
                assert request.content == encoded
        finally:
            await client.aclose()

    if async_mode:
        asyncio.run(apply_async())
    else:
        try:
            for hook in client.event_hooks["request"]:
                hook(request)
                encoded = request.content
                hook(request)
                assert request.content == encoded
        finally:
            client.close()
    assert int(request.headers["Content-Length"]) == len(request.content)
    if body_size <= 131072:
        assert request.headers["Content-Encoding"] == "zstd"
        assert zstd.decompress(request.content) == raw
    else:
        assert "Content-Encoding" not in request.headers
        assert request.content == raw

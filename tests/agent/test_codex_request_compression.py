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

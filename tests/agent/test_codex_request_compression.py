"""Official Codex Responses use plain JSON unless zstd is explicitly enabled."""

import asyncio
import json

import pytest

from agent.process_bootstrap import build_keepalive_http_client

OFFICIAL = "https://chatgpt.com/backend-api/codex"


@pytest.fixture
def enable_codex_compression(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "agent:\n  codex_request_compression: true\n", encoding="utf-8"
    )


def _send_through_hooks(client, url: str, payload: dict):
    request = client.build_request("POST", url, json=payload)
    for hook in client.event_hooks["request"]:
        hook(request)
    return request


def test_official_codex_opt_in_is_zstd_and_lossless(enable_codex_compression):
    zstd = pytest.importorskip("compression.zstd")
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
def test_other_destinations_stay_raw_json(base_url, path, enable_codex_compression):
    payload = {"model": "x", "input": "hello"}
    client = build_keepalive_http_client(base_url)
    try:
        request = _send_through_hooks(client, f"{base_url}{path}", payload)
    finally:
        client.close()
    assert "Content-Encoding" not in request.headers
    assert json.loads(request.content) == payload


@pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
def test_official_codex_defaults_to_plain_json(tmp_path, monkeypatch, async_mode):
    """A default profile cannot hit the provider's zstd rejection (#148898)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    client = build_keepalive_http_client(OFFICIAL, async_mode=async_mode)
    assert client is not None
    try:
        payload = {"model": "gpt-6.1-sol", "input": "hello"}
        request = client.build_request("POST", f"{OFFICIAL}/responses", json=payload)
        for hook in client.event_hooks["request"]:
            if async_mode:
                asyncio.run(hook(request))
            else:
                hook(request)
        assert "Content-Encoding" not in request.headers
        assert json.loads(request.content) == payload
    finally:
        if async_mode:
            asyncio.run(client.aclose())
        else:
            client.close()

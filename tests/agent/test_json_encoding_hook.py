import json
import httpx
import pytest

from agent.process_bootstrap import build_keepalive_http_client


def test_sync_client_json_encoding_hook_preserves_utf8_emoji():
    client = build_keepalive_http_client("https://example.com/v1", async_mode=False)
    assert isinstance(client, httpx.Client)

    received_body = []

    def handle_request(request: httpx.Request) -> httpx.Response:
        received_body.append(request.read())
        return httpx.Response(200, json={"ok": True})

    mock_transport = httpx.MockTransport(handle_request)
    client._mounts = {pattern: mock_transport for pattern in client._mounts}

    payload = {"role": "user", "content": "Hello 🌍 🚀"}
    raw_json = json.dumps(payload, ensure_ascii=True).encode("utf-8")

    req = client.build_request(
        "POST",
        "https://example.com/v1/chat/completions",
        content=raw_json,
        headers={"Content-Type": "application/json"},
    )

    resp = client.send(req)
    assert resp.status_code == 200
    assert len(received_body) == 1
    decoded = received_body[0].decode("utf-8")
    assert "Hello 🌍 🚀" in decoded
    assert "\\u" not in decoded


@pytest.mark.anyio
async def test_async_client_json_encoding_hook_preserves_utf8_emoji():
    client = build_keepalive_http_client("https://example.com/v1", async_mode=True)
    assert isinstance(client, httpx.AsyncClient)

    received_body = []

    def handle_request(request: httpx.Request) -> httpx.Response:
        received_body.append(request.read())
        return httpx.Response(200, json={"ok": True})

    mock_transport = httpx.MockTransport(handle_request)
    client._mounts = {pattern: mock_transport for pattern in client._mounts}

    payload = {"role": "user", "content": "Gemini emoji: ✨🔥🎉"}
    raw_json = json.dumps(payload, ensure_ascii=True).encode("utf-8")

    req = client.build_request(
        "POST",
        "https://example.com/v1/chat/completions",
        content=raw_json,
        headers={"Content-Type": "application/json"},
    )

    resp = await client.send(req)
    assert resp.status_code == 200
    assert len(received_body) == 1
    decoded = received_body[0].decode("utf-8")
    assert "Gemini emoji: ✨🔥🎉" in decoded
    assert "\\u" not in decoded

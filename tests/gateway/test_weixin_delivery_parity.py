"""Tencent CDN retry behavior and metadata on profile-local one-shot delivery."""

import json

import pytest

from gateway.platforms import weixin


class Response:
    def __init__(self, status=200, headers=None, payload=None):
        self.status = status
        self.ok = status < 400
        self.headers = headers or {}
        self.payload = payload or {"ret": 0, "message_id": "server-message-id"}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def read(self):
        return b""

    async def text(self):
        return json.dumps(self.payload)


class Session:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.posts = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def post(self, url, **kwargs):
        self.posts.append((url, kwargs))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


@pytest.mark.asyncio
@pytest.mark.parametrize("first_response", [Response(503), Response(200), OSError("connection reset")])
async def test_cdn_upload_retries_recoverable_failures_with_same_ciphertext(first_response):
    session = Session([first_response, Response(200, {"x-encrypted-param": "download-query"})])

    assert await weixin._upload_ciphertext(session, ciphertext=b"ciphertext", upload_url="https://cdn.example/upload") == "download-query"
    assert len(session.posts) == 2
    assert all(request["data"] == b"ciphertext" for _, request in session.posts)


@pytest.mark.asyncio
async def test_cdn_upload_does_not_retry_client_rejections():
    session = Session([Response(403), Response(200, {"x-encrypted-param": "unexpected-retry"})])

    with pytest.raises(RuntimeError, match="403"):
        await weixin._upload_ciphertext(session, ciphertext=b"ciphertext", upload_url="https://cdn.example/upload")
    assert len(session.posts) == 1


@pytest.mark.asyncio
async def test_cdn_upload_stops_after_three_failed_attempts():
    session = Session([Response(503)] * 4)

    with pytest.raises(RuntimeError, match="503"):
        await weixin._upload_ciphertext(session, ciphertext=b"ciphertext", upload_url="https://cdn.example/upload")
    assert len(session.posts) == 3


@pytest.mark.asyncio
async def test_one_shot_send_uses_own_metadata_without_a_live_gateway(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(weixin, "_LIVE_ADAPTERS", {})
    session = Session([Response()])
    monkeypatch.setattr(weixin, "_new_session", lambda **kwargs: session)

    result = await weixin.send_weixin_direct(
        token="one-shot-token", chat_id="recipient", message="Hello",
        extra={"account_id": "one-shot-bot", "bot_agent": "Scheduled/1.0", "route_tag": 42},
    )

    assert result["success"]
    assert result["message_id"] == "server-message-id"
    _, request = session.posts[0]
    assert request["headers"]["SKRouteTag"] == "42"
    assert json.loads(request["data"])["base_info"]["bot_agent"] == "Scheduled/1.0"

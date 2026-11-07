"""QQ Bot standalone sender honors ``platforms.qqbot.extra.markdown_support``.

The in-process adapter sends ``msg_type=2`` (markdown) in C2C and group scenes when the
switch is on; the standalone sender used for ``send_message`` / cron delivery / plugin
notifications hardcoded ``msg_type=0``, so markdown arrived as literal asterisks.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

from plugins.platforms.qqbot.send import _qqbot_send_text_message
from plugins.platforms.qqbot.send import send_qqbot as _send_qqbot


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {"id": "msg-1"}

    def json(self) -> dict:
        return self._payload


def _endpoint(url: str) -> str:
    if "/channels/" in url:
        return "channel"
    return "c2c" if "/users/" in url else "group"


class _FakeClient:
    """Records POST bodies; ``fail_channel`` mimics a C2C openid that is not a channel id."""

    def __init__(self, fail_channel: bool = True):
        self.fail_channel = fail_channel
        self.calls: list[tuple[str, dict]] = []

    async def post(self, url: str, json: dict | None = None, headers=None, **kwargs):
        body = json or {}
        self.calls.append((_endpoint(url), body))
        if self.fail_channel and "/channels/" in url:
            return _FakeResponse(400, {"message": "unknown channel"})
        return _FakeResponse(200)


def test_guild_channel_stays_plain_even_with_markdown_enabled():
    """Guild markdown is invitation-only; the adapter's _send_guild_text() sends content too."""
    client = _FakeClient(fail_channel=False)
    result = asyncio.run(_qqbot_send_text_message(client, {}, "guild-1", "**bold**", markdown=True))

    assert result.get("error") is None
    assert client.calls == [("channel", {"content": "**bold**", "msg_type": 0})]


def test_c2c_gets_markdown_body_when_channel_is_not_found():
    client = _FakeClient(fail_channel=True)
    result = asyncio.run(_qqbot_send_text_message(client, {}, "openid-1", "**bold**", markdown=True))

    assert result.get("error") is None
    kind, body = client.calls[-1]
    assert kind == "c2c"
    assert body == {"markdown": {"content": "**bold**"}, "msg_type": 2}


def test_every_endpoint_is_plain_when_markdown_disabled():
    client = _FakeClient(fail_channel=True)
    asyncio.run(_qqbot_send_text_message(client, {}, "openid-1", "**bold**", markdown=False))

    assert [(kind, body["msg_type"]) for kind, body in client.calls] == [("channel", 0), ("c2c", 0)]
    assert all("markdown" not in body for _kind, body in client.calls)


class _FakeAsyncClient:
    """Stands in for ``httpx.AsyncClient`` inside ``send_qqbot``."""

    instances: list[_FakeAsyncClient] = []

    def __init__(self, *args, **kwargs):
        self.inner = _FakeClient(fail_channel=True)
        _FakeAsyncClient.instances.append(self)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url: str, json: dict | None = None, headers=None, **kwargs):
        if "getAppAccessToken" in url:
            return _FakeResponse(200, {"access_token": "token-1"})
        return await self.inner.post(url, json=json, headers=headers, **kwargs)


def _run_send(extra: dict) -> list[dict]:
    _FakeAsyncClient.instances.clear()
    pconfig = SimpleNamespace(token="client-secret", extra={"app_id": "app-123", **extra}, enabled=True)
    with patch("httpx.AsyncClient", _FakeAsyncClient):
        asyncio.run(_send_qqbot(pconfig, "openid-1", "**bold**"))
    return [body for _kind, body in _FakeAsyncClient.instances[0].inner.calls]


def test_send_qqbot_reads_the_switch():
    """Default (no extra key) matches the adapter's True; an explicit false opts out."""
    assert _run_send({})[-1] == {"markdown": {"content": "**bold**"}, "msg_type": 2}
    assert [body["msg_type"] for body in _run_send({"markdown_support": False})] == [0, 0]

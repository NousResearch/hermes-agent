"""``wecom_callback`` must deliver out-of-process like its sibling ``wecom``.

A platform entry without a ``standalone_sender_fn`` makes the registry fallback
in ``tools/send_message_tool.py`` hit ``sender is None``: cron jobs carrying
``deliver=wecom_callback:<user_id>`` and ``send_message(platform=
"wecom_callback")`` fail when they run outside the gateway (#125072). The
sibling ``wecom`` entry registers one at its registration site; ``wecom_callback``
registers none.

The callback sender deliberately does NOT reuse the WebSocket-based
``wecom`` pattern (``connect()`` → send → ``disconnect()``): here ``connect()``
binds the callback listener port and refuses when the gateway already holds it.
Callback delivery is outbound only, so the sender opens just the outbound HTTP
client and closes it again.
"""
from __future__ import annotations

import asyncio
import importlib
import json

import pytest

wecom_adapter = importlib.import_module("plugins.platforms.wecom.adapter")


class _FakeResponse:
    def __init__(self, url: str, payload: dict):
        self._payload = payload
        self._url = url

    def json(self):
        return self._payload

    @property
    def status_code(self) -> int:
        return 200


class _RecordingClient:
    """Records outbound calls; answers a token request then each send."""

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []
        self.closed = False

    async def get(self, url, params=None):
        self.calls.append(("GET", {"url": url, "params": params}))
        return _FakeResponse(url, {
            "errcode": 0,
            "access_token": "fake-access-token",
            "expires_in": 7200,
        })

    async def post(self, url, json=None, **kwargs):
        self.calls.append(("POST", {"url": url, "json": json}))
        return _FakeResponse(url, {"errcode": 0, "msgid": "MSG-1"})

    async def aclose(self):
        self.closed = True


class _FakeCtx:
    def __init__(self):
        self.registrations: dict[str, dict] = {}

    def register_platform(self, name, **kwargs):
        self.registrations[name] = kwargs


def _callback_registration() -> dict:
    ctx = _FakeCtx()
    wecom_adapter.register(ctx)
    assert "wecom_callback" in ctx.registrations, "the callback platform must register"
    return ctx.registrations["wecom_callback"]


def test_wecom_callback_registers_a_standalone_sender():
    """The registry needs a callable or out-of-process sends cannot route."""
    registration = _callback_registration()

    sender = registration.get("standalone_sender_fn")
    assert sender is not None, (
        "wecom_callback registers no standalone_sender_fn, so "
        "tools/send_message_tool.py has nothing to call for cron/CLI delivery"
    )
    assert callable(sender)


def test_wecom_entry_keeps_its_own_sender():
    """Guard against collapsing the two registrations into one sender."""
    ctx = _FakeCtx()
    wecom_adapter.register(ctx)

    assert ctx.registrations["wecom"]["standalone_sender_fn"] is not None
    assert (
        ctx.registrations["wecom"]["standalone_sender_fn"]
        is not ctx.registrations["wecom_callback"]["standalone_sender_fn"]
    ), "the callback sender must not reuse the WebSocket-based wecom sender"


def test_callback_sender_delivers_without_connecting(monkeypatch):
    """Outbound only: token + send over the HTTP client, no listener bind.

    ``connect()`` binds the callback port and refuses when the gateway already
    holds it, so the sender must not go through it — and must not leave a
    dangling aiohttp application either.
    """
    from plugins.platforms.wecom import callback_adapter as cb_module

    client = _RecordingClient()

    import httpx

    monkeypatch.setattr(
        httpx, "AsyncClient", lambda *a, **kw: client, raising=True
    )
    connect_calls: list[bool] = []
    monkeypatch.setattr(
        cb_module.WecomCallbackAdapter,
        "connect",
        lambda self, **kw: connect_calls.append(True) or asyncio.sleep(0),
        raising=True,
    )

    from plugins.platforms.wecom.callback_adapter import WecomCallbackAdapter

    def _fake_factory(pconfig):
        adapter = WecomCallbackAdapter.__new__(WecomCallbackAdapter)
        adapter._apps = [{
            "name": "default",
            "corp_id": "wwcorp",
            "corp_secret": "secret",
            "agent_id": "1000002",
            "token": "",
            "encoding_aes_key": "",
        }]
        adapter._http_client = client
        adapter._access_tokens = {}
        adapter._user_app_map = {}
        return adapter

    monkeypatch.setattr(wecom_adapter, "_build_callback_adapter", _fake_factory)

    registration = _callback_registration()
    sender = registration["standalone_sender_fn"]

    result = asyncio.run(
        sender(
            None,
            "wwcorp:zhangsan",
            "hello from cron",
        )
    )

    assert not connect_calls, "the standalone sender must not bind the callback listener"
    assert result.get("success") is True, result
    assert result.get("platform") == "wecom_callback"
    assert client.closed, "the outbound client must be closed afterwards"

    methods = [call for call, _ in client.calls]
    assert "GET" in methods and "POST" in methods
    post_body = next(payload for name, payload in client.calls if name == "POST")
    assert post_body["json"]["touser"] == "zhangsan"
    assert post_body["json"]["text"]["content"] == "hello from cron"


def test_callback_sender_reports_missing_requirements(monkeypatch):
    """A missing dependency must come back as an error, not an exception."""
    from plugins.platforms.wecom import callback_adapter as cb_module

    monkeypatch.setattr(
        cb_module, "check_wecom_callback_requirements", lambda: False, raising=True
    )

    registration = _callback_registration()
    sender = registration["standalone_sender_fn"]

    result = asyncio.run(sender(None, "wwcorp:zhangsan", "hi"))

    assert "error" in result
    assert "requirement" in result["error"].lower()


def test_callback_sender_reports_a_failed_send(monkeypatch):
    """A rejected send surfaces the adapter's error string."""
    from plugins.platforms.wecom.callback_adapter import SendResult, WecomCallbackAdapter

    class _FailingClient(_RecordingClient):
        async def post(self, url, json=None, **kwargs):
            self.calls.append(("POST", {"url": url, "json": json}))
            return _FakeResponse(url, {"errcode": 40014, "errmsg": "invalid token"})

    client = _FailingClient()
    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **kw: client, raising=True)

    def _fake_factory(pconfig):
        adapter = WecomCallbackAdapter.__new__(WecomCallbackAdapter)
        adapter._apps = [{
            "name": "default", "corp_id": "wwcorp", "corp_secret": "secret",
            "agent_id": "1", "token": "", "encoding_aes_key": "",
        }]
        adapter._http_client = client
        adapter._access_tokens = {}
        adapter._user_app_map = {}
        return adapter

    monkeypatch.setattr(wecom_adapter, "_build_callback_adapter", _fake_factory)

    registration = _callback_registration()
    result = asyncio.run(
        registration["standalone_sender_fn"](None, "wwcorp:zhangsan", "hi")
    )

    assert "error" in result
    assert client.closed, "the client is closed even when the send fails"

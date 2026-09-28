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


# ── Multi-app routing (#125092) ──────────────────────────────────────────────
# The standalone sender never binds a listener, so ``_user_app_map`` — whose
# only writer is the inbound HTTP handler — is always empty for it. Routing a
# send through ``_resolve_app_for_chat`` therefore fell all the way through to
# the ``self._apps[0]`` default, and every outbound delivery in a multi-app
# config used the FIRST app's appid: dept-b's user was sent under dept-a's
# chat_agent_id, into the wrong enterprise, while the caller was told
# ``{"success": True}``.


def _multi_app_adapter_class():
    from plugins.platforms.wecom.callback_adapter import WecomCallbackAdapter

    return WecomCallbackAdapter


def _multi_app_pair(client):
    """Two apps shaped like ``_normalize_apps`` output for a multi-app config."""
    WecomCallbackAdapter = _multi_app_adapter_class()
    adapter = WecomCallbackAdapter.__new__(WecomCallbackAdapter)
    adapter._apps = [
        {
            "name": "dept-a",
            "corp_id": "ww_corp_a",
            "corp_secret": "secret-a",
            "agent_id": "1000002",
            "token": "",
            "encoding_aes_key": "",
        },
        {
            "name": "dept-b",
            "corp_id": "ww_corp_b",
            "corp_secret": "secret-b",
            "agent_id": "1000003",
            "token": "",
            "encoding_aes_key": "",
        },
    ]
    adapter._http_client = client
    adapter._access_tokens = {}
    adapter._user_app_map = {}
    return adapter


def _token_params(client):
    """The corpid each token request asked for — i.e. which app authenticated."""
    return [payload["params"].get("corpid") for name, payload in client.calls if name == "GET"]


def test_dept_b_user_is_not_sent_through_dept_a(monkeypatch):
    """The reviewer's repro table: dept-b must authenticate as dept-b.

    Before the fix both rows resolved to ``apps[0]``, so the dept-b row
    requested dept-a's token and carried dept-a's agentid.
    """
    client = _RecordingClient()
    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **kw: client, raising=True)

    adapter = _multi_app_pair(client)
    monkeypatch.setattr(
        wecom_adapter, "_build_callback_adapter", lambda pconfig: adapter, raising=True
    )

    sender = _callback_registration()["standalone_sender_fn"]

    result = asyncio.run(sender(None, "ww_corp_b:zhangsan", "hi dept-b"))

    assert result.get("success") is True, result
    assert _token_params(client) == ["ww_corp_b"], (
        "dept-b delivery must authenticate against dept-b's corp_id"
    )
    post_body = next(payload for name, payload in client.calls if name == "POST")
    assert post_body["json"]["agentid"] == 1000003, "dept-b's own agentid must be used"
    assert post_body["json"]["touser"] == "zhangsan", "the user id itself is unchanged"


def test_dept_a_user_still_resolves_dept_a(monkeypatch):
    """The second row of the table: dept-a must keep working, and independently."""
    client = _RecordingClient()
    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **kw: client, raising=True)

    adapter = _multi_app_pair(client)
    monkeypatch.setattr(
        wecom_adapter, "_build_callback_adapter", lambda pconfig: adapter, raising=True
    )

    sender = _callback_registration()["standalone_sender_fn"]

    result = asyncio.run(sender(None, "ww_corp_a:zhangsan", "hi dept-a"))

    assert result.get("success") is True, result
    assert _token_params(client) == ["ww_corp_a"]
    post_body = next(payload for name, payload in client.calls if name == "POST")
    assert post_body["json"]["agentid"] == 1000002


def test_unknown_corp_id_fails_instead_of_defaulting(monkeypatch):
    """A corp prefix no app owns must error, NOT silently fall back to apps[0]."""
    client = _RecordingClient()
    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **kw: client, raising=True)

    adapter = _multi_app_pair(client)
    monkeypatch.setattr(
        wecom_adapter, "_build_callback_adapter", lambda pconfig: adapter, raising=True
    )

    sender = _callback_registration()["standalone_sender_fn"]

    result = asyncio.run(sender(None, "ww_corp_c:zhangsan", "are you there?"))

    assert result.get("success") is not True, result
    assert "ww_corp_c" in (result.get("error") or ""), (
        "the error must name the corp it could not route, so the operator can fix the config"
    )
    assert not [c for c in client.calls if c[0] == "POST"], "no message may be sent on a failed route"


def test_bare_user_id_in_multi_app_does_not_guess(monkeypatch):
    """A legacy bare user_id cannot be routed among several apps — fail loudly.

    Guessing ``apps[0]`` here is the same cross-corp misdelivery as the
    corp-scoped case, just without a prefix to detect it by.
    """
    client = _RecordingClient()
    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **kw: client, raising=True)

    adapter = _multi_app_pair(client)
    monkeypatch.setattr(
        wecom_adapter, "_build_callback_adapter", lambda pconfig: adapter, raising=True
    )

    sender = _callback_registration()["standalone_sender_fn"]

    result = asyncio.run(sender(None, "zhangsan", "hi"))

    assert result.get("success") is not True, result
    assert "error" in result, result
    assert not [c for c in client.calls if c[0] == "POST"], "no message may be sent on a failed route"


def test_single_app_still_defaults_for_a_bare_user_id(monkeypatch):
    """The pre-scoping config must keep working: one app is not a guess."""
    client = _RecordingClient()
    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **kw: client, raising=True)

    WecomCallbackAdapter = _multi_app_adapter_class()
    adapter = WecomCallbackAdapter.__new__(WecomCallbackAdapter)
    adapter._apps = [
        {
            "name": "default",
            "corp_id": "wwcorp",
            "corp_secret": "secret",
            "agent_id": "1000002",
            "token": "",
            "encoding_aes_key": "",
        }
    ]
    adapter._http_client = client
    adapter._access_tokens = {}
    adapter._user_app_map = {}
    monkeypatch.setattr(
        wecom_adapter, "_build_callback_adapter", lambda pconfig: adapter, raising=True
    )

    sender = _callback_registration()["standalone_sender_fn"]

    result = asyncio.run(sender(None, "zhangsan", "hi"))

    assert result.get("success") is True, result
    assert _token_params(client) == ["wwcorp"]


def test_inbound_bound_user_wins_over_the_prefix(monkeypatch):
    """The inbound map stays authoritative when it has an entry.

    ``_user_app_map`` is written by the inbound handler with the corp the user
    actually belongs to; a prefix collision must not override that truth.
    """
    client = _RecordingClient()
    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **kw: client, raising=True)

    adapter = _multi_app_pair(client)
    # The inbound handler bound this exact chat_key to dept-a — that is the
    # corp the user's callback actually arrived from.
    adapter._user_app_map = {"ww_corp_b:zhangsan": "dept-a"}
    monkeypatch.setattr(
        wecom_adapter, "_build_callback_adapter", lambda pconfig: adapter, raising=True
    )

    sender = _callback_registration()["standalone_sender_fn"]

    result = asyncio.run(sender(None, "ww_corp_b:zhangsan", "hi"))

    assert result.get("success") is True, result
    post_body = next(payload for name, payload in client.calls if name == "POST")
    assert post_body["json"]["agentid"] == 1000002, (
        "a bound user must stay on the corp the inbound handler recorded"
    )


def test_media_is_reported_not_silently_dropped(monkeypatch):
    """Attachments must fail loudly, not vanish into a ``success: True``.

    The sender used to ``del media_files`` — the caller was told the message
    was delivered while the attachment was gone.
    """
    client = _RecordingClient()
    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **kw: client, raising=True)

    adapter = _multi_app_pair(client)
    monkeypatch.setattr(
        wecom_adapter, "_build_callback_adapter", lambda pconfig: adapter, raising=True
    )

    sender = _callback_registration()["standalone_sender_fn"]

    result = asyncio.run(sender(None, "ww_corp_a:zhangsan", "see attached", media_files=["/tmp/a.png"]))

    assert result.get("success") is not True, result
    assert "text-only" in (result.get("error") or "")
    assert not client.calls, "nothing may be sent when the request cannot be honoured"

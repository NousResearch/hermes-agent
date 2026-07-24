"""Tests for gateway/wake.py — background wake delivery.

Two strategies:
* push-capable adapters keep the synthetic MessageEvent / handle_message path;
* the stateless API server (supports_async_delivery=False) self-POSTs
  /v1/chat/completions with the RAW session id in X-Hermes-Session-Id, so the
  wake turn resumes the REAL session instead of a parallel invisible one
  keyed by build_session_key().
"""

import asyncio

import pytest

from gateway.config import Platform
from gateway.session import SessionSource
from gateway.wake import deliver_wake, adapter_supports_push


class PushAdapter:
    """Default adapter shape — no supports_async_delivery attribute."""

    def __init__(self):
        self.handled = []

    async def handle_message(self, event):
        self.handled.append(event)


class ApiServerLikeAdapter:
    supports_async_delivery = False

    def __init__(self, host="0.0.0.0", port=0, key="test-key", model="hermes"):
        self._host = host
        self._port = port
        self._api_key = key
        self._model_name = model

    async def handle_message(self, event):  # pragma: no cover — must NOT be hit
        raise AssertionError("non-push adapter must not receive handle_message wakes")


def _source():
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="chat-1",
        chat_type="group",
    )


def test_adapter_supports_push_default_true():
    assert adapter_supports_push(PushAdapter()) is True
    assert adapter_supports_push(ApiServerLikeAdapter()) is False


def test_deliver_wake_push_adapter_uses_handle_message():
    adapter = PushAdapter()
    asyncio.run(deliver_wake(adapter, text="wake up", source=_source()))
    assert len(adapter.handled) == 1
    evt = adapter.handled[0]
    assert evt.text == "wake up"
    assert evt.internal is True
    assert evt.source.chat_id == "chat-1"


def test_deliver_wake_push_adapter_requires_source():
    with pytest.raises(ValueError):
        asyncio.run(deliver_wake(PushAdapter(), text="x", session_id="sid"))


def test_deliver_wake_non_push_requires_session_id():
    with pytest.raises(ValueError):
        asyncio.run(deliver_wake(ApiServerLikeAdapter(), text="x", source=_source()))


def test_deliver_wake_non_push_requires_api_key():
    """Session continuation is 403-gated on API_SERVER_KEY — a missing key
    must fail loudly instead of running the wake in a fresh session."""
    adapter = ApiServerLikeAdapter(key="")
    with pytest.raises(RuntimeError, match="API_SERVER_KEY"):
        asyncio.run(deliver_wake(adapter, text="x", session_id="raw-sid"))


async def _serve(handler):
    """Spin an in-process aiohttp server on an ephemeral loopback port."""
    from aiohttp import web

    app = web.Application()
    app.router.add_post("/v1/chat/completions", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, port


def test_deliver_wake_non_push_self_posts_raw_session_id(monkeypatch):
    """The self-post carries the RAW session id header + bearer auth and a
    single user message with stream=false — the exact entry point real
    gateway turns use."""
    from aiohttp import web

    seen = {}

    async def handler(request):
        seen["session_id"] = request.headers.get("X-Hermes-Session-Id")
        seen["auth"] = request.headers.get("Authorization")
        seen["body"] = await request.json()
        return web.json_response({"choices": [{"message": {"content": "ok"}}]})

    async def run():
        runner, port = await _serve(handler)
        try:
            adapter = ApiServerLikeAdapter(host="0.0.0.0", port=port, key="sekrit")
            await deliver_wake(adapter, text="task done — wake", session_id="raw-sid-42")
        finally:
            await runner.cleanup()

    asyncio.run(run())
    assert seen["session_id"] == "raw-sid-42"
    assert seen["auth"] == "Bearer sekrit"
    assert seen["body"]["stream"] is False
    assert seen["body"]["messages"] == [
        {"role": "user", "content": "task done — wake"}
    ]


def test_deliver_wake_retries_429_then_succeeds(monkeypatch):
    """HTTP 429 (max_concurrent_runs cap) is transient — retried with backoff."""
    from aiohttp import web

    import gateway.wake as wake_mod

    monkeypatch.setattr(wake_mod, "_RETRY_DELAYS_SECONDS", (0.01, 0.01, 0.01))
    calls = {"n": 0}

    async def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return web.json_response({"error": "busy"}, status=429)
        return web.json_response({"choices": []})

    async def run():
        runner, port = await _serve(handler)
        try:
            adapter = ApiServerLikeAdapter(port=port)
            await deliver_wake(adapter, text="x", session_id="sid")
        finally:
            await runner.cleanup()

    asyncio.run(run())
    assert calls["n"] == 2


def test_deliver_wake_raises_on_permanent_http_error(monkeypatch):
    """Auth/validation errors (403/400) are permanent — raise immediately so
    the caller can rewind instead of treating the event as delivered."""
    from aiohttp import web

    calls = {"n": 0}

    async def handler(request):
        calls["n"] += 1
        return web.json_response({"error": "forbidden"}, status=403)

    async def run():
        runner, port = await _serve(handler)
        try:
            adapter = ApiServerLikeAdapter(port=port)
            with pytest.raises(RuntimeError, match="HTTP 403"):
                await deliver_wake(adapter, text="x", session_id="sid")
        finally:
            await runner.cleanup()

    asyncio.run(run())
    assert calls["n"] == 1


def test_deliver_wake_raises_after_exhausted_retries(monkeypatch):
    """Connection failures raise after bounded retries — never silent."""
    import gateway.wake as wake_mod

    monkeypatch.setattr(wake_mod, "_RETRY_DELAYS_SECONDS", (0.01,))
    # Nothing is listening on this port.
    adapter = ApiServerLikeAdapter(host="127.0.0.1", port=1, key="k")
    with pytest.raises(RuntimeError, match="gave up"):
        asyncio.run(deliver_wake(adapter, text="x", session_id="sid"))

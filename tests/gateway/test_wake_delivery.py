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


def test_persist_delegation_delivery_appends_delivery_row(tmp_path):
    """#85957: the delegation completion lands in the session transcript as a
    display_kind=async_delegation_complete delivery row (real SessionDB), and
    NO self-post / agent turn is involved."""
    from pathlib import Path

    from gateway.wake import persist_delegation_delivery
    from hermes_state import SessionDB

    db = SessionDB(db_path=Path(tmp_path) / "state.db")
    sid = "raw-hq-sid"
    db.create_session(sid, source="api_server")
    db.append_message(sid, "user", content="please confirm before writing")
    db.append_message(sid, "assistant", content="awaiting confirmation",
                      finish_reason="stop")

    class DbAdapter(ApiServerLikeAdapter):
        def _ensure_session_db(self):
            return db

    evt = {
        "type": "async_delegation",
        "delegation_id": "deleg_x",
        "results": [{"status": "completed"}, {"status": "failed"}],
        "total_duration_seconds": 12.5,
    }
    asyncio.run(persist_delegation_delivery(
        DbAdapter(), text="[ASYNC DELEGATION BATCH COMPLETE — deleg_x]",
        session_id=sid, evt=evt,
    ))

    rows = db.get_messages(sid)
    assert len(rows) == 3
    delivery = rows[-1]
    assert delivery["role"] == "user"
    assert delivery["display_kind"] == "async_delegation_complete"
    meta = delivery["display_metadata"]
    assert meta["delegation_id"] == "deleg_x"
    assert meta["task_count"] == 2
    assert meta["failed_count"] == 1
    assert meta["duration_seconds"] == 12.5


def test_persist_delegation_delivery_raises_without_db():
    """DB unavailable must RAISE so the durable claim is released for retry."""
    from gateway.wake import persist_delegation_delivery

    class NoDbAdapter(ApiServerLikeAdapter):
        def _ensure_session_db(self):
            return None

    with pytest.raises(RuntimeError, match="SessionDB unavailable"):
        asyncio.run(persist_delegation_delivery(
            NoDbAdapter(), text="x", session_id="sid",
        ))



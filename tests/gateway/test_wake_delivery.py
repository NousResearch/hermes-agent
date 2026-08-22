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


def _bare_session_lock_adapter():
    from gateway.platforms.api_server import APIServerAdapter

    adapter = APIServerAdapter.__new__(APIServerAdapter)
    adapter._session_turn_locks = {}
    adapter._session_turn_lock_refs = {}
    adapter._session_turn_locks_guard = asyncio.Lock()
    return adapter


def test_session_turn_lock_serializes_same_session():
    """Same session_id must not overlap across _run_agent and /v1/runs (#84235)."""
    adapter = _bare_session_lock_adapter()

    active = {"n": 0}
    max_active = {"n": 0}
    order = []

    async def hold(label: str, delay: float):
        async with adapter._hold_session_turn_lock("sess-1"):
            active["n"] += 1
            max_active["n"] = max(max_active["n"], active["n"])
            order.append(f"{label}:start")
            await asyncio.sleep(delay)
            order.append(f"{label}:end")
            active["n"] -= 1

    async def run():
        await asyncio.gather(hold("a", 0.05), hold("b", 0.01))

    asyncio.run(run())
    assert max_active["n"] == 1
    assert order == ["a:start", "a:end", "b:start", "b:end"] or order == [
        "b:start",
        "b:end",
        "a:start",
        "a:end",
    ]
    assert adapter._session_turn_locks == {}
    assert adapter._session_turn_lock_refs == {}


def test_session_turn_lock_allows_different_sessions_in_parallel():
    adapter = _bare_session_lock_adapter()

    active = {"n": 0}
    max_active = {"n": 0}
    gate = asyncio.Event()

    async def hold(session_id: str):
        async with adapter._hold_session_turn_lock(session_id):
            active["n"] += 1
            max_active["n"] = max(max_active["n"], active["n"])
            await gate.wait()
            active["n"] -= 1

    async def run():
        t1 = asyncio.create_task(hold("sess-a"))
        t2 = asyncio.create_task(hold("sess-b"))
        for _ in range(50):
            if max_active["n"] >= 2:
                break
            await asyncio.sleep(0.01)
        gate.set()
        await asyncio.gather(t1, t2)

    asyncio.run(run())
    assert max_active["n"] == 2
    assert adapter._session_turn_locks == {}
    assert adapter._session_turn_lock_refs == {}


def test_session_turn_lock_prunes_idle_and_ephemeral_entries():
    """Idle session ids must not accumulate Lock objects (review on #84876)."""
    adapter = _bare_session_lock_adapter()

    async def run():
        async with adapter._hold_session_turn_lock("sess-1"):
            assert "sess-1" in adapter._session_turn_locks
            assert adapter._session_turn_lock_refs["sess-1"] == 1
        assert adapter._session_turn_locks == {}
        assert adapter._session_turn_lock_refs == {}

        for i in range(20):
            async with adapter._hold_session_turn_lock(f"ephemeral-{i}"):
                pass
        assert adapter._session_turn_locks == {}
        assert adapter._session_turn_lock_refs == {}

        async with adapter._hold_session_turn_lock(""):
            pass
        async with adapter._hold_session_turn_lock("   "):
            pass
        assert adapter._session_turn_locks == {}

    asyncio.run(run())


def test_session_turn_lock_keeps_entry_while_waiter_queued():
    """Do not pop on release while another turn already checked out the Lock."""
    adapter = _bare_session_lock_adapter()

    async def run():
        holder_started = asyncio.Event()
        release_holder = asyncio.Event()

        async def hold_a():
            async with adapter._hold_session_turn_lock("sess-1"):
                holder_started.set()
                await release_holder.wait()

        async def hold_b():
            async with adapter._hold_session_turn_lock("sess-1"):
                pass

        t_a = asyncio.create_task(hold_a())
        await holder_started.wait()
        t_b = asyncio.create_task(hold_b())
        for _ in range(50):
            if adapter._session_turn_lock_refs.get("sess-1", 0) >= 2:
                break
            await asyncio.sleep(0.01)
        assert adapter._session_turn_lock_refs["sess-1"] == 2
        assert "sess-1" in adapter._session_turn_locks
        release_holder.set()
        await asyncio.gather(t_a, t_b)
        assert adapter._session_turn_locks == {}
        assert adapter._session_turn_lock_refs == {}

    asyncio.run(run())


def test_session_turn_lock_prunes_after_cancelled_waiter():
    adapter = _bare_session_lock_adapter()

    async def run():
        holder_started = asyncio.Event()
        release_holder = asyncio.Event()

        async def hold_a():
            async with adapter._hold_session_turn_lock("sess-1"):
                holder_started.set()
                await release_holder.wait()

        async def hold_b():
            async with adapter._hold_session_turn_lock("sess-1"):
                pass

        t_a = asyncio.create_task(hold_a())
        await holder_started.wait()
        t_b = asyncio.create_task(hold_b())
        for _ in range(50):
            if adapter._session_turn_lock_refs.get("sess-1", 0) >= 2:
                break
            await asyncio.sleep(0.01)
        assert adapter._session_turn_lock_refs["sess-1"] == 2
        t_b.cancel()
        with pytest.raises(asyncio.CancelledError):
            await t_b
        assert adapter._session_turn_lock_refs["sess-1"] == 1
        assert "sess-1" in adapter._session_turn_locks
        release_holder.set()
        await t_a
        assert adapter._session_turn_locks == {}
        assert adapter._session_turn_lock_refs == {}

    asyncio.run(run())


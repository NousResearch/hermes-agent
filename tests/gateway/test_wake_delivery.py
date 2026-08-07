"""Tests for gateway/wake.py — background wake delivery.

Two strategies:
* push-capable adapters keep the synthetic MessageEvent / handle_message path;
* the stateless API server (supports_async_delivery=False) self-POSTs
  /v1/chat/completions with the RAW session id in X-Hermes-Session-Id, so the
  wake turn resumes the REAL session instead of a parallel invisible one
  keyed by build_session_key().
"""

import asyncio
import re
from unittest.mock import AsyncMock

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


def test_deliver_wake_push_path_ignores_idempotency_identity():
    """Adding retry identity must not change the synthetic push contract."""
    adapter = PushAdapter()
    source = _source()

    asyncio.run(deliver_wake(
        adapter,
        text="push wake",
        session_id="push-session",
        source=source,
        producer_identity=("delegation", "deleg-push-1"),
    ))

    assert len(adapter.handled) == 1
    event = adapter.handled[0]
    assert event.text == "push wake"
    assert event.source is source
    assert event.internal is True


async def _serve(handler):
    """Spin an in-process aiohttp server on an ephemeral loopback port."""
    from aiohttp import web

    app = web.Application()
    app.router.add_post("/v1/chat/completions", handler)
    return await _serve_app(app)


async def _serve_app(app):
    """Spin an aiohttp application on an ephemeral loopback port."""
    from aiohttp import web

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


def test_deliver_wake_timeout_retry_reuses_nonempty_idempotency_key(monkeypatch):
    """A transport-ambiguous retry represents the same logical wake."""
    from aiohttp import web

    import gateway.wake as wake_mod

    monkeypatch.setattr(wake_mod, "WAKE_TURN_TIMEOUT_SECONDS", 0.03)
    monkeypatch.setattr(wake_mod, "_RETRY_DELAYS_SECONDS", (0,))
    attempt_keys = []
    release_first = asyncio.Event()

    async def handler(request):
        attempt_keys.append(request.headers.get("Idempotency-Key"))
        if len(attempt_keys) == 1:
            await release_first.wait()
        return web.json_response({"choices": []})

    async def run():
        runner, port = await _serve(handler)
        try:
            adapter = ApiServerLikeAdapter(port=port)
            await deliver_wake(adapter, text="retry me", session_id="sid-timeout")
        finally:
            release_first.set()
            await runner.cleanup()

    asyncio.run(run())

    assert len(attempt_keys) == 2
    assert attempt_keys[0]
    assert attempt_keys[0] == attempt_keys[1]


def test_wake_idempotency_keys_are_scoped_opaque_and_bounded():
    """Target and producer scope the key without leaking their raw values."""
    from aiohttp import web

    seen_keys = []
    session_a = "raw-session-alpha-secret"
    session_b = "raw-session-beta-secret"
    producer_a = (
        "async_delegation",
        "delegation-secret-123",
        "process-secret-456",
        "board-secret-789",
        "task-secret-abc",
    )
    producer_b = ("async_delegation", "delegation-secret-other")

    async def handler(request):
        seen_keys.append(request.headers.get("Idempotency-Key"))
        return web.json_response({"choices": []})

    async def run():
        runner, port = await _serve(handler)
        try:
            adapter = ApiServerLikeAdapter(port=port)
            await deliver_wake(
                adapter,
                text="same payload",
                session_id=session_a,
                producer_identity=producer_a,
            )
            await deliver_wake(
                adapter,
                text="same payload",
                session_id=session_b,
                producer_identity=producer_a,
            )
            await deliver_wake(
                adapter,
                text="same payload",
                session_id=session_a,
                producer_identity=producer_b,
            )
        finally:
            await runner.cleanup()

    asyncio.run(run())

    assert all(seen_keys)
    assert seen_keys[0] != seen_keys[1], "target session must scope wake identity"
    assert seen_keys[0] != seen_keys[2], "producer must scope wake identity"
    for key in seen_keys:
        assert len(key) <= 128
        assert re.fullmatch(r"(?:[a-z][a-z0-9_-]*:)*[0-9a-f]{32,64}", key)
        for raw_component in (
            session_a,
            session_b,
            "delegation-secret-123",
            "delegation-secret-other",
            "process-secret-456",
            "board-secret-789",
            "task-secret-abc",
        ):
            assert raw_component not in key


def test_wake_retry_joins_real_chat_completion_handler_inflight_work(
    monkeypatch, tmp_path,
):
    """A timed-out self-post retry joins the handler's shielded agent task."""
    from aiohttp import web

    import gateway.platforms.api_server as api_server_mod
    import gateway.wake as wake_mod

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(wake_mod, "WAKE_TURN_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(wake_mod, "_RETRY_DELAYS_SECONDS", (0,))
    monkeypatch.setattr(
        api_server_mod,
        "_idem_cache",
        api_server_mod._IdempotencyCache(),
    )

    attempt_keys = []
    agent_started = asyncio.Event()
    second_attempt = asyncio.Event()
    release_agent = asyncio.Event()
    run_agent_calls = 0

    @web.middleware
    async def observe_attempts(request, handler):
        attempt_keys.append(request.headers.get("Idempotency-Key"))
        if len(attempt_keys) >= 2:
            second_attempt.set()
        return await handler(request)

    async def run_agent(**_kwargs):
        nonlocal run_agent_calls
        run_agent_calls += 1
        agent_started.set()
        await release_agent.wait()
        return (
            {"final_response": "delivered", "messages": [], "api_calls": 1},
            {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        )

    async def run():
        adapter = api_server_mod.APIServerAdapter(
            api_server_mod.PlatformConfig(
                enabled=True,
                extra={"key": "test-key"},
            )
        )
        monkeypatch.setattr(
            adapter,
            "_ensure_session_db_async",
            AsyncMock(return_value=None),
        )
        monkeypatch.setattr(adapter, "_run_agent", run_agent)

        app = web.Application(middlewares=[observe_attempts])
        app.router.add_post(
            "/v1/chat/completions",
            adapter._handle_chat_completions,
        )
        runner, port = await _serve_app(app)
        adapter._host = "127.0.0.1"
        adapter._port = port
        delivery = asyncio.create_task(deliver_wake(
            adapter,
            text="durable completion",
            session_id="raw-integration-session",
        ))
        try:
            await asyncio.wait_for(agent_started.wait(), timeout=1)
            await asyncio.wait_for(second_attempt.wait(), timeout=1)
            release_agent.set()
            await asyncio.wait_for(delivery, timeout=1)
        finally:
            release_agent.set()
            if not delivery.done():
                delivery.cancel()
            await asyncio.gather(delivery, return_exceptions=True)
            await runner.cleanup()

    asyncio.run(run())

    assert len(attempt_keys) >= 2
    assert all(attempt_keys)
    assert len(set(attempt_keys)) == 1
    assert run_agent_calls == 1

"""A client-side timeout on the wake self-post must not re-POST the same turn.

Regression for #146361: the non-streaming self-POST runs the whole agent turn
server-side, so a client timeout means the turn is already running — retrying the
POST starts the same turn again (the reporter saw one wake run four turns).
"""

import asyncio
from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from gateway import wake


def _adapter(port):
    return SimpleNamespace(_host="127.0.0.1", _port=port, _api_key="k", _model_name="m")


def _fast_retries(monkeypatch, total=0.2):
    monkeypatch.setattr(wake, "WAKE_TURN_TIMEOUT_SECONDS", total)
    monkeypatch.setattr(wake, "_RETRY_DELAYS_SECONDS", (0.01, 0.01, 0.01))


@pytest.mark.asyncio
async def test_response_timeout_does_not_repost_the_turn(monkeypatch):
    """The server accepts and runs the turn; the client times out waiting: one POST total."""
    _fast_retries(monkeypatch)
    posts = 0

    async def hang(request):
        nonlocal posts
        posts += 1
        await asyncio.Event().wait()  # the turn is still running server-side
        raise AssertionError("unreachable")

    app = web.Application()
    app.router.add_post("/v1/chat/completions", hang)
    server = TestServer(app)
    await server.start_server()
    try:
        await wake._self_post_chat_completion(
            _adapter(server.port), text="wake", session_id="sess-1")
    finally:
        await server.close()
    assert posts == 1


@pytest.mark.asyncio
async def test_concurrency_cap_429_still_retries(monkeypatch):
    """429 means the turn never started: retrying it is safe and must keep working."""
    _fast_retries(monkeypatch)
    posts = 0

    async def cap_then_ok(request):
        nonlocal posts
        posts += 1
        if posts == 1:
            return web.Response(status=429)
        return web.json_response({"ok": True})

    app = web.Application()
    app.router.add_post("/v1/chat/completions", cap_then_ok)
    server = TestServer(app)
    await server.start_server()
    try:
        await wake._self_post_chat_completion(
            _adapter(server.port), text="wake", session_id="sess-1")
    finally:
        await server.close()
    assert posts == 2


@pytest.mark.asyncio
async def test_connection_refused_still_raises_after_retries(monkeypatch):
    """Nothing started server-side: the caller must get the error so it can rewind/retry."""
    # Total stays above this host's loopback refusal latency (~2s): the refused
    # connect must surface as a connection error, not a total-timeout.
    _fast_retries(monkeypatch, total=15.0)
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        closed_port = probe.getsockname()[1]
    # Leaving the `with` block closes it: nothing listens, so connects refuse fast.
    with pytest.raises(RuntimeError, match="gave up"):
        await wake._self_post_chat_completion(
            _adapter(closed_port), text="wake", session_id="sess-1")

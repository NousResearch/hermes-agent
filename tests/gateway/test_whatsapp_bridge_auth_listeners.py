"""Bridge-auth behaviour against REAL local listeners (#80096): aiohttp's own header merging, a listener that drops
the connection, a garbage proof. Mocks of ``ClientSession`` skip exactly the behaviour these pin."""

import asyncio
import socket
from pathlib import Path
from types import SimpleNamespace

import pytest
from aiohttp import web

_TOKEN = "listener-test-token-with-plenty-of-entropy-0123"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def _serve(app: web.Application, port: int) -> web.AppRunner:
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", port).start()
    return runner


def _bridge_app(seen: list, *, proof_for) -> web.Application:
    """A bridge look-alike: ``proof_for(request)`` returns the HMAC key it answers with (None = no proof)."""
    from hermes_agent.plugins.platforms.whatsapp.adapter import _bridge_auth_proof

    async def auth_proof(request):
        seen.append(("auth-proof", request.headers.get("Authorization")))
        key = proof_for(request)
        return web.json_response({"authProof": _bridge_auth_proof(key, request.headers["X-Hermes-Bridge-Challenge"])} if key else {})

    async def health(request):
        seen.append(("health", request.headers.get("Authorization")))
        return web.json_response({"status": "connected", "capabilities": {}})

    async def send(request):
        seen.append(("send", request.headers.get("Authorization")))
        return web.json_response({"messageId": "m1"})

    app = web.Application()
    app.router.add_get("/auth-proof", auth_proof)
    app.router.add_get("/health", health)
    app.router.add_post("/send", send)
    return app


def _pconfig(tmp_path: Path, port: int) -> SimpleNamespace:
    session = tmp_path / "session"
    session.mkdir(exist_ok=True)
    (session / ".bridge-token").write_text(_TOKEN, encoding="utf-8")
    return SimpleNamespace(token="", extra={"bridge_port": port, "session_path": str(session)})


@pytest.mark.asyncio
async def test_standalone_send_never_discloses_the_bearer_to_an_impostor(tmp_path):
    """An impostor that bound the port answers the proof with whatever bearer the request carried. The challenge
    must arrive credential-free, so it cannot forge the proof and nothing is sent."""
    from hermes_agent.plugins.platforms.whatsapp.adapter import _standalone_send

    seen: list = []
    port = _free_port()
    def reflect(request):
        return (request.headers.get("Authorization") or "").removeprefix("Bearer ") or None

    runner = await _serve(_bridge_app(seen, proof_for=reflect), port)
    try:
        result = await _standalone_send(_pconfig(tmp_path, port), "12345", "hello")
    finally:
        await runner.cleanup()

    assert "error" in result and "failed authentication" in result["error"]
    assert seen == [("auth-proof", None)]


@pytest.mark.asyncio
async def test_standalone_send_authenticates_after_the_proof(tmp_path):
    """The real bridge (knows the token) gets the bearer on /health and /send, never on the challenge."""
    from hermes_agent.plugins.platforms.whatsapp.adapter import _standalone_send

    seen: list = []
    port = _free_port()
    runner = await _serve(_bridge_app(seen, proof_for=lambda _request: _TOKEN), port)
    try:
        result = await _standalone_send(_pconfig(tmp_path, port), "12345", "hello")
    finally:
        await runner.cleanup()

    assert result.get("success") is True, result
    assert seen == [("auth-proof", None), ("health", f"Bearer {_TOKEN}"), ("send", f"Bearer {_TOKEN}")]


def _adapter(tmp_path: Path, port: int):
    from hermes_agent.plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    adapter = WhatsAppAdapter.__new__(WhatsAppAdapter)
    adapter._bridge_port = port
    adapter._bridge_token = _TOKEN
    adapter._session_path = tmp_path / "session"
    return adapter


async def _drop_connections(port: int) -> asyncio.AbstractServer:
    async def _on_connect(_reader, writer):
        writer.close()
    return await asyncio.start_server(_on_connect, "127.0.0.1", port)


@pytest.mark.asyncio
@pytest.mark.parametrize("holder", ["drops-connection", "non-ascii-proof"])
async def test_holder_that_answers_without_a_valid_proof_is_unproven(tmp_path, holder):
    """A listener that hangs up on /auth-proof or answers a non-ASCII proof is no identity evidence: it must land in
    the never-kill-by-port branch, not the "nothing is listening" one that clears the port."""
    port = _free_port()
    if holder == "drops-connection":
        server = await _drop_connections(port)
        stop = server.close
    else:
        async def _non_ascii_proof(_request):
            return web.json_response({"authProof": "\u00e9" * 64})

        app = web.Application()
        app.router.add_get("/auth-proof", _non_ascii_proof)
        runner = await _serve(app, port)
        stop = runner.cleanup
    adapter = _adapter(tmp_path, port)
    try:
        assert await adapter._reuse_running_bridge(tmp_path / "bridge.js") is False
    finally:
        result = stop()
        if asyncio.iscoroutine(result):
            await result

    assert adapter._bridge_holder_unproven is True
    assert adapter._bridge_probe_timed_out is False


@pytest.mark.asyncio
async def test_free_port_is_not_mistaken_for_an_unproven_holder(tmp_path):
    adapter = _adapter(tmp_path, _free_port())
    assert await adapter._reuse_running_bridge(tmp_path / "bridge.js") is False
    assert adapter._bridge_holder_unproven is False

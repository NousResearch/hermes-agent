"""Authentication boundary for the local WhatsApp bridge HTTP API."""

from __future__ import annotations

import asyncio
import os
import stat
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.whatsapp.adapter import (
    WhatsAppAdapter,
    _bridge_auth_headers,
    _bridge_client_session,
    _bridge_ipc_endpoint,
    _file_content_hash,
    _read_bridge_token,
    _rotate_bridge_token,
    _standalone_send,
)


class _AsyncContext:
    def __init__(self, value):
        self.value = value

    async def __aenter__(self):
        return self.value

    async def __aexit__(self, *_exc):
        return False


def test_bridge_token_round_trip_is_private_and_rotates(tmp_path: Path) -> None:
    session_path = tmp_path / "session"
    session_path.mkdir()

    first = _rotate_bridge_token(session_path)
    token_path = session_path / ".bridge-token"

    assert _read_bridge_token(session_path) == first
    if os.name != "nt":
        assert stat.S_IMODE(token_path.stat().st_mode) == 0o600
    assert _bridge_auth_headers(first) == {"Authorization": f"Bearer {first}"}

    second = _rotate_bridge_token(session_path)
    assert second != first
    assert _read_bridge_token(session_path) == second
    if os.name != "nt":
        assert stat.S_IMODE(token_path.stat().st_mode) == 0o600


def test_bridge_token_reader_rejects_malformed_credentials(tmp_path: Path) -> None:
    session_path = tmp_path / "session"
    session_path.mkdir()
    token_path = session_path / ".bridge-token"

    for malformed in ("", "short", "token with spaces", "a" * 129):
        token_path.write_text(malformed, encoding="utf-8")
        assert _read_bridge_token(session_path) == ""


@pytest.mark.skipif(os.name == "nt", reason="POSIX file permissions only")
def test_bridge_token_reader_repairs_overbroad_mode(tmp_path: Path) -> None:
    session_path = tmp_path / "session"
    session_path.mkdir()
    token = _rotate_bridge_token(session_path)
    token_path = session_path / ".bridge-token"
    token_path.chmod(0o644)

    assert _read_bridge_token(session_path) == token
    assert stat.S_IMODE(token_path.stat().st_mode) == 0o600


@pytest.mark.skipif(os.name == "nt", reason="POSIX file permissions only")
def test_bridge_token_reader_fails_closed_when_mode_repair_fails(tmp_path: Path) -> None:
    session_path = tmp_path / "session"
    session_path.mkdir()
    _rotate_bridge_token(session_path)
    (session_path / ".bridge-token").chmod(0o644)

    with patch("plugins.platforms.whatsapp.adapter.os.fchmod", side_effect=OSError("denied")):
        assert _read_bridge_token(session_path) == ""


@pytest.mark.asyncio
@pytest.mark.skipif(os.name == "nt", reason="adversarial Unix-socket transport test")
async def test_standalone_send_ignores_fake_loopback_listener(tmp_path: Path) -> None:
    """A process owning the configured TCP port sees neither auth nor payload."""
    from aiohttp import web

    session_path = tmp_path / "session"
    session_path.mkdir()
    token = _rotate_bridge_token(session_path)
    fake_requests = []
    genuine_requests = []

    async def fake_handler(request):
        fake_requests.append((dict(request.headers), await request.read()))
        return web.json_response({"messageId": "forged"})

    fake_app = web.Application()
    fake_app.router.add_route("*", "/{tail:.*}", fake_handler)
    fake_runner = web.AppRunner(fake_app)
    await fake_runner.setup()
    fake_site = web.TCPSite(fake_runner, "127.0.0.1", 0)
    await fake_site.start()
    bridge_port = fake_site._server.sockets[0].getsockname()[1]

    async def genuine_send(request):
        genuine_requests.append((request.path, request.headers.get("Authorization"), await request.json()))
        return web.json_response({"messageId": "genuine"})

    genuine_app = web.Application()
    genuine_app.router.add_post("/send", genuine_send)
    genuine_app.router.add_post("/send-media", genuine_send)
    genuine_runner = web.AppRunner(genuine_app)
    await genuine_runner.setup()
    endpoint = _bridge_ipc_endpoint(session_path, bridge_port, token)
    genuine_site = web.UnixSite(genuine_runner, endpoint)
    await genuine_site.start()
    Path(endpoint).chmod(0o600)

    config = SimpleNamespace(
        token="",
        extra={"bridge_port": bridge_port, "session_path": str(session_path)},
    )
    media_path = tmp_path / "private.jpg"
    media_path.write_bytes(b"image")
    try:
        result = await _standalone_send(config, "15551234567", "private message")
        media_result = await _standalone_send(
            config,
            "15551234567",
            "",
            media_files=[(str(media_path), False)],
            caption="private caption",
        )
    finally:
        await genuine_runner.cleanup()
        await fake_runner.cleanup()

    assert result["success"] is True
    assert media_result["success"] is True
    assert result["message_id"] == "genuine"
    assert genuine_requests == [
        (
            "/send",
            f"Bearer {token}",
            {"chatId": "15551234567@s.whatsapp.net", "message": "private message"},
        ),
        (
            "/send-media",
            f"Bearer {token}",
            {
                "chatId": "15551234567@s.whatsapp.net",
                "filePath": str(media_path),
                "mediaType": "image",
                "caption": "private caption",
            },
        ),
    ]
    assert fake_requests == []


@pytest.mark.asyncio
@pytest.mark.skipif(os.name == "nt", reason="adversarial Unix-socket transport test")
async def test_standalone_send_rejects_untrusted_socket_before_http(tmp_path: Path) -> None:
    """An endpoint without the private socket invariant receives no request."""
    from aiohttp import web

    session_path = tmp_path / "session"
    session_path.mkdir()
    token = _rotate_bridge_token(session_path)
    bridge_port = 30124
    received = []

    async def fake_send(request):
        received.append((dict(request.headers), await request.read()))
        return web.json_response({"messageId": "forged"})

    app = web.Application()
    app.router.add_post("/send", fake_send)
    runner = web.AppRunner(app)
    await runner.setup()
    endpoint = _bridge_ipc_endpoint(session_path, bridge_port, token)
    site = web.UnixSite(runner, endpoint)
    await site.start()
    Path(endpoint).chmod(0o666)
    config = SimpleNamespace(
        token="",
        extra={"bridge_port": bridge_port, "session_path": str(session_path)},
    )
    try:
        result = await _standalone_send(config, "15551234567", "secret")
    finally:
        await runner.cleanup()

    assert "permissions are not private" in result["error"]
    assert received == []


@pytest.mark.asyncio
@pytest.mark.skipif(os.name == "nt", reason="adversarial Unix-socket transport test")
async def test_persistent_session_rejects_replacement_listener_on_reconnect(
    tmp_path: Path,
) -> None:
    """A live client revalidates OS identity before a replacement connection."""
    import aiohttp
    from aiohttp import web

    session_path = tmp_path / "session"
    session_path.mkdir()
    token = _rotate_bridge_token(session_path)
    bridge_port = 30125
    endpoint = _bridge_ipc_endpoint(session_path, bridge_port, token)

    async def genuine_health(_request):
        return web.json_response({"status": "connected"})

    genuine_app = web.Application()
    genuine_app.router.add_get("/health", genuine_health)
    genuine_runner = web.AppRunner(genuine_app)
    await genuine_runner.setup()
    genuine_site = web.UnixSite(genuine_runner, endpoint)
    await genuine_site.start()
    Path(endpoint).chmod(0o600)

    received = []

    async def replacement_send(request):
        received.append((dict(request.headers), await request.read()))
        return web.json_response({"messageId": "forged"})

    session = _bridge_client_session(aiohttp, session_path, bridge_port, token)
    replacement_runner = None
    try:
        async with session.get("http://localhost/health") as response:
            assert response.status == 200
        await genuine_runner.cleanup()
        Path(endpoint).unlink(missing_ok=True)

        replacement_app = web.Application()
        replacement_app.router.add_post("/send", replacement_send)
        replacement_runner = web.AppRunner(replacement_app)
        await replacement_runner.setup()
        replacement_site = web.UnixSite(replacement_runner, endpoint)
        await replacement_site.start()
        Path(endpoint).chmod(0o666)

        with pytest.raises(OSError, match="permissions are not private"):
            await session.post(
                "http://localhost/send",
                json={"message": "must remain private"},
            )
    finally:
        await session.close()
        if replacement_runner is not None:
            await replacement_runner.cleanup()
        else:
            await genuine_runner.cleanup()

    assert received == []


@pytest.mark.asyncio
@pytest.mark.skipif(os.name == "nt", reason="adversarial Unix-socket transport test")
async def test_fake_loopback_listener_cannot_forge_health_or_persistent_calls(
    tmp_path: Path,
) -> None:
    """Reuse and the persistent client stay on the authenticated OS channel."""
    from aiohttp import web

    bridge_dir = tmp_path / "bridge"
    bridge_dir.mkdir()
    bridge_script = bridge_dir / "bridge.js"
    bridge_script.write_text("// current bridge\n", encoding="utf-8")
    package_json = bridge_dir / "package.json"
    package_json.write_text('{"name":"bridge"}\n', encoding="utf-8")
    node_modules = bridge_dir / "node_modules"
    node_modules.mkdir()
    (node_modules / ".hermes-pkg-hash").write_text(
        _file_content_hash(package_json), encoding="utf-8"
    )
    session_path = tmp_path / "session"
    session_path.mkdir()
    (session_path / "creds.json").write_text("{}", encoding="utf-8")
    token = _rotate_bridge_token(session_path)
    fake_requests = []
    genuine_sends = []

    async def fake_handler(request):
        fake_requests.append((dict(request.headers), await request.read()))
        return web.json_response({
            "status": "connected",
            "scriptHash": _file_content_hash(bridge_script),
            "sendReadReceipts": False,
            "messageId": "forged",
        })

    fake_app = web.Application()
    fake_app.router.add_route("*", "/{tail:.*}", fake_handler)
    fake_runner = web.AppRunner(fake_app)
    await fake_runner.setup()
    fake_site = web.TCPSite(fake_runner, "127.0.0.1", 0)
    await fake_site.start()
    bridge_port = fake_site._server.sockets[0].getsockname()[1]

    async def genuine_health(_request):
        return web.json_response({
            "status": "connected",
            "scriptHash": _file_content_hash(bridge_script),
            "sendReadReceipts": False,
        })

    async def genuine_send(request):
        genuine_sends.append(await request.json())
        return web.json_response({"messageId": "genuine"})

    async def genuine_messages(_request):
        return web.json_response([])

    genuine_app = web.Application()
    genuine_app.router.add_get("/health", genuine_health)
    genuine_app.router.add_post("/send", genuine_send)
    genuine_app.router.add_get("/messages", genuine_messages)
    genuine_runner = web.AppRunner(genuine_app)
    await genuine_runner.setup()
    endpoint = _bridge_ipc_endpoint(session_path, bridge_port, token)
    genuine_site = web.UnixSite(genuine_runner, endpoint)
    await genuine_site.start()
    Path(endpoint).chmod(0o600)

    adapter = WhatsAppAdapter(PlatformConfig(enabled=True, extra={
        "bridge_script": str(bridge_script),
        "session_path": str(session_path),
        "bridge_port": bridge_port,
    }))

    def consume_poll_task(coroutine):
        coroutine.close()
        return MagicMock()

    try:
        with patch("plugins.platforms.whatsapp.adapter.check_whatsapp_requirements", return_value=True), \
             patch.object(adapter, "_acquire_platform_lock", return_value=True), \
             patch("plugins.platforms.whatsapp.adapter.asyncio.create_task", side_effect=consume_poll_task):
            connected = await adapter.connect()
        sent = await adapter.send("15551234567", "persistent private message")
        async with adapter._http_session.get("http://localhost/messages") as response:
            messages = await response.json()
    finally:
        if adapter._http_session and not adapter._http_session.closed:
            await adapter._http_session.close()
        await genuine_runner.cleanup()
        await fake_runner.cleanup()

    assert connected is True
    assert sent.success is True
    assert messages == []
    assert sent.message_id == "genuine"
    assert genuine_sends == [{
        "chatId": "15551234567@s.whatsapp.net",
        "message": "persistent private message",
    }]
    assert fake_requests == []


@pytest.mark.asyncio
@pytest.mark.skipif(os.name == "nt", reason="adversarial Unix-socket transport test")
async def test_standalone_send_revalidates_replaced_bridge_endpoint(tmp_path: Path) -> None:
    """Each standalone call reconnects through the owner-only socket path."""
    from aiohttp import web

    session_path = tmp_path / "session"
    session_path.mkdir()
    token = _rotate_bridge_token(session_path)
    bridge_port = 30123
    endpoint = _bridge_ipc_endpoint(session_path, bridge_port, token)
    config = SimpleNamespace(
        token="",
        extra={"bridge_port": bridge_port, "session_path": str(session_path)},
    )
    responders = []

    async def run_one(message_id):
        async def send_handler(request):
            responders.append((message_id, await request.json()))
            return web.json_response({"messageId": message_id})

        app = web.Application()
        app.router.add_post("/send", send_handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.UnixSite(runner, endpoint)
        await site.start()
        Path(endpoint).chmod(0o600)
        try:
            return await _standalone_send(config, "15551234567", message_id)
        finally:
            await runner.cleanup()
            Path(endpoint).unlink(missing_ok=True)

    first = await run_one("first-bridge")
    second = await run_one("replacement-bridge")

    assert first["message_id"] == "first-bridge"
    assert second["message_id"] == "replacement-bridge"
    assert [name for name, _payload in responders] == [
        "first-bridge",
        "replacement-bridge",
    ]


@pytest.mark.asyncio
async def test_matching_running_bridge_is_reused_with_bearer_token(tmp_path: Path) -> None:
    bridge_dir = tmp_path / "bridge"
    bridge_dir.mkdir()
    bridge_script = bridge_dir / "bridge.js"
    bridge_script.write_text("// current bridge\n", encoding="utf-8")
    package_json = bridge_dir / "package.json"
    package_json.write_text('{"name":"bridge"}\n', encoding="utf-8")
    node_modules = bridge_dir / "node_modules"
    node_modules.mkdir()
    (node_modules / ".hermes-pkg-hash").write_text(
        _file_content_hash(package_json), encoding="utf-8"
    )
    session_path = tmp_path / "session"
    session_path.mkdir()
    (session_path / "creds.json").write_text("{}", encoding="utf-8")
    token = _rotate_bridge_token(session_path)
    adapter = WhatsAppAdapter(PlatformConfig(enabled=True, extra={
        "bridge_script": str(bridge_script),
        "session_path": str(session_path),
        "bridge_port": 19876,
    }))

    health_response = MagicMock()
    health_response.status = 200
    health_response.json = AsyncMock(return_value={
        "status": "connected",
        "scriptHash": _file_content_hash(bridge_script),
        "sendReadReceipts": False,
    })
    transient_session = MagicMock()
    transient_session.get.return_value = _AsyncContext(health_response)
    persistent_session = MagicMock()
    session_factory = MagicMock(side_effect=[
        _AsyncContext(transient_session),
        persistent_session,
    ])

    def consume_poll_task(coroutine):
        coroutine.close()
        return MagicMock()

    connector = MagicMock()
    with patch("plugins.platforms.whatsapp.adapter.check_whatsapp_requirements", return_value=True), \
         patch("aiohttp.ClientSession", session_factory), \
         patch("plugins.platforms.whatsapp.adapter._bridge_unix_connector", return_value=connector), \
         patch("plugins.platforms.whatsapp.adapter._validate_bridge_endpoint"), \
         patch.object(adapter, "_acquire_platform_lock", return_value=True), \
         patch("plugins.platforms.whatsapp.adapter.asyncio.create_task", side_effect=consume_poll_task):
        connected = await adapter.connect()

    assert connected is True
    assert adapter._bridge_process is None
    assert adapter._bridge_token == token
    assert "headers" not in transient_session.get.call_args.kwargs
    assert session_factory.call_args_list[0].kwargs == {
        "connector": connector,
        "headers": {"Authorization": f"Bearer {token}"},
    }
    assert session_factory.call_args_list[1].kwargs["headers"] == {
        "Authorization": f"Bearer {token}"
    }


def test_standalone_sender_uses_session_bearer_token(tmp_path: Path) -> None:
    session_path = tmp_path / "session"
    session_path.mkdir()
    token = _rotate_bridge_token(session_path)
    config = SimpleNamespace(
        token="",
        extra={"bridge_port": 3000, "session_path": str(session_path)},
    )
    response = AsyncMock()
    response.status = 200
    response.json = AsyncMock(return_value={"messageId": "outbound-1"})
    response_context = MagicMock()
    response_context.__aenter__ = AsyncMock(return_value=response)
    response_context.__aexit__ = AsyncMock(return_value=False)
    session = MagicMock()
    session.post.return_value = response_context
    session_context = MagicMock()
    session_context.__aenter__ = AsyncMock(return_value=session)
    session_context.__aexit__ = AsyncMock(return_value=False)

    connector = MagicMock()
    with patch("aiohttp.ClientSession", return_value=session_context) as client_session, \
         patch("plugins.platforms.whatsapp.adapter._bridge_unix_connector", return_value=connector), \
         patch("plugins.platforms.whatsapp.adapter._validate_bridge_endpoint"):
        result = asyncio.run(_standalone_send(config, "15551234567", "hello"))

    assert result["success"] is True
    client_session.assert_called_once_with(
        connector=connector,
        headers={"Authorization": f"Bearer {token}"}
    )


def test_standalone_sender_fails_closed_without_session_token(tmp_path: Path) -> None:
    config = SimpleNamespace(
        token="",
        extra={"bridge_port": 3000, "session_path": str(tmp_path / "missing")},
    )

    with patch("aiohttp.ClientSession") as client_session:
        result = asyncio.run(_standalone_send(config, "15551234567", "hello"))

    assert "authentication is unavailable" in result["error"]
    client_session.assert_not_called()

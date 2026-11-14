"""Hermes thread routing against stock discord.py and a real loopback HTTP server."""

import asyncio
from types import MethodType, SimpleNamespace

import pytest


discord = pytest.importorskip("discord")
from aiohttp import web
from discord.http import HTTPClient, Route, handle_message_parameters

from gateway.config import PlatformConfig
from plugins.platforms.discord.adapter import DiscordAdapter
from plugins.platforms.discord import adapter_auto_thread


@pytest.mark.asyncio
@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("outcome", ["known", "unknown", "sdk_route_wait", "sdk_global_wait"])
async def test_native_create_uses_stock_policy_and_surfaces_only_terminal_errors(
    monkeypatch, tmp_path, fallback, outcome,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(adapter_auto_thread, "time", SimpleNamespace(
        time=lambda: 1700000000.25, monotonic=lambda: 10,
    ))
    requests = []
    notices = []
    creates = []
    successful = outcome.startswith("sdk_")

    def json_response(data, **kwargs):
        response = web.json_response(data, **kwargs)
        # discord.py compares this header exactly; Discord omits aiohttp's charset.
        response.headers["Content-Type"] = "application/json"
        return response

    async def handle(request):
        requests.append((request.method, request.path))
        if request.path == "/users/@me":
            return json_response({"id": "999", "username": "test", "discriminator": "0", "avatar": None})
        if request.method == "DELETE":
            return web.Response(status=204)
        if request.path.endswith("/threads"):
            creates.append(request.path)
            if fallback and request.path.endswith("/123/threads"):
                return json_response({"message": "Cannot create from this message", "code": 50035}, status=400)
            relevant = [path for path in creates if not fallback or "/456/" in path]
            if successful and len(relevant) > 1:
                return json_response({
                    "id": "789", "parent_id": "800", "owner_id": "999", "name": "hello", "type": 11,
                    "message_count": 0, "member_count": 1,
                    "thread_metadata": {"archived": False, "auto_archive_duration": 1440,
                                        "archive_timestamp": "2026-01-01T00:00:00+00:00", "locked": False},
                })
            # No Via on terminal errors: stock discord.py surfaces these rather than retrying.
            headers = {"Retry-After": "60.01"} if outcome == "known" else {}
            if successful:
                headers["Via"] = "loopback"
            return json_response({"message": "Rate limited", "retry_after": 0.01,
                                      "global": outcome == "sdk_global_wait"}, status=429, headers=headers)
        data = await request.json()
        if data["content"].startswith("⚠️") or "could not confirm" in data["content"]:
            notices.append(data["content"])
        return json_response({"id": "456"})

    app = web.Application()
    app.router.add_route("*", "/{path:.*}", handle)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    monkeypatch.setattr(Route, "BASE", f"http://127.0.0.1:{port}")
    http = HTTPClient(asyncio.get_running_loop())
    guild = SimpleNamespace(id=100)
    state = SimpleNamespace(http=http)
    channel = SimpleNamespace(id=800)

    def message(message_id):
        msg = SimpleNamespace(id=message_id, content="hello", author=None,
                              channel=channel, guild=guild, _state=state)
        # Public SDK methods enforce stock signatures and use its real HTTP transport.
        msg.create_thread = MethodType(discord.Message.create_thread, msg)
        msg.delete = MethodType(discord.Message.delete, msg)
        return msg

    async def send(content):
        with handle_message_parameters(content=content) as params:
            data = await http.send_message(channel.id, params=params)
        return message(int(data["id"]))

    channel.send = send
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test", extra={}))
    try:
        await http.static_login("loopback-only")
        result = await adapter._auto_create_thread_or_notify(message(123))
        assert http.max_ratelimit_timeout is None
        assert http._global_over.is_set()
        if successful:
            assert isinstance(result, discord.Thread)
            assert result.id == 789
            assert len(creates) == (3 if fallback else 2)
            assert not notices
        else:
            assert result is None
            assert len(creates) == (2 if fallback else 1)
            assert len(notices) == 1
            assert "not processed" not in notices[0]
            if outcome == "known":
                assert "<t:1700000061:R>" in notices[0]
                assert "<t:1700000061:F>" in notices[0]
            else:
                assert "Try again later" in notices[0]
                assert "<t:" not in notices[0]
            assert ("DELETE", "/channels/800/messages/456") in requests if fallback else True
    finally:
        await http.close()
        await runner.cleanup()

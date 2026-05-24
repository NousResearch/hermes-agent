from types import SimpleNamespace

import pytest

pytest.importorskip("httpx")

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.platforms.base import EphemeralReply, MessageEvent
from gateway.session import SessionSource


def _make_event(text="/xsearch status", platform=Platform.DISCORD):
    source = SessionSource(
        platform=platform,
        user_id="user-1",
        chat_id="chat-1",
        user_name="tester",
    )
    return MessageEvent(text=text, source=source)


def _make_runner():
    return object.__new__(gateway_run.GatewayRunner)


@pytest.mark.asyncio
async def test_handle_xsearch_command_uses_platform_key(monkeypatch):
    captured = {}

    def _fake_run(command, *, platform="cli"):
        captured["command"] = command
        captured["platform"] = platform
        return SimpleNamespace(output="ok", reset_session=False)

    monkeypatch.setattr("hermes_cli.xsearch_command.run_xsearch_command", _fake_run)

    result = await _make_runner()._handle_xsearch_command(_make_event())

    assert result == "ok"
    assert captured == {"command": "/xsearch status", "platform": "discord"}


@pytest.mark.asyncio
async def test_handle_xsearch_command_resets_session_when_requested(monkeypatch):
    def _fake_run(command, *, platform="cli"):
        assert command == "/xsearch enable"
        assert platform == "discord"
        return SimpleNamespace(output="enabled", reset_session=True)

    async def _fake_reset(_event):
        return EphemeralReply("fresh session", ttl_seconds=45)

    runner = _make_runner()
    runner._handle_reset_command = _fake_reset

    monkeypatch.setattr("hermes_cli.xsearch_command.run_xsearch_command", _fake_run)

    result = await runner._handle_xsearch_command(_make_event("/xsearch enable"))

    assert isinstance(result, EphemeralReply)
    assert "enabled" in result
    assert "fresh session" in result
    assert result.ttl_seconds == 45

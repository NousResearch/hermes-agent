"""Tests for gateway /ooo slash wrapper delegating to the native router."""

from __future__ import annotations

import sys
from typing import Any

import pytest

from gateway.config import Platform
from gateway.ouroboros_native import OooNativeResponse
from gateway.ouroboros_state import OooStateStore
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource
from gateway.slash_commands import GatewaySlashCommandsMixin


def _discord_thread_event(text: str = "/ooo interview hello") -> MessageEvent:
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="thread-1",
        chat_type="thread",
        user_id="user-1",
        user_id_alt="alt-user-1",
        user_name="User One",
        guild_id="guild-1",
        parent_chat_id="channel-1",
        thread_id="thread-1",
        message_id="source-msg-1",
        profile="default",
    )
    return MessageEvent(
        text=text,
        message_type=MessageType.COMMAND,
        source=source,
        message_id="event-msg-1",
        platform_update_id=12345,
    )


def _install_native_fake(monkeypatch: pytest.MonkeyPatch, *, text: str = "native ok") -> list[tuple[str, Any]]:
    calls: list[tuple[str, Any]] = []

    async def fake_handle(raw: str, ctx: Any) -> OooNativeResponse:
        calls.append((raw, ctx))
        return OooNativeResponse(text=text, payload={"ok": True})

    monkeypatch.setattr("gateway.ouroboros_native.handle_ooo_native", fake_handle)
    return calls


@pytest.mark.asyncio
async def test_handle_ooo_delegates_to_native_and_returns_response_text(monkeypatch: pytest.MonkeyPatch):
    calls = _install_native_fake(monkeypatch, text="native delegated")
    runner = GatewaySlashCommandsMixin()

    response = await runner._handle_ooo_command(_discord_thread_event())

    assert response == "native delegated"
    assert len(calls) == 1
    assert calls[0][0] == "interview hello"


@pytest.mark.asyncio
async def test_handle_ooo_builds_native_context_from_discord_thread_source(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("gateway.slash_commands.os.getcwd", lambda: "/gateway-cwd")
    calls = _install_native_fake(monkeypatch)
    runner = GatewaySlashCommandsMixin()

    await runner._handle_ooo_command(_discord_thread_event())

    _raw, ctx = calls[0]
    assert ctx.cwd == "/gateway-cwd"
    assert isinstance(ctx.state_store, OooStateStore)
    assert ctx.allow_mutating_side_effects is False
    assert ctx.idempotency_key == "source-msg-1"
    assert ctx.state_context.platform == "discord"
    assert ctx.state_context.guild_id == "guild-1"
    assert ctx.state_context.channel_id == "channel-1"
    assert ctx.state_context.thread_id == "thread-1"
    assert ctx.state_context.user_id == "alt-user-1"
    assert ctx.state_context.profile == "default"


@pytest.mark.asyncio
async def test_handle_ooo_uses_deterministic_idempotency_fallback(monkeypatch: pytest.MonkeyPatch):
    calls = _install_native_fake(monkeypatch)
    runner = GatewaySlashCommandsMixin()
    event = _discord_thread_event("/ooo run --seed-path seed.yaml")
    event.source.message_id = None
    event.message_id = None
    event.platform_update_id = None

    first = await runner._handle_ooo_command(event, session_key="session-key-1")
    second = await runner._handle_ooo_command(event, session_key="session-key-1")

    assert first == "native ok"
    assert second == "native ok"
    first_key = calls[0][1].idempotency_key
    second_key = calls[1][1].idempotency_key
    assert first_key == second_key
    assert first_key.startswith("ooo:")


@pytest.mark.asyncio
async def test_handle_ooo_prefers_raw_message_id_before_hash_fallback(monkeypatch: pytest.MonkeyPatch):
    calls = _install_native_fake(monkeypatch)
    runner = GatewaySlashCommandsMixin()
    event = _discord_thread_event("/ooo run --seed-path seed.yaml")
    event.source.message_id = None
    event.message_id = None
    event.platform_update_id = None
    event.raw_message = type("RawInteraction", (), {"id": "raw-interaction-789"})()

    response = await runner._handle_ooo_command(event, session_key="session-key-1")

    assert response == "native ok"
    assert calls[0][1].idempotency_key == "raw-interaction-789"


@pytest.mark.asyncio
async def test_handle_ooo_run_does_not_use_cli_fast_path(monkeypatch: pytest.MonkeyPatch):
    calls = _install_native_fake(monkeypatch, text="native run")

    async def fail_cli(self, subcommand: str, args_text: str) -> str:  # pragma: no cover - should not run
        raise AssertionError("legacy CLI fast-path should not be called")

    monkeypatch.setattr(GatewaySlashCommandsMixin, "_run_ooo_cli_command", fail_cli, raising=False)
    runner = GatewaySlashCommandsMixin()

    response = await runner._handle_ooo_command(_discord_thread_event("/ooo run --seed-path seed.yaml"))

    assert response == "native run"
    assert calls[0][0] == "run --seed-path seed.yaml"


@pytest.mark.asyncio
async def test_handle_ooo_does_not_rewrite_for_skill_fallthrough(monkeypatch: pytest.MonkeyPatch):
    calls = _install_native_fake(monkeypatch, text="native no fallthrough")
    monkeypatch.setitem(sys.modules, "agent.skill_commands", None)
    runner = GatewaySlashCommandsMixin()
    event = _discord_thread_event("/ooo interview original request")

    response = await runner._handle_ooo_command(event, session_key="sess-1", allow_agent_fallthrough=True)

    assert response == "native no fallthrough"
    assert calls[0][0] == "interview original request"
    assert event.text == "/ooo interview original request"


@pytest.mark.asyncio
async def test_handle_ooo_native_exception_returns_safe_error(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    async def fail_native(raw: str, ctx: Any) -> OooNativeResponse:
        raise RuntimeError("boom\nTraceback: secret stack")

    monkeypatch.setattr("gateway.ouroboros_native.handle_ooo_native", fail_native)
    caplog.set_level("WARNING", logger="gateway.run")
    runner = GatewaySlashCommandsMixin()

    response = await runner._handle_ooo_command(_discord_thread_event())

    assert isinstance(response, str)
    assert response
    assert "Traceback" not in response
    assert "secret stack" not in response
    assert "Ouroboros" in response or "/ooo" in response
    assert any(
        record.exc_info and "Native /ooo router failed" in record.getMessage()
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_handle_ooo_running_agent_mode_still_uses_native(monkeypatch: pytest.MonkeyPatch):
    calls = _install_native_fake(monkeypatch, text="native while running")
    runner = GatewaySlashCommandsMixin()

    response = await runner._handle_ooo_command(
        _discord_thread_event("/ooo interview still route"),
        allow_agent_fallthrough=False,
    )

    assert response == "native while running"
    assert calls[0][0] == "interview still route"

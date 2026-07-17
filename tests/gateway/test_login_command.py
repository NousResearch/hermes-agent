"""Channel-safe Codex device-code login command tests."""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource
from hermes_cli.auth import CodexDeviceCodePrompt


def _event(text: str = "/login codex", *, chat_type: str = "dm", user_id: str = "owner"):
    return MessageEvent(
        text=text,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="chat-1",
            user_id=user_id,
            user_name="tester",
            chat_type=chat_type,
        ),
    )


def _runner(*, extra=None):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={
            Platform.TELEGRAM: PlatformConfig(
                enabled=True,
                token="***",
                extra=extra or {},
            )
        }
    )
    adapter = SimpleNamespace(
        send=AsyncMock(return_value=SimpleNamespace(success=True))
    )
    runner.adapters = {Platform.TELEGRAM: adapter}
    return runner, adapter


def test_login_is_registered_for_gateway_and_telegram_menu():
    from hermes_cli.commands import (
        GATEWAY_KNOWN_COMMANDS,
        telegram_bot_commands,
    )

    assert "login" in GATEWAY_KNOWN_COMMANDS
    assert ("login", "Pair Codex with a private device code") in telegram_bot_commands()


@pytest.mark.asyncio
async def test_login_codex_sends_code_before_completion(monkeypatch):
    runner, adapter = _runner()

    def fake_login(on_verification):
        on_verification(
            CodexDeviceCodePrompt(
                verification_url="https://auth.openai.com/codex/device",
                user_code="ABCD-EFGH",
                expires_in_seconds=900,
            )
        )

    monkeypatch.setattr(
        "hermes_cli.auth.login_openai_codex_device",
        fake_login,
    )

    result = await runner._handle_login_command(_event())

    assert result == "Codex login complete. Try your request again now."
    sent = adapter.send.await_args.args[1]
    assert "https://auth.openai.com/codex/device" in sent
    assert "ABCD-EFGH" in sent
    assert "Never share it" in sent


@pytest.mark.asyncio
async def test_login_codex_rejects_group_without_starting_auth(monkeypatch):
    runner, adapter = _runner()
    login = AsyncMock()
    monkeypatch.setattr("hermes_cli.auth.login_openai_codex_device", login)

    result = await runner._handle_login_command(_event(chat_type="group"))

    assert "only sent in a private conversation" in result
    login.assert_not_awaited()
    adapter.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_login_codex_is_admin_only_when_command_roles_are_configured(monkeypatch):
    runner, adapter = _runner(extra={"allow_admin_from": ["owner"]})
    login = AsyncMock()
    monkeypatch.setattr("hermes_cli.auth.login_openai_codex_device", login)

    result = await runner._handle_login_command(_event(user_id="member"))

    assert result == "⛔ /login is admin-only because it replaces Hermes credentials."
    login.assert_not_awaited()
    adapter.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_login_codex_dedupes_active_conversation_flow(monkeypatch):
    runner, _adapter = _runner()
    started = threading.Event()
    release = threading.Event()

    def blocking_login(on_verification):
        on_verification(
            CodexDeviceCodePrompt(
                verification_url="https://auth.openai.com/codex/device",
                user_code="FIRST-CODE",
                expires_in_seconds=900,
            )
        )
        started.set()
        release.wait(timeout=5)

    monkeypatch.setattr(
        "hermes_cli.auth.login_openai_codex_device",
        blocking_login,
    )

    first = asyncio.create_task(runner._handle_login_command(_event()))
    await asyncio.to_thread(started.wait, 5)
    second = await runner._handle_login_command(_event())
    release.set()

    assert "already active" in second
    assert await first == "Codex login complete. Try your request again now."

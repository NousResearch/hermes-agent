"""Channel-safe, profile-scoped Codex gateway login tests."""

import asyncio
import json
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource
from hermes_cli.auth import CodexDeviceCodePrompt


def _event(
    text: str = "/login codex",
    *,
    chat_type: str = "dm",
    user_id: str = "owner",
    profile: str | None = None,
):
    return MessageEvent(
        text=text,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="chat-1",
            user_id=user_id,
            user_name="tester",
            chat_type=chat_type,
            profile=profile,
        ),
    )


def _runner(*, extra=None, multiplex_profiles: bool = False):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    platform_config = PlatformConfig(
        enabled=True,
        token="***",
        extra=extra or {},
    )
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: platform_config},
        multiplex_profiles=multiplex_profiles,
    )
    adapter = SimpleNamespace(
        send=AsyncMock(return_value=SimpleNamespace(success=True)),
        config=platform_config,
    )
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._adapter_for_source = lambda _source: adapter
    runner._reply_anchor_for_event = lambda _event: None
    runner._thread_metadata_for_source = lambda _source, _anchor=None: {}
    return runner, adapter


def _prompt(code: str = "PRIVATE-CODE"):
    return CodexDeviceCodePrompt(
        verification_url="https://auth.openai.com/codex/device",
        user_code=code,
        expires_in_seconds=900,
    )


def test_login_is_registered_with_explicit_busy_rejection():
    from hermes_cli.commands import (
        GATEWAY_KNOWN_COMMANDS,
        resolve_command,
        telegram_bot_commands,
    )

    command = resolve_command("login")

    assert command is not None
    assert command.gateway_only is True
    assert command.busy_policy == "reject"
    assert "login" in GATEWAY_KNOWN_COMMANDS
    assert ("login", "Pair Codex with a private device code") in telegram_bot_commands()


@pytest.mark.asyncio
async def test_login_rejects_while_agent_is_busy():
    from hermes_cli.commands import resolve_command

    runner, _adapter = _runner()
    command = resolve_command("login")

    result = await runner._dispatch_busy_slash_command(
        _event(),
        command,
        "agent:main:telegram:dm:chat-1",
        _event().source,
    )

    assert result == (
        "⏳ Agent is running — `/login` can't run mid-turn. "
        "Wait for the current response or `/stop` first."
    )


@pytest.mark.asyncio
async def test_login_codex_sends_code_before_auth_continues(monkeypatch):
    runner, adapter = _runner()
    events = []

    async def send(_chat_id, _message, **_kwargs):
        events.append("send")
        return SimpleNamespace(success=True)

    adapter.send = AsyncMock(side_effect=send)

    def fake_login(on_verification):
        on_verification(_prompt())
        events.append("poll")

    monkeypatch.setattr(
        "hermes_cli.auth.login_openai_codex_to_pool",
        fake_login,
    )

    result = await runner._handle_login_command(_event())

    assert events == ["send", "poll"]
    assert result == "Codex login complete. The account was added to this profile."
    sent = adapter.send.await_args.args[1]
    assert "https://auth.openai.com/codex/device" in sent
    assert "PRIVATE-CODE" in sent
    assert "PRIVATE-CODE" not in result


@pytest.mark.asyncio
async def test_login_codex_rejects_group_without_starting_auth(monkeypatch):
    runner, adapter = _runner()
    login = MagicMock()
    monkeypatch.setattr("hermes_cli.auth.login_openai_codex_to_pool", login)

    result = await runner._handle_login_command(_event(chat_type="group"))

    assert "only sent in a private conversation" in result
    login.assert_not_called()
    adapter.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_login_codex_is_admin_only_when_command_roles_are_configured(
    monkeypatch,
):
    runner, adapter = _runner(extra={"allow_admin_from": ["owner"]})
    login = MagicMock()
    monkeypatch.setattr("hermes_cli.auth.login_openai_codex_to_pool", login)

    result = await runner._handle_login_command(_event(user_id="member"))

    assert result == "⛔ /login is admin-only because it adds Hermes credentials."
    login.assert_not_called()
    adapter.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_login_codex_dedupes_active_conversation_flow(monkeypatch):
    runner, _adapter = _runner()
    started = threading.Event()
    release = threading.Event()

    def blocking_login(on_verification):
        on_verification(_prompt("FIRST-CODE"))
        started.set()
        release.wait(timeout=5)

    monkeypatch.setattr(
        "hermes_cli.auth.login_openai_codex_to_pool",
        blocking_login,
    )

    first = asyncio.create_task(runner._handle_login_command(_event()))
    assert await asyncio.to_thread(started.wait, 5)
    second = await runner._handle_login_command(_event())
    release.set()

    assert "already active" in second
    assert await first == "Codex login complete. The account was added to this profile."


@pytest.mark.asyncio
async def test_login_codex_handler_cancellation_retains_dedupe_until_worker_exits(
    monkeypatch,
):
    runner, _adapter = _runner()
    started = threading.Event()
    release = threading.Event()
    exited = threading.Event()

    def blocking_login(on_verification):
        on_verification(_prompt("CANCEL-CODE"))
        started.set()
        release.wait(timeout=5)
        exited.set()

    monkeypatch.setattr(
        "hermes_cli.auth.login_openai_codex_to_pool",
        blocking_login,
    )

    handler = asyncio.create_task(runner._handle_login_command(_event()))
    assert await asyncio.to_thread(started.wait, 5)
    flow_key = next(iter(runner._active_codex_login_flows))
    worker = runner._active_codex_login_workers[flow_key]

    handler.cancel()
    with pytest.raises(asyncio.CancelledError):
        await handler

    assert flow_key in runner._active_codex_login_flows
    assert runner._active_codex_login_workers[flow_key] is worker
    duplicate = await runner._handle_login_command(_event())
    assert "already active" in duplicate

    release.set()
    assert await asyncio.to_thread(exited.wait, 5)
    assert await worker is True
    assert runner._active_codex_login_flows == set()
    assert runner._active_codex_login_workers == {}


@pytest.mark.asyncio
async def test_login_codex_failure_clears_dedupe_without_disclosing_exception(
    monkeypatch, caplog
):
    runner, _adapter = _runner()
    attempts = 0

    def flaky_login(on_verification):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("PRIVATE-TOKEN")
        on_verification(_prompt("SECOND-CODE"))

    monkeypatch.setattr(
        "hermes_cli.auth.login_openai_codex_to_pool",
        flaky_login,
    )

    first = await runner._handle_login_command(_event())
    second = await runner._handle_login_command(_event())

    assert first == "Codex login did not complete. Send `/login codex` to request a new code."
    assert "PRIVATE-TOKEN" not in first
    assert "PRIVATE-TOKEN" not in caplog.text
    assert runner._active_codex_login_flows == set()
    assert runner._active_codex_login_workers == {}
    assert second == "Codex login complete. The account was added to this profile."


@pytest.mark.asyncio
async def test_login_codex_adds_distinct_accounts_to_routed_profile(
    tmp_path, monkeypatch
):
    default_home = tmp_path / "default"
    routed_home = tmp_path / "profiles" / "coder"
    default_home.mkdir(parents=True)
    routed_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(default_home))

    existing = {
        "id": "existing",
        "label": "existing@example.com",
        "auth_type": "oauth",
        "priority": 0,
        "source": "manual:device_code",
        "access_token": "existing-access",
        "refresh_token": "existing-refresh",
        "base_url": "https://chatgpt.com/backend-api/codex",
    }
    (routed_home / "auth.json").write_text(
        json.dumps(
            {
                "version": 1,
                "active_provider": "anthropic",
                "providers": {},
                "credential_pool": {"openai-codex": [existing]},
            }
        )
    )

    runner, adapter = _runner(multiplex_profiles=True)

    def resolve_profile_home(source):
        assert source.profile == "coder"
        return routed_home

    runner._resolve_profile_home_for_source = resolve_profile_home
    logins = iter(
        [
            {
                "code": "FIRST-CODE",
                "access": "first-access",
                "refresh": "first-refresh",
            },
            {
                "code": "SECOND-CODE",
                "access": "second-access",
                "refresh": "second-refresh",
            },
        ]
    )

    def fake_device_login(on_verification=None):
        login = next(logins)
        assert on_verification is not None
        on_verification(_prompt(login["code"]))
        return {
            "tokens": {
                "access_token": login["access"],
                "refresh_token": login["refresh"],
            },
            "base_url": "https://chatgpt.com/backend-api/codex",
            "last_refresh": "2026-07-30T00:00:00Z",
        }

    monkeypatch.setattr(
        "hermes_cli.auth._codex_device_code_login",
        fake_device_login,
    )

    first_result = await runner._handle_login_command(_event(profile="coder"))
    second_result = await runner._handle_login_command(_event(profile="coder"))

    payload = json.loads((routed_home / "auth.json").read_text())
    entries = payload["credential_pool"]["openai-codex"]
    assert [entry["access_token"] for entry in entries] == [
        "existing-access",
        "first-access",
        "second-access",
    ]
    assert [entry["refresh_token"] for entry in entries] == [
        "existing-refresh",
        "first-refresh",
        "second-refresh",
    ]
    assert payload["active_provider"] == "anthropic"
    assert not (default_home / "auth.json").exists()

    returned = first_result + second_result
    sent = " ".join(call.args[1] for call in adapter.send.await_args_list)
    for secret in (
        "first-access",
        "first-refresh",
        "second-access",
        "second-refresh",
    ):
        assert secret not in returned
        assert secret not in sent
    assert "FIRST-CODE" not in returned
    assert "SECOND-CODE" not in returned
    assert "FIRST-CODE" in sent
    assert "SECOND-CODE" in sent

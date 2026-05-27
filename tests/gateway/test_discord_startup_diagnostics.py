"""Secret-safe Discord startup diagnostics."""

from __future__ import annotations

import types

import pytest

from gateway.config import PlatformConfig
from tests.gateway._plugin_adapter_loader import load_plugin_adapter


class _FakeOpus:
    @staticmethod
    def is_loaded():
        return True


class _FakeIntents:
    message_content = False
    dm_messages = False
    guild_messages = False
    members = False
    voice_states = False

    @classmethod
    def default(cls):
        return cls()


@pytest.mark.asyncio
async def test_discord_missing_token_sets_nonretryable_fatal_error(monkeypatch):
    discord_mod = load_plugin_adapter("discord")

    monkeypatch.setattr(discord_mod, "DISCORD_AVAILABLE", True)
    monkeypatch.setattr(
        discord_mod,
        "discord",
        types.SimpleNamespace(opus=_FakeOpus()),
    )

    adapter = discord_mod.DiscordAdapter(PlatformConfig(enabled=True, token=None))

    result = await adapter.connect()

    assert result is False
    assert adapter.has_fatal_error is True
    assert adapter.fatal_error_retryable is False
    assert adapter.fatal_error_code == "discord_missing_token"
    assert adapter.fatal_error_message is not None
    assert "DISCORD_BOT_TOKEN" in adapter.fatal_error_message


@pytest.mark.asyncio
async def test_discord_login_failure_sets_redacted_nonretryable_fatal_error(monkeypatch):
    discord_mod = load_plugin_adapter("discord")

    class FakeLoginFailure(Exception):
        pass

    class FakeBot:
        user = object()

        def __init__(self, *args, **kwargs):
            self.tree = types.SimpleNamespace()

        def event(self, func):
            return func

        async def start(self, token):
            assert token == "super-secret-token"
            raise FakeLoginFailure("Improper token has been passed: super-secret-token")

        def is_closed(self):
            return True

    monkeypatch.setattr(discord_mod, "DISCORD_AVAILABLE", True)
    monkeypatch.setattr(
        discord_mod,
        "discord",
        types.SimpleNamespace(
            opus=_FakeOpus(),
            Intents=_FakeIntents,
            LoginFailure=FakeLoginFailure,
            PrivilegedIntentsRequired=type("PrivilegedIntentsRequired", (Exception,), {}),
            MessageType=types.SimpleNamespace(default=0, reply=1),
            DMChannel=type("DMChannel", (), {}),
        ),
    )
    monkeypatch.setattr(discord_mod, "commands", types.SimpleNamespace(Bot=FakeBot))
    monkeypatch.setattr(discord_mod, "_build_allowed_mentions", lambda: None)

    adapter = discord_mod.DiscordAdapter(
        PlatformConfig(enabled=True, token="super-secret-token", extra={"slash_commands": False})
    )
    monkeypatch.setattr(adapter, "_acquire_platform_lock", lambda *args, **kwargs: True)
    released = {"called": False}
    monkeypatch.setattr(adapter, "_release_platform_lock", lambda: released.__setitem__("called", True))

    result = await adapter.connect()

    assert result is False
    assert adapter.has_fatal_error is True
    assert adapter.fatal_error_retryable is False
    assert adapter.fatal_error_code == "discord_auth_failed"
    assert adapter.fatal_error_message is not None
    assert "DISCORD_BOT_TOKEN" in adapter.fatal_error_message
    assert "super-secret-token" not in adapter.fatal_error_message
    assert released["called"] is True

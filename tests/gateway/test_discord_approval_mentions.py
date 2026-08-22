"""Discord blocking prompts can opt into owner mentions."""

import os
from types import SimpleNamespace

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.discord import adapter as discord_adapter
from plugins.platforms.discord.adapter import DiscordAdapter, _apply_yaml_config


class _FakeObject:
    def __init__(self, *, id):
        self.id = id


class _FakeAllowedMentions:
    def __init__(self, *, users, roles, everyone, replied_user):
        self.users = users
        self.roles = roles
        self.everyone = everyone
        self.replied_user = replied_user


class _FakeChannel:
    def __init__(self):
        self.sent_kwargs = None

    async def send(self, **kwargs):
        self.sent_kwargs = kwargs
        return SimpleNamespace(id=12345)


class _FakeClient:
    def __init__(self, channel):
        self.channel = channel

    def get_channel(self, channel_id):
        return self.channel


@pytest.fixture(autouse=True)
def _stable_discord_types(monkeypatch):
    monkeypatch.setattr(discord_adapter.discord, "Object", _FakeObject)
    monkeypatch.setattr(discord_adapter.discord, "AllowedMentions", _FakeAllowedMentions)


def _make_adapter(*, extra=None, allowed_user_ids=None):
    channel = _FakeChannel()
    adapter = object.__new__(DiscordAdapter)
    adapter._client = _FakeClient(channel)
    adapter._allowed_user_ids = allowed_user_ids or {"222", "111", "alice"}
    adapter._allowed_role_ids = set()
    adapter.config = PlatformConfig(enabled=True, extra=extra or {})
    return adapter, channel


def _assert_owner_ping(channel):
    assert channel.sent_kwargs is not None
    sent = channel.sent_kwargs
    assert sent["content"].startswith("<@111> <@222>\n")
    assert "<@alice>" not in sent["content"]
    allowed_mentions = sent["allowed_mentions"]
    assert [user.id for user in allowed_mentions.users] == [111, 222]
    assert allowed_mentions.roles is False
    assert allowed_mentions.everyone is False
    assert allowed_mentions.replied_user is False
    assert len(sent["content"]) <= 2000
    return sent


@pytest.mark.asyncio
async def test_exec_approval_mentions_allowed_users_when_enabled(monkeypatch):
    monkeypatch.setenv("DISCORD_APPROVAL_MENTIONS", "true")
    adapter, channel = _make_adapter()

    result = await adapter.send_exec_approval(
        chat_id="99",
        command="make check",
        session_key="session-1",
        description="dangerous command",
    )

    assert result.success is True
    sent = _assert_owner_ping(channel)
    assert "make check" in sent["content"]
    assert sent["embed"].title.endswith("Command Approval Required")


@pytest.mark.asyncio
async def test_slash_confirmation_mentions_allowed_users_when_enabled(monkeypatch):
    monkeypatch.setenv("DISCORD_APPROVAL_MENTIONS", "true")
    adapter, channel = _make_adapter()

    result = await adapter.send_slash_confirm(
        chat_id="99",
        title="Confirm reset",
        message="Reset session?",
        session_key="session-1",
        confirm_id="confirm-1",
    )

    assert result.success is True
    sent = _assert_owner_ping(channel)
    assert "Reset session?" in sent["content"]


@pytest.mark.asyncio
@pytest.mark.parametrize("choices", [None, ["staging", "production"]])
async def test_clarify_mentions_allowed_users_when_enabled(monkeypatch, choices):
    monkeypatch.setenv("DISCORD_APPROVAL_MENTIONS", "true")
    adapter, channel = _make_adapter()

    result = await adapter.send_clarify(
        chat_id="99",
        question="Which environment? <@999>",
        choices=choices,
        clarify_id="clarify-1",
        session_key="session-1",
    )

    assert result.success is True
    sent = _assert_owner_ping(channel)
    assert "Which environment?" in sent["content"]
    assert 999 not in [user.id for user in sent["allowed_mentions"].users]
    assert ("view" in sent) is bool(choices)


@pytest.mark.asyncio
async def test_update_prompt_mentions_allowed_users_when_enabled(monkeypatch):
    monkeypatch.setenv("DISCORD_APPROVAL_MENTIONS", "true")
    adapter, channel = _make_adapter()

    result = await adapter.send_update_prompt(
        chat_id="99",
        prompt="Restore stashed changes?",
        session_key="session-1",
    )

    assert result.success is True
    sent = _assert_owner_ping(channel)
    assert "Restore stashed changes?" in sent["content"]


@pytest.mark.asyncio
async def test_clarify_does_not_mention_when_disabled(monkeypatch):
    monkeypatch.delenv("DISCORD_APPROVAL_MENTIONS", raising=False)
    adapter, channel = _make_adapter(extra={"approval_mentions": "false"})

    result = await adapter.send_clarify(
        chat_id="99",
        question="Question?",
        choices=None,
        clarify_id="clarify-1",
        session_key="session-1",
    )

    assert result.success is True
    assert channel.sent_kwargs is not None
    assert not channel.sent_kwargs["content"].startswith("<@")
    assert "allowed_mentions" not in channel.sent_kwargs


@pytest.mark.asyncio
async def test_multiplex_profile_extra_false_ignores_process_env(monkeypatch):
    monkeypatch.setenv("DISCORD_APPROVAL_MENTIONS", "true")
    monkeypatch.setattr(discord_adapter, "_multiplex_active", lambda: True)
    adapter, channel = _make_adapter(extra={"approval_mentions": "false"})

    result = await adapter.send_clarify(
        chat_id="99",
        question="Question?",
        choices=None,
        clarify_id="clarify-1",
        session_key="session-1",
    )

    assert result.success is True
    assert channel.sent_kwargs is not None
    assert not channel.sent_kwargs["content"].startswith("<@")
    assert "allowed_mentions" not in channel.sent_kwargs


def test_single_profile_env_override_wins(monkeypatch):
    monkeypatch.setenv("DISCORD_APPROVAL_MENTIONS", "true")
    monkeypatch.setattr(discord_adapter, "_multiplex_active", lambda: False)
    adapter, _ = _make_adapter(extra={"approval_mentions": "false"})

    assert adapter._approval_mention_content() == "<@111> <@222>"


@pytest.mark.asyncio
async def test_large_allowlist_keeps_prompt_within_discord_limit(monkeypatch):
    monkeypatch.setenv("DISCORD_APPROVAL_MENTIONS", "true")
    allowed_user_ids = {str(100_000_000_000_000_000 + index) for index in range(120)}
    adapter, channel = _make_adapter(allowed_user_ids=allowed_user_ids)

    result = await adapter.send_clarify(
        chat_id="99",
        question="Q" * 4_000,
        choices=None,
        clarify_id="clarify-1",
        session_key="session-1",
    )

    assert result.success is True
    assert channel.sent_kwargs is not None
    content = channel.sent_kwargs["content"]
    assert content.startswith("<@")
    assert len(content) <= adapter.MAX_MESSAGE_LENGTH
    assert content.count("<@") < len(allowed_user_ids)
    assert len(channel.sent_kwargs["allowed_mentions"].users) == content.count("<@")


def test_yaml_mapping_exposes_approval_mentions(monkeypatch):
    monkeypatch.delenv("DISCORD_APPROVAL_MENTIONS", raising=False)

    seeded = _apply_yaml_config({}, {"approval_mentions": True})

    assert os.environ["DISCORD_APPROVAL_MENTIONS"] == "true"
    assert seeded == {"approval_mentions": "true"}


def test_multiplex_yaml_mapping_stays_profile_scoped(monkeypatch):
    monkeypatch.delenv("DISCORD_APPROVAL_MENTIONS", raising=False)
    monkeypatch.setattr(discord_adapter, "_profile_scoped_config_load", lambda: True)

    seeded = _apply_yaml_config({}, {"approval_mentions": True})
    adapter, _ = _make_adapter(extra=seeded)

    assert seeded == {"approval_mentions": "true"}
    assert "DISCORD_APPROVAL_MENTIONS" not in os.environ
    assert adapter._approval_mention_content() == "<@111> <@222>"


def test_yaml_config_seeds_websocket_health_with_primary_precedence(monkeypatch):
    for key in (
        "HERMES_DISCORD_LIVENESS_INTERVAL_SECONDS",
        "HERMES_DISCORD_LIVENESS_FAILURE_THRESHOLD",
    ):
        monkeypatch.delenv(key, raising=False)

    seeded = _apply_yaml_config(
        {},
        {
            "websocket_liveness_interval_seconds": 11,
            "liveness_interval_seconds": 99,
            "websocket_liveness_failure_threshold": 2,
            "websocket_heartbeat_ack_max_age_seconds": 75,
            "websocket_max_latency_seconds": 30,
        },
    )

    assert os.environ["HERMES_DISCORD_LIVENESS_INTERVAL_SECONDS"] == "11"
    assert os.environ["HERMES_DISCORD_LIVENESS_FAILURE_THRESHOLD"] == "2"
    assert seeded == {
        "websocket_liveness_interval_seconds": 11,
        "websocket_liveness_failure_threshold": 2,
        "websocket_heartbeat_ack_max_age_seconds": 75,
        "websocket_max_latency_seconds": 30,
    }

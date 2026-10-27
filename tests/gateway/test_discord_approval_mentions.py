"""Discord approval prompts can opt into owner mentions.

Also covers the sibling opt-in mention knobs for the other interactive Discord
prompts (clarify, slash confirm, update prompt) — each gated behind its own
``discord.*_mentions`` config key / ``DISCORD_*_MENTIONS`` env var, default off.
"""

import os
from types import SimpleNamespace

import pytest

from plugins.platforms.discord.adapter import (
    DiscordAdapter,
    _apply_yaml_config,
)


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


@pytest.mark.asyncio
async def test_exec_approval_mentions_allowed_users_when_enabled(monkeypatch):
    monkeypatch.setenv("DISCORD_APPROVAL_MENTIONS", "true")
    channel = _FakeChannel()
    adapter = object.__new__(DiscordAdapter)
    adapter._client = _FakeClient(channel)
    adapter._allowed_user_ids = {"222", "111", "alice"}
    adapter._allowed_role_ids = set()
    adapter.config = SimpleNamespace(extra=None)

    result = await adapter.send_exec_approval(
        chat_id="99",
        command="make check",
        session_key="session-1",
        description="dangerous command",
    )

    assert result.success is True
    # Mentions are prepended to the (always present) content mirror.
    assert channel.sent_kwargs["content"].startswith("<@111> <@222>\n")
    assert "make check" in channel.sent_kwargs["content"]
    assert "allowed_mentions" in channel.sent_kwargs


def _mention_adapter(monkeypatch, env_key: str | None) -> DiscordAdapter:
    if env_key is None:
        monkeypatch.delenv("DISCORD_CLARIFY_MENTIONS", raising=False)
        monkeypatch.delenv("DISCORD_SLASH_CONFIRM_MENTIONS", raising=False)
        monkeypatch.delenv("DISCORD_UPDATE_PROMPT_MENTIONS", raising=False)
    else:
        monkeypatch.setenv(env_key, "true")
    channel = _FakeChannel()
    adapter = object.__new__(DiscordAdapter)
    adapter._client = _FakeClient(channel)
    adapter._allowed_user_ids = {"222", "111", "alice"}
    adapter._allowed_role_ids = set()
    adapter.config = SimpleNamespace(extra=None)
    return adapter


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("send", "env_key"),
    [
        (lambda a: a.send_clarify(chat_id="99", question="Which one?", choices=["A", "B"],
                                  clarify_id="c1", session_key="session-1"),
         "DISCORD_CLARIFY_MENTIONS"),
        (lambda a: a.send_slash_confirm(chat_id="99", title="Reload MCP", message="Proceed?",
                                        session_key="session-1", confirm_id="s1"),
         "DISCORD_SLASH_CONFIRM_MENTIONS"),
        (lambda a: a.send_update_prompt(chat_id="99", prompt="Update available.", default="1.2.3",
                                        session_key="session-1"),
         "DISCORD_UPDATE_PROMPT_MENTIONS"),
    ],
)
async def test_prompt_mentions_allowed_users_when_enabled(monkeypatch, send, env_key):
    adapter = _mention_adapter(monkeypatch, env_key)
    result = await send(adapter)

    assert result.success is True
    sent = adapter._client.channel.sent_kwargs
    # Mentions are prepended to the (always present) content mirror.
    assert sent["content"].startswith("<@111> <@222>\n")
    assert "allowed_mentions" in sent


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("send",),
    [
        (lambda a: a.send_clarify(chat_id="99", question="Which one?", choices=["A", "B"],
                                  clarify_id="c1", session_key="session-1"),),
        (lambda a: a.send_slash_confirm(chat_id="99", title="Reload MCP", message="Proceed?",
                                        session_key="session-1", confirm_id="s1"),),
        (lambda a: a.send_update_prompt(chat_id="99", prompt="Update available.", default="1.2.3",
                                        session_key="session-1"),),
    ],
)
async def test_prompt_mentions_default_off(monkeypatch, send):
    adapter = _mention_adapter(monkeypatch, None)
    result = await send(adapter)

    assert result.success is True
    sent = adapter._client.channel.sent_kwargs
    assert not sent["content"].startswith("<@")
    assert "allowed_mentions" not in sent


def test_yaml_config_seeds_prompt_mention_keys(monkeypatch):
    for key in (
        "DISCORD_APPROVAL_MENTIONS",
        "DISCORD_CLARIFY_MENTIONS",
        "DISCORD_SLASH_CONFIRM_MENTIONS",
        "DISCORD_UPDATE_PROMPT_MENTIONS",
    ):
        monkeypatch.delenv(key, raising=False)

    seeded = _apply_yaml_config(
        {},
        {
            "approval_mentions": True,
            "clarify_mentions": True,
            "slash_confirm_mentions": True,
            "update_prompt_mentions": True,
        },
    )

    for key in (
        "DISCORD_APPROVAL_MENTIONS",
        "DISCORD_CLARIFY_MENTIONS",
        "DISCORD_SLASH_CONFIRM_MENTIONS",
        "DISCORD_UPDATE_PROMPT_MENTIONS",
    ):
        assert os.environ[key] == "true"
    assert seeded["approval_mentions"] is True
    assert seeded["clarify_mentions"] is True
    assert seeded["slash_confirm_mentions"] is True
    assert seeded["update_prompt_mentions"] is True


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



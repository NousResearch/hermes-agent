"""Regression coverage for Discord bot-ingress YAML configuration."""

from types import SimpleNamespace
from typing import Any, cast

import plugins.platforms.discord.adapter as discord_adapter
from plugins.platforms.discord.adapter import DiscordAdapter, _apply_yaml_config


def test_yaml_allow_bots_reaches_adapter_ingress_policy(monkeypatch):
    """discord.allow_bots must configure the live adapter, not only validate."""
    monkeypatch.delenv("DISCORD_ALLOW_BOTS", raising=False)
    monkeypatch.setattr(discord_adapter, "_profile_scoped_config_load", lambda: True)

    seeded = _apply_yaml_config({}, {"allow_bots": "mentions"})

    assert seeded is not None
    assert seeded["allow_bots"] == "mentions"
    adapter = object.__new__(DiscordAdapter)
    adapter._gate_env_snapshot = {"DISCORD_ALLOW_BOTS": ""}
    adapter.config = cast(Any, SimpleNamespace(extra=seeded))
    assert adapter._get_allow_bots() == "mentions"


def test_env_allow_bots_overrides_yaml_policy(monkeypatch):
    """Legacy env configuration must retain precedence over profile YAML."""
    monkeypatch.setenv("DISCORD_ALLOW_BOTS", "all")
    monkeypatch.setattr(discord_adapter, "_profile_scoped_config_load", lambda: False)

    seeded = _apply_yaml_config({}, {"allow_bots": "mentions"})

    assert seeded is not None
    assert seeded["allow_bots"] == "mentions"
    assert discord_adapter.os.environ["DISCORD_ALLOW_BOTS"] == "all"
    adapter = object.__new__(DiscordAdapter)
    adapter._gate_env_snapshot = {"DISCORD_ALLOW_BOTS": "all"}
    adapter.config = cast(Any, SimpleNamespace(extra=seeded))
    assert adapter._get_allow_bots() == "all"

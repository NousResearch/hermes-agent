"""Config contract tests for Discord Native Multi-Bot Protocol v2.

Slice 0A is schema/validation only. These tests intentionally assert that v2
stays default-off and does not reuse legacy single-token Discord credentials.
"""

from __future__ import annotations

import importlib
import json

import pytest

from gateway.config import (
    DiscordNativeMultibotConfig,
    GatewayConfig,
    Platform,
)


DISCORD_ENV_VARS = (
    "DISCORD_BOT_TOKEN",
    "DISCORD_HOME_CHANNEL",
    "DISCORD_REPLY_TO_MODE",
)


@pytest.fixture(autouse=True)
def _clean_discord_env(monkeypatch):
    for name in DISCORD_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def _identity(**overrides):
    data = {
        "agent_id": "bohumil",
        "hermes_profile": "default",
        "discord_application_id": "111111111111111111",
        "discord_bot_user_id": "222222222222222222",
        "token_secret_ref": "secret://hermes/discord/bohumil-token",
        "capabilities": ["intake", "reply", "approve_projection"],
        "allowed_scopes": {"guild_ids": ["333333333333333333"]},
    }
    data.update(overrides)
    return data


def _v2_config(**overrides):
    data = {
        "enabled": True,
        "mode": "active",
        "guild_allowlist": ["333333333333333333"],
        "default_intake_agent_id": "bohumil",
        "identities": [_identity()],
    }
    data.update(overrides)
    return data


def test_absent_config_defaults_to_off_and_no_identities():
    config = GatewayConfig.from_dict({})

    assert config.discord_native_multibot.enabled is False
    assert config.discord_native_multibot.mode == "off"
    assert config.discord_native_multibot.guild_allowlist == []
    assert config.discord_native_multibot.identities == []


def test_legacy_discord_env_still_enables_legacy_adapter_only(monkeypatch):
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "legacy-single-bot-token")

    gateway_config = importlib.import_module("gateway.config")
    config = gateway_config.load_gateway_config()

    assert config.discord_native_multibot.enabled is False
    assert config.discord_native_multibot.mode == "off"
    assert config.discord_native_multibot.identities == []
    assert config.platforms[Platform.DISCORD].enabled is True
    assert config.platforms[Platform.DISCORD].token == "legacy-single-bot-token"


def test_legacy_platform_discord_token_stays_legacy_only():
    config = GatewayConfig.from_dict(
        {
            "platforms": {
                "discord": {"enabled": True, "token": "legacy-config-token"}
            }
        }
    )

    assert config.discord_native_multibot.enabled is False
    assert config.discord_native_multibot.identities == []
    assert config.platforms[Platform.DISCORD].enabled is True
    assert config.platforms[Platform.DISCORD].token == "legacy-config-token"


@pytest.mark.parametrize("mode", ["off", "shadow", "listen_only", "active"])
def test_exact_modes_are_accepted(mode):
    data = _v2_config(mode=mode)
    if mode == "off":
        data["enabled"] = False
        data["guild_allowlist"] = []
        data["identities"] = []

    parsed = DiscordNativeMultibotConfig.from_dict(data)

    assert parsed.mode == mode


@pytest.mark.parametrize("mode", ["", "listen-only", "diagnostic", "ACTIVE_NOW", "webhook"])
def test_invalid_modes_are_rejected(mode):
    with pytest.raises(ValueError, match="mode"):
        DiscordNativeMultibotConfig.from_dict(_v2_config(mode=mode))


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"enabled": False}, "enabled=true"),
        ({"guild_allowlist": []}, "guild_allowlist"),
        ({"default_intake_agent_id": ""}, "default_intake_agent_id"),
        ({"default_intake_agent_id": "missing-agent"}, "default intake identity"),
        ({"identities": [_identity(enabled=False)]}, "default intake identity"),
    ],
)
def test_active_mode_requires_enabled_allowlist_and_default_intake_identity(
    overrides, message
):
    with pytest.raises(ValueError, match=message):
        DiscordNativeMultibotConfig.from_dict(_v2_config(**overrides))


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"enabled": False}, "enabled=false"),
        ({"guild_allowlist": []}, "guild_allowlist"),
    ],
)
def test_listen_only_requires_enabled_and_allowlist(overrides, message):
    data = _v2_config(mode="listen_only", identities=[], **overrides)

    with pytest.raises(ValueError, match=message):
        DiscordNativeMultibotConfig.from_dict(data)


def test_listen_only_accepts_allowlist_without_identity_tokens():
    parsed = DiscordNativeMultibotConfig.from_dict(
        {
            "enabled": True,
            "mode": "listen_only",
            "guild_allowlist": ["333333333333333333"],
            "identities": [],
        }
    )

    assert parsed.mode == "listen_only"
    assert parsed.identities == []


def test_legacy_discord_bot_token_does_not_satisfy_v2_identity_requirement(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "legacy-single-bot-token")
    (tmp_path / "config.yaml").write_text(
        """
discord_native_multibot:
  enabled: true
  mode: active
  guild_allowlist: ["333333333333333333"]
  default_intake_agent_id: bohumil
  identities: []
""".lstrip(),
        encoding="utf-8",
    )

    gateway_config = importlib.import_module("gateway.config")
    with pytest.raises(ValueError, match="default intake identity"):
        gateway_config.load_gateway_config()


def test_secret_ref_identity_is_accepted_and_round_trips():
    parsed = DiscordNativeMultibotConfig.from_dict(_v2_config())

    identity = parsed.identities[0]
    assert identity.agent_id == "bohumil"
    assert identity.token_secret_ref == "secret://hermes/discord/bohumil-token"
    assert identity.allowed_scopes.guild_ids == ["333333333333333333"]
    assert parsed.to_dict()["identities"][0]["token_secret_ref"] == (
        "secret://hermes/discord/bohumil-token"
    )


@pytest.mark.parametrize(
    "bad_key",
    ["token", "bot_token", "discord_token", "DISCORD_BOT_TOKEN", "discord_bot_token"],
)
def test_legacy_or_plaintext_token_keys_are_rejected(bad_key):
    bad_identity = _identity()
    bad_identity[bad_key] = "plaintext-token-value"

    with pytest.raises(ValueError, match="forbidden credential key"):
        DiscordNativeMultibotConfig.from_dict(_v2_config(identities=[bad_identity]))


@pytest.mark.parametrize(
    "bad_ref",
    [
        "plaintext-token-value",
        "Bot plaintext-token-value",
        "secret://",
        "secret://   ",
        "env://DISCORD_BOT_TOKEN",
        "DISCORD_BOT_TOKEN",
    ],
)
def test_raw_or_legacy_token_secret_refs_are_rejected(bad_ref):
    with pytest.raises(ValueError, match="token_secret_ref"):
        DiscordNativeMultibotConfig.from_dict(
            _v2_config(identities=[_identity(token_secret_ref=bad_ref)])
        )


def test_serialization_and_redacted_snapshots_do_not_include_plaintext_or_resolver_output():
    plaintext = "resolved-discord-token-plaintext"
    data = _v2_config(identities=[_identity(resolver_output=plaintext)])

    config = GatewayConfig.from_dict({"discord_native_multibot": data})
    serialized = json.dumps(config.to_dict(), sort_keys=True)
    redacted = json.dumps(config.redacted_snapshot(), sort_keys=True)

    assert plaintext not in serialized
    assert plaintext not in redacted
    assert "secret://<redacted>" in redacted
    assert "resolved-discord-token-plaintext" not in redacted

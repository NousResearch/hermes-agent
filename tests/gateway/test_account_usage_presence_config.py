from pathlib import Path

from gateway.config import AccountUsagePresenceConfig, GatewayConfig, load_gateway_config
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


def test_account_usage_presence_defaults_off_and_unconfigured():
    config = AccountUsagePresenceConfig()

    assert config.enabled is False
    assert config.provider is None
    assert config.platforms == ()
    assert config.update_interval_seconds == 300
    assert config.stale_after_seconds == 900
    assert config.is_configured is False


def test_account_usage_presence_config_roundtrip_normalizes_values():
    parsed = AccountUsagePresenceConfig.from_dict(
        {
            "enabled": "true",
            "provider": " Anthropic ",
            "platforms": ["Telegram", "discord", "telegram", ""],
            "update_interval_seconds": 10,
            "stale_after_seconds": 20,
            "window_label": " Seven day ",
        }
    )

    assert parsed.enabled is True
    assert parsed.provider == "anthropic"
    assert parsed.platforms == ("telegram", "discord")
    assert parsed.update_interval_seconds == 300
    assert parsed.stale_after_seconds == 300
    assert parsed.window_label == "Seven day"
    assert parsed.is_configured is True
    assert AccountUsagePresenceConfig.from_dict(parsed.to_dict()) == parsed


def test_enabled_config_requires_explicit_provider_and_platforms():
    assert AccountUsagePresenceConfig.from_dict({"enabled": True}).is_configured is False
    assert (
        AccountUsagePresenceConfig.from_dict(
            {"enabled": True, "provider": "openai-codex"}
        ).is_configured
        is False
    )
    assert (
        AccountUsagePresenceConfig.from_dict(
            {"enabled": True, "platforms": ["telegram"]}
        ).is_configured
        is False
    )


def test_ambiguous_provider_aliases_are_not_explicit_accounts():
    for provider in ("auto", "main"):
        parsed = AccountUsagePresenceConfig.from_dict(
            {
                "enabled": True,
                "provider": provider,
                "platforms": ["telegram"],
            }
        )
        assert parsed.provider is None
        assert parsed.is_configured is False


def test_unsupported_provider_and_platform_values_are_ignored():
    parsed = AccountUsagePresenceConfig.from_dict(
        {
            "enabled": True,
            "provider": "not-a-real-provider",
            "platforms": ["telegram", "matrix", "discord"],
        }
    )

    assert parsed.provider is None
    assert parsed.platforms == ("telegram", "discord")
    assert parsed.is_configured is False


def test_gateway_config_roundtrip_includes_account_usage_presence():
    original = GatewayConfig(
        account_usage_presence=AccountUsagePresenceConfig.from_dict(
            {
                "enabled": True,
                "provider": "openrouter",
                "platforms": ["discord"],
            }
        )
    )

    restored = GatewayConfig.from_dict(original.to_dict())

    assert restored.account_usage_presence == original.account_usage_presence


def test_load_gateway_config_reads_nested_gateway_account_usage_presence(tmp_path: Path):
    token = set_hermes_home_override(tmp_path)
    try:
        (tmp_path / "config.yaml").write_text(
            """
gateway:
  account_usage_presence:
    enabled: true
    provider: openai-codex
    platforms: [telegram, discord]
    update_interval_seconds: 600
    stale_after_seconds: 1800
    window_label: Session
""".lstrip(),
            encoding="utf-8",
        )

        loaded = load_gateway_config()
    finally:
        reset_hermes_home_override(token)

    assert loaded.account_usage_presence.enabled is True
    assert loaded.account_usage_presence.provider == "openai-codex"
    assert loaded.account_usage_presence.platforms == ("telegram", "discord")
    assert loaded.account_usage_presence.update_interval_seconds == 600
    assert loaded.account_usage_presence.stale_after_seconds == 1800
    assert loaded.account_usage_presence.window_label == "Session"

"""Tests for Discord config and policy helpers."""

from gateway.config import PlatformConfig
from gateway.platforms.discord_impl import config as discord_config


def test_clean_discord_id_strips_common_prefixes():
    assert discord_config.clean_discord_id(" user:123 ") == "123"
    assert discord_config.clean_discord_id("<@123>") == "123"
    assert discord_config.clean_discord_id("<@!123>") == "123"
    assert discord_config.clean_discord_id("teknium") == "teknium"


def test_parse_allowed_users_cleans_and_filters_entries():
    parsed = discord_config.parse_allowed_users(" 123, <@!456>, user:teknium, , <@789> ")

    assert parsed == {"123", "456", "teknium", "789"}


def test_parse_free_response_channels_accepts_lists_and_strings():
    parsed = discord_config.parse_free_response_channels(["123", " 456 ", 789, ""])

    assert parsed == {"123", "456", "789"}


def test_get_bot_filter_policy_defaults_to_none(monkeypatch):
    monkeypatch.delenv("DISCORD_ALLOW_BOTS", raising=False)

    assert discord_config.get_bot_filter_policy() == "none"


def test_get_bot_filter_policy_normalizes_case_and_whitespace(monkeypatch):
    monkeypatch.setenv("DISCORD_ALLOW_BOTS", " Mentions ")

    assert discord_config.get_bot_filter_policy() == "mentions"


def test_get_free_response_channels_parses_ids(monkeypatch):
    monkeypatch.setenv("DISCORD_FREE_RESPONSE_CHANNELS", " 123,456 , ,789 ")

    assert discord_config.get_free_response_channels() == {"123", "456", "789"}


def test_is_mention_required_defaults_true(monkeypatch):
    monkeypatch.delenv("DISCORD_REQUIRE_MENTION", raising=False)

    assert discord_config.is_mention_required() is True


def test_is_mention_required_accepts_falsey_env_values(monkeypatch):
    for value in ("false", "0", "no"):
        monkeypatch.setenv("DISCORD_REQUIRE_MENTION", value)
        assert discord_config.is_mention_required() is False


def test_is_auto_thread_enabled_defaults_true(monkeypatch):
    monkeypatch.delenv("DISCORD_AUTO_THREAD", raising=False)

    assert discord_config.is_auto_thread_enabled() is True


def test_is_auto_thread_enabled_accepts_truthy_and_falsey_values(monkeypatch):
    for value in ("true", "1", "yes"):
        monkeypatch.setenv("DISCORD_AUTO_THREAD", value)
        assert discord_config.is_auto_thread_enabled() is True

    monkeypatch.setenv("DISCORD_AUTO_THREAD", "false")
    assert discord_config.is_auto_thread_enabled() is False


def test_load_policy_config_reads_env_defaults(monkeypatch):
    monkeypatch.setenv("DISCORD_ALLOWED_USERS", "123,<@!456>")
    monkeypatch.setenv("DISCORD_ALLOW_BOTS", "mentions")
    monkeypatch.setenv("DISCORD_FREE_RESPONSE_CHANNELS", "1,2")
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")
    monkeypatch.setenv("DISCORD_AUTO_THREAD", "false")

    policy = discord_config.load_policy_config()

    assert policy.allowed_users == {"123", "456"}
    assert policy.bot_filter_policy == "mentions"
    assert policy.free_response_channels == {"1", "2"}
    assert policy.require_mention is False
    assert policy.auto_thread is False


def test_load_policy_config_prefers_platform_extra_over_env(monkeypatch):
    monkeypatch.setenv("DISCORD_ALLOWED_USERS", "111")
    monkeypatch.setenv("DISCORD_ALLOW_BOTS", "none")
    monkeypatch.setenv("DISCORD_FREE_RESPONSE_CHANNELS", "10")
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "true")
    monkeypatch.setenv("DISCORD_AUTO_THREAD", "true")

    config = PlatformConfig(
        enabled=True,
        extra={
            "allowed_users": ["<@!222>", "user:teknium"],
            "allow_bots": "all",
            "free_response_channels": ["20", "30"],
            "require_mention": False,
            "auto_thread": False,
        },
    )

    policy = discord_config.load_policy_config(config)

    assert policy.allowed_users == {"222", "teknium"}
    assert policy.bot_filter_policy == "all"
    assert policy.free_response_channels == {"20", "30"}
    assert policy.require_mention is False
    assert policy.auto_thread is False


def test_load_policy_config_applies_runtime_overrides_over_platform_extra(monkeypatch):
    config = PlatformConfig(
        enabled=True,
        extra={
            "allow_bots": "none",
            "free_response_channels": ["20"],
            "require_mention": True,
            "auto_thread": True,
        },
    )

    policy = discord_config.load_policy_config(
        config,
        overrides={
            "allow_bots": "mentions",
            "free_response_channels": ["44", "55"],
            "require_mention": False,
            "auto_thread": False,
        },
    )

    assert policy.bot_filter_policy == "mentions"
    assert policy.free_response_channels == {"44", "55"}
    assert policy.require_mention is False
    assert policy.auto_thread is False

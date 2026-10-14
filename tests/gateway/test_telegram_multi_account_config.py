"""Multi-account Telegram configuration parsing — #8287.

One gateway, N Telegram bot accounts: tokens arrive as
``TELEGRAM_BOT_TOKEN_<ACCOUNT>`` env vars (secrets stay in .env), behavioral
settings as ``platforms.telegram.accounts.<name>`` in config.yaml. The
unsuffixed ``TELEGRAM_BOT_TOKEN`` remains the default account, so single-bot
configurations parse byte-identically to before.
"""

import pytest

from gateway.config import Platform, PlatformConfig, load_gateway_config


def test_single_bot_config_has_no_accounts_key(monkeypatch, tmp_path):
    """Backward compatibility: an unsuffixed token must not grow an
    accounts block — existing single-bot setups stay byte-identical."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:default-token")

    config = load_gateway_config()
    tg = config.platforms[Platform.TELEGRAM]
    assert tg.token == "123:default-token"
    assert "accounts" not in tg.extra


def test_suffixed_env_tokens_declare_accounts(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:default-token")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN_SUPPORT", "456:support-token")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN_SALES", "789:sales-token")

    config = load_gateway_config()
    tg = config.platforms[Platform.TELEGRAM]

    # Default account untouched.
    assert tg.token == "123:default-token"
    # Suffix names are lowercased account names carrying only the credential.
    accounts = tg.extra["accounts"]
    assert accounts["support"]["token"] == "456:support-token"
    assert accounts["sales"]["token"] == "789:sales-token"


def test_suffixed_token_alone_enables_platform(monkeypatch, tmp_path):
    """A gateway configured with only account tokens (no default) still
    enables Telegram — the registry decides which accounts to start."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN_SUPPORT", "456:support-token")

    config = load_gateway_config()
    tg = config.platforms[Platform.TELEGRAM]
    assert tg.enabled
    assert tg.token is None
    assert tg.extra["accounts"]["support"]["token"] == "456:support-token"


def test_empty_suffix_or_value_is_ignored(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:default-token")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN_", "999:no-name")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN_EMPTY", "")

    config = load_gateway_config()
    tg = config.platforms[Platform.TELEGRAM]
    assert "accounts" not in tg.extra


def test_yaml_accounts_block_parses_and_normalizes_names():
    cfg = PlatformConfig.from_dict({
        "enabled": True,
        "accounts": {
            "Support": {"display_name": "Support Bot", "allowed_users": [1, 2]},
            "  SALES ": {"home_channel": {"chat_id": "-100123"}},
        },
    })
    accounts = cfg.extra["accounts"]
    assert set(accounts) == {"support", "sales"}
    assert accounts["support"]["display_name"] == "Support Bot"
    assert accounts["support"]["allowed_users"] == [1, 2]


def test_yaml_accounts_survive_via_extra_bridge():
    """The shared-key loop can bridge accounts into extra — both routes
    normalize identically (the gateway_restart_notification pattern)."""
    cfg = PlatformConfig.from_dict({
        "enabled": True,
        "extra": {"accounts": {"Support": {"display_name": "S"}}},
    })
    assert cfg.extra["accounts"]["support"]["display_name"] == "S"


def test_env_token_merges_into_yaml_account_block(monkeypatch, tmp_path):
    """config.yaml declares the behavioral block; .env supplies the token.
    The two merge on the same account name."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "gateway:\n"
        "  platforms:\n"
        "    telegram:\n"
        "      enabled: true\n"
        "      accounts:\n"
        "        support:\n"
        "          display_name: Support Bot\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN_SUPPORT", "456:support-token")

    config = load_gateway_config()
    tg = config.platforms[Platform.TELEGRAM]
    support = tg.extra["accounts"]["support"]
    assert support["token"] == "456:support-token"
    assert support.get("display_name") == "Support Bot"


def test_accounts_round_trip_through_to_dict():
    cfg = PlatformConfig.from_dict({
        "enabled": True,
        "accounts": {"support": {"display_name": "S"}},
    })
    rebuilt = PlatformConfig.from_dict(cfg.to_dict())
    assert rebuilt.extra["accounts"]["support"]["display_name"] == "S"

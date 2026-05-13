import os


_BANKR_ENV_KEYS = (
    "BANKR_AGENT_WALLET_ADDRESS",
    "BANKR_AGENT_API_KEY",
    "BANKR_AGENT_WALLET_ID",
    "BANKR_AGENT_WITHDRAWAL_DESTINATION",
    "BANKR_API_KEY",
    "BANKR_WALLET_ADDRESS",
)


def _clear_bankr_env(monkeypatch):
    for key in _BANKR_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def _clear_config_caches(config_module):
    config_module._LOAD_CONFIG_CACHE.clear()
    config_module._RAW_CONFIG_CACHE.clear()
    config_module._LAST_EXPANDED_CONFIG_BY_PATH.clear()


def test_load_config_exports_bankr_wallet_environment_aliases(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        """
bankr:
  walletAddress: "0x000000000000000000000000000000000000ba5e"
  apiKey: "bk_agent_cleartext_secret"
  walletId: "wlt_123"
  withdrawalDestination: "0x000000000000000000000000000000000000feed"
""",
        encoding="utf-8",
    )

    _clear_bankr_env(monkeypatch)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    from hermes_cli import config as config_module

    _clear_config_caches(config_module)
    config_module.load_config()

    assert os.environ["BANKR_AGENT_WALLET_ADDRESS"] == "0x000000000000000000000000000000000000ba5e"
    assert os.environ["BANKR_AGENT_API_KEY"] == "bk_agent_cleartext_secret"
    assert os.environ["BANKR_AGENT_WALLET_ID"] == "wlt_123"
    assert os.environ["BANKR_AGENT_WITHDRAWAL_DESTINATION"] == "0x000000000000000000000000000000000000feed"
    assert os.environ["BANKR_API_KEY"] == "bk_agent_cleartext_secret"
    assert os.environ["BANKR_WALLET_ADDRESS"] == "0x000000000000000000000000000000000000ba5e"


def test_load_config_cached_result_reexports_bankr_environment(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        """
bankr:
  walletAddress: "0x000000000000000000000000000000000000cafe"
  apiKey: "bk_cached_secret"
  walletId: "wlt_cached"
  withdrawalDestination: "0x000000000000000000000000000000000000beef"
""",
        encoding="utf-8",
    )

    _clear_bankr_env(monkeypatch)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    from hermes_cli import config as config_module

    _clear_config_caches(config_module)
    config_module.load_config()
    _clear_bankr_env(monkeypatch)

    config_module.load_config()

    assert os.environ["BANKR_AGENT_WALLET_ADDRESS"] == "0x000000000000000000000000000000000000cafe"
    assert os.environ["BANKR_AGENT_API_KEY"] == "bk_cached_secret"
    assert os.environ["BANKR_AGENT_WALLET_ID"] == "wlt_cached"
    assert os.environ["BANKR_AGENT_WITHDRAWAL_DESTINATION"] == "0x000000000000000000000000000000000000beef"
    assert os.environ["BANKR_API_KEY"] == "bk_cached_secret"
    assert os.environ["BANKR_WALLET_ADDRESS"] == "0x000000000000000000000000000000000000cafe"


def test_save_config_exports_bankr_environment(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()

    _clear_bankr_env(monkeypatch)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    from hermes_cli import config as config_module

    _clear_config_caches(config_module)
    config_module.save_config({
        "bankr": {
            "walletAddress": "0x000000000000000000000000000000000000f00d",
            "apiKey": "bk_saved_secret",
            "walletId": "wlt_saved",
            "withdrawalDestination": "0x000000000000000000000000000000000000d00d",
        }
    })

    assert os.environ["BANKR_AGENT_WALLET_ADDRESS"] == "0x000000000000000000000000000000000000f00d"
    assert os.environ["BANKR_AGENT_API_KEY"] == "bk_saved_secret"
    assert os.environ["BANKR_AGENT_WALLET_ID"] == "wlt_saved"
    assert os.environ["BANKR_AGENT_WITHDRAWAL_DESTINATION"] == "0x000000000000000000000000000000000000d00d"
    assert os.environ["BANKR_API_KEY"] == "bk_saved_secret"
    assert os.environ["BANKR_WALLET_ADDRESS"] == "0x000000000000000000000000000000000000f00d"

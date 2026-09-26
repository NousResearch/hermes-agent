"""Env-only setups must read as connected in the setup picker (#120870).

`hermes gateway setup` resolves each plugin platform's status with a synthetic empty
``PlatformConfig(enabled=True)``; the checkers that read ``extra`` alone therefore report
"not configured" for installs whose credentials live entirely in ``.env`` — the exact shape
the wizard itself writes. These tests pin the env rung on the three affected checkers
(feishu, wecom, wecom_callback), mirroring the dingtalk sibling.
"""

import pytest

from gateway.config import PlatformConfig


FEISHU_ENV = ("FEISHU_APP_ID", "FEISHU_APP_SECRET")
WECOM_ENV = ("WECOM_BOT_ID", "WECOM_SECRET")
WECOM_CALLBACK_ENV = ("WECOM_CALLBACK_CORP_ID", "WECOM_CALLBACK_CORP_SECRET")
ALL_ENV = FEISHU_ENV + WECOM_ENV + WECOM_CALLBACK_ENV


@pytest.fixture(autouse=True)
def _clean_platform_env(monkeypatch):
    for name in ALL_ENV:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def feishu_is_connected():
    from plugins.platforms.feishu.adapter import _is_connected
    return _is_connected


@pytest.fixture
def wecom_is_connected():
    from plugins.platforms.wecom.adapter import _is_connected
    return _is_connected


@pytest.fixture
def wecom_callback_is_connected():
    from plugins.platforms.wecom.adapter import _callback_is_connected
    return _callback_is_connected


class TestFeishuEnvCredentials:
    def test_env_only_pair_reads_connected(self, monkeypatch, feishu_is_connected):
        """The wizard's synthetic empty config must still see env-only credentials."""
        monkeypatch.setenv("FEISHU_APP_ID", "cli_a1")
        monkeypatch.setenv("FEISHU_APP_SECRET", "s3cret")
        assert feishu_is_connected(PlatformConfig(enabled=True)) is True

    def test_half_env_pair_stays_not_connected(self, monkeypatch, feishu_is_connected):
        """connect() rejects a missing secret; the picker must not call a half-configured install ready."""
        monkeypatch.setenv("FEISHU_APP_ID", "cli_a1")
        assert feishu_is_connected(PlatformConfig(enabled=True)) is False

    def test_extra_pair_still_connected(self, feishu_is_connected):
        assert feishu_is_connected(PlatformConfig(enabled=True, extra={"app_id": "cli_a1", "app_secret": "s3cret"})) is True

    def test_nothing_configured(self, feishu_is_connected):
        assert feishu_is_connected(PlatformConfig(enabled=True)) is False


class TestWecomEnvCredentials:
    def test_env_only_pair_reads_connected(self, monkeypatch, wecom_is_connected):
        monkeypatch.setenv("WECOM_BOT_ID", "bot-1")
        monkeypatch.setenv("WECOM_SECRET", "s3cret")
        assert wecom_is_connected(PlatformConfig(enabled=True)) is True

    def test_half_env_pair_stays_not_connected(self, monkeypatch, wecom_is_connected):
        monkeypatch.setenv("WECOM_BOT_ID", "bot-1")
        assert wecom_is_connected(PlatformConfig(enabled=True)) is False

    def test_extra_pair_still_connected(self, wecom_is_connected):
        assert wecom_is_connected(PlatformConfig(enabled=True, extra={"bot_id": "bot-1", "secret": "s3cret"})) is True

    def test_nothing_configured(self, wecom_is_connected):
        assert wecom_is_connected(PlatformConfig(enabled=True)) is False


class TestWecomCallbackEnvCredentials:
    def test_env_only_pair_reads_connected(self, monkeypatch, wecom_callback_is_connected):
        monkeypatch.setenv("WECOM_CALLBACK_CORP_ID", "corp-1")
        monkeypatch.setenv("WECOM_CALLBACK_CORP_SECRET", "s3cret")
        assert wecom_callback_is_connected(PlatformConfig(enabled=True)) is True

    def test_half_env_pair_stays_not_connected(self, monkeypatch, wecom_callback_is_connected):
        monkeypatch.setenv("WECOM_CALLBACK_CORP_ID", "corp-1")
        assert wecom_callback_is_connected(PlatformConfig(enabled=True)) is False

    def test_extra_pair_still_connected(self, wecom_callback_is_connected):
        assert wecom_callback_is_connected(
            PlatformConfig(enabled=True, extra={"corp_id": "corp-1", "corp_secret": "s3cret"})) is True

    def test_multi_app_block_still_connected(self, wecom_callback_is_connected):
        """The multi-app YAML block has no env equivalent and stays a config-only rung."""
        apps = [{"name": "a", "corp_id": "corp-1"}]
        assert wecom_callback_is_connected(PlatformConfig(enabled=True, extra={"apps": apps})) is True

    def test_nothing_configured(self, wecom_callback_is_connected):
        assert wecom_callback_is_connected(PlatformConfig(enabled=True)) is False

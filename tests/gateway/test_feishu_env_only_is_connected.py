"""The Feishu connectivity checker must answer for the credentials the adapter reads.

``hermes gateway setup``'s picker (``hermes_cli.gateway_setup_wizard._platform_status``)
hands every registry platform a synthetic ``PlatformConfig(enabled=True)`` — no ``extra``,
no ``token``. An env-only install (``FEISHU_APP_ID`` / ``FEISHU_APP_SECRET`` in the
profile's ``.env``, which is the shape the wizard itself writes) therefore read as
"not configured" even though the gateway's adapter was live-connected, because the checker
consulted ``extra`` alone while ``_load_settings`` resolves through ``extra_or_secret``
(env → YAML). #120870

The checker must also stay in agreement with ``connect()``: the adapter refuses to start
without an ``app_secret``, so a half-configured install must not read as connected.
"""

import os

import pytest

from gateway.config import PlatformConfig


@pytest.fixture
def feishu_is_connected():
    """The plugin's real checker; skips when the optional lark SDK is unavailable."""
    pytest.importorskip("plugins.platforms.feishu.adapter")
    from plugins.platforms.feishu.adapter import _is_connected

    return _is_connected


@pytest.fixture(autouse=True)
def _clean_feishu_env(monkeypatch):
    """No ambient credentials: each case states its own."""
    for name in ("FEISHU_APP_ID", "FEISHU_APP_SECRET"):
        monkeypatch.delenv(name, raising=False)


def test_env_only_reads_connected_with_empty_config(feishu_is_connected, monkeypatch):
    """The wizard's synthetic config + .env credentials must read as configured."""
    monkeypatch.setenv("FEISHU_APP_ID", "cli_env_only")
    monkeypatch.setenv("FEISHU_APP_SECRET", "secret_env_only")

    assert feishu_is_connected(PlatformConfig(enabled=True)) is True


def test_yaml_extra_still_reads_connected(feishu_is_connected):
    """The pre-existing YAML path (credentials in config.extra) must keep working."""
    config = PlatformConfig(enabled=True, extra={"app_id": "cli_yaml", "app_secret": "secret_yaml"})

    assert feishu_is_connected(config) is True


def test_env_beats_yaml_like_the_adapter(feishu_is_connected, monkeypatch):
    """Env wins over YAML in extra_or_secret, so the checker reports the adapter's view."""
    monkeypatch.setenv("FEISHU_APP_ID", "cli_from_env")
    monkeypatch.setenv("FEISHU_APP_SECRET", "secret_from_env")
    config = PlatformConfig(enabled=True, extra={"app_id": "", "app_secret": ""})

    assert feishu_is_connected(config) is True


def test_half_configured_app_id_only_is_not_connected(feishu_is_connected, monkeypatch):
    """``connect()`` rejects a missing app_secret, so the picker must not call it ready."""
    monkeypatch.setenv("FEISHU_APP_ID", "cli_half")
    config = PlatformConfig(enabled=True, extra={"app_id": "cli_half"})

    assert feishu_is_connected(config) is False


def test_unconfigured_is_not_connected(feishu_is_connected):
    """No env, no extra: still honestly "not configured"."""
    assert feishu_is_connected(PlatformConfig(enabled=True)) is False


def test_blank_values_are_unset(feishu_is_connected, monkeypatch):
    """Blank env / blank YAML count as unset — never as credentials."""
    monkeypatch.setenv("FEISHU_APP_ID", "   ")
    monkeypatch.setenv("FEISHU_APP_SECRET", "")
    config = PlatformConfig(enabled=True, extra={"app_id": "  ", "app_secret": ""})

    assert feishu_is_connected(config) is False


def test_checker_agrees_with_adapter_settings(feishu_is_connected, monkeypatch):
    """Agreement invariant: checker True means ``_load_settings`` resolves both halves."""
    monkeypatch.setenv("FEISHU_APP_ID", "cli_agreement")
    monkeypatch.setenv("FEISHU_APP_SECRET", "secret_agreement")

    from plugins.platforms.feishu.adapter import FeishuAdapter

    settings = FeishuAdapter._load_settings({})
    assert feishu_is_connected(PlatformConfig(enabled=True)) is True
    assert settings.app_id and settings.app_secret

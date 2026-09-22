"""The startup "No env user allowlists configured" reminder fires only when a messaging
platform is enabled — an API-server-only gateway has no sender to gate (#115439)."""

import logging

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.run import GatewayRunner

_WARNING = "No env user allowlists configured"


def _runner(platforms, tmp_path):
    config = GatewayConfig(platforms=platforms, sessions_dir=tmp_path / "sessions")
    return GatewayRunner(config)


@pytest.mark.parametrize(
    ("platforms", "expect_warning"),
    [
        ({Platform.API_SERVER: PlatformConfig(enabled=True, extra={"key": "k" * 40})}, False),
        ({Platform.API_SERVER: PlatformConfig(enabled=True), Platform.TELEGRAM: PlatformConfig(enabled=True)}, True),
        ({Platform.TELEGRAM: PlatformConfig(enabled=False)}, False),
    ],
    ids=["api_server_only", "api_server_plus_telegram", "telegram_disabled"],
)
def test_allowlist_warning_requires_an_enabled_messaging_platform(
    monkeypatch, tmp_path, caplog, platforms, expect_warning,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for var in list(GatewayRunner._BUILTIN_ALLOWED_USERS_VARS) + list(GatewayRunner._BUILTIN_ALLOW_ALL_VARS):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)
    runner = _runner(platforms, tmp_path)

    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        refused = runner._start_check_access_policy()

    assert refused is False
    assert (_WARNING in caplog.text) is expect_warning

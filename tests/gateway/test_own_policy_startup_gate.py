"""Regression tests for own-policy open startup gate in hermes_agent/gateway/run.py."""

import pytest

from hermes_agent.gateway.config import GatewayConfig, Platform, PlatformConfig
from hermes_agent.gateway.run import GatewayRunner


@pytest.mark.asyncio
async def test_unrelated_allow_all_does_not_bypass_own_policy_open_gate(
    monkeypatch, tmp_path,
):
    """TELEGRAM_ALLOW_ALL_USERS must not satisfy Weixin's open-policy opt-in."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)
    monkeypatch.delenv("WEIXIN_ALLOW_ALL_USERS", raising=False)
    monkeypatch.setenv("TELEGRAM_ALLOW_ALL_USERS", "true")

    config = GatewayConfig(
        platforms={
            Platform.WEIXIN: PlatformConfig(
                enabled=True,
                extra={"dm_policy": "open"},
            ),
        },
        sessions_dir=tmp_path / "sessions",
    )
    runner = GatewayRunner(config)

    ok = await runner.start()

    assert ok is True
    assert runner.should_exit_cleanly is True
    assert "weixin" in (runner.exit_reason or "").lower()



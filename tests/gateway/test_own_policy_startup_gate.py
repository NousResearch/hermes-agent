"""Regression tests for own-policy open startup gate in gateway/run.py."""

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.run import GatewayRunner


@pytest.mark.asyncio
async def test_unrelated_allow_all_does_not_bypass_yuanbao_open_gate(
    monkeypatch, tmp_path,
):
    """TELEGRAM_ALLOW_ALL_USERS must not satisfy Yuanbao's open-policy opt-in."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)
    monkeypatch.delenv("YUANBAO_ALLOW_ALL_USERS", raising=False)
    monkeypatch.setenv("TELEGRAM_ALLOW_ALL_USERS", "true")

    config = GatewayConfig(
        platforms={
            Platform.YUANBAO: PlatformConfig(
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
    assert "yuanbao" in (runner.exit_reason or "").lower()


@pytest.mark.asyncio
async def test_gateway_allow_all_satisfies_yuanbao_open_gate(monkeypatch, tmp_path):
    """GATEWAY_ALLOW_ALL_USERS is the intended global open-policy opt-in."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "true")
    monkeypatch.delenv("YUANBAO_ALLOW_ALL_USERS", raising=False)
    monkeypatch.delenv("TELEGRAM_ALLOW_ALL_USERS", raising=False)

    config = GatewayConfig(
        platforms={
            Platform.YUANBAO: PlatformConfig(
                enabled=True,
                extra={"dm_policy": "open"},
            ),
        },
        sessions_dir=tmp_path / "sessions",
    )
    runner = GatewayRunner(config)
    monkeypatch.setattr(runner, "_create_adapter", lambda platform, cfg: None)

    ok = await runner.start()

    assert ok is True
    assert runner.should_exit_cleanly is False


@pytest.mark.asyncio
async def test_unrelated_allow_all_does_not_bypass_feishu_open_gate(
    monkeypatch, tmp_path,
):
    """TELEGRAM_ALLOW_ALL_USERS must not satisfy Feishu's open-policy opt-in.

    Mirrors the Yuanbao gate above. Feishu's config-file key for this is
    ``default_group_policy`` (not ``group_policy`` — see
    FeishuAdapterSettings.default_group_policy), so this also pins that the
    startup guard reads the key Feishu's adapter actually honors.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)
    monkeypatch.delenv("FEISHU_ALLOW_ALL_USERS", raising=False)
    monkeypatch.setenv("TELEGRAM_ALLOW_ALL_USERS", "true")

    config = GatewayConfig(
        platforms={
            Platform.FEISHU: PlatformConfig(
                enabled=True,
                extra={"default_group_policy": "open"},
            ),
        },
        sessions_dir=tmp_path / "sessions",
    )
    runner = GatewayRunner(config)

    ok = await runner.start()

    assert ok is True
    assert runner.should_exit_cleanly is True
    assert "feishu" in (runner.exit_reason or "").lower()


@pytest.mark.asyncio
async def test_feishu_group_policy_env_var_triggers_gate(monkeypatch, tmp_path):
    """FEISHU_GROUP_POLICY=open (the interactive-setup-wizard path — see
    plugins/platforms/feishu/adapter.py's interactive_setup, which calls
    save_env_value("FEISHU_GROUP_POLICY", "open") for the recommended
    "respond only when @mentioned" choice) must also require the allow-all
    opt-in, not just the config.yaml extra key."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)
    monkeypatch.delenv("FEISHU_ALLOW_ALL_USERS", raising=False)
    monkeypatch.setenv("FEISHU_GROUP_POLICY", "open")

    config = GatewayConfig(
        platforms={
            Platform.FEISHU: PlatformConfig(enabled=True, extra={}),
        },
        sessions_dir=tmp_path / "sessions",
    )
    runner = GatewayRunner(config)

    ok = await runner.start()

    assert ok is True
    assert runner.should_exit_cleanly is True
    assert "feishu" in (runner.exit_reason or "").lower()


@pytest.mark.asyncio
async def test_gateway_allow_all_satisfies_feishu_open_gate(monkeypatch, tmp_path):
    """GATEWAY_ALLOW_ALL_USERS is the intended global open-policy opt-in."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "true")
    monkeypatch.delenv("FEISHU_ALLOW_ALL_USERS", raising=False)
    monkeypatch.delenv("TELEGRAM_ALLOW_ALL_USERS", raising=False)

    config = GatewayConfig(
        platforms={
            Platform.FEISHU: PlatformConfig(
                enabled=True,
                extra={"default_group_policy": "open"},
            ),
        },
        sessions_dir=tmp_path / "sessions",
    )
    runner = GatewayRunner(config)
    monkeypatch.setattr(runner, "_create_adapter", lambda platform, cfg: None)

    ok = await runner.start()

    assert ok is True
    assert runner.should_exit_cleanly is False
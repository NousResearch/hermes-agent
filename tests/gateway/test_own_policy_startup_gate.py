"""Regression tests for own-policy open startup gate in gateway/run.py."""

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter
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


class _StubAdapter(BasePlatformAdapter):
    """Minimal adapter that connects successfully, so start() reaches the connect bookkeeping."""

    def __init__(self, platform: Platform):
        super().__init__(PlatformConfig(enabled=True, token="***"), platform)

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        raise NotImplementedError

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


async def _no_secondary_profiles() -> int:
    """Stub for the secondary-profile fan-out; this file only exercises the default profile."""
    return 0


def _clear_open_policy_env(monkeypatch, tmp_path) -> None:
    """No allow-all opt-in anywhere, so an ``open`` policy is a genuine violation."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for var in (
        "GATEWAY_ALLOW_ALL_USERS", "WHATSAPP_ALLOW_ALL_USERS", "YUANBAO_ALLOW_ALL_USERS",
        "WECOM_ALLOW_ALL_USERS", "WEIXIN_ALLOW_ALL_USERS", "QQ_ALLOW_ALL_USERS",
        "WHATSAPP_DM_POLICY", "WHATSAPP_GROUP_POLICY", "YUANBAO_DM_POLICY", "YUANBAO_GROUP_POLICY",
    ):
        monkeypatch.delenv(var, raising=False)


def _open_policy_config(tmp_path, platforms) -> GatewayConfig:
    return GatewayConfig(platforms=platforms, sessions_dir=tmp_path / "sessions")


def test_all_offending_platforms_are_reported(monkeypatch, tmp_path):
    """The gate reports *every* offender, not just the first one it trips over.

    Short-circuiting on the first offender is what made the failure all-or-nothing: the caller
    could only ever learn about one platform, so it had no way to disable them individually.
    """
    from gateway.run import _own_policy_open_violations

    _clear_open_policy_env(monkeypatch, tmp_path)
    config = _open_policy_config(
        tmp_path,
        {
            Platform.TELEGRAM: PlatformConfig(enabled=True, token="***"),
            Platform.WHATSAPP: PlatformConfig(enabled=True, extra={"dm_policy": "open"}),
            Platform.YUANBAO: PlatformConfig(enabled=True, extra={"group_policy": "open"}),
        },
    )

    violations = _own_policy_open_violations(config)

    assert {p for p, _ in violations} == {Platform.WHATSAPP, Platform.YUANBAO}
    # Each offender carries the flag that would re-enable it, so the error can name it.
    assert dict(violations)[Platform.WHATSAPP] == "WHATSAPP_ALLOW_ALL_USERS"
    assert dict(violations)[Platform.YUANBAO] == "YUANBAO_ALLOW_ALL_USERS"


@pytest.mark.asyncio
async def test_quarantine_keeps_healthy_platform_serving(monkeypatch, tmp_path):
    """One misconfigured platform must not take the whole gateway down.

    The reported outage: an enabled-but-unpaired WhatsApp left on ``dm_policy: open`` kept a
    healthy Telegram bot offline, and ``Restart=always`` then crash-looped the unit. Driven
    through ``GatewayRunner.start()`` so the real startup branch is what gets exercised.
    """
    _clear_open_policy_env(monkeypatch, tmp_path)
    config = _open_policy_config(
        tmp_path,
        {
            Platform.TELEGRAM: PlatformConfig(enabled=True, token="***"),
            Platform.WHATSAPP: PlatformConfig(enabled=True, extra={"dm_policy": "open"}),
        },
    )
    runner = GatewayRunner(config)
    stamped: list = []
    monkeypatch.setattr(runner, "_create_adapter", lambda platform, _cfg: _StubAdapter(platform))
    monkeypatch.setattr(runner, "_start_secondary_profile_adapters", _no_secondary_profiles)
    monkeypatch.setattr(
        runner, "_update_platform_runtime_status",
        lambda platform, **kw: stamped.append((platform, kw)),
    )

    await runner.start()

    # The gateway kept running, and the healthy platform is actually serving.
    assert runner.should_exit_cleanly is False
    assert Platform.TELEGRAM in runner.adapters
    # The offender is disabled — fail-closed for it, so it accepts nothing.
    assert config.platforms[Platform.WHATSAPP].enabled is False
    assert Platform.WHATSAPP not in runner.adapters
    # ...and it says *why* it is down, rather than keeping whatever it last reported. Only the
    # quarantine writes for the offender; the healthy platform's own connect stamps are ignored.
    assert [kw for platform, kw in stamped if platform == Platform.WHATSAPP.value] == [{
        "platform_state": "disabled",
        "error_code": "open_policy_no_opt_in",
        "error_message": (
            "Disabled at startup: dm_policy/group_policy is 'open' but neither "
            "GATEWAY_ALLOW_ALL_USERS nor WHATSAPP_ALLOW_ALL_USERS is enabled."
        ),
    }]


@pytest.mark.asyncio
async def test_opt_in_flag_still_suppresses_the_gate(monkeypatch, tmp_path):
    """With the allow-all opt-in set there is no violation, so nothing is quarantined."""
    _clear_open_policy_env(monkeypatch, tmp_path)
    monkeypatch.setenv("WHATSAPP_ALLOW_ALL_USERS", "true")
    config = _open_policy_config(
        tmp_path,
        {
            Platform.TELEGRAM: PlatformConfig(enabled=True, token="***"),
            Platform.WHATSAPP: PlatformConfig(enabled=True, token="***", extra={"dm_policy": "open"}),
        },
    )
    runner = GatewayRunner(config)
    monkeypatch.setattr(runner, "_create_adapter", lambda platform, _cfg: _StubAdapter(platform))
    monkeypatch.setattr(runner, "_start_secondary_profile_adapters", _no_secondary_profiles)

    await runner.start()

    assert runner.should_exit_cleanly is False
    assert config.platforms[Platform.WHATSAPP].enabled is True
    assert Platform.WHATSAPP in runner.adapters


@pytest.mark.asyncio
async def test_every_platform_offending_still_refuses_startup(monkeypatch, tmp_path):
    """When nothing survives the gate, refusing to start is still the right answer.

    Quarantining every platform would leave a gateway running with no transports at all, so the
    original refuse-to-start path is preserved for that case.
    """
    _clear_open_policy_env(monkeypatch, tmp_path)
    config = _open_policy_config(
        tmp_path,
        {
            Platform.WHATSAPP: PlatformConfig(enabled=True, extra={"dm_policy": "open"}),
            Platform.YUANBAO: PlatformConfig(enabled=True, extra={"dm_policy": "open"}),
        },
    )
    runner = GatewayRunner(config)

    ok = await runner.start()

    assert ok is True
    assert runner.should_exit_cleanly is True
    assert "open policy without allow-all opt-in" in (runner.exit_reason or "")
    assert not runner.adapters


def test_quarantined_state_survives_both_status_consumers():
    """``disabled`` must mean "known, not serving, not an error" to every status reader.

    The first cut of this fix wrote ``stopped``, which the health snapshot does not recognize and
    bounds to ``unknown`` — losing the reason. These are the two consumers that classify the value.
    """
    from agent.monitoring.gateway_health import (
        _FATAL_PLATFORM_STATES, _KNOWN_PLATFORM_STATES, _RUNNING_PLATFORM_STATES,
    )
    from hermes_cli.web_server_gateway import _PLATFORM_DEAD_STATES

    # Known, so the snapshot keeps it instead of bounding it to "unknown".
    assert "disabled" in _KNOWN_PLATFORM_STATES
    # Not serving its port, so the dashboard does not show a quarantined adapter as live.
    assert "disabled" in _PLATFORM_DEAD_STATES
    # A configuration choice is neither "up" nor an error-severity page.
    assert "disabled" not in _RUNNING_PLATFORM_STATES
    assert "disabled" not in _FATAL_PLATFORM_STATES

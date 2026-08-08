"""Regression tests for multiplex profile-aware own-policy authorization."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.session import SessionSource


def _clear_auth_env(monkeypatch) -> None:
    for key in (
        "WECOM_ALLOWED_USERS",
        "GATEWAY_ALLOWED_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
        "WECOM_ALLOW_ALL_USERS",
    ):
        monkeypatch.delenv(key, raising=False)


def _make_multiplex_runner(monkeypatch):
    """Runner with default allowlist WeCom and secondary open-policy WeCom."""
    from gateway.run import GatewayRunner

    _clear_auth_env(monkeypatch)

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)

    default_adapter = SimpleNamespace(
        send=AsyncMock(),
        enforces_own_access_policy=True,
        _dm_policy="allowlist",
        _group_policy="pairing",
    )
    secondary_adapter = SimpleNamespace(
        send=AsyncMock(),
        enforces_own_access_policy=True,
        _dm_policy="open",
        _group_policy="open",
    )

    runner.adapters = {Platform.WECOM: default_adapter}
    runner._profile_adapters = {
        "coder": {Platform.WECOM: secondary_adapter},
    }
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = False
    return runner, default_adapter, secondary_adapter


def test_default_profile_still_trusts_own_allowlist(monkeypatch):
    """Default-profile allowlist trust is unchanged when profile is unstamped."""
    runner, _default_adapter, _secondary_adapter = _make_multiplex_runner(monkeypatch)

    source = SessionSource(
        platform=Platform.WECOM,
        user_id="allowed-user",
        chat_id="dm-chat",
        user_name="allowed-user",
        chat_type="dm",
        profile=None,
    )

    assert runner._is_user_authorized(source) is True


def test_active_profile_stamp_resolves_primary_adapter(monkeypatch):
    """A single-profile gateway stamps its active profile but stores adapters as primary."""
    runner, default_adapter, _secondary_adapter = _make_multiplex_runner(monkeypatch)
    runner._active_profile_name = lambda: "dev"

    assert runner._authorization_adapter(Platform.WECOM, profile="dev") is default_adapter


def test_secondary_allowlist_dm_behavior_ignores_unauthorized(monkeypatch):
    """Unauthorized-DM behavior must read the secondary adapter's dm_policy."""
    runner, _default_adapter, secondary_adapter = _make_multiplex_runner(monkeypatch)
    secondary_adapter._dm_policy = "allowlist"

    assert runner._get_unauthorized_dm_behavior(
        Platform.WECOM,
        profile="coder",
    ) == "ignore"
    assert runner._get_unauthorized_dm_behavior(Platform.WECOM) == "ignore"


def test_adapter_auth_check_stamps_secondary_profile(monkeypatch):
    """The adapter auth-check callback must stamp its own secondary profile.

    Regression for the gap where ``_make_adapter_auth_check`` built a
    profile-less ``SessionSource``, so a secondary adapter's external-context
    authorization (e.g. Slack/Discord thread-reply lookups) silently
    resolved the *active* profile's allowlist scope instead of its own.
    """
    from gateway.run import GatewayRunner

    _clear_auth_env(monkeypatch)

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)

    captured: dict = {}

    def fake_is_user_authorized(source):
        captured["profile"] = source.profile
        return True

    runner._is_user_authorized = fake_is_user_authorized

    check = runner._make_adapter_auth_check(Platform.WECOM, profile_name="coder")
    assert check("some-user", "dm", "dm-chat") is True
    assert captured["profile"] == "coder"


def test_secondary_open_policy_fails_startup_guard(monkeypatch):
    """Secondary profiles must pass the same open-policy startup guard."""
    from gateway.run import _own_policy_open_startup_violation

    _clear_auth_env(monkeypatch)

    secondary_cfg = GatewayConfig(multiplex_profiles=True)
    secondary_cfg.platforms = {
        Platform.WECOM: PlatformConfig(
            enabled=True,
            extra={"dm_policy": "open"},
        ),
    }

    violation = _own_policy_open_startup_violation(secondary_cfg)
    assert violation is not None
    assert "wecom" in violation
    assert "open policy" in violation


def _clear_whatsapp_auth_env(monkeypatch) -> None:
    for key in (
        "GATEWAY_ALLOW_ALL_USERS",
        "WHATSAPP_ALLOW_ALL_USERS",
        "WHATSAPP_DM_POLICY",
        "WHATSAPP_GROUP_POLICY",
        "WECOM_ALLOW_ALL_USERS",
    ):
        monkeypatch.delenv(key, raising=False)


def test_open_policy_violations_reports_every_offender(monkeypatch):
    """The gate must surface all offenders, not short-circuit on the first."""
    from gateway.run import _own_policy_open_violations

    _clear_whatsapp_auth_env(monkeypatch)

    cfg = GatewayConfig()
    cfg.platforms = {
        Platform.WHATSAPP: PlatformConfig(enabled=True, extra={"dm_policy": "open"}),
        Platform.WECOM: PlatformConfig(enabled=True, extra={"group_policy": "open"}),
        Platform.TELEGRAM: PlatformConfig(enabled=True),
    }

    offenders = {platform for platform, _ in _own_policy_open_violations(cfg)}
    assert offenders == {Platform.WHATSAPP, Platform.WECOM}


def _quarantine_runner(monkeypatch, platforms):
    """Runner wired to exercise the real ``_quarantine_open_policy_platforms``.

    Only the two collaborators the method reaches outside itself are stubbed —
    the runtime-status writer and the clean-exit request — so the production
    control flow, logging and config mutation all run for real.
    """
    from gateway.run import GatewayRunner

    _clear_whatsapp_auth_env(monkeypatch)

    cfg = GatewayConfig()
    cfg.platforms = platforms

    runner = object.__new__(GatewayRunner)
    runner.config = cfg
    status_writes: list = []
    clean_exits: list = []
    runner._update_platform_runtime_status = (
        lambda platform, **kw: status_writes.append((platform, kw))
    )
    runner._request_clean_exit = clean_exits.append
    return runner, cfg, status_writes, clean_exits


def test_open_policy_quarantines_offender_and_keeps_gateway_up(monkeypatch):
    """A misconfigured platform is disabled; healthy ones keep serving.

    Regression for the case where an enabled-but-unpaired WhatsApp aborted the
    whole gateway, taking a perfectly healthy Telegram down with it.
    """
    runner, cfg, _writes, clean_exits = _quarantine_runner(
        monkeypatch,
        {
            Platform.WHATSAPP: PlatformConfig(enabled=True, extra={"dm_policy": "open"}),
            Platform.TELEGRAM: PlatformConfig(enabled=True),
        },
    )

    assert runner._quarantine_open_policy_platforms() is False
    assert cfg.platforms[Platform.WHATSAPP].enabled is False
    assert cfg.platforms[Platform.TELEGRAM].enabled is True
    assert clean_exits == []


def test_open_policy_quarantine_aborts_when_nothing_left_to_serve(monkeypatch):
    """With every enabled platform quarantined, startup must still refuse."""
    runner, cfg, _writes, clean_exits = _quarantine_runner(
        monkeypatch,
        {
            Platform.WHATSAPP: PlatformConfig(enabled=True, extra={"dm_policy": "open"}),
            Platform.WECOM: PlatformConfig(enabled=True, extra={"group_policy": "open"}),
        },
    )

    assert runner._quarantine_open_policy_platforms() is True
    assert not any(pc.enabled for pc in cfg.platforms.values())
    assert len(clean_exits) == 1
    assert "open policy without allow-all opt-in" in clean_exits[0]


def test_open_policy_quarantine_noop_without_violations(monkeypatch):
    """A compliant config must not touch platforms or request an exit."""
    runner, cfg, writes, clean_exits = _quarantine_runner(
        monkeypatch,
        {
            Platform.WHATSAPP: PlatformConfig(
                enabled=True, extra={"dm_policy": "allowlist"}
            ),
            Platform.TELEGRAM: PlatformConfig(enabled=True),
        },
    )

    assert runner._quarantine_open_policy_platforms() is False
    assert all(pc.enabled for pc in cfg.platforms.values())
    assert writes == []
    assert clean_exits == []


def test_open_policy_opt_in_keeps_platform_enabled(monkeypatch):
    """The platform allow-all flag still suppresses the gate entirely."""
    from gateway.run import _own_policy_open_violations

    _clear_whatsapp_auth_env(monkeypatch)
    monkeypatch.setenv("WHATSAPP_ALLOW_ALL_USERS", "true")

    cfg = GatewayConfig()
    cfg.platforms = {
        Platform.WHATSAPP: PlatformConfig(enabled=True, extra={"dm_policy": "open"}),
    }

    assert _own_policy_open_violations(cfg) == []


def test_open_policy_quarantine_records_platform_runtime_status(monkeypatch):
    """Quarantining a platform must stamp its runtime status, not leave it stale.

    Without this the status file keeps the platform's last reported state, so a
    previously-connected adapter still reads "connected" after being disabled.
    """
    runner, _cfg, writes, _exits = _quarantine_runner(
        monkeypatch,
        {
            Platform.WHATSAPP: PlatformConfig(enabled=True, extra={"dm_policy": "open"}),
            Platform.TELEGRAM: PlatformConfig(enabled=True),
        },
    )

    runner._quarantine_open_policy_platforms()

    assert len(writes) == 1
    name, kwargs = writes[0]
    assert name == "whatsapp"
    assert kwargs["platform_state"] == "disabled"
    assert kwargs["error_code"] == "open_policy_no_opt_in"


def test_quarantine_state_is_known_to_health_monitor_and_dashboard():
    """The state written must survive both consumers unchanged.

    ``_bounded_state`` rewrites anything outside ``_KNOWN_PLATFORM_STATES`` to
    ``"unknown"``, which would silently erase the reason a platform is down;
    the dashboard separately needs it classified as not-serving.
    """
    from agent.monitoring.gateway_health import (
        _FATAL_PLATFORM_STATES,
        _KNOWN_PLATFORM_STATES,
        _RUNNING_PLATFORM_STATES,
    )
    from hermes_cli.web_server import _PLATFORM_DEAD_STATES

    assert "disabled" in _KNOWN_PLATFORM_STATES
    assert "disabled" in _PLATFORM_DEAD_STATES
    # Not "up", and not an error-severity alert for a deliberate config choice.
    assert "disabled" not in _RUNNING_PLATFORM_STATES
    assert "disabled" not in _FATAL_PLATFORM_STATES


def test_open_policy_all_platforms_offending_leaves_nothing_enabled(monkeypatch):
    """When every platform fails the gate, nothing remains to serve."""
    from gateway.run import _own_policy_open_violations

    _clear_whatsapp_auth_env(monkeypatch)

    cfg = GatewayConfig()
    cfg.platforms = {
        Platform.WHATSAPP: PlatformConfig(enabled=True, extra={"dm_policy": "open"}),
    }

    for platform, _ in _own_policy_open_violations(cfg):
        cfg.platforms[platform].enabled = False

    assert not any(pc.enabled for pc in cfg.platforms.values())

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


def test_session_context_records_primary_transport_owner_for_routed_runtime(monkeypatch):
    """A runtime route must not relabel the bot that will send notifications."""
    runner, default_adapter, _secondary_adapter = _make_multiplex_runner(monkeypatch)
    runner._active_profile_name = lambda: "default"
    source = SessionSource(
        platform=Platform.WECOM,
        user_id="allowed-user",
        chat_id="routed-chat",
        chat_type="dm",
        profile="daily",
    )
    setattr(source, "_transport_adapter_ref", lambda: default_adapter)

    assert runner._adapter_for_source(source) is default_adapter
    assert runner._adapter_profile_for_source(source) == "default"

    from gateway.session_context import clear_session_vars, get_session_env

    tokens = runner._set_session_env(
        SimpleNamespace(source=source, session_key="daily-routed-session")
    )  # type: ignore[arg-type]
    try:
        assert get_session_env("HERMES_SESSION_PROFILE") == "daily"
        assert get_session_env("HERMES_SESSION_TRANSPORT_PROFILE") == "default"
    finally:
        clear_session_vars(tokens)


def test_transport_owner_keeps_distinct_secondary_adapter_isolated(monkeypatch):
    runner, default_adapter, secondary_adapter = _make_multiplex_runner(monkeypatch)
    runner._active_profile_name = lambda: "default"
    source = SessionSource(
        platform=Platform.WECOM,
        user_id="allowed-user",
        chat_id="coder-chat",
        chat_type="dm",
        profile="coder",
    )
    setattr(source, "_transport_adapter_ref", lambda: secondary_adapter)

    assert runner._adapter_profile_for_source(source) == "coder"
    assert runner._adapter_for_source(source) is secondary_adapter
    assert runner._adapter_for_source(source) is not default_adapter


def test_transport_owner_uses_active_profile_for_routed_relay(monkeypatch):
    """The shared Relay socket belongs to the active gateway, not the runtime route."""
    runner, _default_adapter, _secondary_adapter = _make_multiplex_runner(monkeypatch)
    runner._active_profile_name = lambda: "default"
    relay_adapter = SimpleNamespace(send=AsyncMock())
    runner.adapters[Platform.RELAY] = relay_adapter  # type: ignore[assignment]
    source = SessionSource(
        platform=Platform.WECOM,
        user_id="allowed-user",
        chat_id="relay-chat",
        chat_type="dm",
        profile="daily",
        delivered_via_upstream_relay=True,
    )
    setattr(source, "_transport_adapter_ref", lambda: relay_adapter)

    assert runner._adapter_for_source(source) is relay_adapter
    assert runner._adapter_profile_for_source(source) == "default"


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

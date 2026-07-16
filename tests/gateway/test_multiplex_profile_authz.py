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


def test_secondary_open_policy_not_authorized_by_default_allowlist(monkeypatch):
    """Secondary-profile open intake must not inherit default allowlist trust."""
    runner, _default_adapter, _secondary_adapter = _make_multiplex_runner(monkeypatch)

    source = SessionSource(
        platform=Platform.WECOM,
        user_id="attacker",
        chat_id="dm-chat",
        user_name="attacker",
        chat_type="dm",
        profile="coder",
    )

    assert runner._adapter_dm_policy(Platform.WECOM, profile="coder") == "open"
    assert runner._adapter_dm_policy(Platform.WECOM) == "allowlist"
    assert runner._is_user_authorized(source) is False


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


def test_secondary_allowlist_still_authorized(monkeypatch):
    """Secondary profile with allowlist policy is trusted on its own adapter."""
    runner, _default_adapter, secondary_adapter = _make_multiplex_runner(monkeypatch)
    secondary_adapter._dm_policy = "allowlist"

    source = SessionSource(
        platform=Platform.WECOM,
        user_id="allowed-user",
        chat_id="dm-chat",
        user_name="allowed-user",
        chat_type="dm",
        profile="coder",
    )

    assert runner._is_user_authorized(source) is True


def test_adapter_for_source_resolves_secondary_profile_adapter(monkeypatch):
    """Ingress adapter lookup must use the stamped profile's adapter map."""
    runner, default_adapter, secondary_adapter = _make_multiplex_runner(monkeypatch)

    source = SessionSource(
        platform=Platform.WECOM,
        user_id="attacker",
        chat_id="dm-chat",
        user_name="attacker",
        chat_type="dm",
        profile="coder",
    )

    assert runner._adapter_for_source(source) is secondary_adapter
    assert runner._adapter_for_source(
        SessionSource(
            platform=Platform.WECOM,
            user_id="allowed-user",
            chat_id="dm-chat",
            user_name="allowed-user",
            chat_type="dm",
            profile=None,
        )
    ) is default_adapter


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


def test_adapter_auth_check_defaults_to_active_profile(monkeypatch):
    """Primary-adapter callbacks (no profile_name) still resolve the active profile."""
    from gateway.run import GatewayRunner

    _clear_auth_env(monkeypatch)

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)

    captured: dict = {}

    def fake_is_user_authorized(source):
        captured["profile"] = source.profile
        return True

    runner._is_user_authorized = fake_is_user_authorized

    check = runner._make_adapter_auth_check(Platform.WECOM)
    assert check("some-user", "dm", "dm-chat") is True
    assert captured["profile"] is None


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

def _routed_runner(monkeypatch, routes):
    """Runner whose Discord/Slack channels are routed by ``profile_routes``."""
    from gateway.profile_routing import parse_profile_routes
    from gateway.run import GatewayRunner

    _clear_auth_env(monkeypatch)
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner.config.profile_routes = parse_profile_routes(routes)
    return runner


def _capture_authorized_profile(runner):
    """Record the profile each authz decision is scoped to."""
    seen = {}

    def _fake_is_user_authorized(source):
        seen["profile"] = source.profile
        return True

    runner._is_user_authorized = _fake_is_user_authorized
    return seen


def test_primary_adapter_callback_resolves_channel_route(monkeypatch):
    """A shared primary adapter must authorize against the ROUTED profile.

    The primary adapter registers without a ``profile_name``, but a channel it
    serves may be routed elsewhere. Without route resolution the check reads the
    active profile's allowlist — marking the routed profile's own users
    ``[unverified]`` while trusting default-profile users.
    """
    runner = _routed_runner(monkeypatch, [
        {
            "name": "coder-channel",
            "platform": "slack",
            "chat_id": "C-CODER",
            "profile": "coder",
        },
    ])
    seen = _capture_authorized_profile(runner)

    check = runner._make_adapter_auth_check(Platform.SLACK)
    assert check("u1", "thread", "C-CODER") is True
    assert seen["profile"] == "coder"


def test_primary_adapter_callback_matches_guild_scoped_route(monkeypatch):
    """Guild-scoped routes match conjunctively — the callback must pass guild_id."""
    runner = _routed_runner(monkeypatch, [
        {
            "name": "coder-guild",
            "platform": "discord",
            "guild_id": "G-CODER",
            "profile": "coder",
        },
    ])
    seen = _capture_authorized_profile(runner)

    check = runner._make_adapter_auth_check(Platform.DISCORD)
    assert check("u1", "group", "chan-1", guild_id="G-CODER") is True
    assert seen["profile"] == "coder"


def test_primary_adapter_callback_unrouted_channel_uses_default(monkeypatch):
    """A channel with no matching route stays on the default profile."""
    runner = _routed_runner(monkeypatch, [
        {
            "name": "coder-channel",
            "platform": "slack",
            "chat_id": "C-CODER",
            "profile": "coder",
        },
    ])
    seen = _capture_authorized_profile(runner)

    check = runner._make_adapter_auth_check(Platform.SLACK)
    assert check("u1", "thread", "C-OTHER") is True
    assert seen["profile"] is None


def test_secondary_adapter_profile_wins_over_routes(monkeypatch):
    """An explicitly bound profile is authoritative — routes must not override it."""
    runner = _routed_runner(monkeypatch, [
        {
            "name": "coder-channel",
            "platform": "slack",
            "chat_id": "C-CODER",
            "profile": "coder",
        },
    ])
    seen = _capture_authorized_profile(runner)

    check = runner._make_adapter_auth_check(Platform.SLACK, profile_name="ops")
    assert check("u1", "thread", "C-CODER") is True
    assert seen["profile"] == "ops"

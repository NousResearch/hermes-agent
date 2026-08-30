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


def test_startup_guard_gateway_allow_all_reads_scope_not_environ(monkeypatch):
    """The GATEWAY_ALLOW_ALL_USERS opt-in check inside the startup guard
    must honor the active profile secret scope (#93522): the default
    profile's env-only opt-in must not leak into a secondary profile that
    never opted in, and a secondary profile's own scoped opt-in must be
    honored."""
    from agent import secret_scope
    from gateway.run import _own_policy_open_startup_violation

    _clear_auth_env(monkeypatch)
    cfg = GatewayConfig(multiplex_profiles=True)
    cfg.platforms = {
        Platform.WECOM: PlatformConfig(enabled=True, extra={"dm_policy": "open"}),
    }

    previous_multiplex = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(True)
    monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "true")
    try:
        token = secret_scope.set_secret_scope({"SOMETHING_ELSE": "x"})
        try:
            violation = _own_policy_open_startup_violation(cfg)
        finally:
            secret_scope.reset_secret_scope(token)
        assert violation is not None, "default profile's env opt-in must not leak into the scoped secondary profile"

        token = secret_scope.set_secret_scope({"GATEWAY_ALLOW_ALL_USERS": "true"})
        try:
            violation = _own_policy_open_startup_violation(cfg)
        finally:
            secret_scope.reset_secret_scope(token)
        assert violation is None, "the secondary profile's own scoped opt-in must be honored"
    finally:
        secret_scope.set_multiplex_active(previous_multiplex)


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


# ─────────────────────────────────────────────────────────────────────
# Plugin-platform extra.allowed_users fallback (#98738 / #82871)
# ─────────────────────────────────────────────────────────────────────

# Buzz has no static Platform member: plugin platforms get a dynamic
# member created on demand by Platform._missing_ (value lookup). Resolve
# it that way — attribute access only works after an earlier lookup in
# the same process, which a fresh CI shard cannot rely on.
_BUZZ = Platform("buzz")


def _make_buzz_multiplex_runner(monkeypatch, extra):
    """Runner whose secondary 'coder' profile runs a live Buzz adapter."""
    from gateway.run import GatewayRunner
    from tests.gateway.test_buzz_adapter import _normalize_user_ref

    for key in (
        "BUZZ_ALLOWED_USERS",
        "BUZZ_ALLOW_ALL_USERS",
        "GATEWAY_ALLOWED_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
    ):
        monkeypatch.delenv(key, raising=False)

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)

    adapter = SimpleNamespace(
        config=PlatformConfig(enabled=True, extra=extra),
        # The Buzz adapter exposes this hook so npub allowlist entries match
        # the hex-pubkey user ids the gateway authorizes.
        normalize_user_id=_normalize_user_ref,
    )
    runner.adapters = {}
    runner._profile_adapters = {"coder": {_BUZZ: adapter}}
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = False
    return runner


def _buzz_source(user_id):
    return SessionSource(
        platform=_BUZZ,
        user_id=user_id,
        chat_id="chat-1",
        user_name="member",
        chat_type="dm",
        profile="coder",
    )


def _patch_buzz_registry(monkeypatch, allowed_users_env="BUZZ_ALLOWED_USERS"):
    from gateway.platform_registry import platform_registry

    real_get = platform_registry.get

    def _get(key):
        if key == "buzz":
            return SimpleNamespace(allowed_users_env=allowed_users_env)
        return real_get(key)

    monkeypatch.setattr(platform_registry, "get", _get)


def test_secondary_buzz_extra_allowed_users_authorizes_listed_user(monkeypatch):
    """A secondary profile's extra.allowed_users must authorize its users when
    the env var only ever carried the default profile's list (#98738/#82871)."""
    from tests.gateway.test_buzz_adapter import SELF_NPUB, SELF_PUBKEY

    runner = _make_buzz_multiplex_runner(
        monkeypatch, extra={"allowed_users": [SELF_NPUB]}
    )
    _patch_buzz_registry(monkeypatch)

    # user_id arrives as the hex pubkey while the allowlist entry is an npub.
    assert runner._is_user_authorized(_buzz_source(SELF_PUBKEY)) is True


def test_secondary_buzz_extra_allowed_users_denies_unlisted_sender(monkeypatch):
    """Default-deny is preserved: a sender not in the profile's allowlist
    stays denied even though the adapter-level list admitted the message."""
    from tests.gateway.test_buzz_adapter import SELF_PUBKEY

    runner = _make_buzz_multiplex_runner(
        monkeypatch, extra={"allowed_users": ["npub1" + "b" * 56]}
    )
    _patch_buzz_registry(monkeypatch)

    assert runner._is_user_authorized(_buzz_source(SELF_PUBKEY)) is False


def test_secondary_buzz_without_extra_allowlist_stays_default_deny(monkeypatch):
    """No extra.allowed_users configured: nothing changes, the default-deny
    path applies (no fail-open via an empty list)."""
    from tests.gateway.test_buzz_adapter import SELF_PUBKEY

    runner = _make_buzz_multiplex_runner(monkeypatch, extra={})
    _patch_buzz_registry(monkeypatch)

    assert runner._is_user_authorized(_buzz_source(SELF_PUBKEY)) is False


def test_extra_allowed_users_not_consulted_without_registry_declaration(monkeypatch):
    """The fallback is gated on the platform's registry entry declaring
    allowed_users_env — a platform without that contract keeps the previous
    behavior even if its extra happens to hold an allowed_users key."""
    from tests.gateway.test_buzz_adapter import SELF_PUBKEY

    runner = _make_buzz_multiplex_runner(
        monkeypatch, extra={"allowed_users": ["someone"]}
    )
    _patch_buzz_registry(monkeypatch, allowed_users_env="")

    assert runner._is_user_authorized(_buzz_source("someone")) is False

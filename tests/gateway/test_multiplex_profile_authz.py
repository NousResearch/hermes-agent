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


# --- _auth_env scoped-miss fail-closed (#86905) -----------------------------
#
# In multiplex mode with a scope installed, a scoped miss must return the
# default instead of leaking os.environ (which holds the DEFAULT profile's
# allowlist). A leaked allowlist would skip the allow-all check and reject
# every sender on a secondary profile's bot (Feishu open_ids are app-scoped).


def test_auth_env_scoped_miss_does_not_leak_os_environ(tmp_path, monkeypatch):
    from gateway.authz_mixin import _auth_env

    import agent.secret_scope as ss

    # Default profile's process env holds an allowlist the scoped profile
    # does not have.
    monkeypatch.setenv("FEISHU_ALLOWED_USERS", "ou_default_view")
    (tmp_path / ".env").write_text(
        "GATEWAY_ALLOW_ALL_USERS=true\n", encoding="utf-8"
    )

    ss.set_multiplex_active(True)
    tok = ss.set_secret_scope(ss.build_profile_secret_scope(tmp_path))
    try:
        assert _auth_env("FEISHU_ALLOWED_USERS") == ""
        assert _auth_env("GATEWAY_ALLOW_ALL_USERS") == "true"
    finally:
        ss.reset_secret_scope(tok)
        ss.set_multiplex_active(False)


def test_auth_env_unscoped_still_reads_os_environ(monkeypatch):
    """Single-profile (multiplex off): legacy os.environ read unchanged."""
    from gateway.authz_mixin import _auth_env

    monkeypatch.setenv("FEISHU_ALLOWED_USERS", "ou_a,ou_b")
    assert _auth_env("FEISHU_ALLOWED_USERS") == "ou_a,ou_b"


def test_is_user_authorized_allow_all_not_skipped_by_leaked_allowlist(
    tmp_path, monkeypatch
):
    """End-to-end shape of #86905: the secondary profile's scope has only
    GATEWAY_ALLOW_ALL_USERS=true; the DEFAULT profile's FEISHU_ALLOWED_USERS
    lives in os.environ. The sender carries the secondary app's open_id
    (absent from the default allowlist) and must still be admitted via the
    allow-all flag — a leaked allowlist must not hijack the decision."""
    import agent.secret_scope as ss
    from gateway.authz_mixin import _auth_env
    from gateway.run import GatewayRunner

    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("FEISHU_ALLOWED_USERS", "ou_default_view")
    (tmp_path / ".env").write_text(
        "GATEWAY_ALLOW_ALL_USERS=true\n", encoding="utf-8"
    )

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner.adapters = {}
    runner._profile_adapters = {}
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = False

    ss.set_multiplex_active(True)
    tok = ss.set_secret_scope(ss.build_profile_secret_scope(tmp_path))
    try:
        assert _auth_env("FEISHU_ALLOWED_USERS") == ""  # no leak
        source = SessionSource(
            platform=Platform.FEISHU,
            user_id="ou_role_view",
            chat_id="dm-chat",
            user_name="owner",
            chat_type="dm",
            profile="role-codex",
        )
        assert runner._is_user_authorized(source) is True
    finally:
        ss.reset_secret_scope(tok)
        ss.set_multiplex_active(False)

"""Tests for the BasicAuthProvider plugin (username/password, scrypt, signed
tokens).

Loads the plugin module directly (it's a bundled backend plugin, not on the
import path as a package) and exercises the provider behaviour + the
``register(ctx)`` entry point's config/env resolution and skip reasons.
"""

from __future__ import annotations

import secrets
from unittest.mock import MagicMock

import pytest

import plugins.dashboard_auth.basic as basic_plugin
from hermes_cli.dashboard_auth import (
    InvalidCredentialsError,
    RefreshExpiredError,
    assert_protocol_compliance,
)


@pytest.fixture(scope="module")
def basic():
    return basic_plugin


@pytest.fixture(autouse=True)
def _clear_basic_env(monkeypatch):
    for var in (
        "HERMES_DASHBOARD_BASIC_AUTH_USERNAME",
        "HERMES_DASHBOARD_BASIC_AUTH_PASSWORD",
        "HERMES_DASHBOARD_BASIC_AUTH_PASSWORD_HASH",
        "HERMES_DASHBOARD_BASIC_AUTH_SECRET",
        "HERMES_DASHBOARD_BASIC_AUTH_TTL_SECONDS",
    ):
        monkeypatch.delenv(var, raising=False)


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


class TestPasswordHashing:
    def test_hash_then_verify_round_trips(self, basic):
        h = basic.hash_password("hunter2")
        assert h.startswith("scrypt$")
        assert basic._verify_password("hunter2", h)

    def test_wrong_password_fails(self, basic):
        h = basic.hash_password("hunter2")
        assert not basic._verify_password("wrong", h)

    def test_malformed_hash_returns_false(self, basic):
        assert not basic._verify_password("x", "not-a-valid-hash")
        assert not basic._verify_password("x", "bcrypt$wrong$scheme")

    def test_two_hashes_of_same_password_differ(self, basic):
        # Distinct random salts → distinct encoded hashes.
        assert basic.hash_password("pw") != basic.hash_password("pw")


# ---------------------------------------------------------------------------
# Provider behaviour
# ---------------------------------------------------------------------------


class TestProvider:
    def _make(self, basic, **kw):
        h = basic.hash_password("hunter2")
        return basic.BasicAuthProvider(
            username="admin",
            password_hash=h,
            secret=secrets.token_bytes(32),
            **kw,
        )

    def test_protocol_compliant(self, basic):
        assert assert_protocol_compliance(basic.BasicAuthProvider) is None

    def test_supports_password_true(self, basic):
        assert basic.BasicAuthProvider.supports_password is True

    def test_login_mints_session(self, basic):
        p = self._make(basic)
        s = p.complete_password_login(username="admin", password="hunter2")
        assert s.user_id == "admin"
        assert s.provider == "basic"
        assert s.access_token and s.refresh_token

    def test_bad_credentials_raise(self, basic):
        p = self._make(basic)
        for u, pw in [("admin", "wrong"), ("ghost", "hunter2"), ("", "")]:
            with pytest.raises(InvalidCredentialsError):
                p.complete_password_login(username=u, password=pw)

    def test_verify_round_trips_and_rejects_tamper(self, basic):
        p = self._make(basic)
        s = p.complete_password_login(username="admin", password="hunter2")
        assert p.verify_session(access_token=s.access_token) is not None
        assert p.verify_session(access_token="garbage") is None

    def test_access_token_not_accepted_as_refresh(self, basic):
        p = self._make(basic)
        s = p.complete_password_login(username="admin", password="hunter2")
        # A refresh token must not verify as an access token and vice
        # versa — the ``kind`` claim is enforced.
        assert p.verify_session(access_token=s.refresh_token) is None
        with pytest.raises(RefreshExpiredError):
            p.refresh_session(refresh_token=s.access_token)

    def test_refresh_round_trips(self, basic):
        p = self._make(basic)
        s = p.complete_password_login(username="admin", password="hunter2")
        r = p.refresh_session(refresh_token=s.refresh_token)
        assert r.user_id == "admin"
        assert p.verify_session(access_token=r.access_token) is not None

    def test_refresh_with_garbage_raises(self, basic):
        p = self._make(basic)
        with pytest.raises(RefreshExpiredError):
            p.refresh_session(refresh_token="garbage")

    def test_cross_secret_token_does_not_verify(self, basic):
        p1 = self._make(basic)
        p2 = self._make(basic)  # different random secret
        s = p1.complete_password_login(username="admin", password="hunter2")
        assert p2.verify_session(access_token=s.access_token) is None

    def test_revoke_is_silent(self, basic):
        p = self._make(basic)
        p.revoke_session(refresh_token="anything")  # must not raise

    def test_oauth_methods_raise_not_implemented(self, basic):
        p = self._make(basic)
        with pytest.raises(NotImplementedError):
            p.start_login(redirect_uri="https://x/auth/callback")
        with pytest.raises(NotImplementedError):
            p.complete_login(
                code="c", state="s", code_verifier="v", redirect_uri="r"
            )

    def test_construction_validates_inputs(self, basic):
        good_hash = basic.hash_password("pw")
        with pytest.raises(ValueError):
            basic.BasicAuthProvider(
                username="", password_hash=good_hash, secret=b"x" * 32
            )
        with pytest.raises(ValueError):
            basic.BasicAuthProvider(
                username="admin", password_hash="", secret=b"x" * 32
            )
        with pytest.raises(ValueError):
            basic.BasicAuthProvider(
                username="admin", password_hash=good_hash, secret=b"short"
            )


# ---------------------------------------------------------------------------
# register() entry point — config/env resolution + skip reasons
# ---------------------------------------------------------------------------


class TestRegister:
    def test_skips_when_no_username(self, basic, monkeypatch):
        monkeypatch.setattr(basic, "_load_config_basic_auth_section", lambda: {})
        ctx = MagicMock()
        basic.register(ctx)
        ctx.register_dashboard_auth_provider.assert_not_called()
        assert "username" in basic.LAST_SKIP_REASON

    def test_skips_when_username_but_no_password(self, basic, monkeypatch):
        monkeypatch.setenv("HERMES_DASHBOARD_BASIC_AUTH_USERNAME", "admin")
        monkeypatch.setattr(basic, "_load_config_basic_auth_section", lambda: {})
        ctx = MagicMock()
        basic.register(ctx)
        ctx.register_dashboard_auth_provider.assert_not_called()
        assert "password" in basic.LAST_SKIP_REASON

    def test_registers_with_env_plaintext_password(self, basic, monkeypatch):
        monkeypatch.setenv("HERMES_DASHBOARD_BASIC_AUTH_USERNAME", "admin")
        monkeypatch.setenv("HERMES_DASHBOARD_BASIC_AUTH_PASSWORD", "hunter2")
        monkeypatch.setattr(basic, "_load_config_basic_auth_section", lambda: {})
        ctx = MagicMock()
        basic.register(ctx)
        ctx.register_dashboard_auth_provider.assert_called_once()
        provider = ctx.register_dashboard_auth_provider.call_args.args[0]
        assert isinstance(provider, basic.BasicAuthProvider)
        # Round-trips: the registered provider authenticates the env creds.
        s = provider.complete_password_login(username="admin", password="hunter2")
        assert s.user_id == "admin"
        assert basic.LAST_SKIP_REASON == ""

    def test_registers_with_precomputed_hash(self, basic, monkeypatch):
        h = basic.hash_password("s3cret")
        monkeypatch.setattr(
            basic,
            "_load_config_basic_auth_section",
            lambda: {"username": "ops", "password_hash": h},
        )
        ctx = MagicMock()
        basic.register(ctx)
        ctx.register_dashboard_auth_provider.assert_called_once()
        provider = ctx.register_dashboard_auth_provider.call_args.args[0]
        assert provider.complete_password_login(
            username="ops", password="s3cret"
        ).user_id == "ops"

    def test_env_password_overrides_config(self, basic, monkeypatch):
        cfg_hash = basic.hash_password("config-pw")
        monkeypatch.setattr(
            basic,
            "_load_config_basic_auth_section",
            lambda: {"username": "admin", "password_hash": cfg_hash},
        )
        # Env plaintext should win over the config hash.
        monkeypatch.setenv("HERMES_DASHBOARD_BASIC_AUTH_PASSWORD", "env-pw")
        ctx = MagicMock()
        basic.register(ctx)
        provider = ctx.register_dashboard_auth_provider.call_args.args[0]
        # env password works ...
        assert provider.complete_password_login(
            username="admin", password="env-pw"
        )
        # ... and the config password no longer does.
        with pytest.raises(InvalidCredentialsError):
            provider.complete_password_login(username="admin", password="config-pw")

    def test_explicit_secret_makes_sessions_portable(self, basic, monkeypatch):
        # Two providers built from the SAME explicit secret accept each
        # other's tokens (the restart-/multi-worker-survival contract).
        shared = secrets.token_bytes(32).hex()
        monkeypatch.setattr(basic, "_load_config_basic_auth_section", lambda: {})
        monkeypatch.setenv("HERMES_DASHBOARD_BASIC_AUTH_USERNAME", "admin")
        monkeypatch.setenv("HERMES_DASHBOARD_BASIC_AUTH_PASSWORD", "hunter2")
        monkeypatch.setenv("HERMES_DASHBOARD_BASIC_AUTH_SECRET", shared)

        ctx1, ctx2 = MagicMock(), MagicMock()
        basic.register(ctx1)
        basic.register(ctx2)
        p1 = ctx1.register_dashboard_auth_provider.call_args.args[0]
        p2 = ctx2.register_dashboard_auth_provider.call_args.args[0]
        s = p1.complete_password_login(username="admin", password="hunter2")
        assert p2.verify_session(access_token=s.access_token) is not None

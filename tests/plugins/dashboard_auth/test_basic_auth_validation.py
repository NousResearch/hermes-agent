"""Regression coverage for dashboard basic-auth credential validation (#110069)."""

from unittest.mock import MagicMock

import pytest

import plugins.dashboard_auth.basic as basic


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


def test_validate_password_hash_accepts_generated_hash():
    encoded = basic.hash_password("hunter2")
    assert basic.validate_password_hash(encoded)


@pytest.mark.parametrize(
    "encoded",
    [
        "not-a-hash",
        "bcrypt$1$2$3$4$5",
        "scrypt$16384$8$1$missing-digest",
        "scrypt$16384$8$1$$",
        "scrypt$not-an-int$8$1$c2FsdA==$ZGs=",
    ],
)
def test_validate_password_hash_rejects_malformed_values(encoded):
    assert not basic.validate_password_hash(encoded)


def test_provider_constructor_rejects_malformed_hash():
    with pytest.raises(ValueError, match="password_hash is malformed"):
        basic.BasicAuthProvider(
            username="admin",
            password_hash="scrypt1638481shell-expanded",
            secret=b"x" * 32,
        )


def test_register_rejects_malformed_config_hash_instead_of_locking_out(monkeypatch):
    monkeypatch.setattr(
        basic,
        "_load_config_basic_auth_section",
        lambda: {
            "username": "admin",
            "password_hash": "scrypt1638481shell-expanded",
            "password": "",
        },
    )
    ctx = MagicMock()

    basic.register(ctx)

    ctx.register_dashboard_auth_provider.assert_not_called()
    assert "password_hash is malformed" in basic.LAST_SKIP_REASON
    assert "quote it" in basic.LAST_SKIP_REASON


def test_env_plaintext_password_still_overrides_malformed_config_hash(monkeypatch):
    monkeypatch.setattr(
        basic,
        "_load_config_basic_auth_section",
        lambda: {
            "username": "admin",
            "password_hash": "scrypt1638481shell-expanded",
        },
    )
    monkeypatch.setenv("HERMES_DASHBOARD_BASIC_AUTH_PASSWORD", "env-password")
    ctx = MagicMock()

    basic.register(ctx)

    provider = ctx.register_dashboard_auth_provider.call_args.args[0]
    session = provider.complete_password_login(username="admin", password="env-password")
    assert session.user_id == "admin"
    assert basic.LAST_SKIP_REASON == ""

"""Behavior tests for live Basic Auth credential rotation."""

import pytest
import yaml

from hermes_cli.dashboard_auth import InvalidCredentialsError
from plugins.dashboard_auth.basic import (
    BasicAuthProvider,
    _load_basic_auth_credentials,
    hash_password,
)


def _use_isolated_dashboard_auth_config(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for name in (
        "HERMES_DASHBOARD_BASIC_AUTH_USERNAME",
        "HERMES_DASHBOARD_BASIC_AUTH_PASSWORD_HASH",
        "HERMES_DASHBOARD_BASIC_AUTH_PASSWORD",
        "HERMES_DASHBOARD_BASIC_AUTH_SECRET",
    ):
        monkeypatch.delenv(name, raising=False)


def test_password_login_uses_rotated_config_without_restarting_provider(tmp_path, monkeypatch):
    """The next password login must observe a changed config.yaml credential."""
    _use_isolated_dashboard_auth_config(tmp_path, monkeypatch)

    config = {
        "dashboard": {
            "basic_auth": {
                "username": "old-admin",
                "password_hash": hash_password("old-password"),
                "secret": "test-secret-that-is-long-enough",
            }
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    credentials = _load_basic_auth_credentials()
    assert credentials is not None

    provider = BasicAuthProvider(
        username=credentials[0],
        password_hash=credentials[1],
        secret=credentials[2],
        ttl_seconds=credentials[3],
        credentials_loader=_load_basic_auth_credentials,
    )

    provider.complete_password_login(username="old-admin", password="old-password")

    config["dashboard"]["basic_auth"]["username"] = "new-admin"
    config["dashboard"]["basic_auth"]["password_hash"] = hash_password("new-password")
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(InvalidCredentialsError):
        provider.complete_password_login(username="old-admin", password="old-password")

    session = provider.complete_password_login(username="new-admin", password="new-password")
    assert session.user_id == "new-admin"


def test_unset_secret_keeps_existing_sessions_valid_across_logins(tmp_path, monkeypatch):
    """An implicit per-process secret must not rotate on every login."""
    _use_isolated_dashboard_auth_config(tmp_path, monkeypatch)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "dashboard": {
                    "basic_auth": {
                        "username": "admin",
                        "password_hash": hash_password("password"),
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    credentials = _load_basic_auth_credentials()
    assert credentials is not None
    assert credentials[2] is None
    provider = BasicAuthProvider(
        username=credentials[0],
        password_hash=credentials[1],
        secret=credentials[2],
        ttl_seconds=credentials[3],
        credentials_loader=_load_basic_auth_credentials,
    )

    first = provider.complete_password_login(username="admin", password="password")
    provider.complete_password_login(username="admin", password="password")

    assert provider.verify_session(access_token=first.access_token) is not None


def test_explicit_secret_rotation_invalidates_existing_sessions(tmp_path, monkeypatch):
    """An operator-supplied new secret deliberately invalidates old tokens."""
    _use_isolated_dashboard_auth_config(tmp_path, monkeypatch)
    config = {
        "dashboard": {
            "basic_auth": {
                "username": "admin",
                "password_hash": hash_password("password"),
                "secret": "first-test-secret-is-long-enough",
            }
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    credentials = _load_basic_auth_credentials()
    assert credentials is not None
    provider = BasicAuthProvider(
        username=credentials[0],
        password_hash=credentials[1],
        secret=credentials[2],
        ttl_seconds=credentials[3],
        credentials_loader=_load_basic_auth_credentials,
    )
    first = provider.complete_password_login(username="admin", password="password")

    config["dashboard"]["basic_auth"]["secret"] = "second-test-secret-is-long-enough"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    provider.complete_password_login(username="admin", password="password")

    assert provider.verify_session(access_token=first.access_token) is None


def test_invalid_secret_update_keeps_existing_credentials_available(tmp_path, monkeypatch):
    """A malformed live edit must not replace a working auth provider."""
    _use_isolated_dashboard_auth_config(tmp_path, monkeypatch)
    config = {
        "dashboard": {
            "basic_auth": {
                "username": "admin",
                "password_hash": hash_password("password"),
                "secret": "stable-test-secret-is-long-enough",
            }
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    credentials = _load_basic_auth_credentials()
    assert credentials is not None
    provider = BasicAuthProvider(
        username=credentials[0],
        password_hash=credentials[1],
        secret=credentials[2],
        ttl_seconds=credentials[3],
        credentials_loader=_load_basic_auth_credentials,
    )

    config["dashboard"]["basic_auth"]["secret"] = "short"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    session = provider.complete_password_login(username="admin", password="password")
    assert provider.verify_session(access_token=session.access_token) is not None

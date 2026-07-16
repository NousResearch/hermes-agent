"""Behavior tests for live Basic Auth credential rotation."""

import pytest
import yaml

from hermes_cli.dashboard_auth import InvalidCredentialsError
from plugins.dashboard_auth.basic import (
    BasicAuthProvider,
    _load_basic_auth_credentials,
    hash_password,
)


def test_password_login_uses_rotated_config_without_restarting_provider(tmp_path, monkeypatch):
    """The next password login must observe a changed config.yaml credential."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_DASHBOARD_BASIC_AUTH_USERNAME", raising=False)
    monkeypatch.delenv("HERMES_DASHBOARD_BASIC_AUTH_PASSWORD_HASH", raising=False)
    monkeypatch.delenv("HERMES_DASHBOARD_BASIC_AUTH_PASSWORD", raising=False)

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

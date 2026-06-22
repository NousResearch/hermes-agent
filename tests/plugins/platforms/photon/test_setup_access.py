"""Tests for `hermes photon setup`'s access auto-configuration.

`_autoconfigure_access` allowlists the operator and points the cron home
channel at their DM, writing to the per-test ~/.hermes/.env (the hermetic
HERMES_HOME fixture isolates this). It must fill only unset keys so a re-run
never clobbers a hand-tuned allowlist.
"""
from __future__ import annotations

import argparse

import pytest

from hermes_cli.config import get_env_value, save_env_value
from plugins.platforms.photon.adapter import _env_enablement
from plugins.platforms.photon import cli


def test_autoconfigure_access_fills_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PHOTON_ALLOWED_USERS", raising=False)
    monkeypatch.delenv("PHOTON_HOME_CHANNEL", raising=False)

    cli._autoconfigure_access("+15551234567")

    assert get_env_value("PHOTON_ALLOWED_USERS") == "+15551234567"
    assert get_env_value("PHOTON_HOME_CHANNEL") == "+15551234567"


def test_autoconfigure_access_preserves_existing_allowlist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PHOTON_ALLOWED_USERS", raising=False)
    monkeypatch.delenv("PHOTON_HOME_CHANNEL", raising=False)
    # A hand-tuned allowlist already in place must survive a setup re-run.
    save_env_value("PHOTON_ALLOWED_USERS", "+19998887777,+15551112222")

    cli._autoconfigure_access("+15551234567")

    assert get_env_value("PHOTON_ALLOWED_USERS") == "+19998887777,+15551112222"
    # The still-unset home channel is filled.
    assert get_env_value("PHOTON_HOME_CHANNEL") == "+15551234567"


def test_env_enablement_seeds_home_channel(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PHOTON_PROJECT_ID", "project_123")
    monkeypatch.setenv("PHOTON_PROJECT_SECRET", "secret_123")
    monkeypatch.setenv("PHOTON_HOME_CHANNEL", "+15551234567")
    monkeypatch.setenv("PHOTON_HOME_CHANNEL_NAME", "Primary DM")

    seed = _env_enablement()

    assert seed is not None
    assert seed["home_channel"] == {
        "chat_id": "+15551234567",
        "name": "Primary DM",
    }


def test_env_enablement_home_channel_defaults_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PHOTON_PROJECT_ID", "project_123")
    monkeypatch.setenv("PHOTON_PROJECT_SECRET", "secret_123")
    monkeypatch.setenv("PHOTON_HOME_CHANNEL", "+15551234567")
    monkeypatch.delenv("PHOTON_HOME_CHANNEL_NAME", raising=False)

    seed = _env_enablement()

    assert seed is not None
    assert seed["home_channel"] == {
        "chat_id": "+15551234567",
        "name": "Home",
    }


def test_setup_hint_uses_gateway_service_command(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr(cli.photon_auth, "load_photon_token", lambda: "token")
    # The dashboard id *is* the Spectrum project id (ids unified), so setup no
    # longer enables Spectrum or fetches a separate spectrumProjectId — it
    # reuses this id directly.
    monkeypatch.setattr(cli.photon_auth, "load_dashboard_project_id", lambda: "dashboard")
    # No existing credentials — first-time setup path.
    monkeypatch.setattr(
        cli.photon_auth, "load_project_credentials", lambda: (None, None),
    )
    monkeypatch.setattr(
        cli.photon_auth,
        "regenerate_project_secret",
        lambda token, dashboard_id: "secret_123",
    )
    monkeypatch.setattr(cli.photon_auth, "store_project_credentials", lambda **kwargs: None)
    monkeypatch.setattr(
        cli.photon_auth,
        "register_user_if_absent",
        lambda *args, **kwargs: ({"id": "user_123", "phoneNumber": "+155****4567"}, True),
    )
    monkeypatch.setattr(cli.photon_auth, "user_assigned_line", lambda user: "+155****4321")
    monkeypatch.setattr(cli.photon_auth, "store_user_numbers", lambda **kwargs: None)
    monkeypatch.setattr(cli, "_install_sidecar", lambda: 0)

    rc = cli._cmd_setup(
        argparse.Namespace(
            project_name=None,
            phone="+155****4567",
            first_name=None,
            last_name=None,
            email=None,
            no_browser=True,
            skip_sidecar_install=False,
        )
    )

    assert rc == 0
    out = capsys.readouterr().out
    assert "Start the gateway:  hermes gateway start" in out
    assert "--platform photon" not in out
    assert "new secret saved" in out
    assert "restart it so the sidecar" in out


def test_setup_reuses_valid_existing_secret(
    monkeypatch: pytest.MonkeyPatch, capsys,
) -> None:
    """Re-running setup with a valid existing secret must NOT regenerate it."""
    regenerate_called = False

    def _fake_regenerate(token, dashboard_id):
        nonlocal regenerate_called
        regenerate_called = True
        return "new_secret"

    monkeypatch.setattr(cli.photon_auth, "load_photon_token", lambda: "token")
    monkeypatch.setattr(cli.photon_auth, "load_dashboard_project_id", lambda: "dashboard")
    monkeypatch.setattr(
        cli.photon_auth,
        "load_project_credentials",
        lambda: ("dashboard", "existing_secret"),
    )
    # list_users succeeds — existing secret is valid.
    monkeypatch.setattr(
        cli.photon_auth, "list_users", lambda pid, secret: [],
    )
    monkeypatch.setattr(cli.photon_auth, "regenerate_project_secret", _fake_regenerate)
    monkeypatch.setattr(cli.photon_auth, "store_project_credentials", lambda **kwargs: None)
    monkeypatch.setattr(
        cli.photon_auth,
        "register_user_if_absent",
        lambda *args, **kwargs: ({"id": "user_123", "phoneNumber": "+155****4567"}, True),
    )
    monkeypatch.setattr(cli.photon_auth, "user_assigned_line", lambda user: "+155****4321")
    monkeypatch.setattr(cli.photon_auth, "store_user_numbers", lambda **kwargs: None)
    monkeypatch.setattr(cli, "_install_sidecar", lambda: 0)

    rc = cli._cmd_setup(
        argparse.Namespace(
            project_name=None,
            phone="+155****4567",
            first_name=None,
            last_name=None,
            email=None,
            no_browser=True,
            skip_sidecar_install=False,
        )
    )

    assert rc == 0
    assert not regenerate_called, "regenerate_project_secret must not be called when existing creds are valid"
    out = capsys.readouterr().out
    assert "existing credentials valid" in out
    assert "restart" not in out.lower()


def test_setup_regenerates_when_existing_secret_invalid(
    monkeypatch: pytest.MonkeyPatch, capsys,
) -> None:
    """When existing credentials are invalid, setup must regenerate."""
    monkeypatch.setattr(cli.photon_auth, "load_photon_token", lambda: "token")
    monkeypatch.setattr(cli.photon_auth, "load_dashboard_project_id", lambda: "dashboard")
    monkeypatch.setattr(
        cli.photon_auth,
        "load_project_credentials",
        lambda: ("dashboard", "stale_secret"),
    )
    # list_users fails — existing secret is invalid.
    monkeypatch.setattr(
        cli.photon_auth,
        "list_users",
        lambda pid, secret: (_ for _ in ()).throw(RuntimeError("auth failed")),
    )
    monkeypatch.setattr(
        cli.photon_auth,
        "regenerate_project_secret",
        lambda token, dashboard_id: "new_secret",
    )
    monkeypatch.setattr(cli.photon_auth, "store_project_credentials", lambda **kwargs: None)
    monkeypatch.setattr(
        cli.photon_auth,
        "register_user_if_absent",
        lambda *args, **kwargs: ({"id": "user_123", "phoneNumber": "+155****4567"}, True),
    )
    monkeypatch.setattr(cli.photon_auth, "user_assigned_line", lambda user: "+155****4321")
    monkeypatch.setattr(cli.photon_auth, "store_user_numbers", lambda **kwargs: None)
    monkeypatch.setattr(cli, "_install_sidecar", lambda: 0)

    rc = cli._cmd_setup(
        argparse.Namespace(
            project_name=None,
            phone="+155****4567",
            first_name=None,
            last_name=None,
            email=None,
            no_browser=True,
            skip_sidecar_install=False,
        )
    )

    assert rc == 0
    out = capsys.readouterr().out
    assert "new secret saved" in out
    assert "restart it so the sidecar" in out

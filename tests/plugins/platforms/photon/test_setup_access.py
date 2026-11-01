"""Tests for `hermes photon setup`'s access auto-configuration.

`_autoconfigure_access` allowlists the operator and points the cron home
channel at their DM, writing to the per-test ~/.hermes/.env (the hermetic
HERMES_HOME fixture isolates this). It must fill only unset keys so a re-run
never clobbers a hand-tuned allowlist.
"""
from __future__ import annotations

import argparse

import pytest

from hermes_cli.config import get_env_value
from plugins.platforms.photon.adapter import _env_enablement
from plugins.platforms.photon import cli


def test_autoconfigure_access_fills_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PHOTON_ALLOWED_USERS", raising=False)
    monkeypatch.delenv("PHOTON_HOME_CHANNEL", raising=False)

    cli._autoconfigure_access("+15551234567")

    assert get_env_value("PHOTON_ALLOWED_USERS") == "+15551234567"
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
    # Token validation (added for #72763) would otherwise hit the network.
    monkeypatch.setattr(cli.photon_auth, "check_photon_token_valid", lambda token: True)
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
    # Token validation (added for #72763) would otherwise hit the network.
    monkeypatch.setattr(cli.photon_auth, "check_photon_token_valid", lambda token: True)
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


def test_setup_without_opt_in_does_not_report_complete(
    monkeypatch: pytest.MonkeyPatch, capsys,
) -> None:
    """A number that never texted its assigned line cannot route: setup must not end on
    "complete" (non-TTY run, so the opt-in wait is skipped rather than blocking)."""
    monkeypatch.setattr(cli.photon_auth, "load_photon_token", lambda: "token")
    monkeypatch.setattr(cli.photon_auth, "check_photon_token_valid", lambda token: True)
    monkeypatch.setattr(cli.photon_auth, "load_dashboard_project_id", lambda: "dashboard")
    monkeypatch.setattr(cli.photon_auth, "load_project_credentials", lambda: ("dashboard", "s"))
    monkeypatch.setattr(cli.photon_auth, "list_users", lambda pid, secret: [])
    monkeypatch.setattr(cli.photon_auth, "store_project_credentials", lambda **kwargs: None)
    monkeypatch.setattr(cli.photon_auth, "register_user_if_absent", lambda *a, **k: (
        {"id": "u1", "phoneNumber": "+15551234567", "assignedPhoneNumber": "+16282679185", "meta": {}}, True))
    monkeypatch.setattr(cli.photon_auth, "store_user_numbers", lambda **kwargs: None)
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False, raising=False)

    rc = cli._cmd_setup(argparse.Namespace(
        project_name=None, phone="+15551234567", first_name=None, last_name=None,
        email=None, no_browser=True, skip_sidecar_install=True))

    out = capsys.readouterr().out
    assert rc == 0
    assert "setup complete" not in out
    assert "routing is NOT live" in out and "+16282679185" in out


@pytest.mark.parametrize("dedicated_line", ["+16282679185", None])
def test_opt_in_wait_always_names_a_line_to_text(
    monkeypatch: pytest.MonkeyPatch, capsys, dedicated_line,
) -> None:
    """A row with no assignedPhoneNumber: the project-line lookup runs BEFORE the wait, and
    with no line at all the wait is skipped (the user would have nothing to text)."""
    monkeypatch.setattr(cli.photon_auth, "load_photon_token", lambda: "token")
    monkeypatch.setattr(cli.photon_auth, "check_photon_token_valid", lambda token: True)
    monkeypatch.setattr(cli.photon_auth, "load_dashboard_project_id", lambda: "dashboard")
    monkeypatch.setattr(cli.photon_auth, "load_project_credentials", lambda: ("dashboard", "s"))
    monkeypatch.setattr(cli.photon_auth, "list_users", lambda pid, secret: [])
    monkeypatch.setattr(cli.photon_auth, "store_project_credentials", lambda **kwargs: None)
    monkeypatch.setattr(cli.photon_auth, "register_user_if_absent", lambda *a, **k: (
        {"id": "u1", "phoneNumber": "+15551234567", "meta": {}}, True))
    monkeypatch.setattr(cli.photon_auth, "get_imessage_line", lambda token, pid: (
        {"phoneNumber": dedicated_line} if dedicated_line else None))
    monkeypatch.setattr(cli.photon_auth, "store_user_numbers", lambda **kwargs: None)
    waited: list = []
    monkeypatch.setattr(cli, "_wait_for_opt_in", lambda d, s, phone, line: waited.append(line))

    rc = cli._cmd_setup(argparse.Namespace(
        project_name=None, phone="+15551234567", first_name=None, last_name=None,
        email=None, no_browser=True, skip_sidecar_install=True))

    out = capsys.readouterr().out
    assert rc == 0 and "setup complete" not in out
    if dedicated_line:
        assert waited == [dedicated_line]
    else:
        assert waited == [] and "skipping the opt-in wait" in out and "hermes photon status" in out

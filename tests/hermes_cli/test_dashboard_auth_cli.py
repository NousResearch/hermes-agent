"""Tests for dashboard native auth CLI helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace


def test_configure_dashboard_auth_writes_hash_not_plaintext(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    from hermes_cli.config import load_config
    from hermes_cli.dashboard_auth import configure_dashboard_auth
    from hermes_cli.web_server import _verify_dashboard_password

    result = configure_dashboard_auth(username="mauri", password="moon-secret")

    assert result["enabled"] is True
    assert result["username"] == "mauri"
    assert result["password_hash"].startswith("pbkdf2_sha256$")
    assert "moon-secret" not in result["password_hash"]
    assert _verify_dashboard_password("moon-secret", result["password_hash"])

    cfg = load_config()
    assert cfg["dashboard"]["auth"]["enabled"] is True
    assert cfg["dashboard"]["auth"]["username"] == "mauri"
    assert cfg["dashboard"]["auth"]["password_hash"] == result["password_hash"]


def test_disable_dashboard_auth_preserves_username_and_clears_hash(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    from hermes_cli.dashboard_auth import configure_dashboard_auth, disable_dashboard_auth, dashboard_auth_status

    configure_dashboard_auth(username="mauri", password="moon-secret")
    disabled = disable_dashboard_auth()

    assert disabled["enabled"] is False
    assert disabled["username"] == "mauri"
    assert disabled["password_hash"] == ""

    status = dashboard_auth_status()
    assert status == {"enabled": False, "configured": False, "username": "mauri"}


def test_dashboard_auth_status_reports_enabled_and_configured(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    from hermes_cli.dashboard_auth import configure_dashboard_auth, dashboard_auth_status

    configure_dashboard_auth(username="mauri", password="moon-secret")

    assert dashboard_auth_status() == {
        "enabled": True,
        "configured": True,
        "username": "mauri",
    }


def test_configure_dashboard_auth_rejects_empty_username_or_password(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    from hermes_cli.dashboard_auth import configure_dashboard_auth

    for username, password in [("", "moon-secret"), ("mauri", "")]:
        try:
            configure_dashboard_auth(username=username, password=password)
        except ValueError as exc:
            assert "dashboard auth" in str(exc)
        else:
            raise AssertionError("expected ValueError")


def test_cmd_dashboard_auth_status_prints_safe_state(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    from hermes_cli.dashboard_auth import configure_dashboard_auth
    from hermes_cli.main import cmd_dashboard

    configure_dashboard_auth(username="mauri", password="moon-secret")

    cmd_dashboard(SimpleNamespace(auth_action="status"))

    out = capsys.readouterr().out
    assert "Dashboard native auth: enabled" in out
    assert "Username: mauri" in out
    assert "moon-secret" not in out
    assert "pbkdf2_sha256" not in out


def test_cmd_dashboard_auth_setup_reads_password_env(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("HERMES_DASHBOARD_PASSWORD", "moon-secret")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    from hermes_cli.config import load_config
    from hermes_cli.main import cmd_dashboard
    from hermes_cli.web_server import _verify_dashboard_password

    cmd_dashboard(
        SimpleNamespace(
            auth_action="setup",
            username="mauri",
            password_env="HERMES_DASHBOARD_PASSWORD",
        )
    )

    out = capsys.readouterr().out
    assert "Dashboard native auth enabled" in out
    cfg = load_config()
    auth = cfg["dashboard"]["auth"]
    assert auth["enabled"] is True
    assert auth["username"] == "mauri"
    assert _verify_dashboard_password("moon-secret", auth["password_hash"])

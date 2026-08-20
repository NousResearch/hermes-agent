from __future__ import annotations

from argparse import Namespace
from pathlib import Path


def _bitwarden_config(*, enabled: bool = True, server_url: str = "") -> dict:
    return {
        "secrets": {
            "bitwarden": {
                "enabled": enabled,
                "access_token_env": "BWS_ACCESS_TOKEN",
                "project_id": "proj-123",
                "server_url": server_url,
                "cache_ttl_seconds": 300,
                "override_existing": True,
                "auto_install": True,
            }
        }
    }


def test_status_surfaces_failed_token_validation(monkeypatch, capsys):
    from hermes_cli import secrets_cli

    seen = {}

    monkeypatch.setattr(
        secrets_cli,
        "load_config",
        lambda: _bitwarden_config(server_url="https://vault.bitwarden.eu"),
    )
    monkeypatch.setenv("BWS_ACCESS_TOKEN", "0.invalid-token")
    monkeypatch.setattr(
        secrets_cli.bw,
        "find_bws",
        lambda install_if_missing=False: Path("/tmp/bws"),
    )
    monkeypatch.setattr(secrets_cli, "_bws_version", lambda _binary: "bws v2.1.0")

    def _fake_list_projects(binary, token, console, *, server_url=""):
        seen["binary"] = binary
        seen["token"] = token
        seen["server_url"] = server_url
        console.print("  bws project list failed: Doesn't contain a decryption key")
        console.print(
            "  This usually means the access token is wrong or revoked. "
            "Double-check it in the Bitwarden web app."
        )
        return None

    monkeypatch.setattr(secrets_cli, "_list_projects", _fake_list_projects)

    assert secrets_cli.cmd_status(Namespace()) == 0

    out = capsys.readouterr().out
    assert "Token in env" in out
    assert "Token validation" in out
    assert "failed" in out
    assert "Doesn't contain a decryption key" in out
    assert "wrong or revoked" in out
    assert seen == {
        "binary": Path("/tmp/bws"),
        "token": "0.invalid-token",
        "server_url": "https://vault.bitwarden.eu",
    }


def test_status_warns_when_token_does_not_look_like_bsm_token(monkeypatch, capsys):
    from hermes_cli import secrets_cli

    monkeypatch.setattr(secrets_cli, "load_config", lambda: _bitwarden_config())
    monkeypatch.setenv("BWS_ACCESS_TOKEN", "not-a-bitwarden-token")
    monkeypatch.setattr(
        secrets_cli.bw,
        "find_bws",
        lambda install_if_missing=False: Path("/tmp/bws"),
    )
    monkeypatch.setattr(secrets_cli, "_bws_version", lambda _binary: "bws v2.1.0")
    monkeypatch.setattr(
        secrets_cli,
        "_list_projects",
        lambda binary, token, console, *, server_url="": [],
    )

    assert secrets_cli.cmd_status(Namespace()) == 0

    out = capsys.readouterr().out
    assert "Token validation" in out
    assert "passed" in out
    assert "doesn't start with '0.'" in out


def test_status_marks_validation_as_not_checked_without_bws_binary(monkeypatch, capsys):
    from hermes_cli import secrets_cli

    monkeypatch.setattr(secrets_cli, "load_config", lambda: _bitwarden_config())
    monkeypatch.setenv("BWS_ACCESS_TOKEN", "0.token-present")
    monkeypatch.setattr(
        secrets_cli.bw,
        "find_bws",
        lambda install_if_missing=False: None,
    )

    assert secrets_cli.cmd_status(Namespace()) == 0

    out = capsys.readouterr().out
    assert "Token validation" in out
    assert "not checked" in out
    assert "bws not installed" in out


# ---------------------------------------------------------------------------
# A configured server_url reaches bws through the CLI's own project-list path,
# which does not go through fetch_bitwarden_secrets — so validation cannot be
# left to fetch callers.  These run the real _list_projects and inspect the
# environment the bws child would have been given.
# ---------------------------------------------------------------------------


def _capture_bws_env(monkeypatch, *, returncode: int = 0, stdout: str = "[]"):
    """Let _list_projects run for real; capture the env handed to bws."""
    from unittest import mock

    from hermes_cli import secrets_cli

    captured: dict = {}

    def _fake_run(cmd, **kwargs):
        captured.update(kwargs["env"])
        return mock.Mock(returncode=returncode, stdout=stdout, stderr="")

    monkeypatch.setattr(secrets_cli.subprocess, "run", _fake_run)
    return captured


def test_status_drops_invalid_configured_server_url(monkeypatch, capsys):
    """`hermes secrets bitwarden status` must not send the token to http://."""
    from hermes_cli import secrets_cli

    monkeypatch.setattr(
        secrets_cli,
        "load_config",
        lambda: _bitwarden_config(server_url="http://attacker.example"),
    )
    monkeypatch.setenv("BWS_ACCESS_TOKEN", "0.token-present")
    monkeypatch.setattr(
        secrets_cli.bw,
        "find_bws",
        lambda install_if_missing=False: Path("/tmp/bws"),
    )
    monkeypatch.setattr(secrets_cli, "_bws_version", lambda _binary: "bws v2.1.0")
    env = _capture_bws_env(monkeypatch)

    assert secrets_cli.cmd_status(Namespace()) == 0

    assert env["BWS_ACCESS_TOKEN"] == "0.token-present"
    assert "BWS_SERVER_URL" not in env
    out = capsys.readouterr().out
    assert "Ignoring the configured Bitwarden server_url" in out


def test_status_keeps_valid_configured_server_url(monkeypatch):
    """The legitimate self-hosted / EU case keeps working."""
    from hermes_cli import secrets_cli

    monkeypatch.setattr(
        secrets_cli,
        "load_config",
        lambda: _bitwarden_config(server_url="https://vault.bitwarden.eu"),
    )
    monkeypatch.setenv("BWS_ACCESS_TOKEN", "0.token-present")
    monkeypatch.setattr(
        secrets_cli.bw,
        "find_bws",
        lambda install_if_missing=False: Path("/tmp/bws"),
    )
    monkeypatch.setattr(secrets_cli, "_bws_version", lambda _binary: "bws v2.1.0")
    env = _capture_bws_env(monkeypatch)

    assert secrets_cli.cmd_status(Namespace()) == 0

    assert env["BWS_SERVER_URL"] == "https://vault.bitwarden.eu"


def test_token_rotation_drops_invalid_configured_server_url(monkeypatch, capsys):
    """Token verification probes Bitwarden too — same endpoint rules apply."""
    from hermes_cli import secrets_cli

    monkeypatch.setattr(
        secrets_cli,
        "load_config",
        lambda: _bitwarden_config(server_url="http://attacker.example"),
    )
    monkeypatch.setattr(
        secrets_cli.bw,
        "find_bws",
        lambda install_if_missing=True: Path("/tmp/bws"),
    )
    monkeypatch.setattr(secrets_cli, "save_env_value", lambda *_a, **_kw: None)
    monkeypatch.setattr(secrets_cli, "get_env_path", lambda: Path("/tmp/.env"))
    monkeypatch.setattr(secrets_cli.bw, "clear_caches", lambda: None)
    env = _capture_bws_env(monkeypatch)

    rc = secrets_cli.cmd_token(
        Namespace(access_token="0.new-token", no_verify=False)
    )

    assert rc == 0
    assert env["BWS_ACCESS_TOKEN"] == "0.new-token"
    assert "BWS_SERVER_URL" not in env
    out = capsys.readouterr().out
    assert "Ignoring the configured Bitwarden server_url" in out

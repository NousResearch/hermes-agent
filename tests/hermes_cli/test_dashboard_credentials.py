"""Tests for ``hermes dashboard credentials``."""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest
import yaml

from hermes_cli.subcommands.dashboard import build_dashboard_parser
from plugins.dashboard_auth.basic import BasicAuthProvider, hash_password


def _parser(handler):
    parser = argparse.ArgumentParser()
    build_dashboard_parser(
        parser.add_subparsers(dest="command"),
        cmd_dashboard=lambda _args: None,
        cmd_dashboard_register=lambda _args: None,
        cmd_dashboard_credentials=handler,
    )
    return parser


def test_credentials_subcommand_routes_to_handler():
    def handler(args):
        return args

    args = _parser(handler).parse_args(["dashboard", "credentials"])
    assert args.func is handler


def test_credentials_subcommand_is_absent_without_handler(capsys):
    parser = argparse.ArgumentParser()
    build_dashboard_parser(
        parser.add_subparsers(dest="command"),
        cmd_dashboard=lambda _args: None,
        cmd_dashboard_register=lambda _args: None,
    )

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["dashboard", "credentials"])

    assert exc_info.value.code == 2
    assert "invalid choice: 'credentials'" in capsys.readouterr().err


def _write_config(home, password="old-password"):
    (home / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "dashboard": {
                    "basic_auth": {
                        "username": "old-user",
                        "password_hash": hash_password(password),
                        "secret": "s" * 32,
                    }
                },
                "plugins": {"disabled": ["basic", "other"]},
            }
        ),
        encoding="utf-8",
    )


def test_credentials_declines_change_without_touching_config(
    tmp_path, monkeypatch, capsys
):
    from hermes_cli.dashboard_credentials import cmd_dashboard_credentials

    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    _write_config(home)
    original = (home / "config.yaml").read_text()
    monkeypatch.setattr("builtins.input", lambda _prompt: "n")

    cmd_dashboard_credentials(SimpleNamespace())

    assert (home / "config.yaml").read_text() == original
    assert "cancelled" in capsys.readouterr().out.lower()


def test_credentials_verifies_old_password_then_rotates(
    tmp_path, monkeypatch, capsys
):
    from hermes_cli.dashboard_credentials import cmd_dashboard_credentials

    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    _write_config(home)
    answers = iter(["old-password", "new-password", "new-password"])
    input_answers = iter(["y", "new-user"])
    prompts = []

    def fake_getpass(prompt):
        prompts.append(prompt)
        return next(answers)

    def fake_input(prompt):
        prompts.append(prompt)
        return next(input_answers)

    monkeypatch.setattr("getpass.getpass", fake_getpass)
    monkeypatch.setattr("builtins.input", fake_input)

    cmd_dashboard_credentials(SimpleNamespace())

    cfg = yaml.safe_load((home / "config.yaml").read_text())
    basic = cfg["dashboard"]["basic_auth"]
    assert basic["username"] == "new-user"
    assert not basic.get("password", "")
    assert basic["password_hash"].startswith("scrypt$")
    assert basic["secret"] != "s" * 32
    assert cfg["plugins"]["disabled"] == ["other"]
    provider = BasicAuthProvider(
        username=basic["username"],
        password_hash=basic["password_hash"],
        secret=basic["secret"].encode(),
    )
    assert provider.complete_password_login(
        username="new-user", password="new-password"
    ).user_id == "new-user"
    output = capsys.readouterr().out.lower()
    assert "keeps credentials in memory" in output
    assert "after restart, existing dashboard sessions will be invalidated" in output
    assert "same command or service manager" in output
    assert prompts == [
        "Change username and password? [y/N]: ",
        "Old password: ",
        "New username [old-user]: ",
        "New password: ",
        "Confirm new password: ",
    ]


def test_credentials_initial_setup_does_not_require_old_password(
    tmp_path, monkeypatch
):
    from hermes_cli.dashboard_credentials import cmd_dashboard_credentials

    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text("{}\n", encoding="utf-8")
    input_answers = iter(["first-user"])
    getpass_answers = iter(["first-password", "first-password"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(input_answers))
    monkeypatch.setattr("getpass.getpass", lambda _prompt: next(getpass_answers))

    cmd_dashboard_credentials(SimpleNamespace())

    basic = yaml.safe_load((home / "config.yaml").read_text())["dashboard"][
        "basic_auth"
    ]
    assert basic["username"] == "first-user"
    assert basic["password_hash"].startswith("scrypt$")


def test_credentials_rejects_mismatched_new_password_without_writing(
    tmp_path, monkeypatch
):
    from hermes_cli.dashboard_credentials import cmd_dashboard_credentials

    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    _write_config(home)
    original = (home / "config.yaml").read_text()
    input_answers = iter(["y", "new-user"])
    getpass_answers = iter(["old-password", "new-password", "different-password"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(input_answers))
    monkeypatch.setattr("getpass.getpass", lambda _prompt: next(getpass_answers))

    with pytest.raises(SystemExit, match="Passwords do not match"):
        cmd_dashboard_credentials(SimpleNamespace())

    assert (home / "config.yaml").read_text() == original


def test_credentials_rejects_wrong_old_password_without_writing(
    tmp_path, monkeypatch
):
    from hermes_cli.dashboard_credentials import cmd_dashboard_credentials

    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    _write_config(home, password="correct-old-password")
    original = (home / "config.yaml").read_text()
    monkeypatch.setattr("getpass.getpass", lambda _prompt: "wrong-old-password")
    monkeypatch.setattr("builtins.input", lambda _prompt: "y")

    with pytest.raises(SystemExit, match="old password"):
        cmd_dashboard_credentials(SimpleNamespace())
    assert (home / "config.yaml").read_text() == original


def test_credentials_refuses_environment_override(monkeypatch):
    from hermes_cli.dashboard_credentials import cmd_dashboard_credentials

    monkeypatch.setenv("HERMES_DASHBOARD_BASIC_AUTH_PASSWORD", "env-secret")
    with pytest.raises(SystemExit, match="environment variable"):
        cmd_dashboard_credentials(SimpleNamespace())


def test_basic_provider_exposes_public_password_verifier():
    from plugins.dashboard_auth.basic import verify_password

    encoded = hash_password("secret")
    assert verify_password("secret", encoded)
    assert not verify_password("wrong", encoded)

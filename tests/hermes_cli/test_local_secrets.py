"""Behavior contracts for terminal-only local secret capture."""

from __future__ import annotations

import argparse
import io

import pytest


def _parser():
    from hermes_cli.subcommands.secrets import build_secrets_parser

    parser = argparse.ArgumentParser()
    build_secrets_parser(parser.add_subparsers(dest="command"))
    return parser


def test_set_never_accepts_secret_in_argv_and_hidden_flow_never_prints_value(
        tmp_path, monkeypatch, capsys):
    parser = _parser()
    argv_args = parser.parse_args(
        ["secrets", "set", "OPENAI_API_KEY", "must-not-be-argv"])
    assert argv_args.func(argv_args) == 2
    argv_output = capsys.readouterr()
    assert "must-not-be-argv" not in argv_output.out + argv_output.err

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(tmp_path / "managed"))
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("hermes_cli.local_secrets.getpass.getpass", lambda _prompt: "super-secret-value")
    args = parser.parse_args(["secrets", "set", "OPENAI_API_KEY"])

    assert args.func(args) == 0
    assert "OPENAI_API_KEY=super-secret-value" in (tmp_path / ".env").read_text(encoding="utf-8")
    output = capsys.readouterr()
    assert "super-secret-value" not in output.out + output.err


@pytest.mark.parametrize("from_stdin", [False, True])
def test_set_rejects_non_ascii_without_disclosing_input(
        tmp_path, monkeypatch, capsys, from_stdin):
    parser = _parser()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(tmp_path / "managed"))
    secret = "\u5bc6\u7801"
    command = ["secrets", "set", "REVIEW_SECRET"]
    if from_stdin:
        command.append("--stdin")
        monkeypatch.setattr("sys.stdin", io.StringIO(secret + "\n"))
    else:
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("hermes_cli.local_secrets.getpass.getpass", lambda _prompt: secret)

    args = parser.parse_args(command)
    assert args.func(args) == 1
    output = capsys.readouterr()
    assert secret not in output.out + output.err
    assert not (tmp_path / ".env").exists()


@pytest.mark.parametrize("from_stdin", [False, True])
def test_set_rejects_nul_without_disclosing_or_persisting_input(
        tmp_path, monkeypatch, capsys, from_stdin):
    parser = _parser()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(tmp_path / "managed"))
    secret = "before\0after"
    command = ["secrets", "set", "REVIEW_SECRET"]
    if from_stdin:
        command.append("--stdin")
        monkeypatch.setattr("sys.stdin", io.StringIO(secret + "\n"))
    else:
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("hermes_cli.local_secrets.getpass.getpass", lambda _prompt: secret)

    args = parser.parse_args(command)
    assert args.func(args) == 1
    output = capsys.readouterr()
    assert "before" not in output.out + output.err
    assert "after" not in output.out + output.err
    assert not (tmp_path / ".env").exists()


def test_set_reports_managed_scope_refusal_as_failure(tmp_path, monkeypatch, capsys):
    from hermes_cli import managed_scope

    managed = tmp_path / "managed"
    managed.mkdir()
    (managed / ".env").write_text("REVIEW_SECRET=managed\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    managed_scope.invalidate_managed_cache()
    monkeypatch.setattr("sys.stdin", io.StringIO("new-value\n"))

    args = _parser().parse_args(
        ["secrets", "set", "REVIEW_SECRET", "--stdin"])
    assert args.func(args) == 1
    output = capsys.readouterr()
    assert "Cannot set REVIEW_SECRET" in output.err
    assert "Stored REVIEW_SECRET" not in output.out + output.err
    assert not (tmp_path / ".env").exists()


def test_set_rejects_redirected_multiline_value_without_disclosure(
        tmp_path, monkeypatch, capsys):
    parser = _parser()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(tmp_path / "managed"))
    secret = "alpha\nbeta"
    monkeypatch.setattr("sys.stdin", io.StringIO(secret + "\n"))

    args = parser.parse_args(
        ["secrets", "set", "REVIEW_SECRET", "--stdin"])
    assert args.func(args) == 1
    output = capsys.readouterr()
    assert "alpha" not in output.out + output.err
    assert "beta" not in output.out + output.err
    assert not (tmp_path / ".env").exists()


def test_stdin_strips_trailing_line_terminators_but_preserves_spaces(
        tmp_path, monkeypatch):
    from hermes_cli.config import invalidate_env_cache, load_env

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(tmp_path / "managed"))
    monkeypatch.setattr("sys.stdin", io.StringIO("space-sensitive  \r\n\n"))

    args = _parser().parse_args(
        ["secrets", "set", "REVIEW_SECRET", "--stdin"])
    assert args.func(args) == 0
    invalidate_env_cache()
    assert load_env()["REVIEW_SECRET"] == "space-sensitive  "


def test_stdin_list_and_delete_keep_values_out_of_output(tmp_path, monkeypatch, capsys):
    parser = _parser()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(tmp_path / "managed"))
    monkeypatch.setattr("sys.stdin", io.StringIO("piped-secret\n"))

    set_args = parser.parse_args(["secrets", "set", "GITHUB_TOKEN", "--stdin"])
    assert set_args.func(set_args) == 0
    list_args = parser.parse_args(["secrets", "list"])
    assert list_args.func(list_args) == 0
    delete_args = parser.parse_args(["secrets", "delete", "GITHUB_TOKEN"])
    assert delete_args.func(delete_args) == 0

    output = capsys.readouterr()
    assert "GITHUB_TOKEN" in output.out
    assert "piped-secret" not in output.out + output.err
    assert "GITHUB_TOKEN" not in (tmp_path / ".env").read_text(encoding="utf-8")


def test_delete_managed_secret_refuses_before_lifecycle_side_effects(
        tmp_path, monkeypatch, capsys):
    import json

    from hermes_cli import managed_scope
    from hermes_cli.config import invalidate_env_cache

    managed = tmp_path / "managed"
    managed.mkdir()
    (managed / ".env").write_text("OPENAI_API_KEY=managed-value\n", encoding="utf-8")
    auth_path = tmp_path / "auth.json"
    auth_text = json.dumps({
        "credential_pool": {"openai": [{
            "id": "managed-env",
            "source": "env:OPENAI_API_KEY",
            "auth_type": "api_key",
        }]},
        "suppressed_sources": {},
    })
    auth_path.write_text(auth_text, encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_text = "model:\n  api_key: managed-value\n"
    config_path.write_text(config_text, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    managed_scope.invalidate_managed_cache()
    invalidate_env_cache()

    args = _parser().parse_args(["secrets", "delete", "OPENAI_API_KEY"])
    assert args.func(args) == 1
    output = capsys.readouterr()
    assert "Cannot remove OPENAI_API_KEY" in output.err
    assert "Deleted OPENAI_API_KEY" not in output.out + output.err
    assert (managed / ".env").read_text(encoding="utf-8") == "OPENAI_API_KEY=managed-value\n"
    assert auth_path.read_text(encoding="utf-8") == auth_text
    assert config_path.read_text(encoding="utf-8") == config_text
    assert not (tmp_path / ".env").exists()


def test_delete_allows_cleanup_of_write_denylisted_names(tmp_path, monkeypatch, capsys):
    from hermes_cli.config import invalidate_env_cache

    env_path = tmp_path / ".env"
    env_path.write_text("LD_PRELOAD=dangerous-value\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(tmp_path / "managed"))
    invalidate_env_cache()

    args = _parser().parse_args(["secrets", "delete", "LD_PRELOAD"])
    assert args.func(args) == 0
    output = capsys.readouterr()
    assert "dangerous-value" not in output.out + output.err
    assert "LD_PRELOAD" not in env_path.read_text(encoding="utf-8")


def test_list_reports_unreadable_env_instead_of_claiming_empty(monkeypatch, capsys):
    env_path = "profile/.env"
    monkeypatch.setattr("hermes_cli.config.get_env_path", lambda: env_path)
    monkeypatch.setattr(
        "hermes_cli.config.load_env_strict",
        lambda: (_ for _ in ()).throw(PermissionError("access denied")),
    )

    assert _parser().parse_args(["secrets", "list"]).func(None) == 1
    output = capsys.readouterr()
    assert "could not read profile/.env" in output.err
    assert "No local secrets" not in output.out + output.err


@pytest.mark.parametrize("command", ["set", "delete"])
def test_mutations_report_filesystem_failures_without_disclosing_values(
        monkeypatch, capsys, command):
    secret = "never-print-this-value"
    if command == "set":
        monkeypatch.setattr("sys.stdin", io.StringIO(secret + "\n"))
        monkeypatch.setattr(
            "hermes_cli.config.save_env_value_secure",
            lambda *_args: (_ for _ in ()).throw(PermissionError("store is read-only")),
        )
        args = _parser().parse_args(
            ["secrets", "set", "REVIEW_SECRET", "--stdin"])
    else:
        monkeypatch.setattr(
            "hermes_cli.credential_lifecycle.remove_provider_env_credential",
            lambda *_args: (_ for _ in ()).throw(PermissionError("store is read-only")),
        )
        args = _parser().parse_args(["secrets", "delete", "REVIEW_SECRET"])

    assert args.func(args) == 1
    output = capsys.readouterr()
    assert "store is read-only" in output.err
    assert secret not in output.out + output.err

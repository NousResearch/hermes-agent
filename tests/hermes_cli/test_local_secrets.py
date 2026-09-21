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
    with pytest.raises(SystemExit):
        parser.parse_args(["secrets", "set", "OPENAI_API_KEY", "must-not-be-argv"])
    capsys.readouterr()

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(tmp_path / "managed"))
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("hermes_cli.local_secrets.getpass.getpass", lambda _prompt: "super-secret-value")
    args = parser.parse_args(["secrets", "set", "OPENAI_API_KEY"])

    assert args.func(args) == 0
    assert "OPENAI_API_KEY=super-secret-value" in (tmp_path / ".env").read_text(encoding="utf-8")
    output = capsys.readouterr()
    assert "super-secret-value" not in output.out + output.err


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
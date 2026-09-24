"""Regression tests for hermes secrets bitwarden setup non-TTY guard.

Issue #40274: cmd_setup() crashes with EOFError when stdin is not a TTY
because getpass.getpass() and console.input() require an interactive terminal.

Issue #120993: the headless flag check must run BEFORE the bws install step —
a scripted caller missing a flag is guaranteed to exit 1 and must not pay the
find/download cost (or any filesystem writes) for a doomed run.
"""
from __future__ import annotations

import argparse
from unittest.mock import patch

import pytest


def _forbid_binary_install(monkeypatch, calls):
    """Wire find_bws/install_bws to record any touch and fail the test if reached."""
    def _record(name):
        def _fn(*a, **kw):
            calls.append(name)
            raise AssertionError(f"doomed headless run must not touch {name}")
        return _fn
    monkeypatch.setattr("hermes_cli.secrets_cli.bw.find_bws", _record("find_bws"))
    monkeypatch.setattr("hermes_cli.secrets_cli.bw.install_bws", _record("install_bws"))


class TestCmdSetupNonTtyGuard:
    """cmd_setup should fail early with a clear error in non-TTY environments."""

    @staticmethod
    def _make_args(**overrides):
        ns = argparse.Namespace(
            access_token=overrides.get("access_token", ""),
            server_url=overrides.get("server_url", ""),
            project_id=overrides.get("project_id", ""),
        )
        return ns


    def test_missing_access_token_only(self, monkeypatch, capsys):
        """Non-TTY with server-url and project-id but no token → reports --access-token."""
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        monkeypatch.setattr(
            "hermes_cli.secrets_cli.bw.find_bws", lambda install_if_missing=False: "/usr/bin/bws"
        )
        monkeypatch.setattr(
            "hermes_cli.secrets_cli._bws_version", lambda _: "2.0.0"
        )

        from hermes_cli.secrets_cli import cmd_setup

        result = cmd_setup(self._make_args(
            server_url="https://vault.bitwarden.com",
            project_id="aaaa-bbbb",
        ))
        assert result == 1
        captured = capsys.readouterr()
        # The "Missing:" line should list --access-token only
        assert "Missing:" in captured.out
        assert "--access-token" in captured.out
        # The usage example contains --server-url and --project-id, so check
        # the missing line specifically: it should NOT list them as missing
        missing_line = [l for l in captured.out.split("\n") if "Missing:" in l][0]
        assert "--access-token" in missing_line
        assert "--server-url" not in missing_line
        assert "--project-id" not in missing_line

    def test_missing_server_url_with_env_var_passes(self, monkeypatch):
        """Non-TTY with BWS_SERVER_URL env set → server-url not required."""
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        monkeypatch.setenv("BWS_SERVER_URL", "https://vault.bitwarden.com")
        monkeypatch.setattr(
            "hermes_cli.secrets_cli.bw.find_bws", lambda install_if_missing=False: "/usr/bin/bws"
        )
        monkeypatch.setattr(
            "hermes_cli.secrets_cli._bws_version", lambda _: "2.0.0"
        )
        monkeypatch.setattr("hermes_cli.secrets_cli.load_config", lambda: {})
        monkeypatch.setattr("hermes_cli.secrets_cli.save_env_value", lambda *a: None)
        monkeypatch.setattr("hermes_cli.secrets_cli.get_env_path", lambda: "/tmp/.env")
        monkeypatch.setattr(
            "hermes_cli.secrets_cli.bw.fetch_bitwarden_secrets",
            lambda **kw: ({"KEY": "val"}, []),
        )

        from hermes_cli.secrets_cli import cmd_setup

        result = cmd_setup(self._make_args(
            access_token="0.valid-token",
            project_id="aaaa-bbbb",
        ))
        assert result == 0


class TestHeadlessFlagCheckBeforeInstall:
    """Regression for #120993: a doomed non-TTY run must not download bws first."""

    def test_doomed_non_tty_run_invokes_neither_find_bws_nor_install_bws(
            self, monkeypatch, capsys):
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        calls: list[str] = []
        _forbid_binary_install(monkeypatch, calls)

        from hermes_cli.secrets_cli import cmd_setup

        rc = cmd_setup(argparse.Namespace(
            access_token="", server_url="", project_id=""))
        assert rc == 1
        assert "Missing:" in capsys.readouterr().out
        assert calls == []

    def test_doomed_run_through_real_argparse_dispatch_also_skips_install(
            self, monkeypatch, capsys):
        from hermes_cli.subcommands.secrets import build_secrets_parser
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers()
        build_secrets_parser(sub)
        # parse-time import of secrets_cli is expected and crypto-free (#86781)
        argv = ["secrets", "bitwarden", "setup"]  # no flags → doomed headless run
        args = parser.parse_args(argv)
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        calls: list[str] = []
        _forbid_binary_install(monkeypatch, calls)
        rc = args.func(args)
        assert rc == 1
        assert "Missing:" in capsys.readouterr().out
        assert calls == []

    def test_fully_flagged_headless_run_still_installs(self, monkeypatch, capsys):
        """Boundary arm: a VALID headless run still performs the binary install."""
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        installed: list[str] = []
        monkeypatch.setattr(
            "hermes_cli.secrets_cli.bw.find_bws",
            lambda install_if_missing=False: None)
        monkeypatch.setattr(
            "hermes_cli.secrets_cli.bw.install_bws",
            lambda: installed.append("bws") or "/managed/bws")
        monkeypatch.setattr(
            "hermes_cli.secrets_cli._bws_version", lambda _: "2.0.0")
        monkeypatch.setattr("hermes_cli.secrets_cli.load_config", lambda: {})
        monkeypatch.setattr("hermes_cli.secrets_cli.save_env_value", lambda *a: None)
        monkeypatch.setattr("hermes_cli.secrets_cli.get_env_path", lambda: "/tmp/.env")
        monkeypatch.setenv("BWS_SERVER_URL", "https://vault.bitwarden.com")
        monkeypatch.setattr(
            "hermes_cli.secrets_cli.bw.fetch_bitwarden_secrets",
            lambda **kw: ({"KEY": "val"}, []))

        from hermes_cli.secrets_cli import cmd_setup

        rc = cmd_setup(argparse.Namespace(
            access_token="0.valid-token",
            server_url="https://vault.bitwarden.com",
            project_id="aaaa-bbbb"))
        assert rc == 0
        assert installed == ["bws"]



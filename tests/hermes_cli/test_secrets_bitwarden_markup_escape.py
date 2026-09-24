"""Regression tests: server/bws-derived text must not reach a markup console raw.

bws stderr, project names, secret names, and fetch warnings all come from the
configured Bitwarden endpoint (``secrets.bitwarden.server_url`` may point at a
self-hosted server). They must print as literal text: Rich markup must not be
parsed and terminal control sequences (ANSI/OSC) must not pass through.
"""
from __future__ import annotations

import argparse
import io
import json
import subprocess
from pathlib import Path

import pytest
from rich.console import Console

import hermes_cli.secrets_cli as secrets_cli

HOSTILE = "token revoked [green]Token accepted[/green] \x1b]8;;http://evil.example\x07CLICKME\x1b]8;;\x07 end"


def _failed_run(*args, **kwargs):
    return subprocess.CompletedProcess(args[0], 1, "", HOSTILE)


def _recording_console() -> Console:
    return Console(file=io.StringIO(), record=True, width=200)


def _render(messages, console=None):
    console = console or _recording_console()
    for m in messages:
        console.print(m)
    return console.export_text(styles=False)


class TestListProjectsHostileStderr:
    """The primary surface: bws stderr interpolated into a markup string."""

    def test_markup_prints_literally_and_escapes_stripped(self, monkeypatch):
        monkeypatch.setattr(secrets_cli.subprocess, "run", _failed_run)
        console = _recording_console()
        assert secrets_cli._list_projects(Path("/bin/bws"), "tok", console) is None
        out = console.export_text(styles=False)
        assert "[green]Token accepted[/green]" in out
        assert "\x1b" not in out and "\x07" not in out

    def test_spawn_error_text_escaped(self, monkeypatch):
        def boom(*a, **k):
            raise OSError("spawn failed [green]ok[/green] \x1b[31m")
        monkeypatch.setattr(secrets_cli.subprocess, "run", boom)
        console = _recording_console()
        assert secrets_cli._list_projects(Path("/bin/bws"), "tok", console) is None
        out = console.export_text(styles=False)
        assert "[green]ok[/green]" in out
        assert "\x1b" not in out


class TestProjectTableEscaping:
    def test_project_names_and_ids_print_literally(self, monkeypatch):
        payload = [{"id": "id\x1b[1m", "name": "proj [green]x[/green]"}]
        monkeypatch.setattr(
            secrets_cli.subprocess, "run",
            lambda *a, **k: subprocess.CompletedProcess(a[0], 0, json.dumps(payload), ""))
        console = _recording_console()
        projects = secrets_cli._list_projects(Path("/bin/bws"), "tok", console)
        assert projects == payload
        # _setup_project renders the table; drive it directly with a fixed pick.
        monkeypatch.setattr(secrets_cli, "prompt_index", lambda *a, **k: 1)
        chosen = secrets_cli._setup_project(Path("/bin/bws"), "tok", console, server_url="")
        assert chosen == "id\x1b[1m"
        out = console.export_text(styles=False)
        assert "[green]x[/green]" in out
        assert "\x1b" not in out


class TestStatusValidationRerelease:
    """The probe console is exported to text, then re-printed through markup in
    cmd_status: literal brackets in the export must not re-parse."""

    def test_exported_details_survive_reprint_literally(self, monkeypatch):
        monkeypatch.setattr(secrets_cli.subprocess, "run", _failed_run)
        status, messages = secrets_cli._token_validation_status(
            enabled=True, binary=Path("/bin/bws"), token="0.abc", server_url="")
        assert status == "[red]failed[/red]"
        out = _render(messages)
        assert "[green]Token accepted[/green]" in out
        assert "\x1b" not in out


class TestSetupAndSyncSurfaces:
    def _setup_stubs(self, monkeypatch, secrets, warnings):
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        monkeypatch.setattr(secrets_cli.bw, "find_bws",
                            lambda install_if_missing=False: Path("/bin/bws"))
        monkeypatch.setattr(secrets_cli, "_bws_version", lambda _b: "2.0.0")
        monkeypatch.setattr(secrets_cli.bw, "fetch_bitwarden_secrets",
                            lambda **k: (secrets, warnings))
        monkeypatch.setattr(secrets_cli, "save_config", lambda _c: None)
        monkeypatch.setattr(secrets_cli, "load_config", lambda: {})

    def test_setup_escapes_secret_names_and_warnings(self, monkeypatch, capsys):
        hostile_key = "K[green]spoof[/green]"
        self._setup_stubs(
            monkeypatch,
            {hostile_key: "v"},
            ["Skipping secret 'n[red]x[/red]': not a valid env-var name \x1b]8;;u\x07"],
        )
        monkeypatch.setattr(secrets_cli.subprocess, "run",
                            lambda *a, **k: subprocess.CompletedProcess(
                                a[0], 0, json.dumps([{"id": "p1", "name": "n"}]), ""))
        args = argparse.Namespace(access_token="0.t", server_url="https://vault.example",
                                  project_id="p1")
        assert secrets_cli.cmd_setup(args) == 0
        out = capsys.readouterr().out
        assert "[green]spoof[/green]" in out
        assert "[red]x[/red]" in out
        assert "\x1b" not in out and "\x07" not in out

    def test_sync_escapes_secret_names_and_warnings(self, monkeypatch, capsys):
        hostile_key = "K[green]spoof[/green]"
        monkeypatch.setattr(
            secrets_cli, "load_config",
            lambda: {"secrets": {"bitwarden": {"enabled": True, "project_id": "p1",
                                             "access_token_env": "BWS_ACCESS_TOKEN"}}})
        monkeypatch.setenv("BWS_ACCESS_TOKEN", "0.t")
        monkeypatch.setattr(secrets_cli.bw, "fetch_bitwarden_secrets",
                            lambda **k: ({hostile_key: "v"}, ["w [green]x[/green] \x1b[31m"]))
        args = argparse.Namespace(apply=False)
        assert secrets_cli.cmd_sync(args) == 0
        out = capsys.readouterr().out
        assert "[green]spoof[/green]" in out
        assert "[green]x[/green]" in out
        assert "\x1b" not in out

    def test_sync_fetch_error_text_escaped(self, monkeypatch, capsys):
        monkeypatch.setattr(
            secrets_cli, "load_config",
            lambda: {"secrets": {"bitwarden": {"enabled": True, "project_id": "p1",
                                             "access_token_env": "BWS_ACCESS_TOKEN"}}})
        monkeypatch.setenv("BWS_ACCESS_TOKEN", "0.t")

        def boom(**k):
            raise RuntimeError("bws failed [green]ok[/green] \x1b]8;;u\x07")
        monkeypatch.setattr(secrets_cli.bw, "fetch_bitwarden_secrets", boom)
        args = argparse.Namespace(apply=False)
        assert secrets_cli.cmd_sync(args) == 1
        out = capsys.readouterr().out
        assert "[green]ok[/green]" in out
        assert "\x1b" not in out and "\x07" not in out


class TestStatusAuthoredMarkupBoundary:
    """Only probe-derived lines get escaped; authored markup in the messages
    list (the '0.' prefix warning) must still render as markup."""

    def test_authored_warning_renders_while_injected_stays_literal(self, monkeypatch):
        monkeypatch.setattr(secrets_cli.subprocess, "run", _failed_run)
        status, messages = secrets_cli._token_validation_status(
            enabled=True, binary=Path("/bin/bws"), token="not-a-bsm-token",
            server_url="")
        assert status == "[red]failed[/red]"
        out = _render(messages)
        # authored [yellow] markup rendered (not shown literally)
        assert "[yellow]" not in out
        assert "doesn't start with '0.'" in out
        # injected markup stayed literal
        assert "[green]Token accepted[/green]" in out

    def test_hint_matching_uses_unescaped_stderr(self, monkeypatch):
        """_PROJECT_LIST_HINTS must match the raw stderr: a fix that escaped
        before matching would silently drop the hints."""
        monkeypatch.setattr(
            secrets_cli.subprocess, "run",
            lambda *a, **k: subprocess.CompletedProcess(
                a[0], 1, "", "Error: invalid_client [green]x[/green]"))
        console = _recording_console()
        assert secrets_cli._list_projects(Path("/bin/bws"), "tok", console) is None
        out = console.export_text(styles=False)
        assert "invalid_client" in out
        assert "from the US identity endpoint" in out
        assert "[green]x[/green]" in out


class TestTokenCommandE2E:
    """`bitwarden token --access-token` verifies through _list_projects on the
    real console — the exact spot where a spoofed 'Token accepted' misleads."""

    def test_token_verify_hostile_stderr_literal(self, monkeypatch, capsys):
        import argparse as _ap

        from hermes_cli.subcommands.secrets import build_secrets_parser
        monkeypatch.setattr(
            secrets_cli, "load_config",
            lambda: {"secrets": {"bitwarden": {"enabled": True, "project_id": "p1",
                                             "access_token_env": "BWS_ACCESS_TOKEN"}}})
        monkeypatch.setattr(secrets_cli.bw, "find_bws",
                            lambda install_if_missing=True: Path("/bin/bws"))
        monkeypatch.setattr(secrets_cli.subprocess, "run", _failed_run)
        saved = []
        monkeypatch.setattr(secrets_cli, "save_env_value",
                            lambda k, v: saved.append((k, v)))
        parser = _ap.ArgumentParser()
        build_secrets_parser(parser.add_subparsers())
        args = parser.parse_args(
            ["secrets", "bitwarden", "token", "--access-token", "0.new"])
        assert args.func(args) == 1
        assert saved == []  # rejected token must not persist
        out = capsys.readouterr().out
        assert "[green]Token accepted[/green]" in out
        assert "\x1b" not in out and "\x07" not in out


class TestOnepasswordSiblingSurfaces:
    """Same defect class in the 1Password secrets CLI (op stderr/server text)."""

    def test_sync_fetch_error_text_escaped(self, monkeypatch, capsys):
        import hermes_cli.onepassword_secrets_cli as op_cli

        monkeypatch.setattr(
            op_cli, "load_config",
            lambda: {"secrets": {"onepassword": {
                "enabled": True, "env": {"K": "op://v/i/f"}}}})

        def boom(**k):
            raise RuntimeError("op failed [green]ok[/green] \x1b]8;;u\x07")
        monkeypatch.setattr(op_cli.op_src, "fetch_onepassword_secrets", boom)
        args = argparse.Namespace(apply=False)
        assert op_cli.cmd_sync(args) == 1
        out = capsys.readouterr().out
        assert "[green]ok[/green]" in out
        assert "\x1b" not in out and "\x07" not in out


class TestStatusCommandE2E:
    """Real parser + dispatcher: hermes secrets bitwarden status."""

    def test_status_e2e_hostile_stderr_literal(self, monkeypatch, capsys):
        import argparse as _ap

        from hermes_cli.subcommands.secrets import build_secrets_parser
        monkeypatch.setattr(
            secrets_cli, "load_config",
            lambda: {"secrets": {"bitwarden": {"enabled": True, "project_id": "p1",
                                             "access_token_env": "BWS_ACCESS_TOKEN"}}})
        monkeypatch.setenv("BWS_ACCESS_TOKEN", "0.t")
        monkeypatch.setattr(secrets_cli.bw, "find_bws",
                            lambda install_if_missing=False: Path("/bin/bws"))
        monkeypatch.setattr(secrets_cli, "_bws_version", lambda _b: "2.0.0")
        monkeypatch.setattr(secrets_cli.subprocess, "run", _failed_run)
        parser = _ap.ArgumentParser()
        build_secrets_parser(parser.add_subparsers())
        args = parser.parse_args(["secrets", "bitwarden", "status"])
        assert args.func(args) == 0
        out = capsys.readouterr().out
        assert "[green]Token accepted[/green]" in out
        assert "\x1b" not in out and "\x07" not in out

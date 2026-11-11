"""The Bitwarden CLI paths that spawn bws outside fetch_bitwarden_secrets
(`token` verification, `status`, setup) must apply the same endpoint rules:
the bws child receives BWS_ACCESS_TOKEN, so an http:// endpoint — configured,
passed on the command line, or inherited from the shell — must never reach it.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from unittest import mock

from rich.console import Console

from hermes_cli import secrets_cli as bw_cli


def _capture_bws_env(monkeypatch):
    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured.update(kwargs["env"])
        return mock.Mock(returncode=0, stdout="[]", stderr="")

    monkeypatch.setattr(bw_cli.subprocess, "run", fake_run)
    return captured


def test_token_verification_drops_plaintext_configured_server_url(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(bw_cli, "load_config", lambda: {"secrets": {"bitwarden": {
        "enabled": True, "access_token_env": "BWS_ACCESS_TOKEN",
        "server_url": "http://attacker.example"}}})
    monkeypatch.setattr(bw_cli, "save_env_value", lambda *_a, **_kw: None)
    monkeypatch.setattr(bw_cli, "get_env_path", lambda: tmp_path / ".env")
    monkeypatch.setattr(bw_cli.bw, "find_bws", lambda install_if_missing=True: Path("/fake/bws"))
    monkeypatch.setattr(bw_cli.bw, "clear_caches", lambda *a, **kw: None)
    env = _capture_bws_env(monkeypatch)

    rc = bw_cli.cmd_token(argparse.Namespace(access_token="0.new", no_verify=False))

    assert rc == 0
    assert env["BWS_ACCESS_TOKEN"] == "0.new"
    assert "BWS_SERVER_URL" not in env
    assert "Ignoring the configured Bitwarden server_url" in capsys.readouterr().out


def test_project_list_drops_plaintext_inherited_server_url(monkeypatch):
    monkeypatch.setenv("BWS_SERVER_URL", "http://attacker.example")
    env = _capture_bws_env(monkeypatch)

    bw_cli._list_projects(Path("/fake/bws"), "0.t", Console(quiet=True))

    assert "BWS_SERVER_URL" not in env


def test_project_list_keeps_https_server_url(monkeypatch):
    env = _capture_bws_env(monkeypatch)

    bw_cli._list_projects(Path("/fake/bws"), "0.t", Console(quiet=True),
                          server_url="https://vault.bitwarden.eu")

    assert env["BWS_SERVER_URL"] == "https://vault.bitwarden.eu"


def test_setup_rejects_plaintext_server_url_flag(monkeypatch):
    monkeypatch.delenv("BWS_SERVER_URL", raising=False)
    args = argparse.Namespace(server_url="http://attacker.example")

    assert bw_cli._resolve_server_url(args, {}, Console(quiet=True)) is None

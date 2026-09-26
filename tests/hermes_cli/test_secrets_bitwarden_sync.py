"""Tests for `hermes secrets bitwarden sync` honouring `override_existing`.

`--apply` must apply the configured policy, not force an override: a var the
dry-run reported as "skip (already set)" must still be set-valued after
`--apply`.  The 1Password backend already delegates `--apply` to the same
code path startup uses; the Bitwarden handler evaluates the policy inline.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from unittest import mock

import pytest

from hermes_cli import secrets_cli as bw_cli


def _cfg(override_existing=...):
    bw = {
        "enabled": True,
        "access_token_env": "BWS_ACCESS_TOKEN",
        "project_id": "proj-1",
        "server_url": "",
    }
    if override_existing is not ...:
        bw["override_existing"] = override_existing
    return {"secrets": {"bitwarden": bw}}


@pytest.fixture
def sync_env(monkeypatch):
    """Config + fake backend + captured action table; yields (rows, secrets)."""
    state = {"rows": None, "secrets": {"EXISTING_VAR": "bw-value", "NEW_VAR": "fresh"}}
    fake_bw = mock.Mock()
    fake_bw.fetch_bitwarden_secrets.side_effect = lambda **kw: (state["secrets"], [])
    monkeypatch.setattr(bw_cli, "_load_bw", lambda: fake_bw)

    def _capture(console, headers, rows, warnings=()):
        state["rows"] = list(rows)

    monkeypatch.setattr(bw_cli, "print_table", _capture)
    monkeypatch.setenv("BWS_ACCESS_TOKEN", "0.token")
    return state


def _sync(apply: bool) -> int:
    return bw_cli.cmd_sync(argparse.Namespace(apply=apply))


def _action(state, key):
    return dict(state["rows"])[key]


def test_apply_respects_override_existing_false(sync_env, monkeypatch):
    monkeypatch.setattr(bw_cli, "load_config", lambda: _cfg(False))
    monkeypatch.setenv("EXISTING_VAR", "mine")
    monkeypatch.delenv("NEW_VAR", raising=False)
    assert _sync(apply=True) == 0
    assert os.environ["EXISTING_VAR"] == "mine"          # not clobbered
    assert _action(sync_env, "EXISTING_VAR") == "[dim]skip (already set)[/dim]"


def test_apply_override_false_still_writes_new_vars(sync_env, monkeypatch):
    monkeypatch.setattr(bw_cli, "load_config", lambda: _cfg(False))
    monkeypatch.delenv("NEW_VAR", raising=False)
    assert _sync(apply=True) == 0
    assert os.environ["NEW_VAR"] == "fresh"
    assert _action(sync_env, "NEW_VAR") == "[green]exported[/green]"


def test_apply_override_true_overrides(sync_env, monkeypatch):
    monkeypatch.setattr(bw_cli, "load_config", lambda: _cfg(True))
    monkeypatch.setenv("EXISTING_VAR", "mine")
    assert _sync(apply=True) == 0
    assert os.environ["EXISTING_VAR"] == "bw-value"
    assert _action(sync_env, "EXISTING_VAR") == "[green]exported[/green] (overrode)"


def test_dry_run_override_false_skips(sync_env, monkeypatch):
    """Control: the dry-run already honoured the flag before the fix."""
    monkeypatch.setattr(bw_cli, "load_config", lambda: _cfg(False))
    monkeypatch.setenv("EXISTING_VAR", "mine")
    assert _sync(apply=False) == 0
    assert os.environ["EXISTING_VAR"] == "mine"
    assert _action(sync_env, "EXISTING_VAR") == "[dim]skip (already set)[/dim]"


def test_missing_key_defaults_to_override(sync_env, monkeypatch):
    """The schema default is True; a hand-edited config without the key must
    preview and apply the same way the startup path does."""
    monkeypatch.setattr(bw_cli, "load_config", lambda: _cfg())
    monkeypatch.setenv("EXISTING_VAR", "mine")
    assert _sync(apply=False) == 0
    assert _action(sync_env, "EXISTING_VAR") == "[green]would export[/green] (overrides)"
    assert _sync(apply=True) == 0
    assert os.environ["EXISTING_VAR"] == "bw-value"


def test_apply_never_writes_token_env(sync_env, monkeypatch):
    monkeypatch.setattr(bw_cli, "load_config", lambda: _cfg(True))
    sync_env["secrets"]["BWS_ACCESS_TOKEN"] = "rotated"
    monkeypatch.setenv("BWS_ACCESS_TOKEN", "0.token")
    assert _sync(apply=True) == 0
    assert os.environ["BWS_ACCESS_TOKEN"] == "0.token"
    assert _action(sync_env, "BWS_ACCESS_TOKEN") == "[dim]skip (bootstrap token)[/dim]"


def test_apply_e2e_through_cli_parser(sync_env, monkeypatch):
    """`sync --apply` through the real argparse tree honours override_existing."""
    monkeypatch.setattr(bw_cli, "load_config", lambda: _cfg(False))
    monkeypatch.setenv("EXISTING_VAR", "mine")
    parser = argparse.ArgumentParser()
    bw_cli.register_cli(parser)
    args = parser.parse_args(["sync", "--apply"])
    assert args.func(args) == 0
    assert os.environ["EXISTING_VAR"] == "mine"


def test_fetch_status_respects_override(monkeypatch):
    monkeypatch.setenv("EXISTING_VAR", "mine")
    assert "will be overwritten" in bw_cli._fetch_status("EXISTING_VAR", "BWS_ACCESS_TOKEN", True)
    assert "will be kept" in bw_cli._fetch_status("EXISTING_VAR", "BWS_ACCESS_TOKEN", False)


def test_status_shows_default_override_true(sync_env, monkeypatch, capsys):
    """A config without the key must report the schema default, not `no`."""
    monkeypatch.setattr(bw_cli, "load_config", lambda: _cfg())
    monkeypatch.setattr(bw_cli, "_load_bw", lambda: mock.Mock(
        find_bws=lambda install_if_missing=False: Path("/fake/bws")))
    monkeypatch.setattr(bw_cli, "_bws_version", lambda _binary: "bws 2.0.0")
    monkeypatch.setattr(bw_cli, "_token_validation_status", lambda **kw: ("[green]passed[/green]", []))
    assert bw_cli.cmd_status(argparse.Namespace()) == 0
    line = next(ln for ln in capsys.readouterr().out.splitlines() if "Override existing" in ln)
    assert "yes" in line

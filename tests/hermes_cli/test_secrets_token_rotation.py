"""Tests for `hermes secrets bitwarden token` / `hermes secrets onepassword token`.

The rotation command must: verify the candidate token BEFORE persisting,
never touch .env on a rejected token, store + clear caches on success,
and fail cleanly without a TTY.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from unittest import mock

import pytest

from hermes_cli import onepassword_secrets_cli as op_cli
from hermes_cli import secrets_cli as bw_cli


# ---------------------------------------------------------------------------
# Bitwarden
# ---------------------------------------------------------------------------


def _bw_args(**overrides):
    return argparse.Namespace(
        access_token=overrides.get("access_token", ""),
        no_verify=overrides.get("no_verify", False),
    )


@pytest.fixture
def bw_env(monkeypatch, tmp_path):
    saved = {}
    monkeypatch.setattr(bw_cli, "load_config", lambda: {
        "secrets": {"bitwarden": {
            "enabled": True,
            "access_token_env": "BWS_ACCESS_TOKEN",
            "project_id": "proj-1",
            "server_url": "",
        }},
    })
    monkeypatch.setattr(
        bw_cli, "save_env_value",
        lambda name, value: saved.__setitem__(name, value),
    )
    monkeypatch.setattr(bw_cli, "get_env_path", lambda: tmp_path / ".env")
    monkeypatch.setattr(
        bw_cli.bw, "find_bws",
        lambda install_if_missing=True: Path("/fake/bws"),
    )
    return saved


def test_bw_token_rejected_token_never_persisted(bw_env, monkeypatch):
    monkeypatch.setattr(
        bw_cli, "_list_projects",
        lambda binary, token, console, server_url="": None,  # probe fails
    )
    rc = bw_cli.cmd_token(_bw_args(access_token="0.bad"))
    assert rc == 1
    assert bw_env == {}  # nothing written to .env


def test_bw_token_accepted_token_persisted_and_caches_cleared(bw_env, monkeypatch):
    cleared = []
    monkeypatch.setattr(
        bw_cli, "_list_projects",
        lambda binary, token, console, server_url="": [{"id": "proj-1"}],
    )
    monkeypatch.setattr(bw_cli.bw, "clear_caches", lambda *a, **kw: cleared.append(True))
    rc = bw_cli.cmd_token(_bw_args(access_token="0.fresh"))
    assert rc == 0
    assert bw_env == {"BWS_ACCESS_TOKEN": "0.fresh"}
    assert cleared


def test_bw_token_warns_when_project_not_visible(bw_env, monkeypatch, capsys):
    monkeypatch.setattr(
        bw_cli, "_list_projects",
        lambda binary, token, console, server_url="": [{"id": "other-proj"}],
    )
    monkeypatch.setattr(bw_cli.bw, "clear_caches", lambda *a, **kw: None)
    rc = bw_cli.cmd_token(_bw_args(access_token="0.fresh"))
    assert rc == 0  # stored anyway — the token itself is valid
    out = capsys.readouterr().out
    assert "proj-1" in out and "not visible" in out


def test_bw_token_no_verify_skips_probe(bw_env, monkeypatch):
    probe = mock.Mock()
    monkeypatch.setattr(bw_cli, "_list_projects", probe)
    monkeypatch.setattr(bw_cli.bw, "clear_caches", lambda *a, **kw: None)
    rc = bw_cli.cmd_token(_bw_args(access_token="0.x", no_verify=True))
    assert rc == 0
    probe.assert_not_called()
    assert bw_env == {"BWS_ACCESS_TOKEN": "0.x"}


def test_bw_token_non_tty_requires_flag(bw_env, monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    rc = bw_cli.cmd_token(_bw_args())
    assert rc == 1
    assert bw_env == {}


# ---------------------------------------------------------------------------
# 1Password
# ---------------------------------------------------------------------------


def _op_args(**overrides):
    return argparse.Namespace(
        token=overrides.get("token", ""),
        no_verify=overrides.get("no_verify", False),
    )


@pytest.fixture
def op_env(monkeypatch, tmp_path):
    saved = {}
    monkeypatch.setattr(op_cli, "load_config", lambda: {
        "secrets": {"onepassword": {
            "enabled": True,
            "service_account_token_env": "OP_SERVICE_ACCOUNT_TOKEN",
        }},
    })
    monkeypatch.setattr(
        op_cli, "save_env_value",
        lambda name, value: saved.__setitem__(name, value),
    )
    monkeypatch.setattr(op_cli, "get_env_path", lambda: tmp_path / ".env")
    monkeypatch.setattr(
        op_cli.op_src, "find_op", lambda binary_path="": Path("/fake/op")
    )
    return saved


def test_op_token_rejected_never_persisted(op_env, monkeypatch):
    monkeypatch.setattr(
        op_cli, "_op_whoami",
        lambda binary, account, token_value="": None,
    )
    rc = op_cli.cmd_token(_op_args(token="ops_bad"))
    assert rc == 1
    assert op_env == {}


def test_op_token_accepted_persisted_and_caches_cleared(op_env, monkeypatch):
    cleared = []
    monkeypatch.setattr(
        op_cli, "_op_whoami",
        lambda binary, account, token_value="": "service-account test",
    )
    monkeypatch.setattr(
        op_cli.op_src, "clear_caches", lambda *a, **kw: cleared.append(True)
    )
    rc = op_cli.cmd_token(_op_args(token="ops_fresh"))
    assert rc == 0
    assert op_env == {"OP_SERVICE_ACCOUNT_TOKEN": "ops_fresh"}
    assert cleared


def test_op_token_probe_uses_candidate_token(op_env, monkeypatch):
    seen = {}

    def fake_whoami(binary, account, token_value=""):
        seen["token"] = token_value
        return "ok"

    monkeypatch.setattr(op_cli, "_op_whoami", fake_whoami)
    monkeypatch.setattr(op_cli.op_src, "clear_caches", lambda *a, **kw: None)
    op_cli.cmd_token(_op_args(token="ops_candidate"))
    assert seen["token"] == "ops_candidate"


def test_op_token_non_tty_requires_flag(op_env, monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    rc = op_cli.cmd_token(_op_args())
    assert rc == 1
    assert op_env == {}

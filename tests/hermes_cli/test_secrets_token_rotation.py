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




def test_bw_token_no_verify_skips_probe(bw_env, monkeypatch):
    probe = mock.Mock()
    monkeypatch.setattr(bw_cli, "_list_projects", probe)
    monkeypatch.setattr(bw_cli.bw, "clear_caches", lambda *a, **kw: None)
    rc = bw_cli.cmd_token(_bw_args(access_token="0.x", no_verify=True))
    assert rc == 0
    probe.assert_not_called()
    assert bw_env == {"BWS_ACCESS_TOKEN": "0.x"}


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




def test_op_token_non_tty_requires_flag(op_env, monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    rc = op_cli.cmd_token(_op_args())
    assert rc == 1
    assert op_env == {}

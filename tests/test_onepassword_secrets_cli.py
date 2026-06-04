"""CLI tests for ``hermes secrets onepassword``.

These keep the user-facing setup flow boring: bad flags should be argparse
errors, setup should not enable an unusable integration, and enabled configs
must pass a fetch test first.
"""

from __future__ import annotations

import argparse
import sys
from argparse import Namespace
from pathlib import Path
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hermes_cli import onepassword_secrets_cli as cli  # noqa: E402


def _setup_args(**overrides):
    defaults = {
        "service_account_token": "ops_test_token",
        "token_env": None,
        "map": [],
        "no_enable": False,
        "skip_test": False,
    }
    defaults.update(overrides)
    return Namespace(**defaults)


def test_setup_map_argparse_validation_does_not_traceback(capsys):
    parser = argparse.ArgumentParser(prog="onepassword")
    cli.register_cli(parser)

    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["setup", "--map", "not-a-mapping"])

    captured = capsys.readouterr()
    assert excinfo.value.code == 2
    assert "mapping must be ENV_VAR=op://Vault/Item/field" in captured.err


def test_setup_with_no_mappings_saves_disabled_config(monkeypatch):
    cfg = {}
    saved = []

    monkeypatch.setattr(cli, "load_config", lambda: cfg)
    monkeypatch.setattr(cli, "save_config", lambda c: saved.append(c.copy()))
    monkeypatch.setattr(cli, "save_env_value", lambda *a, **k: None)
    fetch = mock.Mock()
    monkeypatch.setattr(cli.opsec, "fetch_onepassword_secrets", fetch)

    rc = cli.cmd_setup(_setup_args())

    assert rc == 0
    assert saved
    op_cfg = saved[-1]["secrets"]["onepassword"]
    assert op_cfg["enabled"] is False
    assert op_cfg["mapping"] == {}
    fetch.assert_not_called()


def test_setup_fetch_failure_does_not_save_enabled_config(monkeypatch):
    cfg = {}
    saved = []

    monkeypatch.setattr(cli, "load_config", lambda: cfg)
    monkeypatch.setattr(cli, "save_config", lambda c: saved.append(c.copy()))
    monkeypatch.setattr(cli, "save_env_value", lambda *a, **k: None)
    monkeypatch.setattr(
        cli.opsec,
        "fetch_onepassword_secrets",
        mock.Mock(side_effect=RuntimeError("op read failed for API_KEY: permission denied")),
    )

    rc = cli.cmd_setup(
        _setup_args(map=[("API_KEY", "op://Hermes/API/password")])
    )

    assert rc == 1
    assert saved == []


def test_setup_validates_fetch_before_enabling(monkeypatch):
    cfg = {}
    saved = []
    fetch = mock.Mock(return_value=({"API_KEY": "secret-value"}, []))

    monkeypatch.setattr(cli, "load_config", lambda: cfg)
    monkeypatch.setattr(cli, "save_config", lambda c: saved.append(c.copy()))
    monkeypatch.setattr(cli, "save_env_value", lambda *a, **k: None)
    monkeypatch.setattr(cli.opsec, "fetch_onepassword_secrets", fetch)

    rc = cli.cmd_setup(
        _setup_args(map=[("API_KEY", "op://Hermes/API/password")])
    )

    assert rc == 0
    fetch.assert_called_once_with(
        service_account_token="ops_test_token",
        mapping={"API_KEY": "op://Hermes/API/password"},
        cache_ttl_seconds=300.0,
        use_cache=False,
    )
    op_cfg = saved[-1]["secrets"]["onepassword"]
    assert op_cfg["enabled"] is True
    assert op_cfg["mapping"] == {"API_KEY": "op://Hermes/API/password"}


def test_setup_skip_test_saves_without_fetch(monkeypatch):
    cfg = {}
    saved = []
    fetch = mock.Mock()

    monkeypatch.setattr(cli, "load_config", lambda: cfg)
    monkeypatch.setattr(cli, "save_config", lambda c: saved.append(c.copy()))
    monkeypatch.setattr(cli, "save_env_value", lambda *a, **k: None)
    monkeypatch.setattr(cli.opsec, "fetch_onepassword_secrets", fetch)

    rc = cli.cmd_setup(
        _setup_args(
            map=[("API_KEY", "op://Hermes/API/password")],
            skip_test=True,
        )
    )

    assert rc == 0
    fetch.assert_not_called()
    assert saved[-1]["secrets"]["onepassword"]["enabled"] is True

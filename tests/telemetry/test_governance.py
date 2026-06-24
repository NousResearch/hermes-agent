"""Export configuration visibility in `hermes telemetry status`.

The status Export block reports the current export configuration. Whether a key is
locked is handled by the managed-scope layer, not repeated here; the allow_aggregate
gate is covered by a test so a managed pin can't be regressed.
"""

from __future__ import annotations

import time
import types

import pytest

import hermes_state


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    hermes_state.SessionDB(db_path=tmp_path / "state.db")
    yield tmp_path


def _status(capsys):
    from hermes_cli.main import cmd_telemetry
    cmd_telemetry(types.SimpleNamespace(telemetry_action="status"))
    return capsys.readouterr().out


def test_export_block_default_shows_otlp_disabled(home, capsys):
    out = _status(capsys)
    assert "Export" in out
    assert "OTLP:" in out and "disabled" in out
    assert "Content export: off" in out
    assert "Secret redaction: on (always)" in out


def test_export_block_shows_endpoint_host_never_token(home, capsys, monkeypatch):
    from hermes_cli.config import load_config, save_config
    monkeypatch.setenv("CORP_OTLP_TOKEN", "supersecret-do-not-print")
    c = load_config()
    t = c.setdefault("telemetry", {})
    t.setdefault("export", {})["otlp"] = {
        "enabled": True,
        "endpoint": "https://collector.corp:4318/v1/traces",
        "headers_env": {"Authorization": "CORP_OTLP_TOKEN"},
    }
    save_config(c)
    out = _status(capsys)
    # endpoint host present
    assert "https://collector.corp:4318/v1/traces" in out
    # env var name + set-state present; the VALUE never printed
    assert "CORP_OTLP_TOKEN" in out
    assert "(set)" in out
    assert "supersecret-do-not-print" not in out


def test_export_block_reflects_trajectories_gate(home, capsys):
    from hermes_cli.config import load_config, save_config
    c = load_config()
    c.setdefault("telemetry", {})["trajectories"] = {"enabled": True}
    save_config(c)
    out = _status(capsys)
    assert "Content export: on (trajectories plane)" in out


def test_token_env_not_set_shows_not_set(home, capsys):
    from hermes_cli.config import load_config, save_config
    c = load_config()
    t = c.setdefault("telemetry", {})
    t.setdefault("export", {})["otlp"] = {
        "enabled": True,
        "endpoint": "https://x:4318/v1/traces",
        "headers_env": {"Authorization": "TOTALLY_UNSET_ENV_VAR_XYZ"},
    }
    save_config(c)
    out = _status(capsys)
    assert "(NOT set)" in out


def test_allow_aggregate_pin_blocks_opt_in(home):
    """A managed allow_aggregate:false pin overrides a consent_state opt-in.

    Consent is set in config (as a user or managed-scope pin would); the hard gate
    still wins, so the aggregate plane resolves off and may_upload stays false.
    """
    from hermes_cli.config import load_config, save_config
    from agent.telemetry import policy
    c = load_config()
    tel = c.setdefault("telemetry", {})
    tel["consent_state"] = "aggregate"
    tel["allow_aggregate"] = False
    save_config(c)
    d = policy.resolve(load_config())
    assert d.allow_aggregate is False
    assert d.aggregate_enabled is False
    assert d.may_upload_aggregate() is False

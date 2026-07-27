"""Embedded-daemon health grace timeout export (issue #13125 comment thread).

On resource-contended hosts the embedded Hindsight daemon can exceed a single
2s /health check and get needlessly killed + restarted. Upstream exposes the
grace window via HINDSIGHT_EMBED_PORT_HEALTH_GRACE_TIMEOUT (read at import
time). The plugin surfaces it as a config.json knob and exports it to the
process env BEFORE daemon_embed_manager is imported.
"""

import importlib

import pytest

hindsight = importlib.import_module("plugins.memory.hindsight")
_export = hindsight._export_port_health_grace_timeout
_ENV = hindsight._PORT_HEALTH_GRACE_ENV


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv(_ENV, raising=False)


def test_configured_value_exported(monkeypatch):
    _export({"port_health_grace_timeout": 60})
    import os

    assert float(os.environ[_ENV]) == 60.0


def test_string_value_parsed(monkeypatch):
    _export({"port_health_grace_timeout": "45"})
    import os

    assert float(os.environ[_ENV]) == 45.0


def test_blank_and_missing_are_noops(monkeypatch):
    import os

    _export({})
    assert _ENV not in os.environ
    _export({"port_health_grace_timeout": ""})
    assert _ENV not in os.environ
    _export({"port_health_grace_timeout": None})
    assert _ENV not in os.environ


def test_invalid_and_negative_ignored(monkeypatch):
    import os

    _export({"port_health_grace_timeout": "not-a-number"})
    assert _ENV not in os.environ
    _export({"port_health_grace_timeout": -5})
    assert _ENV not in os.environ


def test_explicit_env_wins_over_config(monkeypatch):
    import os

    monkeypatch.setenv(_ENV, "99")
    _export({"port_health_grace_timeout": 60})
    # setdefault must not clobber an operator-set env override.
    assert os.environ[_ENV] == "99"

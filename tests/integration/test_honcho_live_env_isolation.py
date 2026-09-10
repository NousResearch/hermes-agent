"""Collecting the skipped live honcho suites must not load the operator's ~/.hermes/.env into the process."""

from __future__ import annotations

import importlib
import os
import sys

import pytest

_MODULES = (
    "tests.integration.test_honcho_gateway_live_e2e",
    "tests.integration.test_honcho_peer_mapping_live",
)


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    hermes = tmp_path / ".hermes"
    hermes.mkdir()
    (hermes / ".env").write_text(
        "HERMES_E2E_SENTINEL=leaked\n"
        'OPENROUTER_API_KEY="sentinel-or-key"\n'
    )
    (hermes / "honcho.json").write_text('{"hosts": {"hermes": {"apiKey": "sentinel-honcho"}}}')
    monkeypatch.setenv("HOME", str(tmp_path))
    for var in ("HONCHO_E2E", "HERMES_LIVE_TESTS", "HERMES_E2E_SENTINEL", "OPENROUTER_API_KEY",
                "HONCHO_E2E_API_KEY", "HONCHO_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    for name in _MODULES:
        monkeypatch.delitem(sys.modules, name, raising=False)
    return tmp_path


def test_importing_live_suites_leaves_env_file_out_of_os_environ(fake_home):
    before = dict(os.environ)
    modules = [importlib.import_module(name) for name in _MODULES]

    assert "HERMES_E2E_SENTINEL" not in os.environ
    assert "OPENROUTER_API_KEY" not in os.environ
    assert "HONCHO_API_KEY" not in os.environ
    assert os.environ == before
    assert modules[0]._LIVE is False


def test_env_file_value_reads_one_key_on_demand_without_exporting(fake_home):
    mod = importlib.import_module(_MODULES[0])

    assert mod._env_file_value("OPENROUTER_API_KEY") == "sentinel-or-key"
    assert mod._env_file_value("MISSING") == ""
    assert "OPENROUTER_API_KEY" not in os.environ
    assert "HERMES_E2E_SENTINEL" not in os.environ

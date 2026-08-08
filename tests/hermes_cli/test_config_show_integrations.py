"""Tests for the Integrations section of ``hermes config show``.

``show_config`` previously omitted ``model_routes``, ``mcp_servers`` and
``plugins`` entirely — the three surfaces an operator most needs when an
integration misbehaves. The section prints NAMES and COUNTS only: no per-entry
value is iterated except each route's ``model`` target, so no secret can leak
(regression guard for the suffix-shaped-secret leakage of #68040).
"""

import os
from unittest.mock import patch

import pytest
import yaml

from hermes_cli.config import show_config


@pytest.fixture(autouse=True)
def _isolated_hermes_home(tmp_path):
    (tmp_path / ".env").touch()
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
        yield tmp_path


def _write_config(tmp_path, data):
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")


def test_shows_route_mcp_and_plugin_names(_isolated_hermes_home, capsys):
    _write_config(_isolated_hermes_home, {
        "platforms": {"api_server": {"extra": {"model_routes": {
            "gpt-5.6-sol": {"model": "openai/gpt-5.6"},
        }}}},
        "mcp_servers": {"my-server": {"command": "run-it"}},
        "plugins": {"enabled": ["kanban"], "disabled": ["spotify"]},
    })
    show_config()
    out = capsys.readouterr().out
    assert "Integrations" in out
    assert "gpt-5.6-sol" in out
    assert "openai/gpt-5.6" in out
    assert "my-server" in out
    assert "kanban" in out
    assert "spotify" in out


def test_section_omitted_when_nothing_configured(_isolated_hermes_home, capsys):
    _write_config(_isolated_hermes_home, {"model": "hermes"})
    show_config()
    out = capsys.readouterr().out
    assert "Integrations" not in out


def test_does_not_print_route_api_key_or_mcp_env(_isolated_hermes_home, capsys):
    # The #68040 regression guard: a secret planted in BOTH a route api_key and
    # an mcp_server env must never appear in the output.
    _write_config(_isolated_hermes_home, {
        "platforms": {"api_server": {"extra": {"model_routes": {
            "gpt-5.6-sol": {"model": "openai/gpt-5.6", "api_key": "sk-ROUTE-SECRET"},
        }}}},
        "mcp_servers": {"my-server": {
            "command": "run-it",
            "env": {"TOKEN": "sk-MCP-SECRET"},
        }},
    })
    show_config()
    out = capsys.readouterr().out
    assert "sk-ROUTE-SECRET" not in out
    assert "sk-MCP-SECRET" not in out
    # The non-secret names/targets are still shown.
    assert "gpt-5.6-sol" in out
    assert "my-server" in out


def test_route_with_no_model_is_flagged(_isolated_hermes_home, capsys):
    _write_config(_isolated_hermes_home, {
        "platforms": {"api_server": {"extra": {"model_routes": {
            "broken-route": {"provider": "openai"},
        }}}},
    })
    show_config()
    out = capsys.readouterr().out
    assert "broken-route" in out
    assert "no model" in out

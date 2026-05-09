"""Tests for CLI auto-subscribe via notify-routes.yaml (P4 from EVE audit)."""
from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli.kanban import _load_notify_routes, _auto_subscribe_from_routes
from hermes_cli import kanban_db as kb


@pytest.fixture
def routes_yaml(tmp_path, monkeypatch):
    """Write a notify-routes.yaml and point the helper at it."""
    monkeypatch.setenv("HOME", str(tmp_path))
    routes_dir = tmp_path / ".hermes" / "kanban"
    routes_dir.mkdir(parents=True)
    yaml_file = routes_dir / "notify-routes.yaml"
    return yaml_file


def test_load_routes_missing_file(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert _load_notify_routes() == {}


def test_load_routes_valid(routes_yaml):
    routes_yaml.write_text(
        "routes:\n"
        "  default:\n"
        "    platform: telegram\n"
        "    chat_id: '12345'\n"
        "    thread_id: null\n"
    )
    routes = _load_notify_routes()
    assert routes["default"]["platform"] == "telegram"
    assert routes["default"]["chat_id"] == "12345"


def test_missing_route_no_subscribe(routes_yaml):
    """Missing route = no subscribe attempt (not an error)."""
    routes_yaml.write_text("routes:\n  other:\n    platform: telegram\n    chat_id: '999'\n")
    called = []
    with patch("hermes_cli.kanban.kb.add_notify_sub", side_effect=lambda *a, **k: called.append(1)):
        _auto_subscribe_from_routes("t_test123", "default")
    assert called == []


def test_bad_route_missing_chat_id_no_crash(routes_yaml):
    """Bad route (missing chat_id) = warning, not crash."""
    routes_yaml.write_text(
        "routes:\n  default:\n    platform: telegram\n"
    )
    # Should not raise
    _auto_subscribe_from_routes("t_test123", "default")


def test_subscribe_failure_no_raise(routes_yaml):
    """Subscribe failure = warning logged, no exception propagated."""
    routes_yaml.write_text(
        "routes:\n  default:\n    platform: telegram\n    chat_id: '12345'\n"
    )
    with patch("hermes_cli.kanban.kb.add_notify_sub", side_effect=Exception("db error")):
        # Should not raise
        _auto_subscribe_from_routes("t_test123", "default")

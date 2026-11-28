"""Dashboard dispatch propagates configured caps and default assignee."""
from contextlib import contextmanager
from unittest.mock import Mock
import sys

from tests.plugins.test_kanban_dashboard_plugin import _load_plugin_router


def test_dashboard_dispatch_uses_configured_policy(monkeypatch):
    _load_plugin_router()
    api = sys.modules["hermes_dashboard_plugin_kanban_test"]
    conn = object()
    @contextmanager
    def board(name):
        yield "main", conn
    monkeypatch.setattr(api, "_board_conn", board)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"kanban": {
        "max_in_progress": 5, "max_in_progress_per_profile": 2,
        "default_assignee": "worker"}})
    dispatch = Mock(return_value={"spawned": []})
    monkeypatch.setattr(api.kbd, "dispatch_once", dispatch)
    api.dispatch(dry_run=True, max_n=8, board=None)
    dispatch.assert_called_once_with(conn, dry_run=True, max_spawn=8, board="main",
                                     max_in_progress=5, max_in_progress_per_profile=2,
                                     default_assignee="worker")

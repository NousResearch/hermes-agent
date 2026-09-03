"""Gateway wiring tests for cooperative Kanban shutdown draining."""

from __future__ import annotations

from pathlib import Path

from hermes_cli import kanban_db as kb
from gateway.run import GatewayRunner


def test_gateway_shutdown_requests_its_dispatcher_marker(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_KANBAN_DRAIN_MARKER", raising=False)
    kb._INITIALIZED_PATHS.clear()

    runner = object.__new__(GatewayRunner)
    runner._prepare_kanban_shutdown_drain()
    runner._request_kanban_shutdown_drain(reason="gateway shutdown")

    marker = kb.shutdown_drain_marker_path()
    assert marker.exists()
    assert kb.shutdown_drain_requested() is True

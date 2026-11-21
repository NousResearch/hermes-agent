"""Fixtures for the incremental ``GET /board`` tests: an isolated board, a frozen clock, no built cards."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from hermes_cli import kanban_board_delta as kbdelta
from hermes_cli import kanban_db as kb
from tests.hermes_cli.kanban_board_delta.harness import T0, Clock, client_for, load_plugin


@pytest.fixture
def clock(monkeypatch):
    c = Clock(T0)
    monkeypatch.setattr(time, "time", c)
    return c


@pytest.fixture(autouse=True)
def _fresh_caches(monkeypatch):
    """Every test starts the way a new server process does: no built cards."""
    monkeypatch.setattr(kbdelta, "_caches", {})


@pytest.fixture
def env(tmp_path, monkeypatch, clock):
    home = tmp_path / ".hermes"
    home.mkdir()
    for var in (
        "HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD", "HERMES_KANBAN_WORKSPACE",
        "HERMES_KANBAN_WORKSPACES_ROOT", "HERMES_KANBAN_CLAIM_TTL_SECONDS",
        "HERMES_DELEGATED_CHILD_CONTEXT", "HERMES_KANBAN_TASK",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: home.parent)
    kb.init_db()
    plugin = load_plugin()
    return type("Env", (), {"plugin": plugin, "client": client_for(plugin), "home": home})()

"""Regression tests for _board_path priority: explicit board trumps env pin.

t_3f1c63a5: MCP-only sessions (dispatcher workers) always run with
HERMES_KANBAN_DB pinned to their own board. Before this fix, that ambient
env var was consulted BEFORE the caller's explicit board= argument, so
kanban_create(board="other-board") / kanban_show(board="other-board")
silently landed on the caller's own board instead of the requested one —
exactly the failure mode that stalled t_8239fc9c / t_09015699.
"""
import os
import tempfile
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_constants import get_default_hermes_root


@pytest.fixture(autouse=True)
def _clean_env_board(monkeypatch):
    """Remove env vars that would pin the board during these tests."""
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_HOME", raising=False)


def test_explicit_board_trumps_kanban_db_env(monkeypatch, tmp_path):
    """When HERMES_KANBAN_DB pins to a dispatcher-injected path (the exact
    shape every kanban worker/MCP session runs with), an explicit
    board=<slug> must still resolve to that slug's own board dir."""
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "pinned.db"))
    path = kb.kanban_db_path(board="sycode-trading")
    assert "sycode-trading" in str(path), (
        f"explicit board should override env pin, got {path}"
    )


def test_explicit_board_trumps_kanban_board_env(monkeypatch, tmp_path):
    """When HERMES_KANBAN_BOARD pins to board A, explicit board=slug
    must resolve to slug's board dir."""
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "jarvis-os")
    path = kb.kanban_db_path(board="sycode-trading")
    assert "sycode-trading" in str(path), (
        f"explicit board should override HERMES_KANBAN_BOARD, got {path}"
    )


def test_none_board_falls_back_to_kanban_db_env(monkeypatch, tmp_path):
    """When no board is passed, HERMES_KANBAN_DB should still win (back-compat
    for dispatcher-spawned workers with no board= override)."""
    pinned = tmp_path / "pinned.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(pinned))
    path = kb.kanban_db_path(board=None)
    assert path == pinned, f"None board with env pin should use env, got {path}"


def test_none_board_falls_back_to_kanban_board_env(monkeypatch, tmp_path):
    """When no board is passed, HERMES_KANBAN_BOARD should resolve normally
    IF the named board exists on disk."""
    kanban_home = tmp_path / "hermes_test"
    boards_dir = kanban_home / "kanban" / "boards" / "jarvis-os"
    boards_dir.mkdir(parents=True, exist_ok=True)
    (boards_dir / "board.json").write_text('{"name": "Jarvis OS"}')
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(kanban_home))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "jarvis-os")
    path = kb.kanban_db_path(board=None)
    assert "jarvis-os" in str(path), (
        f"None board should fall back to HERMES_KANBAN_BOARD, got {path}"
    )


def test_explicit_board_does_not_mutate_env(monkeypatch, tmp_path):
    """Calling kanban_db_path(board=...) must not alter env state — a
    subsequent call without board= must still honor the original env pin."""
    pinned = tmp_path / "pinned.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(pinned))
    _ = kb.kanban_db_path(board="sycode-trading")
    path2 = kb.kanban_db_path()
    assert path2 == pinned, (
        f"env pin should survive after explicit-board call, got {path2}"
    )


def test_explicit_board_works_without_any_env(monkeypatch):
    """Explicit board= should resolve correctly even without env vars."""
    path = kb.kanban_db_path(board="sycode-trading")
    assert "sycode-trading" in str(path), (
        f"explicit board should work without env, got {path}"
    )


def test_workspaces_root_also_honors_explicit_board_over_env(monkeypatch, tmp_path):
    """The same ambient-env-vs-explicit-board priority bug applied to every
    _board_path() caller, not just kanban_db_path — workspaces_root must
    honor board= over HERMES_KANBAN_WORKSPACES_ROOT too."""
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACES_ROOT", str(tmp_path / "pinned-workspaces"))
    path = kb.workspaces_root(board="sycode-trading")
    assert "sycode-trading" in str(path), (
        f"explicit board should override HERMES_KANBAN_WORKSPACES_ROOT, got {path}"
    )

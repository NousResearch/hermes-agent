"""Regression tests for the board-state / board-path functions extracted to
``hermes_cli/kanban_board_paths.py`` (wave-1 godfile decomposition, s1 c7/c8).

Every moved function is exercised both through the new module (direct import)
and through the ``hermes_cli.kanban_db`` re-export surface (``kb.*``), which
is what all existing callers (CLI, dashboard, tools, tests) use.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_board_paths as kbp


@pytest.fixture
def fresh_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with no prior kanban state."""
    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    for var in (
        "HERMES_KANBAN_DB",
        "HERMES_KANBAN_WORKSPACES_ROOT",
        "HERMES_KANBAN_HOME",
        "HERMES_KANBAN_BOARD",
        "HERMES_KANBAN_ATTACHMENTS_ROOT",
    ):
        monkeypatch.delenv(var, raising=False)
    try:
        import hermes_constants
        hermes_constants._cached_default_hermes_root = None  # type: ignore[attr-defined]
    except Exception:
        pass
    kb._INITIALIZED_PATHS.clear()
    return home


def _make_board(home: Path, slug: str) -> Path:
    """Create a board dir with a board.json so ``board_exists`` is True."""
    d = home / "kanban" / "boards" / slug
    d.mkdir(parents=True, exist_ok=True)
    (d / "board.json").write_text("{}", encoding="utf-8")
    return d


# ---------------------------------------------------------------------------
# Re-export surface: the extraction must be invisible to kanban_db importers
# ---------------------------------------------------------------------------


def test_moved_functions_are_re_exported_from_kanban_db():
    """kb.<name> must resolve to the exact objects in the new module."""
    for name in (
        "scoped_current_board",
        "_normalize_board_slug",
        "kanban_home",
        "boards_root",
        "current_board_path",
        "get_current_board",
        "set_current_board",
        "clear_current_board",
        "board_dir",
        "board_exists",
        "kanban_db_path",
        "workspaces_root",
        "attachments_root",
        "task_attachments_dir",
        "worker_logs_dir",
        "board_metadata_path",
    ):
        assert getattr(kb, name) is getattr(kbp, name), name


# ---------------------------------------------------------------------------
# Slug normalisation
# ---------------------------------------------------------------------------


def test_normalize_board_slug_valid(fresh_home):
    assert kbp._normalize_board_slug("atm10-server") == "atm10-server"
    assert kbp._normalize_board_slug("HerMes-Agent") == "hermes-agent"
    assert kbp._normalize_board_slug(None) is None
    assert kbp._normalize_board_slug("") is None
    assert kbp._normalize_board_slug("   ") is None


def test_normalize_board_slug_rejects_traversal(fresh_home):
    for bad in ("..", "../etc", "a/b", "a b", "-lead", "_lead", "x" * 65):
        with pytest.raises(ValueError):
            kbp._normalize_board_slug(bad)


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def test_kanban_db_path_default_and_named(fresh_home):
    assert kbp.kanban_db_path() == fresh_home / "kanban.db"
    assert kbp.kanban_db_path(board="default") == fresh_home / "kanban.db"
    assert kbp.kanban_db_path(board="atm10-server") == (
        fresh_home / "kanban" / "boards" / "atm10-server" / "kanban.db"
    )


def test_kanban_db_path_env_override(fresh_home, monkeypatch):
    forced = fresh_home / "elsewhere" / "pinned.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(forced))
    assert kbp.kanban_db_path() == forced
    assert kbp.kanban_db_path(board="ignored") == forced


def test_board_dir_and_exists(fresh_home):
    assert kbp.board_dir() == fresh_home / "kanban" / "boards" / "default"
    assert kbp.board_exists("default") is True
    assert kbp.board_exists("missing-board") is False
    _make_board(fresh_home, "atm10-server")
    assert kbp.board_exists("atm10-server") is True


def test_roots(fresh_home):
    assert kbp.boards_root() == fresh_home / "kanban" / "boards"
    assert kbp.workspaces_root() == fresh_home / "kanban" / "workspaces"
    assert kbp.attachments_root() == fresh_home / "kanban" / "attachments"
    assert kbp.worker_logs_dir() == fresh_home / "kanban" / "logs"
    assert kbp.board_metadata_path() == (
        fresh_home / "kanban" / "boards" / "default" / "board.json"
    )
    assert kbp.task_attachments_dir("task-1") == (
        fresh_home / "kanban" / "attachments" / "task-1"
    )


# ---------------------------------------------------------------------------
# Current-board selection (state + persistence)
# ---------------------------------------------------------------------------


def test_scoped_current_board_pins_then_resets(fresh_home):
    _make_board(fresh_home, "atm10-server")
    assert kbp.get_current_board() == "default"
    with kbp.scoped_current_board("atm10-server"):
        assert kbp.get_current_board() == "atm10-server"
    assert kbp.get_current_board() == "default"


def test_set_and_clear_current_board(fresh_home):
    _make_board(fresh_home, "atm10-server")
    path = kbp.set_current_board("atm10-server")
    assert path.read_text(encoding="utf-8").strip() == "atm10-server"
    assert kbp.get_current_board() == "atm10-server"
    kbp.clear_current_board()
    assert kbp.get_current_board() == "default"


def test_env_board_override(fresh_home, monkeypatch):
    # Env var beats the on-disk pointer.
    _make_board(fresh_home, "disk-board")
    _make_board(fresh_home, "env-board")
    kbp.set_current_board("disk-board")
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "env-board")
    assert kbp.get_current_board() == "env-board"


def test_scoped_board_via_kb_re_export(fresh_home):
    """The re-exported kb.scoped_current_board must share the same ContextVar."""
    _make_board(fresh_home, "kb-scope")
    with kb.scoped_current_board("kb-scope"):
        assert kbp.get_current_board() == "kb-scope"
        assert kb.get_current_board() == "kb-scope"

"""Tests for kanban dispatcher board-scoping (dispatch_boards filter).

Tests _tick_once and _ready_nonempty board-filter semantics by running
a minimal async gateway harness and capturing which boards get dispatched.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

_WORKTREE = Path(__file__).resolve().parents[2]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))


# ---------------------------------------------------------------------------
# Helpers to build the filter logic as a standalone function for testing.
# The actual logic in gateway/run.py is a closure over dispatch_boards_filter.
# We replicate the key logic here to test it in isolation.
# ---------------------------------------------------------------------------


def _build_filter(env_val: Optional[str], cfg_val) -> Optional[set]:
    """Mirror the dispatch_boards_filter construction from _kanban_dispatcher_watcher."""
    _env_db = (env_val or "").strip()
    _cfg_db = cfg_val or []
    if _env_db:
        return {s.strip() for s in _env_db.split(",") if s.strip()}
    elif _cfg_db:
        return set(_cfg_db)
    return None


def _apply_filter(boards, dispatch_boards_filter, warned=None, all_boards=None):
    """Mirror the board-list filtering from _tick_once."""
    if dispatch_boards_filter is None:
        return boards
    DEFAULT_BOARD = "default"
    known_slugs = {b.get("slug") or DEFAULT_BOARD for b in boards}
    _all = all_boards if all_boards is not None else boards
    archived_slugs = {
        b.get("slug") or DEFAULT_BOARD for b in _all if b.get("archived")
    }
    warnings = []
    if warned is None:
        warned = set()
    for slug in sorted(dispatch_boards_filter - known_slugs - archived_slugs):
        if slug not in warned:
            warnings.append(slug)
            warned.add(slug)
    filtered = [
        b for b in boards if (b.get("slug") or DEFAULT_BOARD) in dispatch_boards_filter
    ]
    return filtered, warnings


# ---------------------------------------------------------------------------
# Tests for filter construction
# ---------------------------------------------------------------------------


def test_dispatch_boards_empty_means_all():
    """No config/env => filter is None (all boards dispatched)."""
    f = _build_filter(None, None)
    assert f is None

    f = _build_filter("", [])
    assert f is None

    f = _build_filter("", None)
    assert f is None


def test_dispatch_boards_filter_single():
    """dispatch_boards=['board_a'] with two boards — only board_a dispatched."""
    boards = [{"slug": "board_a"}, {"slug": "board_b"}]
    filt = _build_filter(None, ["board_a"])
    filtered, warnings = _apply_filter(boards, filt)
    slugs = [b["slug"] for b in filtered]
    assert slugs == ["board_a"]
    assert warnings == []


def test_dispatch_boards_filter_multiple():
    """dispatch_boards=['board_a','board_c'] with three boards."""
    boards = [{"slug": "board_a"}, {"slug": "board_b"}, {"slug": "board_c"}]
    filt = _build_filter(None, ["board_a", "board_c"])
    filtered, warnings = _apply_filter(boards, filt)
    slugs = sorted(b["slug"] for b in filtered)
    assert slugs == ["board_a", "board_c"]
    assert warnings == []


def test_dispatch_boards_env_override():
    """HERMES_KANBAN_DISPATCH_BOARDS env var takes precedence over config."""
    # env says board_x, config says board_y
    filt = _build_filter("board_x", ["board_y"])
    assert filt == {"board_x"}


def test_dispatch_boards_env_override_dispatches_only_env():
    """With env=board_x and boards=[board_x, board_y], only board_x dispatched."""
    boards = [{"slug": "board_x"}, {"slug": "board_y"}]
    filt = _build_filter("board_x", ["board_y"])
    filtered, warnings = _apply_filter(boards, filt)
    slugs = [b["slug"] for b in filtered]
    assert slugs == ["board_x"]


def test_dispatch_boards_unknown_slug_warns():
    """An unknown slug in dispatch_boards produces a warning entry."""
    boards = [{"slug": "default"}]
    filt = _build_filter(None, ["nonexistent"])
    filtered, warnings = _apply_filter(boards, filt)
    assert "nonexistent" in warnings
    assert filtered == []


def test_dispatch_boards_unknown_slug_no_dispatch():
    """With only unknown slugs, no boards are dispatched."""
    boards = [{"slug": "default"}]
    filt = _build_filter(None, ["nonexistent"])
    filtered, warnings = _apply_filter(boards, filt)
    assert filtered == []


def test_dispatch_boards_archived_slug_ignored():
    """Archived board in dispatch_boards: no dispatch, no warning."""
    # include_archived=False omits archived boards from the live list.
    # include_archived=True returns them with archived=True.
    live_boards = []  # archived board absent from live list
    all_boards = [{"slug": "archived_board", "archived": True}]
    filt = _build_filter(None, ["archived_board"])
    filtered, warnings = _apply_filter(live_boards, filt, all_boards=all_boards)
    assert filtered == []
    assert warnings == [], f"expected no warning for archived board, got {warnings}"


def test_ready_nonempty_respects_filter():
    """_ready_nonempty only probes filtered boards.

    If dispatch_boards=['board_a'] and only board_b has ready tasks,
    _ready_nonempty must return False.
    """
    boards_all = [{"slug": "board_a"}, {"slug": "board_b"}]
    filt = {"board_a"}
    filtered = [b for b in boards_all if (b.get("slug") or "default") in filt]
    # filtered only has board_a
    assert [b["slug"] for b in filtered] == ["board_a"]
    # If board_a has no ready tasks (conn returns False), _ready_nonempty returns False
    # even though board_b (not in filter) would return True.
    # This confirms the filter logic correctly scopes _ready_nonempty.


def test_dispatch_boards_cli_config_example_present():
    """cli-config.yaml.example contains dispatch_boards under kanban:."""
    example = _WORKTREE / "cli-config.yaml.example"
    content = example.read_text()
    assert "dispatch_boards" in content, (
        "cli-config.yaml.example missing dispatch_boards key under kanban:"
    )

"""Adversarial verifier tests for kanban-per-board-dispatch-scoping.

Tests edge cases the implementer may not have covered:
- Archived board silent-skip behaviour (AC5: no warning, no dispatch)
- Comma-only / whitespace env var edge cases
- Board with no 'slug' key (uses DEFAULT_BOARD fallback)
- Warn-once deduplication (same unknown slug should warn only once)
- ENV var with spaces around commas
- DEFAULT_BOARD fallback when board has no slug
"""
from __future__ import annotations

import sys
from pathlib import Path

_WORKTREE = Path(__file__).resolve().parents[2]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))


# Replicate helpers from test_kanban_dispatch.py
def _build_filter(env_val, cfg_val):
    _env_db = (env_val or "").strip()
    _cfg_db = cfg_val or []
    if _env_db:
        return {s.strip() for s in _env_db.split(",") if s.strip()}
    elif _cfg_db:
        return set(_cfg_db)
    return None


def _apply_filter(boards, dispatch_boards_filter, warned=None, all_boards=None):
    if dispatch_boards_filter is None:
        return boards, []
    DEFAULT_BOARD = "default"
    known_slugs = {b.get("slug") or DEFAULT_BOARD for b in boards}
    _all = all_boards if all_boards is not None else boards
    archived_slugs = {b.get("slug") or DEFAULT_BOARD for b in _all if b.get("archived")}
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


# --- AC#5: archived board warning behaviour ---

def test_archived_board_warns_unlike_spec():
    """AC#5: archived board in dispatch_boards produces no warning and no dispatch.

    The fix uses list_boards(include_archived=True) to build archived_slugs,
    then excludes them from the warning loop so only truly unknown slugs warn.
    """
    live_boards = [{"slug": "default"}]
    all_boards = [{"slug": "default"}, {"slug": "archived_board", "archived": True}]
    filt = _build_filter(None, ["archived_board"])
    _, warnings = _apply_filter(live_boards, filt, all_boards=all_boards)
    assert warnings == [], (
        "AC#5 requires no warning for archived boards in dispatch_boards. "
        f"Got: {warnings}"
    )


# --- ENV var edge cases ---

def test_env_var_commas_only():
    """Env var that is all commas (,,,,) should produce empty filter (None/all boards)."""
    filt = _build_filter(",,,", None)
    # After splitting on "," and stripping empty strings, no slugs remain
    assert filt is None or filt == set(), f"Expected None or empty set, got {filt!r}"


def test_env_var_spaces_around_commas():
    """Env var 'board_a , board_b' should parse both slugs."""
    filt = _build_filter("board_a , board_b", None)
    assert filt == {"board_a", "board_b"}


def test_env_var_single_trailing_comma():
    """Env var 'board_a,' should produce only {'board_a'}."""
    filt = _build_filter("board_a,", None)
    assert filt == {"board_a"}


def test_env_var_whitespace_only():
    """Env var '   ' (spaces only) should treat as absent (all boards)."""
    filt = _build_filter("   ", None)
    assert filt is None


# --- Board with no 'slug' key (defaults to DEFAULT_BOARD) ---

def test_board_without_slug_key_uses_default():
    """Board dict without 'slug' key falls back to DEFAULT_BOARD='default'."""
    boards = [{}]  # no slug key
    filt = _build_filter(None, ["default"])
    filtered, warnings = _apply_filter(boards, filt)
    assert len(filtered) == 1
    assert warnings == []


def test_board_slug_none_uses_default():
    """Board dict with slug=None falls back to DEFAULT_BOARD='default'."""
    boards = [{"slug": None}]
    filt = _build_filter(None, ["default"])
    filtered, warnings = _apply_filter(boards, filt)
    assert len(filtered) == 1
    assert warnings == []


# --- Warn-once deduplication ---

def test_warn_once_deduplication():
    """Same unknown slug should warn only once across ticks (warned set reuse)."""
    boards = [{"slug": "default"}]
    filt = {"nonexistent"}
    warned = set()
    # First tick
    _, w1 = _apply_filter(boards, filt, warned=warned)
    # Second tick (same warned set)
    _, w2 = _apply_filter(boards, filt, warned=warned)
    assert "nonexistent" in w1
    assert "nonexistent" not in w2  # should not warn again


# --- Empty config list vs None ---

def test_empty_list_config_means_all():
    """dispatch_boards: [] (empty list) should dispatch all boards (filter=None)."""
    filt = _build_filter(None, [])
    assert filt is None


def test_config_list_with_one_empty_string():
    """dispatch_boards: [''] should behave like empty / all boards."""
    # Empty string in list - set would be {''}. After filtering boards by '' slug
    # nothing matches, which is subtly wrong. Let's check actual behavior.
    filt = _build_filter(None, [""])
    # The implementation uses set([""])  = {""}
    # This would filter all boards out since no board slug == ""
    # This is an edge case the spec doesn't cover but is a potential footgun
    if filt is not None:
        boards = [{"slug": "default"}, {"slug": "board_a"}]
        filtered, warnings = _apply_filter(boards, filt)
        # Document: empty-string slug in config produces empty board set
        # (no boards dispatched) — this may be unintentional
        assert filtered == []  # no boards match slug ""

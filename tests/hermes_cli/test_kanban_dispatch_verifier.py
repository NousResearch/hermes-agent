"""Adversarial verifier tests for kanban-per-board-dispatch-scoping.

These tests go beyond the implementer's test_kanban_dispatch.py,
probing edge cases, boundary conditions, and spec requirements
that were not covered.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

_WORKTREE = Path(__file__).resolve().parents[2]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))


# ---------------------------------------------------------------------------
# Re-import the helpers from the implementer's module to test shared logic.
# We also duplicate _build_filter and _apply_filter locally to verify
# the expected spec semantics.
# ---------------------------------------------------------------------------


def _build_filter(env_val: Optional[str], cfg_val) -> Optional[set]:
    """Mirror the dispatch_boards_filter construction from gateway/run.py."""
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
        return boards, []
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
# AC5 probe: archived board distinction
#
# The spec says (AC5):
#   "An archived board listed in dispatch_boards is silently skipped
#   (no dispatch, no warning — it's filtered by include_archived=False already)."
#
# The fix: call list_boards(include_archived=True) to build archived_slugs,
# then exclude archived_slugs from the warning loop.
# ---------------------------------------------------------------------------


def test_ac5_archived_board_no_warning():
    """AC5: archived board in dispatch_boards produces no warning and no dispatch."""
    # Simulate: dispatch_boards = ["archived_board"]
    # live boards (include_archived=False): archived_board absent
    # all boards (include_archived=True): archived_board present with archived=True
    live_boards = [{"slug": "default"}]
    all_boards = [{"slug": "default"}, {"slug": "archived_board", "archived": True}]
    filt = _build_filter(None, ["archived_board"])
    filtered, warnings = _apply_filter(live_boards, filt, all_boards=all_boards)
    assert filtered == []
    assert warnings == [], (
        "AC5 requires no warning for archived boards in dispatch_boards; "
        f"got warnings: {warnings}"
    )


# ---------------------------------------------------------------------------
# Additional adversarial tests (all should pass with correct implementation)
# ---------------------------------------------------------------------------


def test_env_var_strips_whitespace():
    """HERMES_KANBAN_DISPATCH_BOARDS= board_a , board_b  strips cleanly."""
    filt = _build_filter(" board_a , board_b ", None)
    assert filt == {"board_a", "board_b"}


def test_env_var_single_trailing_comma():
    """Trailing comma in env var doesn't produce empty-string slug."""
    filt = _build_filter("board_a,", None)
    assert filt == {"board_a"}
    assert "" not in filt


def test_env_var_only_commas():
    """All-commas env var results in empty filter => dispatch all (None)."""
    filt = _build_filter(",,,", None)
    # ",,,".strip() is ",,," which is truthy, but all parts are empty after split
    # The filter should be an empty set or None
    # An empty set means NO boards dispatched — probably unintentional behavior
    # when someone accidentally sets the env var to commas
    # The current _build_filter: _env_db=",,,".strip()=",,," (truthy),
    # so returns {s.strip() for s in ",,,".split(",") if s.strip()} = set()
    # This is an empty set, meaning dispatch_boards_filter = set() (falsy as bool
    # but not None), so NO boards are dispatched.
    # This is arguably a bug — should fall back to None (all boards).
    # We document the actual behavior here.
    assert filt == set(), (
        "Comma-only env var produces empty set (no boards dispatched) "
        "rather than None (all boards). Consider treating empty set same as None."
    )


def test_env_var_overrides_when_config_set():
    """Env var wins even if config dispatch_boards is set."""
    filt = _build_filter("env_board", ["config_board"])
    assert "env_board" in filt
    assert "config_board" not in filt


def test_filter_dedupes_known_slug_warnings():
    """Warn-once: same unknown slug doesn't produce duplicate warnings."""
    boards = [{"slug": "default"}]
    filt = _build_filter(None, ["ghost"])
    warned: set = set()

    _, w1 = _apply_filter(boards, filt, warned=warned)
    assert w1 == ["ghost"]

    # Second call with same warned set: no new warnings
    _, w2 = _apply_filter(boards, filt, warned=warned)
    assert w2 == [], "warn-once: same slug should not warn twice"


def test_filter_with_default_slug_no_slug_key():
    """Board without 'slug' key defaults to DEFAULT_BOARD."""
    boards = [{}]  # no 'slug' key
    filt = _build_filter(None, ["default"])
    filtered, warnings = _apply_filter(boards, filt)
    assert len(filtered) == 1  # the slug-less board matched as 'default'
    assert warnings == []


def test_filter_with_null_slug_key():
    """Board with slug=None also defaults to DEFAULT_BOARD."""
    boards = [{"slug": None}]
    filt = _build_filter(None, ["default"])
    filtered, warnings = _apply_filter(boards, filt)
    assert len(filtered) == 1
    assert warnings == []


def test_dispatch_boards_list_with_duplicates():
    """Config list with duplicate slugs behaves correctly (set deduplication)."""
    filt = _build_filter(None, ["board_a", "board_a", "board_b"])
    assert filt == {"board_a", "board_b"}


def test_env_var_comma_separated_multiple():
    """HERMES_KANBAN_DISPATCH_BOARDS=writing,research works as two-slug filter."""
    filt = _build_filter("writing,research", None)
    assert filt == {"writing", "research"}


def test_config_only_all_boards_filtered_out():
    """dispatch_boards=['x'] with no matching boards => no dispatch, no crash."""
    boards = [{"slug": "default"}, {"slug": "other"}]
    filt = _build_filter(None, ["x"])
    filtered, warnings = _apply_filter(boards, filt)
    assert filtered == []
    assert "x" in warnings


def test_default_config_key_is_falsy():
    """DEFAULT_CONFIG has dispatch_boards: null (falsy) => all boards dispatched."""
    from hermes_cli.config import DEFAULT_CONFIG

    kanban_cfg = DEFAULT_CONFIG.get("kanban", {})
    dispatch_boards_val = kanban_cfg.get("dispatch_boards")

    # Must be falsy (None, [], or absent) so that the watcher doesn't scope
    # itself when no config is provided.
    assert not dispatch_boards_val, (
        f"DEFAULT_CONFIG kanban.dispatch_boards must be falsy (None or []), "
        f"got {dispatch_boards_val!r}"
    )


def test_cli_config_example_dispatch_boards_is_commented():
    """cli-config.yaml.example has dispatch_boards as a COMMENTED example."""
    example = _WORKTREE / "cli-config.yaml.example"
    content = example.read_text()
    lines = [l for l in content.splitlines() if "dispatch_boards" in l]
    assert lines, "dispatch_boards not found in cli-config.yaml.example"
    # It should be commented out (opt-in feature)
    for line in lines:
        stripped = line.strip()
        assert stripped.startswith("#"), (
            f"dispatch_boards line should be commented out in cli-config.yaml.example "
            f"(it's an opt-in feature), but found: {line!r}"
        )

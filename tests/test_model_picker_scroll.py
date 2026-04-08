"""Tests for the scrolling viewport logic in _curses_prompt_choice (issue #5755).

The "More providers" submenu has 13 entries (11 extended + custom + cancel).
Before the fix, _curses_prompt_choice rendered items starting unconditionally
from index 0 with no scroll offset.  On terminals shorter than ~16 rows, items
near the bottom were never drawn.  When the cursor wrapped from 0 to the last
item (Cancel) via UP-arrow, the highlight rendered off-screen, leaving the menu
looking like only "Cancel" existed.

The fix adds a scroll_offset that tracks the cursor so the highlighted item
is always within the visible window.  These tests exercise that logic in
isolation without requiring a real TTY.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Pure scroll-offset logic extracted from _curses_menu for unit testing
# ---------------------------------------------------------------------------

def _compute_scroll_offset(cursor: int, scroll_offset: int, visible: int, n_choices: int) -> int:
    """Mirror of the scroll adjustment block inside _curses_menu."""
    if cursor < scroll_offset:
        scroll_offset = cursor
    elif cursor >= scroll_offset + visible:
        scroll_offset = cursor - visible + 1
    scroll_offset = max(0, min(scroll_offset, max(0, n_choices - visible)))
    return scroll_offset


def _visible_indices(cursor: int, scroll_offset: int, visible: int, n_choices: int):
    """Return the list indices that would be rendered for the given state."""
    scroll_offset = _compute_scroll_offset(cursor, scroll_offset, visible, n_choices)
    return list(range(scroll_offset, min(scroll_offset + visible, n_choices)))


# ---------------------------------------------------------------------------
# Tests: scroll offset calculation
# ---------------------------------------------------------------------------

class TestScrollOffsetLogic:
    N = 13  # typical extended-providers list length

    def test_cursor_at_zero_no_scroll(self):
        """Start position: offset stays 0, first items visible."""
        assert _compute_scroll_offset(0, 0, 8, self.N) == 0

    def test_cursor_within_window_unchanged(self):
        """Cursor inside the current window: offset unchanged."""
        assert _compute_scroll_offset(5, 0, 8, self.N) == 0

    def test_cursor_at_last_item_scrolls_down(self):
        """Cursor on Cancel (index 12) with 8-row window: offset = 12 - 8 + 1 = 5."""
        offset = _compute_scroll_offset(12, 0, 8, self.N)
        assert offset == 5
        assert 12 in _visible_indices(12, 0, 8, self.N)

    def test_cursor_wraps_to_cancel_via_up(self):
        """UP from index 0 wraps to last item; last item must be visible."""
        wrapped_cursor = (0 - 1) % self.N  # == 12
        indices = _visible_indices(wrapped_cursor, 0, 8, self.N)
        assert wrapped_cursor in indices

    def test_cursor_above_window_scrolls_up(self):
        """Cursor above current window: offset tracks cursor."""
        # window currently shows [5..12], cursor moves to 3
        offset = _compute_scroll_offset(3, 5, 8, self.N)
        assert offset == 3
        assert 3 in _visible_indices(3, 5, 8, self.N)

    def test_visible_window_never_exceeds_list(self):
        """Offset is clamped so the window never starts past the list end."""
        offset = _compute_scroll_offset(12, 0, 20, self.N)  # window larger than list
        assert offset == 0

    def test_single_item_list(self):
        """Edge case: one choice, cursor 0."""
        assert _compute_scroll_offset(0, 0, 8, 1) == 0

    def test_list_fits_in_window_no_scroll_needed(self):
        """If all choices fit in the visible window, offset is always 0."""
        for cursor in range(self.N):
            offset = _compute_scroll_offset(cursor, 0, 20, self.N)
            assert offset == 0, f"cursor={cursor} should not scroll when window > list"

    def test_cursor_always_in_visible_range(self):
        """Invariant: cursor is always within the rendered window after adjustment."""
        visible = 5
        for cursor in range(self.N):
            indices = _visible_indices(cursor, 0, visible, self.N)
            assert cursor in indices, f"cursor={cursor} not in visible={indices}"

    def test_full_navigation_down_cursor_always_visible(self):
        """Simulate pressing DOWN through all items; cursor always in view."""
        visible = 6
        scroll_offset = 0
        cursor = 0
        for _ in range(self.N + 2):  # wrap around twice
            scroll_offset = _compute_scroll_offset(cursor, scroll_offset, visible, self.N)
            rendered = list(range(scroll_offset, min(scroll_offset + visible, self.N)))
            assert cursor in rendered, f"cursor={cursor} not in rendered={rendered}"
            cursor = (cursor + 1) % self.N

    def test_full_navigation_up_cursor_always_visible(self):
        """Simulate pressing UP through all items; cursor always in view."""
        visible = 6
        scroll_offset = 0
        cursor = 0
        for _ in range(self.N + 2):
            scroll_offset = _compute_scroll_offset(cursor, scroll_offset, visible, self.N)
            rendered = list(range(scroll_offset, min(scroll_offset + visible, self.N)))
            assert cursor in rendered, f"cursor={cursor} not in rendered={rendered}"
            cursor = (cursor - 1) % self.N

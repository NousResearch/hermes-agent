"""Tests for the prompt_toolkit /model picker scroll viewport.

Regression for: when a provider exposes many models (e.g. Ollama Cloud's
36+), the picker rendered every choice into a Window with no max height,
clipping the bottom border and any items past the terminal's last row.
The viewport helper now caps visible items and slides the offset to keep
the cursor on screen.
"""
from cli import HermesCLI


_compute = HermesCLI._compute_model_picker_viewport


class TestPickerViewport:
    def test_short_list_no_scroll(self):
        offset, visible = _compute(selected=0, scroll_offset=0, n=5, term_rows=30)
        assert offset == 0
        assert visible == 5

    def test_long_list_caps_visible_to_chrome_budget(self):
        # 36 models, 30 terminal rows, chrome_reserve=14 → max_visible=16.
        # Position counter lives in the bottom border, no row reserved.
        offset, visible = _compute(selected=0, scroll_offset=0, n=36, term_rows=30)
        assert visible == 16
        assert offset == 0

    def test_cursor_past_window_scrolls_down(self):
        offset, visible = _compute(selected=20, scroll_offset=0, n=36, term_rows=30)
        assert visible == 16
        assert 20 in range(offset, offset + visible)

    def test_cursor_above_window_scrolls_up(self):
        offset, visible = _compute(selected=3, scroll_offset=15, n=36, term_rows=30)
        assert offset == 3
        assert 3 in range(offset, offset + visible)

    def test_offset_clamped_to_bottom(self):
        # Selected on last item — offset must keep visible window full, not
        # walk past the end of the list.
        offset, visible = _compute(selected=35, scroll_offset=0, n=36, term_rows=30)
        assert offset + visible == 36
        assert 35 in range(offset, offset + visible)

    def test_tiny_terminal_uses_minimum_visible(self):
        # term_rows below chrome_reserve falls back to the floor of 3 rows.
        _, visible = _compute(selected=0, scroll_offset=0, n=20, term_rows=10)
        assert visible == 3  # max(3, 10 - 14) == 3

    def test_offset_recovers_after_stage_switch(self):
        # When the user backs out of the model stage and re-enters with
        # selected=0, a stale offset from the previous stage must collapse.
        offset, visible = _compute(selected=0, scroll_offset=25, n=36, term_rows=30)
        assert offset == 0
        assert 0 in range(offset, offset + visible)

    def test_full_navigation_keeps_cursor_visible(self):
        offset = 0
        for cursor in list(range(36)) + list(range(35, -1, -1)):
            offset, visible = _compute(cursor, offset, n=36, term_rows=30)
            assert cursor in range(offset, offset + visible), (
                f"cursor={cursor} out of view: offset={offset} visible={visible}"
            )

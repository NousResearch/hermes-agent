"""Tests for input-box scrolling of long pasted text.

Regression for the Wispr Flow / long-paste issue: the input ``TextArea`` is
capped at a few visible rows and Up/Down browse shell history for a single
wrapped line, leaving no way to scroll to the top of a long dictated blob.

The scroll logic lives in inline closures inside ``HermesCLI.run()`` and
relies on prompt_toolkit's ``Window.vertical_scroll_2`` (sub-line scroll for
wrapped lines) and ``Window.vertical_scroll`` (line scroll for multi-line).
These tests verify the pure helper ``_estimate_tui_input_height`` that feeds
the scroll math, plus structural invariants of the keybinding registration.
"""

import cli as cli_mod


class TestEstimateTuiInputHeight:
    """The height estimate determines visible_height for scroll calculations."""

    def test_single_short_line_is_one_row(self):
        assert cli_mod._estimate_tui_input_height(["hello"], "⚔ ", 80) == 1

    def test_long_line_wraps_to_multiple_rows(self):
        # 200 chars at 10 columns = 20 visual rows, capped at 8
        assert cli_mod._estimate_tui_input_height(["x" * 200], "", 10) == 8

    def test_prompt_only_on_first_wrapped_row(self):
        # Prompt "⚔ " (2 cells) + "abcdef" (6 cells) = 8 cells at 3 columns
        # = 3 visual rows (2+1, 3, 3), but first row has prompt
        assert cli_mod._estimate_tui_input_height(["abcdef"], "⚔ ", 3) == 3

    def test_wide_characters_use_cell_width(self):
        # 10 CJK chars (20 cells) + prompt (2 cells) = 22 cells at 14 columns
        # = 2 visual rows
        assert cli_mod._estimate_tui_input_height(["你" * 10], "❯ ", 14) == 2

    def test_zero_columns_treated_as_one(self):
        assert cli_mod._estimate_tui_input_height(["abcd"], "", 0) == 4

    def test_multiline_text_capped_at_max_height(self):
        lines = [f"line{i}" for i in range(20)]
        assert cli_mod._estimate_tui_input_height(lines, "", 80) == 8

    def test_multiline_text_under_cap_shows_all(self):
        lines = [f"line{i}" for i in range(3)]
        assert cli_mod._estimate_tui_input_height(lines, "", 80) == 3


class TestComputeInputPageCursor:
    """The _compute_input_page_cursor helper is still used for fallback cursor math.

    Verify it produces valid in-bounds positions for various inputs.
    """

    def test_empty_text_is_noop(self):
        assert cli_mod._compute_input_page_cursor("", 0, -1, 8, 80) == 0
        assert cli_mod._compute_input_page_cursor("", 0, +1, 8, 80) == 0

    def test_pageup_single_line_moves_toward_start(self):
        text = "x" * 200
        pos = cli_mod._compute_input_page_cursor(text, 200, -1, 8, 10)
        assert 0 <= pos < 200

    def test_pagedown_single_line_moves_toward_end(self):
        text = "x" * 200
        pos = cli_mod._compute_input_page_cursor(text, 0, +1, 8, 10)
        assert 0 < pos <= 200

    def test_repeated_pageup_reaches_start(self):
        text = "x" * 200
        pos = 200
        for _ in range(10):
            pos = cli_mod._compute_input_page_cursor(text, pos, -1, 8, 10)
        assert pos == 0

    def test_cursor_always_in_bounds(self):
        text = "hello world\nthis is a test\nmore text"
        for direction in (-1, +1):
            for start in range(len(text) + 1):
                pos = cli_mod._compute_input_page_cursor(text, start, direction, 8, 10)
                assert 0 <= pos <= len(text)

    def test_multiline_pageup_moves_up(self):
        text = "\n".join(f"line{i}" for i in range(20))
        pos = cli_mod._compute_input_page_cursor(text, len(text), -1, 8, 80)
        assert pos < len(text)

    def test_multiline_pagedown_moves_down(self):
        text = "\n".join(f"line{i}" for i in range(20))
        pos = cli_mod._compute_input_page_cursor(text, 0, +1, 8, 80)
        assert pos > 0

    def test_defensive_against_bad_inputs(self):
        assert cli_mod._compute_input_page_cursor("hello", 999, -1, 0, 0) >= 0
        assert cli_mod._compute_input_page_cursor("hello", -5, +1, 0, 0) <= 5

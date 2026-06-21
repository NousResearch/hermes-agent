"""Tests for PageUp/PageDown input-box scrolling of long pasted text.

Regression for the Wispr Flow / long-paste issue: the input ``TextArea`` is
capped at a few visible rows and Up/Down browse shell history for a single
wrapped line, leaving no way to scroll to the top of a long dictated blob.
``_compute_input_page_cursor`` is the pure cursor-math behind the PageUp /
PageDown keybindings.
"""

import cli as cli_mod


class TestComputeInputPageCursor:
    def test_empty_text_is_noop(self):
        assert cli_mod._compute_input_page_cursor("", 0, -1, 8, 80) == 0
        assert cli_mod._compute_input_page_cursor("", 0, +1, 8, 80) == 0

    def test_pageup_on_single_wrapped_line_moves_toward_start(self):
        # A single 200-char line at 10 columns wraps to 20 visual rows. With an
        # 8-row viewport the cursor sits at the end (200). PageUp should move
        # the cursor left by one viewport (8 * 10 = 80 chars).
        text = "x" * 200
        pos = cli_mod._compute_input_page_cursor(text, 200, -1, 8, 10)
        assert pos == 120  # 200 - 80

    def test_repeated_pageup_reaches_start_of_single_line(self):
        text = "x" * 200
        pos = 200
        for _ in range(10):
            pos = cli_mod._compute_input_page_cursor(text, pos, -1, 8, 10)
        assert pos == 0

    def test_pagedown_on_single_wrapped_line_moves_toward_end(self):
        text = "x" * 200
        pos = cli_mod._compute_input_page_cursor(text, 0, +1, 8, 10)
        assert pos == 80

    def test_pagedown_clamps_to_text_length(self):
        text = "x" * 50
        pos = cli_mod._compute_input_page_cursor(text, 0, +1, 8, 10)
        assert pos == 50  # would be 80, clamped to len(text)

    def test_pageup_multiline_moves_up_by_viewport_rows(self):
        # 20 logical lines, cursor at end of line 19, 8-row viewport.
        text = "\n".join(f"line{i}" for i in range(20))
        cursor = len(text)  # very end
        pos = cli_mod._compute_input_page_cursor(text, cursor, -1, 8, 80)
        # Should land on line 11 (row 19 - 8), preserving the cursor's column
        # (6 = len("line19")).
        lines = text.split("\n")
        expected = sum(len(lines[r]) + 1 for r in range(11)) + len(lines[11])
        assert pos == expected

    def test_pageup_multiline_clamps_to_first_row(self):
        text = "a\nb\nc\nd"
        pos = cli_mod._compute_input_page_cursor(text, 3, -1, 8, 80)
        # Row 1 -> row 0, column preserved (1 = end of "a").
        assert pos == 1

    def test_pagedown_multiline_moves_down_by_viewport_rows(self):
        text = "\n".join(f"line{i}" for i in range(20))
        # Cursor at start of line 0.
        pos = cli_mod._compute_input_page_cursor(text, 0, +1, 8, 80)
        lines = text.split("\n")
        expected = sum(len(lines[r]) + 1 for r in range(8))
        assert pos == expected

    def test_pagedown_multiline_clamps_to_last_row(self):
        text = "a\nb\nc\nd"
        # Cursor at end (row 3), page down can't go further.
        pos = cli_mod._compute_input_page_cursor(text, len(text), +1, 8, 80)
        assert pos == len(text)

    def test_pageup_preserves_column_within_row_length(self):
        # Cursor at column 5 of row 10; page up to row 2 (which is long enough).
        text = "\n".join("0123456789" for _ in range(20))
        # Place cursor at row 10, col 5.
        cursor = sum(len("0123456789") + 1 for _ in range(10)) + 5
        pos = cli_mod._compute_input_page_cursor(text, cursor, -1, 8, 80)
        expected = sum(len("0123456789") + 1 for _ in range(2)) + 5
        assert pos == expected

    def test_pageup_column_clamped_to_shorter_target_row(self):
        # Row 0 is long; row 2 is short. Cursor at col 9 of row 10; page up to
        # row 2 whose length is 1, so column clamps to 1.
        rows = ["0123456789"] * 11
        rows[2] = "Z"
        text = "\n".join(rows)
        cursor = sum(len(rows[r]) + 1 for r in range(10)) + 9
        pos = cli_mod._compute_input_page_cursor(text, cursor, -1, 8, 80)
        expected = sum(len(rows[r]) + 1 for r in range(2)) + 1
        assert pos == expected

    def test_invalid_inputs_are_defensive(self):
        # Zero/negative height/columns should be coerced, not crash.
        assert cli_mod._compute_input_page_cursor("hello world", 11, -1, 0, 0) >= 0
        assert cli_mod._compute_input_page_cursor("hello world", 0, +1, 0, 0) >= 0

    def test_cursor_out_of_range_is_clamped(self):
        text = "abc"
        assert cli_mod._compute_input_page_cursor(text, 999, -1, 8, 80) == 0
        assert cli_mod._compute_input_page_cursor(text, -5, +1, 8, 80) <= len(text)

"""Regression tests for TUI resize ghost-row cleanup."""

from prompt_toolkit.layout.screen import Char, Screen

from cli import (
    _estimate_reflowed_height_for_narrower_width,
    _extra_rows_after_terminal_shrink,
)


def _make_screen(rows: list[str]) -> Screen:
    screen = Screen()
    screen.height = len(rows)
    screen.width = max((len(r) for r in rows), default=0)
    for y, row_text in enumerate(rows):
        for x, ch in enumerate(row_text):
            screen.data_buffer[y][x] = Char(ch, "")
    return screen


class TestResizeReflowEstimation:
    def test_counts_reflow_per_rendered_row_instead_of_global_multiplier(self):
        screen = _make_screen([
            "=" * 10,
            "> short",
            "x" * 10,
        ])

        # Narrowing from width 10 to width 5 reflows rows independently:
        # 10 -> 2 rows, 7 -> 2 rows, 10 -> 2 rows, total = 6.
        assert _estimate_reflowed_height_for_narrower_width(screen, 5) == 6
        assert _extra_rows_after_terminal_shrink(screen, 5) == 3

    def test_blank_rows_still_count_as_one_terminal_row(self):
        screen = _make_screen([
            "",
            "abcde",
            "",
        ])

        assert _estimate_reflowed_height_for_narrower_width(screen, 2) == 5
        assert _extra_rows_after_terminal_shrink(screen, 2) == 2

    def test_explicit_space_cells_contribute_to_row_width(self):
        screen = _make_screen([
            "          ",
            "abc",
        ])

        # The first row is ten explicit spaces; it still occupies width and will
        # wrap to two rows when the terminal narrows to five columns.
        assert _estimate_reflowed_height_for_narrower_width(screen, 5) == 3
        assert _extra_rows_after_terminal_shrink(screen, 5) == 1

    def test_invalid_inputs_return_zero_extra_rows(self):
        assert _estimate_reflowed_height_for_narrower_width(None, 10) == 0
        assert _extra_rows_after_terminal_shrink(None, 10) == 0

        screen = _make_screen(["abc"])
        assert _estimate_reflowed_height_for_narrower_width(screen, 0) == 0
        assert _extra_rows_after_terminal_shrink(screen, 0) == 0

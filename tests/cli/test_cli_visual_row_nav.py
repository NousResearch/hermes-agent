"""Tests for classic-CLI visual-row Up/Down navigation helpers."""

from __future__ import annotations

import cli as cli_mod


class _FakeBuffer:
    def __init__(
        self,
        text: str = "history entry",
        cursor: int = 0,
        *,
        working_index: int = 0,
        working_line_count: int = 2,
        draft_text: str = "draft text",
    ):
        self.text = text
        self.cursor_position = cursor
        self.working_index = working_index
        self._working_lines = [""] * max(1, working_line_count)
        self._draft_text = draft_text
        self.forward_calls = 0

    def history_forward(self, count: int = 1) -> None:
        self.forward_calls += count
        # Mirror prompt_toolkit: jump to end of the first line of the new entry.
        self.working_index = min(self.working_index + count, len(self._working_lines) - 1)
        if self.working_index >= len(self._working_lines) - 1:
            self.text = self._draft_text
        first_line = self.text.split("\n", 1)[0]
        self.cursor_position = len(first_line)


def test_build_rows_soft_wrap_uses_exclusive_ends_without_overlap():
    # prompt takes 2 cells → first visual row holds 3 content cells at width 5.
    rows = cli_mod._build_input_visual_rows("abcdefghij", window_width=5, prompt_width=2)
    assert rows == [(0, 3, 0), (3, 8, 0), (8, 10, 0)]

    # Wrap boundary belongs to the next row, not both.
    assert cli_mod._find_input_visual_row(rows, 3, 10) == 1
    assert cli_mod._find_input_visual_row(rows, 2, 10) == 0


def test_build_rows_hard_newlines_and_empty_line():
    text = "hi\n\nbye"
    rows = cli_mod._build_input_visual_rows(text, window_width=80, prompt_width=2)
    assert rows == [(0, 2, 0), (3, 3, 1), (4, 7, 2)]

    # Cursor on the newline after "hi" maps to end of that logical line.
    assert cli_mod._find_input_visual_row(rows, 2, len(text)) == 0
    # Empty middle line.
    assert cli_mod._find_input_visual_row(rows, 3, len(text)) == 1
    # EOF sits on the last row.
    assert cli_mod._find_input_visual_row(rows, len(text), len(text)) == 2


def test_cjk_wrapping_uses_display_cell_width():
    # Ten CJK chars = 20 cells. With prompt width 2 and width 14:
    # first row content cells = 12 → 6 CJK chars; remainder 4 chars on row 2.
    text = "你" * 10
    rows = cli_mod._build_input_visual_rows(text, window_width=14, prompt_width=2)
    assert rows == [(0, 6, 0), (6, 10, 0)]

    # len()-based wrapping would have put 12 code points on row 0; display
    # width must keep the wrap at 6 characters.
    assert cli_mod._find_input_visual_row(rows, 6, len(text)) == 1


def test_visual_move_soft_wrap_preserves_screen_column():
    text = "abcdefghij"
    # Cursor on 'e' (index 4) in the second visual row.
    # Screen column on row 1 is content col 1.
    down_from_b = cli_mod._visual_row_move_cursor(text, 1, +1, 5, 2)
    # 'b' is content col 1 on prompt row → screen col 3 → row1 content col 3 → 'g'
    assert down_from_b == 6

    up_from_g = cli_mod._visual_row_move_cursor(text, 6, -1, 5, 2)
    assert up_from_g == 1


def test_visual_move_hard_newline_and_history_boundaries():
    text = "hello\nworld"
    # From col 2 of "hello" down to col 2 of "world" (screen-aware via prompt).
    # content col 2 on row 0 → screen 4 → dest content col 4 → index 10 ('d' end-ish)
    moved = cli_mod._visual_row_move_cursor(text, 2, +1, 80, 2)
    assert moved == 10

    assert cli_mod._visual_row_move_cursor(text, 1, -1, 80, 2) is None
    assert cli_mod._visual_row_move_cursor(text, 8, +1, 80, 2) is None


def test_visual_move_cjk_targets_matching_display_column():
    text = "你" * 10
    # Index 7 is the 8th CJK char → second visual row, content display col 2.
    up = cli_mod._visual_row_move_cursor(text, 7, -1, 14, 2)
    # screen col = 2; on prompt row content col = 0 → first CJK char.
    assert up == 0

    down = cli_mod._visual_row_move_cursor(text, 1, +1, 14, 2)
    # content col 2 (你你), screen col 4 → second-row content col 4 → index 8.
    assert down == 8


def test_history_forward_restores_saved_draft_cursor():
    # Mid-line draft cursor must survive Up→history→Down→draft.
    buf = _FakeBuffer(text="history entry", cursor=0, working_index=0, working_line_count=2)
    cli_mod._history_forward_restore_draft_cursor(buf, count=1, draft_cursor=6)
    assert buf.forward_calls == 1
    assert buf.text == "draft text"
    assert buf.cursor_position == 6


def test_history_forward_restores_cursor_on_multiline_draft():
    # prompt_toolkit alone would park at end-of-first-line (0 for a leading blank).
    buf = _FakeBuffer(
        text="history entry",
        cursor=0,
        working_index=0,
        working_line_count=2,
        draft_text="\n\ntrailing",
    )
    cli_mod._history_forward_restore_draft_cursor(buf, count=1, draft_cursor=2)
    assert buf.cursor_position == 2


def test_history_forward_leaves_history_entry_cursor_alone():
    buf = _FakeBuffer(text="old\nentry", cursor=0, working_index=0, working_line_count=3)
    # Still browsing history (not yet on draft) — do not override pt's placement.
    cli_mod._history_forward_restore_draft_cursor(buf, count=1, draft_cursor=9)
    assert buf.working_index == 1
    assert buf.cursor_position == len("old")  # end of first line after forward


def test_remember_draft_cursor_ignores_climb_to_history():
    slot: dict[str, int | None] = {"pos": 12}
    # Climbing Up through rows suppresses tracking so position 0 cannot clobber.
    cli_mod._remember_draft_history_cursor(slot, 0, on_draft=True, suppress=True)
    assert slot["pos"] == 12
    cli_mod._remember_draft_history_cursor(slot, 0, on_draft=True, suppress=False)
    assert slot["pos"] == 0
    cli_mod._remember_draft_history_cursor(slot, 5, on_draft=False, suppress=False)
    assert slot["pos"] == 0

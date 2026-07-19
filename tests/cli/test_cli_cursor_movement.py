"""Tests for cursor-movement keybindings in cli.py.

Covers:
  - Pure helper functions (visual-row math, wide-char handling, round-trip)
  - Buffer-level integration with real prompt_toolkit Buffer/Document objects
  - Up/Down always navigate history (never move cursor)

For the PR targeting upstream: only Ctrl+Up/Down cursor movement is proposed.
Left/Right overrides are intentionally excluded from this test suite.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from prompt_toolkit.utils import get_cwidth


# ═══════════════════════════════════════════════════════════════════════════
# Import the REAL helpers from cli.py and wrap them to supply get_cwidth.
#
# cli.py's helpers take get_cwidth as their final argument (it is imported
# locally inside the handlers, not at module level). These thin wrappers pass
# prompt_toolkit's get_cwidth so the tests exercise the actual shipping
# functions in cli.py rather than a private copy that could silently drift.
# ═══════════════════════════════════════════════════════════════════════════

from cli import _visual_row_col as _cli_visual_row_col
from cli import _max_visual_row as _cli_max_visual_row
from cli import _pos_from_visual as _cli_pos_from_visual


def _visual_row_col(text, pos, cols, prompt_width):
    return _cli_visual_row_col(text, pos, cols, prompt_width, get_cwidth)


def _max_visual_row(text, cols, prompt_width):
    return _cli_max_visual_row(text, cols, prompt_width, get_cwidth)


def _pos_from_visual(text, target_row, target_col, cols, prompt_width):
    return _cli_pos_from_visual(text, target_row, target_col, cols, prompt_width, get_cwidth)


# ═══════════════════════════════════════════════════════════════════════════
# _visual_row_col tests
# ═══════════════════════════════════════════════════════════════════════════

class TestVisualRowCol:
    """Tests for _visual_row_col — maps char position to visual (row, col)."""

    def test_empty_text(self):
        assert _visual_row_col("", 0, 80, 2) == (0, 2)  # cursor sits just after the prompt

    def test_first_visual_row_no_prompt(self):
        assert _visual_row_col("hello world", 5, 80, 0) == (0, 5)

    def test_first_visual_row_with_prompt(self):
        assert _visual_row_col("abcdefghij", 5, 80, 2) == (0, 7)  # 2 (prompt) + 5

    def test_position_zero(self):
        assert _visual_row_col("hello", 0, 80, 2) == (0, 2)  # start of text = after prompt

    def test_wrap_to_second_row_no_prompt(self):
        # 10-col terminal: "0123456789AB" — after 10 chars, cursor wraps to row 1
        assert _visual_row_col("0123456789AB", 0, 10, 0) == (0, 0)
        assert _visual_row_col("0123456789AB", 9, 10, 0) == (0, 9)
        assert _visual_row_col("0123456789AB", 10, 10, 0) == (1, 0)
        assert _visual_row_col("0123456789AB", 11, 10, 0) == (1, 1)

    def test_wrap_with_prompt(self):
        # 10-col term, 2-char prompt: row 0 holds 8 chars (screen cols 2-9),
        # then wraps. Columns are screen columns (prompt offset included).
        assert _visual_row_col("01234567ABCDEFGH", 0, 10, 2) == (0, 2)
        assert _visual_row_col("01234567ABCDEFGH", 7, 10, 2) == (0, 9)
        assert _visual_row_col("01234567ABCDEFGH", 8, 10, 2) == (1, 0)
        assert _visual_row_col("01234567ABCDEFGH", 9, 10, 2) == (1, 1)

    def test_wide_characters_cjk(self):
        assert _visual_row_col("a你b", 0, 80, 0) == (0, 0)
        assert _visual_row_col("a你b", 1, 80, 0) == (0, 1)
        assert _visual_row_col("a你b", 2, 80, 0) == (0, 3)

    def test_wide_char_wrapping(self):
        text = "abc你d"
        cols, pw = 5, 0
        assert _visual_row_col(text, 0, cols, pw) == (0, 0)
        assert _visual_row_col(text, 1, cols, pw) == (0, 1)
        assert _visual_row_col(text, 2, cols, pw) == (0, 2)
        assert _visual_row_col(text, 3, cols, pw) == (0, 3)
        assert _visual_row_col(text, 4, cols, pw) == (1, 0)

    def test_narrow_terminal_prompt_larger_than_cols(self):
        # Degenerate: prompt wider than terminal. pos 0 stays on row 0;
        # the first character then wraps. (Pathological — a sane prompt is
        # always far narrower than the terminal.)
        text = "abcdef"
        assert _visual_row_col(text, 0, 10, 12) == (0, 12)
        assert _visual_row_col(text, 1, 10, 12) == (1, 1)

    def test_position_at_end_of_text(self):
        text = "hello"
        assert _visual_row_col(text, 5, 80, 2) == (0, 7)  # 2 (prompt) + 5

    def test_hard_newline_creates_new_row(self):
        """\\n forces a new visual row."""
        assert _visual_row_col("hi\nthere", 0, 80, 0) == (0, 0)
        assert _visual_row_col("hi\nthere", 2, 80, 0) == (0, 2)
        assert _visual_row_col("hi\nthere", 3, 80, 0) == (1, 0)
        assert _visual_row_col("hi\nthere", 4, 80, 0) == (1, 1)

    def test_multiple_newlines(self):
        assert _visual_row_col("a\n\nb", 1, 80, 0) == (0, 1)
        assert _visual_row_col("a\n\nb", 2, 80, 0) == (1, 0)
        assert _visual_row_col("a\n\nb", 3, 80, 0) == (2, 0)
        assert _visual_row_col("a\n\nb", 4, 80, 0) == (2, 1)


# ═══════════════════════════════════════════════════════════════════════════
# _max_visual_row tests
# ═══════════════════════════════════════════════════════════════════════════

class TestMaxVisualRow:
    """Tests for _max_visual_row — returns last visual row index."""

    def test_empty_text(self):
        assert _max_visual_row("", 80, 2) == -1

    def test_single_row(self):
        assert _max_visual_row("hello", 80, 2) == 0

    def test_two_rows_no_prompt(self):
        assert _max_visual_row("0123456789", 5, 0) == 1

    def test_two_rows_with_prompt(self):
        assert _max_visual_row("0123456789012345678", 10, 2) == 2

    def test_wide_chars_affect_row_count(self):
        assert _max_visual_row("a你c", 5, 0) == 0
        assert _max_visual_row("a你cd", 5, 0) == 0
        assert _max_visual_row("a你cde", 5, 0) == 1

    def test_newline_rows(self):
        assert _max_visual_row("hello\nworld", 80, 0) == 1
        assert _max_visual_row("a\n\nb", 80, 0) == 2


# ═══════════════════════════════════════════════════════════════════════════
# _pos_from_visual tests
# ═══════════════════════════════════════════════════════════════════════════

class TestPosFromVisual:
    """Tests for _pos_from_visual — maps visual (row, col) to char position."""

    def test_first_row_simple(self):
        text = "hello world"
        assert _pos_from_visual(text, 0, 0, 80, 0) == 0
        assert _pos_from_visual(text, 0, 5, 80, 0) == 5
        assert _pos_from_visual(text, 0, 10, 80, 0) == 10

    def test_second_row(self):
        text = "01234ABCDE"
        assert _pos_from_visual(text, 0, 4, 5, 0) == 4
        assert _pos_from_visual(text, 1, 0, 5, 0) == 5
        assert _pos_from_visual(text, 1, 4, 5, 0) == 9

    def test_clamp_to_row_end(self):
        text = "abcDEF"
        assert _pos_from_visual(text, 0, 10, 3, 0) == 3

    def test_clamp_past_end_of_text(self):
        text = "hi"
        assert _pos_from_visual(text, 5, 5, 80, 0) == 2

    def test_with_prompt(self):
        # cols=5, prompt=1 -> row 0 chars A,B,C,D at screen cols 1-4; E wraps.
        # Screen col 0 on row 0 is the prompt zone -> clamps to first char (A).
        text = "ABCDEFGHI"
        assert _pos_from_visual(text, 0, 0, 5, 1) == 0   # prompt-zone clamp -> 'A'
        assert _pos_from_visual(text, 0, 3, 5, 1) == 2   # screen col 3 = 'C'
        assert _pos_from_visual(text, 1, 0, 5, 1) == 4   # 'E'
        assert _pos_from_visual(text, 1, 4, 5, 1) == 8   # 'I'

    def test_wide_char_positioning(self):
        text = "abc你def"
        assert _pos_from_visual(text, 0, 0, 80, 0) == 0
        assert _pos_from_visual(text, 0, 2, 80, 0) == 2
        assert _pos_from_visual(text, 0, 3, 80, 0) == 3
        assert _pos_from_visual(text, 0, 4, 80, 0) == 3
        assert _pos_from_visual(text, 0, 5, 80, 0) == 4

    def test_newline_mapping(self):
        """Targeting a row with a newline returns the newline position."""
        text = "hello\nworld"
        assert _pos_from_visual(text, 0, 0, 80, 0) == 0
        assert _pos_from_visual(text, 0, 5, 80, 0) == 5
        assert _pos_from_visual(text, 1, 0, 80, 0) == 6


# ═══════════════════════════════════════════════════════════════════════════
# Integration: round-trip tests
# ═══════════════════════════════════════════════════════════════════════════

class TestRoundTrip:
    """Round-trip: visual_row_col → pos_from_visual should be identity."""

    @pytest.mark.parametrize("text,cols,pw", [
        ("hello", 80, 2),
        ("abcdefghijklmnopqrstuvwxyz", 5, 0),
        ("abcdefghijklmnopqrstuvwxyz", 10, 3),
        ("hello\nworld", 80, 2),
        ("a\n\nb", 80, 0),
        ("a你b好c", 80, 2),
        ("a你b好cdefghij", 5, 0),
        ("😀😀😀😀😀", 4, 0),
        ("", 80, 2),
        ("x", 80, 2),
    ])
    def test_round_trip(self, text, cols, pw):
        """For every position, pos_from_visual(visual_row_col(pos)) ≈ pos."""
        for pos in range(len(text) + 1):
            vr, vc = _visual_row_col(text, pos, cols, pw)
            recovered = _pos_from_visual(text, vr, vc, cols, pw)
            assert recovered <= pos, (
                f"pos={pos} recovered={recovered} (should not exceed original)\n"
                f"text={text!r} cols={cols} pw={pw} vr={vr} vc={vc}"
            )
            vr2, vc2 = _visual_row_col(text, recovered, cols, pw)
            assert vr2 == vr, (
                f"Recovered row {vr2} != original row {vr} for pos={pos}\n"
                f"text={text!r} cols={cols} pw={pw}"
            )


# ═══════════════════════════════════════════════════════════════════════════
# _cursor_prompt_width tests (import from cli.py)
# ═══════════════════════════════════════════════════════════════════════════

class TestCursorPromptWidth:
    """Tests for _cursor_prompt_width (requires mock HermesCLI instance)."""

    def test_returns_minimum_2(self):
        mock_cli = MagicMock()
        mock_cli._get_tui_prompt_fragments.side_effect = AttributeError("no method")

        from cli import _cursor_prompt_width
        mock_event = MagicMock()
        result = _cursor_prompt_width(mock_cli, mock_event)
        assert result == 2

    def test_returns_prompt_width(self):
        mock_cli = MagicMock()
        mock_cli._get_tui_prompt_fragments.return_value = [
            ("class:prompt", "> "),
        ]

        from cli import _cursor_prompt_width
        mock_event = MagicMock()
        result = _cursor_prompt_width(mock_cli, mock_event)
        assert result == 2

    def test_cjk_prompt(self):
        mock_cli = MagicMock()
        mock_cli._get_tui_prompt_fragments.return_value = [
            ("class:prompt", "你好"),
        ]

        from cli import _cursor_prompt_width
        mock_event = MagicMock()
        result = _cursor_prompt_width(mock_cli, mock_event)
        assert result == 4


# ═══════════════════════════════════════════════════════════════════════════
# Buffer-level cursor movement integration tests
# ═══════════════════════════════════════════════════════════════════════════

def _simulate_move_up(buf, cols, pw):
    """Simulate what Ctrl+Up does: move cursor up one visual row."""
    vr, vc = _visual_row_col(buf.text, buf.cursor_position, cols, pw)
    if vr > 0:
        buf.cursor_position = _pos_from_visual(buf.text, vr - 1, vc, cols, pw)
    return buf.cursor_position


def _simulate_move_down(buf, cols, pw):
    """Simulate what Ctrl+Down does: move cursor down one visual row."""
    vr, vc = _visual_row_col(buf.text, buf.cursor_position, cols, pw)
    max_row = _max_visual_row(buf.text, cols, pw)
    if vr < max_row:
        buf.cursor_position = _pos_from_visual(buf.text, vr + 1, vc, cols, pw)
    return buf.cursor_position


class TestBufferCursorMovement:
    """Cursor movement using real prompt_toolkit Buffer/Document objects."""

    def test_ctrl_up_wrapped_text(self):
        """Ctrl+Up from second visual row of a wrapped line moves to first row."""
        from prompt_toolkit.buffer import Buffer
        from prompt_toolkit.document import Document

        text = "0123456789AB"
        buf = Buffer(document=Document(text, cursor_position=10))
        cols, pw = 10, 0

        vr, vc = _visual_row_col(buf.text, 10, cols, pw)
        assert vr == 1, f"pos 10 should be on visual row 1, got {vr}"

        new_pos = _simulate_move_up(buf, cols, pw)
        assert new_pos == 0
        assert buf.cursor_position == 0

    def test_ctrl_up_at_first_row_is_noop(self):
        """Ctrl+Up when already at first visual row does nothing."""
        from prompt_toolkit.buffer import Buffer
        from prompt_toolkit.document import Document

        text = "hello"
        buf = Buffer(document=Document(text, cursor_position=3))
        cols, pw = 80, 2

        vr, _ = _visual_row_col(buf.text, buf.cursor_position, cols, pw)
        assert vr == 0
        old_pos = buf.cursor_position
        _simulate_move_up(buf, cols, pw)
        assert buf.cursor_position == old_pos

    def test_ctrl_up_at_top_of_wrapped_text(self):
        """Ctrl+Up at pos 0 of a wrapped text stays at pos 0."""
        from prompt_toolkit.buffer import Buffer
        from prompt_toolkit.document import Document

        text = "0123456789ABCDEFGHIJ"
        buf = Buffer(document=Document(text, cursor_position=0))
        cols, pw = 10, 0

        vr, _ = _visual_row_col(buf.text, buf.cursor_position, cols, pw)
        assert vr == 0
        old_pos = buf.cursor_position
        _simulate_move_up(buf, cols, pw)
        assert buf.cursor_position == old_pos

    def test_ctrl_down_wrapped_text(self):
        """Ctrl+Down from first visual row moves to second row, preserves column."""
        from prompt_toolkit.buffer import Buffer
        from prompt_toolkit.document import Document

        text = "0123456789AB"
        buf = Buffer(document=Document(text, cursor_position=5))
        cols, pw = 10, 0

        vr, vc = _visual_row_col(buf.text, buf.cursor_position, cols, pw)
        assert vr == 0
        assert vc == 5

        max_row = _max_visual_row(text, cols, pw)
        assert max_row == 1

        _simulate_move_down(buf, cols, pw)
        assert buf.cursor_position == len(text)

    def test_ctrl_down_at_last_row_is_noop(self):
        """Ctrl+Down when already at last visual row does nothing."""
        from prompt_toolkit.buffer import Buffer
        from prompt_toolkit.document import Document

        text = "0123456789AB"
        buf = Buffer(document=Document(text, cursor_position=11))
        cols, pw = 10, 0

        vr, _ = _visual_row_col(buf.text, buf.cursor_position, cols, pw)
        max_row = _max_visual_row(text, cols, pw)
        assert vr == max_row
        old_pos = buf.cursor_position
        _simulate_move_down(buf, cols, pw)
        assert buf.cursor_position == old_pos

    def test_preserves_visual_column_when_moving_up(self):
        """Moving up preserves the approximate visual column."""
        from prompt_toolkit.buffer import Buffer
        from prompt_toolkit.document import Document

        text = "0123456789" * 3
        buf = Buffer(document=Document(text, cursor_position=25))
        cols, pw = 10, 0

        vr, vc = _visual_row_col(buf.text, buf.cursor_position, cols, pw)
        assert vr == 2
        assert vc == 5

        _simulate_move_up(buf, cols, pw)
        assert buf.cursor_position == 15

    def test_preserves_visual_column_when_moving_down(self):
        """Moving down preserves the approximate visual column."""
        from prompt_toolkit.buffer import Buffer
        from prompt_toolkit.document import Document

        text = "0123456789" * 3
        buf = Buffer(document=Document(text, cursor_position=5))
        cols, pw = 10, 0

        vr, vc = _visual_row_col(buf.text, buf.cursor_position, cols, pw)
        assert vr == 0
        assert vc == 5

        _simulate_move_down(buf, cols, pw)
        assert buf.cursor_position == 15

    def test_cursor_movement_across_hard_newlines(self):
        """Ctrl+Up/Down works across hard newlines."""
        from prompt_toolkit.buffer import Buffer
        from prompt_toolkit.document import Document

        text = "hello\nworld"
        buf = Buffer(document=Document(text, cursor_position=7))
        cols, pw = 80, 2

        vr, vc = _visual_row_col(buf.text, buf.cursor_position, cols, pw)
        assert vr == 1

        _simulate_move_up(buf, cols, pw)
        assert buf.document.cursor_position_row == 0
        assert buf.document.current_char == "h"

    def test_cursor_movement_empty_buffer(self):
        """Ctrl+Up/Down on empty buffer is a no-op (no crash)."""
        from prompt_toolkit.buffer import Buffer
        from prompt_toolkit.document import Document

        text = ""
        buf = Buffer(document=Document(text, cursor_position=0))
        cols, pw = 80, 2

        vr, _ = _visual_row_col(buf.text, buf.cursor_position, cols, pw)
        max_row = _max_visual_row(text, cols, pw)
        assert vr == 0
        assert max_row == -1
        old_pos = buf.cursor_position
        _simulate_move_up(buf, cols, pw)
        _simulate_move_down(buf, cols, pw)
        assert buf.cursor_position == old_pos


# ═══════════════════════════════════════════════════════════════════════════
# Up/Down history navigation: verify cli.py uses history, not auto_up/down
# ═══════════════════════════════════════════════════════════════════════════

class TestUpDownBindsHistoryNotCursor:
    """Verify the cli.py source calls history_backward/forward, not auto_up/down."""

    def test_up_arrow_calls_history_backward_not_auto_up(self):
        import re
        with open(__import__('cli').__file__, encoding='utf-8') as f:
            source = f.read()

        up_block = re.search(
            r"@kb\.add\('up', filter=_normal_input\).*?"
            r"def history_up\(event\):.*?"
            r"(event\.app\.current_buffer\.\w+\(count=event\.arg\))",
            source, re.DOTALL
        )
        assert up_block is not None, "Could not find up keybinding in cli.py"
        call = up_block.group(1)
        assert "history_backward" in call, (
            f"up arrow should call history_backward, found: {call}"
        )
        assert "auto_up" not in call, (
            f"up arrow should NOT call auto_up, found: {call}"
        )

    def test_down_arrow_calls_history_forward_not_auto_down(self):
        import re
        with open(__import__('cli').__file__, encoding='utf-8') as f:
            source = f.read()

        down_block = re.search(
            r"@kb\.add\('down', filter=_normal_input\).*?"
            r"def history_down\(event\):.*?"
            r"(event\.app\.current_buffer\.\w+\(count=event\.arg\))",
            source, re.DOTALL
        )
        assert down_block is not None, "Could not find down keybinding in cli.py"
        call = down_block.group(1)
        assert "history_forward" in call, (
            f"down arrow should call history_forward, found: {call}"
        )
        assert "auto_down" not in call, (
            f"down arrow should NOT call auto_down, found: {call}"
        )

    def test_cursor_movement_uses_separate_keybindings(self):
        import re
        with open(__import__('cli').__file__, encoding='utf-8') as f:
            source = f.read()

        for key in ('c-up', 's-up', 'c-down', 's-down'):
            pattern = rf"@kb\.add\('{re.escape(key)}', filter=_normal_input\)"
            assert re.search(pattern, source), (
                f"Missing keybinding: {key}"
            )

    def test_cursor_movement_handlers_exist(self):
        import re
        with open(__import__('cli').__file__, encoding='utf-8') as f:
            source = f.read()

        for func in ('_move_cursor_up', '_move_cursor_down'):
            assert re.search(rf'def {func}\(event\):', source), (
                f"Missing function: {func}"
            )

    def test_move_handlers_have_get_cwidth_in_scope(self):
        import re
        with open(__import__('cli').__file__, encoding='utf-8') as f:
            source = f.read()

        for func in ('_move_cursor_up', '_move_cursor_down'):
            m = re.search(
                rf'( {{8}}def {func}\(event\):.*?)(?=\n {{8}}(?:@kb\.add|def )\b)',
                source,
                re.DOTALL,
            )
            assert m is not None, f"Could not isolate body of {func}"
            body = m.group(1)
            assert 'get_cwidth' in body, f"{func} should reference get_cwidth"
            assert 'from prompt_toolkit.utils import get_cwidth' in body, (
                f"{func} calls get_cwidth but never imports it in scope "
                f"(get_cwidth is not module-level in cli.py) — pressing "
                f"Ctrl+Up/Down would raise NameError at runtime"
            )


# ═══════════════════════════════════════════════════════════════════════════
# Edge cases: emoji, zero-width, terminal boundary conditions
# ═══════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Cursor movement edge cases that could cause off-by-one or crashes."""

    def test_emoji_round_trip(self):
        text = "😀😀😀😀😀"
        cols, pw = 4, 0
        for pos in range(len(text) + 1):
            vr, vc = _visual_row_col(text, pos, cols, pw)
            recovered = _pos_from_visual(text, vr, vc, cols, pw)
            assert recovered <= pos

    def test_zero_width_chars_dont_crash(self):
        text = "e\u0301"
        cols, pw = 80, 2
        vr, vc = _visual_row_col(text, 2, cols, pw)
        assert vr == 0
        assert _pos_from_visual(text, vr, vc, cols, pw) <= 2

    def test_single_char_buffer(self):
        text = "x"
        cols, pw = 80, 2
        for pos in range(len(text) + 1):
            vr, vc = _visual_row_col(text, pos, cols, pw)
            max_row = _max_visual_row(text, cols, pw)
            assert vr <= max_row

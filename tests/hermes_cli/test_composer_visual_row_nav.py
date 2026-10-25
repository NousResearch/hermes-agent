"""Regression for #22009 — Up/Down must walk rendered rows before touching history.

The classic CLI composer is a soft-wrapped prompt_toolkit ``TextArea``. prompt_toolkit's
``Buffer.auto_up`` / ``Buffer.auto_down`` decide "am I on the first/last line?" from
``Document.cursor_position_row``, which counts only hard newlines. Every row of a wrapped draft
therefore looked like the first line, so pressing Up on a long prompt immediately swapped the
draft for a history entry instead of moving the cursor up one visual row.

The rendered-row helpers and the bindings live in ``hermes_cli.cli_composer_nav``; the stub below
mixes in the real ``CLITuiMixin`` so the binding-level cases exercise the shipped wiring.

These tests pin the rendered-row splitter, the resulting cursor movement (same visual column,
clamped to the target row, wide CJK characters counted as two cells, the prompt column charged to
row 0 only), the draft cursor that history navigation has to hand back, and the binding-level
orchestration for both directions.
"""

from __future__ import annotations

from types import SimpleNamespace

from hermes_cli.cli_composer_nav import _composer_rows, _display_width
from hermes_cli.cli_tui_mixin import CLITuiMixin


class _ComposerStub(CLITuiMixin):
    """Just the collaborators the composer bindings ask the CLI for."""

    def __init__(self, columns: int, prompt: str = "") -> None:
        self.columns = columns
        self.prompt = prompt
        self._tui_composer_draft: tuple[str, int] | None = None
        self._tui_composer_cursor_before_walk: int | None = None
        self._skip_paste_collapse = False

    def _tui_composer_columns(self) -> int:
        return self.columns

    def _get_tui_prompt_text(self) -> str:
        return self.prompt


class _BufferStub:
    """Text + cursor + history working line, without a terminal behind it."""

    def __init__(self, text: str = "", cursor: int = 0, *, history_depth: int = 0,
                 working_index: int = 0) -> None:
        self.text = text
        self.cursor_position = cursor
        self.working_index = working_index
        self._working_lines = [""] * (history_depth + 1)


class _HistoryBufferStub(_BufferStub):
    """Emulates what ``Buffer.auto_up`` / ``auto_down`` do with a one-entry history.

    ``auto_up`` steps into history from the first line; ``auto_down`` steps back to the draft and
    parks the cursor at position 0 — the behaviour that lost the edit position before this fix.
    """

    def __init__(self, draft: str, cursor: int, recalled: str = "a recalled entry") -> None:
        # The draft is the LAST working line; history entries sit before it (prompt_toolkit
        # appends the entry the user was editing after the recalled ones).
        super().__init__(draft, cursor, history_depth=1, working_index=1)
        self._working_lines = [recalled, draft]
        self.auto_calls: list[str] = []

    def auto_up(self, count: int = 1) -> None:
        self.auto_calls.append("up")
        self.working_index = 0
        self.text = self._working_lines[0]
        self.cursor_position = len(self.text)

    def auto_down(self, count: int = 1) -> None:
        self.auto_calls.append("down")
        self.working_index = len(self._working_lines) - 1
        self.text = self._working_lines[-1]
        self.cursor_position = 0                      # prompt_toolkit parks at the line start


def _event(buf):
    return SimpleNamespace(arg=1, app=SimpleNamespace(current_buffer=buf))


def _move(cli, text: str, cursor: int, direction: int):
    buf = _BufferStub(text, cursor)
    moved = cli._tui_move_composer_cursor(buf, direction)
    return moved, buf.cursor_position, buf.text


# --------------------------------------------------------------------------- rendered rows


def test_rows_split_a_wrapped_ascii_draft():
    assert _composer_rows("a" * 25, 0, 10) == [(0, 10), (10, 20), (20, 25)]


def test_prompt_width_is_charged_to_the_first_row_only():
    # Row 0 holds 8 cells (12 - 4), later rows take the full width.
    assert _composer_rows("a" * 20, 4, 12) == [(0, 8), (8, 16), (16, 20)]


def test_wide_characters_wrap_by_display_cells():
    assert _display_width("中文abc") == 7
    assert _composer_rows("中" * 7, 0, 10) == [(0, 5), (5, 7)]


def test_hard_newlines_start_a_new_row_without_counting_the_break():
    assert _composer_rows("ab\ncd", 0, 10) == [(0, 2), (3, 5)]


def test_exact_multiple_fills_whole_rows():
    assert _composer_rows("a" * 20, 0, 10) == [(0, 10), (10, 20)]


# --------------------------------------------------------------------------- cursor movement


def test_up_moves_one_rendered_row_keeping_the_column():
    cli = _ComposerStub(columns=20)
    draft = "x" * 50                      # rows: 0-19, 20-39, 40-49
    assert _move(cli, draft, 50, -1) == (True, 30, draft)
    assert _move(cli, draft, 30, -1) == (True, 10, draft)


def test_up_off_the_first_rendered_row_reports_no_movement():
    cli = _ComposerStub(columns=20)
    # False is what lets the binding fall through to history.
    assert _move(cli, "x" * 50, 10, -1) == (False, 10, "x" * 50)


def test_down_moves_one_rendered_row_and_stops_at_the_last():
    cli = _ComposerStub(columns=20)
    draft = "x" * 50
    assert _move(cli, draft, 10, 1) == (True, 30, draft)
    assert _move(cli, draft, 45, 1) == (False, 45, draft)


def test_wide_characters_keep_the_visual_column():
    cli = _ComposerStub(columns=20)       # 10 wide chars per row
    cjk = "汉" * 12
    assert _move(cli, cjk, 12, -1) == (True, 2, cjk)      # 4 cells in on row 2 -> row 1
    assert _move(cli, cjk, 11, -1) == (True, 1, cjk)


def test_cursor_moves_across_hard_newlines():
    cli = _ComposerStub(columns=20)
    draft = "line one\nline two"
    assert _move(cli, draft, len(draft), -1) == (True, 8, draft)
    assert _move(cli, draft, 8, 1) == (True, 17, draft)


def test_prompt_column_is_not_applied_to_later_rows():
    cli = _ComposerStub(columns=20, prompt=">> ")
    draft = "y" * 30                      # row 0 = 17 cells, row 1 = the last 13
    assert _move(cli, draft, 30, -1) == (True, 13, draft)


def test_movement_never_edits_the_draft():
    cli = _ComposerStub(columns=16, prompt="> ")
    draft = "一条很长的中文草稿，长到会折行显示，光标应该只在行之间移动。"
    for cursor in range(len(draft) + 1):
        for direction in (-1, 1):
            _moved, position, text = _move(cli, draft, cursor, direction)
            assert text == draft, "cursor movement must not rewrite the composer"
            assert 0 <= position <= len(draft)


# --------------------------------------------------------------------------- bindings


def test_up_mid_draft_moves_the_cursor_and_leaves_history_alone():
    cli = _ComposerStub(columns=20)
    draft = "x" * 50
    buf = _HistoryBufferStub(draft, 50)
    cli._tui_history_up(_event(buf))
    assert buf.auto_calls == []
    assert (buf.text, buf.cursor_position) == (draft, 30)


def test_up_walks_the_rows_then_recalls_history_from_the_first_one():
    cli = _ComposerStub(columns=20)
    draft = "x" * 50                       # rows: 0-19, 20-39, 40-49 — cursor at the end
    buf = _HistoryBufferStub(draft, 50)

    cli._tui_history_up(_event(buf))        # last row -> row 2
    assert (buf.auto_calls, buf.text, buf.cursor_position) == ([], draft, 30)
    cli._tui_history_up(_event(buf))        # row 2 -> row 1
    assert (buf.auto_calls, buf.text, buf.cursor_position) == ([], draft, 10)

    cli._tui_history_up(_event(buf))        # first rendered row -> history
    assert buf.auto_calls == ["up"]
    assert buf.working_index == 0
    assert buf.text != draft
    assert cli._tui_composer_draft == (draft, 50), "the draft and its edit cursor are kept"


def test_down_returns_the_draft_with_the_edit_cursor():
    cli = _ComposerStub(columns=20)
    draft = "x" * 50
    buf = _HistoryBufferStub(draft, 50)
    for _ in range(3):
        cli._tui_history_up(_event(buf))    # walk the rows, then step into history

    cli._tui_history_down(_event(buf))
    assert buf.auto_calls == ["up", "down"]
    assert buf.text == draft
    assert buf.cursor_position == 50, "history_forward parks at 0; the draft cursor wins"


# --------------------------------------------------------------------------- draft cursor


def test_draft_is_remembered_only_for_the_working_line():
    cli = _ComposerStub(columns=20)
    cli._tui_remember_composer_draft(_BufferStub("draft", 3))
    assert cli._tui_composer_draft == ("draft", 3)

    # Browsing history must not overwrite the saved draft with the entry on screen.
    cli._tui_remember_composer_draft(_BufferStub("recalled entry", 14, history_depth=2,
                                                working_index=1))
    assert cli._tui_composer_draft == ("draft", 3)


def test_empty_composer_is_not_worth_remembering():
    cli = _ComposerStub(columns=20)
    cli._tui_remember_composer_draft(_BufferStub("", 0))
    assert cli._tui_composer_draft is None


def test_the_first_row_walk_parks_the_cursor_and_later_ones_do_not():
    cli = _ComposerStub(columns=20)
    buf = _BufferStub("x" * 50, 44)        # e.g. after a few Lefts
    cli._tui_park_composer_cursor(buf)
    buf.cursor_position = 30
    cli._tui_park_composer_cursor(buf)     # walking on must not move the parked position
    assert cli._tui_composer_cursor_before_walk == 44


def test_a_new_edit_re_parks_the_cursor():
    cli = _ComposerStub(columns=20)
    cli._tui_composer_cursor_before_walk = 44
    cli._tui_composer_cursor_before_walk = None    # what _tui_on_text_changed resets to
    buf = _BufferStub("x" * 50, 51)
    cli._tui_park_composer_cursor(buf)
    assert cli._tui_composer_cursor_before_walk == 51


def test_remember_prefers_the_parked_position_over_the_live_cursor():
    cli = _ComposerStub(columns=20)
    draft = "x" * 149
    cli._tui_composer_cursor_before_walk = 149     # typed at the end, then walked Up
    cli._tui_remember_composer_draft(_BufferStub(draft, 51))
    assert cli._tui_composer_draft == (draft, 149)


def test_remember_falls_back_to_the_live_cursor_when_nothing_is_parked():
    cli = _ComposerStub(columns=20)
    cli._tui_remember_composer_draft(_BufferStub("x" * 30, 12))
    assert cli._tui_composer_draft == ("x" * 30, 12)


def test_restore_puts_the_cursor_back_and_repark_there():
    cli = _ComposerStub(columns=20)
    cli._tui_composer_draft = ("a draft", 4)
    buf = _BufferStub("a draft", 0)        # history_forward parks the cursor at position 0
    cli._tui_restore_composer_draft_cursor(buf)
    assert buf.cursor_position == 4
    assert cli._tui_composer_cursor_before_walk == 4


def test_restore_leaves_other_text_alone():
    cli = _ComposerStub(columns=20)
    cli._tui_composer_draft = ("a draft", 4)
    buf = _BufferStub("a different history entry", 0)
    cli._tui_restore_composer_draft_cursor(buf)
    assert buf.cursor_position == 0

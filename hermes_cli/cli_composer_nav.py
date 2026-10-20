"""Rendered-row navigation for the classic CLI composer (Up/Down, #22009).

The composer is a soft-wrapped prompt_toolkit ``TextArea``. prompt_toolkit's
``Buffer.auto_up`` / ``auto_down`` answer "is the cursor on the first/last line?" from
``Document.cursor_position_row``, which counts only hard newlines — so every rendered row of a
wrapped draft looks like the first line, and Up jumps straight into history, dropping the draft
the user was editing.

This module works in rendered rows instead: the wrap width comes from the terminal, row 0 gives
its leading cells to the prompt, and wide characters (CJK) count as two cells. Up/Down move the
cursor one rendered row and only fall through to history from the first/last rendered row, which
keeps prompt_toolkit's history semantics — and the draft line it keeps for the way back — intact.
"""
from __future__ import annotations

import shutil
from collections.abc import Callable
from typing import Optional


def _cwidth(char: str) -> int:
    """Terminal cells for one character (wide CJK counts two)."""
    try:
        from prompt_toolkit.utils import get_cwidth
    except ImportError:  # partial prompt_toolkit stub (#96075)
        return 1
    return get_cwidth(char)


def _display_width(text: Optional[str]) -> int:
    """Terminal cells for ``text``, matching what prompt_toolkit's renderer lays out."""
    return sum(_cwidth(char) for char in (text or ""))


def _composer_rows(text: str, prompt_width: int, columns: int) -> list[tuple[int, int]]:
    """Character-offset ``(start, end)`` of every *rendered* composer row.

    A long draft is one logical line but several rendered rows, which is the unit Up/Down has to
    move in. Only row 0 gives cells to the prompt; later rows get the full width. ``end`` is
    exclusive and never includes the ``"\\n"`` that ended the logical line.
    """
    columns = max(1, int(columns or 1))
    prompt_width = max(0, int(prompt_width or 0))

    rows: list[tuple[int, int]] = []
    line_start = 0
    for index, line in enumerate((text or "").split("\n")):
        available = max(1, columns - prompt_width) if index == 0 else columns
        row_start, used = line_start, 0
        for offset, char in enumerate(line):
            width = _cwidth(char)
            if width and used + width > available:
                rows.append((row_start, line_start + offset))
                row_start, used = line_start + offset, 0
            used += width
        rows.append((row_start, line_start + len(line)))
        line_start += len(line) + 1  # skip the "\n" itself
    return rows


def _move_cursor_one_row(buf, direction: int, prompt_width: int, columns: int) -> bool:
    """Move ``buf``'s cursor one rendered row up (-1) or down (+1).

    False means there is no row in that direction, which is what lets the bindings fall through to
    history. The same visual column is kept where the target row is long enough, like any editor.
    """
    text = buf.text or ""
    rows = _composer_rows(text, prompt_width, columns)
    cursor = min(max(0, buf.cursor_position), len(text))
    current = 0
    for index, (start, end) in enumerate(rows):
        # Last match wins: a cursor exactly on a row boundary belongs to the row below it.
        if start <= cursor <= end:
            current = index
    target = current + direction
    if target < 0 or target >= len(rows):
        return False

    start, end = rows[target]
    row_text = text[start:end]
    column = min(_display_width(text[rows[current][0]:cursor]), _display_width(row_text))
    position, used = start, 0
    for char in row_text:
        width = _cwidth(char)
        if used + width > column:
            break
        used += width
        position += 1
    buf.cursor_position = position
    return True


class ComposerNavMixin:
    """Up/Down bindings that walk the composer's rendered rows before touching history.

    Applies to the classic CLI composer (``CLITuiMixin``). ``auto_up`` / ``auto_down`` keep doing
    the history work — including parking the draft the user was editing, which is what allows the
    way back — they are just reached from the first/last rendered row instead of the first/last
    logical line.
    """

    # Draft text+cursor remembered across a history recall, restored when it comes back.
    _tui_composer_draft: tuple[str, int] | None = None
    # Where the cursor sat before the current Up/Down row walk started (None until then).
    _tui_composer_cursor_before_walk: int | None = None

    # Provided by the CLI the mixin is mixed into (``CLITuiMixin``).
    _get_tui_prompt_text: Callable[[], str]
    _tui_recall_without_recollapse: Callable[..., None]

    def _tui_composer_columns(self) -> int:
        """Rendered composer width in cells (the terminal's columns)."""
        try:
            from prompt_toolkit.application import get_app
            columns = get_app().output.get_size().columns
        except (AttributeError, ImportError, RuntimeError):
            columns = shutil.get_terminal_size((80, 24)).columns
        return max(1, int(columns or 80))

    def _tui_move_composer_cursor(self, buf, direction: int) -> bool:
        """Move the composer cursor one rendered row; False when there is no row that way."""
        return _move_cursor_one_row(
            buf, direction, _display_width(self._get_tui_prompt_text()), self._tui_composer_columns())

    def _tui_history_up(self, event):
        """Up: one rendered row up; browse history from the first rendered row.

        The draft and its edit cursor are remembered so Down can come back to the same position.
        """
        buf = event.app.current_buffer
        self._tui_park_composer_cursor(buf)
        if self._tui_move_composer_cursor(buf, -1):
            return
        self._tui_remember_composer_draft(buf)
        self._tui_recall_without_recollapse(buf, lambda: buf.auto_up(count=event.arg))

    def _tui_history_down(self, event):
        """Down: one rendered row down; history forward from the last rendered row."""
        buf = event.app.current_buffer
        self._tui_park_composer_cursor(buf)
        if self._tui_move_composer_cursor(buf, 1):
            return
        self._tui_recall_without_recollapse(buf, lambda: buf.auto_down(count=event.arg))
        self._tui_restore_composer_draft_cursor(buf)

    def _tui_park_composer_cursor(self, buf) -> None:
        """Remember the cursor once, before the first row walk of a navigation sequence.

        Coming back from history should land where the user was editing — including a position
        they reached with Left/Right — not on the row this walk happened to end on.
        """
        if self._tui_composer_cursor_before_walk is None:
            self._tui_composer_cursor_before_walk = buf.cursor_position

    def _tui_remember_composer_draft(self, buf) -> None:
        """Save the working draft's text+cursor before a history recall replaces the composer.

        ``history_backward`` parks the cursor at the end of the recalled entry and
        ``history_forward`` parks it at position 0, so without this the draft comes back with the
        user's edit position lost. Only the working line is remembered: while browsing history the
        entry on screen must not overwrite the saved draft.
        """
        lines = getattr(buf, "_working_lines", None)
        if not buf.text or (lines is not None and buf.working_index < len(lines) - 1):
            return
        parked = self._tui_composer_cursor_before_walk
        self._tui_composer_draft = (buf.text, buf.cursor_position if parked is None else parked)

    def _tui_restore_composer_draft_cursor(self, buf) -> None:
        """Put the cursor back where the user was editing once history returns the draft."""
        draft = self._tui_composer_draft
        if draft and buf.text == draft[0]:
            buf.cursor_position = min(draft[1], len(buf.text))
            self._tui_composer_cursor_before_walk = buf.cursor_position

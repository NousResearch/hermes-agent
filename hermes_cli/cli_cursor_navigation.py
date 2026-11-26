"""Visual-row navigation for the classic CLI's wrapped prompt buffer."""

from prompt_toolkit.formatted_text import fragment_list_to_text
from prompt_toolkit.utils import get_cwidth


def navigate_prompt(cli, event, direction: int) -> None:
    """History recalls must not be reinterpreted as freshly pasted input."""
    buffer = event.current_buffer
    before = buffer.text
    cli._skip_paste_collapse = True
    cursor_only = (cli.config or {}).get("display", {}).get("arrow_key_mode") == "cursor"
    move_prompt_cursor(event, direction, cursor_only=cursor_only)
    if buffer.text == before:
        cli._skip_paste_collapse = False


def _visual_rows(control, width: int, height: int) -> list[list[tuple[int, int]]]:
    """Map rendered cells back to buffer offsets, including prompt processors.

    BufferControl owns source/display translation (also used by its mouse handler).
    Rebuild it for each key: fast typing can get ahead of the last painted frame.
    """
    content = control.create_content(width, height)
    document = control.buffer.document
    rows: list[list[tuple[int, int]]] = [[]]
    for line_number, line in enumerate(document.lines):
        if line_number:
            rows.append([])
        processed = control._last_get_processed_line(line_number)
        first = processed.source_to_display(0)
        last = processed.source_to_display(len(line))
        text = fragment_list_to_text(content.get_line(line_number))[:last]
        column = 0
        for display_offset in range(len(text) + 1):
            cell_width = get_cwidth(text[display_offset]) if display_offset < len(text) else 1
            if column + cell_width > width:
                rows.append([])
                column = 0
            if display_offset >= first:
                source = processed.display_to_source(display_offset)
                offset = document.translate_row_col_to_index(line_number, source)
                rows[-1].append((column, offset))
            column += cell_width
    return rows


def move_prompt_cursor(event, direction: int, *, cursor_only: bool = False) -> None:
    """Move one visual row; history is available at the edges in auto mode."""
    buffer = event.current_buffer
    if buffer.complete_state:
        (buffer.complete_previous if direction < 0 else buffer.complete_next)(count=event.arg)
        return
    window = event.app.layout.current_window
    info = window.render_info
    if info is None:
        if cursor_only:
            (buffer.cursor_up if direction < 0 else buffer.cursor_down)(count=event.arg)
        else:
            (buffer.auto_up if direction < 0 else buffer.auto_down)(count=event.arg)
        return
    rows = _visual_rows(window.content, max(1, info.window_width), info.window_height)
    current = buffer.cursor_position
    locations = [(row, column) for row, cells in enumerate(rows) for column, offset in cells if offset == current]
    if not locations:
        return
    row, column = locations[-1]
    previous = getattr(buffer, "_hermes_vertical_goal", None)
    goal = previous[2] if previous and previous[:2] == (buffer.text, current) else column
    target = row + direction * event.arg
    if 0 <= target < len(rows) and rows[target]:
        _, offset = min(rows[target], key=lambda cell: abs(cell[0] - goal))
        buffer.cursor_position = offset
        buffer._hermes_vertical_goal = (buffer.text, offset, goal)
    elif not cursor_only and not buffer.selection_state:
        (buffer.history_backward if direction < 0 else buffer.history_forward)(count=event.arg)
        buffer._hermes_vertical_goal = None

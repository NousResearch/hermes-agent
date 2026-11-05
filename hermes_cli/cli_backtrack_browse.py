"""Transcript pager for the classic CLI's backtrack browse mode (Codex inline-overlay equivalent).

prompt_toolkit cannot scroll the terminal's scrollback, so browsing renders the session's prompts
and replies into a panel of its own: ``› prompt`` / ``• reply`` rows wrapped by terminal cells, the
selected prompt's rows in reverse video, sized to the screen and scrolled to keep the selection in
view. Pure helpers: the mixin in ``cli_backtrack`` owns the state and the keys.
"""

from __future__ import annotations

import shutil

from hermes_cli.cli_panel import _Panel
from hermes_cli.cli_render import _clip_cells, _wrap_panel_text

PROMPT_STYLE = "class:backtrack-prompt bold"
SELECTED_STYLE = "class:backtrack-prompt-selected reverse"
REPLY_STYLE = "class:backtrack-reply"
_BORDER = "class:clarify-border"


def transcript_turns(cli) -> list[tuple[str, str]]:
    """``(prompt, reply)`` per real user turn, oldest first; index = the ordinal /undo rewinds.
    The reply is the visible assistant text up to the next turn (tool calls and handoffs dropped)."""
    from agent.compaction_display import project_compaction_message_for_display
    from hermes_cli.cli_session_mixin import _user_turn_indices
    history = cli.conversation_history or []
    starts = _user_turn_indices(history)
    prompts = cli._backtrack_prompts()
    turns = []
    for k, start in enumerate(starts):
        end = starts[k + 1] if k + 1 < len(starts) else len(history)
        parts = []
        for message in history[start + 1:end]:
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            shown = project_compaction_message_for_display(message) or {}
            text = cli._undo_content_to_text(shown.get("content")).strip()
            if text:
                parts.append(text)
        turns.append((prompts[k] if k < len(prompts) else "", "\n".join(parts)))
    return turns


def _block(marker: str, text: str, width: int) -> list[str]:
    """``marker text`` wrapped to ``width`` cells, continuation rows indented under the text."""
    rows = [row for line in (text or "").splitlines() or [""]
            for row in (_wrap_panel_text(line, width - 2) if line.strip() else [""])]
    return [(marker if i == 0 else " ") + " " + row for i, row in enumerate(rows)]


def transcript_rows(turns, selected_ordinal: int, width: int):
    """Styled rows for the pager plus the ``(first, end)`` row span of the selected prompt."""
    rows, span = [], (0, 0)
    for ordinal, (prompt, reply) in enumerate(turns):
        style = SELECTED_STYLE if ordinal == selected_ordinal else PROMPT_STYLE
        first = len(rows)
        rows.extend((style, row) for row in _block("›", prompt, width))
        if ordinal == selected_ordinal:
            span = (first, len(rows))
        if reply:
            rows.append((REPLY_STYLE, ""))
            rows.extend((REPLY_STYLE, row) for row in _block("•", reply, width))
        if ordinal + 1 < len(turns):
            rows.append((REPLY_STYLE, ""))
    return rows, span


def pager_geometry(reserved_below: int, hint_rows: int = 1) -> tuple[int, int]:
    """``(inner text width, visible body rows)`` for the current terminal: the panel spans the
    width (one column spare so the right border never auto-wraps) and leaves the CLI's chrome."""
    size = shutil.get_terminal_size((100, 24))
    inner = max(20, size.columns - 5)
    body = max(3, size.lines - reserved_below - 2 - hint_rows)  # top and bottom rules
    return inner, body


def scroll_into_view(offset: int, span: tuple[int, int], visible: int, total: int) -> int:
    """Smallest scroll that shows the selected prompt (its first row when taller than the view)."""
    first, end = span
    if first < offset:
        offset = first
    elif end > offset + visible:
        offset = min(first, end - visible)
    return max(0, min(offset, max(0, total - visible)))


def fit_browse_hint(hint: str, width: int) -> str:
    """The browse footer for ``width`` cells, shedding hints the way Codex's prompt-navigation footer
    does: full, then scroll/prompt keys without their words, then without fork, then edit + back,
    then back, then the label. ``hint`` is the localized ``label · keys word · …`` line (5 hints)."""
    from prompt_toolkit.utils import get_cwidth
    label, *hints = hint.split(" · ")
    if len(hints) != 5 or get_cwidth(hint) <= width:
        return hint
    scroll, prompts, edit, fork, back = hints
    nav = f"{scroll.split(' ')[0]} {prompts.split(' ')[0]}"
    for rung in ([nav, edit, fork, back], [nav, edit, back], [edit, back], [back]):
        line = " · ".join([label, *rung])
        if get_cwidth(line) <= width:
            return line
    return label


def render_pager(state, turns, title: str, hint: str, reserved_below: int) -> list:
    """Panel fragments. ``state["_scroll_offset"]`` is clamped in place; a pending
    ``state["_reveal"]`` (set on open and by ←/→) first scrolls the selected prompt into view.
    ``state["_page"]`` records the visible row count for PgUp/PgDn."""
    inner, _ = pager_geometry(reserved_below)
    hint_rows = _wrap_panel_text(fit_browse_hint(hint, inner), inner)
    inner, visible = pager_geometry(reserved_below, len(hint_rows))
    ordinal = state["entries"][state["selected"]][0]
    rows, span = transcript_rows(turns, ordinal, inner)
    offset = max(0, min(state.get("_scroll_offset", 0), max(0, len(rows) - visible)))
    if state.pop("_reveal", False):
        offset = scroll_into_view(offset, span, visible, len(rows))
    state["_scroll_offset"], state["_page"] = offset, visible
    panel = _Panel(_BORDER, inner + 2, _clip_cells(title, inner - 2), "class:clarify-title")
    for style, text in rows[offset:offset + visible]:
        panel.row(style, text)
    for row in hint_rows:
        panel.row("class:clarify-hint", row)
    return panel.close()

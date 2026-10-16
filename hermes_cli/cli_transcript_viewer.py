"""Ctrl+O: a temporary full-screen view of the recent turns' tool calls (full output, exit code,
duration, diffs) and reasoning, over a bounded in-memory log the classic CLI fills as it prints.

The scrollback stays append-only: the viewer is its own prompt_toolkit Application run
``in_terminal`` (the Ctrl+T monitor's pattern), so leaving it returns to the untouched transcript.
Frontend only; nothing here reaches the model.
"""
from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from agent.i18n import t
from hermes_cli import cli_tool_render as render

MAX_ENTRIES = 300
MAX_TEXT_CHARS = 200_000


def _bounded(text: str) -> str:
    if len(text) <= MAX_TEXT_CHARS:
        return text
    half = MAX_TEXT_CHARS // 2
    return f"{text[:half]}\n{t('cli.transcript.chars_omitted', count=len(text) - MAX_TEXT_CHARS)}\n{text[-half:]}"


@dataclass
class TranscriptEntry:
    kind: str  # "tool" | "reasoning"
    turn: int
    name: str = ""
    args: dict = field(default_factory=dict)
    result: str = ""
    exit_code: int | None = None
    duration: float | None = None
    failed: bool = False
    diff: str = ""
    call_id: str = ""


class TranscriptLog:
    """The last ``MAX_ENTRIES`` tool calls / reasoning blocks, tagged with the turn they belong to."""

    def __init__(self, maxlen: int = MAX_ENTRIES):
        self.entries: deque[TranscriptEntry] = deque(maxlen=maxlen)
        self.turn = 0

    def begin_turn(self) -> None:
        self.turn += 1

    def add_tool(self, name: str, args: dict | None, result: Any, *, duration: float | None, failed: bool,
                 call_id: str = "") -> TranscriptEntry:
        entry = TranscriptEntry("tool", self.turn, name=name or "", args=dict(args or {}),
                                result=_bounded(render.result_text(result)), exit_code=render.exit_code_of(result),
                                duration=duration, failed=failed, call_id=call_id or "")
        self.entries.append(entry)
        return entry

    def add_reasoning(self, text: str, duration: float | None) -> None:
        if text and text.strip():
            self.entries.append(TranscriptEntry("reasoning", self.turn, result=_bounded(text.strip()),
                                                duration=duration))

    def attach_diff(self, call_id: str, name: str, diff: str) -> None:
        """Attach an edit's diff to its call (by id, else the latest call of that tool)."""
        for entry in reversed(self.entries):
            if entry.kind == "tool" and ((call_id and entry.call_id == call_id) or (not call_id and entry.name == name)):
                entry.diff = _bounded(diff)
                return

    def latest_turn(self) -> int | None:
        return self.entries[-1].turn if self.entries else None

    def select(self, *, all_turns: bool) -> list[TranscriptEntry]:
        latest = self.latest_turn()
        return [e for e in self.entries if all_turns or e.turn == latest]


# ── plain-text document ──────────────────────────────────────────────────────────────────────

def _indent(text: str) -> list[str]:
    return [f"    {line}" for line in render.clean_text(text).splitlines()]


def _tool_header(entry: TranscriptEntry) -> str:
    from agent.display import cute_tool_rows
    rows, _, _ = cute_tool_rows(entry.name, entry.args, None)
    label = " · ".join(row.removeprefix("┊ ") for row in rows)
    meta = [render.format_duration(entry.duration)] if entry.duration is not None else []
    if entry.exit_code is not None:
        meta.append(t("cli.transcript.exit_code", code=entry.exit_code))
    from agent.redact import redact_sensitive_text
    label = render.clean_text(redact_sensitive_text(label))
    return f"{'✗' if entry.failed else '✓'} {label}" + (f"   {' · '.join(meta)}" if meta else "")


def _args_line(entry: TranscriptEntry) -> list[str]:
    from agent.display import redact_tool_args_for_display
    from agent.redact import redact_sensitive_text
    args = redact_tool_args_for_display(entry.name, entry.args) or entry.args
    if not args:
        return []
    text = redact_sensitive_text(json.dumps(args, ensure_ascii=False, default=str))
    return [f"  {t('cli.transcript.args_label')} {render.clip(render.clean_text(text), 400)}"]


def _tool_block(entry: TranscriptEntry) -> list[str]:
    from agent.redact import redact_sensitive_text
    lines = [_tool_header(entry), *_args_line(entry)]
    body = redact_sensitive_text(entry.result) if entry.result.strip() else ""
    lines.extend(_indent(body) if body else [f"    {t('cli.tool_render.no_output')}"])
    if entry.diff:
        lines.append(f"  {t('cli.transcript.diff_label')}")
        lines.extend(_indent(entry.diff))
    return lines


def _reasoning_block(entry: TranscriptEntry) -> list[str]:
    label = (t("cli.tool_render.thought_for", duration=render.format_duration(entry.duration))
             if entry.duration and entry.duration >= 1 else t("cli.tool_render.thought"))
    return [f"✻ {label}", *_indent(entry.result)]


def document(entries: list[TranscriptEntry]) -> str:
    """The viewer body: one block per entry, a turn rule whenever the turn changes."""
    if not entries:
        return t("cli.transcript.empty")
    lines: list[str] = []
    turn = None
    for entry in entries:
        if entry.turn != turn:
            turn = entry.turn
            if lines:
                lines.append("")
            lines.append(f"── {t('cli.transcript.turn_label', number=turn)} " + "─" * 30)
        lines.append("")
        lines.extend(_reasoning_block(entry) if entry.kind == "reasoning" else _tool_block(entry))
    return "\n".join(lines)


# ── application ──────────────────────────────────────────────────────────────────────────────

def _line_style(line: str) -> str:
    if line.startswith("✗"):
        return "class:transcript.error"
    if line.startswith(("✓", "✻", "──")):
        return "class:transcript.heading"
    stripped = line[4:] if line.startswith("    ") else ""
    if stripped[:1] == "+" and not stripped.startswith("+++"):
        return "class:transcript.added"
    if stripped[:1] == "-" and not stripped.startswith("---"):
        return "class:transcript.removed"
    return ""


def _lexer():
    from prompt_toolkit.lexers import Lexer

    class TranscriptLexer(Lexer):
        def lex_document(self, document):
            lines = document.lines

            def line(number: int):
                return [(_line_style(lines[number]), lines[number])] if number < len(lines) else []
            return line

    return TranscriptLexer()


def _styles():
    from prompt_toolkit.styles import Style, merge_styles
    from hermes_cli.skin_engine import get_prompt_toolkit_style_overrides
    ok, bad = render.skin_hex("ui_ok", "#4caf50"), render.skin_hex("ui_error", "#ef5350")
    return merge_styles([
        Style.from_dict(get_prompt_toolkit_style_overrides()),
        Style.from_dict({"transcript.heading": "bold", "transcript.error": f"bold {bad}",
                         "transcript.added": ok, "transcript.removed": bad,
                         "transcript.bar": "reverse"}),
    ])


def _search_next(area, *, backwards: bool) -> None:
    from prompt_toolkit.search import SearchDirection
    state = area.control.search_state
    if not state.text:
        return
    state.direction = SearchDirection.BACKWARD if backwards else SearchDirection.FORWARD
    area.buffer.apply_search(state, include_current_position=False, count=1)


def build_viewer_application(log: TranscriptLog, **kwargs):
    """Full-screen scrollable view of the last turn (``a`` toggles every retained turn)."""
    from prompt_toolkit.application import Application
    from prompt_toolkit.filters import Condition, is_searching
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import HSplit, Layout, Window
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.search import start_search
    from prompt_toolkit.widgets import SearchToolbar, TextArea

    state = {"all": False}
    search = SearchToolbar()
    area = TextArea(read_only=True, scrollbar=True, wrap_lines=True, lexer=_lexer(), search_field=search)

    def load():
        area.text = document(log.select(all_turns=state["all"]))
        area.buffer.cursor_position = 0

    def header():
        scope = t("cli.transcript.scope_all" if state["all"] else "cli.transcript.scope_last")
        return [("class:transcript.bar", f" {t('cli.transcript.title')} · {scope} ")]

    kb = KeyBindings()
    browsing = ~is_searching

    @kb.add("/", filter=browsing)
    def find(event):
        start_search(area.control)

    @kb.add("n", filter=browsing)
    def next_match(event):
        _search_next(area, backwards=False)

    @kb.add("N", filter=browsing)
    def prev_match(event):
        _search_next(area, backwards=True)

    @kb.add("a", filter=browsing)
    def toggle_scope(event):
        state["all"] = not state["all"]
        load()

    @kb.add("g", filter=browsing)
    def top(event):
        area.buffer.cursor_position = 0

    @kb.add("G", filter=browsing)
    def bottom(event):
        area.buffer.cursor_position = len(area.text)

    @kb.add("space", filter=browsing)
    def page_down(event):
        from prompt_toolkit.key_binding.bindings.scroll import scroll_page_down
        scroll_page_down(event)

    @kb.add("q", filter=browsing)
    @kb.add("escape", filter=browsing, eager=True)
    @kb.add("c-o", filter=browsing)
    @kb.add("c-c")
    def close(event):
        event.app.exit()

    load()
    layout = Layout(HSplit([
        Window(FormattedTextControl(header), height=1),
        area,
        search,
        Window(FormattedTextControl(lambda: [("class:transcript.bar", f" {t('cli.transcript.footer')} ")]), height=1),
    ]), focused_element=area)
    kwargs.setdefault("style", _styles())
    return Application(layout=layout, key_bindings=kb, full_screen=True, mouse_support=False, **kwargs)


def open_viewer(cli) -> None:
    """Ctrl+O handler: run the viewer in the terminal, then hand the screen back to the prompt."""
    import asyncio
    from prompt_toolkit.application import in_terminal
    log = getattr(cli, "_transcript_log", None)
    if log is None or getattr(cli, "_transcript_viewer_open", False):
        return
    cli._transcript_viewer_open = True

    async def run():
        try:
            async with in_terminal():
                await build_viewer_application(log).run_async()
        finally:
            cli._transcript_viewer_open = False
            cli._invalidate()

    asyncio.get_running_loop().create_task(run())


def bind_keys(cli, kb) -> None:
    """Ctrl+O opens the viewer (not in vi insert mode, where Ctrl+O is vi's one-shot normal mode)."""
    from prompt_toolkit.filters import Condition, vi_insert_mode
    from hermes_cli.cli_subagent_monitor import modal_prompt_active
    free = Condition(lambda: not modal_prompt_active(cli)) & ~vi_insert_mode
    kb.add("c-o", filter=free)(lambda event: open_viewer(cli))

"""Rich Markdown in the classic CLI, with prompt_toolkit owning all live drawing.

The unfinished block lives in the application layout; completed blocks print above
it through run_in_terminal. Reference links retain their document context until final.
"""
from __future__ import annotations

import asyncio
from io import StringIO
from threading import RLock

from markdown_it import MarkdownIt
from prompt_toolkit.data_structures import Point
from prompt_toolkit.formatted_text import ANSI, to_formatted_text
from prompt_toolkit.layout import Window
from prompt_toolkit.layout.controls import UIContent, UIControl
from prompt_toolkit.layout.dimension import Dimension
from rich.console import Console
from rich.markdown import CodeBlock, Heading, Markdown, Paragraph
from rich.syntax import Syntax
from rich.theme import Theme
from rich.text import Text

from hermes_cli.markdown_rendering import normalize_source


class _Heading(Heading):
    # Rich 14 removed the base class' LEVEL_ALIGN constant; keep our heading
    # alignment override compatible with both the older and newer APIs.
    LEVEL_ALIGN = dict.fromkeys(range(1, 7), "left")


class _CodeBlock(CodeBlock):
    def __rich_console__(self, console, options):
        # No decorative padding: copying code must preserve its original indentation.
        yield Syntax(str(self.text).rstrip("\n"), self.lexer_name, theme=self.theme,
                     word_wrap=True, padding=0, background_color="#20252b")


class ReadableMarkdown(Markdown):
    elements = {**Markdown.elements, "heading_open": _Heading,
                "fence": _CodeBlock, "code_block": _CodeBlock}


class _TerminalParagraph(Paragraph):
    @classmethod
    def create(cls, markdown, token):
        # Only top-level prose can use terminal soft wrapping; nested lists need hanging indents.
        return (cls if token.level == 0 else Paragraph)(justify="left")

    def __rich_console__(self, console, options):
        yield from console.render(self.text, options.update(no_wrap=True, overflow="ignore"))


class _ScrollbackMarkdown(ReadableMarkdown):
    elements = {**ReadableMarkdown.elements, "paragraph_open": _TerminalParagraph}


def make_markdown(source: str, width: int, *, terminal_wrap: bool = False) -> Markdown:
    """One normalized Rich renderable for final output, streaming and command views."""
    source = normalize_source(source, width)
    markdown = _ScrollbackMarkdown if terminal_wrap else ReadableMarkdown
    return markdown(source, code_theme="github-dark", justify="left", hyperlinks=False)


def render_markdown(source: str, width: int, *, color: bool = True,
                    terminal_wrap: bool = False) -> str:
    buf = StringIO()
    console = Console(file=buf, width=max(1, width), height=25, force_terminal=color,
                      color_system="truecolor" if color else None,
                      theme=Theme({"markdown.h1": "bold", "markdown.h2": "bold",
                                   "markdown.h3": "bold", "markdown.code": "bold cyan"}))
    console.print(make_markdown(source, width, terminal_wrap=terminal_wrap), crop=False)
    return buf.getvalue().rstrip("\n")


def assistant_label(width: int, *, color: bool = True) -> str:
    from hermes_cli.skin_engine import get_active_skin
    skin = get_active_skin()
    buf = StringIO()
    Console(file=buf, width=max(1, width), height=25, force_terminal=color,
            color_system="truecolor" if color else None).print(Text(
                skin.get_branding("response_label", "⚕ Hermes").strip(),
                style=skin.get_color("banner_dim", "#8B949E")))
    return buf.getvalue().rstrip("\n")


def print_markdown(committed: str, *, live: bool = False, label: bool = False) -> None:
    """Commit complete Markdown without creating live-stream state."""
    from hermes_cli.cli_conversation_display import emit_display_event
    if committed.strip():
        def render(width, source=committed):
            import sys
            color = live or sys.stdout.isatty()
            prefix = assistant_label(width, color=color) + "\n" if label else ""
            return prefix + render_markdown(source, width, color=color, terminal_wrap=True) + "\n"
        emit_display_event(render)


class MarkdownAccumulator:
    """Synchronous Markdown boundary tracker; it owns source, pending, and committed blocks."""

    def __init__(self):
        self.pending = ""
        self._parser = MarkdownIt().enable("table").enable("strikethrough")
        self.message_start = True

    def append(self, text: str, *, final: bool = False) -> list[tuple[str, bool]]:
        self.pending += text
        cut = len(self.pending) if final else self.stable_prefix(self.pending)
        committed, self.pending = self.pending[:cut], self.pending[cut:]
        label = self.message_start
        if committed.strip():
            self.message_start = False
        if final:
            self.message_start = True
        return [(committed, label)] if committed.strip() else []

    def stable_prefix(self, source: str) -> int:
        if len(source) > 8192:
            return 0
        env = {}
        tokens = self._parser.parse(source, env)
        if env.get("references") or any(
            child.type == "text" and "[" in child.content
            for token in tokens for child in (token.children or ())
        ):
            return 0
        starts = [t.map[0] for t in tokens if t.level == 0 and t.map and t.nesting != -1]
        if len(starts) < 2:
            return 0
        return sum(len(line) for line in source.splitlines(keepends=True)[:starts[-1]])


class MarkdownStream:
    def __init__(self, cli):
        self.cli = cli
        self.accumulator = MarkdownAccumulator()
        self._lock = RLock()
        self._cache = None
        self._operations = asyncio.Lock()

    @property
    def pending(self):
        return self.accumulator.pending

    def preview(self, width: int) -> list[str]:
        with self._lock:
            key = (self.pending, width, self.accumulator.message_start)
            if self._cache is None or self._cache[0] != key:
                if len(self.pending) > 8192:
                    # A bounded literal tail keeps input responsive for huge unfinished
                    # fences/lists. Full source still receives Markdown formatting on commit.
                    buf = StringIO()
                    Console(file=buf, width=max(1, width)).print(Text("… pending block (tail)\n" + self.pending[-4096:]))
                    lines = buf.getvalue().splitlines()
                else:
                    lines = render_markdown(self.pending, width).splitlines() if self.pending else []
                if self.pending and self.accumulator.message_start:
                    lines = assistant_label(width).splitlines() + lines
                self._cache = key, lines
            return self._cache[1]

    def feed(self, text: str, *, final: bool = False, after=None) -> None:
        """Callbacks arrive from the agent worker; finish printing before it emits tool status."""
        app = getattr(self.cli, "_app", None)
        if app is None or not app.is_running:
            self._update(text, final, live=False)
            if after is not None:
                after()
            return

        async def update():
            from prompt_toolkit.application import run_in_terminal
            # Serialize the entire handoff: UI notifications must not overtake worker
            # deltas or compute a pending-source snapshot across another commit.
            async with self._operations:
                with self._lock:
                    source = self.pending + text
                    cut = len(source) if final else self._stable_prefix(source)
                if cut or after is not None:
                    def commit():
                        self._update(text, final, live=True)
                        if after is not None:
                            after()
                    await run_in_terminal(commit)
                else:
                    with self._lock:
                        self.accumulator.append(text, final=final)
                app.invalidate()

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is app.loop:
            app.create_background_task(update())
        else:
            asyncio.run_coroutine_threadsafe(update(), app.loop).result()

    def _update(self, text: str, final: bool, *, live: bool) -> None:
        with self._lock:
            blocks = self.accumulator.append(text, final=final)
        for committed, label in blocks:
            print_markdown(committed, live=live, label=label)

    def _stable_prefix(self, source: str) -> int:
        return self.accumulator.stable_prefix(source)


class _PreviewControl(UIControl):
    def __init__(self, cli):
        self.cli = cli

    def _lines(self, width):
        stream = getattr(self.cli, "_markdown_stream", None)
        return stream.preview(width) if stream else []

    def preferred_height(self, width, max_available_height, wrap_lines, get_line_prefix):
        return min(len(self._lines(width)), max_available_height)

    def create_content(self, width, height):
        lines = self._lines(width)
        fragments = [to_formatted_text(ANSI(line)) for line in lines]
        return UIContent(get_line=lambda i: fragments[i] if i < len(fragments) else [], line_count=len(lines),
                         cursor_position=Point(x=0, y=max(0, len(lines)-1)), show_cursor=False)


def preview_window(cli):
    from prompt_toolkit.application import get_app

    def height():
        # Approval and question widgets already budget the screen; give them their rows back.
        modal_states = ("_approval_state", "_clarify_state", "_sudo_state", "_secret_state",
                        "_slash_confirm_state", "_model_picker_state", "_command_palette_state")
        if any(getattr(cli, name, None) for name in modal_states):
            return Dimension.exact(0)
        return Dimension(min=0, max=max(1, get_app().output.get_size().rows // 2))

    return Window(_PreviewControl(cli), wrap_lines=False, always_hide_cursor=True, height=height)

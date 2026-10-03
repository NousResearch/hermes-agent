"""Exercise the real footer fragments through prompt_toolkit's diff renderer."""
import io
import re

import pytest
from prompt_toolkit.application import Application
from prompt_toolkit.application.current import set_app
from prompt_toolkit.data_structures import Size
from prompt_toolkit.input import DummyInput
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.output.vt100 import Vt100_Output

from hermes_cli.cli_status_bar_mixin import CLIStatusBarMixin


class _Terminal:
    """Small VT grid for the emitted cursor moves; VS16 expands a glyph to two cells."""
    def __init__(self):
        self.grid = [[" "] * 120 for _ in range(4)]
        self.x = self.y = 0
        self.last = None

    def feed(self, data):
        i = 0
        while i < len(data):
            ch = data[i]
            if ch == "\x1b":
                match = re.match(r"\x1b\[([?0-9;]*)([A-Za-z])", data[i:])
                if match:
                    arg, cmd = match.groups()
                    n = int(arg) if arg.isdigit() else 1
                    if cmd == "C":
                        self.x = min(119, self.x + n)
                    elif cmd == "D":
                        self.x = max(0, self.x - n)
                    elif cmd == "A":
                        self.y = max(0, self.y - n)
                    elif cmd == "B":
                        self.y = min(3, self.y + n)
                    elif cmd == "K" or (cmd == "J" and not arg):
                        self.grid[self.y][self.x:] = [" "] * (120 - self.x)
                        if cmd == "J":
                            for row in range(self.y + 1, 4):
                                self.grid[row] = [" "] * 120
                    i += len(match.group())
                    continue
                i += 1
                continue
            if ch == "\r":
                self.x = 0
            elif ch == "\n":
                self.y = min(3, self.y + 1)
            elif ch == "\ufe0f" and self.last is not None:
                row, col = self.last
                self.grid[row][col + 1] = ""
                self.x = col + 2
            elif ord(ch) >= 0x20:
                if self.x < 120:
                    self.grid[self.y][self.x] = ch
                    self.last = (self.y, self.x)
                self.x += 1
            i += 1

    def row(self):
        return "".join(self.grid[1]).strip()


class _Output(io.StringIO):
    encoding = "utf-8"


@pytest.mark.parametrize("compressions,enabled", [(0, True), (1, True), (12, True), (1, False)])
def test_timer_diff_redraw_after_compression(compressions, enabled):
    cli = CLIStatusBarMixin()
    snapshot = dict(model_short="test", duration="1h", context_percent=None,
                    context_length=0, compressions=compressions)
    fields = {"prompt_elapsed"}
    if enabled:
        fields.add("compressions")
    fragments = []
    out = _Output()
    terminal = _Terminal()
    layout = Layout(HSplit([
        Window(FormattedTextControl([("", "> "), ("[SetCursorPosition]", "")],
                                    focusable=True, show_cursor=True), height=1),
        Window(FormattedTextControl(lambda: fragments), height=1),
    ]))
    app = Application(layout=layout, input=DummyInput(),
                      output=Vt100_Output(out, lambda: Size(rows=4, columns=120),
                                          term="xterm-256color"), full_screen=False)
    for seconds in range(295, 312):
        minutes, secs = divmod(seconds, 60)
        expected = f"{minutes}m {secs}s" if secs else f"{minutes}m"
        snapshot["prompt_elapsed"] = f"⏱ {expected}"
        segments = cli._status_bar_segments(snapshot, 120, fields, False, styled=True)
        fragments[:] = []
        for segment in segments:
            if fragments:
                fragments.append(("", " | "))
            fragments.extend(segment)
        app.render_counter += 1
        with set_app(app):
            app.renderer.render(app, layout)
        terminal.feed(out.getvalue())
        out.seek(0)
        out.truncate()
        assert terminal.row().split("⏱ ")[1] == expected
    assert (len(segments) == 2) == bool(compressions and enabled)

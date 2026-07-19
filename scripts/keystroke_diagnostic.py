#!/usr/bin/env python3
"""Diagnose how prompt_toolkit identifies keystrokes in the current terminal.

Useful when adding a keybinding to Hermes (or any prompt_toolkit app) and you
need to know what the terminal actually delivers — particularly on Windows,
where terminals can collapse, intercept, or silently remap key combinations.

Usage:
    # POSIX
    python scripts/keystroke_diagnostic.py

    # Windows (PowerShell / git-bash / cmd)
    python scripts\\keystroke_diagnostic.py

Press the key combinations you care about. Each keystroke prints the
prompt_toolkit `Keys.*` identifier and the raw escape bytes the terminal
sent. The last 20 keystrokes stay on screen. Ctrl+Q or Ctrl+C to quit.

Common questions this answers:
    - Does my terminal distinguish Ctrl+Enter from plain Enter?
      (On Windows Terminal: yes, Ctrl+Enter → c-j, Enter → c-m.)
    - Does Alt+Enter reach the app, or does the terminal eat it?
      (Windows Terminal eats it for fullscreen; mintty may too.)
    - Does Shift+Enter register as a separate key?
      (Almost never — most terminals collapse it to Enter.)
    - What byte sequence does Home/End/PageUp/etc. produce?

Example output for Ctrl+Enter on Windows Terminal + PowerShell:
    key=<Keys.ControlJ: 'c-j'> data='\\n'

Then in Hermes, bind the newline behaviour to that key:
    @kb.add('c-j')
    def handle_ctrl_enter(event):
        event.current_buffer.insert_text('\\n')
"""
from prompt_toolkit import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import Window
from prompt_toolkit.layout.controls import FormattedTextControl


_HISTORY: list[str] = []


def _header() -> list[str]:
    return [
        "Keystroke diagnostic — press keys to see how prompt_toolkit sees them.",
        "Try: Enter, Ctrl+Enter, Shift+Enter, Alt+Enter, Ctrl+J, Ctrl+M, arrows, Home/End.",
        "Ctrl+Q or Ctrl+C to quit. Last 20 keystrokes shown.",
        "",
    ]


def _render_text() -> str:
    return "\n".join(_header() + _HISTORY[-20:])


def main() -> None:
    kb = KeyBindings()

    @kb.add("<any>")
    def _on_any(event):  # noqa: ANN001 — prompt_toolkit event type
        parts = []
        for kp in event.key_sequence:
            parts.append(f"key={kp.key!r} data={kp.data!r}")
        _HISTORY.append(" | ".join(parts))
        event.app.invalidate()

    @kb.add("c-q")
    @kb.add("c-c")
    def _quit(event):  # noqa: ANN001
        event.app.exit()

    control = FormattedTextControl(text=_render_text)
    layout = Layout(Window(content=control))
    Application(layout=layout, key_bindings=kb, full_screen=False).run()


if __name__ == "__main__":
    main()

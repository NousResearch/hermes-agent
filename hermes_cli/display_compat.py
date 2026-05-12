"""ASCII fallbacks for terminals with weak Unicode rendering.

Git Bash / MSYS2 on Windows can accept UTF-8 bytes while still rendering some
decorative CLI glyphs as ``?``.  Keep the normal rich glyphs everywhere else,
but downgrade display-only adornments to ASCII in those terminals.
"""

from __future__ import annotations

import os
import sys

_ASCII_SPINNER_FRAMES = ("-", "\\", "|", "/")
_ASCII_WAITING_FACES = ["(...)", "(wait)", "(idle)"]
_ASCII_THINKING_FACES = ["(...)", "(think)", "(work)"]

_ASCII_REPLACEMENTS = (
    ("◆ Hermes:", "* Hermes:"),
    ("● You:", "* You:"),
    ("⚕ Hermes Agent", "Hermes Agent"),
    ("⚕ Hermes", "Hermes"),
    ("⚠️", "!"),
    ("⚠", "!"),
    ("💾", "[save]"),
    ("📎", "[img]"),
    ("👁️", "view"),
    ("👁", "view"),
    ("🖼️", "img"),
    ("🖼", "img"),
    ("⌨️", "key"),
    ("⌨", "key"),
    ("✍️", "write"),
    ("✍", "write"),
    ("🕸️", "web"),
    ("🕸", "web"),
    ("🔍", "search"),
    ("📄", "file"),
    ("💻", "$"),
    ("⚙️", "proc"),
    ("⚙", "proc"),
    ("📖", "read"),
    ("🔧", "patch"),
    ("🔎", "grep"),
    ("🌐", "web"),
    ("📸", "snap"),
    ("👆", "click"),
    ("📋", "plan"),
    ("🧠", "mem"),
    ("📚", "skills"),
    ("🎨", "art"),
    ("🔊", "speak"),
    ("📨", "send"),
    ("⏰", "cron"),
    ("🧪", "rl"),
    ("🐍", "py"),
    ("🔀", "->"),
    ("◀️", "back"),
    ("◀", "back"),
    ("✨", "*"),
    ("✦", "*"),
    ("⚡", "*"),
    ("⏳", "*"),
    ("✓", "OK"),
    ("✔", "OK"),
    ("✗", "x"),
    ("✘", "x"),
    ("❯", ">"),
    ("›", ">"),
    ("»", ">>"),
    ("→", "->"),
    ("←", "<-"),
    ("↑", "^"),
    ("↓", "v"),
    ("┊", "|"),
)


def terminal_prefers_ascii() -> bool:
    """Return True when the current terminal should avoid decorative Unicode."""
    forced = os.getenv("HERMES_FORCE_ASCII_DISPLAY")
    if forced is not None:
        return forced.strip().lower() not in {"", "0", "false", "no", "off"}

    if sys.platform != "win32":
        return False

    if os.getenv("MSYSTEM"):
        return True

    ostype = (os.getenv("OSTYPE") or "").lower()
    term = (os.getenv("TERM") or "").lower()
    return any(token in ostype or token in term for token in ("msys", "mingw"))


def make_terminal_display_safe(text: str) -> str:
    """Downgrade display-only glyphs to ASCII when the terminal needs it."""
    rendered = str(text)
    if not rendered or not terminal_prefers_ascii():
        return rendered

    for source, replacement in _ASCII_REPLACEMENTS:
        rendered = rendered.replace(source, replacement)
    return rendered


def get_ascii_spinner_frames() -> tuple[str, ...]:
    return _ASCII_SPINNER_FRAMES


def get_ascii_waiting_faces() -> list[str]:
    return list(_ASCII_WAITING_FACES)


def get_ascii_thinking_faces() -> list[str]:
    return list(_ASCII_THINKING_FACES)

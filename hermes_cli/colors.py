"""Shared ANSI color utilities for Hermes CLI modules."""

import os
import sys


def should_use_color() -> bool:
    """Return True when colored output is appropriate."""
    if os.environ.get("NO_COLOR") is not None or os.environ.get("TERM") == "dumb":
        return False
    return bool(sys.stdout.isatty())


class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"


class Palette:
    """Semantic color roles for the Hermes CLI.

    Callers name the intent (success, warning, heading) and the palette
    decides the ANSI codes. Raw :class:`Colors` values stay available.
    Every role is a tuple so ``color`` and :func:`semantic` can flatten it.
    ``HEADING`` is cyan + bold, matching section titles elsewhere in the CLI.
    """

    SUCCESS = (Colors.GREEN,)
    WARNING = (Colors.YELLOW,)
    ERROR = (Colors.RED,)
    INFO = (Colors.DIM,)
    MUTED = (Colors.DIM,)
    HEADING = (Colors.CYAN, Colors.BOLD)
    EMPHASIS = (Colors.BOLD,)
    PROMPT = (Colors.YELLOW,)


def _flatten_codes(codes) -> list[str]:
    """Flatten nested palette tuples so ``"".join`` only sees ANSI strings."""
    flat: list[str] = []
    for code in codes:
        if isinstance(code, (tuple, list)):
            flat.extend(_flatten_codes(code))
        else:
            flat.append(code)
    return flat


def color(text: str, *codes) -> str:
    """Apply color codes to text (only when color output is appropriate).

    ``codes`` may be raw :class:`Colors` strings, a :class:`Palette` role
    tuple, or a mix. Nested tuples and lists are flattened, so both
    ``color(t, Colors.GREEN)`` and ``color(t, Palette.SUCCESS)`` work, and
    a group such as ``(Palette.HEADING, Colors.DIM)`` does not leave a tuple
    for ``"".join``.
    """
    if not should_use_color():
        return text
    return "".join(_flatten_codes(codes)) + text + Colors.RESET


def semantic(text: str, role) -> str:
    """Color ``text`` with a semantic :class:`Palette` role."""
    return color(text, role)

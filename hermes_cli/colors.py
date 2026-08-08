"""Shared ANSI color utilities for Hermes CLI modules."""

import os
import sys


def should_use_color() -> bool:
    """Return True when colored output is appropriate.

    Respects the NO_COLOR environment variable (https://no-color.org/)
    and TERM=dumb, in addition to the existing TTY check.
    """
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    if not sys.stdout.isatty():
        return False
    return True


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
    """Semantic color roles for the Hermes CLI design system.

    Centralizes *which raw color means what* so callers express intent
    ("this is a success message") instead of hand-picking ``Colors.GREEN``
    in one place and ``Colors.CYAN`` in another for the same role. This is
    the single source of truth for the CLI's visual identity; raw
    :class:`Colors` codes remain available for one-off needs and for
    backwards compatibility.

    Roles map to ANSI escape sequences (the same strings exposed by
    :class:`Colors`). Pass them to :func:`color` exactly like the raw codes::

        print(color("saved", Palette.SUCCESS))
        print(color("section title", *Palette.HEADING))

    ``HEADING`` and ``EMPHASIS`` are tuples because they combine two codes
    (a base color plus ``BOLD``); :func:`color` already accepts any number
    of codes, and :func:`semantic` unpacks tuples for you.
    """

    SUCCESS = (Colors.GREEN,)
    WARNING = (Colors.YELLOW,)
    ERROR = (Colors.RED,)
    INFO = (Colors.DIM,)
    MUTED = (Colors.DIM,)
    HEADING = (Colors.CYAN, Colors.BOLD)
    EMPHASIS = (Colors.BOLD,)
    PROMPT = (Colors.YELLOW,)


def color(text: str, *codes) -> str:
    """Apply color codes to text (only when color output is appropriate).

    ``codes`` may be raw :class:`Colors` strings, a :class:`Palette` role
    tuple, or a mix; nested tuples/lists are flattened so callers can write
    either ``color(t, Colors.GREEN)`` or ``color(t, Palette.SUCCESS)``.
    """
    flat: list[str] = []
    for c in codes:
        if isinstance(c, (tuple, list)):
            flat.extend(c)
        else:
            flat.append(c)
    if not should_use_color():
        return text
    return "".join(flat) + text + Colors.RESET


def semantic(text: str, role) -> str:
    """Color ``text`` using a semantic :class:`Palette` role.

    Thin wrapper over :func:`color` that reads at call sites as intent::

        semantic("✓ done", Palette.SUCCESS)
        semantic("Section", Palette.HEADING)
    """
    return color(text, role)

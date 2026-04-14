"""Shared ANSI color utilities for Hermes CLI modules."""

import os
import sys
from typing import Optional


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


# Per issue #8526: on light-background terminals (e.g. the default macOS
# Terminal.app profile), yellow text renders near-invisible against white
# and dim text loses nearly all contrast. In "light" mode we swap those
# two codes for high-contrast alternatives. Red/green/cyan/blue already
# have acceptable contrast on both backgrounds and are left alone so
# existing semantic meaning (error=red, ok=green, info=cyan) is preserved.
_LIGHT_REMAP = {
    Colors.YELLOW: Colors.MAGENTA,
    # Dropping DIM means the text renders in the terminal's default
    # foreground, which on a light background is a dark color and is
    # comfortably readable.
    Colors.DIM: "",
}


def _parse_colorfgbg_bg(value: str) -> Optional[int]:
    """Return the background color index from a COLORFGBG value, or None.

    COLORFGBG is conventionally ``<fg>;<bg>`` (sometimes with a middle
    segment). Only the trailing segment is the background. Non-integer
    values (e.g. ``default``) yield None so callers can fall through
    rather than guessing.
    """
    if not value:
        return None
    parts = value.split(";")
    if not parts:
        return None
    bg = parts[-1].strip()
    try:
        return int(bg)
    except ValueError:
        return None


def _detect_theme_from_colorfgbg() -> Optional[str]:
    """Best-effort background-color detection from the COLORFGBG env var.

    Returns ``"light"`` for indices commonly used for white / light-grey
    backgrounds (7, 15), ``"dark"`` for clearly dark indices (0-6, 8),
    and ``None`` otherwise so we fall back to the default behavior.
    """
    bg = _parse_colorfgbg_bg(os.environ.get("COLORFGBG", ""))
    if bg is None:
        return None
    if bg in (7, 15):
        return "light"
    if 0 <= bg <= 6 or bg == 8:
        return "dark"
    return None


def _resolve_theme() -> Optional[str]:
    """Return ``"light"``, ``"dark"``, or ``None`` (unknown / default).

    Priority:

    1. ``HERMES_THEME`` env var — ``light`` / ``dark`` are explicit
       overrides; ``auto`` (or unset) falls through to step 2.
    2. ``COLORFGBG``-based auto-detection.

    Anything else (including unrecognized ``HERMES_THEME`` values) returns
    ``None`` so callers keep their current dark-optimized defaults.
    """
    explicit = (os.environ.get("HERMES_THEME") or "").strip().lower()
    if explicit in ("light", "dark"):
        return explicit
    if explicit in ("", "auto"):
        return _detect_theme_from_colorfgbg()
    return None


def color(text: str, *codes) -> str:
    """Apply color codes to text (only when color output is appropriate)."""
    if not should_use_color():
        return text
    if _resolve_theme() == "light":
        codes = tuple(_LIGHT_REMAP.get(c, c) for c in codes)
    return "".join(codes) + text + Colors.RESET

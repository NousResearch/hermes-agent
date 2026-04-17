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


def is_light_background() -> bool:
    """Detect light terminal backgrounds via COLORFGBG or common indicators.

    COLORFGBG is set by some terminals (konsole, yakuake, some xterm configs)
    as ``<fg>;<bg>`` where 15=white, 0=black, 7=light gray, etc.

    Returns True when the terminal likely has a light background, so callers
    can switch to dark-friendly colors (e.g. brown instead of bright yellow).
    (#11300)
    """
    colorfgbg = os.environ.get("COLORFGBG", "")
    if ";" in colorfgbg:
        try:
            _, bg = colorfgbg.rsplit(";", 1)
            bg = int(bg)
            # Light backgrounds: white(15), light gray(7), yellow(11), etc.
            if bg in (7, 10, 11, 12, 13, 14, 15):
                return True
            # Dark backgrounds: black(0), dark blue(1), dark green(2), etc.
            if bg in (0, 1, 2, 3, 4, 5, 6, 8, 9):
                return False
        except (ValueError, IndexError):
            pass
    # Check for common light-theme env vars
    if os.environ.get("TERM_BACKGROUND") == "light":
        return True
    return False


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
    # Bright variants (readable on dark backgrounds)
    BRIGHT_YELLOW = "\033[93m"
    # Dark variants (readable on light backgrounds)
    DARK_YELLOW = "\033[33m"
    BROWN = "\033[38;5;130m"  # Dark orange/brown


def color(text: str, *codes) -> str:
    """Apply color codes to text (only when color output is appropriate)."""
    if not should_use_color():
        return text
    return "".join(codes) + text + Colors.RESET

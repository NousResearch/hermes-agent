"""Shared ANSI color utilities for Hermes CLI modules."""

import sys


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


def color(text: str, *codes) -> str:
    """Apply color codes to text (only when output is a TTY)."""
    if not sys.stdout.isatty():
        return text
    return "".join(codes) + text + Colors.RESET

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

    @classmethod
    def set_skin(cls, skin_dict: dict) -> None:
        """Update color codes to 24-bit ANSI from a skin theme dict."""
        def _to_ansi(key: str, fallback: str) -> str:
            val = skin_dict.get(key, "")
            h = val.split()[0].lstrip("#")
            if len(h) == 6:
                try:
                    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
                    return f"\033[38;2;{r};{g};{b}m"
                except ValueError:
                    pass
            return fallback
        cls.CYAN    = _to_ansi("ui-label",  "\033[36m")
        cls.GREEN   = _to_ansi("ui-ok",     "\033[32m")
        cls.YELLOW  = _to_ansi("ui-warn",   "\033[33m")
        cls.RED     = _to_ansi("ui-error",  "\033[31m")


def color(text: str, *codes) -> str:
    """Apply color codes to text (only when output is a TTY)."""
    if not sys.stdout.isatty():
        return text
    return "".join(codes) + text + Colors.RESET

"""Shared ANSI color utilities for Hermes CLI modules.

Provides adaptive color support that automatically detects light terminal
backgrounds and adjusts colors for readability.  On light backgrounds,
standard ANSI colors (like yellow, green, cyan) are remapped to darker
variants that provide adequate contrast.
"""

import logging
import os
import sys

logger = logging.getLogger(__name__)

# ─── Terminal background detection ───────────────────────────────────────────

# Cached result of background detection (None = not yet detected).
_is_light_background: bool | None = None


def _parse_osc11_response(data: str) -> bool | None:
    """Parse OSC 11 response like '\\033]11;rgb:ffff/dddd/aaaa\\033\\\\'
    and return True if the background is perceived as light."""
    import re
    m = re.search(r'11;rgb:([0-9a-fA-F]{2,4})/([0-9a-fA-F]{2,4})/([0-9a-fA-F]{2,4})', data)
    if not m:
        return None
    try:
        r = int(m.group(1)[:2], 16)
        g = int(m.group(2)[:2], 16)
        b = int(m.group(3)[:2], 16)
    except ValueError:
        return None
    # Rec. 709 luminance
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return luminance > 186


def _detect_light_background() -> bool:
    """Detect whether the terminal has a light background.

    Strategies (in order):
    1. HERMES_LIGHT_BACKGROUND env var override
    2. COLORFGBG env var (common on KDE/GNOME; format: fg;bg)
    3. OSC 11 query (works on xterm-compatible terminals, macOS Terminal, iTerm2)
    4. Fallback to dark (False)
    """
    # 1. Explicit env var
    env_val = os.environ.get("HERMES_LIGHT_BACKGROUND", "").strip().lower()
    if env_val in ("1", "true", "yes"):
        return True
    if env_val in ("0", "false", "no"):
        return False

    # 2. COLORFGBG
    colorfgbg = os.environ.get("COLORFGBG", "")
    if ";" in colorfgbg:
        try:
            bg_str = colorfgbg.rsplit(";", 1)[-1].strip()
            bg = int(bg_str)
            # Common light background color indices
            if bg in (7, 15, 231):
                return True
            if bg > 231:
                return True  # grayscale range 232-255
        except (ValueError, IndexError):
            pass

    # 3. OSC 11 query
    if sys.stdout.isatty() and sys.stdin.isatty():
        try:
            import select
            import termios
            fd = sys.stdin.fileno()
            old = termios.tcgetattr(fd)
            new = termios.tcgetattr(fd)
            new[3] = new[3] & ~termios.ECHO & ~termios.ICANON
            new[6][termios.VMIN] = 0
            new[6][termios.VTIME] = 1  # 100ms timeout
            termios.tcsetattr(fd, termios.TCSANOW, new)
            try:
                sys.stdout.write("\033]11;?\033\\")
                sys.stdout.flush()
                data = b""
                for _ in range(10):
                    r, _, _ = select.select([sys.stdin], [], [], 0.05)
                    if r:
                        chunk = sys.stdin.buffer.read(64)
                        if chunk:
                            data += chunk
                        else:
                            break
                    else:
                        break
                if data:
                    result = _parse_osc11_response(data.decode("utf-8", errors="replace"))
                    if result is not None:
                        return result
            finally:
                termios.tcsetattr(fd, termios.TCSANOW, old)
        except Exception:
            pass

    return False


def is_light_background() -> bool:
    """Return True if the terminal background is light.  Result is cached."""
    global _is_light_background
    if _is_light_background is None:
        _is_light_background = _detect_light_background()
    return _is_light_background


def set_light_background(light: bool) -> None:
    """Explicitly set the light/dark background mode (bypasses detection)."""
    global _is_light_background
    _is_light_background = bool(light)


def reset_background_cache() -> None:
    """Clear the cached background detection so it re-runs on next call."""
    global _is_light_background
    _is_light_background = None


# ─── Color output gating ─────────────────────────────────────────────────────

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


# ─── Adaptive ANSI colors ────────────────────────────────────────────────────

# Dark-background (default) ANSI codes
_DARK_COLORS = {
    "RED":     "\033[31m",
    "GREEN":   "\033[32m",
    "YELLOW":  "\033[33m",
    "BLUE":    "\033[34m",
    "MAGENTA": "\033[35m",
    "CYAN":    "\033[36m",
}

# Light-background alternatives (darker shades for contrast on white/light bg)
_LIGHT_COLORS = {
    "RED":     "\033[38;5;124m",   # dark red
    "GREEN":   "\033[38;5;28m",    # dark green
    "YELLOW":  "\033[38;5;136m",   # dark yellow/ochre
    "BLUE":    "\033[38;5;25m",    # dark blue
    "MAGENTA": "\033[38;5;90m",    # dark magenta
    "CYAN":    "\033[38;5;23m",    # dark teal
}


class _ColorResolver:
    """Dynamically resolves ANSI color codes based on terminal background.

    Usage is identical to the old static ``Colors`` class::

        from hermes_cli.colors import Colors, color
        print(color("done", Colors.GREEN, Colors.BOLD))
    """

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    def __getattr__(self, name: str) -> str:
        if name in _DARK_COLORS:
            if is_light_background():
                return _LIGHT_COLORS[name]
            return _DARK_COLORS[name]
        raise AttributeError(f"'Colors' object has no attribute {name!r}")


Colors: _ColorResolver = _ColorResolver()


def color(text: str, *codes) -> str:
    """Apply color codes to text (only when color output is appropriate)."""
    if not should_use_color():
        return text
    return "".join(codes) + text + Colors.RESET

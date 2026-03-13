"""Shared ANSI color utilities for Hermes CLI modules."""

import os
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


def detect_terminal_background() -> str:
    """Detect whether the terminal has a light or dark background.

    Uses the OSC 11 escape sequence to query the terminal's background color.
    Returns "light", "dark", or "unknown" if detection fails.
    """
    if sys.platform == "win32":
        return "unknown"

    if not sys.stdin.isatty() or not sys.stdout.isatty():
        return "unknown"

    # Some terminals/environments don't support OSC queries
    term = os.environ.get("TERM", "")
    if term == "dumb" or os.environ.get("NO_COLOR"):
        return "unknown"

    import select
    import termios
    import tty

    fd = sys.stdin.fileno()
    try:
        old_attrs = termios.tcgetattr(fd)
    except termios.error:
        return "unknown"

    try:
        tty.setraw(fd)
        # Send OSC 11 query: "what is the background color?"
        sys.stdout.write("\033]11;?\033\\")
        sys.stdout.flush()

        # Wait for response with a short timeout
        if not select.select([sys.stdin], [], [], 0.15)[0]:
            return "unknown"

        # Read response: expected format is \033]11;rgb:RRRR/GGGG/BBBB\033\\
        response = ""
        while select.select([sys.stdin], [], [], 0.05)[0]:
            ch = sys.stdin.read(1)
            response += ch
            if ch == "\\" or len(response) > 64:
                break

        # Parse the response
        # Format: \033]11;rgb:RRRR/GGGG/BBBB\033\\ or \033]11;rgb:RR/GG/BB\a
        if "rgb:" in response:
            rgb_part = response.split("rgb:")[1].split("\033")[0].split("\a")[0]
            components = rgb_part.split("/")
            if len(components) == 3:
                # Take the first 2 hex digits of each component (they may be 2 or 4 digits)
                r = int(components[0][:2], 16)
                g = int(components[1][:2], 16)
                b = int(components[2][:2], 16)
                # Perceived luminance (ITU-R BT.601)
                luminance = 0.299 * r + 0.587 * g + 0.114 * b
                return "light" if luminance > 128 else "dark"

        return "unknown"
    except Exception:
        return "unknown"
    finally:
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)
        except Exception:
            pass

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


def _detect_via_colorfgbg() -> str:
    """Check the COLORFGBG env var (set by some terminals like rxvt, iTerm2)."""
    val = os.environ.get("COLORFGBG", "")
    if ";" in val:
        # Format: "fg;bg" where bg is a color index (0-15)
        # 0-6 are dark colors, 7-15 are light colors
        try:
            bg = int(val.rsplit(";", 1)[1])
            return "light" if bg >= 7 else "dark"
        except (ValueError, IndexError):
            pass
    return "unknown"


def _detect_via_macos_appearance() -> str:
    """Check macOS system appearance (Dark Mode vs Light Mode)."""
    if sys.platform != "darwin":
        return "unknown"
    import subprocess
    try:
        result = subprocess.run(
            ["defaults", "read", "-g", "AppleInterfaceStyle"],
            capture_output=True, text=True, timeout=2,
        )
        # If the key exists and returns "Dark", system is in dark mode.
        # If the command fails (exit code 1), the key doesn't exist = light mode.
        if result.returncode == 0 and "dark" in result.stdout.strip().lower():
            return "dark"
        return "light"
    except Exception:
        return "unknown"


def _detect_via_osc11() -> str:
    """Query the terminal background color via OSC 11 escape sequence."""
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        return "unknown"

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
        sys.stdout.write("\033]11;?\033\\")
        sys.stdout.flush()

        if not select.select([sys.stdin], [], [], 0.15)[0]:
            return "unknown"

        response = ""
        while select.select([sys.stdin], [], [], 0.05)[0]:
            ch = sys.stdin.read(1)
            response += ch
            if ch == "\\" or len(response) > 64:
                break

        if "rgb:" in response:
            rgb_part = response.split("rgb:")[1].split("\033")[0].split("\a")[0]
            components = rgb_part.split("/")
            if len(components) == 3:
                r = int(components[0][:2], 16)
                g = int(components[1][:2], 16)
                b = int(components[2][:2], 16)
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


def detect_terminal_background() -> str:
    """Detect whether the terminal has a light or dark background.

    Tries multiple detection methods in order:
    1. COLORFGBG env var (fast, set by some terminals)
    2. OSC 11 escape sequence (most accurate, but not all terminals respond)
    3. macOS system appearance (AppleInterfaceStyle)
    Returns "light", "dark", or "unknown" if all methods fail.
    """
    if sys.platform == "win32":
        return "unknown"

    # Method 1: COLORFGBG env var
    result = _detect_via_colorfgbg()
    if result != "unknown":
        return result

    # Method 2: OSC 11 terminal query
    result = _detect_via_osc11()
    if result != "unknown":
        return result

    # Method 3: macOS system appearance
    result = _detect_via_macos_appearance()
    if result != "unknown":
        return result

    return "unknown"

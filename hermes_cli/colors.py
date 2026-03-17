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


# =============================================================================
# Terminal background detection (light vs dark)
# =============================================================================


def _detect_via_colorfgbg() -> str:
    """Check the COLORFGBG environment variable.

    Some terminals (rxvt, xterm, iTerm2) set COLORFGBG to ``<fg>;<bg>``
    where bg >= 8 usually means a dark background.
    Returns "light", "dark", or "unknown".
    """
    val = os.environ.get("COLORFGBG", "")
    if not val:
        return "unknown"
    parts = val.split(";")
    try:
        bg = int(parts[-1])
    except (ValueError, IndexError):
        return "unknown"
    # Standard terminal colors 0-6 are dark, 7+ are light.
    # bg < 7 → dark background; bg >= 7 → light background.
    if bg >= 7:
        return "light"
    return "dark"


def _detect_via_macos_appearance() -> str:
    """Check macOS AppleInterfaceStyle via ``defaults read``.

    Returns "light", "dark", or "unknown".
    """
    if sys.platform != "darwin":
        return "unknown"
    try:
        import subprocess
        result = subprocess.run(
            ["defaults", "read", "-g", "AppleInterfaceStyle"],
            capture_output=True, text=True, timeout=2,
        )
        if result.returncode == 0 and "dark" in result.stdout.lower():
            return "dark"
        # If the key doesn't exist, macOS is in light mode.
        return "light"
    except Exception:
        return "unknown"


def _detect_via_osc11() -> str:
    """Query the terminal background colour via the OSC 11 escape sequence.

    Writes ``\\e]11;?\\a`` and reads the response to determine luminance.
    Only works when stdin/stdout are connected to a real TTY (not piped).
    Returns "light", "dark", or "unknown".
    """
    if sys.platform == "win32":
        return "unknown"
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return "unknown"
    try:
        import select
        import termios
        import tty

        fd = sys.stdin.fileno()
        old_attrs = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            # Send OSC 11 query
            sys.stdout.write("\x1b]11;?\x07")
            sys.stdout.flush()
            # Wait briefly for response
            if not select.select([fd], [], [], 0.1)[0]:
                return "unknown"
            response = b""
            while select.select([fd], [], [], 0.05)[0]:
                response += os.read(fd, 128)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)

        # Parse response: \x1b]11;rgb:RRRR/GGGG/BBBB\x07  (or \x1b\\)
        text = response.decode("latin-1", errors="replace")
        if "rgb:" not in text:
            return "unknown"
        rgb_part = text.split("rgb:")[-1].split("\x07")[0].split("\x1b")[0]
        channels = rgb_part.split("/")
        if len(channels) < 3:
            return "unknown"
        # Each channel is 2 or 4 hex digits; normalise to 0-255
        vals = []
        for ch in channels[:3]:
            ch = ch.strip()
            if len(ch) <= 2:
                vals.append(int(ch, 16))
            else:
                vals.append(int(ch[:2], 16))  # take high byte
        # Perceived luminance (ITU-R BT.601)
        luminance = 0.299 * vals[0] + 0.587 * vals[1] + 0.114 * vals[2]
        return "light" if luminance > 128 else "dark"
    except Exception:
        return "unknown"


def detect_terminal_background() -> str:
    """Detect whether the terminal has a light or dark background.

    Tries three strategies in order:
    1. COLORFGBG environment variable
    2. macOS appearance setting
    3. OSC 11 escape sequence query

    Returns "light", "dark", or "unknown" if detection fails.
    """
    for detector in (_detect_via_colorfgbg, _detect_via_macos_appearance, _detect_via_osc11):
        result = detector()
        if result != "unknown":
            return result
    return "unknown"

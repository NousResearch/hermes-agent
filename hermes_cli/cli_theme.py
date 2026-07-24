"""CLI color, ANSI escape, and light/dark terminal mode detection.

Extracted from ``cli.py`` to keep that god-file smaller and make theme
utilities independently testable.  ``cli.py`` imports everything from here
so all existing call-sites keep working unchanged.
"""

from __future__ import annotations

import os
import re
import sys
import time
from typing import Dict

# ── ANSI building blocks ─────────────────────────────────────────────
_ACCENT_ANSI_DEFAULT = "\033[1;38;2;255;215;0m"  # True-color #FFD700 bold — fallback
_BOLD = "\033[1m"
_RST = "\033[0m"
_STREAM_PAD = "    "  # 4-space indent for streamed response text (matches Panel padding)


# ── Hex ↔ ANSI ───────────────────────────────────────────────────────

def _hex_to_ansi(hex_color: str, *, bold: bool = False) -> str:
    """Convert a hex color like '#268bd2' to a true-color ANSI escape.

    Auto-remaps known dark-mode-tuned colors to readable light-mode
    equivalents when running on a light terminal (see
    _maybe_remap_for_light_mode + _LIGHT_MODE_REMAP).
    """
    hex_color = _maybe_remap_for_light_mode(hex_color)
    try:
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        prefix = "1;" if bold else ""
        return f"\033[{prefix}38;2;{r};{g};{b}m"
    except (ValueError, IndexError):
        return _ACCENT_ANSI_DEFAULT if bold else "\033[38;2;184;134;11m"


# ── Luminance ────────────────────────────────────────────────────────

def _luminance_from_hex(hex_str: str) -> float | None:
    s = (hex_str or "").strip().lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    if len(s) != 6 or not all(c in "0123456789abcdefABCDEF" for c in s):
        return None
    try:
        r, g, b = int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
    except ValueError:
        return None
    # Rec.709 luma
    return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0


# ── Light/dark terminal mode detection ───────────────────────────────
#
# Mirrors ui-tui/src/theme.ts detectLightMode().  Used to decide whether
# to remap "near-white" skin colors (e.g. #FFF8DC banner_text, #B8860B
# banner_dim) to darker equivalents that are readable on a light
# Terminal.app / iTerm2 background.
#
# Detection priority:
#   1. HERMES_LIGHT / HERMES_TUI_LIGHT env (true/false) — explicit override
#   2. HERMES_TUI_THEME=light|dark — explicit theme
#   3. HERMES_TUI_BACKGROUND=#RRGGBB — explicit bg hint
#   4. COLORFGBG env (set by xterm/Konsole/urxvt) — bg slot 7/15 = light
#   5. OSC 11 query (\x1b]11;?\x1b\\) — ask the terminal directly
#   6. Default: assume dark (matches the legacy Hermes assumption)
#
# Cached after first call so we don't query the terminal repeatedly.
_LIGHT_MODE_CACHE: bool | None = None
_TRUE_RE = re.compile(r"^(1|true|on|yes|y)$")
_FALSE_RE = re.compile(r"^(0|false|off|no|n)$")
_LIGHT_DEFAULT_TERM_PROGRAMS = frozenset()  # Apple_Terminal doesn't reliably indicate; require explicit


def _query_osc11_background() -> str | None:
    """Ask the terminal for its background color via OSC 11.

    Most modern terminals reply with \\x1b]11;rgb:RRRR/GGGG/BBBB\\x1b\\\\
    within a few ms.  We wait up to 100ms total before giving up.
    Returns "#RRGGBB" or None on timeout / non-tty.

    Skipped over SSH: the round-trip routinely exceeds our 100ms budget, so a
    late reply lands after prompt_toolkit has grabbed the tty — its payload
    leaks in as typed text and the BEL terminator reads as Ctrl+G (open
    editor), trapping the user in a stray editor. Remote sessions fall back to
    COLORFGBG / env hints / the dark default instead.
    """
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        return None
    if any(os.environ.get(v) for v in ("SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY")):
        return None
    try:
        import termios
        import tty
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
    except Exception:
        return None
    try:
        try:
            tty.setcbreak(fd)
        except Exception:
            return None
        try:
            sys.stdout.write("\x1b]11;?\x1b\\")
            sys.stdout.flush()
        except Exception:
            return None
        # Read up to ~50ms for the response
        import select
        deadline = time.monotonic() + 0.1
        buf = b""
        while time.monotonic() < deadline:
            r, _, _ = select.select([fd], [], [], deadline - time.monotonic())
            if not r:
                continue
            try:
                chunk = os.read(fd, 64)
            except OSError:
                break
            if not chunk:
                break
            buf += chunk
            if b"\x1b\\" in buf or b"\x07" in buf:
                break
        # Parse: \x1b]11;rgb:RRRR/GGGG/BBBB\x1b\\
        m = re.search(rb"rgb:([0-9a-fA-F]+)/([0-9a-fA-F]+)/([0-9a-fA-F]+)", buf)
        if not m:
            return None
        # Each component is 1-4 hex digits — normalize to 8-bit
        def norm(h: bytes) -> int:
            v = int(h, 16)
            # Scale to 0-255 based on hex length
            bits = len(h) * 4
            return (v * 255) // ((1 << bits) - 1) if bits else 0
        r, g, b = norm(m.group(1)), norm(m.group(2)), norm(m.group(3))
        return f"#{r:02X}{g:02X}{b:02X}"
    finally:
        # TCSAFLUSH discards any unread input as it restores the original
        # attributes — scrubs a slow/partial OSC 11 reply out of the tty
        # buffer before prompt_toolkit can read it as keystrokes.
        try:
            termios.tcsetattr(fd, termios.TCSAFLUSH, old)
        except Exception:
            pass


def _detect_light_mode() -> bool:
    global _LIGHT_MODE_CACHE
    if _LIGHT_MODE_CACHE is not None:
        return _LIGHT_MODE_CACHE
    result = False
    try:
        # 1. Explicit env override
        for var in ("HERMES_LIGHT", "HERMES_TUI_LIGHT"):
            v = (os.environ.get(var) or "").strip().lower()
            if _TRUE_RE.match(v):
                result = True
                _LIGHT_MODE_CACHE = result
                return result
            if _FALSE_RE.match(v):
                _LIGHT_MODE_CACHE = result
                return result
        # 2. Theme hint
        theme = (os.environ.get("HERMES_TUI_THEME") or "").strip().lower()
        if theme == "light":
            result = True
            _LIGHT_MODE_CACHE = result
            return result
        if theme == "dark":
            _LIGHT_MODE_CACHE = result
            return result
        # 3. Explicit bg hex
        bg_hint = os.environ.get("HERMES_TUI_BACKGROUND") or ""
        bg_lum = _luminance_from_hex(bg_hint)
        if bg_lum is not None:
            result = bg_lum >= 0.5
            _LIGHT_MODE_CACHE = result
            return result
        # 4. COLORFGBG (xterm/Konsole/urxvt)
        cfgbg = (os.environ.get("COLORFGBG") or "").strip()
        if cfgbg:
            last = cfgbg.split(";")[-1] if ";" in cfgbg else cfgbg
            if last.isdigit():
                bg = int(last)
                if bg in {7, 15}:
                    result = True
                    _LIGHT_MODE_CACHE = result
                    return result
                if 0 <= bg < 16:
                    _LIGHT_MODE_CACHE = result
                    return result
        # 5. OSC 11 query (best-effort, only when stdin/stdout are TTY)
        bg_color = _query_osc11_background()
        if bg_color:
            lum = _luminance_from_hex(bg_color)
            if lum is not None:
                result = lum >= 0.5
                _LIGHT_MODE_CACHE = result
                return result
        # 6. TERM_PROGRAM allow-list (currently empty)
        tp = (os.environ.get("TERM_PROGRAM") or "").strip()
        if tp in _LIGHT_DEFAULT_TERM_PROGRAMS:
            result = True
    except Exception:
        result = False
    _LIGHT_MODE_CACHE = result
    return result


# ── Light-mode color remapping ───────────────────────────────────────
#
# Light-mode equivalents of skin colors that are unreadable on cream
# Terminal.app backgrounds.  Used by _SkinAwareAnsi to remap colors
# at resolution time when light mode is detected.
#
# IMPORTANT: only remap colors that are used as STANDALONE foregrounds
# on the terminal's background.  Don't remap colors that are paired
# with a dark bg (e.g. status bar text on bg:#1a1a2e) — those would
# become invisible the OTHER direction (dark gray on dark navy).
_LIGHT_MODE_REMAP: dict[str, str] = {
    # Original (dark-mode) -> Light-mode replacement (darker, readable)
    "#FFF8DC": "#1A1A1A",   # cornsilk -> near-black
    "#FFD700": "#9A6B00",   # gold -> dark goldenrod (readable on cream)
    "#FFBF00": "#8A5A00",   # amber -> dark amber
    "#B8860B": "#5C4500",   # dark goldenrod -> deeper brown (more contrast)
    "#DAA520": "#6B4F00",   # goldenrod -> dark olive
    "#F1E6CF": "#1A1A1A",   # cream -> near-black
    "#c9d1d9": "#24292F",   # github-light fg
    "#EAF7FF": "#0F1B26",   # ice
    "#F5F5F5": "#1A1A1A",
    "#FFF0D4": "#1A1A1A",
    "#CD7F32": "#8A4F1A",   # bronze -> darker bronze
    "#FFEFB5": "#3A2A00",
    # NOTE: skipping #C0C0C0/#888888/#555555/#8B8682 — those are
    # status-bar foregrounds paired with dark navy bg, where dark
    # remap values would become invisible.
}

# Pre-uppercased lookup table for case-insensitive remapping
_LIGHT_MODE_REMAP_UPPER = {k.upper(): v for k, v in _LIGHT_MODE_REMAP.items()}


def _maybe_remap_for_light_mode(hex_color: str) -> str:
    """If we're in light mode, remap a dark-mode-tuned color to a
    higher-contrast equivalent.  No-op in dark mode."""
    if not _detect_light_mode():
        return hex_color
    if not hex_color or not hex_color.startswith("#"):
        return hex_color
    # Case-insensitive lookup
    upper = hex_color.upper()
    if upper in _LIGHT_MODE_REMAP_UPPER:
        return _LIGHT_MODE_REMAP_UPPER[upper]
    return hex_color


# ── Skin-aware ANSI wrapper ──────────────────────────────────────────

class _SkinAwareAnsi:
    """Lazy ANSI escape that resolves from the skin engine on first use.

    Acts as a string in f-strings and concatenation.  Call ``.reset()`` to
    force re-resolution after a ``/skin`` switch.
    """

    def __init__(self, skin_key: str, fallback_hex: str = "#FFD700", *, bold: bool = False):
        self._skin_key = skin_key
        self._fallback_hex = fallback_hex
        self._bold = bold
        self._cached: str | None = None

    def __str__(self) -> str:
        if self._cached is None:
            try:
                from hermes_cli.skin_engine import get_active_skin
                self._cached = _hex_to_ansi(
                    get_active_skin().get_color(self._skin_key, self._fallback_hex),
                    bold=self._bold,
                )
            except Exception:
                self._cached = _hex_to_ansi(self._fallback_hex, bold=self._bold)
        return self._cached

    def __add__(self, other: str) -> str:
        return str(self) + other

    def __radd__(self, other: str) -> str:
        return other + str(self)

    def reset(self) -> None:
        """Clear the cached ANSI escape so the next ``str()`` re-resolves."""
        self._cached = None

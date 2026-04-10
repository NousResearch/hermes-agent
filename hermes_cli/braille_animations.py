"""Braille-based terminal animations — pure frame data, zero dependencies.

Ported from the unicode-animations npm package (by Gunnar Gray, Perplexity).
Each animation is a dict with:
  - "frames": tuple[str, ...]  — each frame is a string of braille characters
  - "interval_ms": int         — recommended millisecond interval between frames

All animations use Unicode braille characters (U+2800..U+28FF) so they
render correctly in any modern terminal.

Usage:
    from hermes_cli.braille_animations import get_frame, get_animation_names

    # Time-based frame selection (for render loops):
    frame = get_frame("breathe", elapsed_ms=1200)

    # Direct access:
    anim = ANIMATIONS["dna"]
    frames, interval = anim["frames"], anim["interval_ms"]
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

# ── Braille grid conversion ─────────────────────────────────────────
#
# Each braille character is a 2-column × 4-row dot grid.
# Base codepoint U+2800, with dots mapped via:
#
#   col 0   col 1
#   ─────   ─────
#   0x01    0x08    row 0
#   0x02    0x10    row 1
#   0x04    0x20    row 2
#   0x40    0x80    row 3

BRAILLE_BASE = 0x2800

BRAILLE_DOT_MAP = [
    [0x01, 0x08],
    [0x02, 0x10],
    [0x04, 0x20],
    [0x40, 0x80],
]


def grid_to_braille(grid: List[List[bool]]) -> str:
    """Convert a 2D boolean grid to a string of braille characters.

    Each braille character encodes a 4-row × 2-column region of the grid.
    The grid can have arbitrary dimensions; partial cells are handled.
    """
    rows = len(grid)
    if rows == 0:
        return ""
    cols = len(grid[0])
    char_count = math.ceil(cols / 2)
    result = []
    for c in range(char_count):
        code = BRAILLE_BASE
        for r in range(min(4, rows)):
            for d in range(2):
                col = c * 2 + d
                if col < cols and grid[r][col]:
                    code |= BRAILLE_DOT_MAP[r][d]
        result.append(chr(code))
    return "".join(result)


def make_grid(rows: int, cols: int) -> List[List[bool]]:
    """Create a rows×cols boolean grid initialized to False."""
    return [[False] * cols for _ in range(rows)]


# ── Animation definitions ────────────────────────────────────────────
#
# Frames are stored as tuples (immutable, zero-copy on repeated access).
# Three animations have hardcoded frames; the remaining 15 use frames
# extracted from the upstream snapshot tests (ground truth).

ANIMATIONS: Dict[str, dict] = {
    # ── Classic braille ──────────────────────────────────────────────
    "braille": {
        "frames": (
            "⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏",
        ),
        "interval_ms": 80,
    },
    "braillewave": {
        "frames": (
            "⠁⠂⠄⡀", "⠂⠄⡀⢀", "⠄⡀⢀⠠", "⡀⢀⠠⠐",
            "⢀⠠⠐⠈", "⠠⠐⠈⠁", "⠐⠈⠁⠂", "⠈⠁⠂⠄",
        ),
        "interval_ms": 100,
    },
    "dna": {
        "frames": (
            "⠋⠉⠙⠚", "⠉⠙⠚⠒", "⠙⠚⠒⠂", "⠚⠒⠂⠂",
            "⠒⠂⠂⠒", "⠂⠂⠒⠲", "⠂⠒⠲⠴", "⠒⠲⠴⠤",
            "⠲⠴⠤⠄", "⠴⠤⠄⠋", "⠤⠄⠋⠉", "⠄⠋⠉⠙",
        ),
        "interval_ms": 80,
    },
    # ── Grid-based animations ────────────────────────────────────────
    "scan": {
        "frames": (
            "⠀⠀⠀⠀", "⡇⠀⠀⠀", "⣿⠀⠀⠀", "⢸⡇⠀⠀",
            "⠀⣿⠀⠀", "⠀⢸⡇⠀", "⠀⠀⣿⠀", "⠀⠀⢸⡇",
            "⠀⠀⠀⣿", "⠀⠀⠀⢸",
        ),
        "interval_ms": 70,
    },
    "rain": {
        "frames": (
            "⢁⠂⠔⠈", "⠂⠌⡠⠐", "⠄⡐⢀⠡", "⡈⠠⠀⢂",
            "⠐⢀⠁⠄", "⠠⠁⠊⡀", "⢁⠂⠔⠈", "⠂⠌⡠⠐",
            "⠄⡐⢀⠡", "⡈⠠⠀⢂", "⠐⢀⠁⠄", "⠠⠁⠊⡀",
        ),
        "interval_ms": 100,
    },
    "scanline": {
        "frames": (
            "⠉⠉⠉", "⠓⠓⠓", "⠦⠦⠦", "⣄⣄⣄", "⠦⠦⠦", "⠓⠓⠓",
        ),
        "interval_ms": 120,
    },
    "pulse": {
        "frames": (
            "⠀⠶⠀", "⠰⣿⠆", "⢾⣉⡷", "⣏⠀⣹", "⡁⠀⢈",
        ),
        "interval_ms": 180,
    },
    "snake": {
        "frames": (
            "⣁⡀", "⣉⠀", "⡉⠁", "⠉⠉", "⠈⠙", "⠀⠛",
            "⠐⠚", "⠒⠒", "⠖⠂", "⠶⠀", "⠦⠄", "⠤⠤",
            "⠠⢤", "⠀⣤", "⢀⣠", "⣀⣀",
        ),
        "interval_ms": 80,
    },
    "sparkle": {
        "frames": (
            "⡡⠊⢔⠡", "⠊⡰⡡⡘", "⢔⢅⠈⢢",
            "⡁⢂⠆⡍", "⢔⠨⢑⢐", "⠨⡑⡠⠊",
        ),
        "interval_ms": 150,
    },
    "cascade": {
        "frames": (
            "⠀⠀⠀⠀", "⠀⠀⠀⠀", "⠁⠀⠀⠀", "⠋⠀⠀⠀",
            "⠞⠁⠀⠀", "⡴⠋⠀⠀", "⣠⠞⠁⠀", "⢀⡴⠋⠀",
            "⠀⣠⠞⠁", "⠀⢀⡴⠋", "⠀⠀⣠⠞", "⠀⠀⢀⡴",
            "⠀⠀⠀⣠", "⠀⠀⠀⢀",
        ),
        "interval_ms": 60,
    },
    "columns": {
        "frames": (
            "⡀⠀⠀", "⡄⠀⠀", "⡆⠀⠀", "⡇⠀⠀",
            "⣇⠀⠀", "⣧⠀⠀", "⣷⠀⠀", "⣿⠀⠀",
            "⣿⡀⠀", "⣿⡄⠀", "⣿⡆⠀", "⣿⡇⠀",
            "⣿⣇⠀", "⣿⣧⠀", "⣿⣷⠀", "⣿⣿⠀",
            "⣿⣿⡀", "⣿⣿⡄", "⣿⣿⡆", "⣿⣿⡇",
            "⣿⣿⣇", "⣿⣿⣧", "⣿⣿⣷", "⣿⣿⣿",
            "⣿⣿⣿", "⠀⠀⠀",
        ),
        "interval_ms": 60,
    },
    "orbit": {
        "frames": (
            "⠃", "⠉", "⠘", "⠰", "⢠", "⣀", "⡄", "⠆",
        ),
        "interval_ms": 100,
    },
    "breathe": {
        "frames": (
            "⠀", "⠂", "⠌", "⡑", "⢕", "⢝", "⣫", "⣟", "⣿",
            "⣟", "⣫", "⢝", "⢕", "⡑", "⠌", "⠂", "⠀",
        ),
        "interval_ms": 100,
    },
    "waverows": {
        "frames": (
            "⠖⠉⠉⠑", "⡠⠖⠉⠉", "⣠⡠⠖⠉", "⣄⣠⡠⠖",
            "⠢⣄⣠⡠", "⠙⠢⣄⣠", "⠉⠙⠢⣄", "⠊⠉⠙⠢",
            "⠜⠊⠉⠙", "⡤⠜⠊⠉", "⣀⡤⠜⠊", "⢤⣀⡤⠜",
            "⠣⢤⣀⡤", "⠑⠣⢤⣀", "⠉⠑⠣⢤", "⠋⠉⠑⠣",
        ),
        "interval_ms": 90,
    },
    "checkerboard": {
        "frames": (
            "⢕⢕⢕", "⡪⡪⡪", "⢊⠔⡡", "⡡⢊⠔",
        ),
        "interval_ms": 250,
    },
    "helix": {
        "frames": (
            "⢌⣉⢎⣉", "⣉⡱⣉⡱", "⣉⢎⣉⢎", "⡱⣉⡱⣉",
            "⢎⣉⢎⣉", "⣉⡱⣉⡱", "⣉⢎⣉⢎", "⡱⣉⡱⣉",
            "⢎⣉⢎⣉", "⣉⡱⣉⡱", "⣉⢎⣉⢎", "⡱⣉⡱⣉",
            "⢎⣉⢎⣉", "⣉⡱⣉⡱", "⣉⢎⣉⢎", "⡱⣉⡱⣉",
        ),
        "interval_ms": 80,
    },
    "fillsweep": {
        "frames": (
            "⣀⣀", "⣤⣤", "⣶⣶", "⣿⣿", "⣿⣿",
            "⣿⣿", "⣶⣶", "⣤⣤", "⣀⣀", "⠀⠀", "⠀⠀",
        ),
        "interval_ms": 100,
    },
    "diagswipe": {
        "frames": (
            "⠁⠀", "⠋⠀", "⠟⠁", "⡿⠋", "⣿⠟", "⣿⡿",
            "⣿⣿", "⣿⣿", "⣾⣿", "⣴⣿", "⣠⣾", "⢀⣴",
            "⠀⣠", "⠀⢀", "⠀⠀", "⠀⠀",
        ),
        "interval_ms": 60,
    },
}


# ── Public API ───────────────────────────────────────────────────────


def get_animation_names() -> Tuple[str, ...]:
    """Return sorted tuple of available animation names."""
    return tuple(sorted(ANIMATIONS.keys()))


def get_animation(name: str) -> dict:
    """Get animation dict by name. Raises KeyError if not found."""
    return ANIMATIONS[name]


def get_frame(name: str, elapsed_ms: int) -> str:
    """Get the current frame for an animation based on elapsed time.

    Automatically wraps based on the animation's interval and frame count.
    """
    anim = ANIMATIONS[name]
    frames = anim["frames"]
    interval = anim["interval_ms"]
    idx = (elapsed_ms // interval) % len(frames)
    return frames[idx]

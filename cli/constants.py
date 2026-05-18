"""Shared constants for the Hermes classic CLI."""

_COMMAND_SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")

_REASONING_TAGS = (
    "REASONING_SCRATCHPAD",
    "think",
    "thinking",
    "reasoning",
    "thought",
)

_ACCENT_ANSI_DEFAULT = "\033[1;38;2;255;215;0m"
_BOLD = "\033[1m"
_RST = "\033[0m"
_STREAM_PAD = "    "

_IMAGE_EXTENSIONS = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".webp",
    ".bmp", ".tiff", ".tif", ".svg", ".ico",
})

_TERMINAL_INPUT_MODE_RESET_SEQ = (
    "\x1b[?1006l"
    "\x1b[?1003l"
    "\x1b[?1002l"
    "\x1b[?1000l"
    "\x1b[?1004l"
    "\x1b[?2004l"
    "\x1b[?1049l"
    "\x1b[<u"
    "\x1b[>4m"
    "\x1b[0m"
    "\x1b[?25h"
)

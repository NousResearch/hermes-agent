"""
Inline image rendering for terminals that support the Kitty or iTerm2 graphics protocol.

Supports: Kitty, Ghostty, WezTerm (kitty protocol), iTerm2 (iterm2 protocol).

Usage from CLI:
    from agent.inline_images import try_render_inline
    try_render_inline("/path/to/screenshot.png")  # renders if supported, no-op otherwise

Config: display.inline_images in config.yaml
    - "auto" (default): auto-detect terminal capability
    - true: force enabled (kitty protocol)
    - false: disabled
"""

import base64
import json
import logging
import os
import struct
import sys
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# ── Terminal capability detection ──────────────────────────────────────


def detect_image_protocol() -> str:
    """Detect which inline image protocol the terminal supports.

    Returns: 'kitty', 'iterm2', or 'none'
    """
    term = os.environ.get("TERM", "").lower()
    term_program = os.environ.get("TERM_PROGRAM", "").lower()
    kitty_window = os.environ.get("KITTY_WINDOW_ID", "")

    # Kitty protocol support
    if kitty_window:
        return "kitty"
    if term_program in ("ghostty", "kitty"):
        return "kitty"
    if "ghostty" in term or "kitty" in term:
        return "kitty"

    # iTerm2 protocol support (also works in WezTerm, Hyper, etc.)
    if term_program in ("iterm2", "iterm2.app", "wezterm", "hyper"):
        return "iterm2"
    if "iterm" in term_program:
        return "iterm2"

    # WezTerm also supports kitty protocol — check TERM
    if "wezterm" in term:
        return "kitty"

    return "none"


# ── PNG dimension reader (no PIL) ──────────────────────────────────────


def _get_png_dimensions(data: bytes) -> Tuple[int, int]:
    """Extract (width, height) from PNG header."""
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("Not a valid PNG file")
    return struct.unpack(">II", data[16:24])


def _get_image_dimensions(path: Path) -> Tuple[int, int]:
    """Get image dimensions. Handles PNG natively, tries PIL for others."""
    data = path.read_bytes()

    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return _get_png_dimensions(data)

    # Try PIL/Pillow for JPEG etc.
    try:
        from PIL import Image

        with Image.open(path) as img:
            return img.size
    except ImportError:
        pass

    # JPEG: try to parse SOF marker
    if data[:2] == b"\xff\xd8":
        return _get_jpeg_dimensions(data)

    return (800, 600)  # fallback estimate


def _get_jpeg_dimensions(data: bytes) -> Tuple[int, int]:
    """Parse JPEG SOF marker for dimensions."""
    i = 2
    while i < len(data) - 1:
        if data[i] != 0xFF:
            break
        marker = data[i + 1]
        if marker in (0xC0, 0xC1, 0xC2):  # SOF markers
            h = struct.unpack(">H", data[i + 5 : i + 7])[0]
            w = struct.unpack(">H", data[i + 7 : i + 9])[0]
            return (w, h)
        if marker == 0xD9:  # EOI
            break
        if marker in (0xD0, 0xD1, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0x01):
            i += 2
            continue
        length = struct.unpack(">H", data[i + 2 : i + 4])[0]
        i += 2 + length
    return (800, 600)


# ── Terminal size helpers ──────────────────────────────────────────────


def _get_terminal_size_pixels() -> Tuple[int, int, int, int]:
    """Return (cols, rows, pixel_width, pixel_height)."""
    try:
        cols, rows = os.get_terminal_size()
    except OSError:
        cols, rows = 80, 24

    try:
        import fcntl
        import termios

        result = fcntl.ioctl((getattr(sys, "__stdout__", None) or sys.stdout).fileno(), termios.TIOCGWINSZ, b"\x00" * 8)
        _, _, xpixel, ypixel = struct.unpack("HHHH", result)
        if xpixel > 0 and ypixel > 0:
            return cols, rows, xpixel, ypixel
    except Exception:
        pass

    # Estimate: typical cell ~8x16
    return cols, rows, cols * 8, rows * 16


def _calculate_display_cells(
    img_w: int, img_h: int, max_cols: Optional[int] = None, max_rows: Optional[int] = None
) -> Tuple[int, int]:
    """Calculate display size in terminal cells, preserving aspect ratio."""
    term_cols, term_rows, px_w, px_h = _get_terminal_size_pixels()

    if max_cols is None:
        max_cols = min(term_cols - 4, 120)
    if max_rows is None:
        max_rows = min(term_rows - 4, 30)

    cell_w = px_w / term_cols if term_cols > 0 else 8
    cell_h = px_h / term_rows if term_rows > 0 else 16

    img_cols = img_w / cell_w
    img_rows = img_h / cell_h

    scale = min(1.0, max_cols / max(img_cols, 0.1), max_rows / max(img_rows, 0.1))

    return max(1, int(img_cols * scale)), max(1, int(img_rows * scale))


# ── Kitty graphics protocol renderer ──────────────────────────────────


def _is_ssh_session() -> bool:
    """Detect if we're running inside an SSH session."""
    return bool(os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_TTY"))


def _render_kitty(image_path: str, max_cols: Optional[int] = None, max_rows: Optional[int] = None) -> str:
    """Render image using Kitty graphics protocol.

    Uses transmit-by-file-path (t=f) locally for speed, but falls back to
    direct base64 data transmission (t=d) over SSH since the local terminal
    can't access remote file paths.
    """
    path = Path(image_path).resolve()
    if not path.exists():
        return ""

    # Over SSH, the terminal is on the local machine and can't read remote
    # file paths.  Send the image data directly through the SSH pipe instead.
    if _is_ssh_session():
        data = path.read_bytes()
        return _render_kitty_from_data(data, max_cols, max_rows)

    try:
        img_w, img_h = _get_image_dimensions(path)
    except Exception:
        img_w, img_h = 800, 600

    display_cols, display_rows = _calculate_display_cells(img_w, img_h, max_cols, max_rows)
    path_b64 = base64.b64encode(str(path).encode()).decode()

    return f"\033_Ga=T,f=100,t=f,c={display_cols},r={display_rows};{path_b64}\033\\"


def _render_kitty_from_data(
    data: bytes, max_cols: Optional[int] = None, max_rows: Optional[int] = None
) -> str:
    """Render image from raw bytes via Kitty protocol (direct transmission)."""
    try:
        img_w, img_h = _get_png_dimensions(data)
    except ValueError:
        img_w, img_h = 800, 600

    display_cols, display_rows = _calculate_display_cells(img_w, img_h, max_cols, max_rows)
    b64_data = base64.b64encode(data).decode()

    chunks = [b64_data[i : i + 4096] for i in range(0, len(b64_data), 4096)]

    if len(chunks) == 1:
        return f"\033_Ga=T,f=100,t=d,c={display_cols},r={display_rows};{chunks[0]}\033\\"

    result = f"\033_Ga=T,f=100,t=d,c={display_cols},r={display_rows},m=1;{chunks[0]}\033\\"
    for chunk in chunks[1:-1]:
        result += f"\033_Gm=1;{chunk}\033\\"
    result += f"\033_Gm=0;{chunks[-1]}\033\\"
    return result


# ── iTerm2 inline image protocol renderer ─────────────────────────────


def _render_iterm2(image_path: str, max_cols: Optional[int] = None, max_rows: Optional[int] = None) -> str:
    """Render image using iTerm2 inline image protocol."""
    path = Path(image_path).resolve()
    if not path.exists():
        return ""

    data = path.read_bytes()
    b64_data = base64.b64encode(data).decode()

    try:
        img_w, img_h = _get_image_dimensions(path)
    except Exception:
        img_w, img_h = 800, 600

    display_cols, _ = _calculate_display_cells(img_w, img_h, max_cols, max_rows)
    name_b64 = base64.b64encode(path.name.encode()).decode()

    return (
        f"\033]1337;File=name={name_b64}"
        f";size={len(data)}"
        f";inline=1"
        f";width={display_cols}"
        f";preserveAspectRatio=1"
        f":{b64_data}\a"
    )


# ── Public API ─────────────────────────────────────────────────────────

# Cached protocol detection (avoid re-checking env vars every call)
_cached_protocol: Optional[str] = None


def _get_protocol(config_value: object = None) -> str:
    """Resolve the inline image protocol from config + environment.

    config_value: display.inline_images from config.yaml
        - "auto" or None → auto-detect
        - True / "true" → force kitty
        - False / "false" → disabled
        - "kitty" / "iterm2" → force specific protocol
    """
    global _cached_protocol

    if config_value is False or config_value == "false":
        return "none"

    if config_value is True or config_value == "true":
        if _cached_protocol is None:
            _cached_protocol = detect_image_protocol()
        return _cached_protocol if _cached_protocol != "none" else "kitty"

    if isinstance(config_value, str) and config_value in ("kitty", "iterm2"):
        return config_value

    # auto or None
    if _cached_protocol is None:
        _cached_protocol = detect_image_protocol()
    return _cached_protocol


def try_render_inline(
    image_path: str,
    max_cols: Optional[int] = None,
    max_rows: Optional[int] = None,
    config_value: object = "auto",
    label: Optional[str] = None,
    indent: str = "  ",
) -> bool:
    """Attempt to render an image inline in the terminal.

    Args:
        image_path: Path to a PNG/JPEG image file
        max_cols: Max display width in terminal columns (default: auto)
        max_rows: Max display height in terminal rows (default: auto)
        config_value: Value of display.inline_images from config
        label: Optional text label to show below the image
        indent: Indentation prefix for the image and label

    Returns:
        True if rendered inline, False if not supported or failed
    """
    protocol = _get_protocol(config_value)
    if protocol == "none":
        return False

    path = Path(image_path)
    if not path.exists():
        logger.debug("Inline image: file not found: %s", image_path)
        return False

    # Check file is a supported image
    suffix = path.suffix.lower()
    if suffix not in (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"):
        logger.debug("Inline image: unsupported format: %s", suffix)
        return False

    try:
        if protocol == "kitty":
            output = _render_kitty(str(path), max_cols, max_rows)
        elif protocol == "iterm2":
            output = _render_iterm2(str(path), max_cols, max_rows)
        else:
            return False

        if not output:
            return False

        # Write to the *real* stdout, bypassing prompt_toolkit's StdoutProxy.
        # prompt_toolkit's patch_stdout replaces sys.stdout with a proxy that
        # swallows raw escape sequences (it only understands ANSI color codes
        # routed through print_formatted_text).  Kitty/iTerm2 graphics protocol
        # uses APC/OSC sequences that the proxy silently discards.
        # sys.__stdout__ is the original file object saved by Python at startup.
        _real_out = getattr(sys, "__stdout__", None) or sys.stdout
        _real_out.write(f"{indent}{output}\n")
        if label:
            _real_out.write(f"{indent}{label}\n")
        _real_out.flush()
        return True

    except Exception as e:
        logger.debug("Inline image rendering failed: %s", e)
        return False


def extract_image_path_from_tool_result(function_name: str, result: str) -> Optional[str]:
    """Extract an image file path from a tool's JSON result string.

    Supports: browser_vision, image_generate, vision_analyze.
    Returns the path if found, None otherwise.
    """
    _IMAGE_TOOLS = {
        "browser_vision",
        "image_generate",
        "vision_analyze",
    }
    if function_name not in _IMAGE_TOOLS:
        return None

    try:
        data = json.loads(result)
    except (json.JSONDecodeError, TypeError):
        return None

    # browser_vision returns {"analysis": "...", "screenshot_path": "/path/..."}
    # image_generate returns {"image": "/path/..." or "https://..."}
    # vision_analyze returns {"analysis": "...", "screenshot_path": "..."}

    for key in ("screenshot_path", "image", "image_path"):
        val = data.get(key)
        if isinstance(val, str) and val and not val.startswith("http"):
            path = Path(val)
            if path.exists() and path.suffix.lower() in (".png", ".jpg", ".jpeg", ".gif", ".webp"):
                return str(path)

    return None

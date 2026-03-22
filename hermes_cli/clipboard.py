"""Clipboard image and text extraction for macOS, Windows, Linux, and WSL2.

Provides functions for extracting images and text from the system clipboard.
No external Python dependencies — uses only OS-level CLI tools that ship
with the platform (or are commonly installed).

Platform support:
  macOS   — osascript (always available), pngpaste (if installed), pbpaste (text)
  Windows — PowerShell with .NET System.Windows.Forms.Clipboard (native win32)
  WSL2    — powershell.exe via .NET System.Windows.Forms.Clipboard
  Linux   — wl-paste (Wayland), xclip (X11)
"""

import base64
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Cache WSL detection (checked once per process)
_wsl_detected: bool | None = None

# Cache powershell.exe path for WSL (resolved once)
_powershell_path: str | None = None


def save_clipboard_image(dest: Path) -> bool:
    """Extract an image from the system clipboard and save it as PNG.

    Returns True if an image was found and saved, False otherwise.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if sys.platform == "darwin":
        return _macos_save(dest)
    if sys.platform == "win32":
        return _windows_save(dest)
    return _linux_save(dest)


def has_clipboard_image() -> bool:
    """Quick check: does the clipboard currently contain an image?

    Lighter than save_clipboard_image — doesn't extract or write anything.
    """
    if sys.platform == "darwin":
        return _macos_has_image()
    if sys.platform == "win32":
        return _windows_has_image()
    if _is_wsl():
        return _wsl_has_image()
    if os.environ.get("WAYLAND_DISPLAY"):
        return _wayland_has_image()
    return _xclip_has_image()


def get_clipboard_text() -> str | None:
    """Get text content from the system clipboard.

    Returns the text string if clipboard contains text, None otherwise.
    """
    if sys.platform == "darwin":
        return _macos_get_text()
    if sys.platform == "win32":
        return _windows_get_text()
    if _is_wsl():
        text = _wsl_get_text()
        if text is not None:
            return text
        # Fall through to wl-paste/xclip for WSLg
    if os.environ.get("WAYLAND_DISPLAY"):
        text = _wayland_get_text()
        if text is not None:
            return text
    return _xclip_get_text()


def has_clipboard_text() -> bool:
    """Quick check: does the clipboard currently contain text?"""
    if sys.platform == "darwin":
        return _macos_has_text()
    if sys.platform == "win32":
        return _windows_has_text()
    if _is_wsl():
        if _wsl_has_text():
            return True
    if os.environ.get("WAYLAND_DISPLAY"):
        if _wayland_has_text():
            return True
    return _xclip_has_text()


# ── macOS ────────────────────────────────────────────────────────────────

def _macos_save(dest: Path) -> bool:
    """Try pngpaste first (fast, handles more formats), fall back to osascript."""
    return _macos_pngpaste(dest) or _macos_osascript(dest)


def _macos_has_image() -> bool:
    """Check if macOS clipboard contains image data."""
    try:
        info = subprocess.run(
            ["osascript", "-e", "clipboard info"],
            capture_output=True, text=True, timeout=3,
        )
        return "«class PNGf»" in info.stdout or "«class TIFF»" in info.stdout
    except Exception:
        return False


def _macos_pngpaste(dest: Path) -> bool:
    """Use pngpaste (brew install pngpaste) — fastest, cleanest."""
    try:
        r = subprocess.run(
            ["pngpaste", str(dest)],
            capture_output=True, timeout=3,
        )
        if r.returncode == 0 and dest.exists() and dest.stat().st_size > 0:
            return True
    except FileNotFoundError:
        pass  # pngpaste not installed
    except Exception as e:
        logger.debug("pngpaste failed: %s", e)
    return False


def _macos_osascript(dest: Path) -> bool:
    """Use osascript to extract PNG data from clipboard (always available)."""
    if not _macos_has_image():
        return False

    # Extract as PNG
    script = (
        'try\n'
        '  set imgData to the clipboard as «class PNGf»\n'
        f'  set f to open for access POSIX file "{dest}" with write permission\n'
        '  write imgData to f\n'
        '  close access f\n'
        'on error\n'
        '  return "fail"\n'
        'end try\n'
    )
    try:
        r = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0 and "fail" not in r.stdout and dest.exists() and dest.stat().st_size > 0:
            return True
    except Exception as e:
        logger.debug("osascript clipboard extract failed: %s", e)
    return False


def _macos_get_text() -> str | None:
    """Get text from macOS clipboard via pbpaste."""
    try:
        r = subprocess.run(
            ["pbpaste"],
            capture_output=True, text=True, timeout=3,
        )
        if r.returncode == 0 and r.stdout:
            return r.stdout
    except Exception as e:
        logger.debug("pbpaste failed: %s", e)
    return None


def _macos_has_text() -> bool:
    """Check if macOS clipboard has text content."""
    try:
        info = subprocess.run(
            ["osascript", "-e", "clipboard info"],
            capture_output=True, text=True, timeout=3,
        )
        return "«class utf8»" in info.stdout or "«class ut16»" in info.stdout
    except Exception:
        return False


# ── Linux ────────────────────────────────────────────────────────────────

def _is_wsl() -> bool:
    """Detect if running inside WSL (1 or 2)."""
    global _wsl_detected
    if _wsl_detected is not None:
        return _wsl_detected
    try:
        with open("/proc/version", "r") as f:
            _wsl_detected = "microsoft" in f.read().lower()
    except Exception:
        _wsl_detected = False
    return _wsl_detected


def _linux_save(dest: Path) -> bool:
    """Try clipboard backends in priority order: WSL → Wayland → X11."""
    if _is_wsl():
        if _wsl_save(dest):
            return True
        # Fall through — WSLg might have wl-paste or xclip working

    if os.environ.get("WAYLAND_DISPLAY"):
        if _wayland_save(dest):
            return True

    return _xclip_save(dest)


# ── Native Windows (sys.platform == "win32") ────────────────────────────

# PowerShell commands for native Windows — same .NET approach as WSL
# but invokes "powershell" (not "powershell.exe") since we're native.
_PS_WIN_CHECK_IMAGE = (
    "Add-Type -AssemblyName System.Windows.Forms;"
    "[System.Windows.Forms.Clipboard]::ContainsImage()"
)

_PS_WIN_EXTRACT_IMAGE = (
    "Add-Type -AssemblyName System.Windows.Forms;"
    "Add-Type -AssemblyName System.Drawing;"
    "$img = [System.Windows.Forms.Clipboard]::GetImage();"
    "if ($null -eq $img) { exit 1 }"
    "$ms = New-Object System.IO.MemoryStream;"
    "$img.Save($ms, [System.Drawing.Imaging.ImageFormat]::Png);"
    "[System.Convert]::ToBase64String($ms.ToArray())"
)

_PS_WIN_CHECK_TEXT = (
    "Add-Type -AssemblyName System.Windows.Forms;"
    "[System.Windows.Forms.Clipboard]::ContainsText()"
)

_PS_WIN_GET_TEXT = (
    "Add-Type -AssemblyName System.Windows.Forms;"
    "$t = [System.Windows.Forms.Clipboard]::GetText();"
    "if ($t) { $t } else { exit 1 }"
)


def _find_powershell_native() -> str:
    """Find PowerShell on native Windows."""
    # Prefer pwsh (PowerShell 7+) over Windows PowerShell 5.1
    for cmd in ("pwsh", "powershell"):
        if shutil.which(cmd):
            return cmd
    return "powershell"


def _windows_has_image() -> bool:
    """Check if Windows clipboard has an image (native win32)."""
    try:
        ps = _find_powershell_native()
        r = subprocess.run(
            [ps, "-NoProfile", "-NonInteractive", "-Command",
             _PS_WIN_CHECK_IMAGE],
            capture_output=True, text=True, timeout=10,
        )
        return r.returncode == 0 and "True" in r.stdout
    except FileNotFoundError:
        logger.debug("PowerShell not found — Windows clipboard unavailable")
    except Exception as e:
        logger.debug("Windows clipboard check failed: %s", e)
    return False


def _windows_save(dest: Path) -> bool:
    """Extract clipboard image on native Windows via PowerShell."""
    try:
        ps = _find_powershell_native()
        r = subprocess.run(
            [ps, "-NoProfile", "-NonInteractive", "-Command",
             _PS_WIN_EXTRACT_IMAGE],
            capture_output=True, text=True, timeout=20,
        )
        if r.returncode != 0:
            return False

        b64_data = r.stdout.strip()
        if not b64_data:
            return False

        png_bytes = base64.b64decode(b64_data)
        dest.write_bytes(png_bytes)
        return dest.exists() and dest.stat().st_size > 0

    except FileNotFoundError:
        logger.debug("PowerShell not found — Windows clipboard unavailable")
    except Exception as e:
        logger.debug("Windows clipboard extraction failed: %s", e)
        dest.unlink(missing_ok=True)
    return False


def _windows_has_text() -> bool:
    """Check if Windows clipboard has text (native win32)."""
    try:
        ps = _find_powershell_native()
        r = subprocess.run(
            [ps, "-NoProfile", "-NonInteractive", "-Command",
             _PS_WIN_CHECK_TEXT],
            capture_output=True, text=True, timeout=10,
        )
        return r.returncode == 0 and "True" in r.stdout
    except Exception as e:
        logger.debug("Windows text clipboard check failed: %s", e)
    return False


def _windows_get_text() -> str | None:
    """Get text from Windows clipboard (native win32)."""
    try:
        ps = _find_powershell_native()
        r = subprocess.run(
            [ps, "-NoProfile", "-NonInteractive", "-Command",
             _PS_WIN_GET_TEXT],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0 and r.stdout:
            return r.stdout.rstrip("\r\n")
    except Exception as e:
        logger.debug("Windows text clipboard get failed: %s", e)
    return None


# ── WSL2 (powershell.exe) ────────────────────────────────────────────────

# PowerShell script: get clipboard image as base64-encoded PNG on stdout.
# Using .NET System.Windows.Forms.Clipboard — always available on Windows.
_PS_CHECK_IMAGE = (
    "Add-Type -AssemblyName System.Windows.Forms;"
    "[System.Windows.Forms.Clipboard]::ContainsImage()"
)

_PS_EXTRACT_IMAGE = (
    "Add-Type -AssemblyName System.Windows.Forms;"
    "Add-Type -AssemblyName System.Drawing;"
    "$img = [System.Windows.Forms.Clipboard]::GetImage();"
    "if ($null -eq $img) { exit 1 }"
    "$ms = New-Object System.IO.MemoryStream;"
    "$img.Save($ms, [System.Drawing.Imaging.ImageFormat]::Png);"
    "[System.Convert]::ToBase64String($ms.ToArray())"
)

_PS_CHECK_TEXT = (
    "Add-Type -AssemblyName System.Windows.Forms;"
    "[System.Windows.Forms.Clipboard]::ContainsText()"
)

_PS_GET_TEXT = (
    "Add-Type -AssemblyName System.Windows.Forms;"
    "$t = [System.Windows.Forms.Clipboard]::GetText();"
    "if ($t) { $t } else { exit 1 }"
)


def _find_powershell_wsl() -> str:
    """Find powershell.exe accessible from WSL.

    Tries in order:
    1. powershell.exe on PATH (works if Windows PATH is appended to WSL PATH)
    2. Common Windows install locations via /mnt/c/
    """
    global _powershell_path
    if _powershell_path is not None:
        return _powershell_path

    # Check PATH first
    path = shutil.which("powershell.exe")
    if path:
        _powershell_path = path
        return path

    # Try common Windows install locations
    common_paths = [
        "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe",
        "/mnt/c/Windows/SysWOW64/WindowsPowerShell/v1.0/powershell.exe",
    ]
    for p in common_paths:
        if os.path.isfile(p) and os.access(p, os.X_OK):
            _powershell_path = p
            logger.debug("Found powershell.exe at %s (not on PATH)", p)
            return p

    # Last resort — just return the name and let subprocess raise FileNotFoundError
    _powershell_path = "powershell.exe"
    return _powershell_path


def _wsl_has_image() -> bool:
    """Check if Windows clipboard has an image (via powershell.exe)."""
    try:
        ps = _find_powershell_wsl()
        r = subprocess.run(
            [ps, "-NoProfile", "-NonInteractive", "-Command",
             _PS_CHECK_IMAGE],
            capture_output=True, text=True, timeout=15,
        )
        return r.returncode == 0 and "True" in r.stdout
    except FileNotFoundError:
        logger.debug("powershell.exe not found — WSL clipboard unavailable")
    except Exception as e:
        logger.debug("WSL clipboard check failed: %s", e)
    return False


def _wsl_save(dest: Path) -> bool:
    """Extract clipboard image via powershell.exe → base64 → decode to PNG."""
    try:
        ps = _find_powershell_wsl()
        r = subprocess.run(
            [ps, "-NoProfile", "-NonInteractive", "-Command",
             _PS_EXTRACT_IMAGE],
            capture_output=True, text=True, timeout=25,
        )
        if r.returncode != 0:
            return False

        b64_data = r.stdout.strip()
        if not b64_data:
            return False

        png_bytes = base64.b64decode(b64_data)
        dest.write_bytes(png_bytes)
        return dest.exists() and dest.stat().st_size > 0

    except FileNotFoundError:
        logger.debug("powershell.exe not found — WSL clipboard unavailable")
    except Exception as e:
        logger.debug("WSL clipboard extraction failed: %s", e)
        dest.unlink(missing_ok=True)
    return False


def _wsl_has_text() -> bool:
    """Check if Windows clipboard has text (via powershell.exe from WSL)."""
    try:
        ps = _find_powershell_wsl()
        r = subprocess.run(
            [ps, "-NoProfile", "-NonInteractive", "-Command",
             _PS_CHECK_TEXT],
            capture_output=True, text=True, timeout=15,
        )
        return r.returncode == 0 and "True" in r.stdout
    except FileNotFoundError:
        logger.debug("powershell.exe not found — WSL text clipboard unavailable")
    except Exception as e:
        logger.debug("WSL text clipboard check failed: %s", e)
    return False


def _wsl_get_text() -> str | None:
    """Get text from Windows clipboard via powershell.exe from WSL."""
    try:
        ps = _find_powershell_wsl()
        r = subprocess.run(
            [ps, "-NoProfile", "-NonInteractive", "-Command",
             _PS_GET_TEXT],
            capture_output=True, text=True, timeout=15,
        )
        if r.returncode == 0 and r.stdout:
            # PowerShell on Windows uses \r\n — normalize to \n
            return r.stdout.rstrip("\r\n").replace("\r\n", "\n")
    except FileNotFoundError:
        logger.debug("powershell.exe not found — WSL text clipboard unavailable")
    except Exception as e:
        logger.debug("WSL text clipboard get failed: %s", e)
    return None


# ── Wayland (wl-paste) ──────────────────────────────────────────────────

def _wayland_has_image() -> bool:
    """Check if Wayland clipboard has image content."""
    try:
        r = subprocess.run(
            ["wl-paste", "--list-types"],
            capture_output=True, text=True, timeout=3,
        )
        return r.returncode == 0 and any(
            t.startswith("image/") for t in r.stdout.splitlines()
        )
    except FileNotFoundError:
        logger.debug("wl-paste not installed — Wayland clipboard unavailable")
    except Exception:
        pass
    return False


def _wayland_save(dest: Path) -> bool:
    """Use wl-paste to extract clipboard image (Wayland sessions)."""
    try:
        # Check available MIME types
        types_r = subprocess.run(
            ["wl-paste", "--list-types"],
            capture_output=True, text=True, timeout=3,
        )
        if types_r.returncode != 0:
            return False
        types = types_r.stdout.splitlines()

        # Prefer PNG, fall back to other image formats
        mime = None
        for preferred in ("image/png", "image/jpeg", "image/bmp",
                          "image/gif", "image/webp"):
            if preferred in types:
                mime = preferred
                break

        if not mime:
            return False

        # Extract the image data
        with open(dest, "wb") as f:
            subprocess.run(
                ["wl-paste", "--type", mime],
                stdout=f, stderr=subprocess.DEVNULL, timeout=5, check=True,
            )

        if not dest.exists() or dest.stat().st_size == 0:
            dest.unlink(missing_ok=True)
            return False

        # BMP needs conversion to PNG (common in WSLg where only BMP
        # is bridged from Windows clipboard via RDP).
        if mime == "image/bmp":
            return _convert_to_png(dest)

        return True

    except FileNotFoundError:
        logger.debug("wl-paste not installed — Wayland clipboard unavailable")
    except Exception as e:
        logger.debug("wl-paste clipboard extraction failed: %s", e)
        dest.unlink(missing_ok=True)
    return False


def _wayland_has_text() -> bool:
    """Check if Wayland clipboard has text content."""
    try:
        r = subprocess.run(
            ["wl-paste", "--list-types"],
            capture_output=True, text=True, timeout=3,
        )
        if r.returncode != 0:
            return False
        types = r.stdout.splitlines()
        return any(t.startswith("text/") or t == "STRING" or t == "UTF8_STRING"
                    for t in types)
    except Exception:
        return False


def _wayland_get_text() -> str | None:
    """Get text from Wayland clipboard via wl-paste."""
    try:
        r = subprocess.run(
            ["wl-paste", "--no-newline"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0 and r.stdout:
            return r.stdout
    except FileNotFoundError:
        logger.debug("wl-paste not installed — Wayland text clipboard unavailable")
    except Exception as e:
        logger.debug("wl-paste text extraction failed: %s", e)
    return None


def _convert_to_png(path: Path) -> bool:
    """Convert an image file to PNG in-place (requires Pillow or ImageMagick)."""
    # Try Pillow first (likely installed in the venv)
    try:
        from PIL import Image
        img = Image.open(path)
        img.save(path, "PNG")
        return True
    except ImportError:
        pass
    except Exception as e:
        logger.debug("Pillow BMP→PNG conversion failed: %s", e)

    # Fall back to ImageMagick convert
    tmp = path.with_suffix(".bmp")
    try:
        path.rename(tmp)
        r = subprocess.run(
            ["convert", str(tmp), "png:" + str(path)],
            capture_output=True, timeout=5,
        )
        if r.returncode == 0 and path.exists() and path.stat().st_size > 0:
            tmp.unlink(missing_ok=True)
            return True
        else:
            # Convert failed — restore the original file
            tmp.rename(path)
    except FileNotFoundError:
        logger.debug("ImageMagick not installed — cannot convert BMP to PNG")
        if tmp.exists() and not path.exists():
            tmp.rename(path)
    except Exception as e:
        logger.debug("ImageMagick BMP→PNG conversion failed: %s", e)
        if tmp.exists() and not path.exists():
            tmp.rename(path)

    # Can't convert — BMP is still usable as-is for most APIs
    return path.exists() and path.stat().st_size > 0


# ── X11 (xclip) ─────────────────────────────────────────────────────────

def _xclip_has_image() -> bool:
    """Check if X11 clipboard has image content."""
    try:
        r = subprocess.run(
            ["xclip", "-selection", "clipboard", "-t", "TARGETS", "-o"],
            capture_output=True, text=True, timeout=3,
        )
        return r.returncode == 0 and "image/png" in r.stdout
    except FileNotFoundError:
        pass
    except Exception:
        pass
    return False


def _xclip_save(dest: Path) -> bool:
    """Use xclip to extract clipboard image (X11 sessions)."""
    # Check if clipboard has image content
    try:
        targets = subprocess.run(
            ["xclip", "-selection", "clipboard", "-t", "TARGETS", "-o"],
            capture_output=True, text=True, timeout=3,
        )
        if "image/png" not in targets.stdout:
            return False
    except FileNotFoundError:
        logger.debug("xclip not installed — X11 clipboard image paste unavailable")
        return False
    except Exception:
        return False

    # Extract PNG data
    try:
        with open(dest, "wb") as f:
            subprocess.run(
                ["xclip", "-selection", "clipboard", "-t", "image/png", "-o"],
                stdout=f, stderr=subprocess.DEVNULL, timeout=5, check=True,
            )
        if dest.exists() and dest.stat().st_size > 0:
            return True
    except Exception as e:
        logger.debug("xclip image extraction failed: %s", e)
        dest.unlink(missing_ok=True)
    return False


def _xclip_has_text() -> bool:
    """Check if X11 clipboard has text content."""
    try:
        r = subprocess.run(
            ["xclip", "-selection", "clipboard", "-t", "TARGETS", "-o"],
            capture_output=True, text=True, timeout=3,
        )
        if r.returncode != 0:
            return False
        targets = r.stdout.splitlines()
        return any(t in ("UTF8_STRING", "STRING", "text/plain")
                    for t in targets)
    except Exception:
        return False


def _xclip_get_text() -> str | None:
    """Get text from X11 clipboard via xclip."""
    try:
        r = subprocess.run(
            ["xclip", "-selection", "clipboard", "-o"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0 and r.stdout:
            return r.stdout
    except FileNotFoundError:
        logger.debug("xclip not installed — X11 text clipboard unavailable")
    except Exception as e:
        logger.debug("xclip text extraction failed: %s", e)
    return None

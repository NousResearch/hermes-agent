"""Clipboard image extraction and text writing for macOS, Windows, Linux, and WSL2.

Provides:
  - save_clipboard_image(dest)    — extract image from clipboard, save as PNG
  - has_clipboard_image()         — quick check for image content
  - write_clipboard_text(text, *) — copy text to the system clipboard, with
                                    native binaries (pbcopy / wl-copy / xclip /
                                    xsel / clip.exe) and OSC 52 as a fallback

No external Python dependencies — uses only OS-level CLI tools that ship
with the platform (or are commonly installed).

Platform support:
  macOS   — osascript (always available), pngpaste (if installed); pbcopy for text
  Windows — PowerShell via WinForms, Get-Clipboard, file-drop fallback; clip.exe for text
  WSL2    — powershell.exe via WinForms, Get-Clipboard, file-drop fallback; clip.exe for text
  Linux   — wl-paste/wl-copy (Wayland), xclip/xsel (X11)
"""

import base64
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

from hermes_constants import is_wsl as _is_wsl

logger = logging.getLogger(__name__)


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


# ── Text writer ──────────────────────────────────────────────────────────

def _is_ssh_session() -> bool:
    """True when the current process is part of an SSH session.

    Both ``SSH_CONNECTION`` and ``SSH_TTY`` are set by sshd; checking either
    is sufficient. We use this to decide between native-first (local) and
    OSC-52-first (remote) clipboard ordering — running pbcopy/xclip on the
    remote host is useless when the user is sitting at a local terminal.
    """
    return bool(os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_TTY"))


def _osc52_sequence(text: str) -> str:
    payload = base64.b64encode(text.encode("utf-8")).decode("ascii")
    return f"\x1b]52;c;{payload}\x07"


def _emit_osc52(text: str, writer: Optional[Callable[[str], None]]) -> bool:
    """Emit an OSC 52 escape sequence via *writer*.

    Returns True if the bytes were flushed to the writer without raising.
    This is **best-effort only** — OSC 52 is fire-and-forget: we cannot
    observe whether the receiving terminal actually honoured the sequence
    (macOS Terminal.app, for example, silently drops it). A True return
    here means "we wrote it"; it does not mean "the system clipboard
    now contains *text*".

    Returns False if no writer was provided or the writer raised.
    """
    if writer is None:
        return False
    try:
        writer(_osc52_sequence(text))
        return True
    except Exception as e:
        logger.debug("OSC 52 writer failed: %s", e)
        return False


def write_clipboard_text(
    text: str,
    *,
    osc52_writer: Optional[Callable[[str], None]] = None,
) -> bool:
    """Copy *text* to the system clipboard.

    Tries the most reliable path for the current environment first:

      - **Local terminal** — native binary (pbcopy on macOS, wl-copy/xclip/xsel
        on Linux, clip.exe on WSL/Windows), then OSC 52 as a fallback.
      - **SSH session** — OSC 52 first (the local terminal honours it across
        the SSH tunnel), then native binaries (which would copy on the
        *remote* host but are still better than nothing).

    *osc52_writer* is a callable that accepts the OSC 52 escape sequence and
    writes it to the user's terminal (e.g. ``prompt_toolkit``'s output).
    Pass ``None`` if no terminal sink is available — in that case only the
    native path is attempted.

    Return value is True if **either** of these happened:

      - A native binary (pbcopy / wl-copy / xclip / xsel / clip / clip.exe)
        exited with code 0 — this is a real confirmation that the system
        clipboard was written.
      - The OSC 52 escape sequence was successfully flushed to the
        terminal sink — this is **best-effort only**: we cannot observe
        whether the terminal honoured it. Several terminals (notably
        macOS Terminal.app) silently drop OSC 52, so a True return on
        this path does not guarantee the user can paste *text* anywhere.

    Returns False only when every available path failed outright (no
    backend installed, no terminal sink, all writes raised). Callers
    should report a real error to the user when this returns False
    rather than claiming success.
    """
    if not text:
        return False

    paths = (_emit_osc52, _native_clipboard_write) if _is_ssh_session() \
        else (_native_clipboard_write, _emit_osc52)
    for path in paths:
        if path(text, osc52_writer):
            return True
    return False


def _native_clipboard_write(text: str, _osc52_writer: Optional[Callable[[str], None]]) -> bool:
    """Dispatch to the platform-appropriate native clipboard binary."""
    if sys.platform == "darwin":
        return _macos_write_text(text)
    if sys.platform == "win32":
        return _windows_write_text(text)
    if _is_wsl():
        return _wsl_write_text(text) or _linux_write_text(text)
    return _linux_write_text(text)


def _run_clipboard_writer(cmd: list[str], text: str, *, label: str) -> bool:
    """Invoke *cmd* with *text* on stdin. Return True on success."""
    try:
        r = subprocess.run(
            cmd,
            input=text.encode("utf-8"),
            capture_output=True,
            timeout=5,
        )
        if r.returncode == 0:
            return True
        logger.debug("%s exited %d: %s", label, r.returncode, r.stderr[:200] if r.stderr else "")
    except FileNotFoundError:
        logger.debug("%s not installed — clipboard write unavailable via this path", label)
    except subprocess.TimeoutExpired:
        logger.debug("%s timed out writing to clipboard", label)
    except Exception as e:
        logger.debug("%s clipboard write failed: %s", label, e)
    return False


def _macos_write_text(text: str) -> bool:
    return _run_clipboard_writer(["pbcopy"], text, label="pbcopy")


def _linux_write_text(text: str) -> bool:
    """Write to Linux clipboard. Wayland first if available, then X11 (xclip → xsel)."""
    if os.environ.get("WAYLAND_DISPLAY"):
        if _run_clipboard_writer(["wl-copy"], text, label="wl-copy"):
            return True
    # X11 fallbacks. Try xclip first, then xsel — both are common on different distros.
    if _run_clipboard_writer(
        ["xclip", "-selection", "clipboard"], text, label="xclip"
    ):
        return True
    return _run_clipboard_writer(
        ["xsel", "--clipboard", "--input"], text, label="xsel"
    )


def _wsl_write_text(text: str) -> bool:
    """WSL has clip.exe on PATH and bridges to the Windows clipboard."""
    return _run_clipboard_writer(["clip.exe"], text, label="clip.exe")


def _windows_write_text(text: str) -> bool:
    return _run_clipboard_writer(["clip"], text, label="clip")


def has_clipboard_image() -> bool:
    """Quick check: does the clipboard currently contain an image?

    Lighter than save_clipboard_image — doesn't extract or write anything.
    """
    if sys.platform == "darwin":
        return _macos_has_image()
    if sys.platform == "win32":
        return _windows_has_image()
    # Match _linux_save fallthrough order: WSL → Wayland → X11
    if _is_wsl() and _wsl_has_image():
        return True
    if os.environ.get("WAYLAND_DISPLAY") and _wayland_has_image():
        return True
    return _xclip_has_image()


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


# ── Shared PowerShell scripts (native Windows + WSL2) ─────────────────────

# .NET System.Windows.Forms.Clipboard — used by both native Windows (powershell)
# and WSL2 (powershell.exe) paths.
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

_PS_CHECK_IMAGE_GET_CLIPBOARD = (
    "try { "
    "$img = Get-Clipboard -Format Image -ErrorAction Stop;"
    "if ($null -ne $img) { 'True' } else { 'False' }"
    "} catch { 'False' }"
)

_PS_EXTRACT_IMAGE_GET_CLIPBOARD = (
    "try { "
    "Add-Type -AssemblyName System.Drawing;"
    "Add-Type -AssemblyName PresentationCore;"
    "Add-Type -AssemblyName WindowsBase;"
    "$img = Get-Clipboard -Format Image -ErrorAction Stop;"
    "if ($null -eq $img) { exit 1 }"
    "$ms = New-Object System.IO.MemoryStream;"
    "if ($img -is [System.Drawing.Image]) {"
    "$img.Save($ms, [System.Drawing.Imaging.ImageFormat]::Png)"
    "} elseif ($img -is [System.Windows.Media.Imaging.BitmapSource]) {"
    "$enc = New-Object System.Windows.Media.Imaging.PngBitmapEncoder;"
    "$enc.Frames.Add([System.Windows.Media.Imaging.BitmapFrame]::Create($img));"
    "$enc.Save($ms)"
    "} else { exit 2 }"
    "[System.Convert]::ToBase64String($ms.ToArray())"
    "} catch { exit 1 }"
)

_FILEDROP_IMAGE_EXTS = "'.png','.jpg','.jpeg','.gif','.webp','.bmp','.tiff','.tif'"

_PS_CHECK_FILEDROP_IMAGE = (
    "try { "
    "$files = Get-Clipboard -Format FileDropList -ErrorAction Stop;"
    f"$exts = @({_FILEDROP_IMAGE_EXTS});"
    "$hit = $files | Where-Object { $exts -contains ([System.IO.Path]::GetExtension($_).ToLowerInvariant()) } | Select-Object -First 1;"
    "if ($null -ne $hit) { 'True' } else { 'False' }"
    "} catch { 'False' }"
)

_PS_EXTRACT_FILEDROP_IMAGE = (
    "try { "
    "$files = Get-Clipboard -Format FileDropList -ErrorAction Stop;"
    f"$exts = @({_FILEDROP_IMAGE_EXTS});"
    "$hit = $files | Where-Object { $exts -contains ([System.IO.Path]::GetExtension($_).ToLowerInvariant()) } | Select-Object -First 1;"
    "if ($null -eq $hit) { exit 1 }"
    "[System.Convert]::ToBase64String([System.IO.File]::ReadAllBytes($hit))"
    "} catch { exit 1 }"
)

_POWERSHELL_HAS_IMAGE_SCRIPTS = (
    _PS_CHECK_IMAGE,
    _PS_CHECK_IMAGE_GET_CLIPBOARD,
    _PS_CHECK_FILEDROP_IMAGE,
)

_POWERSHELL_EXTRACT_IMAGE_SCRIPTS = (
    _PS_EXTRACT_IMAGE,
    _PS_EXTRACT_IMAGE_GET_CLIPBOARD,
    _PS_EXTRACT_FILEDROP_IMAGE,
)


def _run_powershell(exe: str, script: str, timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(
        [exe, "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True, text=True, timeout=timeout,
    )


def _write_base64_image(dest: Path, b64_data: str) -> bool:
    image_bytes = base64.b64decode(b64_data, validate=True)
    dest.write_bytes(image_bytes)
    return dest.exists() and dest.stat().st_size > 0


def _powershell_has_image(exe: str, *, timeout: int, label: str) -> bool:
    for script in _POWERSHELL_HAS_IMAGE_SCRIPTS:
        try:
            r = _run_powershell(exe, script, timeout=timeout)
            if r.returncode == 0 and "True" in r.stdout:
                return True
        except FileNotFoundError:
            logger.debug("%s not found — clipboard unavailable", exe)
            return False
        except Exception as e:
            logger.debug("%s clipboard image check failed: %s", label, e)
    return False


def _powershell_save_image(exe: str, dest: Path, *, timeout: int, label: str) -> bool:
    for script in _POWERSHELL_EXTRACT_IMAGE_SCRIPTS:
        try:
            r = _run_powershell(exe, script, timeout=timeout)
            if r.returncode != 0:
                continue

            b64_data = r.stdout.strip()
            if not b64_data:
                continue

            if _write_base64_image(dest, b64_data):
                return True
        except FileNotFoundError:
            logger.debug("%s not found — clipboard unavailable", exe)
            return False
        except Exception as e:
            logger.debug("%s clipboard image extraction failed: %s", label, e)
            dest.unlink(missing_ok=True)
    return False


# ── Native Windows ────────────────────────────────────────────────────────

# Native Windows uses ``powershell`` (Windows PowerShell 5.1, always present)
# or ``pwsh`` (PowerShell 7+, optional).  Discovery is cached per-process.


def _find_powershell() -> str | None:
    """Return the first available PowerShell executable, or None."""
    for name in ("powershell", "pwsh"):
        try:
            r = subprocess.run(
                [name, "-NoProfile", "-NonInteractive", "-Command", "echo ok"],
                capture_output=True, text=True, timeout=5,
            )
            if r.returncode == 0 and "ok" in r.stdout:
                return name
        except FileNotFoundError:
            continue
        except Exception:
            continue
    return None


# Cache the resolved PowerShell executable (checked once per process)
_ps_exe: str | None | bool = False  # False = not yet checked


def _get_ps_exe() -> str | None:
    global _ps_exe
    if _ps_exe is False:
        _ps_exe = _find_powershell()
    return _ps_exe


def _windows_has_image() -> bool:
    """Check if the Windows clipboard contains an image."""
    ps = _get_ps_exe()
    if ps is None:
        return False
    return _powershell_has_image(ps, timeout=5, label="Windows")


def _windows_save(dest: Path) -> bool:
    """Extract clipboard image on native Windows via PowerShell → base64 PNG."""
    ps = _get_ps_exe()
    if ps is None:
        logger.debug("No PowerShell found — Windows clipboard image paste unavailable")
        return False
    return _powershell_save_image(ps, dest, timeout=15, label="Windows")


# ── Linux ────────────────────────────────────────────────────────────────

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


# ── WSL2 (powershell.exe) ────────────────────────────────────────────────
# Reuses _PS_CHECK_IMAGE / _PS_EXTRACT_IMAGE defined above.

def _wsl_has_image() -> bool:
    """Check if Windows clipboard has an image (via powershell.exe)."""
    return _powershell_has_image("powershell.exe", timeout=8, label="WSL")


def _wsl_save(dest: Path) -> bool:
    """Extract clipboard image via powershell.exe → base64 → decode to PNG."""
    return _powershell_save_image("powershell.exe", dest, timeout=15, label="WSL")


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

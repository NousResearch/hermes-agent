#!/usr/bin/env python3
"""
Cross-platform desktop wallpaper setter.

Sets the given image as the desktop wallpaper using the appropriate native
method for the current OS and desktop environment.  Stdlib-only; Python 3.10+.

Usage:
    python3 set_wallpaper.py /path/to/image.png
    python3 set_wallpaper.py --mode center  /path/to/image.png
    python3 set_wallpaper.py --mode stretch /path/to/image.png

Output (JSON):
    {"status": "ok", "method": "gsettings", "path": "/abs/path/to/image.png"}
    {"status": "error", "error": "No supported desktop environment detected"}
"""

from __future__ import annotations

import ctypes
import json
import shutil
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Mode mapping — "fill" is the default (best for wallpapers)
# ---------------------------------------------------------------------------
FIT_MODES: dict[str, dict[str, str]] = {
    "fill": {
        "gsettings": "scaled",  # GNOME scales keeping aspect ratio (closest to fill)
        "feh": "--bg-fill",
        "sway": "fill",
        "kde": "2",           # Scaled, keep proportions
        "xfce": "5",          # Zoomed
    },
    "center": {
        "gsettings": "centered",
        "feh": "--bg-center",
        "sway": "center",
        "kde": "1",           # Centered
        "xfce": "1",          # Centered
    },
    "stretch": {
        "gsettings": "stretched",
        "feh": "--bg-scale",  # feh doesn't have stretch; --bg-scale is closest
        "sway": "stretch",
        "kde": "0",           # Scaled (tiled)
        "xfce": "2",          # Tiled
    },
    "fit": {
        "gsettings": "scaled",
        "feh": "--bg-max",    # Fit inside, borders if needed
        "sway": "fit",
        "kde": "2",
        "xfce": "4",          # Scaled
    },
    "tile": {
        "gsettings": "wallpaper",
        "feh": "--bg-tile",
        "sway": "tile",
        "kde": "3",           # Tiled
        "xfce": "3",          # Tiled
    },
}


def _which(executable: str) -> str | None:
    """Return the path to *executable* or None if not on PATH."""
    return shutil.which(executable)


def _run(*args: str, timeout: int = 15) -> subprocess.CompletedProcess:
    """Run a command; return CompletedProcess.  Stderr captured but ignored."""
    return subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _image_uri(path: str) -> str:
    """Convert a local filesystem path to a ``file://`` URI."""
    abs_path = str(Path(path).resolve())
    # Prefix with file:// — GNOME gsettings expects this
    return f"file://{abs_path}"


# ---------------------------------------------------------------------------
# Per-platform / per-DE setters
# ---------------------------------------------------------------------------

def _try_gnome(path: str, mode: str) -> bool:
    """Set wallpaper via GNOME gsettings (also works on Budgie, Cinnamon, Unity)."""
    gsettings = _which("gsettings")
    if gsettings is None:
        return False

    uri = _image_uri(path)
    picture_option = FIT_MODES[mode]["gsettings"]

    # Try the dark-scheme key first (Ubuntu 22.04+, GNOME 42+), fall back to
    # the legacy key.  Both are set so light/dark mode both get the wallpaper.
    ok = False
    for key in (
        "org.gnome.desktop.background",
        "org.gnome.desktop.background",
    ):
        light = _run(gsettings, "set", key, "picture-uri", uri)
        dark = _run(gsettings, "set", key, "picture-uri-dark", uri)
        opts = _run(gsettings, "set", key, "picture-options", picture_option)
        if light.returncode == 0:
            ok = True
    return ok


def _try_kde(path: str, mode: str) -> bool:
    """Set wallpaper via KDE Plasma dbus scripting."""
    # Plasma 5.26+ supports a simple dbus call via plasmashell
    plasma_script = _which("plasma-apply-wallpaperimage")
    if plasma_script is not None:
        result = _run(plasma_script, str(Path(path).resolve()))
        return result.returncode == 0

    # Fallback: kwriteconfig5 + dbus reload
    kwrite = _which("kwriteconfig5")
    if kwrite is not None:
        abs_path = str(Path(path).resolve())
        _run(kwrite, "plasma-org.kde.plasma.desktop", "",
             "wallpaper", abs_path)
        # Try to reload
        dbus_send = _which("dbus-send")
        if dbus_send is not None:
            _run(dbus_send, "--session", "--dest=org.kde.plasmashell",
                 "--type=method_call", "/PlasmaShell",
                 "org.kde.PlasmaShell.refreshCurrentShell")
        return True

    # Last resort: qdbus
    qdbus = _which("qdbus")
    if qdbus is not None:
        abs_path = str(Path(path).resolve())
        script = (
            'var allDesktops = desktops();'
            'for (i=0;i<allDesktops.length;i++) {{'
            '  d = allDesktops[i];'
            '  d.wallpaperPlugin = "org.kde.image";'
            '  d.currentConfigGroup = Array("Wallpaper", "org.kde.image", "General");'
            '  d.writeConfig("Image", "file://{path}");'
            '  d.writeConfig("FillMode", "{mode}");'
            '}}'
        ).format(path=abs_path, mode=FIT_MODES[mode]["kde"])
        result = _run(qdbus, "org.kde.plasmashell", "/PlasmaShell",
                      "org.kde.PlasmaShell.evaluateScript", script)
        return result.returncode == 0

    return False


def _try_xfce(path: str, mode: str) -> bool:
    """Set wallpaper via XFCE xfconf-query."""
    xfconf = _which("xfconf-query")
    if xfconf is None:
        return False

    abs_path = str(Path(path).resolve())
    picture_mode = FIT_MODES[mode]["xfce"]

    # XFCE stores per-monitor settings; we set all monitors at once
    # by writing to the default workspace
    props = [
        ("/backdrop/screen0/monitor0/workspace0/last-image", abs_path),
        ("/backdrop/screen0/monitor0/workspace0/image-style", picture_mode),
    ]
    ok = True
    for prop, value in props:
        result = _run(xfconf, "-c", "xfce4-desktop", "-p", prop, "-s", value)
        if result.returncode != 0:
            ok = False
    return ok


def _try_sway(path: str, mode: str) -> bool:
    """Set wallpaper via SwayWM swaymsg."""
    swaymsg = _which("swaymsg")
    if swaymsg is None:
        return False

    abs_path = str(Path(path).resolve())
    fit = FIT_MODES[mode]["sway"]
    result = _run(swaymsg, f"output * bg {abs_path} {fit}")
    return result.returncode == 0


def _try_feh(path: str, mode: str) -> bool:
    """Set wallpaper via feh (generic X11 fallback)."""
    feh = _which("feh")
    if feh is None:
        return False

    abs_path = str(Path(path).resolve())
    flag = FIT_MODES[mode]["feh"]
    result = _run(feh, flag, abs_path)
    return result.returncode == 0


def _try_nitrogen(path: str, _mode: str) -> bool:
    """Set wallpaper via nitrogen (another generic X11 tool)."""
    nitrogen = _which("nitrogen")
    if nitrogen is None:
        return False

    abs_path = str(Path(path).resolve())
    result = _run(nitrogen, "--set-zoom-fill", abs_path)
    return result.returncode == 0


def _try_macos(path: str, _mode: str) -> bool:
    """Set wallpaper on macOS via osascript / System Events."""
    osascript = _which("osascript")
    if osascript is None:
        return False

    abs_path = str(Path(path).resolve())
    # Set picture for every desktop (space) on every display
    script = (
        'tell application "System Events" to '
        'set picture of every desktop to POSIX file "{path}"'
    ).format(path=abs_path)
    result = _run(osascript, "-e", script, timeout=30)
    return result.returncode == 0


def _try_windows(path: str, _mode: str) -> bool:
    """Set wallpaper on Windows via SystemParametersInfoW."""
    abs_path = str(Path(path).resolve())
    SPI_SETDESKWALLPAPER = 20
    SPIF_UPDATEINIFILE = 0x01
    SPIF_SENDCHANGE = 0x02

    try:
        result = ctypes.windll.user32.SystemParametersInfoW(
            SPI_SETDESKWALLPAPER,
            0,
            abs_path,
            SPIF_UPDATEINIFILE | SPIF_SENDCHANGE,
        )
        return bool(result)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Ordered fallback chains per platform
# ---------------------------------------------------------------------------

_LINUX_SETTERS = [
    ("gnome", _try_gnome),
    ("kde", _try_kde),
    ("xfce", _try_xfce),
    ("sway", _try_sway),
    ("feh", _try_feh),
    ("nitrogen", _try_nitrogen),
]

_MACOS_SETTERS = [
    ("osascript", _try_macos),
]

_WINDOWS_SETTERS = [
    ("win32-api", _try_windows),
]


def set_wallpaper(path: str, mode: str = "fill") -> dict:
    """
    Set the desktop wallpaper to *path*.  Tries platform-appropriate
    methods in order; stops at the first success.

    Returns a result dict with keys ``status``, ``method``, ``path``
    (or ``error`` on failure).
    """
    image = Path(path).resolve()
    if not image.is_file():
        return {
            "status": "error",
            "error": f"File not found: {path}",
            "path": str(image),
        }

    if mode not in FIT_MODES:
        return {
            "status": "error",
            "error": f"Unknown fit mode: {mode!r}. Choices: {', '.join(FIT_MODES)}",
            "path": str(image),
        }

    platform = sys.platform
    if platform == "darwin":
        chain = _MACOS_SETTERS
    elif platform == "win32":
        chain = _WINDOWS_SETTERS
    else:
        chain = _LINUX_SETTERS

    for name, fn in chain:
        try:
            if fn(str(image), mode):
                return {
                    "status": "ok",
                    "method": name,
                    "path": str(image),
                }
        except Exception as exc:
            # One setter failing shouldn't prevent trying the next
            print(f"[set_wallpaper] {name} failed: {exc}", file=sys.stderr)

    return {
        "status": "error",
        "error": "No supported desktop environment detected. "
                 "Tried: " + ", ".join(name for name, _ in chain),
        "path": str(image),
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Set the desktop wallpaper (cross-platform).",
    )
    parser.add_argument(
        "image",
        help="Path to the image file to set as wallpaper.",
    )
    parser.add_argument(
        "--mode", "-m",
        default="fill",
        choices=list(FIT_MODES),
        help="How to fit the image on the desktop (default: fill).",
    )
    args = parser.parse_args()

    result = set_wallpaper(args.image, args.mode)
    print(json.dumps(result))
    sys.exit(0 if result["status"] == "ok" else 1)


if __name__ == "__main__":
    main()

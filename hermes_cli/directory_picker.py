"""Native folder picker for the Hermes dashboard (local server only)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from typing import Optional


class DirectoryPickerUnavailable(Exception):
    """Raised when no native folder picker is available on this host."""


def server_cwd() -> str:
    """Return the Hermes web server process cwd as an absolute path."""
    return os.path.abspath(os.getcwd())


def pick_directory(*, initial_dir: Optional[str] = None) -> Optional[str]:
    """Open a folder-only native dialog.

    Returns an absolute path, or ``None`` when the user cancels.
    """
    start: Optional[str] = None
    if initial_dir:
        expanded = os.path.abspath(os.path.expanduser(str(initial_dir)))
        if os.path.isdir(expanded):
            start = expanded

    if sys.platform == "darwin":
        return _pick_directory_macos(start)
    if sys.platform.startswith("win"):
        return _pick_directory_windows(start)
    return _pick_directory_linux(start)


def _pick_directory_macos(initial_dir: Optional[str]) -> Optional[str]:
    script = 'POSIX path of (choose folder with prompt "Select workspace folder"'
    if initial_dir:
        escaped = initial_dir.replace("\\", "\\\\").replace('"', '\\"')
        script += f' default location (POSIX file "{escaped}")'
    script += ")"
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise DirectoryPickerUnavailable(str(exc)) from exc
    if result.returncode != 0:
        return None
    chosen = (result.stdout or "").strip()
    return os.path.abspath(chosen) if chosen else None


def _pick_directory_windows(initial_dir: Optional[str]) -> Optional[str]:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError as exc:
        raise DirectoryPickerUnavailable("tkinter is not available") from exc

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        kwargs: dict = {"mustexist": True}
        if initial_dir:
            kwargs["initialdir"] = initial_dir
        chosen = filedialog.askdirectory(**kwargs)
    finally:
        root.destroy()
    if not chosen:
        return None
    return os.path.abspath(chosen)


def _pick_directory_linux(initial_dir: Optional[str]) -> Optional[str]:
    if shutil.which("zenity"):
        cmd = [
            "zenity",
            "--file-selection",
            "--directory",
            "--title=Select workspace folder",
        ]
        if initial_dir:
            cmd.extend(["--filename", initial_dir + os.sep])
        runner = subprocess.run
    elif shutil.which("kdialog"):
        cmd = [
            "kdialog",
            "--getexistingdirectory",
            initial_dir or os.path.expanduser("~"),
            "--title",
            "Select workspace folder",
        ]
        runner = subprocess.run
    else:
        raise DirectoryPickerUnavailable(
            "No folder picker found (install zenity or kdialog)",
        )

    try:
        result = runner(cmd, capture_output=True, text=True, timeout=300, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise DirectoryPickerUnavailable(str(exc)) from exc
    if result.returncode != 0:
        return None
    chosen = (result.stdout or "").strip()
    return os.path.abspath(chosen) if chosen else None

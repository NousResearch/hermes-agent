"""Shared helpers for attaching Hermes to a local Chrome CDP port."""

from __future__ import annotations

import os
import platform
import shlex
import shutil
import subprocess

from hermes_constants import get_hermes_home


DEFAULT_BROWSER_CDP_PORT = 9222
DEFAULT_BROWSER_CDP_URL = f"http://127.0.0.1:{DEFAULT_BROWSER_CDP_PORT}"

_DARWIN_APPS = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
)

_WINDOWS_INSTALL_PARTS = (
    ("Google", "Chrome", "Application", "chrome.exe"),
    ("Chromium", "Application", "chrome.exe"),
    ("Chromium", "Application", "chromium.exe"),
    ("BraveSoftware", "Brave-Browser", "Application", "brave.exe"),
    ("Microsoft", "Edge", "Application", "msedge.exe"),
)

_LINUX_BIN_NAMES = (
    "google-chrome", "google-chrome-stable", "chromium-browser",
    "chromium", "brave-browser", "microsoft-edge",
)

_WINDOWS_BIN_NAMES = (
    "chrome.exe", "msedge.exe", "brave.exe", "chromium.exe",
    "chrome", "msedge", "brave", "chromium",
)


def get_chrome_debug_candidates(system: str) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()

    def add(path: str | None) -> None:
        if not path:
            return
        normalized = os.path.normcase(os.path.normpath(path))
        if normalized in seen or not os.path.isfile(path):
            return
        candidates.append(path)
        seen.add(normalized)

    def add_install_paths(bases: tuple[str | None, ...]) -> None:
        for base in filter(None, bases):
            for parts in _WINDOWS_INSTALL_PARTS:
                add(os.path.join(base, *parts))

    if system == "Darwin":
        for app in _DARWIN_APPS:
            add(app)
        return candidates

    if system == "Windows":
        for name in _WINDOWS_BIN_NAMES:
            add(shutil.which(name))
        add_install_paths((
            os.environ.get("ProgramFiles"),
            os.environ.get("ProgramFiles(x86)"),
            os.environ.get("LOCALAPPDATA"),
        ))
        return candidates

    for name in _LINUX_BIN_NAMES:
        add(shutil.which(name))
    add_install_paths(("/mnt/c/Program Files", "/mnt/c/Program Files (x86)"))
    return candidates


def chrome_debug_data_dir() -> str:
    return str(get_hermes_home() / "chrome-debug")


def _chrome_debug_args(port: int) -> list[str]:
    return [
        f"--remote-debugging-port={port}",
        f"--user-data-dir={chrome_debug_data_dir()}",
        "--no-first-run",
        "--no-default-browser-check",
    ]


def manual_chrome_debug_command(port: int = DEFAULT_BROWSER_CDP_PORT, system: str | None = None) -> str | None:
    system = system or platform.system()
    candidates = get_chrome_debug_candidates(system)

    if candidates:
        argv = [candidates[0], *_chrome_debug_args(port)]
        return subprocess.list2cmdline(argv) if system == "Windows" else shlex.join(argv)

    if system == "Darwin":
        data_dir = chrome_debug_data_dir()
        return (
            f'open -a "Google Chrome" --args --remote-debugging-port={port} '
            f'--user-data-dir="{data_dir}" --no-first-run --no-default-browser-check'
        )

    return None


def _detach_kwargs(system: str) -> dict:
    if system != "Windows":
        return {"start_new_session": True}
    flags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(
        subprocess, "CREATE_NEW_PROCESS_GROUP", 0
    )
    return {"creationflags": flags} if flags else {}


def try_launch_chrome_debug(port: int = DEFAULT_BROWSER_CDP_PORT, system: str | None = None) -> bool:
    system = system or platform.system()
    candidates = get_chrome_debug_candidates(system)
    if not candidates:
        return False

    os.makedirs(chrome_debug_data_dir(), exist_ok=True)
    try:
        subprocess.Popen(
            [candidates[0], *_chrome_debug_args(port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            **_detach_kwargs(system),
        )
        return True
    except Exception:
        return False

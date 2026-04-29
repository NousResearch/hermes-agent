"""Shared helpers for attaching Hermes to a local Chrome CDP port."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess

from hermes_constants import get_hermes_home


DEFAULT_BROWSER_CDP_PORT = 9222
DEFAULT_BROWSER_CDP_URL = f"http://127.0.0.1:{DEFAULT_BROWSER_CDP_PORT}"


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

    def add_from_path(*names: str) -> None:
        for name in names:
            add(shutil.which(name))

    if system == "Darwin":
        for app in (
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/Applications/Chromium.app/Contents/MacOS/Chromium",
            "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
            "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
        ):
            add(app)
    elif system == "Windows":
        add_from_path(
            "chrome.exe", "msedge.exe", "brave.exe", "chromium.exe",
            "chrome", "msedge", "brave", "chromium",
        )
        for base in (
            os.environ.get("ProgramFiles"),
            os.environ.get("ProgramFiles(x86)"),
            os.environ.get("LOCALAPPDATA"),
        ):
            if not base:
                continue
            for parts in (
                ("Google", "Chrome", "Application", "chrome.exe"),
                ("Chromium", "Application", "chrome.exe"),
                ("Chromium", "Application", "chromium.exe"),
                ("BraveSoftware", "Brave-Browser", "Application", "brave.exe"),
                ("Microsoft", "Edge", "Application", "msedge.exe"),
            ):
                add(os.path.join(base, *parts))
    else:
        add_from_path(
            "google-chrome", "google-chrome-stable", "chromium-browser",
            "chromium", "brave-browser", "microsoft-edge",
        )

    return candidates


def chrome_debug_data_dir() -> str:
    return str(get_hermes_home() / "chrome-debug")


def manual_chrome_debug_command(port: int = DEFAULT_BROWSER_CDP_PORT, system: str | None = None) -> str:
    system = system or platform.system()
    data_dir = chrome_debug_data_dir()
    if system == "Darwin":
        return (
            'open -a "Google Chrome" --args'
            f" --remote-debugging-port={port}"
            f' --user-data-dir="{data_dir}"'
            " --no-first-run --no-default-browser-check"
        )
    if system == "Windows":
        return (
            f"chrome.exe --remote-debugging-port={port}"
            f' --user-data-dir="{data_dir}"'
            " --no-first-run --no-default-browser-check"
        )
    return (
        f"google-chrome --remote-debugging-port={port}"
        f' --user-data-dir="{data_dir}"'
        " --no-first-run --no-default-browser-check"
    )


def try_launch_chrome_debug(port: int = DEFAULT_BROWSER_CDP_PORT, system: str | None = None) -> bool:
    candidates = get_chrome_debug_candidates(system or platform.system())
    if not candidates:
        return False

    os.makedirs(chrome_debug_data_dir(), exist_ok=True)
    try:
        subprocess.Popen(
            [
                candidates[0],
                f"--remote-debugging-port={port}",
                f"--user-data-dir={chrome_debug_data_dir()}",
                "--no-first-run",
                "--no-default-browser-check",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return True
    except Exception:
        return False

"""Lazy dependency bootstrapper for non-Python runtime deps.

Wraps install.sh --ensure to install node, browser, ripgrep, ffmpeg
on first use. Prompts interactively unless told not to.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

_DEP_CHECKS = {
    "node": lambda: shutil.which("node") is not None,
    "browser": lambda: (
        shutil.which("agent-browser") is not None
        or _has_system_browser()
        or _has_hermes_agent_browser()
    ),
    "ripgrep": lambda: shutil.which("rg") is not None,
    "ffmpeg": lambda: shutil.which("ffmpeg") is not None,
}

_DEP_DESCRIPTIONS = {
    "node": "Node.js (required for browser tools and TUI)",
    "browser": "Browser engine (Chromium, for web browsing tools)",
    "ripgrep": "ripgrep (fast file search)",
    "ffmpeg": "ffmpeg (TTS voice messages)",
}


def _has_system_browser() -> bool:
    for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
        if shutil.which(name):
            return True
    return False


def _has_hermes_agent_browser() -> bool:
    hermes_home = os.environ.get("HERMES_HOME", str(Path.home() / ".hermes"))
    return (Path(hermes_home) / "node_modules" / ".bin" / "agent-browser").is_file()


def _find_install_script(
    package_dir: Path | None = None,
    repo_root: Path | None = None,
) -> Path | None:
    """Locate install.sh — bundled in wheel or in git checkout."""
    if package_dir is None:
        package_dir = Path(__file__).parent
    if repo_root is None:
        repo_root = package_dir.parent

    bundled = package_dir / "scripts" / "install.sh"
    if bundled.is_file():
        return bundled
    repo = repo_root / "scripts" / "install.sh"
    if repo.is_file():
        return repo
    return None


def ensure_dependency(dep: str, interactive: bool = True) -> bool:
    """Ensure a non-Python dependency is available. Returns True if available."""
    check = _DEP_CHECKS.get(dep)
    if check and check():
        return True

    script = _find_install_script()
    if script is None:
        if interactive:
            desc = _DEP_DESCRIPTIONS.get(dep, dep)
            print(f"  {desc} is not installed and install.sh was not found.")
            print(f"  Install {dep} manually and try again.")
        return False

    if interactive and sys.stdin.isatty():
        desc = _DEP_DESCRIPTIONS.get(dep, dep)
        try:
            reply = input(f"{desc} is not installed. Install now? [Y/n] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        if reply not in ("", "y", "yes"):
            return False

    result = subprocess.run(
        ["bash", str(script), "--ensure", dep],
        env={**os.environ, "IS_INTERACTIVE": "false"},
    )
    if result.returncode != 0:
        return False

    if check:
        return check()
    return True

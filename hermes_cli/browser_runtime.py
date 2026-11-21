"""Read-only full-Chromium selection shared by browser launchers.

Nothing here installs, downloads or starts a browser: a caller that may provision (the launcher's
one-shot ``pm.ensure("chromium")``) does so itself and then asks again. The readiness probes below
only ask the dynamic loader about a binary (``ldd``) — they never run it.
"""

from __future__ import annotations

import importlib.util
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pm

# Written by the Docker image build with the pm Chromium binary it staged. The binary's path differs
# per architecture, so the build resolves it
# once; it also survives a volume over the runtime dir that hides pm's facts.
BAKED_EXECUTABLE_FILE = Path("/etc/hermes/chromium-executable-path")

# Distribution command names for a system Chromium, used only where pm builds none for this target
# (Termux: `pkg install chromium` from x11-repo).
_SYSTEM_CHROMIUM_COMMANDS = ("chromium", "chromium-browser")

# Where Playwright documents the distribution packages Chromium needs that no browser download carries.
SYSTEM_DEPENDENCY_GUIDE = "https://playwright.dev/python/docs/browsers#install-system-dependencies"

# `ldd` only resolves names; a broker/VFS that hangs it must not stall the readiness check.
_LDD_TIMEOUT_S = 10

# The dynamic loader's own words when a shared object is absent — the one launch failure where
# reinstalling the browser cannot help.
_MISSING_LIBRARY_MARKER = "error while loading shared libraries"


def _configured_executable() -> str | None:
    """``browser.executable_path`` from the active profile's config (managed overlay applied)."""
    from hermes_cli.config import cfg_get, load_config_readonly

    value = cfg_get(load_config_readonly(), "browser", "executable_path", default="")
    value = os.path.expanduser(value.strip()) if isinstance(value, str) else ""
    return value or None


def _pm_executable() -> str | None:
    installed = pm.installed_package("chromium")
    return str(installed.binary) if installed and installed.binary is not None else None


def _baked_executable() -> str | None:
    try:
        value = BAKED_EXECUTABLE_FILE.read_text(encoding="utf-8-sig").strip()
    except OSError:  # not a Hermes Docker image: nothing baked
        return None
    return value if value and os.path.isfile(value) else None


def _pm_has_no_chromium() -> bool:
    """pm publishes no Chromium for this target (Termux/Android, musl)."""
    return pm.get_package("chromium").missing_reason(pm.current_target()) is not None


def _system_executable() -> str | None:
    from hermes_platform.resolver import locate_command

    for name in _SYSTEM_CHROMIUM_COMMANDS:
        found = locate_command(name).command
        if found:
            return found[0]
    return None


def chromium_executable(*, allow_override: bool = True) -> str | None:
    """The full Chromium a browser launcher should run, or ``None`` when none is available.

    Order: ``browser.executable_path`` (skipped with ``allow_override=False``, for callers that need
    pm's own build) -> pm's installed ``chromium`` -> the Docker image's baked path -> a system
    ``chromium`` / ``chromium-browser`` on PATH, only on targets pm builds no Chromium for (Termux).
    """
    if allow_override and (override := _configured_executable()):
        return override
    if found := _pm_executable() or _baked_executable():
        return found
    return _system_executable() if _pm_has_no_chromium() else None


def chromium_missing_hint() -> str:
    """One actionable line for "no Chromium" on this machine. Never suggests npm."""
    from hermes_platform.host.runtime import is_termux

    if pinned := _configured_executable():  # the pin wins over every other source, so name it
        return (f"browser.executable_path is set to {pinned}, which is not a browser on this machine: fix the "
                "path, or clear it to use Hermes's Chromium.")
    if is_termux():
        return ("No Chromium found: pm builds none for Termux. Install one with "
                "`pkg install x11-repo && pkg install chromium`, or set browser.executable_path.")
    if BAKED_EXECUTABLE_FILE.exists():
        return ("The image's Chromium is missing: pull the current Hermes image and recreate the "
                "container (a volume mounted over the image's tool store hides it).")
    if _pm_has_no_chromium():
        return ("No Chromium found: pm builds none for this platform. Install your distribution's "
                "chromium package, or set browser.executable_path.")
    return "Chromium is not installed: run `hermes pm install chromium`."


def missing_shared_libraries(executable: str, *, platform: str | None = None) -> list[str]:
    """Shared objects the dynamic loader cannot resolve for *executable*, empty when none or unknown.

    A standalone Chrome download carries no libraries — ``libnspr4.so`` and friends are host-owned — so
    an installed binary can still exit 127 on launch (#148660). Only a resolved answer reports a
    problem: a non-Linux host, a missing/unreadable binary, no ``ldd``, or a loader that hangs all
    answer ``[]``, because hiding a working browser behind a warning is the worse failure.
    """
    if (platform or sys.platform) != "linux" or not os.path.isfile(executable):
        return []
    ldd = shutil.which("ldd")
    if ldd is None:
        return []
    try:
        result = subprocess.run([ldd, executable], capture_output=True, text=True, encoding="utf-8",
                                errors="replace", timeout=_LDD_TIMEOUT_S, check=False)
    except (OSError, subprocess.SubprocessError):  # not executable, killed, or a hung loader
        return []
    missing = []
    for line in (result.stdout or "").splitlines():
        name, separator, resolution = line.partition("=>")
        if separator and resolution.strip().startswith("not found") and name.strip():
            missing.append(name.strip())
    return missing


def system_library_install_command() -> str | None:
    """``python -m playwright install-deps chromium`` when this host can actually run it, else None."""
    if shutil.which("apt-get") is None:
        return None
    try:
        if importlib.util.find_spec("playwright") is None:
            return None
    except ImportError:  # a half-installed distribution: treat it as absent
        return None
    return shlex.join([sys.executable, "-m", "playwright", "install-deps", "chromium"])


def install_dependency_lines() -> list[str]:
    """Print-ready lines telling the user how to install Chromium's host libraries."""
    command = system_library_install_command()
    if command:
        return [f"Install missing system libraries with: {command}"]
    return ["System dependency installation guide:", f"  {SYSTEM_DEPENDENCY_GUIDE}"]


def local_browser_failure_advice(error: str) -> str:
    """What to do after a local Chromium launch failure, judged from the launcher's own error text.

    A missing shared object is the one case where ``hermes tools`` → Browser Automation cannot help:
    the archive is fine, the distribution's libraries are not.
    """
    if _MISSING_LIBRARY_MARKER in (error or ""):
        return ("The bundled Chromium is fine; this host is missing system libraries it needs. "
                + " ".join(line.strip() for line in install_dependency_lines())
                + " Then retry — reinstalling Chromium will not help.")
    return "Run `hermes tools` → Browser Automation to (re)install Chromium, or switch backends."

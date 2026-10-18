"""Install the host X11 client libraries cua-driver links against.

cua-driver-rs is a native binary. On headless Linux hosts (containers,
minimal servers) its ``--version`` probe dies on the FIRST missing X11
client library — the operator would otherwise chase libX11, then libXi,
then libXtst… one apt round per message. ``ldd`` enumerates the full set
in one pass; this module installs the distro packages that provide them
so the staged entry passes verification after a re-probe.

Mirrors pm/libatomic.py: the repair is installer-only
(``CuaDriver.repair_staged_verification``), never runs from
verify()/doctor/status, and uses ``sudo -n`` when it cannot be root
because PM holds its install lock during the repair.
"""

from __future__ import annotations

import shlex
import shutil
import subprocess
from pathlib import Path

# soname → distro package, per family. The sonames are stable across the
# families; only the package names differ.
_DEBIAN = {
    "libX11.so.6": "libx11-6",
    "libXau.so.6": "libxau6",
    "libXcursor.so.1": "libxcursor1",
    "libXdmcp.so.6": "libxdmcp6",
    "libXext.so.6": "libxext6",
    "libXfixes.so.3": "libxfixes3",
    "libXi.so.6": "libxi6",
    "libXinerama.so.1": "libxinerama1",
    "libXrandr.so.2": "libxrandr2",
    "libXrender.so.1": "libxrender1",
    "libXtst.so.6": "libxtst6",
    "libxkbcommon.so.0": "libxkbcommon0",
    "libxkbcommon-x11.so.0": "libxkbcommon-x11-0",
}
# Fedora/RHEL: same sonames, X.Org packages keep their upstream names (the
# Debian table must never be reused here — dnf aborts the whole transaction
# on a single unmatched spec). Verified on packages.fedoraproject.org:
# libXrandr provides libXrandr.so.2, libXcursor provides libXcursor.so.1, …
_FEDORA = {
    "libX11.so.6": "libX11",
    "libXau.so.6": "libXau",
    "libXcursor.so.1": "libXcursor",
    "libXdmcp.so.6": "libXdmcp",
    "libXext.so.6": "libXext",
    "libXfixes.so.3": "libXfixes",
    "libXi.so.6": "libXi",
    "libXinerama.so.1": "libXinerama",
    "libXrandr.so.2": "libXrandr",
    "libXrender.so.1": "libXrender",
    "libXtst.so.6": "libXtst",
    "libxkbcommon.so.0": "libxkbcommon",
    "libxkbcommon-x11.so.0": "libxkbcommon-x11",
}
_FAMILIES: dict[str, dict[str, str]] = {
    "apt-get": _DEBIAN,
    "dnf": _FEDORA,
    "yum": _FEDORA,
    "zypper": {"libX11.so.6": "libX11-6", "libXcursor.so.1": "libXcursor1",
               "libXi.so.6": "libXi6", "libXtst.so.6": "libXtst6",
               "libxkbcommon.so.0": "libxkbcommon0", "libxkbcommon-x11.so.0": "libxkbcommon-x11-0"},
    "pacman": {"libX11.so.6": "libx11", "libXcursor.so.1": "libxcursor",
               "libXi.so.6": "libxi", "libXtst.so.6": "libxtst",
               "libxkbcommon.so.0": "libxkbcommon", "libxkbcommon-x11.so.0": "libxkbcommon-x11"},
    "apk": {"libX11.so.6": "libx11", "libXcursor.so.1": "libxcursor",
            "libXi.so.6": "libxi", "libXtst.so.6": "libxtst",
            "libxkbcommon.so.0": "libxkbcommon"},
}

# One attempt per process (mirrors pm.libatomic._ATTEMPT): every root that
# depends on cua-driver would otherwise re-run the package manager while PM
# holds its install lock. A later process retries.
_ATTEMPT: tuple[bool, str] | None = None


def missing_shared_libraries(binary: Path) -> list[str]:
    """Sonames the loader cannot resolve for `binary`, via ldd.

    [] when ldd is unavailable or the listing is unreadable — callers then
    keep the original probe reason unchanged.
    """
    try:
        proc = subprocess.run(["ldd", str(binary)], capture_output=True, text=True,
                              timeout=60, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return []
    out: list[str] = []
    for line in ((proc.stdout or "") + (proc.stderr or "")).splitlines():
        if "not found" not in line or "=>" not in line:
            continue
        soname = line.split("=>", 1)[0].strip()
        if soname and soname not in out:
            out.append(soname)
    return out


def _packages_for(sonames: list[str]) -> tuple[str, list[str]]:
    """(manager, distro package list) for the first available package manager.

    Sonames the family table does not map are dropped from the command but
    stay in the operator-facing message verbatim.
    """
    from pm.libatomic import _manager_order, _release_tokens

    for manager in _manager_order(_release_tokens()):
        if shutil.which(manager):
            table = _FAMILIES.get(manager, {})
            return manager, [table[s] for s in sonames if s in table]
    return "", []


def _install_command(manager: str, pkgs: list[str]) -> tuple[str, ...]:
    # Command shapes mirror pm.libatomic._INSTALLERS (manager, install, packages…),
    # extended to a multi-package payload.
    return {
        "apt-get": ("apt-get", "install", "-y"),
        "dnf": ("dnf", "install", "-y"),
        "yum": ("yum", "install", "-y"),
        "zypper": ("zypper", "--non-interactive", "install"),
        "pacman": ("pacman", "-S", "--needed", "--noconfirm"),
        "apk": ("apk", "add"),
    }[manager] + tuple(pkgs)


def _manager_first() -> str:
    from pm.libatomic import _manager_order, _release_tokens

    for manager in _manager_order(_release_tokens()):
        if shutil.which(manager):
            return manager
    return ""


def try_install_x11_client_libs(sonames: list[str]) -> tuple[bool, str]:
    """Install the distro packages providing `sonames`; return (attempted_ok, hint).

    Success means the package-manager command completed. The staged-entry
    re-probe remains the authority and is always repeated by the caller.
    """
    global _ATTEMPT
    if _ATTEMPT is None:
        _ATTEMPT = _attempt(sonames)
    return _ATTEMPT


def _attempt(sonames: list[str]) -> tuple[bool, str]:
    from pm.libatomic import _command_plan, _install

    missing = [s for s in sonames if s]
    if not missing:
        return False, ""
    shown = ", ".join(missing)
    manager, pkgs = _packages_for(missing)
    hint = f"install the packages providing these libraries and rerun: {shown}"
    if not manager or not pkgs:
        return False, hint
    command = _install_command(manager, pkgs)
    argv, attempt_display, remedy_display = _command_plan(command)
    remedy = f"install the missing libraries ({shown}) with {remedy_display} and rerun"
    if argv is None:
        return False, remedy
    print(f"→ cua-driver needs host libraries ({shown}); "
          f"trying non-interactive {attempt_display}", flush=True)
    return _install(argv[: len(argv) - len(command)], command), remedy


def manual_remedy(binary: Path) -> str:
    """Human remedy for a shared-library probe failure, for non-repair callers."""
    sonames = missing_shared_libraries(binary)
    if not sonames:
        return ""
    _, pkgs = _packages_for(sonames)
    shown = ", ".join(sonames)
    if not pkgs:
        return f"install the packages providing these libraries and rerun: {shown}"
    manager = _manager_first()
    if not manager:
        return f"install the packages providing these libraries and rerun: {shown}"
    command = shlex.join(_install_command(manager, pkgs))
    as_root = True  # non-interactive repair uses sudo -n; a manual remedy may ask normally
    return (f"install the missing libraries ({shown}) with "
            f"{command if as_root else 'the distro package manager'} and rerun")
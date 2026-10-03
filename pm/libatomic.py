"""Install the host libatomic package required by official Linux Node.

This module is intentionally called only from PM's staged-install repair hook.
Package verification itself remains diagnostic and side-effect free.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from pathlib import Path


_INSTALLERS: dict[str, tuple[str, ...]] = {
    "apt-get": ("apt-get", "install", "-y", "libatomic1"),
    "dnf": ("dnf", "install", "-y", "libatomic"),
    "yum": ("yum", "install", "-y", "libatomic"),
    "zypper": ("zypper", "--non-interactive", "install", "libatomic1"),
    "pacman": ("pacman", "-S", "--needed", "--noconfirm", "gcc-libs"),
    "apk": ("apk", "add", "libatomic"),
}

_DEBIAN = {"debian", "ubuntu", "linuxmint", "pop", "raspbian", "kali", "elementary"}
_RPM = {"rhel", "fedora", "centos", "rocky", "almalinux", "ol", "oracle", "amzn"}
_ARCH = {"arch", "manjaro", "endeavouros", "garuda", "artix"}


def _release_tokens() -> set[str]:
    fields: dict[str, str] = {}
    for path in (Path("/etc/os-release"), Path("/usr/lib/os-release")):
        try:
            text = path.read_text(encoding="utf-8-sig")
        except OSError:
            continue
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            fields[key] = value.strip().strip('"').strip("'")
        break
    blob = f"{fields.get('ID', '')} {fields.get('ID_LIKE', '')}".lower()
    return set(blob.replace(",", " ").split())


def _manager_order(tokens: set[str]) -> tuple[str, ...]:
    if tokens & _DEBIAN:
        return ("apt-get",)
    if tokens & _RPM:
        return ("dnf", "yum")
    if any("suse" in token or token.startswith("sles") for token in tokens):
        return ("zypper",)
    if tokens & _ARCH:
        return ("pacman",)
    if "alpine" in tokens:
        return ("apk",)
    return ("dnf", "yum", "apt-get", "zypper", "pacman", "apk")


def _host_install_command() -> tuple[str, ...] | None:
    for manager in _manager_order(_release_tokens()):
        if shutil.which(manager):
            return _INSTALLERS[manager]
    return None


def _is_root() -> bool:
    return bool(hasattr(os, "geteuid") and os.geteuid() == 0)


def _command_plan(command: tuple[str, ...]) -> tuple[list[str] | None, str, str]:
    """Return auto-repair argv, its display form, and a runnable manual remedy.

    Host-package repair runs while PM owns its publication lock, so it must never
    wait on an interactive privilege prompt. Non-root repair therefore uses
    ``sudo -n`` even on a TTY. The manual remedy may use normal ``sudo``
    outside that lock. If sudo is unavailable, don't advertise it at all.
    """
    if _is_root():
        argv = list(command)
        shown = shlex.join(argv)
        return argv, shown, shown
    sudo = shutil.which("sudo")
    if sudo:
        argv = [sudo, "-n", *command]
        return argv, shlex.join(["sudo", "-n", *command]), shlex.join(["sudo", *command])
    return None, "", f"as root: {shlex.join(command)}"


def try_install_libatomic() -> tuple[bool, str]:
    """Try the distro-native libatomic package; return (attempt_succeeded, hint).

    Success means the package-manager command completed successfully. The Node
    smoke probe remains the authority and is always repeated by the caller.
    """
    command = _host_install_command()
    if command is None:
        return (
            False,
            "official Node needs libatomic.so.1; install the distro package that provides it and rerun",
        )

    argv, attempt_display, remedy_display = _command_plan(command)
    remedy = f"install the missing runtime library with {remedy_display} and rerun"
    if argv is None:
        return False, remedy

    env = dict(os.environ)
    env.setdefault("DEBIAN_FRONTEND", "noninteractive")
    print(
        f"→ Node needs libatomic.so.1; trying non-interactive {attempt_display}",
        flush=True,
    )
    try:
        result = subprocess.run(
            argv,
            stdin=subprocess.DEVNULL,
            env=env,
            timeout=300,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False, remedy
    return result.returncode == 0, remedy

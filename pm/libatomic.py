"""Install the host libatomic package required by official Linux Node.

This module is intentionally called only from PM's staged-install repair hook.
Package verification itself remains diagnostic and side-effect free.
"""

from __future__ import annotations

import os
import platform
import shlex
import shutil
import subprocess


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

# One attempt per process: every root that depends on node (npm, browser
# tools) re-stages it after a failure, and each would otherwise re-run the
# package manager under PM's install lock. A later process retries.
_ATTEMPT: tuple[bool, str] | None = None


def _release_tokens() -> set[str]:
    try:
        fields = platform.freedesktop_os_release()
    except OSError:
        return set()
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


def warm_sudo_before_install() -> None:
    """Let a terminal user authorize sudo before PM takes its install lock.

    The staged repair can only use ``sudo -n``, so on a password-sudo host it
    succeeds only with a cached ticket. Updates enter PM directly, without the
    shell installer's prerequisites stage, so cache the ticket here: only on a
    glibc Linux host still missing libatomic.so.1, non-root, interactive, and
    only when sudo would actually ask.
    """
    import ctypes
    import sys

    from pm.store import MUSL_TARGETS, current_target

    if not sys.platform.startswith("linux") or _is_root():
        return
    if not (sys.stdin and sys.stdin.isatty() and sys.stdout and sys.stdout.isatty()):
        return
    if current_target() in MUSL_TARGETS or current_target().endswith("-bionic"):
        return
    try:
        ctypes.CDLL("libatomic.so.1")
        return
    except OSError:
        pass
    sudo = shutil.which("sudo")
    if sudo is None or subprocess.run([sudo, "-n", "true"], stdin=subprocess.DEVNULL,
                                      capture_output=True, check=False).returncode == 0:
        return
    print("→ Node.js needs libatomic.so.1; sudo may ask for your password to install it", flush=True)
    try:
        subprocess.run([sudo, "-v"], timeout=300, check=False)
    except (OSError, subprocess.TimeoutExpired):
        pass


def _run(argv: list[str]) -> bool:
    try:
        return subprocess.run(argv, stdin=subprocess.DEVNULL, timeout=300, check=False).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def try_install_libatomic() -> tuple[bool, str]:
    """Try the distro-native libatomic package; return (attempt_succeeded, hint).

    Success means the package-manager command completed successfully. The Node
    smoke probe remains the authority and is always repeated by the caller.
    """
    global _ATTEMPT
    if _ATTEMPT is None:
        _ATTEMPT = _attempt()
    return _ATTEMPT


def _attempt() -> tuple[bool, str]:
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

    print(f"→ Node needs libatomic.so.1; trying non-interactive {attempt_display}", flush=True)
    if _run(argv):
        return True, remedy
    if command[0] != "apt-get":
        return False, remedy
    # Minimal Debian/Ubuntu images ship empty package lists.
    update = argv[: len(argv) - len(command)] + ["apt-get", "update", "-qq"]
    return _run(update) and _run(argv), remedy

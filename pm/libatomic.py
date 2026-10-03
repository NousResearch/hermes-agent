"""Host libatomic for official Node linux binaries.

nodejs.org linux builds link ``libatomic.so.1`` (Node 25+). Debian and
Ubuntu do not install ``libatomic1`` by default, and RHEL-family distros
ship the same library in a separate ``libatomic`` package. The dynamic
loader then exits 127, and PM rejects the staged Node. Install that
package once, then probe again.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

# Manager binary -> argv that installs the package providing libatomic.so.1.
# argv is without sudo; the displayed command adds sudo when we are not root.
_INSTALL = {
    "apt-get": ["apt-get", "install", "-y", "libatomic1"],
    "dnf": ["dnf", "install", "-y", "libatomic"],
    "yum": ["yum", "install", "-y", "libatomic"],
    "zypper": ["zypper", "--non-interactive", "install", "libatomic1"],
    "pacman": ["pacman", "-S", "--needed", "--noconfirm", "gcc-libs"],
    "apk": ["apk", "add", "libatomic"],
}

_DEB = {"debian", "ubuntu", "linuxmint", "pop", "raspbian", "kali", "elementary"}
_RPM = {
    "fedora", "rhel", "centos", "rocky", "almalinux", "ol", "oracle",
    "amzn", "nobara", "photon", "scientific",
}
_ARCH = {"arch", "manjaro", "endeavouros", "garuda", "artix"}
_FALLBACK = ("apt-get", "dnf", "yum", "zypper", "pacman", "apk")


def parse_os_release(text: str) -> dict[str, str]:
    """The KEY=VALUE fields of an os-release file. Quotes are stripped."""
    fields: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        fields[key] = value.strip().strip('"').strip("'")
    return fields


def _release_tokens(release: dict[str, str]) -> set[str]:
    blob = f"{release.get('ID', '')} {release.get('ID_LIKE', '')}".lower()
    return set(blob.replace(",", " ").split())


def preferred_manager(release: dict[str, str], available: list[str]) -> Optional[str]:
    """Which installer to use for this os-release, among binaries that exist.

    Family comes from ID / ID_LIKE so a host with both apt and dnf (rare,
    but Alma must not be handed Debian's ``libatomic1`` name) gets the
    package name its own manager ships. A known family never falls back
    to another family's manager. Unknown releases need one unambiguous
    manager; dnf and yum count as one RPM choice, with dnf preferred.
    """
    tokens = _release_tokens(release)
    wanted: list[str] = []
    if tokens & _DEB:
        wanted = ["apt-get"]
    elif tokens & _RPM:
        wanted = ["dnf", "yum"]
    elif any("suse" in token or token.startswith("sles") for token in tokens):
        wanted = ["zypper"]
    elif tokens & _ARCH:
        wanted = ["pacman"]
    elif "alpine" in tokens:
        wanted = ["apk"]
    present = set(available)
    for name in wanted:
        if name in present:
            return name
    if wanted:
        return None
    candidates = [name for name in _FALLBACK if name in present]
    if "dnf" in candidates and "yum" in candidates:
        candidates.remove("yum")
    return candidates[0] if len(candidates) == 1 else None


def read_os_release() -> dict[str, str]:
    for path in (Path("/etc/os-release"), Path("/usr/lib/os-release")):
        try:
            return parse_os_release(path.read_text(encoding="utf-8-sig"))
        except OSError:
            continue
    return {}


def available_managers() -> list[str]:
    return [name for name in _FALLBACK if shutil.which(name)]


def _is_root() -> bool:
    return hasattr(os, "geteuid") and os.geteuid() == 0  # windows-footgun: ok — Linux package install only


def install_command(manager: str, *, root: bool) -> str:
    """The command a person would type. ``sudo`` is omitted for root."""
    argv = list(_INSTALL[manager])
    if not root:
        argv.insert(0, "sudo")
    return " ".join(argv)


def remediation_for_host() -> str:
    """What to tell the user when Node still cannot load libatomic."""
    manager = preferred_manager(read_os_release(), available_managers())
    if manager is None:
        return (
            "official Node links libatomic.so.1, which is not installed; "
            "install the distro package that provides it and rerun `hermes update`"
        )
    command = install_command(manager, root=_is_root())
    return (
        "official Node links libatomic.so.1, which is not installed; "
        f"install it with `{command}` and rerun `hermes update`"
    )


def libatomic_present() -> bool:
    """True when the dynamic loader can open libatomic.so.1.

    Non-Linux hosts do not use this library for official Node. A failed
    load is the same condition as Node's own exit 127.
    """
    if not sys.platform.startswith("linux"):
        return True
    try:
        ctypes.CDLL("libatomic.so.1")
    except OSError:
        return False
    return True


def _stdin_is_tty() -> bool:
    stdin = sys.stdin
    if stdin is None or not hasattr(stdin, "isatty"):
        return False
    try:
        return bool(stdin.isatty())
    except (ValueError, OSError):
        return False


def _sudo_noninteractive_ok() -> bool:
    if shutil.which("sudo") is None:
        return False
    try:
        proc = subprocess.run(
            ["sudo", "-n", "true"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0


def privileged_argv(argv: list[str]) -> Optional[list[str]]:
    """argv to run now, or None when we must not block waiting for a password.

    Passwordless sudo (and root) run immediately. A terminal gets a normal
    ``sudo`` so the update can finish the way a person would. Anywhere else,
    ``sudo`` would sit on a password prompt until the update is killed.
    """
    if _is_root():
        return list(argv)
    if shutil.which("sudo") is None:
        return None
    if _sudo_noninteractive_ok():
        return ["sudo", "-n", *argv]
    if _stdin_is_tty():
        return ["sudo", *argv]
    return None


def _run_install(argv: list[str]) -> None:
    env = os.environ.copy()
    env.setdefault("DEBIAN_FRONTEND", "noninteractive")
    # ``sudo -n`` must not consume a prompt. A real sudo on a terminal has to
    # read the password from the user's stdin; closing it makes the update fail
    # again at the exact moment someone could have typed it.
    interactive = argv[:1] == ["sudo"] and "-n" not in argv[1:3]
    subprocess.run(
        argv,
        stdin=None if interactive else subprocess.DEVNULL,
        timeout=300,
        env=env,
        check=False,
    )


def try_install_libatomic() -> bool:
    """Install the host package when libatomic.so.1 will not load.

    Returns whether the loader can open it afterwards. Never raises: a
    failed install leaves the probe error to carry the command instead.
    """
    if libatomic_present():
        return True
    manager = preferred_manager(read_os_release(), available_managers())
    if manager is None:
        return False
    argv = privileged_argv(_INSTALL[manager])
    if argv is None:
        return False
    print(f"→ Node needs libatomic.so.1; trying `{' '.join(argv)}`", flush=True)
    try:
        _run_install(argv)
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"→ Could not install libatomic.so.1 ({exc})", flush=True)
        return False
    if libatomic_present():
        print("✓ libatomic.so.1 is available", flush=True)
        return True
    return False


def should_repair_node_libatomic(*, target: str, host: str, reason: str) -> bool:
    """Only the native glibc probe runs the binary, so only it can be repaired."""
    return (
        "libatomic.so.1" in reason
        and target == host
        and target.startswith("linux")
        and not target.endswith("-bionic")
    )


def repair_node_libatomic_failure(
    reason: str,
    *,
    target: str,
    host: str,
    retry: Callable[[], str],
) -> str:
    """After a Node smoke probe dies on libatomic, install it and probe once more."""
    if not reason or not should_repair_node_libatomic(target=target, host=host, reason=reason):
        return reason
    if try_install_libatomic():
        reason = retry()
        if "libatomic.so.1" not in reason:
            return reason
    hint = remediation_for_host()
    if hint and hint not in reason:
        return f"{reason} — {hint}"
    return reason

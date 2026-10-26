"""Chromium's Linux sandbox under AppArmor's unprivileged-userns restriction (#130177).

With ``kernel.apparmor_restrict_unprivileged_userns=1`` (Ubuntu 23.10+), any unconfined program
may still create a user namespace, but it loses every capability inside it unless an AppArmor
profile attached to that program's path grants ``userns``. ``unshare --user true`` therefore
succeeds while Chromium, which needs those capabilities, aborts with ``No usable sandbox!``.
Only the Hermes binary itself, under whatever profile attaches to its path, can answer whether
its namespace sandbox starts, so on a restricted host the launcher asks that binary.
"""

from __future__ import annotations

import hashlib
import mmap
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

RESTRICT_SYSCTL = Path("/proc/sys/kernel/apparmor_restrict_unprivileged_userns")
# Answered by apps/desktop/electron/entry.ts before any window, lock or backend exists.
SANDBOX_PROBE_SWITCH = "--hermes-sandbox-probe"
_PROBE_TIMEOUT_S = 15
_SUDO_TIMEOUT_S = 300  # room for a typed password, never an unbounded hang
_PROFILE_DIR = Path("/etc/apparmor.d")


def apparmor_restricts_userns() -> bool:
    try:
        return RESTRICT_SYSCTL.read_text(encoding="utf-8-sig").strip() == "1"
    except OSError:
        return False


def _build_answers_probe(packaged_executable: Path) -> bool:
    """True when this packaged build's entry point honours the probe switch.

    A build from before the switch would start the whole app instead of exiting, so the probe is
    only sent to a build whose main bundle carries the switch literal. electron-builder unpacks
    ``dist/`` beside the asar; a fully packed asar holds the bundle verbatim.
    """
    resources = packaged_executable.parent / "resources"
    needle = SANDBOX_PROBE_SWITCH.encode()
    for bundle in (resources / "app.asar.unpacked" / "dist" / "electron-main.mjs", resources / "app.asar"):
        try:
            with bundle.open("rb") as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as data:
                if data.find(needle) != -1:
                    return True
        except (OSError, ValueError):
            continue
    return False


def chromium_userns_sandbox_verdict(packaged_executable: Path) -> Optional[bool]:
    """Whether Chromium's namespace sandbox starts for *packaged_executable* on this host.

    ``None`` when the host has no AppArmor userns restriction or the build cannot answer the
    probe; the caller then keeps its generic probe. Chromium creates its sandboxed zygote before
    any app JavaScript runs, so a probe that reaches the entry point exits 0 and a blocked one
    dies with ``No usable sandbox!``. A probe that hangs or cannot start counts as blocked.
    """
    if sys.platform != "linux" or not apparmor_restricts_userns():
        return None
    if not _build_answers_probe(packaged_executable):
        return None
    env = {key: value for key, value in os.environ.items() if key != "ELECTRON_RUN_AS_NODE"}
    try:
        completed = subprocess.run(
            [str(packaged_executable), "--disable-setuid-sandbox", SANDBOX_PROBE_SWITCH],
            cwd=packaged_executable.parent, env=env, stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=_PROBE_TIMEOUT_S, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


def profile_name(packaged_executable: Path) -> str:
    """One profile per install path, so two checkouts never overwrite each other's grant."""
    digest = hashlib.sha256(str(packaged_executable.resolve()).encode()).hexdigest()[:10]
    return f"hermes-desktop-{digest}"


def profile_text(packaged_executable: Path) -> str:
    """The profile Chrome and VS Code ship for this case: unconfined, plus the ``userns`` grant."""
    name = profile_name(packaged_executable)
    exe = str(packaged_executable.resolve()).replace("\\", "\\\\").replace('"', '\\"')
    return (
        "# Hermes Desktop: let Chromium's user-namespace sandbox start under\n"
        "# kernel.apparmor_restrict_unprivileged_userns=1 (written by `hermes desktop --install-apparmor-profile`).\n"
        "abi <abi/4.0>,\n"
        "include <tunables/global>\n\n"
        f'profile {name} "{exe}" flags=(unconfined) {{\n'
        "  userns,\n\n"
        f"  include if exists <local/{name}>\n"
        "}\n"
    )


def profile_path(packaged_executable: Path) -> Path:
    return _PROFILE_DIR / profile_name(packaged_executable)


def install_profile(packaged_executable: Path) -> bool:
    """Write and load the profile through sudo (prompting on a terminal). Prints the outcome."""
    target = profile_path(packaged_executable)
    from hermes_platform.resolver import locate_command

    sudo = next(iter(locate_command("sudo").command), None)
    parser = next(iter(locate_command("apparmor_parser", known_dirs=("/usr/sbin", "/sbin")).command),
                  "apparmor_parser")
    if not sudo:
        print("✗ Installing the AppArmor profile needs sudo, which was not found.")
        print(f"  As root, write {target} with:\n\n{profile_text(packaged_executable)}")
        print(f"  then load it: apparmor_parser -r {target}")
        return False
    print(f"→ Installing AppArmor profile {target} (sudo required)...")
    steps = (
        ([sudo, "tee", str(target)], profile_text(packaged_executable)),
        ([sudo, parser, "-r", str(target)], None),
    )
    for command, stdin_text in steps:
        try:
            completed = subprocess.run(
                command, input=stdin_text, text=True, check=False, timeout=_SUDO_TIMEOUT_S,
                stdout=subprocess.DEVNULL if stdin_text is not None else None)
        except (OSError, subprocess.TimeoutExpired) as exc:
            print(f"✗ {' '.join(command)} failed: {exc}")
            return False
        if completed.returncode != 0:
            print(f"✗ {' '.join(command)} exited {completed.returncode}")
            return False
    verdict = chromium_userns_sandbox_verdict(packaged_executable)
    if verdict is False:
        print("✗ The profile is loaded, but Chromium's sandbox still cannot start for this binary.")
        return False
    print(f"✓ AppArmor profile installed; Hermes Desktop can use Chromium's user-namespace sandbox ({target}).")
    return True


def print_blocked_remedies(packaged_executable: Path) -> None:
    """The actionable lines for a restricted host where the setuid helper could not be set up."""
    helper = packaged_executable.parent / "chrome-sandbox"
    print("  Ubuntu's AppArmor policy (kernel.apparmor_restrict_unprivileged_userns=1) blocks Chromium's")
    print("  user-namespace sandbox for Hermes Desktop. To keep the sandbox on, run ONE of these once:")
    print("    hermes desktop --install-apparmor-profile    (AppArmor profile for this app; survives rebuilds)")
    print(f"    sudo chown root:root {helper} && sudo chmod 4755 {helper}    (redo after each rebuild)")

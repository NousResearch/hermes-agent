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
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

RESTRICT_SYSCTL = Path("/proc/sys/kernel/apparmor_restrict_unprivileged_userns")
# Answered by apps/desktop/electron/entry.ts before any window, lock or backend exists.
SANDBOX_PROBE_SWITCH = "--hermes-sandbox-probe"
_PROBE_TIMEOUT_S = 15
_SUDO_TIMEOUT_S = 300  # room for a typed password, never an unbounded hang
_PROFILE_DIR = Path("/etc/apparmor.d")
# apparmor.d(5) "Globbing (AARE)": ``?*[]{}^`` stay glob syntax inside a quoted attachment path, so
# they would widen the grant to other binaries or fail the parse; ``{`` also opens ``@{var}``
# expansion, and ``"``/``\`` end or escape the quoted string. Such paths are refused, not escaped.
_AARE_UNSAFE = frozenset('?*[]{}^"\\')


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
    exe = str(packaged_executable.resolve())
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


def unsafe_path_characters(packaged_executable: Path) -> str:
    """The characters of the install path an AppArmor attachment cannot carry literally."""
    path = str(packaged_executable.resolve())
    return "".join(sorted({ch for ch in path if ch in _AARE_UNSAFE or not ch.isprintable()}))


def _user_can_replace(packaged_executable: Path) -> bool:
    exe = packaged_executable.resolve()
    return os.access(exe, os.W_OK) or any(os.access(parent, os.W_OK) for parent in exe.parents)


def _run(command: list[str], stdin_text: Optional[str] = None) -> Optional[subprocess.CompletedProcess]:
    """Run one install step; ``None`` (after printing why) when it could not start or timed out."""
    try:
        return subprocess.run(
            command, input=stdin_text, text=True, encoding="utf-8", errors="replace",
            check=False, timeout=_SUDO_TIMEOUT_S,
            stdout=subprocess.DEVNULL if stdin_text is not None else None)
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"✗ {' '.join(command)} failed: {exc}")
        return None


def _profile_parses(parser: str, text: str) -> bool:
    """Compile *text* as this user without loading it, so a bad profile never reaches /etc."""
    from hermes_constants import get_scratch_dir

    with tempfile.TemporaryDirectory(dir=get_scratch_dir(), prefix="apparmor-") as scratch:
        candidate = Path(scratch) / "hermes-desktop.profile"
        candidate.write_text(text, encoding="utf-8")
        completed = _run([parser, "--skip-kernel-load", "--skip-cache", str(candidate)])
    if completed is None:
        print("✗ Could not check the generated profile; nothing was written to /etc/apparmor.d.")
        return False
    if completed.returncode != 0:
        print("✗ apparmor_parser rejected the generated profile; nothing was written to /etc/apparmor.d.")
        return False
    return True


def _write_and_load(sudo: str, parser: str, target: Path, text: str) -> bool:
    """Stage the profile beside ``target``, load it, and only then rename it into place, so a
    failed write or load leaves an already-installed profile intact. The stage is a dotfile,
    which apparmor_parser skips when it loads the directory at boot."""
    staged = target.with_name(f".{target.name}.hermes-new")
    failure = ""
    written = _run([sudo, "tee", str(staged)], text)
    if written is None or written.returncode != 0:
        failure = f"✗ Could not write {target}."
    else:
        loaded = _run([sudo, parser, "-r", str(staged)])
        if loaded is None or loaded.returncode != 0:
            failure = f"✗ apparmor_parser could not load the profile for {target}."
        else:
            moved = _run([sudo, "mv", "-f", str(staged), str(target)])
            if moved is not None and moved.returncode == 0:
                return True
            failure = f"✗ The profile is loaded until reboot, but moving it to {target} failed."
    removed = _run([sudo, "rm", "-f", str(staged)])
    print(failure + (" Any profile already there is unchanged."
                     if removed is not None and removed.returncode == 0 else f" Remove {staged} with sudo rm."))
    return False


def install_profile(packaged_executable: Path) -> bool:
    """Write and load the profile through sudo (prompting on a terminal). Prints the outcome."""
    if sys.platform != "linux":
        print("✗ --install-apparmor-profile is Linux only: AppArmor profiles do not exist on this OS.")
        return False
    target = profile_path(packaged_executable)
    unsafe = unsafe_path_characters(packaged_executable)
    if unsafe:
        print(f"✗ The install path {packaged_executable.resolve()} contains {unsafe!r}, which AppArmor reads as")
        print("  pattern syntax (apparmor.d(5), Globbing), so a profile could not name this binary alone.")
        print("  Move the checkout to a path without these characters, or use the setuid chrome-sandbox helper.")
        return False
    from hermes_platform.resolver import locate_command

    sudo = next(iter(locate_command("sudo").command), None)
    parser = next(iter(locate_command("apparmor_parser", known_dirs=("/usr/sbin", "/sbin")).command),
                  "apparmor_parser")
    text = profile_text(packaged_executable)
    if _user_can_replace(packaged_executable):
        print("⚠ This profile lets whatever program sits at this path, which your account can replace, use user"
              " namespaces, so any code running as you gains what the restriction withholds; the setuid"
              " chrome-sandbox helper avoids that.")
    if not sudo:
        print("✗ Installing the AppArmor profile needs sudo, which was not found.")
        print(f"  As root, write {target} with:\n\n{text}")
        print(f"  then load it: apparmor_parser -r {target}")
        return False
    if not _profile_parses(parser, text):
        return False
    print(f"→ Installing AppArmor profile {target} (sudo required)...")
    if not _write_and_load(sudo, parser, target, text):
        return False
    verdict = chromium_userns_sandbox_verdict(packaged_executable)
    if verdict is False:
        print("✗ The profile is loaded, but Chromium's sandbox still cannot start for this binary.")
        return False
    print(f"✓ AppArmor profile installed; Hermes Desktop can use Chromium's user-namespace sandbox ({target}).")
    return True


def print_no_sandbox_fallback(packaged_executable: Path) -> None:
    """Why the launch falls back to ``--no-sandbox``; the AppArmor fix-it lines only on a host whose
    userns restriction is the reason (``ELECTRON_DISABLE_SANDBOX=1`` alone is the user's choice)."""
    if not apparmor_restricts_userns():
        print("⚠ Falling back to --no-sandbox because ELECTRON_DISABLE_SANDBOX=1 is set and the Electron"
              " sandbox helper could not be configured.")
        return
    print("⚠ Falling back to --no-sandbox because this Linux host restricts unprivileged user namespaces"
          " and the Electron sandbox helper could not be configured.")
    helper = shlex.quote(str(packaged_executable.parent / "chrome-sandbox"))
    print("  Ubuntu's AppArmor policy (kernel.apparmor_restrict_unprivileged_userns=1) blocks Chromium's")
    print("  user-namespace sandbox for Hermes Desktop. To keep the sandbox on, run ONE of these:")
    print(f"    sudo chown root:root {helper} && sudo chmod 4755 {helper}")
    print("        Chromium's setuid sandbox helper (recommended; redo after each rebuild)")
    print("    hermes desktop --install-apparmor-profile")
    print("        AppArmor profile for this path (survives rebuilds; any program running as you can use it)")

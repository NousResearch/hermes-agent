"""pm.shell(): the one place Hermes resolves the shell it runs commands with.

Owned by pm because the shell is a bundled tool on Windows (Git for Windows
carries bash.exe), and the store is the authority on whether it exists.
Callers that need bash (the terminal backend, `_find_bash`) call this instead
of hunting fixed locations.

Resolution order:
  1. Windows: the git Package's staged bash (via facts.json) — the store
     structurally guarantees it in a bundle; no hunt.
  2. Windows: explicit override, Program Files, per-user and PortableGit,
     then PATH —
     minus C:\\Windows\\System32\bash.exe (the WSL launcher stub, first on PATH
     on most machines; #116818) and WindowsApps\bash.exe (an MSIX alias that
     only spawns inside its package). See windows_bash_candidates().
  3. Provisioned PATH: shutil.which("bash") — the store dirs are on the
     process PATH after pm.activate() ran.
  4. POSIX fallback table for non-bundle / daemon-launch PATH edge cases
     (/usr/bin/bash, /bin/bash, $SHELL, /bin/sh). A systemd/cron-launched
     gateway may have a minimal PATH and macOS /bin/bash is not on PATH by
     default, so `which` alone is not enough there.
"""

from __future__ import annotations

import ntpath
import os
import shutil
import subprocess
from collections.abc import Mapping

from pm import paths
from pm.lock import Facts


def _staged_bash() -> str | None:
    """The git Package's bash.exe under the store (Windows bundles only)."""
    import platform

    if platform.system() != "Windows":
        return None
    facts_path = paths.facts_path()
    if not facts_path.is_file():
        return None
    try:
        facts = Facts(facts_path)
        fact = facts.get("git")
        if fact is None or "entry" not in fact:
            return None
        entry = paths.store_root() / fact["entry"]
        for candidate in (
            entry / "usr" / "bin" / "bash.exe",
            entry / "bin" / "bash.exe",
        ):
            if candidate.is_file():
                return str(candidate)
    except Exception:
        return None
    return None


# PATH dirs whose bash.exe is not a shell: System32 holds the WSL launcher
# stub (prints "no installed distributions", exits 1) and WindowsApps holds
# MSIX execution aliases that fail with WinError 5 from an arbitrary process.
_WINDOWS_BASH_STUB_DIRS = ("system32", "windowsapps")


def windows_bash_candidates(on_path: str | None, env: Mapping[str, str]) -> list[str]:
    """Ordered bash.exe candidates for a Windows host, as pure data: the
    explicit override and Git for Windows roots first, then ``on_path``
    (the ``shutil.which("bash")`` result) unless it is a stub. System32
    precedes Git on most PATHs (#116818)."""
    programfiles = env.get("ProgramFiles", r"C:\Program Files")
    candidates = []
    if env.get("HERMES_GIT_BASH_PATH"):
        candidates.append(env["HERMES_GIT_BASH_PATH"])
    roots = [ntpath.join(programfiles, "Git")]
    if env.get("ProgramFiles(x86)"):
        roots.append(ntpath.join(env["ProgramFiles(x86)"], "Git"))
    if env.get("LOCALAPPDATA"):
        roots.extend((ntpath.join(env["LOCALAPPDATA"], "hermes", "git"),
                      ntpath.join(env["LOCALAPPDATA"], "Programs", "Git")))
    for root in roots:
        candidates.extend((ntpath.join(root, "bin", "bash.exe"),
                           ntpath.join(root, "usr", "bin", "bash.exe")))
    if on_path:
        norm = ntpath.normpath(on_path).lower()
        if not any(stub in norm for stub in _WINDOWS_BASH_STUB_DIRS):
            candidates.append(on_path)
    return list(dict.fromkeys(candidates))


def _bash_starts(candidate: str) -> bool:
    """An existing bash.exe can still be broken or be a launcher stub.

    The budget is deliberately generous: under host memory pressure an MSYS
    bash start can take 7-13 s while native spawns stay sub-second, so a
    tight timeout misclassifies a healthy bash as broken and bash() returns
    None (#133756). The budget is env-tunable and a transient timeout gets
    one retry; a candidate that starts and exits non-zero within the budget
    is still rejected on the first attempt.
    """
    try:
        timeout = float(os.environ.get("HERMES_BASH_PROBE_TIMEOUT", "15") or "15")
    except ValueError:
        timeout = 15.0
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    for attempt in (1, 2):
        try:
            probe = subprocess.run(
                [candidate, "-c", "exit 0"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=timeout,
                creationflags=flags,
                check=False,
            )
            return probe.returncode == 0
        except subprocess.TimeoutExpired:
            if attempt == 2:
                return False
        except OSError:
            return False
    return False


def bash() -> str | None:
    """Resolve the bash binary to use, or None if none is available."""
    staged = _staged_bash()
    if staged and _bash_starts(staged):
        return staged

    on_path = shutil.which("bash")
    if os.name == "nt":
        return next((c for c in windows_bash_candidates(on_path, os.environ)
                     if os.path.isfile(c) and _bash_starts(c)), None)
    if on_path:
        return on_path

    # POSIX fallbacks for minimal-PATH daemon launches / macOS /bin/bash.
    for candidate in (
        "/usr/bin/bash",
        "/bin/bash",
        os.environ.get("SHELL"),
        "/bin/sh",
    ):
        if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None

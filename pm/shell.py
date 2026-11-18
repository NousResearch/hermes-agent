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


def _is_stub_bash(path: str) -> bool:
    """True for a bash.exe that is a launcher stub for another execution host."""
    norm = ntpath.normpath(path).lower()
    return any(stub in norm for stub in _WINDOWS_BASH_STUB_DIRS)


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
        if not _is_stub_bash(on_path):
            candidates.append(on_path)
    return list(dict.fromkeys(candidates))


# ``_bash_starts`` spawns the binary, and the terminal asks for the shell before every
# command: the probe is remembered against the file's identity, so only a bash.exe that
# actually changed is worth 100ms. Keyed by the resolved path (a monkeypatched PATH or
# ProgramFiles yields a different file, hence a different key) plus its stat.
_bash_starts_cache: dict[tuple[str, int, int], bool] = {}


def _bash_starts(candidate: str) -> bool:
    """An existing bash.exe can still be broken or be a launcher stub."""
    try:
        info = os.stat(candidate)
    except OSError:
        # Missing (or unreadable) is the same verdict the spawn would reach, without it.
        return False
    key = (os.path.normcase(os.path.abspath(candidate)), info.st_mtime_ns, info.st_size)
    known = _bash_starts_cache.get(key)
    if known is not None:
        return known
    try:
        starts = subprocess.run(
            [candidate, "-c", "exit 0"], stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=5, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            check=False,
        ).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        starts = False
    if len(_bash_starts_cache) >= 32:
        _bash_starts_cache.clear()
    _bash_starts_cache[key] = starts
    return starts


# The resolved shell, with the environment it was resolved from. ``shutil.which`` walks
# every PATH dir, so an answer is reused until the PATH (or the Git roots, or the store's
# facts, all of which it reads) changes.
_bash_cache: tuple[tuple, str | None] | None = None


def forget_bash() -> None:
    """Drop the memoized shell: the state it resolved from changed (a tool was installed)."""
    global _bash_cache
    _bash_cache = None


def bash() -> str | None:
    """Resolve the bash binary to use, or None if none is available."""
    global _bash_cache
    key = (os.name, os.environ.get("PATH"), os.environ.get("ProgramFiles"),
           os.environ.get("ProgramFiles(x86)"), os.environ.get("LOCALAPPDATA"),
           os.environ.get("HERMES_GIT_BASH_PATH"), os.environ.get("SHELL"),
           os.environ.get("HERMES_RUNTIME_DIR"), os.environ.get("HERMES_HOME"),
           _bash_starts, _staged_bash, paths.facts_path, paths.store_root, shutil.which)
    if _bash_cache is not None and _bash_cache[0] == key:
        return _bash_cache[1]
    resolved = _resolve_bash()
    _bash_cache = (key, resolved)
    return resolved


def _resolve_bash() -> str | None:
    staged = _staged_bash()
    if staged and _bash_starts(staged):
        return staged

    if os.name == "nt":
        # Git for Windows roots first, the PATH entry last: ``windows_bash_candidates``
        # orders it that way, and ``shutil.which`` walks every PATH dir to find it.
        for candidate in windows_bash_candidates(None, os.environ):
            if os.path.isfile(candidate) and _bash_starts(candidate):
                return candidate
        on_path = shutil.which("bash")
        if on_path and not _is_stub_bash(on_path) and _bash_starts(on_path):
            return on_path
        return None

    on_path = shutil.which("bash")
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

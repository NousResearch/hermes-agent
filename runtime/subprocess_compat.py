"""Shared subprocess platform, launch, and bounded-capture compatibility helpers.

These primitives are runtime infrastructure: they describe host subprocess behavior without
depending on CLI orchestration or Gateway policy. General process-tree lifecycle remains a
separate ownership concern.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from typing import Mapping, Sequence


__all__ = [
    "IS_WINDOWS",
    "bounded_probe_run",
    "resolve_node_command",
    "split_command_line",
    "suppress_platform_ver_console",
    "windows_detach_flags",
    "windows_detach_flags_without_breakaway",
    "windows_hide_flags",
    "windows_detach_popen_kwargs",
]


IS_WINDOWS = sys.platform == "win32"


def split_command_line(line: str) -> list[str]:
    """Split a user-supplied command line into tokens, Windows-safely.

    ``shlex.split`` (posix=True) treats every backslash as an escape, mangling Windows paths. On
    Windows use ``posix=False`` and strip one layer of matching quotes per token; on POSIX this is
    exactly ``shlex.split``. Raises ValueError on unbalanced quotes.

    ``shlex.split(line)`` (posix=True) treats every backslash as an escape character, so Windows paths are
    silently mangled: ``C:\\Users\\me\\out.txt`` becomes ``C:Usersmeout.txt`` — no error, just a wrong path
    that then "succeeds" against a mangled relative filename (#83934) or makes a valid hook script report
    "not executable" (#78293).
    """
    import shlex

    if not IS_WINDOWS:
        return shlex.split(line)
    out: list[str] = []
    for tok in shlex.split(line, posix=False):
        if len(tok) >= 2 and tok[0] == tok[-1] and tok[0] in ("'", '"'):
            tok = tok[1:-1]
        out.append(tok)
    return out


def resolve_node_command(name: str, argv: Sequence[str]) -> list[str]:
    """Resolve a Node-ecosystem command name (``npm``, ``npx``, ``yarn``…) to an absolute-path argv.

    On Windows these ship as ``.cmd`` batch shims that CreateProcessW won't execute by bare name;
    ``shutil.which`` resolves via PATHEXT to a fully-qualified path whose extension routes it
    through ``cmd.exe /c``.
    """
    resolved = shutil.which(name)
    return [resolved or name, *argv]


# Win32 CreationFlags — defined here because CREATE_NO_WINDOW / DETACHED_PROCESS aren't guaranteed
# to exist on stdlib subprocess for older Pythons or non-Windows builds.
_CREATE_NEW_PROCESS_GROUP = 0x00000200
# DETACHED_PROCESS (0x00000008) is intentionally NOT part of any flag bundle — do not re-add it
# (the recurring console-flash bug #54220 / #56747): (1) MSDN: CREATE_NO_WINDOW "is ignored if used with either
# CREATE_NEW_CONSOLE or DETACHED_PROCESS"; (2) a DETACHED_PROCESS child has NO console, so every
# console-subsystem descendant (git, gh, cmd, node, powershell, …) allocates its own — a visible
# flash per spawn, including inside third-party libraries no per-site sweep can reach. A
# CREATE_NO_WINDOW child instead OWNS a hidden console all descendants inherit (A/B verified on
# Windows 11 by the desktop backend fix, commit aa2ae36c3f: with per-site hide flags neutered,
# naive git/gh/cmd spawns don't flash under a hidden-console parent and do under a console-less one).
# 1. Combining them means DETACHED_PROCESS governs and the no-window bit is dead. 2. See #54220, #56747.
_CREATE_NO_WINDOW = 0x08000000
# Escape any Win32 job object the parent belongs to. Without this a detached child inherits the
# parent's job, and when that parent (Electron, Tauri, Windows Terminal, the Desktop bootstrap
# installer) dies the OS tears down the whole job — taking the "detached" child with it. Critical
# for the post-update gateway watcher spawned from inside Electron's job.
_CREATE_BREAKAWAY_FROM_JOB = 0x01000000


def windows_detach_flags() -> int:
    """Win32 creationflags detaching a child from the parent console/group; 0 elsewhere.

    Pair with the default ``start_new_session=False`` (POSIX uses ``start_new_session=True``).
    CREATE_NEW_PROCESS_GROUP stops Ctrl+C propagating; CREATE_NO_WINDOW gives the child a hidden
    console descendants (git, gh, cmd, node, …) inherit so they don't flash — deliberately replacing
    the old DETACHED_PROCESS approach, which re-created the per-descendant console-flash bug
    (#54220/#56747) at every spawn; CREATE_BREAKAWAY_FROM_JOB escapes Electron/Tauri job objects. A
    job that forbids breakaway yields PermissionError from Popen — callers catch OSError and fall
    back to :func:`windows_detach_flags_without_breakaway`.

    Rationale: This both detaches it from the parent's console lifetime (closing the launching terminal
    doesn't CTRL_CLOSE it) AND gives every console-subsystem descendant (git, gh, cmd, node, …) a console to
    inherit, so they don't allocate visible flashing ones. This deliberately replaces the old
    ``DETACHED_PROCESS`` approach: MSDN specifies CREATE_NO_WINDOW is *ignored* when combined with
    DETACHED_PROCESS, and a truly console-less daemon re-creates the per-descendant console-flash bug
    (#54220/#56747) at every spawn — see the note on ``_DETACHED_PROCESS`` above. Electron (Desktop app) and
    Tauri (bootstrap installer) wrap their children in job objects; without breakaway, those children die
    when the parent process exits even though they have their own console. This was the missing flag that
    made the post-update gateway respawn watcher silently die alongside the Tauri updater after the Electron
    Desktop's update flow finished.
    """
    if not IS_WINDOWS:
        return 0
    return _CREATE_NEW_PROCESS_GROUP | _CREATE_NO_WINDOW | _CREATE_BREAKAWAY_FROM_JOB


def windows_detach_flags_without_breakaway() -> int:
    """:func:`windows_detach_flags` minus ``CREATE_BREAKAWAY_FROM_JOB``; 0 on non-Windows."""
    if not IS_WINDOWS:
        return 0
    return _CREATE_NEW_PROCESS_GROUP | _CREATE_NO_WINDOW


def windows_hide_flags() -> int:
    """Win32 creationflags hiding the child's console without detaching it; 0 elsewhere.

    For short-lived synchronous helpers (``taskkill``, ``where``, version probes): no flash, but the
    child stays in the parent's process group and job so Ctrl+C and job teardown still propagate.
    Stdio is inherited, so ``capture_output=True`` works.
    """
    return _CREATE_NO_WINDOW if IS_WINDOWS else 0


def suppress_platform_ver_console() -> None:
    """Stub ``platform._syscmd_ver`` on Windows so it never flashes a console. No-op elsewhere.

    ``platform.win32_ver()`` shells out ``cmd /c ver`` without CREATE_NO_WINDOW, so a windowless
    parent (pythonw gateway, kanban workers) flashes a cmd window whenever a dependency touches
    ``platform.uname()`` at import. With the stub, ``win32_ver()`` takes its documented fallback to
    ``sys.getwindowsversion()`` — same data, in-process. Call before heavy imports.
    """
    if not IS_WINDOWS:
        return
    try:
        import platform

        if hasattr(platform, "_syscmd_ver"):
            def _quiet_syscmd_ver(system="", release="", version="",
                                  supported_platforms=("win32", "win16", "dos")):
                return system, release, version

            platform._syscmd_ver = _quiet_syscmd_ver
    except Exception:
        pass  # Purely cosmetic hardening — never let it break startup.


def windows_detach_popen_kwargs() -> dict:
    """Popen kwargs detaching a child on Windows, or ``start_new_session=True`` on POSIX.

    Bare ``start_new_session=True`` is accepted but has no effect on Windows: the child stays
    attached to the parent console and dies when it closes.
    """
    if IS_WINDOWS:
        return {"creationflags": windows_detach_flags()}
    return {"start_new_session": True}


def _close_job(job) -> None:
    if job is None:
        return
    try:
        job.close()
    except Exception:
        pass


def bounded_probe_run(
    argv: Sequence[str], *, timeout: float, errors: str = "replace",
    env: "Mapping[str, str] | None" = None, cwd: "str | os.PathLike[str] | None" = None,
    raise_on_spawn_failure: bool = False,
) -> "subprocess.CompletedProcess[str] | None":
    """Deadlock-safe ``subprocess.run(argv, capture_output=True, timeout=.)`` for fail-open probes.

    Returns a ``CompletedProcess`` when the child finished within *timeout* (any exit code), or
    ``None`` on spawn failure or timeout. With ``raise_on_spawn_failure=True`` the ``Popen``
    exception propagates instead, so callers that treat a *timeout* as a verdict can still tell
    "our own probe never started" apart from "the child hung".

    Why not ``subprocess.run``: on Windows, ``run()``'s post-timeout cleanup calls an *unbounded*
    ``communicate()`` after killing the direct child. Killing it can leave a descendant (``git.exe`` under a
    launcher shim, ``conhost.exe`` under wmic/powershell) holding duplicates of the captured stdout/stderr
    handles, so the pipes never reach EOF and the reader-thread join blocks forever. The wmic /
    ``Get-CimInstance Win32_Process`` gateway scan hit exactly this during ``hermes update`` on slow-WMI
    machines (#87134); the git probes hit it first (#68609 / #66037).
    """
    _popen_kwargs: dict = {"creationflags": windows_hide_flags()} if IS_WINDOWS else {"process_group": 0}
    job = None
    try:
        # Windows: contain the probe in a Job Object. `taskkill /T` walks LIVE parent pids, and a
        # Cygwin/MSYS `exec` lets the forked stub exit once the new image runs, so a Git Bash grandchild
        # (`sleep`, `cat`) has a dead parent and survives the tree-kill holding our pipes (#73403, proven
        # on windows-latest). KILL_ON_JOB_CLOSE reaches it regardless of ancestry.
        from runtime.processes import kill_popen_process_tree, spawn_contained_process

        proc, job = spawn_contained_process(
            list(argv), stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL,
            text=True, encoding="utf-8", errors=errors,
            env=dict(env) if env is not None else None, cwd=cwd, **_popen_kwargs)
    except Exception:
        if raise_on_spawn_failure:
            raise
        return None
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except Exception:
        # Timeout OR any other communicate() failure (torn-down pipe, decode error): tree-kill and
        # drain bounded - leaving it running would leak the suspended-descendant class this guards.
        _close_job(job)
        kill_popen_process_tree(proc)
        try:
            proc.communicate(timeout=1)
        except Exception:
            pass
        return None
    # The probe exited on its own; anything it left behind (`&` jobs) goes with the job.
    _close_job(job)
    return subprocess.CompletedProcess(list(argv), proc.returncode, stdout, stderr)
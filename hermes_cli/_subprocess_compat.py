"""Windows subprocess compatibility helpers.

Hermes is developed on Linux / macOS and tested natively on Windows too.
Several common subprocess patterns break silently-or-loudly on Windows:

* ``["npm", "install", ...]`` — on Windows ``npm`` is ``npm.cmd``, a batch
  shim.  ``subprocess.Popen(["npm", ...])`` fails with WinError 193
  ("not a valid Win32 application") because CreateProcessW can't run a
  ``.cmd`` file without ``shell=True`` or PATHEXT resolution.

* ``start_new_session=True`` — on POSIX, this maps to ``os.setsid()`` and
  actually detaches the child.  On Windows it's silently ignored; the
  Windows equivalent is ``CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS``
  creationflags, which Python only applies when you pass them explicitly.

* Console-window flashes — every ``subprocess.Popen`` of a ``.exe`` on
  Windows spawns a cmd window briefly unless ``CREATE_NO_WINDOW`` is
  passed.  Cosmetic but jarring for background daemons.

This module centralizes the platform-branching logic so the rest of the
codebase doesn't sprinkle ``if sys.platform == "win32":`` everywhere.

**All helpers are no-ops on non-Windows** — calling them in Linux/macOS
code paths is safe by design.  That's the "do no damage on POSIX"
guarantee.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import threading
from typing import Sequence

logger = logging.getLogger(__name__)

__all__ = [
    "IS_WINDOWS",
    "resolve_node_command",
    "windows_detach_flags",
    "windows_detach_flags_without_breakaway",
    "windows_hide_flags",
    "windows_detach_popen_kwargs",
    "bounded_git_probe",
    "spawn_bash_with_kill_on_exit",
]


IS_WINDOWS = sys.platform == "win32"


# -----------------------------------------------------------------------------
# Node ecosystem launcher resolution
# -----------------------------------------------------------------------------


def resolve_node_command(name: str, argv: Sequence[str]) -> list[str]:
    """Resolve a Node-ecosystem command name to an absolute-path argv.

    On Windows, commands like ``npm``, ``npx``, ``yarn``, ``pnpm``,
    ``playwright``, ``prettier`` ship as ``.cmd`` files (batch shims).
    ``subprocess.Popen(["npm", "install"])`` fails with WinError 193
    because CreateProcessW doesn't execute batch files directly.

    ``shutil.which(name)`` *does* resolve ``.cmd`` via PATHEXT and returns
    the fully-qualified path — which CreateProcessW accepts because the
    extension tells Windows to route through ``cmd.exe /c``.

    On POSIX ``shutil.which`` also returns a fully-qualified path when
    found.  That's a small change from bare-name resolution (the OS does
    its own PATH search) but functionally identical and has the side
    benefit of making the argv reproducible in logs.

    Behavior when the command is not on PATH:
    - On Windows: return the bare name — caller can still try with
      ``shell=True`` as a last resort, OR the subsequent Popen will
      raise FileNotFoundError with a readable error we want to surface.
    - On POSIX: same.  Bare ``npm`` on a Linux box without npm installed
      fails the same way it did before this function existed.

    Args:
        name: The command name to resolve (``npm``, ``npx``, ``node`` …).
        argv: The remaining arguments.  Must NOT include ``name`` itself —
            this function builds the full argv list.

    Returns:
        A list suitable for passing to subprocess.Popen/run/call.
    """
    resolved = shutil.which(name)
    if resolved:
        return [resolved, *argv]
    return [name, *argv]


# -----------------------------------------------------------------------------
# Detached / hidden process creation
# -----------------------------------------------------------------------------


# Win32 CreationFlags — defined here rather than imported from subprocess
# because CREATE_NO_WINDOW and DETACHED_PROCESS aren't guaranteed to be
# present on stdlib subprocess on older Pythons or non-Windows builds.
_CREATE_NEW_PROCESS_GROUP = 0x00000200
_DETACHED_PROCESS = 0x00000008
_CREATE_NO_WINDOW = 0x08000000
# Escape any Win32 job object the parent process belongs to. Without this,
# a detached child still inherits its parent's job object membership, and
# when that parent (Electron, Tauri, Windows Terminal, the Desktop GUI's
# bootstrap-installer) dies, the OS tears down the whole job — taking the
# "detached" child with it. Critical for the post-update gateway watcher:
# Electron spawns the Tauri updater inside its own job, the updater spawns
# the watcher subprocess; without BREAKAWAY the watcher dies the instant
# Electron exits, so the gateway never gets respawned after a `hermes
# update` triggered from the GUI. See fix/windows-gateway-reliability.
_CREATE_BREAKAWAY_FROM_JOB = 0x01000000


def windows_detach_flags() -> int:
    """Return Win32 creationflags that detach a child from the parent
    console and process group.  0 on non-Windows.

    Pair with ``start_new_session=False`` (default) when calling
    subprocess.Popen — on POSIX use ``start_new_session=True`` instead,
    which maps to ``os.setsid()`` in the child.

    Rationale:
    - ``CREATE_NEW_PROCESS_GROUP`` — child has its own process group so
      Ctrl+C in the parent console doesn't propagate.
    - ``DETACHED_PROCESS`` — child has no console at all.  Necessary for
      background daemons (gateway watchers, update respawners) because
      without it, closing the console kills the child.
    - ``CREATE_NO_WINDOW`` — suppress the brief cmd flash that would
      otherwise appear when launching a console app.  Redundant with
      DETACHED_PROCESS but explicit for clarity.
    - ``CREATE_BREAKAWAY_FROM_JOB`` — escape any job object the parent is
      in.  Electron (Desktop app) and Tauri (bootstrap installer) wrap
      their children in job objects; without breakaway, those children
      die when the parent process exits even if they were spawned with
      DETACHED_PROCESS.  This was the missing flag that made the
      post-update gateway respawn watcher silently die alongside the
      Tauri updater after the Electron Desktop's update flow finished.

    If a process is in a job that disallows breakaway (rare —
    JOB_OBJECT_LIMIT_BREAKAWAY_OK isn't set), CreateProcess returns
    ERROR_ACCESS_DENIED.  Python surfaces that as ``PermissionError``
    on the ``subprocess.Popen`` call.  Callers in this codebase already
    wrap detached spawns in ``try/except OSError`` and fall back to a
    cmd.exe wrapper, so the breakaway-denied case degrades gracefully
    rather than crashing.
    """
    if not IS_WINDOWS:
        return 0
    return (
        _CREATE_NEW_PROCESS_GROUP
        | _DETACHED_PROCESS
        | _CREATE_NO_WINDOW
        | _CREATE_BREAKAWAY_FROM_JOB
    )


def windows_detach_flags_without_breakaway() -> int:
    """Same as :func:`windows_detach_flags` minus ``CREATE_BREAKAWAY_FROM_JOB``.

    The docstring on :func:`windows_detach_flags` notes that a process in
    a job which disallows breakaway (no ``JOB_OBJECT_LIMIT_BREAKAWAY_OK``)
    will see ``ERROR_ACCESS_DENIED`` from CreateProcess, surfacing as
    ``OSError`` (``PermissionError``) on the ``subprocess.Popen`` call.
    Callers that want to recover — by retrying without the breakaway
    bit — can pair the two helpers symbolically rather than coding the
    ``& ~0x01000000`` magic at every site:

    .. code-block:: python

        try:
            subprocess.Popen(argv, creationflags=windows_detach_flags(), …)
        except OSError:
            subprocess.Popen(
                argv,
                creationflags=windows_detach_flags_without_breakaway(),
                …,
            )

    See ``gateway_windows.py::_spawn_detached`` for the canonical
    implementation of this pattern.  Returns 0 on non-Windows.
    """
    if not IS_WINDOWS:
        return 0
    return _CREATE_NEW_PROCESS_GROUP | _DETACHED_PROCESS | _CREATE_NO_WINDOW


def windows_hide_flags() -> int:
    """Return Win32 creationflags that merely hide the child's console
    window without detaching the child.  0 on non-Windows.

    Use for short-lived console apps spawned as part of a larger
    operation (``taskkill``, ``where``, version probes) where we want no
    flash but also want to collect stdout/exit code synchronously.

    The key difference from :func:`windows_detach_flags`: NO
    ``DETACHED_PROCESS`` — the child still inherits stdio handles so
    ``capture_output=True`` works.  ``DETACHED_PROCESS`` would sever
    stdio and break stdout capture.
    """
    if not IS_WINDOWS:
        return 0
    return _CREATE_NO_WINDOW


def windows_detach_popen_kwargs() -> dict:
    """Return a dict of Popen kwargs that detach a child on Windows and
    fall back to the POSIX equivalent (``start_new_session=True``) on
    Linux/macOS.

    Usage pattern:

    .. code-block:: python

        subprocess.Popen(
            argv,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            close_fds=True,
            **windows_detach_popen_kwargs(),
        )

    This replaces the unsafe-on-Windows pattern:

    .. code-block:: python

        subprocess.Popen(..., start_new_session=True)

    which silently fails to detach on Windows (the flag is accepted but
    has no effect — the child stays attached to the parent's console
    and dies when the console closes).
    """
    if IS_WINDOWS:
        return {"creationflags": windows_detach_flags()}
    return {"start_new_session": True}


# -----------------------------------------------------------------------------
# Bounded, fail-open git probing (Windows post-kill deadlock guard)
# -----------------------------------------------------------------------------


def _kill_git_process_tree(proc: "subprocess.Popen") -> None:
    """Best-effort terminate *proc* and, on Windows, its descendants.

    ``proc.kill()`` alone only terminates the PATH-resolved ``git`` launcher; a
    suspended descendant ``git.exe`` can survive holding duplicates of the
    captured pipe handles, which keeps the pipes from reaching EOF and leaks two
    reader threads + the process per fired timeout. ``taskkill /T /F`` takes the
    whole tree down so the bounded drain that follows can actually reach EOF.

    All failures are swallowed — this is cleanup on an already-failing path, and
    the caller's contract is to fail open. ``kill()`` can raise (access denied,
    already reaped); an unhandled raise here would escape the caller's ``except``
    handler and break that contract. The ``taskkill`` spawn itself cannot
    re-enter the deadlock class it fixes: it captures no pipes (DEVNULL), so its
    own timeout cleanup has no reader threads to join.
    """
    try:
        proc.kill()
    except OSError:
        pass
    if IS_WINDOWS:
        try:
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                timeout=2,
                check=False,
                creationflags=windows_hide_flags(),
            )
        except Exception:
            pass


def bounded_git_probe(argv: Sequence[str], *, timeout: float) -> str:
    """Run a short, throwaway ``git`` probe and return stripped stdout, or ``""``
    on ANY failure (nonzero exit, timeout, spawn error, decode error).

    This is the shared, deadlock-safe replacement for
    ``subprocess.run(["git", ...], timeout=...)`` at fail-open probe call sites
    (``tui_gateway.git_probe.run_git``, ``agent.coding_context._git``).

    Why not ``subprocess.run``: on Windows, ``run()``'s post-timeout cleanup
    calls an *unbounded* ``communicate()`` after killing git. Killing the
    PATH-resolved launcher can leave a suspended descendant ``git.exe`` holding
    duplicates of the captured stdout/stderr handles, so the pipes never reach
    EOF and the reader-thread join blocks forever. On the Desktop agent-build
    path (``_start_agent_build → _session_info → branch() → run_git``) that turned
    an optional branch label into ``agent initialization timed out``
    (issues #68609 / #66037).

    The bounded flow: an explicit ``communicate(timeout)``, then on any failure a
    tree-kill (see :func:`_kill_git_process_tree`) plus a bounded 1s post-kill
    drain; if the pipes are still held after that, they're abandoned (the orphaned
    reader threads are daemonic and cost nothing).

    The normal-path spawn contract mirrors the previous ``run`` call byte-for-byte:
    PIPE/PIPE/DEVNULL, ``text`` with UTF-8 ``errors="replace"`` decoding, and the
    hidden-window ``creationflags`` on Windows only.
    """
    _popen_kwargs = {"creationflags": windows_hide_flags()} if IS_WINDOWS else {}
    try:
        proc = subprocess.Popen(
            list(argv),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            **_popen_kwargs,
        )
    except Exception:
        return ""
    try:
        stdout, _ = proc.communicate(timeout=timeout)
    except Exception:
        # Timeout OR any other communicate() failure (torn-down pipe, decode
        # error): terminate the child + descendants and drain bounded. Leaving
        # it running would leak the same suspended-descendant class this guards.
        _kill_git_process_tree(proc)
        try:
            proc.communicate(timeout=1)
        except Exception:
            pass
        return ""
    return stdout.strip() if proc.returncode == 0 else ""


# -----------------------------------------------------------------------------
# Kill-on-parent-exit Job Object (terminal shell orphan cleanup, Windows)
# -----------------------------------------------------------------------------
#
# On POSIX, terminal-tool shells are spawned with ``start_new_session=True``
# (os.setsid), and the existing pgid-kill machinery (see
# ``LocalEnvironment._kill_process``) reaps the whole process group when the
# session ends normally. Neither of those covers the parent (Hermes) itself
# dying ungracefully (crash, force-kill, TUI restart) — but on POSIX, orphaned
# grandchildren are at least re-parented to init and don't accumulate CPU
# unless something is still feeding them work.
#
# On Windows, ``start_new_session=True`` is a silent no-op (Python's
# subprocess module maps it to nothing on win32 — the flag only affects
# ``os.setsid`` on POSIX). That means a terminal-tool shell (bash.exe) spawned
# by the LOCAL backend, or by the docker/ssh/singularity backends' shared
# ``_popen_bash``, stays fully attached to nothing in particular: it is not
# detached (good, we want the pipes), but it is also not tied to the parent's
# lifetime in any way Windows enforces. If Hermes exits ungracefully, bash.exe
# and everything it spawned (find, grep, node, …) is simply orphaned and keeps
# running — this was observed accumulating 20+ stray processes, one consuming
# ~8 CPU-hours.
#
# The Windows primitive for "this process and everything under it dies when I
# do, even if I'm hard-killed" is a Job Object with
# ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE``. Unlike a job's usual close-on-last-
# handle-release semantics, this flag makes the OS *kill every process still
# assigned to the job* the moment the job's last handle closes — including on
# process termination without any cleanup code running. We keep exactly one
# handle open (a module-global on the Hermes process), so the job's lifetime
# is the Hermes process's lifetime, full stop.
#
# NOTE: this is the OPPOSITE of ``gateway.py``'s job-object usage, which uses
# ``CREATE_BREAKAWAY_FROM_JOB`` to *escape* the job so a detached background
# watcher can outlive Electron/Tauri. Here we deliberately attach the child so
# it *cannot* outlive Hermes. Do not reuse ``windows_detach_flags()`` for this
# path — the two are opposite intents for different callers.
try:
    if IS_WINDOWS:
        import win32api  # type: ignore
        import win32con  # type: ignore
        import win32job  # type: ignore
        import win32process  # type: ignore

        _WIN32_JOB_AVAILABLE = True
    else:
        _WIN32_JOB_AVAILABLE = False
except ImportError:  # pragma: no cover - environment without pywin32
    _WIN32_JOB_AVAILABLE = False

_kill_on_exit_job = None  # module-global handle; its lifetime == process lifetime

# Guards the check/create/publish sequence in _get_kill_on_exit_job(). Two
# threads racing the first call (e.g. two terminal-tool invocations landing
# concurrently at startup) could otherwise both see `_kill_on_exit_job is
# None`, each create their own job (A and B), and both publish — the loser's
# job object (say A) then has its last Python-side handle reference dropped
# by GC, which triggers JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE and kills whatever
# was *already* assigned to A, including a shell some other thread just
# spawned. Double-checked locking closes that window without paying a lock
# on every call once the singleton is warm.
_job_singleton_lock = threading.Lock()

# Serializes the CreateProcess monkeypatch below across threads. Only one
# suspend/assign/resume "critical section" may be in flight at a time because
# the patch is a single process-wide attribute swap
# (subprocess._winapi.CreateProcess) — concurrent spawns must not observe
# each other's patched function.
_create_process_patch_lock = threading.Lock()

_warned_job_assignment_unavailable = False


def _warn_job_assignment_once(message: str) -> None:
    """Emit exactly one ``logger.warning`` for job-assignment failures.

    Fail-open means every individual failure is swallowed, but silently
    swallowing forever means the whole cleanup mechanism can be dark with no
    signal it's not doing anything. One warning is enough to surface that in
    logs without spamming per-spawn.
    """
    global _warned_job_assignment_unavailable
    if _warned_job_assignment_unavailable:
        return
    _warned_job_assignment_unavailable = True
    logger.warning(
        "Windows kill-on-exit job assignment unavailable/failed (%s); "
        "terminal-tool child processes will not be swept up if Hermes exits "
        "ungracefully. This warning is logged once per process.",
        message,
    )


def _get_kill_on_exit_job():
    """Lazily create (once) the process-wide kill-on-close Job Object.

    Thread-safe via double-checked locking: the fast path (job already
    created) takes no lock; only the first-ever call per process pays for
    synchronization.

    Returns ``None`` if unavailable (non-Windows, pywin32 missing, or the
    job could not be created/configured) — every caller must treat ``None``
    as "skip job assignment, spawn works as today" (fail open).
    """
    global _kill_on_exit_job
    if not _WIN32_JOB_AVAILABLE:
        return None
    if _kill_on_exit_job is not None:
        return _kill_on_exit_job
    with _job_singleton_lock:
        if _kill_on_exit_job is not None:
            return _kill_on_exit_job
        try:
            job = win32job.CreateJobObject(None, "")
            info = win32job.QueryInformationJobObject(
                job, win32job.JobObjectExtendedLimitInformation
            )
            info["BasicLimitInformation"]["LimitFlags"] |= (
                win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            )
            win32job.SetInformationJobObject(
                job, win32job.JobObjectExtendedLimitInformation, info
            )
        except Exception:
            # Fail open: no job cleanup, but spawning must never be blocked
            # by this — the pre-existing (leaky) behavior is still better
            # than a crash in the terminal tool.
            _warn_job_assignment_once("job creation failed")
            return None
        _kill_on_exit_job = job
        return job


# Win32 CreateProcess's suspend flag. Defined locally (see the rationale on
# ``_CREATE_NO_WINDOW`` et al. above) rather than pulled from ``win32con`` so
# this module's Windows-only constants stay grep-able in one place.
_CREATE_SUSPENDED = 0x00000004

# Positional index of the ``creation_flags`` argument in the call signature
# CPython's ``subprocess.Popen._execute_child`` uses for
# ``_winapi.CreateProcess`` (application_name, command_line, proc_attrs,
# thread_attrs, inherit_handles, creation_flags, env_mapping,
# current_directory, startup_info) — verified against CPython 3.12's
# ``subprocess.py`` source, the version this project targets. If a future
# CPython changes this signature, ``_patched_create_process`` defensively
# falls back to an unmodified (un-suspended, unassigned) spawn rather than
# indexing into the wrong argument.
_CREATION_FLAGS_ARG_INDEX = 5
_EXPECTED_CREATE_PROCESS_ARGC = 9


def _make_patched_create_process(job, real_create_process):
    """Build a drop-in replacement for ``subprocess._winapi.CreateProcess``
    that suspends the child at birth, assigns it to *job* while it cannot yet
    run a single instruction, then resumes it.

    This closes the race the handle-only approach could not: a native
    pipeline shell (``bash -c "find ... | grep ... | head ..."``) can spawn
    its own children within microseconds of ``CreateProcess`` returning —
    faster than a second Win32 call (``AssignProcessToJobObject``) issued
    from the parent after ``subprocess.Popen()`` returns can reliably land,
    especially under parent-thread scheduler pressure, where the window is
    unbounded rather than merely small. Suspending the child's main thread at
    creation means literally zero child code — including spawning further
    children — can execute until we explicitly resume it, so assignment is
    guaranteed to happen before any descendant can exist.

    ``subprocess.Popen`` does not expose the child's *thread* handle (only
    raw ``CreateProcess`` returns it, and ``Popen._execute_child`` closes it
    immediately after spawning, before ``Popen()`` returns to the caller) —
    so this can't be done by post-processing a ``Popen`` result. Intercepting
    the ``_winapi.CreateProcess`` call ``Popen._execute_child`` makes
    internally is what lets us reach the thread handle in the small window
    while it still exists, without reimplementing everything else
    ``Popen._execute_child`` does (pipe wiring, handle inheritance,
    startupinfo, error handling) via a raw ``win32process.CreateProcess``
    call at every spawn site.

    Empirically verified: see
    ``tests/test_windows_terminal_kill_on_exit_job.py`` — patching this
    around a ``Popen()`` call whose child immediately forks a grandchild
    (``bash -c "sleep 30 & wait"``, and a Python equivalent) shows the
    grandchild is *always* inside the job, because the child cannot run the
    fork/exec syscall that creates it until ``ResumeThread`` runs.
    """

    def _patched(*args, **kwargs):
        if len(args) != _EXPECTED_CREATE_PROCESS_ARGC or kwargs:
            # Signature mismatch (future CPython change) — don't guess at
            # argument positions. Fail open to an un-suspended, unassigned
            # spawn; the process still runs, it just isn't swept.
            _warn_job_assignment_once(
                "unexpected CreateProcess signature; suspend/assign/resume skipped"
            )
            return real_create_process(*args, **kwargs)

        patched_args = list(args)
        patched_args[_CREATION_FLAGS_ARG_INDEX] = (
            patched_args[_CREATION_FLAGS_ARG_INDEX] | _CREATE_SUSPENDED
        )
        hp, ht, pid, tid = real_create_process(*patched_args)
        try:
            win32job.AssignProcessToJobObject(job, int(hp))
        except Exception:
            # Fail open: assignment can legitimately fail (already in a
            # non-nesting job on pre-Windows-8, access denied). The process
            # must still be resumed either way — it just won't be swept.
            _warn_job_assignment_once("AssignProcessToJobObject failed")
        finally:
            try:
                win32process.ResumeThread(int(ht))
            except Exception:
                # If resume itself fails, the child is permanently stuck
                # suspended, which is worse than not cleaning it up. There is
                # no further fallback available here — this mirrors the
                # narrow, already-degenerate failure mode of ResumeThread
                # raising (invalid handle from a race with process exit).
                _warn_job_assignment_once("ResumeThread failed after suspend")
        return hp, ht, pid, tid

    return _patched


def spawn_bash_with_kill_on_exit(
    popen_fn,
) -> "subprocess.Popen":
    """Call *popen_fn* (a zero-arg callable that performs the
    ``subprocess.Popen(...)`` call) and, on Windows, spawn the child
    suspended, assign it to the shared kill-on-exit Job Object, then resume
    it — so it can never outlive Hermes, including any descendants it spawns
    before the parent gets a chance to observe them. No-op wrapper on POSIX
    (returns ``popen_fn()`` unchanged) — the POSIX cleanup story is already
    correct via ``start_new_session=True`` (os.setsid) plus the existing
    pgid-kill machinery.

    Centralizing this as a wrapper (rather than duplicating the
    create-job/assign/patch calls at each spawn site) is what keeps the
    local backend and the shared ``_popen_bash`` (docker/ssh/singularity)
    from drifting the way the issue warns about.
    """
    if not IS_WINDOWS:
        return popen_fn()
    job = _get_kill_on_exit_job()
    if job is None:
        return popen_fn()

    real_create_process = subprocess._winapi.CreateProcess  # type: ignore[attr-defined]
    patched = _make_patched_create_process(job, real_create_process)
    with _create_process_patch_lock:
        subprocess._winapi.CreateProcess = patched  # type: ignore[attr-defined]
        try:
            return popen_fn()
        finally:
            subprocess._winapi.CreateProcess = real_create_process  # type: ignore[attr-defined]

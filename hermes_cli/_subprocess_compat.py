"""Windows subprocess compatibility helpers.

Hermes is developed on Linux / macOS and tested natively on Windows too.
Several common subprocess patterns break silently-or-loudly on Windows:

* ``["npm", "install", ...]`` — on Windows ``npm`` is ``npm.cmd``, a batch
  shim.  ``subprocess.Popen(["npm", ...])`` fails with WinError 193
  ("not a valid Win32 application") because CreateProcessW can't run a
  ``.cmd`` file without ``shell=True`` or PATHEXT resolution.

* ``start_new_session=True`` — on POSIX, this maps to ``os.setsid()`` and
  actually detaches the child.  On Windows it's silently ignored; the
  Windows equivalent is the ``CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW``
  creationflags bundle, which Python only applies when you pass it
  explicitly.

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

import os
import shutil
import subprocess
import sys
from typing import Mapping, Sequence

__all__ = [
    "IS_WINDOWS",
    "resolve_node_command",
    "split_command_line",
    "suppress_platform_ver_console",
    "windows_detach_flags",
    "windows_detach_flags_without_breakaway",
    "windows_hide_flags",
    "windows_detach_popen_kwargs",
    "bounded_git_probe",
    "bounded_probe_run",
    "noninteractive_git_env",
]


IS_WINDOWS = sys.platform == "win32"

# Private launcher-to-child metadata. This is diagnostic state, not user config.
_WINDOWS_GATEWAY_BREAKAWAY_ENV = "_HERMES_GATEWAY_BREAKAWAY"


def split_command_line(line: str) -> list[str]:
    """Split a user-supplied command line into tokens, Windows-safely.

    ``shlex.split(line)`` (posix=True) treats every backslash as an escape
    character, so Windows paths are silently mangled: ``C:\\Users\\me\\out.txt``
    becomes ``C:Usersmeout.txt`` — no error, just a wrong path that then
    "succeeds" against a mangled relative filename (#83934) or makes a valid
    hook script report "not executable" (#78293).

    On Windows this uses ``posix=False``, which preserves backslashes while
    still honoring double-quoted tokens ("path with spaces"). The trade-off
    is that posix=False keeps surrounding quotes on quoted tokens, so we
    strip one layer of matching double quotes per token — that matches how
    Windows command lines are conventionally parsed. On POSIX the behavior
    is exactly ``shlex.split``.

    Raises ValueError for unbalanced quotes, same as ``shlex.split``.
    """
    if not IS_WINDOWS:
        import shlex

        return shlex.split(line)
    import shlex

    tokens = shlex.split(line, posix=False)
    out: list[str] = []
    for tok in tokens:
        if len(tok) >= 2 and tok[0] == tok[-1] and tok[0] in ("'", '"'):
            tok = tok[1:-1]
        out.append(tok)
    return out


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
# DETACHED_PROCESS is intentionally NOT part of any flag bundle here — do not
# re-add it.  Two reasons (the recurring console-flash bug #54220 / #56747):
#
# 1. MSDN (Process Creation Flags): CREATE_NO_WINDOW "is ignored if used with
#    either CREATE_NEW_CONSOLE or DETACHED_PROCESS".  Combining them means
#    DETACHED_PROCESS governs and the no-window bit is dead.
# 2. A DETACHED_PROCESS child has NO console at all, so every console-subsystem
#    descendant it ever spawns (git, gh, cmd, node, wmic, powershell, …) must
#    allocate its OWN console — a visible flash per spawn, including spawns
#    inside third-party libraries that no per-call-site CREATE_NO_WINDOW sweep
#    can reach.  A CREATE_NO_WINDOW child instead OWNS a hidden console that
#    all descendants inherit, making "no flashing windows" a property of the
#    one daemon launch.  Root cause isolated + A/B verified on Windows 11 by
#    the desktop backend fix (commit aa2ae36c3f): with per-site hide flags
#    neutered, naive git/gh/cmd spawns don't flash under a hidden-console
#    parent and do flash under a console-less one.
_DETACHED_PROCESS = 0x00000008  # kept for reference; must stay out of bundles
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
    console and process group without leaving it console-less.  0 on
    non-Windows.

    Pair with ``start_new_session=False`` (default) when calling
    subprocess.Popen — on POSIX use ``start_new_session=True`` instead,
    which maps to ``os.setsid()`` in the child.

    Rationale:
    - ``CREATE_NEW_PROCESS_GROUP`` — child has its own process group so
      Ctrl+C in the parent console doesn't propagate.
    - ``CREATE_NO_WINDOW`` — the child gets its own fresh console that is
      never shown.  This both detaches it from the parent's console
      lifetime (closing the launching terminal doesn't CTRL_CLOSE it) AND
      gives every console-subsystem descendant (git, gh, cmd, node, …) a
      console to inherit, so they don't allocate visible flashing ones.
      This deliberately replaces the old ``DETACHED_PROCESS`` approach:
      MSDN specifies CREATE_NO_WINDOW is *ignored* when combined with
      DETACHED_PROCESS, and a truly console-less daemon re-creates the
      per-descendant console-flash bug (#54220/#56747) at every spawn —
      see the note on ``_DETACHED_PROCESS`` above.
    - ``CREATE_BREAKAWAY_FROM_JOB`` — escape any job object the parent is
      in.  Electron (Desktop app) and Tauri (bootstrap installer) wrap
      their children in job objects; without breakaway, those children
      die when the parent process exits even though they have their own
      console.  This was the missing flag that made the post-update
      gateway respawn watcher silently die alongside the Tauri updater
      after the Electron Desktop's update flow finished.

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
    return _CREATE_NEW_PROCESS_GROUP | _CREATE_NO_WINDOW


def windows_hide_flags() -> int:
    """Return Win32 creationflags that merely hide the child's console
    window without detaching the child.  0 on non-Windows.

    Use for short-lived console apps spawned as part of a larger
    operation (``taskkill``, ``where``, version probes) where we want no
    flash but also want to collect stdout/exit code synchronously.

    The difference from :func:`windows_detach_flags`: no
    ``CREATE_NEW_PROCESS_GROUP`` / ``CREATE_BREAKAWAY_FROM_JOB`` — the
    child stays in the parent's process group and job so Ctrl+C and job
    teardown propagate normally, as a short-lived helper wants.  Stdio
    handles are inherited either way, so ``capture_output=True`` works
    with both bundles.
    """
    if not IS_WINDOWS:
        return 0
    return _CREATE_NO_WINDOW


def suppress_platform_ver_console() -> None:
    """Stub out ``platform._syscmd_ver`` on Windows so it can never flash a
    console window.  No-op on non-Windows.

    CPython's ``platform.win32_ver()`` — reached by ``platform.uname()``,
    ``platform.version()``, and ``platform.platform()`` — unconditionally
    shells out ``cmd /c ver`` via ``subprocess.check_output(..., shell=True)``
    with no ``CREATE_NO_WINDOW``.  From a windowless parent (the pythonw
    gateway and every kanban worker it spawns) that allocates a fresh
    *visible* console: one flashing ``cmd`` window per process, triggered by
    any dependency that merely touches ``platform.uname()`` at import time.

    With ``_syscmd_ver`` stubbed to return its inputs, ``win32_ver()`` hits
    the documented ``ValueError`` fallback and reads the version from
    ``sys.getwindowsversion().platform_version`` — same information, queried
    in-process, no subprocess, no window.  Verified equivalent on
    CPython 3.11 (``platform()`` → ``Windows-10-10.0.xxxxx-SP0`` either way).

    Call early, before heavyweight imports — the flash typically happens
    during a dependency's import, not from Hermes' own code.
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
        # Purely cosmetic hardening — never let it break startup.
        pass


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
# Non-interactive git environment (credential-prompt hang guard)
# -----------------------------------------------------------------------------


def noninteractive_git_env(
    base: "Mapping[str, str] | None" = None,
) -> dict[str, str]:
    """Environment for *internal* git invocations that must never prompt.

    Hermes shells out to git from many non-interactive contexts — MCP catalog
    installs, plugin install/update, profile distribution staging, worktree
    base fetches, desktop review-pane fetch/push. When the remote is private,
    misconfigured, or requires auth, git's default behavior is to prompt on
    the inherited terminal (or via an askpass helper), which silently hangs
    the operation until its timeout — or forever at call sites without one.
    Ported from openai/codex#34540 / #34612 ("detach non-interactive
    subprocesses from stdin"): a background tool invocation must fail fast
    with a readable error, not wait for input nobody can type.

    Returns a copy of ``base`` (default ``os.environ``) with:

    * ``GIT_TERMINAL_PROMPT=0`` — git fails with "terminal prompts disabled"
      instead of prompting for credentials.
    * ``GCM_INTERACTIVE=Never`` — Git Credential Manager (the default
      credential helper on Windows installs) never pops its own dialog.

    ``GIT_ASKPASS`` / ``SSH_ASKPASS`` are deliberately left alone: when the
    user has a *working* askpass helper or ssh-agent configured, auth should
    still succeed non-interactively. The env only disables paths that block
    on a human.

    Pair with ``stdin=subprocess.DEVNULL`` so git (and any credential helper
    it spawns) also can't read the parent's inherited stdin.

    This is for internal plumbing calls only — the agent-facing terminal tool
    has its own policy layer and user-visible PTY, where prompting can be
    legitimate.
    """
    env = dict(base if base is not None else os.environ)
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GCM_INTERACTIVE"] = "Never"
    return env


# -----------------------------------------------------------------------------
# Bounded, fail-open git probing (Windows post-kill deadlock guard)
# -----------------------------------------------------------------------------


def kill_process_tree(proc: "subprocess.Popen") -> None:
    """Best-effort terminate *proc* and its descendants on both platforms.

    ``proc.kill()`` alone only terminates the direct child. On Windows a
    suspended descendant (e.g. ``git.exe``) can survive holding duplicates of the
    captured pipe handles, which keeps the pipes from reaching EOF and leaks two
    reader threads + the process per fired timeout — ``taskkill /T /F`` takes the
    whole tree down so the bounded drain that follows can actually reach EOF.
    On POSIX the same class exists: killing the launcher leaves descendants
    (credential helpers, ``git-remote-https``, hook children) running and
    holding the pipe write ends. Callers spawn the child in its own process
    group (``process_group=0``, Python ≥3.11), so when — and only
    when — the child leads its own group (``pgid == pid``), the entire group is
    signalled with ``os.killpg``. The ownership check means a fallback spawn
    that shares our group can never cause us to kill unrelated processes.
    Ported from openai/codex#36793 ("Terminate timed-out Git process trees");
    generalized for the shell-hook runner via openai/codex#37527
    ("Terminate timed-out hook process trees").

    All failures are swallowed — this is cleanup on an already-failing path, and
    the caller's contract is to fail open. ``kill()`` can raise (access denied,
    already reaped); an unhandled raise here would escape the caller's ``except``
    handler and break that contract. The ``taskkill`` spawn itself cannot
    re-enter the deadlock class it fixes: it captures no pipes (DEVNULL), so its
    own timeout cleanup has no reader threads to join.
    """
    if not IS_WINDOWS:
        # Group-kill first: verify the child actually leads its own process
        # group before signalling it, so we never blast a shared group.
        try:
            import signal as _signal

            pgid = os.getpgid(proc.pid)
            if pgid == proc.pid:
                os.killpg(pgid, _signal.SIGKILL)  # windows-footgun: ok — inside `if not IS_WINDOWS` gate
        except Exception:
            pass
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


def bounded_probe_run(
    argv: Sequence[str],
    *,
    timeout: float,
    errors: str = "replace",
) -> "subprocess.CompletedProcess[str] | None":
    """Deadlock-safe ``subprocess.run(argv, capture_output=True, timeout=...)``
    for fail-open probe call sites. Returns a ``CompletedProcess`` when the
    child finished within *timeout* (any exit code), or ``None`` on spawn
    failure or timeout.

    Why not ``subprocess.run``: on Windows, ``run()``'s post-timeout cleanup
    calls an *unbounded* ``communicate()`` after killing the direct child.
    Killing it can leave a descendant (``git.exe`` under a launcher shim,
    ``conhost.exe`` under wmic/powershell) holding duplicates of the captured
    stdout/stderr handles, so the pipes never reach EOF and the reader-thread
    join blocks forever. The wmic / ``Get-CimInstance Win32_Process`` gateway
    scan hit exactly this during ``hermes update`` on slow-WMI machines
    (#87134); the git probes hit it first (#68609 / #66037).

    The bounded flow: an explicit ``communicate(timeout)``, then on any
    failure a tree-kill (see :func:`kill_process_tree`) plus a bounded 1s
    post-kill drain; if the pipes are still held after that, they're abandoned
    (the orphaned reader threads are daemonic and cost nothing).

    The spawn contract mirrors the ``run`` calls it replaces: PIPE/PIPE/DEVNULL,
    ``text`` with UTF-8 decoding (*errors* configurable — the process scans use
    ``"ignore"``), and the hidden-window ``creationflags`` on Windows only. On
    POSIX the child is placed in its own process group (``process_group=0``,
    Python ≥3.11) so timeout cleanup can take down descendants with the
    launcher instead of orphaning them.
    """
    _popen_kwargs: dict = {"creationflags": windows_hide_flags()} if IS_WINDOWS else {"process_group": 0}
    try:
        proc = subprocess.Popen(
            list(argv),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors=errors,
            **_popen_kwargs,
        )
    except Exception:
        return None
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except Exception:
        # Timeout OR any other communicate() failure (torn-down pipe, decode
        # error): terminate the child + descendants and drain bounded. Leaving
        # it running would leak the same suspended-descendant class this guards.
        kill_process_tree(proc)
        try:
            proc.communicate(timeout=1)
        except Exception:
            pass
        return None
    return subprocess.CompletedProcess(list(argv), proc.returncode, stdout, stderr)


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
    hidden-window ``creationflags`` on Windows only. On POSIX the probe is
    additionally placed in its own process group (``process_group=0``,
    Python ≥3.11) so timeout cleanup can take down descendants — credential
    helpers, ``git-remote-https``, hook children — with the launcher instead of
    orphaning them (see :func:`_kill_git_process_tree`; port of
    openai/codex#36793). ``process_group`` only changes which group the child
    belongs to; it does not detach the terminal or alter the fast path.
    """
    result = bounded_probe_run(argv, timeout=timeout)
    if result is None or result.returncode != 0:
        return ""
    return (result.stdout or "").strip()


# Backward-compat alias — existing call sites/tests import the historical name.
_kill_git_process_tree = kill_process_tree

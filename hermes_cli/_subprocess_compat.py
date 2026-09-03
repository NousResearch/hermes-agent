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

import logging
import os
import re
import shutil
import subprocess
import sys
import threading
from typing import Mapping, Sequence

logger = logging.getLogger(__name__)

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
    "NO_DRIVER_DIFF_FLAGS",
    "pid_is_hermes",
    "spawn_bash_with_kill_on_exit",
]

# Flags that neutralize *attribute-scoped* diff drivers on any diff-rendering
# git command (``diff``, ``log -p``, ``show``, ``blame``). A malicious repo can
# name a driver in ``.gitattributes`` (``* diff=evil``) and point it at an
# arbitrary program via ``[diff "evil"] command=/textconv=`` in ``.git/config``.
# Because the attacker chooses the driver name, ``GIT_CONFIG_KEY`` overrides in
# ``noninteractive_git_env`` cannot enumerate and disable it — only these
# command-line flags do. ``--no-ext-diff`` kills ``command=``; ``--no-textconv``
# kills ``textconv=``. Both are required (verified empirically: each alone
# leaves the other live). Smudge/clean filters are neutralized by the env
# layer's ``core.hooksPath`` + running against the index without checkout.
NO_DRIVER_DIFF_FLAGS = ("--no-ext-diff", "--no-textconv")

# Subcommands that render diffs and therefore invoke ``.gitattributes``-scoped
# diff/textconv drivers. Only these accept ``NO_DRIVER_DIFF_FLAGS`` — ``status``
# and friends reject the flags (``unknown option``), so the helper must gate on
# this set rather than blanket-prepending.
_DIFF_RENDERING_SUBCOMMANDS = frozenset({"diff", "show", "log", "blame"})


def harden_git_argv(args: Sequence[str]) -> list[str]:
    """Return a copy of subcommand-first git *args* with diff-driver flags
    inserted for diff-rendering subcommands.

    *args* is the argument list WITHOUT the leading ``"git"`` (e.g.
    ``["diff", "HEAD"]`` or ``["-c", "core.quotePath=false", "diff", ...]``).
    The first non-option token is treated as the subcommand; if it is one of
    :data:`_DIFF_RENDERING_SUBCOMMANDS`, :data:`NO_DRIVER_DIFF_FLAGS` is
    inserted immediately after it. Non-diff subcommands are returned unchanged.

    Pair with :func:`noninteractive_git_env`: the env layer disables
    fsmonitor/hooks/pager/editor/credential sinks, this closes the one class
    (attacker-named attribute drivers) env overrides cannot reach.
    """
    out = list(args)
    # Options that consume the FOLLOWING token as their value, so that value is
    # never mistaken for the subcommand (``-C diff`` is a path; ``-c diff=x`` is
    # a config pair — neither is the diff subcommand).
    _value_opts = {"-C", "-c", "--git-dir", "--work-tree", "--namespace", "--exec-path"}
    i = 0
    while i < len(out):
        tok = out[i]
        if tok in _value_opts:
            i += 2
            continue
        if tok.startswith("-"):
            i += 1
            continue
        if tok in _DIFF_RENDERING_SUBCOMMANDS:
            return out[: i + 1] + list(NO_DRIVER_DIFF_FLAGS) + out[i + 1 :]
        # First non-option token is the subcommand; if it isn't a diff renderer
        # there is nothing to harden.
        return out
    return out


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
    * isolated git config — inherited ``GIT_CONFIG_*`` overrides, global/system
      config, pagers, editors, fsmonitor, external diff, and hooks are disabled
      for the child process. A user's repo/global config should not be able to
      hang or mutate Hermes's internal plumbing calls.

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

    # Do not inherit caller-supplied config injection. We rebuild the
    # GIT_CONFIG_COUNT block below so ambient -c values cannot re-enable
    # pagers, hooks, fsmonitor, editors, or credential prompts.
    for key in list(env):
        if (
            key == "GIT_CONFIG_PARAMETERS"
            or key.startswith("GIT_CONFIG_KEY_")
            or key.startswith("GIT_CONFIG_VALUE_")
        ):
            env.pop(key, None)
    env.pop("GIT_CONFIG_COUNT", None)

    devnull = os.devnull
    env["GIT_CONFIG_GLOBAL"] = devnull
    env["GIT_CONFIG_SYSTEM"] = devnull
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    env["GIT_PAGER"] = "cat"
    env["PAGER"] = "cat"
    env["GIT_EDITOR"] = "true"

    config_overrides = {
        "credential.helper": "",
        "core.askPass": "",
        "core.fsmonitor": "false",
        "core.untrackedCache": "false",
        "core.hooksPath": devnull,
        "core.pager": "cat",
        "core.editor": "true",
        "sequence.editor": "true",
        "diff.external": "",
    }
    env["GIT_CONFIG_COUNT"] = str(len(config_overrides))
    for idx, (key, value) in enumerate(config_overrides.items()):
        env[f"GIT_CONFIG_KEY_{idx}"] = key
        env[f"GIT_CONFIG_VALUE_{idx}"] = value

    return env


# -----------------------------------------------------------------------------
# Bounded, fail-open git probing (Windows post-kill deadlock guard)
# -----------------------------------------------------------------------------



def _process_start_time(pid: int) -> int | None:
    """Return the repository's stable process-start fingerprint, if available."""
    try:
        from gateway.status import get_process_start_time

        return get_process_start_time(pid)
    except Exception:
        return None


def _text_names_hermes(text: str) -> bool:
    """True when *text* names Hermes at a path-segment / token boundary.

    A bare ``"hermes" in text`` substring test would also match unrelated
    processes whose paths merely contain the letters (``...\\shermesa\\...``),
    which is exactly the false-positive class this guard exists to prevent.
    Instead, split on path separators and whitespace and require a segment
    that *starts with* ``hermes`` (``hermes``, ``hermes.exe``, ``hermes_cli``,
    ``hermes-agent``, ``hermes-runtime``) or the hidden-dir form
    ``.hermes``/``.hermes-runtime``.
    """
    for token in re.split(r"[\\/\s=,;\"']+", text.lower()):
        if token.startswith("hermes") or token.startswith(".hermes"):
            return True
    return False


def _process_command_is_hermes(pid: int) -> bool:
    """Best-effort check that *pid* currently runs Hermes code."""
    try:
        import psutil

        process = psutil.Process(pid)
        command = " ".join(process.cmdline() or [])
        executable = process.exe() or ""
        return _text_names_hermes(f"{command} {executable}")
    except Exception:
        return False


def pid_is_hermes(
    pid: int,
    *,
    expected_start_time: int | None = None,
) -> bool:
    """Return whether it is safe to use ``taskkill`` for *pid*.

    The PID must be valid, currently exist, and identify a Hermes process. When
    the caller captured a start-time fingerprint before the destructive action,
    the live process must still have the same ``(pid, start_time)`` identity.
    Any ambiguity fails closed. Non-Windows callers have no ``taskkill`` path,
    so a valid PID with no (or a matching) explicit expectation is accepted
    there — but a caller-provided fingerprint that no longer matches is a
    recycled PID on every platform and is always refused.
    """
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return False
    if not IS_WINDOWS:
        if expected_start_time is None:
            return True
        try:
            return _process_start_time(pid) == expected_start_time
        except Exception:
            return False

    try:
        current_start_time = _process_start_time(pid)
    except Exception:
        return False
    if current_start_time is None:
        return False
    if (
        expected_start_time is not None
        and current_start_time != expected_start_time
    ):
        return False
    try:
        return _process_command_is_hermes(pid)
    except Exception:
        return False


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

    Delegates the tree-kill to :func:`agent.deadline.kill_process_tree`
    (#85125 4d) — same taskkill /T /F on Windows and killpg-when-leader on
    POSIX, plus a psutil descendant sweep that also reaches descendants that
    ``setsid``'d into their own sessions. On any import/delegation failure it
    falls back to the original local implementation
    (:func:`_legacy_kill_process_tree`), so the fail-open contract holds even
    in stripped environments.
    """
    try:
        from agent.deadline import kill_process_tree as _deadline_kill_tree

        _deadline_kill_tree(proc.pid)
    except Exception:
        _legacy_kill_process_tree(proc)
        return
    # Ensure Popen's own bookkeeping sees the exit (matches the legacy body:
    # a direct kill() so communicate()/wait() cannot hang on a stale handle).
    try:
        proc.kill()
    except OSError:
        pass


def _legacy_kill_process_tree(proc: "subprocess.Popen") -> None:
    """Pre-#85125 local tree-kill — fallback when agent.deadline is unavailable.

    Kept verbatim so ``kill_process_tree`` can honor its swallow-everything
    contract even when the delegation path itself fails (partial install,
    import cycle during teardown).
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
        # No identity guard here on purpose: *proc* is our own retained
        # ``Popen`` handle. The child cannot be reaped (and its PID cannot be
        # recycled) while we still hold the handle, so an identity check could
        # only ever false-refuse a legitimate cleanup. The fail-closed
        # ``pid_is_hermes`` guard is for BARE pids from state files or process
        # scans, where recycling is real.
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
    env: "Mapping[str, str] | None" = None,
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
            env=dict(env) if env is not None else None,
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

    **Security (GHSA-7x36-8jrh-v4pw):** these probes run automatically against
    whatever directory the session sits in — the coding-workspace snapshot and
    the gateway project-tree build fire ``git status`` / ``git branch`` before
    any tool call, approval, or trust prompt. An index refresh executes the
    repository-configured ``core.fsmonitor`` program, and other config keys
    (hooks, pager, editor, credential helper) are execution sinks too. A repo
    delivered as files with its ``.git`` directory intact (a shared zip, sync
    folder, or USB stick — ``git clone`` never transfers ``.git/config``) would
    otherwise get host code execution as the user. Every probe now runs under
    :func:`noninteractive_git_env`, which pins those keys to inert values via
    ``GIT_CONFIG_*`` and ignores global/system config. Diff-rendering callers
    additionally pass :data:`NO_DRIVER_DIFF_FLAGS` (attribute-scoped drivers
    can't be disabled through env overrides).

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
    result = bounded_probe_run(argv, timeout=timeout, env=noninteractive_git_env())
    if result is None or result.returncode != 0:
        return ""
    return (result.stdout or "").strip()


# Backward-compat alias — existing call sites/tests import the historical name.
_kill_git_process_tree = kill_process_tree

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
# Module-level placeholders so ``win32*`` are always attributes of this module,
# even on non-Windows. The real pywin32 modules overwrite them on Windows; on
# other platforms they stay ``None`` (guarded by ``_WIN32_JOB_AVAILABLE``).
# Keeping the names defined lets the platform-independent unit tests
# ``monkeypatch.setattr(_subprocess_compat, "win32job", fake)`` on Linux CI
# instead of failing with AttributeError.
win32api = None  # type: ignore
win32con = None  # type: ignore
win32job = None  # type: ignore
win32process = None  # type: ignore
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
        _warn_job_assignment_once("pywin32 unavailable")
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
# CPython changes this signature, the gated wrapper defensively falls back
# to an unmodified (un-suspended, unassigned) spawn rather than indexing
# into the wrong argument.
_CREATION_FLAGS_ARG_INDEX = 5
_EXPECTED_CREATE_PROCESS_ARGC = 9

# --- Owner-thread-gated CreateProcess patch -------------------------------
#
# The patch below is installed on ``subprocess._winapi.CreateProcess``
# **exactly once, permanently**, rather than swapped in/out around each
# spawn. Two earlier per-call-monkeypatch approaches were rejected after
# adversarial review (issue #69033):
#
# 1. Capturing "the current CreateProcess" and restoring it after the call
#    is only safe if capture and restore happen while holding the same lock
#    the whole time. Capturing *before* acquiring the lock lets a second
#    caller capture the first caller's already-patched function as "the
#    original" while the first caller is still active; when the first
#    caller finishes and restores what *it* captured (the true original),
#    then the second caller finishes and restores what *it* captured (the
#    first caller's patched function) — CreateProcess is now permanently
#    stuck patched, process-wide, forever.
#
# 2. Even with capture/restore correctly serialized under one lock,
#    ``subprocess._winapi.CreateProcess`` is a single process-wide function.
#    While our lock is held, *any other thread* calling ``subprocess.Popen``
#    for any unrelated reason (test fixtures, MCP subprocess spawns, any
#    library) is routed through our patch too — suspended, assigned to our
#    job, and made to inherit ``KILL_ON_JOB_CLOSE`` without ever asking for
#    it. That manifests as random unrelated subprocesses dying whenever
#    Hermes drops the job handle.
#
# The fix: install the patched function once (idempotent, guarded only for
# the one-time install), and never touch ``subprocess._winapi.CreateProcess``
# again afterward. The patch itself is inert for every thread by default —
# it only suspends/assigns/resumes when the *calling thread* has opted in via
# ``_spawn_owner.job`` (a ``threading.local``), which
# ``spawn_bash_with_kill_on_exit`` sets for the duration of exactly the one
# ``popen_fn()`` call it wraps, on the calling thread only. Other threads'
# ``_spawn_owner.job`` is unset, so their ``CreateProcess`` calls pass
# straight through to the true original with zero observable difference —
# this is what makes the patch safe to leave permanently installed.
_original_create_process = None  # type: ignore[assignment]
_create_process_patch_install_lock = threading.Lock()
_spawn_owner = threading.local()


def _job_owned_create_process(*args, **kwargs):
    """The permanently-installed replacement for
    ``subprocess._winapi.CreateProcess``.

    Passes straight through to the true original for every thread except
    the one currently inside ``spawn_bash_with_kill_on_exit`` (identified by
    ``_spawn_owner.job`` being set on this thread's local storage) — see the
    module-level comment above for why this thread-local gate, rather than a
    capture/restore swap, is what keeps the patch from bleeding into
    unrelated spawns on other threads.
    """
    job = getattr(_spawn_owner, "job", None)
    if job is None:
        return _original_create_process(*args, **kwargs)

    if len(args) != _EXPECTED_CREATE_PROCESS_ARGC or kwargs:
        # Signature mismatch (future CPython change) — don't guess at
        # argument positions. Fail open to an un-suspended, unassigned
        # spawn; the process still runs, it just isn't swept.
        _warn_job_assignment_once(
            "unexpected CreateProcess signature; suspend/assign/resume skipped"
        )
        return _original_create_process(*args, **kwargs)

    patched_args = list(args)
    patched_args[_CREATION_FLAGS_ARG_INDEX] = (
        patched_args[_CREATION_FLAGS_ARG_INDEX] | _CREATE_SUSPENDED
    )
    hp, ht, pid, tid = _original_create_process(*patched_args)
    try:
        win32job.AssignProcessToJobObject(job, int(hp))
    except Exception:
        # Fail open: assignment can legitimately fail (already in a
        # non-nesting job on pre-Windows-8, access denied). The process must
        # still be resumed either way — it just won't be swept.
        _warn_job_assignment_once("AssignProcessToJobObject failed")

    try:
        win32process.ResumeThread(int(ht))
    except Exception:
        # Unlike assignment failure, a resume failure is NOT safe to fail
        # open on: the child is a live process whose main thread will never
        # run, so any caller that reads its stdout (every caller here does)
        # hangs forever waiting for output that can never be produced. There
        # is no valid "let it run" fallback for a permanently-suspended
        # process, so terminate it and raise -- the caller's existing
        # spawn-failure handling (subprocess.Popen already raises OSError
        # for a variety of CreateProcess failures) is the right path for
        # this to surface through, not a silent hang.
        _warn_job_assignment_once("ResumeThread failed after suspend; terminating child")
        try:
            win32process.TerminateProcess(int(hp), 1)
        except Exception:
            # Visible but non-fatal: a suspended child may survive if this
            # also fails (already exited, access denied), which is worse
            # than a silent swallow used to be, but silently leaving a
            # permanently-suspended process around with no signal at all is
            # worse still. The original ResumeThread failure is still
            # raised below either way.
            logger.warning(
                "Windows kill-on-exit: TerminateProcess also failed after "
                "ResumeThread failed; a suspended child process may be "
                "left behind.",
                exc_info=True,
            )
        # CPython's ``_winapi.CreateProcess`` returns plain integer handles,
        # not PyHANDLE wrapper objects — there is no ``.Close()`` method on
        # them. The previous ``handle.Close()`` attempt raised AttributeError
        # on every call, which the surrounding ``except Exception`` silently
        # swallowed, so both handles leaked on every ResumeThread failure.
        # ``_winapi.CloseHandle`` is the correct API for raw int handles.
        for handle in (hp, ht):
            try:
                subprocess._winapi.CloseHandle(int(handle))  # type: ignore[attr-defined]
            except Exception:
                pass
        raise
    return hp, ht, pid, tid


def _install_job_owned_create_process_once() -> None:
    """Idempotently install :func:`_job_owned_create_process` over
    ``subprocess._winapi.CreateProcess`` exactly once per process.

    Double-checked locking mirrors :func:`_get_kill_on_exit_job` — the fast
    path (already installed) takes no lock. Because the install is
    idempotent and the *original* is captured only inside the lock on the
    very first install, there is no capture/restore dance and therefore no
    equivalent of the stale-patch race the old per-call approach had.
    """
    global _original_create_process
    if _original_create_process is not None:
        return
    with _create_process_patch_install_lock:
        if _original_create_process is not None:
            return
        current = subprocess._winapi.CreateProcess  # type: ignore[attr-defined]
        if getattr(current, "_hermes_job_owned", False):
            # A module reload (importlib.reload) re-executes this module's
            # top-level code in the SAME (mutated-in-place) module
            # namespace, which resets ``_original_create_process`` back to
            # ``None`` here -- but ``subprocess._winapi.CreateProcess``
            # still points at the wrapper function object a PRIOR
            # generation of this module installed; reload never uninstalls
            # it. Blindly re-capturing "the current CreateProcess" in that
            # case captures our OWN already-installed wrapper as "the
            # original", so every future call recurses into itself
            # (RecursionError) -- reproduced directly against this
            # scenario. Recognize our own wrapper via a marker attribute
            # and recover the true original from an attribute stashed on
            # the wrapper function object itself: function objects (and
            # attributes set on them) survive reload: only names inside the
            # module dict get reassigned, not objects already handed out to
            # code outside the module.
            logger.debug(
                "Windows kill-on-exit: CreateProcess already wraps our own "
                "hermes job-owned patch (module reload detected); reusing "
                "the existing wrapper instead of re-patching."
            )
            _original_create_process = current._hermes_true_original
            return
        real_create_process = current
        _original_create_process = real_create_process
        _job_owned_create_process._hermes_true_original = real_create_process
        _job_owned_create_process._hermes_job_owned = True
        subprocess._winapi.CreateProcess = _job_owned_create_process  # type: ignore[attr-defined]


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
    create-job/assign calls at each spawn site) is what keeps the local
    backend and the shared ``_popen_bash`` (docker/ssh/singularity) from
    drifting the way the issue warns about.

    Only the calling thread's own ``popen_fn()`` call is affected — see the
    module comment above ``_job_owned_create_process`` for how the
    thread-local gate keeps concurrent, unrelated ``subprocess.Popen`` calls
    on other threads from being swept into our job.
    """
    if not IS_WINDOWS:
        return popen_fn()
    job = _get_kill_on_exit_job()
    if job is None:
        return popen_fn()

    _install_job_owned_create_process_once()
    _spawn_owner.job = job
    try:
        return popen_fn()
    finally:
        _spawn_owner.job = None

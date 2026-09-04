"""Windows subprocess compatibility helpers.

* ``["npm", ...]`` — on Windows ``npm`` is ``npm.cmd``, a batch shim; ``Popen`` fails with
  WinError 193 because CreateProcessW can't run a ``.cmd`` without ``shell=True``/PATHEXT.
* ``start_new_session=True`` — POSIX ``os.setsid()`` detach; silently ignored on Windows, whose
  equivalent is the ``CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW`` creationflags bundle.
"""

from __future__ import annotations

import os
import re
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
    "NO_DRIVER_DIFF_FLAGS",
    "pid_is_hermes",
]

# Flags that neutralize *attribute-scoped* diff drivers on any diff-rendering git command. A
# malicious repo can name a driver in ``.gitattributes`` (``* diff=evil``) and point it at an
# arbitrary program via ``[diff "evil"] command=/textconv=`` in ``.git/config``; because the
# attacker chooses the name, ``GIT_CONFIG_KEY`` overrides in ``noninteractive_git_env`` cannot
# enumerate it — only these flags do. ``--no-ext-diff`` kills ``command=``; ``--no-textconv`` kills
# ``textconv=``; each alone leaves the other live. Smudge/clean filters are neutralized by the env
# layer's ``core.hooksPath`` + running against the index without checkout.
NO_DRIVER_DIFF_FLAGS = ("--no-ext-diff", "--no-textconv")

# Only these subcommands accept ``NO_DRIVER_DIFF_FLAGS`` — ``status`` and friends reject them
# (``unknown option``), so the helper gates on this set rather than blanket-prepending.
_DIFF_RENDERING_SUBCOMMANDS = frozenset({"diff", "show", "log", "blame"})

# Options that consume the FOLLOWING token, so that value is never mistaken for the subcommand
# (``-C diff`` is a path; ``-c diff=x`` is a config pair).
_GIT_VALUE_OPTS = {"-C", "-c", "--git-dir", "--work-tree", "--namespace", "--exec-path"}


def harden_git_argv(args: Sequence[str]) -> list[str]:
    """Copy of subcommand-first git *args* (no leading ``"git"``) with :data:`NO_DRIVER_DIFF_FLAGS`
    inserted right after a diff-rendering subcommand; other subcommands are returned unchanged.

    Pair with :func:`noninteractive_git_env`: the env layer disables fsmonitor/hooks/pager/editor/
    credential sinks, this closes the one class (attacker-named attribute drivers) env cannot reach.
    """
    out = list(args)
    i = 0
    while i < len(out):
        tok = out[i]
        if tok in _GIT_VALUE_OPTS:
            i += 2
            continue
        if tok.startswith("-"):
            i += 1
            continue
        if tok in _DIFF_RENDERING_SUBCOMMANDS:
            return out[: i + 1] + list(NO_DRIVER_DIFF_FLAGS) + out[i + 1 :]
        return out  # first non-option token is a non-diff subcommand
    return out


IS_WINDOWS = sys.platform == "win32"

# Private launcher-to-child metadata. This is diagnostic state, not user config.
_WINDOWS_GATEWAY_BREAKAWAY_ENV = "_HERMES_GATEWAY_BREAKAWAY"


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


# GIT_CONFIG_KEY_n/VALUE_n overrides for internal git children: no credential/askpass prompts, no
# repo-configured fsmonitor/hooks/pager/editor/external-diff programs.
_GIT_CONFIG_INJECT_PREFIXES = ("GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_")
_GIT_CONFIG_OVERRIDES = {
    "credential.helper": "",
    "core.askPass": "",
    "core.fsmonitor": "false",
    "core.untrackedCache": "false",
    "core.hooksPath": os.devnull,
    "core.pager": "cat",
    "core.editor": "true",
    "sequence.editor": "true",
    "diff.external": "",
}


def noninteractive_git_env(base: "Mapping[str, str] | None" = None) -> dict[str, str]:
    """Environment for *internal* git invocations that must never prompt.

    Copy of ``base`` (default ``os.environ``) with ``GIT_TERMINAL_PROMPT=0`` (fail instead of
    prompting), ``GCM_INTERACTIVE=Never`` (no Git Credential Manager dialog), and isolated git
    config: inherited ``GIT_CONFIG_*`` injection, global/system config, pagers, editors, fsmonitor,
    external diff and hooks are all disabled so a user's repo/global config cannot hang or mutate
    Hermes's plumbing calls. ``GIT_ASKPASS``/``SSH_ASKPASS`` are deliberately left alone: a
    *working* askpass helper or ssh-agent should still succeed non-interactively. Pair with
    ``stdin=subprocess.DEVNULL``. Internal plumbing only — the agent-facing terminal tool has its
    own policy layer and visible PTY.

    Hermes shells out to git from many non-interactive contexts — MCP catalog installs, plugin
    install/update, profile distribution staging, worktree base fetches, desktop review-pane fetch/push.
    When the remote is private, misconfigured, or requires auth, git's default behavior is to prompt on the
    inherited terminal (or via an askpass helper), which silently hangs the operation until its timeout — or
    forever at call sites without one. Ported from openai/codex#34540 / #34612 ("detach non-interactive
    subprocesses from stdin"): a background tool invocation must fail fast with a readable error, not wait
    for input nobody can type.
    """
    env = dict(base if base is not None else os.environ)
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GCM_INTERACTIVE"] = "Never"
    # Drop caller-supplied config injection; the GIT_CONFIG_COUNT block is rebuilt below so
    # ambient -c values cannot re-enable pagers, hooks, fsmonitor, editors or credential prompts.
    for key in list(env):
        if key == "GIT_CONFIG_PARAMETERS" or key.startswith(_GIT_CONFIG_INJECT_PREFIXES):
            env.pop(key, None)
    env.pop("GIT_CONFIG_COUNT", None)
    env["GIT_CONFIG_GLOBAL"] = os.devnull
    env["GIT_CONFIG_SYSTEM"] = os.devnull
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    env["GIT_PAGER"] = "cat"
    env["PAGER"] = "cat"
    env["GIT_EDITOR"] = "true"
    env["GIT_CONFIG_COUNT"] = str(len(_GIT_CONFIG_OVERRIDES))
    for idx, (key, value) in enumerate(_GIT_CONFIG_OVERRIDES.items()):
        env[f"GIT_CONFIG_KEY_{idx}"] = key
        env[f"GIT_CONFIG_VALUE_{idx}"] = value
    return env


def _process_start_time(pid: int) -> int | None:
    """The repository's stable process-start fingerprint, if available."""
    try:
        from gateway.status import get_process_start_time

        return get_process_start_time(pid)
    except Exception:
        return None


def _text_names_hermes(text: str) -> bool:
    r"""True when *text* names Hermes at a path-segment / token boundary.

    A bare ``"hermes" in text`` substring test would also match unrelated processes whose paths
    merely contain the letters (``...\shermesa\...``) — the false-positive class this prevents.
    """
    return any(token.startswith(("hermes", ".hermes"))
               for token in re.split(r"[\\/\s=,;\"']+", text.lower()))


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


def pid_is_hermes(pid: int, *, expected_start_time: int | None = None) -> bool:
    """Whether it is safe to use ``taskkill`` for *pid*.

    The PID must be valid, currently exist, and identify a Hermes process. When the caller captured
    a start-time fingerprint before the destructive action, the live process must still have the
    same ``(pid, start_time)`` identity. Any ambiguity fails closed.
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
    if expected_start_time is not None and current_start_time != expected_start_time:
        return False
    try:
        return _process_command_is_hermes(pid)
    except Exception:
        return False


def kill_process_tree(proc: "subprocess.Popen") -> None:
    """Best-effort terminate *proc* and its descendants on both platforms; never raises.

    ``proc.kill()`` alone only terminates the direct child. This is cleanup on an already-failing
    path whose contract is to fail open, so every failure (access denied, already reaped) is
    swallowed rather than escaping the caller's ``except``.

    On Windows a suspended descendant (e.g. ``git.exe``) can survive holding duplicates of the captured pipe
    handles, which keeps the pipes from reaching EOF and leaks two reader threads + the process per fired
    timeout — ``taskkill /T /F`` takes the whole tree down so the bounded drain that follows can actually
    reach EOF. On POSIX the same class exists: killing the launcher leaves descendants (credential helpers,
    ``git-remote-https``, hook children) running and holding the pipe write ends. Callers spawn the child in
    its own process group (``process_group=0``, Python ≥3.11), so when — and only when — the child leads its
    own group (``pgid == pid``), the entire group is signalled with ``os.killpg``. The ownership check means
    a fallback spawn that shares our group can never cause us to kill unrelated processes. Ported from
    openai/codex#36793 ("Terminate timed-out Git process trees"); generalized for the shell-hook runner via
    openai/codex#37527 ("Terminate timed-out hook process trees").
    """
    try:
        from agent.deadline import kill_process_tree as _deadline_kill_tree

        _deadline_kill_tree(proc.pid)
    except Exception:
        _legacy_kill_process_tree(proc)
        return
    # Ensure Popen's own bookkeeping sees the exit so communicate()/wait() cannot hang.
    try:
        proc.kill()
    except OSError:
        pass


def _legacy_kill_process_tree(proc: "subprocess.Popen") -> None:
    """Local tree-kill fallback when agent.deadline is unavailable (partial install, cycle)."""
    if not IS_WINDOWS:
        # Verify the child leads its own process group before signalling, never a shared group.
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
        # No identity guard on purpose: *proc* is our own retained Popen handle, so the PID cannot
        # be recycled while we hold it. The fail-closed ``pid_is_hermes`` guard is for BARE pids.
        try:
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           stdin=subprocess.DEVNULL, timeout=2, check=False,
                           creationflags=windows_hide_flags())
        except Exception:
            pass


def bounded_probe_run(
    argv: Sequence[str], *, timeout: float, errors: str = "replace",
    env: "Mapping[str, str] | None" = None,
) -> "subprocess.CompletedProcess[str] | None":
    """Deadlock-safe ``subprocess.run(argv, capture_output=True, timeout=…)`` for fail-open probes.

    Returns a ``CompletedProcess`` when the child finished within *timeout* (any exit code), or
    ``None`` on spawn failure or timeout.

    Why not ``subprocess.run``: on Windows, ``run()``'s post-timeout cleanup calls an *unbounded*
    ``communicate()`` after killing the direct child. Killing it can leave a descendant (``git.exe`` under a
    launcher shim, ``conhost.exe`` under wmic/powershell) holding duplicates of the captured stdout/stderr
    handles, so the pipes never reach EOF and the reader-thread join blocks forever. The wmic /
    ``Get-CimInstance Win32_Process`` gateway scan hit exactly this during ``hermes update`` on slow-WMI
    machines (#87134); the git probes hit it first (#68609 / #66037).
    """
    _popen_kwargs: dict = {"creationflags": windows_hide_flags()} if IS_WINDOWS else {"process_group": 0}
    try:
        proc = subprocess.Popen(
            list(argv), stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL,
            text=True, encoding="utf-8", errors=errors,
            env=dict(env) if env is not None else None, **_popen_kwargs)
    except Exception:
        return None
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except Exception:
        # Timeout OR any other communicate() failure (torn-down pipe, decode error): tree-kill and
        # drain bounded — leaving it running would leak the suspended-descendant class this guards.
        kill_process_tree(proc)
        try:
            proc.communicate(timeout=1)
        except Exception:
            pass
        return None
    return subprocess.CompletedProcess(list(argv), proc.returncode, stdout, stderr)


def bounded_git_probe(argv: Sequence[str], *, timeout: float) -> str:
    """Run a short ``git`` probe and return stripped stdout, or ``""`` on ANY failure.

    On Windows ``run()``'s post-timeout cleanup calls an unbounded ``communicate()``; a suspended
    descendant git.exe holding the pipe handles then blocks forever. Here: bounded ``communicate``,
    tree-kill plus a 1s drain, then abandon the pipes; on POSIX the probe gets its own process
    group so cleanup also takes down credential/remote helpers.

    Security (GHSA-7x36-8jrh-v4pw): these probes run automatically against whatever directory the
    session sits in, before any tool call or trust prompt, and an index refresh executes the
    repo-configured ``core.fsmonitor`` program. Every probe therefore runs under
    :func:`noninteractive_git_env`; diff-rendering callers additionally pass
    :data:`NO_DRIVER_DIFF_FLAGS` (attribute-scoped drivers can't be disabled via env).

    Killing the PATH-resolved launcher can leave a suspended descendant ``git.exe`` holding duplicates of
    the captured stdout/stderr handles, so the pipes never reach EOF and the reader-thread join blocks
    forever. On the Desktop agent-build path (``_start_agent_build → _session_info → branch() → run_git``)
    that turned an optional branch label into ``agent initialization timed out`` (issues #68609 / #66037).
    The normal-path spawn contract mirrors the previous ``run`` call byte-for-byte: PIPE/PIPE/DEVNULL,
    ``text`` with UTF-8 ``errors="replace"`` decoding, and the hidden-window ``creationflags`` on Windows
    only. On POSIX the probe is additionally placed in its own process group (``process_group=0``, Python
    ≥3.11) so timeout cleanup can take down descendants — credential helpers, ``git-remote-https``, hook
    children — with the launcher instead of orphaning them (see :func:`kill_process_tree`; port of
    openai/codex#36793). ``process_group`` only changes which group the child belongs to; it does not detach
    the terminal or alter the fast path.
    """
    result = bounded_probe_run(argv, timeout=timeout, env=noninteractive_git_env())
    if result is None or result.returncode != 0:
        return ""
    return (result.stdout or "").strip()


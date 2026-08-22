"""``!<command>`` shell mode for the interactive CLI.

Typing ``!git status`` at the composer runs the command directly in the
session's working directory. The model is never invoked: no user message, no
assistant message, no tool result enters the conversation history, so a bang
command costs zero tokens and cannot perturb role alternation or the prompt
cache.

A user-typed command still goes through the SAME dangerous-pattern approval
gate the terminal tool uses (``tools.approval.check_all_command_guards``),
reached here through ``tools.terminal_tool._check_all_guards`` so the CLI
approval callback and Docker host-access handling behave identically.

CLI-only by design: gateway/API/cron sessions have their own shells and no
composer, so :func:`bang_shell_enabled` gates the feature off there.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from typing import Optional

USAGE_HINT = "Usage: !<command> — run a shell command without spending a model turn (e.g. !git status)"

# Bang commands are interactive convenience, not agent work. Keep the ceiling
# well under the terminal tool's foreground cap: a user watching output can
# Ctrl+C, and an accidental `!sleep 999` should not wedge the composer.
DEFAULT_TIMEOUT = 120

# How long to keep draining once the shell itself has exited. A backgrounded
# grandchild (`!npm run dev &`) inherits the write end of our stdout pipe via
# fork(), so the pipe can stay open long after the shell is gone — waiting for
# EOF there is waiting for the grandchild. Flush whatever is still arriving,
# then stop. Mirrors the terminal tool's drain in tools/environments/base.py,
# which stops ~300ms after bash exits for exactly this reason.
_DRAIN_IDLE_GRACE = 0.3
_DRAIN_MAX_TAIL = 2.0


def is_bang_command(text: Optional[str]) -> bool:
    """Return True when *text* is a ``!`` shell-mode submission.

    Only a leading ``!`` (after surrounding whitespace) counts. A line that
    merely *contains* ``!`` mid-text (``fix the bug!``, ``echo hi!``) is an
    ordinary prompt and must reach the agent untouched.
    """
    if not isinstance(text, str):
        return False
    return text.strip().startswith("!")


def parse_bang_command(text: str) -> str:
    """Return the shell command inside a bang submission (``""`` when bare).

    ``!ls`` → ``ls``; ``!  ls -la`` → ``ls -la``; ``!!`` → ``!`` (a literal
    second bang is part of the command, e.g. history expansion the user's
    shell will handle); ``!`` alone → ``""``.
    """
    if not isinstance(text, str):
        return ""
    stripped = text.strip()
    if not stripped.startswith("!"):
        return ""
    return stripped[1:].strip()


def bang_shell_enabled() -> bool:
    """True only for interactive local CLI sessions.

    Gateway, API, and cron sessions never reach the composer and their users
    already have a shell; running arbitrary commands for them would be a
    remote-execution surface with no approving human at the keyboard.
    """
    try:
        from utils import env_var_enabled
    except Exception:  # pragma: no cover - utils is always importable in-tree
        def env_var_enabled(name, default=""):  # type: ignore[misc]
            return str(os.getenv(name, default)).strip().lower() in {"1", "true", "yes", "on"}

    if env_var_enabled("HERMES_GATEWAY_SESSION"):
        return False
    if env_var_enabled("HERMES_CRON_SESSION"):
        return False
    if (os.getenv("HERMES_SESSION_PLATFORM") or "").strip():
        return False
    return True


def resolve_bang_cwd(session_key: Optional[str] = None) -> Optional[str]:
    """Return the directory a bang command should run in.

    Mirrors the terminal tool's resolution order so ``!pwd`` matches where the
    agent's own commands land: the session's recorded ``cd`` state first
    (``terminal_tool.get_session_cwd``, updated after every agent command),
    then the configured ``TERMINAL_CWD``/backend default. ``None`` means "let
    the subprocess inherit the process cwd".
    """
    try:
        from tools.terminal_tool import _get_env_config, get_session_cwd

        recorded = get_session_cwd(session_key)
        if recorded:
            return recorded
        configured = (_get_env_config() or {}).get("cwd")
        if configured:
            return configured
    except Exception:
        pass
    return None


def check_bang_approval(command: str) -> dict:
    """Run *command* through the terminal tool's approval gate.

    Reuses ``tools.terminal_tool._check_all_guards`` — the exact function
    ``terminal_tool()`` calls before executing anything — so the hardline
    blocklist, user deny rules, tirith findings, and the interactive
    dangerous-command prompt all apply to user-typed bang commands too. A
    command the agent would need approval for still needs approval when the
    user types it; ``!`` is a latency/cost shortcut, not a security bypass.

    Returns the gate's decision dict (``{"approved": bool, "message": ...}``).
    Falls back to *approved* only when the gate itself cannot be imported,
    which would mean a broken install rather than a policy decision.
    """
    try:
        from tools.terminal_tool import _check_all_guards
    except Exception:
        return {"approved": True, "message": None}

    # env_type mirrors the terminal tool: bang commands always run locally in
    # the CLI process, never inside a remote/sandbox backend.
    return _check_all_guards(command, "local", has_host_access=False)


def _bang_env() -> dict:
    """Environment for a bang command, with Hermes-managed secrets filtered.

    The CLI process holds every configured provider API key in ``os.environ``.
    A bang command is user-typed, but it can still be a third-party script, so
    reuse the same sanitizer ``quick_commands`` and the local terminal backend
    use rather than handing the whole keyring to an arbitrary subprocess.
    """
    try:
        from tools.environments.local import _sanitize_subprocess_env

        return _sanitize_subprocess_env(os.environ.copy())
    except Exception:
        return os.environ.copy()


def run_bang_command(
    command: str,
    *,
    cwd: Optional[str] = None,
    timeout: int = DEFAULT_TIMEOUT,
    writer=None,
) -> int:
    """Execute *command* and stream its output, returning the exit code.

    stdout and stderr are merged and written through *writer* (defaults to
    ``print``) as they arrive, so long-running commands show progress instead
    of buffering to the end. Nothing is returned to a caller for insertion
    into conversation history — the output exists only on the user's terminal.

    ``timeout`` is a wall-clock ceiling, enforced by waiting on the shell while
    a daemon thread drains the pipe. Draining inline instead — ``for line in
    proc.stdout`` — cannot enforce it: that reads to EOF, and EOF only arrives
    once the shell *and every descendant that inherited its write end* have
    closed stdout, so the deadline was only ever applied after the work was
    already over. It is the hang ``tools/environments/base.py`` documents for
    the terminal tool (issue #8340).
    """
    emit = writer or (lambda line: print(line, end="" if line.endswith("\n") else "\n"))

    run_cwd = cwd if (cwd and os.path.isdir(os.path.expanduser(cwd))) else None
    if run_cwd:
        run_cwd = os.path.expanduser(run_cwd)

    try:
        from hermes_cli._subprocess_compat import (
            windows_detach_flags_without_breakaway,
            windows_hide_flags,
        )

        # OR the new-process-group bit INTO the hide flags rather than
        # replacing them — the child still needs CREATE_NO_WINDOW.
        creationflags = windows_hide_flags() | windows_detach_flags_without_breakaway()
    except Exception:
        creationflags = 0

    try:
        # shell=True is intentional and matches quick_commands: this is a
        # command the human typed into their own composer, not model output.
        #
        # start_new_session (POSIX) / CREATE_NEW_PROCESS_GROUP (Windows) give
        # the command its own process group so a timeout or Ctrl+C can take
        # down the whole tree, matching how tools/environments/local.py spawns
        # the terminal tool's commands.
        proc = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=run_cwd,
            env=_bang_env(),
            start_new_session=True,
            creationflags=creationflags,
        )
    except Exception as exc:
        emit(f"!: failed to run command: {exc}")
        return 127

    stop = threading.Event()
    last_output = time.monotonic()

    def _drain(stream) -> None:
        nonlocal last_output
        try:
            for line in stream:
                if stop.is_set():
                    # Control has already returned to the composer, so this
                    # line must not be printed into it. Keep draining anyway
                    # rather than closing the read end: a descendant the user
                    # deliberately left running (`!npm run dev &`) would take
                    # EPIPE on its next write, or block once the pipe filled.
                    # Discarding costs one blocked daemon thread, released as
                    # soon as that descendant exits.
                    continue
                last_output = time.monotonic()
                emit(line.rstrip("\n"))
        except Exception:
            pass
        finally:
            # This thread owns the stream and closes it once the last writer
            # is gone — see the note in the outer finally.
            try:
                stream.close()
            except Exception:
                pass

    reader: Optional[threading.Thread] = None
    if proc.stdout is not None:
        reader = threading.Thread(
            target=_drain, args=(proc.stdout,), name="bang-shell-drain", daemon=True
        )
        reader.start()

    def _flush_tail() -> None:
        """Drain what the shell already wrote, then stop draining.

        Keeps waiting while output is still arriving, so an ordinary tail is
        delivered in full, but gives up once the pipe has gone idle for
        ``_DRAIN_IDLE_GRACE`` — an EOF that depends on a backgrounded
        grandchild may never come at all. ``_DRAIN_MAX_TAIL`` is a hard cap on
        top of that: output still streaming that long after the shell exited
        is coming from a descendant, not the command, and is dropped rather
        than allowed to hold the composer indefinitely.

        The idle window is measured from whichever is later: the last line
        seen, or entry to this function. A command that is silent for a minute
        and then prints on its way out must still get a full grace window, or
        its final line would be raced away.
        """
        if reader is None:
            return
        entered = time.monotonic()
        hard_stop = entered + _DRAIN_MAX_TAIL
        while reader.is_alive():
            now = time.monotonic()
            quiet_since = max(last_output, entered)
            wait = min(_DRAIN_IDLE_GRACE - (now - quiet_since), hard_stop - now)
            if wait <= 0:
                break
            reader.join(timeout=wait)
        stop.set()

    try:
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_bang_process_tree(proc)
            _flush_tail()
            try:
                proc.wait(timeout=_DRAIN_MAX_TAIL)
            except Exception:
                pass
            emit(f"!: command timed out after {timeout}s")
            return 124
        _flush_tail()
    except KeyboardInterrupt:
        # Ctrl+C interrupts the command, not the Hermes session. The command
        # leads its own process group, so signalling only the shell would
        # leave its descendants running as orphans — the same reason
        # tools/environments/base.py kills the group before re-raising.
        _kill_bang_process_tree(proc)
        stop.set()
        emit("!: interrupted")
        return 130
    finally:
        try:
            # The drain thread closes the stream on its way out. Closing it
            # here while that thread is blocked inside read() would deadlock
            # on the buffered reader's lock — the very hang being removed.
            if (reader is None or not reader.is_alive()) and proc.stdout is not None:
                proc.stdout.close()
        except Exception:
            pass

    return int(proc.returncode or 0)


def _kill_bang_process_tree(proc: subprocess.Popen[str]) -> None:
    """Best-effort kill of *proc* and every descendant it spawned.

    ``proc.kill()`` alone signals only the shell wrapper, leaving the
    grandchildren the user actually launched still running — and, because they
    inherited the write end of our stdout pipe, still holding it open. The
    command is spawned into its own process group precisely so the whole group
    can be taken down here, which is also what
    ``tools/environments/local.py::_kill_process`` already does for the
    terminal tool's commands.
    """
    try:
        from hermes_cli._subprocess_compat import _kill_git_process_tree

        # Platform-generic despite the name: ``os.killpg`` on POSIX (only when
        # the child leads its own group, so a shared group is never blasted)
        # and ``taskkill /T /F`` on Windows.
        _kill_git_process_tree(proc)
        return
    except Exception:
        pass
    try:
        proc.kill()
    except Exception:
        pass


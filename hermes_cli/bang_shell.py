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
from typing import Optional

USAGE_HINT = "Usage: !<command> — run a shell command without spending a model turn (e.g. !git status)"

# Bang commands are interactive convenience, not agent work. Keep the ceiling
# well under the terminal tool's foreground cap: a user watching output can
# Ctrl+C, and an accidental `!sleep 999` should not wedge the composer.
DEFAULT_TIMEOUT = 120


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
    """
    emit = writer or (lambda line: print(line, end="" if line.endswith("\n") else "\n"))

    run_cwd = cwd if (cwd and os.path.isdir(os.path.expanduser(cwd))) else None
    if run_cwd:
        run_cwd = os.path.expanduser(run_cwd)

    try:
        from hermes_cli._subprocess_compat import windows_hide_flags

        creationflags = windows_hide_flags()
    except Exception:
        creationflags = 0

    try:
        # shell=True is intentional and matches quick_commands: this is a
        # command the human typed into their own composer, not model output.
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
            creationflags=creationflags,
        )
    except Exception as exc:
        emit(f"!: failed to run command: {exc}")
        return 127

    try:
        if proc.stdout is not None:
            for line in proc.stdout:
                emit(line.rstrip("\n"))
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        emit(f"!: command timed out after {timeout}s")
        return 124
    except KeyboardInterrupt:
        # Ctrl+C interrupts the command, not the Hermes session.
        proc.kill()
        emit("!: interrupted")
        return 130
    finally:
        try:
            if proc.stdout is not None:
                proc.stdout.close()
        except Exception:
            pass

    return int(proc.returncode or 0)


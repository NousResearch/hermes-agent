"""Shared validation for /loop --verify commands.

Loop verification commands run without a shell.  This module centralizes the
argv splitting and shell-syntax rejection used at both create time and runtime
so the two safety boundaries cannot drift independently.
"""

from __future__ import annotations

import re
import shlex


_SHELL_META_RE = re.compile(r"[|&;<>()`$\\\n\r]")


class VerifyCommandError(ValueError):
    """Raised when a loop --verify command cannot be run safely."""


def split_verify_command(command: str) -> list[str]:
    """Validate and split a shell-free /loop --verify command.

    The scheduler passes the returned argv directly to ``subprocess.run`` with
    ``shell=False``.  Rejecting common shell metacharacters keeps user-facing
    semantics honest: ``--verify`` accepts a simple argv-style command, not a
    shell script fragment.
    """
    if _SHELL_META_RE.search(command):
        raise VerifyCommandError(
            "shell metacharacters are not allowed; use a simple argv-style command"
        )
    try:
        argv = shlex.split(command)
    except ValueError as exc:
        raise VerifyCommandError(f"invalid quoting ({exc})") from exc
    if not argv:
        raise VerifyCommandError("empty command")
    return argv

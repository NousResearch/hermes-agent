"""Helpers for user-defined quick command execution."""

from __future__ import annotations

import shlex
from typing import Any


def quick_command_subprocess_args(qcmd: dict[str, Any]) -> tuple[Any, bool]:
    """Return the subprocess payload and shell mode for an exec quick command.

    Quick commands are operator-defined, but they are also reachable from remote
    messaging surfaces. Defaulting to argv execution keeps simple commands easy
    while requiring an explicit ``shell: true`` opt-in for shell syntax.
    """
    command = qcmd.get("command", "")
    use_shell = bool(qcmd.get("shell", False))
    if use_shell:
        return command, True
    if isinstance(command, list):
        return [str(part) for part in command], False
    return shlex.split(str(command)), False

"""Parsing helpers for cron script command strings."""

from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path


def _split_windows_command_line(command: str) -> list[str]:
    """Parse the quoting emitted by ``subprocess.list2cmdline`` into argv."""
    args: list[str] = []
    length = len(command)
    index = 0
    while index < length:
        while index < length and command[index] in " \t":
            index += 1
        if index >= length:
            break

        arg: list[str] = []
        in_quotes = False
        while index < length:
            if command[index] in " \t" and not in_quotes:
                break
            if command[index] == "\\":
                start = index
                while index < length and command[index] == "\\":
                    index += 1
                backslashes = index - start
                if index < length and command[index] == '"':
                    arg.extend("\\" * (backslashes // 2))
                    if backslashes % 2:
                        arg.append('"')
                    else:
                        in_quotes = not in_quotes
                    index += 1
                else:
                    arg.extend("\\" * backslashes)
                continue
            if command[index] == '"':
                in_quotes = not in_quotes
                index += 1
                continue
            arg.append(command[index])
            index += 1

        if in_quotes:
            raise ValueError("No closing quotation")
        args.append("".join(arg))
        while index < length and command[index] in " \t":
            index += 1
    return args


def parse_script_command(
    script_command: str,
    scripts_dir: Path | None = None,
) -> tuple[str, list[str]]:
    """Split a cron script command into its executable path and arguments.

    Existing literal filenames are checked first to preserve legacy jobs whose
    paths contain unquoted spaces. Otherwise the command is split without ever
    invoking a shell. Windows uses its native argv quoting rules so backslashes
    and embedded apostrophes survive a format/parse round trip.
    """
    text = script_command.strip()
    if scripts_dir is not None and text:
        root = scripts_dir.expanduser().resolve()
        raw_literal = Path(text).expanduser()
        literal = (
            raw_literal.resolve()
            if raw_literal.is_absolute()
            else (root / raw_literal).resolve()
        )
        try:
            literal.relative_to(root)
        except ValueError:
            pass
        else:
            if literal.is_file():
                return text, []

    try:
        parts = (
            _split_windows_command_line(text)
            if sys.platform == "win32"
            else shlex.split(text, posix=True)
        )
    except ValueError as exc:
        raise ValueError(f"invalid script command: {exc}") from exc
    if not parts:
        raise ValueError("empty script path")
    return parts[0], parts[1:]


def format_script_command(executable: str, args: list[str]) -> str:
    """Return a stable shell-free storage representation of an argv list."""
    argv = [executable, *args]
    if sys.platform == "win32":
        return subprocess.list2cmdline(argv)
    return shlex.join(argv)

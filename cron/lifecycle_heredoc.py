"""Prove Python stdin boundaries without treating Python bodies as inert data."""

from __future__ import annotations

import re

from tools.shell_heredoc import (
    _find_heredoc_close,
    _parse_heredoc_operator,
    _scan_heredoc_command_unit,
)

# Deliberately only bare interpreters with an explicit stdin operand. Wrappers,
# executable paths, other redirects/options, and compound openers keep the shell
# walk: their spelling alone cannot prove who consumes stdin.
_PYTHON_STDIN = re.compile(r"\s*python(?:[23](?:\.\d+)*)?\s+-\s+(?=<<)")


def split_python_heredocs(command: str) -> tuple[str, list[str]]:
    """Return shell source and Python bodies for proven quoted stdin heredocs.

    The shared shell scanner owns quotes, command units and delimiter matching.
    All heredocs (including non-Python ones) are traversed so a fake opener inside
    another consumer's body cannot establish a language boundary. Any incomplete
    delimiter proof leaves the entire input unchanged. Call only after charging
    the original source against the lifecycle walk budget.
    """
    if "<<" not in command:
        return command, []
    last_opener = command.rfind("<<")
    start = 0
    ranges: list[tuple[int, int]] = []
    bodies: list[str] = []
    while start <= last_opener:
        end, specs, unknown, compound = _scan_heredoc_command_unit(command, start)
        if unknown:
            return command, []
        if not specs:
            start = end + 1
            continue
        if end >= len(command):
            return command, []
        cursor = end + 1
        body_start = cursor
        for delimiter, strip_tabs, _quoted in specs:
            # Empty delimiters have an ambiguous EOF boundary; keep the original walk.
            if not delimiter:
                return command, []
            close = _find_heredoc_close(command, cursor, delimiter, strip_tabs)
            if close is None:
                return command, []
            cursor = close

        opener = command[start:end]
        match = _PYTHON_STDIN.match(opener)
        if match and len(specs) == 1 and specs[0][2] and not compound:
            parsed = _parse_heredoc_operator(opener, match.end())
            # No second redirect, substitution, wrapper tail or command can be
            # hidden by this proof. A trailing shell comment is harmless.
            if parsed is not None:
                tail = opener[parsed[0]:].lstrip(" \t")
                if not tail or tail.startswith("#"):
                    # The shared close finder includes the delimiter line.
                    close_start = command.rfind("\n", body_start, cursor - 1) + 1
                    close_start = max(body_start, close_start)
                    body = command[body_start:close_start]
                    if specs[0][1]:
                        body = "".join(line.lstrip("\t") for line in body.splitlines(keepends=True))
                    bodies.append(body)
                    ranges.append((body_start, cursor))
        start = cursor

    parts: list[str] = []
    previous = 0
    for start, end in ranges:
        parts.extend((command[previous:start], "\n" * command.count("\n", start, end)))
        previous = end
    return "".join(parts) + command[previous:], bodies

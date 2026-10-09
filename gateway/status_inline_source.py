"""Inline-source (``python -c <src> …``) command-line parsing for gateway process identity.

Everything after ``-c`` is data for the inline program, not the process's own identity -- except
Hermes' own bootstraps, which run an entry point in-process and are mapped back to the equivalent
``python -m <module> <argv…>`` (see #107002).
"""

from __future__ import annotations

import re


def inline_source_flag_index(tokens: list[str]) -> int | None:
    """Index of the ``-c`` token when *tokens* is an interpreter running INLINE SOURCE, else None.

    Everything after ``-c`` is data the inline program receives, not this process's own identity.
    The detached gateway restart watcher (``gateway._spawn_gateway_restart_watcher``) is spawned as
    ``python -c <watcher source> <old_pid> <python> -m hermes_cli.main gateway run``: its trailing
    argv is the command the watcher will LATER spawn, so every argv matcher used to read it as a
    live gateway. See #107002 and the "never infer process identity from argv substrings" rule.

    Only interpreter options may precede ``-c``; the first non-option token ends the option block
    (``python -m hermes_cli.main …`` therefore never matches).

    The walk is VALUE-AWARE: ``-X``/``-W``/``-Q`` and ``--check-hash-based-pycs``/``--jit`` take a
    SEPARATE operand, so a naive "first non-option token ends the block" walk mistakes that operand
    for the end of the block and never reaches the ``-c`` behind it (``python -X utf8 -c <src> …``
    was still read as a live gateway). The operand sets are the canonical ones in
    ``hermes_state_holders``, not a second hand-rolled copy.

    *tokens* must be CASE-PRESERVING: the operand-taking ``-Q``/``-W``/``-X`` differ from the
    operand-less ``-q``/``-b``, so a lowercased argv would skip the token after a plain ``-q``.
    """
    from hermes_state_holders import (
        _PYTHON_LONG_OPTIONS_WITH_OPERANDS,
        _PYTHON_SHORT_OPTIONS_WITH_OPERANDS,
    )

    index = 1
    while index < len(tokens):
        token = tokens[index]
        if token == "--":
            return None
        if token in _PYTHON_LONG_OPTIONS_WITH_OPERANDS:
            index += 2  # the next token is this option's operand, not the end of the option block
            continue
        if token.startswith("--"):
            index += 1  # ``--opt=value`` and operand-less long options
            continue
        if not token.startswith("-") or token == "-":
            return None
        # Clustered short options (``-uc``, ``-IsB``). An operand-taking letter consumes the rest of
        # the cluster as its attached value, or the following token when the cluster ends there --
        # so ``-Xc`` is ``-X c``, NOT an inline-source ``-c``. ``-m`` ends the option block with the
        # module name, so the ``c`` in ``-mhermes_cli.main`` is not an inline-source flag either.
        cluster = token[1:]
        for position, letter in enumerate(cluster):
            if letter == "c":
                return index
            if letter == "m":
                return None
            if letter in _PYTHON_SHORT_OPTIONS_WITH_OPERANDS:
                index += 1 if cluster[position + 1 :] else 2
                break
        else:
            index += 1
    return None


def command_line_runs_inline_source(tokens: list[str]) -> bool:
    """True when *tokens* is an interpreter running INLINE SOURCE (``python -c <src> [args]``)."""
    return inline_source_flag_index(tokens) is not None


# Hermes' own inline bootstraps hand control to a Hermes entry point IN this process, so the argv
# they run with is this process's own identity; every other ``-c`` program keeps its trailing argv
# as data (#107002). Each pattern is one emitted source shape, anchored at both ends so a program
# merely CARRYING a bootstrap command line (the restart watcher's respawn argv) never matches.
_Q = r"""['"]?"""
_MAIN = rf"{_Q}__main__{_Q}"
_RUN_MODULE = rf"runpy\.run_module\(\s*{_Q}(?P<target>[\w.]+){_Q}\s*,\s*run_name\s*=\s*{_MAIN}\s*,\s*alter_sys\s*=\s*True\s*\)"
_BOOTSTRAPS = (
    # hermes_cli._launchers.runtime_command (store launcher, the Windows updater's relaunch)
    ("module", re.compile(rf"import os, sys, runpy;.*\b{_RUN_MODULE}", re.DOTALL)),
    # hermes_cli.venv_sync.relaunch_command: argv is assigned inside the source
    ("module", re.compile(rf"import sys, runpy; sys\.path\.insert\(.*\b{_RUN_MODULE}", re.DOTALL)),
    ("path", re.compile(
        rf"import sys, runpy; sys\.path\.insert\(.*\brunpy\.run_path\(\s*{_Q}(?P<target>[^'\"]+?){_Q}\s*,\s*run_name\s*=\s*{_MAIN}\s*\)",
        re.DOTALL)),
    # hermes_cli.venv_sync.relaunch_command re-entering a ``-c`` launcher: the launcher's source is
    # exec'd as a string literal, its argv assigned before it; resolved through the rows above/below
    ("exec", re.compile(r"import sys, runpy; sys\.path\.insert\(.*?;\s*exec\((?P<target>.+)\)", re.DOTALL)),
    # hermes_cli._launchers._launcher_script (the published POSIX shell / Windows .cmd launcher)
    ("entry", re.compile(r"import os, re, sys\s.*\bfrom\s+(?P<target>[\w.]+)\s+import\s+(?P<func>\w+)\b.*\bsys\.exit\(\s*(?P=func)\(\)\s*\)", re.DOTALL)),
    # hermes_cli._launchers._write_cmd_launcher: the launcher script, base64-encoded
    ("base64", re.compile(rf"import base64; exec\(base64\.b64decode\({_Q}(?P<target>[A-Za-z0-9+/=]+){_Q}\)\)")),
)
_ASSIGNED_ARGV = re.compile(r"\bsys\.argv\s*=\s*\[(.*?)\]\s*;")
_EXEC_LAUNCHER = re.compile(
    # no ``\b`` before ``from``/``sys``: an escaped newline normalizes to ``/n`` and abuts them
    r"import os, re, sys\W.*?from\s+(?P<target>[\w.]+)\s+import\s+(?P<func>\w+)\b.*sys\.exit\(\s*(?P=func)\(\)\s*\)", re.DOTALL)
_EXEC_BASE64 = re.compile(r"import base64; exec\(base64\.b64decode\(\W*(?P<b64>[A-Za-z0-9+/=]+)")


def _bootstrap_entry(source: str, argv: list[str]) -> list[str] | None:
    """``[-m, <module>, *argv]`` (or ``[<path>, *argv]``) the inline *source* runs in-process, else None."""
    source = source.strip()
    kind, match = next(((k, m) for k, p in _BOOTSTRAPS if (m := p.fullmatch(source))), (None, None))
    if match is None:
        return None
    target = match["target"]
    if kind == "base64":
        import base64
        import binascii
        try:
            return _bootstrap_entry(base64.b64decode(target, validate=True).decode("utf-8"), argv)
        except (binascii.Error, UnicodeDecodeError):
            return None
    if kind == "exec":
        # The launcher source is a string literal here, and readers normalize its escapes (``\\n`` ->
        # ``/n``), so it is matched in place by the launcher row's anchors, never decoded.
        if assigned := _ASSIGNED_ARGV.search(source):
            argv = [item.strip().strip("'\"") for item in assigned.group(1).split(",")][1:]
        if wrapped := _EXEC_BASE64.search(target):  # the .cmd launcher: its script, base64-encoded
            import base64
            import binascii
            try:
                return _bootstrap_entry(base64.b64decode(wrapped["b64"], validate=True).decode("utf-8"), argv)
            except (binascii.Error, UnicodeDecodeError):
                return None
        launcher = _EXEC_LAUNCHER.search(target)
        if launcher is None:
            return None
        kind, target = "entry", launcher["target"]
    if kind == "entry":  # the launcher script's own ``--run-module <module>`` switch
        return ["-m", argv[1], *argv[2:]] if argv[:1] == ["--run-module"] and len(argv) > 1 else ["-m", target, *argv]
    if assigned := _ASSIGNED_ARGV.search(source):
        argv = [item.strip().strip("'\"") for item in assigned.group(1).split(",")][1:]
    return [target, *argv] if kind == "path" else ["-m", target, *argv]


def inline_bootstrap_argv(tokens: list[str]) -> list[str] | None:
    """*tokens* as the equivalent ``python -m <module> <argv…>`` when this interpreter's ``-c`` source
    is a Hermes bootstrap running an entry point in-process; None for any other inline source.

    Command lines usually arrive space-joined (``/proc``, psutil, ``ps``), which splits the source
    across tokens; the shortest token run that ends in a recognised tail is the source, whatever
    joined it, and the tokens after it are the entry point's argv.
    """
    index = inline_source_flag_index(tokens)
    if index is None:
        return None
    for end in range(index + 1, len(tokens)):
        if tokens[end].rstrip().endswith(")"):
            entry = _bootstrap_entry(" ".join(tokens[index + 1 : end + 1]), tokens[end + 1 :])
            if entry is not None:
                return [tokens[0], *entry]
    return None

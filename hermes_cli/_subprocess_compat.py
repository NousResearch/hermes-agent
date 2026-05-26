"""Windows subprocess compatibility helpers.

Hermes is developed on Linux / macOS and tested natively on Windows too.
Several common subprocess patterns break silently-or-loudly on Windows:

* ``["npm", "install", ...]`` — on Windows ``npm`` is ``npm.cmd``, a batch
  shim.  ``subprocess.Popen(["npm", ...])`` fails with WinError 193
  ("not a valid Win32 application") because CreateProcessW can't run a
  ``.cmd`` file without ``shell=True`` or PATHEXT resolution.

* ``start_new_session=True`` — on POSIX, this maps to ``os.setsid()`` and
  actually detaches the child.  On Windows it's silently ignored; the
  Windows equivalent is ``CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS``
  creationflags, which Python only applies when you pass them explicitly.

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
from typing import Optional, Sequence, Union

__all__ = [
    "IS_WINDOWS",
    "build_cmd_shim_command_line",
    "is_cmd_or_bat_shim",
    "quote_for_cmd_shim",
    "resolve_node_command",
    "safe_subprocess_argv",
    "windows_detach_flags",
    "windows_hide_flags",
    "windows_detach_popen_kwargs",
]


IS_WINDOWS = sys.platform == "win32"


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
# .cmd / .bat shim argument escaping (CVE-2024-24576-class hardening, #31419)
# -----------------------------------------------------------------------------
#
# Background.  When ``subprocess.Popen([target, *args])`` is called on
# Windows and ``target`` ends in ``.cmd`` or ``.bat``, CreateProcessW
# silently launches ``cmd.exe /c <generated-command-line>`` because
# Windows can't directly execute batch files.  cmd.exe then **re-parses**
# that command line with its own metacharacter rules — ``|``, ``&``,
# ``<``, ``>``, ``^``, ``"``, ``(``, ``)``, ``%``, ``!`` are all
# special.  Python's stdlib ``list2cmdline`` only handles the standard
# CRT quoting pass; it doesn't know about cmd.exe's re-parse, so an
# ``arg`` like ``a|b`` (no whitespace, no quotes) round-trips untouched
# through ``list2cmdline`` and gets interpreted by cmd.exe as a pipeline.
# At best the child sees a truncated argv; at worst, untrusted argv
# content (LLM-supplied prompts, JSON quoted from chat, Markdown with
# ``&``, etc.) becomes shell injection.  Issue #31419.
#
# Fix.  When the target is a ``.cmd`` / ``.bat`` shim, build the command
# line ourselves with cmd.exe-aware escaping (caret-prefix the metas,
# CRT-quote the args, then caret-prefix the generated quotes too) and
# pass the resulting STRING to ``subprocess.Popen`` (with
# ``shell=False``).  Popen passes the string straight to CreateProcessW,
# bypassing ``list2cmdline``.  cmd.exe then sees the carets, removes
# them as escape characters, and the batch script gets the original
# metacharacters as literal argv content.
#
# Python 3.13 fixed the underlying issue in stdlib (bpo-114539); we
# require Python >=3.11 in pyproject.toml so the helpers below are
# unconditional belt-and-suspenders for 3.11/3.12 users.

# Cmd.exe metacharacters that re-parse outside double quotes.
# ``%`` and ``!`` are also expanded INSIDE quotes (variable / delayed
# expansion), so we caret-escape them everywhere we caret-escape the
# others — even though they don't strictly break the parse.
_CMD_METACHARACTERS = frozenset('()%!^"<>&|')


def is_cmd_or_bat_shim(path: str) -> bool:
    """Return True iff *path* is a Windows batch shim (.cmd / .bat).

    Case-insensitive — ``Foo.CMD`` matches.  Returns False on POSIX so
    callers don't need their own ``sys.platform`` branch.

    The leading-element extension check matches what
    ``shutil.which()`` returns when it resolves a Node-ecosystem
    command on Windows (``shutil.which("npm")`` → ``"...\\npm.cmd"``).
    """
    if not IS_WINDOWS or not path:
        return False
    lower = path.lower()
    return lower.endswith(".cmd") or lower.endswith(".bat")


def quote_for_cmd_shim(arg: str) -> str:
    """Quote *arg* for safe inclusion in a cmd.exe-bound command line.

    Two-pass escape:

      1. **CRT quoting** — wrap in ``"..."`` and double the backslashes
         that precede embedded ``"``, per the documented CRT argv
         parser rules
         (https://learn.microsoft.com/en-us/cpp/cpp/main-function-command-line-args).
      2. **Cmd.exe meta-escape** — caret-prefix every cmd.exe metacharacter
         (including the quotes we just added) so cmd.exe's outer parse
         pass sees the entire arg as literal token data.

    Empty strings become ``""`` (the documented way to pass an empty
    argv element).  Inputs free of metacharacters AND whitespace are
    returned unchanged so we don't gratuitously alter the static argv
    elements that dominate Hermes's spawn sites.
    """
    if not arg:
        return '""'

    has_meta = any(c in _CMD_METACHARACTERS for c in arg)
    has_space = any(c.isspace() for c in arg)
    if not has_meta and not has_space:
        return arg

    # Pass 1: CRT quoting.
    # Backslashes are only special when they precede a ``"`` — in that
    # case ``n`` backslashes followed by a ``"`` parse as ``n/2`` literal
    # backslashes plus a quote escape (or a closing quote when ``n`` is
    # even).  To produce a literal ``\\`` followed by a literal ``"`` we
    # emit ``2n+1`` backslashes plus ``\"``.
    crt_chars: list[str] = []
    backslashes = 0
    for c in arg:
        if c == '\\':
            backslashes += 1
            continue
        if c == '"':
            # Double pending backslashes, then escape the quote.
            crt_chars.append('\\' * (backslashes * 2 + 1))
            crt_chars.append('"')
            backslashes = 0
            continue
        # Flush pending backslashes as-is, they were not before a quote.
        if backslashes:
            crt_chars.append('\\' * backslashes)
            backslashes = 0
        crt_chars.append(c)
    if backslashes:
        # Trailing backslashes precede the closing ``"`` we'll append below
        # — same doubling rule applies.
        crt_chars.append('\\' * (backslashes * 2))
    crt_quoted = '"' + ''.join(crt_chars) + '"'

    # Pass 2: cmd.exe meta escape.  Every metacharacter (including the
    # outer quotes from pass 1) gets a caret prefix so cmd.exe's outer
    # parse treats it as literal during the re-parse pass.
    return ''.join(
        ('^' + c) if c in _CMD_METACHARACTERS else c
        for c in crt_quoted
    )


def build_cmd_shim_command_line(target: str, args: Sequence[str]) -> str:
    """Build a Windows command-line *string* targeting a .cmd / .bat shim.

    The result is suitable for passing as the first positional argument
    to ``subprocess.Popen(..., shell=False)``.  Each token is wrapped
    via :func:`quote_for_cmd_shim` so neither the CRT parser nor cmd.exe
    can re-interpret embedded metacharacters.

    The target path itself is also escaped — paths under
    ``%PROGRAMFILES%`` or ``%USERPROFILE%`` regularly contain spaces
    and the occasional ``(``/``)`` (e.g. ``Program Files (x86)``).
    """
    parts = [quote_for_cmd_shim(target)]
    parts.extend(quote_for_cmd_shim(a) for a in args)
    return ' '.join(parts)


def safe_subprocess_argv(argv: Sequence[str]) -> Union[list[str], str]:
    """Return a value safe to pass as the first arg of ``subprocess.Popen``.

    On Windows when ``argv[0]`` is a ``.cmd`` / ``.bat`` shim, returns a
    pre-quoted command-line **string** with cmd.exe-aware escaping.
    Call as::

        proc = subprocess.run(safe_subprocess_argv([npm, "install", pkg]), ...)

    so that whichever form is returned is acceptable to ``Popen`` —
    ``Popen`` accepts either a list or a string.

    Elsewhere returns ``list(argv)`` unchanged: POSIX targets and
    Windows ``.exe`` targets go through ``list2cmdline`` /
    ``CreateProcessW`` correctly because there's no cmd.exe re-parse
    pass to worry about.

    Idempotent — passing a list whose head is already an absolute
    ``.cmd`` path produces the same string each time.

    Issue #31419.  Also see Python 3.13's bpo-114539 stdlib fix; this
    helper is the equivalent for 3.11 / 3.12 users.
    """
    seq = list(argv)
    if not seq:
        return seq
    if not is_cmd_or_bat_shim(seq[0]):
        return seq
    return build_cmd_shim_command_line(seq[0], seq[1:])


# -----------------------------------------------------------------------------
# Detached / hidden process creation
# -----------------------------------------------------------------------------


# Win32 CreationFlags — defined here rather than imported from subprocess
# because CREATE_NO_WINDOW and DETACHED_PROCESS aren't guaranteed to be
# present on stdlib subprocess on older Pythons or non-Windows builds.
_CREATE_NEW_PROCESS_GROUP = 0x00000200
_DETACHED_PROCESS = 0x00000008
_CREATE_NO_WINDOW = 0x08000000


def windows_detach_flags() -> int:
    """Return Win32 creationflags that detach a child from the parent
    console and process group.  0 on non-Windows.

    Pair with ``start_new_session=False`` (default) when calling
    subprocess.Popen — on POSIX use ``start_new_session=True`` instead,
    which maps to ``os.setsid()`` in the child.

    Rationale:
    - ``CREATE_NEW_PROCESS_GROUP`` — child has its own process group so
      Ctrl+C in the parent console doesn't propagate.
    - ``DETACHED_PROCESS`` — child has no console at all.  Necessary for
      background daemons (gateway watchers, update respawners) because
      without it, closing the console kills the child.
    - ``CREATE_NO_WINDOW`` — suppress the brief cmd flash that would
      otherwise appear when launching a console app.  Redundant with
      DETACHED_PROCESS but explicit for clarity.
    """
    if not IS_WINDOWS:
        return 0
    return _CREATE_NEW_PROCESS_GROUP | _DETACHED_PROCESS | _CREATE_NO_WINDOW


def windows_hide_flags() -> int:
    """Return Win32 creationflags that merely hide the child's console
    window without detaching the child.  0 on non-Windows.

    Use for short-lived console apps spawned as part of a larger
    operation (``taskkill``, ``where``, version probes) where we want no
    flash but also want to collect stdout/exit code synchronously.

    The key difference from :func:`windows_detach_flags`: NO
    ``DETACHED_PROCESS`` — the child still inherits stdio handles so
    ``capture_output=True`` works.  ``DETACHED_PROCESS`` would sever
    stdio and break stdout capture.
    """
    if not IS_WINDOWS:
        return 0
    return _CREATE_NO_WINDOW


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

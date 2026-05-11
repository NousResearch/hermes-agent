"""
Unified self-relaunch for Hermes CLI.

Preserves critical flags (--tui, --dev, --profile, --model, etc.) across
process replacement so that ``hermes sessions browse`` or post-setup relaunch
doesn't silently drop the user's UI mode or other preferences.

Also works when ``hermes`` is not on PATH (e.g. ``nix run`` or ``python -m``).
"""

import os
import shutil
import sys
from typing import Optional, Sequence

from hermes_cli._parser import (
    PRE_ARGPARSE_INHERITED_FLAGS,
    build_top_level_parser,
)


def _build_inherited_flag_table() -> list[tuple[str, bool]]:
    """Build the ``(option_string, takes_value)`` table of flags that must
    survive a self-relaunch, by introspecting the real parser used by
    ``hermes`` itself.

    A flag participates if its argparse Action carries
    ``inherit_on_relaunch = True`` — set by ``_parser._inherited_flag``.
    """
    parser, _subparsers, chat_parser = build_top_level_parser()

    table: list[tuple[str, bool]] = []
    seen: set[tuple[str, bool]] = set()
    for p in (parser, chat_parser):
        for action in p._actions:
            if not action.option_strings:
                continue  # positional / no flag form
            if not getattr(action, "inherit_on_relaunch", False):
                continue
            takes_value = action.nargs != 0  # store_true/false set nargs=0
            for opt in action.option_strings:
                key = (opt, takes_value)
                if key not in seen:
                    seen.add(key)
                    table.append(key)

    table.extend(PRE_ARGPARSE_INHERITED_FLAGS)
    return table


_INHERITED_FLAGS_TABLE = _build_inherited_flag_table()


def _extract_inherited_flags(argv: Sequence[str]) -> list[str]:
    """Pull out flags that should carry over into a self-relaunched hermes."""
    flags: list[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if "=" in arg:
            key = arg.split("=", 1)[0]
            for flag, _ in _INHERITED_FLAGS_TABLE:
                if key == flag:
                    flags.append(arg)
                    break
            i += 1
            continue

        for flag, takes_value in _INHERITED_FLAGS_TABLE:
            if arg == flag:
                flags.append(arg)
                if takes_value and i + 1 < len(argv) and not argv[i + 1].startswith("-"):
                    flags.append(argv[i + 1])
                    i += 1
                break
        i += 1
    return flags


def resolve_hermes_bin() -> Optional[str]:
    """Find the hermes entry point.

    Priority:
      1. ``sys.argv[0]`` if it resolves to a real executable.
      2. ``shutil.which("hermes")`` on PATH.
      3. ``None`` → caller should fall back to ``python -m hermes_cli.main``.

    Windows note: ``os.access(path, os.X_OK)`` returns True for ``.py`` and
    ``.pyc`` files on Windows (the OS treats anything listed in PATHEXT as
    executable, and Python files are often registered there).  But
    ``subprocess.run([script.py, ...])`` can't actually execute a .py
    directly — CreateProcessW needs a real .exe, not a script associated
    with the Python launcher.  On Windows we therefore skip the argv[0]
    fast-path when it points at a .py file and fall through to either
    ``hermes.exe`` on PATH or the ``sys.executable -m hermes_cli.main``
    fallback.
    """
    argv0 = sys.argv[0]
    _is_windows = sys.platform == "win32"

    def _is_python_script(p: str) -> bool:
        return p.lower().endswith((".py", ".pyc"))

    # Absolute path to an executable (covers nix store, venv wrappers, etc.)
    if os.path.isabs(argv0) and os.path.isfile(argv0) and os.access(argv0, os.X_OK):
        if not (_is_windows and _is_python_script(argv0)):
            return argv0

    # Relative path — resolve against CWD
    if not argv0.startswith("-") and os.path.isfile(argv0):
        abs_path = os.path.abspath(argv0)
        if os.access(abs_path, os.X_OK):
            if not (_is_windows and _is_python_script(abs_path)):
                return abs_path

    # PATH lookup
    path_bin = shutil.which("hermes")
    if path_bin:
        return path_bin

    return None


def build_relaunch_argv(
    extra_args: Sequence[str],
    *,
    preserve_inherited: bool = True,
    original_argv: Optional[Sequence[str]] = None,
) -> list[str]:
    """Construct an argv list for replacing the current process with hermes.

    Args:
        extra_args: Arguments to append (e.g. ``["--resume", id]``).
        preserve_inherited: Whether to carry over UI / behaviour flags
            tagged with ``inherit_on_relaunch`` in the parser.
        original_argv: The original argv to scan for flags (defaults to
            ``sys.argv[1:]``).
    """
    bin_path = resolve_hermes_bin()

    if bin_path:
        argv = [bin_path]
    else:
        argv = [sys.executable, "-m", "hermes_cli.main"]

    src = list(original_argv) if original_argv is not None else list(sys.argv[1:])

    if preserve_inherited:
        argv.extend(_extract_inherited_flags(src))

    argv.extend(extra_args)
    return argv


def relaunch(
    extra_args: Sequence[str],
    *,
    preserve_inherited: bool = True,
    original_argv: Optional[Sequence[str]] = None,
) -> None:
    """Replace the current process with a fresh hermes invocation.

    On POSIX we use ``os.execvp`` which replaces the running process with
    the new one in place — same PID, no double-fork.  That's what the
    relaunch contract wants: "run hermes again as if the user had typed
    the new argv".

    Windows has no native exec semantics — ``os.execvp`` on Windows
    *emulates* exec by spawning the child and exiting the parent, but
    only works when the target is a real Win32 executable.  Our target
    is usually ``hermes.exe`` (a Python console-script shim that wraps
    ``python -m hermes_cli.main``) or a ``.cmd`` batch file, and both
    raise ``OSError(8, "Exec format error")`` on Windows' execvp.

    The Windows-correct pattern is: spawn the child with ``subprocess.run``
    (which routes through ``cmd.exe`` via ``shell=False`` + PATHEXT resolution),
    wait for it to exit, then propagate its exit code via ``sys.exit``.
    That's functionally equivalent — the user sees "hermes exited, then
    new hermes started" — just with two PIDs in play instead of one.
    """
    new_argv = build_relaunch_argv(
        extra_args, preserve_inherited=preserve_inherited, original_argv=original_argv
    )
    if sys.platform == "win32":
        # Windows: subprocess + exit, because execvp can't swap to .cmd/.exe shims.
        import subprocess
        try:
            result = subprocess.run(new_argv)
            sys.exit(result.returncode)
        except KeyboardInterrupt:
            sys.exit(130)
        except OSError as exc:
            # Surface a helpful error rather than the raw OSError — the
            # caller used to see ``[Errno 8] Exec format error`` which is
            # cryptic.  Common causes: ``hermes`` not on PATH yet (install
            # hasn't propagated User PATH into this shell) or a stale shim.
            print(
                f"\nHermes relaunch failed: {exc}\n"
                f"Command: {' '.join(new_argv)}\n"
                f"Fix: open a new terminal so PATH picks up, then re-run hermes.",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        os.execvp(new_argv[0], new_argv)
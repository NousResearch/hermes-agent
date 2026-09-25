"""Unified self-relaunch for Hermes CLI: preserves inherited flags (--tui, --dev, --profile, --model…)
across process replacement so ``hermes sessions browse`` / post-setup relaunch keep the user's mode."""

import os
import pathlib
import sys
from typing import Optional, Sequence

from hermes_cli._parser import (
    PRE_ARGPARSE_INHERITED_FLAGS, build_top_level_parser
)


def _build_inherited_flag_table() -> list[tuple[str, bool]]:
    """``(option_string, takes_value)`` for every parser Action carrying ``inherit_on_relaunch``
    (set by ``_parser._inherited_flag``), plus the pre-argparse flags."""
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
            if any(arg.split("=", 1)[0] == flag for flag, _ in _INHERITED_FLAGS_TABLE):
                flags.append(arg)
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
    """Return an absolute, safe ``sys.argv[0]`` executable, or ``None``.

    ``None`` makes relaunch use :func:`hermes_cli._launchers.current_installation_command`,
    which is independent of the current directory and inherited PATH/Python paths.
    Python launchers are never returned: on Windows a ``.py`` can't be exec'd directly, and on
    POSIX the git installer runs the extensionless source launcher under the managed venv
    interpreter while its ``#!/usr/bin/env python3`` shebang would pick the system Python and
    lose the venv.
    """
    argv0 = sys.argv[0]
    _is_windows = sys.platform == "win32"

    def _is_python_script(p: str) -> bool:
        return p.lower().endswith((".py", ".pyc"))

    def _is_unsafe_python_launcher(p: str) -> bool:
        if _is_windows:
            return _is_python_script(p)
        # A console script pinned to the running venv (``#!<venv>/bin/python``) keeps the venv and
        # stays exec-able; a shebang that resolves elsewhere (``env python3``) loses it.
        from hermes_cli.linux_desktop_entry import _needs_interpreter

        return _needs_interpreter(pathlib.Path(p))

    # Never interpret a bare argv[0] relative to the worker CWD: that directory
    # is often an unrelated Kanban worktree which may contain its own `hermes`.
    if (
        os.path.isabs(argv0) and os.path.isfile(argv0) and os.access(argv0, os.X_OK)
        and not _is_unsafe_python_launcher(argv0)
    ):
        return argv0
    return None


def build_relaunch_argv(
    extra_args: Sequence[str], *, preserve_inherited: bool = True, original_argv: Optional[Sequence[str]] = None
) -> list[str]:
    """Construct an argv list for replacing the current process with this Hermes installation."""
    bin_path = resolve_hermes_bin()
    if bin_path:
        argv = [bin_path]
    else:
        from hermes_cli._launchers import current_installation_command

        argv = current_installation_command()
    src = list(original_argv) if original_argv is not None else list(sys.argv[1:])
    if preserve_inherited:
        argv.extend(_extract_inherited_flags(src))
    argv.extend(extra_args)
    return argv


def relaunch(
    extra_args: Sequence[str], *, preserve_inherited: bool = True, original_argv: Optional[Sequence[str]] = None
) -> None:
    """Replace the current process with a fresh hermes invocation.

    POSIX: ``os.execvp`` in place (same PID, no double-fork). Windows has no real exec — its
    ``execvp`` emulation only works for a real Win32 executable, so spawn + exit instead.
    """
    new_argv = build_relaunch_argv(extra_args, preserve_inherited=preserve_inherited, original_argv=original_argv)
    if sys.platform == "win32":
        import subprocess
        try:
            result = subprocess.run(new_argv)
            sys.exit(result.returncode)
        except KeyboardInterrupt:
            sys.exit(130)
        except OSError as exc:
            # A missing install-bound launcher or a damaged runtime bootstrap is more likely than PATH
            # here: managed relaunches no longer depend on ambient command discovery.
            print(
                f"\nHermes relaunch failed: {exc}\n"
                f"Command: {' '.join(new_argv)}\n"
                f"Fix: repair the Hermes installation, then re-run the command.",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        os.execvp(new_argv[0], new_argv)
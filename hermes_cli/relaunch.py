"""Unified self-relaunch for Hermes CLI: preserves inherited flags (--tui, --dev, --profile, --model…)
across process replacement so ``hermes sessions browse`` / post-setup relaunch keep the user's mode."""

import os
import shutil
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
    """Hermes entry point: ``sys.argv[0]`` if a real executable, else ``which hermes``, else ``None``
    (caller falls back to ``python -m hermes_cli.main``)."""
    argv0 = sys.argv[0]
    _is_windows = sys.platform == "win32"

    def _is_python_script(p: str) -> bool:
        return p.lower().endswith((".py", ".pyc"))

    # Absolute executable (nix store, venv wrappers, …), then relative-to-CWD, then PATH.
    if (
        os.path.isabs(argv0) and os.path.isfile(argv0) and os.access(argv0, os.X_OK)
        and not (_is_windows and _is_python_script(argv0))
    ):
        return argv0
    if not argv0.startswith("-") and os.path.isfile(argv0):
        abs_path = os.path.abspath(argv0)
        if os.access(abs_path, os.X_OK) and not (_is_windows and _is_python_script(abs_path)):
            return abs_path
    return shutil.which("hermes") or None


def build_relaunch_argv(
    extra_args: Sequence[str], *, preserve_inherited: bool = True, original_argv: Optional[Sequence[str]] = None
) -> list[str]:
    """Construct an argv list for replacing the current process with hermes."""
    bin_path = resolve_hermes_bin()
    argv = [bin_path] if bin_path else [sys.executable, "-m", "hermes_cli.main"]
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
            # Raw ``[Errno 8] Exec format error`` is cryptic; usual causes are ``hermes`` not on
            # PATH yet (install hasn't propagated User PATH into this shell) or a stale shim.
            print(
                f"\nHermes relaunch failed: {exc}\n"
                f"Command: {' '.join(new_argv)}\n"
                f"Fix: open a new terminal so PATH picks up, then re-run hermes.",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        os.execvp(new_argv[0], new_argv)
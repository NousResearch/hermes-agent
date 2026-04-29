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


# (option_string, takes_value) — flags whose presence (and value, where
# applicable) on the original argv must survive a self-relaunch.
_CRITICAL_FLAGS: list[tuple[str, bool]] = [
    ("--tui", False),
    ("--dev", False),
    ("--profile", True),
    ("-p", True),
    ("--model", True),
    ("-m", True),
    ("--provider", True),
    ("--yolo", False),
    ("--ignore-user-config", False),
    ("--ignore-rules", False),
    ("--pass-session-id", False),
    ("--accept-hooks", False),
    ("--worktree", False),
    ("-w", False),
    ("--skills", True),
    ("-s", True),
    ("--quiet", False),
    ("-Q", False),
    ("--verbose", False),
    ("-v", False),
    ("--source", True),
]


def _extract_critical_flags(argv: Sequence[str]) -> list[str]:
    """Pull out flags that affect session behaviour / UI mode."""
    flags: list[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if "=" in arg:
            key = arg.split("=", 1)[0]
            for flag, _ in _CRITICAL_FLAGS:
                if key == flag:
                    flags.append(arg)
                    break
            i += 1
            continue

        for flag, takes_value in _CRITICAL_FLAGS:
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
    """
    argv0 = sys.argv[0]

    # Absolute path to an executable (covers nix store, venv wrappers, etc.)
    if os.path.isabs(argv0) and os.path.isfile(argv0) and os.access(argv0, os.X_OK):
        return argv0

    # Relative path — resolve against CWD
    if not argv0.startswith("-") and os.path.isfile(argv0):
        abs_path = os.path.abspath(argv0)
        if os.access(abs_path, os.X_OK):
            return abs_path

    # PATH lookup
    path_bin = shutil.which("hermes")
    if path_bin:
        return path_bin

    return None


def build_relaunch_argv(
    extra_args: Sequence[str],
    *,
    preserve_critical: bool = True,
    original_argv: Optional[Sequence[str]] = None,
) -> list[str]:
    """Construct an argv list for replacing the current process with hermes.

    Args:
        extra_args: Arguments to append (e.g. ``["--resume", id]``).
        preserve_critical: Whether to carry over UI / behaviour flags.
        original_argv: The original argv to scan for flags (defaults to
            ``sys.argv[1:]``).
    """
    bin_path = resolve_hermes_bin()

    if bin_path:
        argv = [bin_path]
    else:
        argv = [sys.executable, "-m", "hermes_cli.main"]

    src = list(original_argv) if original_argv is not None else list(sys.argv[1:])

    if preserve_critical:
        argv.extend(_extract_critical_flags(src))

    argv.extend(extra_args)
    return argv


def relaunch(
    extra_args: Sequence[str],
    *,
    preserve_critical: bool = True,
    original_argv: Optional[Sequence[str]] = None,
) -> None:
    """Replace the current process with a fresh hermes invocation."""
    new_argv = build_relaunch_argv(
        extra_args, preserve_critical=preserve_critical, original_argv=original_argv
    )
    os.execvp(new_argv[0], new_argv)


def relaunch_chat(
    *,
    preserve_critical: bool = True,
    original_argv: Optional[Sequence[str]] = None,
) -> None:
    """Convenience wrapper: relaunch into ``hermes chat``."""
    relaunch(["chat"], preserve_critical=preserve_critical, original_argv=original_argv)

"""Windows UTF-8 bootstrap for Hermes entry points (no-op on POSIX).

Windows binds stdio to the console code page (cp1252), so ``print("café")`` raises
``UnicodeEncodeError``, and Python children inherit the same default unless
``PYTHONUTF8``/``PYTHONIOENCODING`` are set. Import this module first in every entry
point (``hermes``, ``hermes-agent``, ``hermes-acp``, ``gateway.run``, ``batch_runner``,
``cron/scheduler``). It does NOT re-exec with ``-X utf8``: ``open()`` in the current
process still needs an explicit ``encoding="utf-8"`` (ruff ``PLW1514``). POSIX is left
alone deliberately — users' ``LANG``/``LC_*`` choices are respected.
"""

from __future__ import annotations

import os
import sys

_IS_WINDOWS = sys.platform == "win32"
_bootstrap_applied = False


def apply_windows_utf8_bootstrap() -> bool:
    """Apply the Windows UTF-8 bootstrap once; True only when it was applied this call."""
    global _bootstrap_applied

    if not _IS_WINDOWS or _bootstrap_applied:
        return False

    # setdefault() so a user can opt out with PYTHONUTF8=0 / PYTHONIOENCODING=...
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    # os.environ changes don't rebind streams bound at interpreter startup, so
    # reconfigure them in-process. errors="replace" keeps a non-UTF-8 legacy
    # pipe on stdin from crashing us (U+FFFD instead of an exception).
    # Non-TextIOWrapper streams (BytesIO in tests, embedded hosts) have no
    # reconfigure(): skip — the env-var fix for children is the bigger win.
    for stream_name in ("stdout", "stderr", "stdin"):
        reconfigure = getattr(getattr(sys, stream_name, None), "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass  # closed, or replaced with something non-reconfigurable

    _bootstrap_applied = True
    return True


def suppress_platform_ver_console() -> None:
    """Stub ``platform._syscmd_ver`` on Windows — decode-crash + console-flash guard.

    ``platform.win32_ver()`` (reached via ``platform.platform()``, which the OpenAI SDK
    calls) shells out ``cmd /c ver`` with ``shell=True`` and no ``CREATE_NO_WINDOW``: a
    windowless parent (pythonw gateway, slash/kanban workers) flashes a console per call,
    and Python 3.11.0/3.11.1 (no ``encoding="locale"`` fix) strict-utf-8-decodes the OEM
    code page output under PEP 540 mode and raises (#69413). Returning the inputs makes
    ``win32_ver()`` fall back to ``sys.getwindowsversion()`` — same data, no subprocess.
    Mirrors ``hermes_cli._subprocess_compat.suppress_platform_ver_console`` for callers
    that never import ``hermes_cli.main``; double application is harmless.
    """
    if not _IS_WINDOWS:
        return
    try:
        import platform

        if hasattr(platform, "_syscmd_ver"):
            def _quiet_syscmd_ver(system="", release="", version="",
                                  supported_platforms=("win32", "win16", "dos")):
                return system, release, version

            platform._syscmd_ver = _quiet_syscmd_ver
    except Exception:
        pass  # hardening only — never break an entry point


def harden_import_path(src_root: str | None = None) -> None:
    """Stop a package in the current directory from shadowing Hermes modules.

    Hermes ships top-level modules with common names (``utils``, ``proxy``, ``ui``); a
    project with its own ``utils/`` launched from its directory would win the import.
    The cwd reaches ``sys.path`` as ``""``/``"."`` (script/``-m`` launches) AND as an
    absolute path (venv activation, PYTHONPATH), so both are handled: relative forms are
    dropped and the Hermes root is *relocated* to the front, not merely inserted when
    absent. ``src_root`` defaults to this module's directory (the repo root for every
    shipped entry point), so no spawner env var is required.
    """
    root = src_root or os.environ.get("HERMES_PYTHON_SRC_ROOT") or os.path.dirname(
        os.path.abspath(__file__)
    )

    sys.path[:] = [p for p in sys.path if p not in ("", ".")]

    root_abs = os.path.abspath(root)
    sys.path[:] = [p for p in sys.path if os.path.abspath(p) != root_abs]
    sys.path.insert(0, root)


# Apply on import — entry points just need ``import hermes_bootstrap``
# (or ``from hermes_bootstrap import apply_windows_utf8_bootstrap``) at
# the very top of their module, before importing anything else.  The
# import side effect does the right thing.
apply_windows_utf8_bootstrap()
suppress_platform_ver_console()

# Every entry point imports this module before its dependency graph.
from pathlib import Path
from hermes_cli.runtime_paths import activate_dependencies
from hermes_cli._early_recovery import recover_if_needed

from hermes_cli._parser import command_argv

_root = Path(__file__).resolve().parent
# Repair needs only stdlib. Do not activate the damaged tree to reach it.
_pm_repair = command_argv(sys.argv[1:])[:2] == ["pm", "repair"]
if not _pm_repair:
    from hermes_cli.venv_sync import prepare_launch, relaunch_command

    try:
        _launch_python = prepare_launch(_root, sys.argv[1:])
        if _launch_python is not None:
            _main_spec = getattr(sys.modules.get("__main__"), "__spec__", None)
            _command = relaunch_command(
                _launch_python, _root, sys.argv, sys.orig_argv,
                getattr(_main_spec, "name", None),
            )
            if os.name == "nt":
                import subprocess

                raise SystemExit(subprocess.call(_command))
            os.execv(str(_launch_python), _command)
    except Exception as exc:
        print(f"hermes: source-update completion failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    recover_if_needed(_root)
    try:
        activate_dependencies(_root)
    except (RuntimeError, OSError) as exc:
        if command_argv(sys.argv[1:])[:1] != ["pm"]:
            print(f"hermes: {exc}; run `hermes pm repair`", file=sys.stderr)
            raise SystemExit(1) from None

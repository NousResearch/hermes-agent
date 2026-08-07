"""Shared resolution of the Hermes CLI executable for spawn/service paths.

Every daemon, spawn and service-definition generator used to hardcode
``python -m hermes_cli.main``. Executing the module as ``__main__`` means a
later ``from hermes_cli.main import ...`` cannot find it under its real name
in ``sys.modules``, so the import machinery re-executes the module body —
including its import-time side effects (see #76705).

This helper resolves the ``hermes`` console script instead, falling back to
the ``-m`` form only when no shim is available.  For service definitions the
interpreter-bound sibling (``Path(interpreter).with_name("hermes")``) is
preferred over ``shutil.which``: a launchd/systemd job runs with a minimal
PATH, so a PATH lookup can miss the shim entirely or resolve a *different*
Hermes install than the interpreter that generated the unit.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def _interpreter_sibling(interpreter: str) -> str | None:
    """Return the console-script shim next to ``interpreter``, if any.

    On Windows pip installs ``hermes.exe`` next to ``python.exe``; on POSIX
    the shim is a bare ``hermes`` script.
    """
    interp = Path(interpreter)
    names = ["hermes", "hermes.exe"] if sys.platform == "win32" else ["hermes"]
    for name in names:
        candidate = interp.with_name(name)
        if candidate.is_file():
            if sys.platform == "win32" or os.access(candidate, os.X_OK):
                return str(candidate)
    return None


def resolve_hermes_argv(
    interpreter: str | None = None,
    *,
    prefer_interpreter_sibling: bool = False,
) -> list[str]:
    """Resolve the Hermes CLI invocation as argv parts.

    Priority:
      1. ``Path(interpreter).with_name("hermes")`` — the console-script shim
         belonging to the exact interpreter/venv that is running (or, for a
         remapped system-unit interpreter, the target user's shim).  This is
         interpreter-bound and independent of PATH, which matters for
         launchd/systemd jobs that run with a minimal PATH.
      2. ``shutil.which("hermes")`` — PATH lookup (interactive spawn paths).
         On Windows, implicit ``.cmd`` / ``.bat`` shims are declined because
         they are not safe as argv[0] for Popen/exec.
      3. ``[interpreter, "-m", "hermes_cli.main"]`` — fallback when no shim
         resolves (pip ``--target`` installs, zipapps, embedded
         interpreters).

    With ``prefer_interpreter_sibling=True`` (service definitions), the PATH
    lookup is skipped entirely: a launchd/systemd job runs with a minimal
    PATH, so ``shutil.which`` can miss the shim or resolve a *different*
    Hermes install than the interpreter that generated the unit.  The
    sibling is the interpreter-bound answer, and ``-m`` is the fallback.

    Returns argv parts ready for ``Popen`` / ``execvpe`` / plist emission.
    """
    interp = interpreter or sys.executable

    sibling = _interpreter_sibling(interp)
    if sibling:
        return [sibling]

    if not prefer_interpreter_sibling:
        path_bin = shutil.which("hermes")
        if path_bin and not (
            sys.platform == "win32" and path_bin.lower().endswith((".cmd", ".bat"))
        ):
            return [path_bin]

    return [interp, "-m", "hermes_cli.main"]

"""Lightweight installation-bound Python command construction.

This module intentionally imports only the standard library so callers below
package-management layers can construct a source-tree bootstrap command.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Bootstrap pins the default home after repairing an interrupted checkout update.
PIN_DEFAULT_HOME_FLAG = "_hermes_pin_default_home"


def _inline_string_literal(value: str) -> str:
    """Keep inline Python source intact through Windows PowerShell's native argv quoting."""
    if os.name != "nt":
        return repr(value)
    escaped = value.encode("unicode_escape").decode("ascii")
    return "'" + escaped.replace("'", "\\x27").replace('"', "\\x22") + "'"


def _published_posix_launcher(repo_root: Path) -> str | None:
    """Prefer this install's executable shim, never a PATH or Windows batch shim."""
    shim = Path(repo_root).resolve() / ".hermes" / "bin" / "hermes"
    if shim.is_file() and os.access(shim, os.X_OK) and not str(shim).lower().endswith((".cmd", ".bat")):
        return str(shim)
    return None


def isolated_hermes_argv(repo_root: Path, *, python: str | Path | None = None) -> list[str]:
    """Isolated Python needs a source-bound launcher even from an unrelated cwd."""
    launcher = _published_posix_launcher(repo_root)
    return [launcher] if launcher else bootstrap_runtime_command(repo_root, python=python)


def bootstrap_runtime_command(
    repo_root: Path,
    args=(),
    *,
    module: str = "hermes_cli.main",
    code: str | None = None,
    python: str | Path | None = None,
    home: str | Path | None = None,
) -> list[str]:
    """Return an isolated command bound to ``repo_root`` and its bootstrap."""
    root = Path(repo_root).resolve()
    python = python or Path(sys.executable)
    entry = f"exec({_inline_string_literal(code)})" if code is not None else (
        f"runpy.run_module({_inline_string_literal(module)}, run_name='__main__', alter_sys=True)")
    # A literal home is pinned up front; the default one needs ``hermes_constants`` from the
    # checkout, so it is pinned only after ``hermes_bootstrap``'s launch-time repair, here again
    # for a bootstrap that predates that hook.
    if home is not None:
        pin, settle = (f"os.environ['HERMES_HOME'] = os.environ.get('HERMES_HOME') or "
                       f"{_inline_string_literal(str(home))}; "), ""
    else:
        pin = f"sys.{PIN_DEFAULT_HOME_FLAG} = True; "
        settle = ("os.environ.get('HERMES_HOME') or os.environ.__setitem__('HERMES_HOME', "
                  "str(__import__('hermes_constants').get_default_hermes_root())); ")
    bootstrap = (
        "import os, sys, runpy; "
        "os.environ.pop('PYTHONHOME', None); os.environ.pop('PYTHONPATH', None); "
        "os.environ.pop('VIRTUAL_ENV', None); "
        f"sys.path.insert(0, {_inline_string_literal(str(root))}); "
        + pin
        + "import hermes_bootstrap; "
        + settle
        + entry
    )
    return [str(python), "-I", "-c", bootstrap, *args]

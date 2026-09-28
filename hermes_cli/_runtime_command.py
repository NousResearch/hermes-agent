"""Lightweight installation-bound Python command construction.

This module intentionally imports only the standard library so callers below
package-management layers can construct a source-tree bootstrap command.
"""

from __future__ import annotations

import sys
from pathlib import Path


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
    entry = (
        f"exec({code!r})"
        if code is not None
        else f"runpy.run_module({module!r}, run_name='__main__', alter_sys=True)"
    )
    default_home = (
        f"{str(home)!r}"
        if home is not None
        else "str(__import__('hermes_constants').get_default_hermes_root())"
    )
    bootstrap = (
        "import os, sys, runpy; "
        "os.environ.pop('PYTHONHOME', None); os.environ.pop('PYTHONPATH', None); "
        "os.environ.pop('VIRTUAL_ENV', None); "
        f"sys.path.insert(0, {str(root)!r}); "
        f"os.environ['HERMES_HOME'] = os.environ.get('HERMES_HOME') or {default_home}; "
        "import hermes_bootstrap; "
        + entry
    )
    return [str(python), "-I", "-c", bootstrap, *args]

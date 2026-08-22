"""Scoped shim that intercepts only ``python -m hermes_cli.main``.

The Desktop prepends this package directory to ``PYTHONPATH``. The shim locates
but deliberately does not execute the real Hermes package until ``main.py`` has
installed the platform process authority. No other Python invocation is
changed because the path is supplied only to Desktop backend launches.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_SHIM_PACKAGE = Path(__file__).resolve().parent
_BOOTSTRAP_ROOT = _SHIM_PACKAGE.parent
_REAL_PACKAGE: Path | None = None

for raw_entry in sys.path:
    if not raw_entry:
        continue
    try:
        candidate = (Path(raw_entry).resolve() / "hermes_cli").resolve()
    except (OSError, RuntimeError):
        continue
    if candidate == _SHIM_PACKAGE:
        continue
    if (candidate / "__init__.py").is_file() and (candidate / "main.py").is_file():
        _REAL_PACKAGE = candidate
        break

if _REAL_PACKAGE is None:
    raise ImportError("Desktop authority bootstrap could not locate the real hermes_cli package")

# Let the scoped main shim import the authority implementation from the real
# package without running the real package initializer first.
__path__ = [str(_SHIM_PACKAGE), str(_REAL_PACKAGE)]
if __spec__ is not None:
    __spec__.submodule_search_locations = list(__path__)

DESKTOP_BOOTSTRAP_ROOT = str(_BOOTSTRAP_ROOT)
DESKTOP_REAL_PACKAGE = str(_REAL_PACKAGE)
DESKTOP_REAL_INIT = str(_REAL_PACKAGE / "__init__.py")
DESKTOP_REAL_MAIN = str(_REAL_PACKAGE / "main.py")

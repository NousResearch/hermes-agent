"""Backwards-compatible import surface for the Hermes classic CLI package."""

from __future__ import annotations

import sys
import importlib
from pathlib import Path

from . import app as _app

# Preserve the old ``cli.py`` behavior: ``import cli`` should expose the same
# module globals that implementation code uses.  Mark ``cli.app`` as package-like
# so ``cli.constants`` / ``cli.input`` remain importable after this alias.
_app.__path__ = [str(Path(__file__).resolve().parent)]  # type: ignore[attr-defined]
_app.__package__ = __name__
_app.app = _app  # type: ignore[attr-defined]
for _submodule in ("commands", "constants", "rendering", "tui"):
    setattr(
        _app,
        _submodule,
        sys.modules.get(f"{__name__}.{_submodule}")
        or importlib.import_module(f"{__name__}.{_submodule}"),
    )
_app.__dict__.pop("input", None)
sys.modules[__name__] = _app

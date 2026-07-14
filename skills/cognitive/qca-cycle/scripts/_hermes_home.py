"""Resolve HERMES_HOME for standalone skill scripts.

Skill scripts may run outside the Hermes process (e.g. system Python,
nix env, CI) where ``hermes_constants`` is not importable.  This module
provides the same ``get_hermes_home()`` contract as ``hermes_constants``
without requiring it on ``sys.path``.

When ``hermes_constants`` IS available it is used directly so that any
future enhancements (profile resolution, Docker detection, etc.) are
picked up automatically.  The fallback path replicates the core logic
from ``hermes_constants.py`` using only the stdlib, including the
platform-native Windows default.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    from hermes_constants import get_hermes_home as get_hermes_home
except (ModuleNotFoundError, ImportError):

    def get_hermes_home() -> Path:
        """Return the Hermes home directory (platform-native default).

        Mirrors ``hermes_constants.get_hermes_home()``."""
        val = os.environ.get("HERMES_HOME", "").strip()
        if val:
            return Path(val)
        if sys.platform == "win32":
            local_appdata = os.environ.get("LOCALAPPDATA", "").strip()
            base = Path(local_appdata) if local_appdata else Path.home() / "AppData" / "Local"
            return base / "hermes"
        return Path.home() / ".hermes"

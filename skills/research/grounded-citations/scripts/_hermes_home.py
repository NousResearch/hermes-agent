"""Resolve HERMES_HOME for standalone skill scripts.

Skill scripts may run outside the Hermes process (system Python, nix env,
CI) where ``hermes_constants`` is not importable.  This module provides the
same ``get_hermes_home()`` contract without requiring it on ``sys.path``.

When ``hermes_constants`` IS available it is used directly so profile
resolution and any future enhancements are picked up automatically.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    from hermes_constants import get_hermes_home as get_hermes_home
except (ModuleNotFoundError, ImportError):

    def _platform_default_home() -> Path:
        """Mirror ``hermes_constants._get_platform_default_hermes_home``."""
        if sys.platform == "win32":
            local_appdata = os.environ.get("LOCALAPPDATA", "").strip()
            base = Path(local_appdata) if local_appdata else Path.home() / "AppData" / "Local"
            return base / "hermes"
        return Path.home() / ".hermes"

    def get_hermes_home() -> Path:
        """Return the Hermes home directory (``HERMES_HOME`` or the platform default)."""
        val = os.environ.get("HERMES_HOME", "").strip()
        return Path(val) if val else _platform_default_home()

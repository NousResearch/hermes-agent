"""Resolve RABBIT_HOME for standalone skill scripts.

Skill scripts may run outside the Rabbit process (system Python, nix env,
CI) where ``rabbit_constants`` is not importable.  This module provides the
same ``get_rabbit_home()`` contract without requiring it on ``sys.path``.

When ``rabbit_constants`` IS available it is used directly so profile
resolution and any future enhancements are picked up automatically.
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    from rabbit_constants import get_rabbit_home as get_rabbit_home
except (ModuleNotFoundError, ImportError):

    def get_rabbit_home() -> Path:
        """Return the Rabbit home directory (default: ``~/.rabbit``)."""
        val = os.environ.get("RABBIT_HOME", "").strip()
        return Path(val) if val else Path.home() / ".rabbit"

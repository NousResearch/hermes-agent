"""Resolve RABBIT_HOME for standalone skill scripts.

Skill scripts may run outside the Rabbit process (e.g. system Python,
nix env, CI) where ``rabbit_constants`` is not importable.  This module
provides the same ``get_rabbit_home()`` and ``display_rabbit_home()``
contracts as ``rabbit_constants`` without requiring it on ``sys.path``.

When ``rabbit_constants`` IS available it is used directly so that any
future enhancements (profile resolution, Docker detection, etc.) are
picked up automatically.  The fallback path replicates the core logic
from ``rabbit_constants.py`` using only the stdlib.

All scripts under ``google-workspace/scripts/`` should import from here
instead of duplicating the ``RABBIT_HOME = Path(os.getenv(...))`` pattern.
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    from rabbit_constants import display_rabbit_home as display_rabbit_home
    from rabbit_constants import get_rabbit_home as get_rabbit_home
except (ModuleNotFoundError, ImportError):

    def get_rabbit_home() -> Path:
        """Return the Rabbit home directory (default: ~/.rabbit).

        Mirrors ``rabbit_constants.get_rabbit_home()``."""
        val = os.environ.get("RABBIT_HOME", "").strip()
        return Path(val) if val else Path.home() / ".rabbit"

    def display_rabbit_home() -> str:
        """Return a user-friendly ``~/``-shortened display string.

        Mirrors ``rabbit_constants.display_rabbit_home()``."""
        home = get_rabbit_home()
        try:
            return "~/" + home.relative_to(Path.home()).as_posix()
        except ValueError:
            return str(home)

"""Resolve HERMES_HOME for standalone Microsoft Graph skill scripts."""

from __future__ import annotations

import os
from pathlib import Path

try:
    from hermes_constants import display_hermes_home as display_hermes_home
    from hermes_constants import get_hermes_home as get_hermes_home
except (ModuleNotFoundError, ImportError):

    def _fallback_hermes_home() -> Path:
        return (Path.home() / ".hermes").resolve()

    def get_hermes_home() -> Path:
        val = os.environ.get("HERMES_HOME", "").strip()
        if not val:
            return _fallback_hermes_home()

        try:
            candidate = Path(val).expanduser()
            if not candidate.is_absolute():
                candidate = Path.cwd() / candidate
            return candidate.resolve()
        except OSError:
            return _fallback_hermes_home()

    def display_hermes_home() -> str:
        home = get_hermes_home()
        try:
            return "~/" + str(home.relative_to(Path.home().resolve()))
        except ValueError:
            return str(home)
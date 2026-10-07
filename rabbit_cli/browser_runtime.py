"""Read-only full-Chromium selection shared by browser launchers."""

from __future__ import annotations

import os

import pm


def chromium_executable(*, allow_override: bool = True) -> str | None:
    """Prefer an explicit override unless the caller needs PM's own Chromium."""
    override = os.environ.get("AGENT_BROWSER_EXECUTABLE_PATH") if allow_override else None
    if override:
        return override
    installed = pm.installed_package("chromium")
    return str(installed.binary) if installed and installed.binary is not None else None

"""Read-only full-Chromium selection shared by browser launchers."""

from __future__ import annotations

import os

import pm


def chromium_executable() -> str | None:
    """Prefer an explicit override, then PM; None leaves resolution to Playwright."""
    override = os.environ.get("AGENT_BROWSER_EXECUTABLE_PATH")
    if override:
        return override
    installed = pm.installed_package("chromium")
    return str(installed.binary) if installed and installed.binary is not None else None

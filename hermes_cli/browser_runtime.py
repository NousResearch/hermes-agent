"""Read-only full-Chromium selection shared by browser launchers."""

from __future__ import annotations

import os

import pm


def chromium_executable(*, allow_override: bool = True) -> str | None:
    """Prefer a usable explicit override unless the caller needs PM's Chromium.

    A stale override must not mask an installed managed browser.  This resolver
    feeds both readiness checks and the agent-browser command environment.
    """
    override = os.environ.get("AGENT_BROWSER_EXECUTABLE_PATH") if allow_override else None
    if override and os.path.isfile(override) and (os.name == "nt" or os.access(override, os.X_OK)):
        return override
    installed = pm.installed_package("chromium")
    return str(installed.binary) if installed and installed.binary is not None else None

"""Shared log sanitization for standalone Microsoft Graph skill scripts."""

from __future__ import annotations

try:
    from copilot_remote.router import _sanitize_for_log as _sanitize_for_log
except (ImportError, ModuleNotFoundError):

    def _sanitize_for_log(value) -> str:
        if value is None:
            return ""
        text = str(value)
        return "".join(" " if (ord(char) < 0x20 or ord(char) == 0x7F) else char for char in text)
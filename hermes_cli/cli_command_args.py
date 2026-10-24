"""Small argument helpers shared by the CLI command mixins (split out of cli_commands_mixin)."""

from __future__ import annotations


def _take_flag(parts: list, flag: str):
    """Pop ``flag VALUE`` out of ``parts``: ``(rest, value, ok)``; ok=False when the value is
    missing (caller prints usage)."""
    if flag not in parts:
        return parts, None, True
    idx = parts.index(flag)
    if idx + 1 >= len(parts):
        return parts, None, False
    return parts[:idx] + parts[idx + 2:], parts[idx + 1], True

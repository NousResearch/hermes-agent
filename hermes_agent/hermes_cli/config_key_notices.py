"""Notices ``hermes config set`` prints for a key path the schema does not know."""

from __future__ import annotations

from typing import Optional

from hermes_cli.colors import Colors, color


def unknown_subkey_refusal(key: str, suggestion: Optional[str]) -> str:
    """The refusal for a known key given under the wrong prefix (``gateway.discord.foo``)."""
    lines = [color(f"✗ '{key}' is not a recognized config key — nothing was written.", Colors.RED)]
    if suggestion:
        lines.append(color(f"  Did you mean: {suggestion}", Colors.YELLOW))
    lines.append(color(
        "  (Custom top-level keys are supported; use --force to write this path anyway.)", Colors.DIM))
    return "\n".join(lines)

"""Notices ``hermes config set`` / ``add`` / ``remove`` print for a key path the schema does not know.

``force_hint=False`` is for the list commands, which have no ``--force`` to point at.
"""

from __future__ import annotations

from typing import Optional

from hermes_cli.colors import Colors, color


def unknown_subkey_refusal(key: str, suggestion: Optional[str], *, force_hint: bool = True) -> str:
    """The refusal for a known key given under the wrong prefix (``gateway.discord.foo``)."""
    lines = [color(f"✗ '{key}' is not a recognized config key — nothing was written.", Colors.RED)]
    if suggestion:
        lines.append(color(f"  Did you mean: {suggestion}", Colors.YELLOW))
    if force_hint:
        lines.append(color(
            "  (Custom top-level keys are supported; use --force to write this path anyway.)", Colors.DIM))
    return "\n".join(lines)


def print_unknown_key_notice(key: str, suggestion: Optional[str], *, force_hint: bool = True) -> None:
    """Post-write notice (#34067): the value IS saved, but the runtime may never read it."""
    from hermes_cli.config import _split_key_path

    print(color(
        f"⚠ '{key}' is not a recognized config key — it was saved anyway, "
        "but Hermes may not read it.", Colors.YELLOW))
    if suggestion:
        print(color(f"  Did you mean: {suggestion}", Colors.YELLOW))
    skip = " Use --force to skip this notice." if force_hint else ""
    # The env bridge covers custom TOP-LEVEL keys only; an unseeded nested path (``stt.provider``)
    # is written but not bridged, so the footer would be a false promise there.
    if len(_split_key_path(key)) == 1:
        print(color(
            "  (Custom top-level keys are supported and bridged to the "
            f"environment for skills/external tools.{skip})", Colors.DIM))
    elif skip:
        print(color(f"  ({skip.strip()})", Colors.DIM))

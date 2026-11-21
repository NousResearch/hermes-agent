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


def unknown_key_notice(key: str, suggestion: Optional[str]) -> None:
    """Say a saved key is not in the schema, and aim at the likely-intended path.

    The value IS written either way (#34067); the runtime may never read it."""
    from hermes_cli.config import _split_key_path  # late: config imports this module

    print(color(
        f"⚠ '{key}' is not a recognized config key — it was saved anyway, "
        "but Hermes may not read it.", Colors.YELLOW))
    if suggestion:
        print(color(f"  Did you mean: {suggestion}", Colors.YELLOW))
    # The env bridge covers custom TOP-LEVEL keys only; an unseeded nested path (``stt.provider``)
    # is written but not bridged, so the footer would be a false promise there.
    if len(_split_key_path(key)) == 1:
        print(color(
            "  (Custom top-level keys are supported and bridged to the "
            "environment for skills/external tools. Use --force to skip "
            "this notice.)", Colors.DIM))
    else:
        print(color("  (Use --force to skip this notice.)", Colors.DIM))

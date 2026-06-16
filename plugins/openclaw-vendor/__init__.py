"""Hermes plugin bridge for in-tree vendor/openclaw-mirror (OpenClaw extensions + packages)."""

from __future__ import annotations

from .cli import openclaw_vendor_command, register_cli


def register(ctx) -> None:
    """Register ``hermes openclaw-vendor`` — syncs extension skills; tools stay in core toolsets."""
    ctx.register_cli_command(
        name="openclaw-vendor",
        help="Install and manage OpenClaw vendor mirror skills and readiness",
        setup_fn=register_cli,
        handler_fn=openclaw_vendor_command,
        description=(
            "Link skills from vendor/openclaw-mirror/extensions (hypura-harness, etc.) "
            "into your Hermes skills directory and report package/tool readiness for "
            "AI-Scientist, ShinkaEvolve, and VRChat/harness integrations. "
            "Does not clone vendor — mirror must exist in the Hermes checkout. "
            "After install, run /skills reload and enable openclaw/vrchat toolsets."
        ),
    )

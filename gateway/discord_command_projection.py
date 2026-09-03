"""Compatibility import for the shared Discord command projection package."""

from gateway.discord_projection import (
    DiscordCommandProjection,
    DiscordProjectedCommand,
    DiscordProjectionError,
    DiscordProjectionMismatch,
    build_relay_discord_manifest,
    build_relay_discord_projection,
    canonicalize_discord_command,
    canonicalize_discord_option,
    project_discord_commands,
    verify_discord_projection_readback,
)

__all__ = [
    "DiscordCommandProjection",
    "DiscordProjectedCommand",
    "DiscordProjectionError",
    "DiscordProjectionMismatch",
    "build_relay_discord_manifest",
    "build_relay_discord_projection",
    "canonicalize_discord_command",
    "canonicalize_discord_option",
    "project_discord_commands",
    "verify_discord_projection_readback",
]

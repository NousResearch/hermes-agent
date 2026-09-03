"""Shared Discord command projection API."""

from .core import (
    DiscordProjectionError,
    DiscordProjectionMismatch,
    canonicalize_discord_command,
    canonicalize_discord_option,
)
from .model import (
    DiscordCommandProjection,
    DiscordProjectedCommand,
    project_discord_commands,
    verify_discord_projection_readback,
)
from .relay import build_relay_discord_manifest, build_relay_discord_projection

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

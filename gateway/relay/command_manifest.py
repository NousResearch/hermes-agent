"""Relay-lane compatibility export for the shared Discord projection.

The source of truth moved to :mod:`gateway.discord_command_projection` so the
native adapter's sync proof and the relay hello manifest use one canonical
normalization, fingerprint, alias identity, and read-back contract.
"""

from __future__ import annotations

from typing import Any, Dict, List

from gateway.discord_command_projection import build_relay_discord_manifest


def build_relay_command_manifest() -> List[Dict[str, Any]]:
    """Return the relay lane's Discord command manifest."""
    return build_relay_discord_manifest()

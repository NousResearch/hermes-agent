"""Compatibility import for the packaged privileged Discord edge service.

The executable implementation lives in :mod:`gateway.discord_edge_service` so
sealed wheel installs never depend on the source-only ``scripts`` namespace.
"""

from gateway.discord_edge_service import *  # noqa: F403

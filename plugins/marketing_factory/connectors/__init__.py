"""Channel connector registry for Marketing Agent Factory.

`PublisherAgent` calls into here to find a connector for a given channel.

Registration model: connectors register themselves at import time. Only
`dry_run` is bundled by default — every real channel connector lives in its
own file (e.g. `connectors/x.py`) that imports `register` and supplies a
`BaseChannelConnector` implementation. Stub files exist for each channel we
want to support but raise NotImplementedError until a human fills in the API
calls. This keeps the test suite import-safe (no creds required) while making
"plug in X live" a one-file change.

Safety contract:
- If `channel_modes[channel] == "live"` but no real connector is registered,
  the Publisher MUST fall back to dry_run and audit the fallback.
- A "real" connector is any registered connector other than `DryRunConnector`.
"""

from __future__ import annotations

from typing import Dict, Optional, Type

from plugins.marketing_factory.connectors.base import BaseChannelConnector
from plugins.marketing_factory.connectors.dry_run import DryRunConnector

_REGISTRY: Dict[str, BaseChannelConnector] = {}

# Always-on dry-run baseline keyed under the literal channel name "*".
# PublisherAgent uses this when channel_modes says dry_run, when a channel has
# no registered live connector, or when a live publish call raises.
_DRY_RUN: BaseChannelConnector = DryRunConnector()


def register(channel: str, connector: BaseChannelConnector) -> None:
    """Register a real connector for a channel. Last-writer-wins."""
    _REGISTRY[channel] = connector


def get_live_connector(channel: str) -> Optional[BaseChannelConnector]:
    """Return the registered live connector for `channel`, or None if none is wired."""
    connector = _REGISTRY.get(channel)
    if connector is None or isinstance(connector, DryRunConnector):
        return None
    return connector


def get_dry_run_connector() -> BaseChannelConnector:
    return _DRY_RUN


__all__ = [
    "BaseChannelConnector",
    "DryRunConnector",
    "register",
    "get_live_connector",
    "get_dry_run_connector",
]

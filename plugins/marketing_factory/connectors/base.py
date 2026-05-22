"""Base channel connector ABC.

A connector takes a stored draft dict and returns a structured result. The
PublisherAgent records the result via `store.record_publish(...)` — it does
NOT update draft state from inside connectors. This keeps connectors
testable in isolation and prevents a buggy connector from corrupting the
store.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseChannelConnector(ABC):
    """Abstract channel connector.

    Implementations:
      - `mode` is "dry_run" or "live"
      - `publish(draft)` returns a dict with at least:
          { mode: str, would_post: bool, posted: bool, channel: str, payload: dict }
        `posted=True` means the draft was actually published to a real channel.
      - On failure, raise `ConnectorError` (caught by PublisherAgent, which falls
        back to dry-run + audit).
    """

    mode: str = "dry_run"
    channel: str = ""

    @abstractmethod
    def publish(self, draft: Dict[str, Any]) -> Dict[str, Any]:
        ...


class ConnectorError(RuntimeError):
    """Raised when a connector fails. PublisherAgent converts to dry-run fallback."""

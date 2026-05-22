"""DryRunConnector — preserves the pre-phase-4 behavior.

`would_post=True, posted=False` means the system *would* have called the real
channel API with this draft body, but it did not. This connector requires no
credentials and is always available.
"""

from __future__ import annotations

from typing import Any, Dict

from plugins.marketing_factory.connectors.base import BaseChannelConnector


class DryRunConnector(BaseChannelConnector):
    mode = "dry_run"
    channel = "*"

    def publish(self, draft: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "mode": "dry_run",
            "would_post": True,
            "posted": False,
            "channel": draft.get("channel"),
            "body": draft.get("body"),
            "payload": {
                "draft_id": draft.get("id"),
                "channel": draft.get("channel"),
                "body": draft.get("body"),
                "cta": draft.get("cta"),
                "assets": list(draft.get("assets") or []),
            },
        }

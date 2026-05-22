"""X (Twitter) connector — STUB.

To go live:
  1. Get developer creds at https://developer.twitter.com/
  2. Set env vars: X_API_KEY, X_API_SECRET, X_ACCESS_TOKEN, X_ACCESS_TOKEN_SECRET
     (or for OAuth 2.0 user context: X_BEARER_TOKEN + X_REFRESH_TOKEN)
  3. Replace `publish()` body with a POST to https://api.twitter.com/2/tweets
     payload: {"text": draft["body"]} — must be ≤ 280 chars (already enforced
     by ReviewSafetyAgent).
  4. Activate by editing `connectors/__init__.py` to:
       from plugins.marketing_factory.connectors.x_stub import XConnector
       register("x", XConnector())

Until then, channel_modes["x"] == "live" falls back to dry_run (audited).
"""

from __future__ import annotations

from typing import Any, Dict

from plugins.marketing_factory.connectors.base import BaseChannelConnector, ConnectorError


class XConnector(BaseChannelConnector):
    mode = "live"
    channel = "x"

    def publish(self, draft: Dict[str, Any]) -> Dict[str, Any]:
        raise ConnectorError(
            "XConnector is a stub. Fill in publish() with the X /2/tweets call before "
            "registering it in connectors/__init__.py."
        )

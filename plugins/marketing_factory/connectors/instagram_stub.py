"""Instagram connector — STUB.

To go live:
  1. Set up a Facebook/Meta app linked to an Instagram Business or Creator account
     at https://developers.facebook.com/
  2. Set env vars: META_APP_ID, META_APP_SECRET, IG_USER_ID, IG_ACCESS_TOKEN
  3. Instagram posting requires a two-step container flow:
       a. POST /{ig-user-id}/media with image_url + caption → returns container id
       b. POST /{ig-user-id}/media_publish with container id → publishes
     Image asset must be hosted at an accessible HTTPS URL — this means the
     factory needs an asset-hosting step before live publish. Tackle that
     separately; this stub assumes draft["assets"][0] is a URL already.
  4. Activate via connectors/__init__.py:
       from plugins.marketing_factory.connectors.instagram_stub import InstagramConnector
       register("instagram", InstagramConnector())

Until then, channel_modes["instagram"] == "live" falls back to dry_run.
"""

from __future__ import annotations

from typing import Any, Dict

from plugins.marketing_factory.connectors.base import BaseChannelConnector, ConnectorError


class InstagramConnector(BaseChannelConnector):
    mode = "live"
    channel = "instagram"

    def publish(self, draft: Dict[str, Any]) -> Dict[str, Any]:
        raise ConnectorError(
            "InstagramConnector is a stub. Wire the two-step Graph API container "
            "flow + asset hosting before registering."
        )

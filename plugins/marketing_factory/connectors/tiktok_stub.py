"""TikTok connector — STUB.

TikTok is unusual: the public Content Posting API requires uploaded video assets
and is gated on a developer + business application approval. There is currently
no first-class "post a text caption" endpoint — every TikTok post is a video.

To go live:
  1. Approval at https://developers.tiktok.com/ — Content Posting API access.
  2. Env vars: TIKTOK_CLIENT_KEY, TIKTOK_CLIENT_SECRET, TIKTOK_ACCESS_TOKEN
  3. Two-step: POST /v2/post/publish/inbox/video/init → get upload URL, PUT video
     bytes, then POST /v2/post/publish/inbox/video/publish with the caption.
  4. Activate via connectors/__init__.py.

Until then, channel_modes["tiktok"] == "live" falls back to dry_run. (The
TikTok drafts the factory generates today are scripts, not finished videos —
so live publish to TikTok is not realistic without an upstream video pipeline.)
"""

from __future__ import annotations

from typing import Any, Dict

from plugins.marketing_factory.connectors.base import BaseChannelConnector, ConnectorError


class TikTokConnector(BaseChannelConnector):
    mode = "live"
    channel = "tiktok"

    def publish(self, draft: Dict[str, Any]) -> Dict[str, Any]:
        raise ConnectorError(
            "TikTokConnector is a stub. TikTok live posting requires a video asset; "
            "wire only after a video pipeline exists upstream."
        )

"""LinkedIn connector — STUB.

To go live:
  1. Apply for LinkedIn Marketing API access at https://learn.microsoft.com/linkedin/
     (organization page posting requires r_organization_social + w_organization_social
     scopes plus partner approval — non-trivial).
  2. Set env vars: LINKEDIN_ACCESS_TOKEN, LINKEDIN_ORG_URN (e.g. urn:li:organization:12345)
  3. POST https://api.linkedin.com/v2/ugcPosts with shareCommentary text. Personal
     posting uses author=urn:li:person:{id}; organization posts use the org URN.
  4. Activate via connectors/__init__.py.

Until then, channel_modes["linkedin"] == "live" falls back to dry_run.
"""

from __future__ import annotations

from typing import Any, Dict

from plugins.marketing_factory.connectors.base import BaseChannelConnector, ConnectorError


class LinkedInConnector(BaseChannelConnector):
    mode = "live"
    channel = "linkedin"

    def publish(self, draft: Dict[str, Any]) -> Dict[str, Any]:
        raise ConnectorError(
            "LinkedInConnector is a stub. LinkedIn org posting requires partner-tier "
            "API access; wire only after that is approved."
        )

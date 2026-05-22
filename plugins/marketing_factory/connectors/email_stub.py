"""Email connector — STUB.

To go live:
  1. Pick a transactional/broadcast provider (Resend, Postmark, Mailgun, AWS SES).
  2. Env vars (Resend example): RESEND_API_KEY, RESEND_FROM_ADDRESS, RESEND_TO_LIST_ID
  3. Email drafts ALREADY contain a "Subject:" first line per
     pipeline._CHANNEL_GUIDANCE — split on the first newline before sending.
  4. Activate via connectors/__init__.py.

Until then, channel_modes["email"] == "live" falls back to dry_run.
"""

from __future__ import annotations

from typing import Any, Dict

from plugins.marketing_factory.connectors.base import BaseChannelConnector, ConnectorError


class EmailConnector(BaseChannelConnector):
    mode = "live"
    channel = "email"

    def publish(self, draft: Dict[str, Any]) -> Dict[str, Any]:
        raise ConnectorError(
            "EmailConnector is a stub. Wire a provider (Resend/Postmark/SES) and "
            "split the 'Subject:' line off the body before sending."
        )

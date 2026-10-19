"""``/secret [KEY_NAME]``: a one-time HTTPS secret entry link, Telegram only (gateway/secret_link.py)."""

from __future__ import annotations

import asyncio
from typing import Optional

from gateway import secret_link, secret_link_store


class GatewaySecretLinkCommandsMixin:
    async def _handle_secret_command(self, event) -> Optional[str]:
        source = event.source
        if not secret_link.is_telegram(source):
            return secret_link.TELEGRAM_ONLY_MESSAGE
        suggested = event.get_command_args().strip()
        if suggested and secret_link.key_error(suggested, ()):
            return "Invalid key name."
        await secret_link.create_link(self, source, suggested_key=suggested)
        return None  # the link (or the refusal notice) went out with link previews off

    async def _secret_link_cleanup_watcher(self) -> None:
        """Drop used and expired links every interval; shutdown (cancel) clears every record."""
        try:
            while True:
                await asyncio.sleep(secret_link.CLEANUP_INTERVAL_SECONDS)
                secret_link_store.STORE.sweep()
        finally:
            secret_link_store.STORE.clear()

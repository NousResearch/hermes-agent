"""Requester notifications after a chat-initiated gateway restart."""

import json
import logging
from typing import Optional

from gateway.config import Platform
from gateway.run_shutdown import _send_error, _send_failed

logger = logging.getLogger("gateway.run")


class GatewayRestartNotificationsMixin:
    async def _send_restart_notification(self) -> Optional[tuple[str, str, Optional[str]]]:
        """Notify the chat that initiated /restart that the gateway is back."""
        from gateway.delivery import resolve_delivery_transport
        from gateway.run import _hermes_home, _non_conversational_metadata
        notify_path = _hermes_home / ".restart_notify.json"
        if not notify_path.exists():
            return None
        try:
            data = json.loads(notify_path.read_text(encoding="utf-8"))
            platform_str = data.get("platform")
            chat_id = data.get("chat_id")
            thread_id = data.get("thread_id")
            if not platform_str or not chat_id:
                return None
            platform = Platform(platform_str)
            # Relay-aware transport over the REQUESTER'S profile adapter map; ``self.adapters`` is the
            # default profile's, so a secondary's "restarted" notice would leave through the wrong bot.
            transport = resolve_delivery_transport(
                platform, self.config, self._adapters_for_profile(self._marker_profile(data)))
            if transport is None:
                logger.debug("Restart notification skipped: no live transport for %s", platform_str)
                return None
            platform_cfg = self.config.platforms.get(platform)
            if platform_cfg is not None and not platform_cfg.gateway_restart_notification:
                logger.info(
                    "Restart notification suppressed: %s has gateway_restart_notification=false", platform_str
                )
                return None
            metadata = self._pending_marker_metadata(platform, chat_id, data, transport.adapter)
            if data.get("delivered_via_upstream_relay") is True:
                metadata = dict(metadata or {})
                for field in ("user_id", "scope_id"):
                    if data.get(field):
                        metadata[field] = str(data[field])
            result = await transport.send(
                platform, str(chat_id), "♻ Gateway restarted successfully. Your session continues.",
                metadata=_non_conversational_metadata(metadata, platform=platform),
            )
            # adapter.send() catches provider errors (e.g. "Chat not found") and returns
            # SendResult(success=False) rather than raising, so inspect the result before claiming success.
            if _send_failed(result):
                logger.warning(
                    "Restart notification to %s:%s was not delivered: %s", platform_str, chat_id, _send_error(result),
                )
                return None
            logger.info("Sent restart notification to %s:%s", platform_str, chat_id)
            return str(platform_str), str(chat_id), str(thread_id) if thread_id else None
        except Exception as e:
            logger.warning("Restart notification failed: %s", e)
            return None
        finally:
            notify_path.unlink(missing_ok=True)


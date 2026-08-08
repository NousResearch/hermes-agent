"""Thread-metadata helpers for ``GatewayRunner``.

Extracted from ``gateway/run.py`` (god-file decomposition campaign, Wave 1
mixin lifts). This mixin holds the thread-metadata cluster: building the
platform-specific metadata dicts that thread-aware replies need
(``_thread_metadata_for_source`` / ``_thread_metadata_for_target``) and the
``_reply_anchor_for_event`` passthrough.

Behavior-neutral: every method is lifted verbatim from ``GatewayRunner``.
``self.*`` calls resolve unchanged via the MRO (``_is_telegram_dm_topic_target``
stays on ``GatewayRunner`` — covered by the open threads-mixin extraction).
The module-level ``logger`` is ``logging.getLogger("gateway.run")`` so log
records keep the exact name (``"gateway.run"``), matching the sibling mixins'
convention.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, _reply_anchor_for_event

logger = logging.getLogger("gateway.run")


class GatewayThreadMetadataMixin:
    def _thread_metadata_for_source(
        self,
        source,
        reply_to_message_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Build the metadata dict platforms need for thread-aware replies."""
        metadata = self._thread_metadata_for_target(
            getattr(source, "platform", None),
            getattr(source, "chat_id", None),
            getattr(source, "thread_id", None),
            chat_type=getattr(source, "chat_type", None),
            reply_to_message_id=reply_to_message_id or getattr(source, "message_id", None),
        )
        if getattr(source, "platform", None) == Platform.SLACK:
            team_id = getattr(source, "scope_id", None)
            if team_id:
                metadata = dict(metadata or {})
                metadata["slack_team_id"] = str(team_id)
        return metadata

    def _thread_metadata_for_target(
        self,
        platform: Optional[Platform],
        chat_id: Optional[str],
        thread_id: Optional[str],
        *,
        chat_type: Optional[str] = None,
        reply_to_message_id: Optional[str] = None,
        adapter: Optional[Any] = None,
    ) -> Optional[Dict[str, Any]]:
        """Build thread metadata for synthetic sends that only have routing state."""
        if thread_id is None:
            return None
        metadata: Dict[str, Any] = {"thread_id": thread_id}
        if self._is_telegram_dm_topic_target(
            platform,
            chat_id,
            thread_id,
            chat_type=chat_type,
            adapter=adapter,
        ):
            metadata["telegram_dm_topic_reply_fallback"] = True
            # Telegram DM topic lanes need direct_messages_topic_id in metadata
            # so synthetic/queued messages (goal continuations, status notices)
            # route to the correct topic even when reply anchor is unavailable.
            tid = str(thread_id)
            if tid and tid not in {"", "1"}:
                metadata["direct_messages_topic_id"] = tid
            if reply_to_message_id is not None:
                metadata["telegram_reply_to_message_id"] = str(reply_to_message_id)
        if platform == Platform.SLACK and reply_to_message_id is not None:
            # Slack's reply_in_thread=false path uses message_id to distinguish
            # real existing threads from synthetic top-level session keys.
            metadata["message_id"] = str(reply_to_message_id)
        return metadata

    @staticmethod
    def _reply_anchor_for_event(event: MessageEvent) -> Optional[str]:
        """Return the platform-specific reply anchor for GatewayRunner sends."""
        return _reply_anchor_for_event(event)

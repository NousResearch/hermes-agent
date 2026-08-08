"""SessionSource-building helper for gateway platform adapters.

Extracted verbatim from ``gateway/platforms/base.py`` (class
``BasePlatformAdapter``) — godfile decomposition shard s5.  Mixed into
``BasePlatformAdapter``; instance state (``self.platform``) resolves through
the adapter via MRO.
"""

from __future__ import annotations

import logging
import weakref
from typing import Optional

from gateway.session import SessionSource

logger = logging.getLogger(__name__)


class SourceBuilderMixin:
    def build_source(
        self,
        chat_id: str,
        chat_name: Optional[str] = None,
        chat_type: str = "dm",
        user_id: Optional[str] = None,
        user_name: Optional[str] = None,
        thread_id: Optional[str] = None,
        chat_topic: Optional[str] = None,
        user_id_alt: Optional[str] = None,
        chat_id_alt: Optional[str] = None,
        is_bot: bool = False,
        scope_id: Optional[str] = None,
        guild_id: Optional[str] = None,
        parent_chat_id: Optional[str] = None,
        message_id: Optional[str] = None,
        role_authorized: bool = False,
        auto_thread_created: bool = False,
        auto_thread_initial_name: Optional[str] = None,
    ) -> SessionSource:
        """Helper to build a SessionSource for this platform.

        When ``gateway.profile_routes`` is configured, the routing engine
        resolves the matching profile from guild/chat/thread and stamps it on
        ``source.profile``. Downstream code (``_resolve_profile_home_for_source``
        in run.py) reads that field to enter ``_profile_runtime_scope`` for
        per-profile HERMES_HOME isolation.
        """
        # Normalize empty topic to None
        if chat_topic is not None and not chat_topic.strip():
            chat_topic = None

        # Resolve profile from configured routes (None when no match / no routes)
        profile = None
        runner = getattr(self, "gateway_runner", None)
        if runner is not None:
            try:
                profile = runner._profile_name_for_source(
                    SessionSource(
                        platform=self.platform,
                        chat_id=str(chat_id),
                        chat_name=chat_name,
                        chat_type=chat_type,
                        user_id=str(user_id) if user_id else None,
                        user_name=user_name,
                        thread_id=str(thread_id) if thread_id else None,
                        chat_topic=chat_topic.strip() if chat_topic else None,
                        user_id_alt=user_id_alt,
                        chat_id_alt=chat_id_alt,
                        is_bot=is_bot,
                        scope_id=str(scope_id) if scope_id else None,
                        guild_id=str(guild_id) if guild_id else None,
                        parent_chat_id=str(parent_chat_id) if parent_chat_id else None,
                        message_id=str(message_id) if message_id else None,
                    )
                )
            except Exception:
                logger.warning(
                    "Profile resolution failed for %s/%s, defaulting to active profile",
                    self.platform, chat_id, exc_info=True,
                )

        source = SessionSource(
            platform=self.platform,
            chat_id=str(chat_id),
            chat_name=chat_name,
            chat_type=chat_type,
            user_id=str(user_id) if user_id else None,
            user_name=user_name,
            thread_id=str(thread_id) if thread_id else None,
            chat_topic=chat_topic.strip() if chat_topic else None,
            user_id_alt=user_id_alt,
            chat_id_alt=chat_id_alt,
            is_bot=is_bot,
            scope_id=str(scope_id) if scope_id else None,
            guild_id=str(guild_id) if guild_id else None,
            parent_chat_id=str(parent_chat_id) if parent_chat_id else None,
            message_id=str(message_id) if message_id else None,
            profile=profile,
            role_authorized=role_authorized,
            auto_thread_created=auto_thread_created,
            auto_thread_initial_name=auto_thread_initial_name,
        )
        # In-process transport provenance is deliberately not serialized by
        # SessionSource.to_dict(). The live receiving adapter is authoritative
        # for this turn even when profile_routes selects a different runtime.
        source._transport_adapter_ref = weakref.ref(self)
        return source

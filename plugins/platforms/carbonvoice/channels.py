"""In-memory cache for Carbon Voice channel metadata.

One ``GET /channel/{id}`` per channel populates two things, both keyed by
``channel_id`` and cached for the process lifetime:

  - **chat_type** ("dm" | "group") — channel kind almost never changes
    after creation (a DM stays a DM forever).
  - **roster** (``{user_guid → display name}``) — derived from the
    channel's ``json_collaborators``. This is the canonical way to
    resolve participant names: the standalone ``GET /v3/users/{id}``
    endpoint is dead (404), and the collaborator list rides on the same
    payload we already fetch for chat-type, so names cost zero extra
    calls.

The first message in a new channel pays one API call; within the TTL
window every message after is free for both axes. A failed *initial*
lookup caches ``"dm"`` + an empty roster so the adapter degrades
gracefully (keeps responding, falls back to the raw guid for names)
rather than re-hitting the API per message.

TTL: the payload is refreshed after ``ttl_s`` (default 30 min) so a
participant who joins mid-conversation shows up in the roster without a
gateway restart. ``chat_type`` is immutable so re-fetching it is wasted
but harmless — keeping one cache policy is simpler than two. A failed
*refresh* keeps the prior good values (we don't blow a known roster away
with an empty one on a transient hiccup).
"""

from __future__ import annotations

import logging
import time
from typing import Dict, Optional

from .api import CarbonVoiceAPI
from .parse import chat_type_from_channel, extract_roster

logger = logging.getLogger(__name__)

# 30 min — long enough that a busy channel stays cache-warm, short enough
# that a new joiner is picked up within a reasonable window. Mirrors the
# thread-context TTL in conversations.py.
DEFAULT_CHANNEL_TTL_S = 1800


class ChannelCache:
    def __init__(
        self, api: CarbonVoiceAPI, *, ttl_s: int = DEFAULT_CHANNEL_TTL_S
    ):
        self._api = api
        self._type_cache: Dict[str, str] = {}
        self._roster_cache: Dict[str, Dict[str, str]] = {}
        self._loaded_at: Dict[str, float] = {}
        self._ttl_s = ttl_s

    async def _ensure_loaded(self, channel_id: str) -> None:
        """Fetch the channel and populate both caches, honoring the TTL.

        Returns early when a cached entry is still within ``ttl_s``. On a
        refresh that fails, the prior cached values are kept (and the
        timestamp bumped so we don't hammer the API on repeated failures).
        """
        now = time.monotonic()
        loaded = self._loaded_at.get(channel_id)
        if loaded is not None and (now - loaded) <= self._ttl_s:
            return
        try:
            data = await self._api.get_channel(channel_id)
        except Exception as exc:
            logger.debug(
                "carbonvoice: get_channel(%s) failed: %s", channel_id, exc
            )
            data = None
        if data is None and channel_id in self._type_cache:
            # Refresh failed but we have prior good values — keep them.
            self._loaded_at[channel_id] = now
            return
        self._type_cache[channel_id] = chat_type_from_channel(data)
        self._roster_cache[channel_id] = extract_roster(data)
        self._loaded_at[channel_id] = now

    async def resolve_chat_type(self, channel_id: str) -> str:
        """Return ``"dm"`` or ``"group"`` for *channel_id*.

        Defaults to ``"dm"`` on any lookup failure so the agent keeps
        responding (previous behavior) rather than going silent because of
        a transient channel-API hiccup.
        """
        if not channel_id:
            return "dm"
        await self._ensure_loaded(channel_id)
        return self._type_cache.get(channel_id, "dm")

    async def get_roster(self, channel_id: str) -> Dict[str, str]:
        """Return ``{user_guid → display name}`` for *channel_id*.

        Empty dict on lookup failure. Shares the cached channel payload
        with :meth:`resolve_chat_type`, so calling both for one message is
        a single HTTP call.
        """
        if not channel_id:
            return {}
        await self._ensure_loaded(channel_id)
        return self._roster_cache.get(channel_id, {})

    async def resolve_name(
        self, channel_id: str, user_guid: str
    ) -> Optional[str]:
        """Display name for *user_guid* in *channel_id*, or ``None``.

        ``None`` means "not in this channel's collaborator list" — callers
        fall back to the raw guid.
        """
        if not channel_id or not user_guid:
            return None
        roster = await self.get_roster(channel_id)
        return roster.get(user_guid)

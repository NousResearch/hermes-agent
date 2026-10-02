"""Workspace-scoped Slack direct-message target resolution."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from plugins.platforms.slack.adapter import SlackAdapter

logger = logging.getLogger("plugins.platforms.slack.adapter")


class SlackOutboundTargetsMixin:
    async def _dm_target(self: SlackAdapter, chat_id: str, metadata: Optional[dict[str, Any]]) -> str:
        """``_ensure_dm_conversation`` scoped by outbound ``metadata``."""
        return await self._ensure_dm_conversation(chat_id, team_id=self._metadata_team_id(metadata))


    async def _ensure_dm_conversation(self: SlackAdapter, chat_id: str, team_id: Optional[str] = None) -> str:
        """Resolve a bare user ID (U/W...) to a DM conversation ID via ``conversations.open``
        (``chat.postMessage``/``files_upload_v2`` reject user IDs); cached per (team, user). Returns
        ``chat_id`` unchanged when not applicable or on failure (downstream surfaces the error).

        Resolution goes through the workspace-scoped client so multi-workspace installs open the DM with the
        right bot token, and results are cached per (team, user) so repeated sends don't re-open. See
        #17261, #19236.
        """
        cid = str(chat_id or "")
        if not cid or cid[0] not in ("U", "W"):
            return chat_id
        cache_key = f"{team_id or ''}:{cid}"
        cached = self._dm_conversation_cache.get(cache_key)
        if cached:
            return cached
        try:
            response = await self._get_client(cid, team_id=team_id).conversations_open(users=cid)
            dm_id = ((response or {}).get("channel") or {}).get("id")
            if dm_id:
                self._dm_conversation_cache[cache_key] = dm_id
                self._trim_oldest_dict_entries(
                    self._dm_conversation_cache, self._DM_CONVERSATION_CACHE_MAX)
                if team_id:
                    self._remember_channel_team(dm_id, team_id)
                return dm_id
        except Exception as e:
            logger.warning(
                "[Slack] conversations.open failed for user target %s: %s "
                "(check the bot's im:write scope)", cid, e)
        return chat_id

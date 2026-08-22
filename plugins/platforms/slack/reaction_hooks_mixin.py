"""Reaction-hook methods for the Slack adapter.

Extracted from ``plugins/platforms/slack/adapter.py`` as part of the god-file
decomposition campaign (epic #78647, target #78638, region R3 slice R3-S1).

This mixin holds the reaction-primitive cluster: adding/removing emoji
reactions, the config/env gate that enables them, and the
``on_processing_start`` / ``on_processing_complete`` base-hook overrides that
paint the in-progress "eyes" reaction and swap it for a final success/failure
reaction.

Behavior-neutral: every method is lifted verbatim from ``SlackAdapter``.
``self.*`` calls resolve unchanged via the MRO (``_get_client``,
``_workspace_message_marker``, ``_reacting_message_ids``, ``_app`` stay in the
adapter). Neutral dependencies import at module top: stdlib only plus
``gateway.platforms.base``, so this module never imports the adapter and there
is no import cycle.
"""

from __future__ import annotations

import logging
import os

from gateway.platforms.base import MessageEvent, ProcessingOutcome

logger = logging.getLogger(__name__)


class SlackReactionHooksMixin:
    # ----- Reactions -----
    async def _add_reaction(
        self, channel: str, timestamp: str, emoji: str, team_id: str = ""
    ) -> bool:
        """Add an emoji reaction to a message. Returns True on success."""
        if not self._app:
            return False
        try:
            await self._get_client(channel, team_id=team_id or None).reactions_add(
                channel=channel, timestamp=timestamp, name=emoji
            )
            return True
        except Exception as e:
            # Don't log as error — may fail if already reacted or missing scope
            logger.debug("[Slack] reactions.add failed (%s): %s", emoji, e)
            return False

    async def _remove_reaction(
        self, channel: str, timestamp: str, emoji: str, team_id: str = ""
    ) -> bool:
        """Remove an emoji reaction from a message. Returns True on success."""
        if not self._app:
            return False
        try:
            await self._get_client(channel, team_id=team_id or None).reactions_remove(
                channel=channel, timestamp=timestamp, name=emoji
            )
            return True
        except Exception as e:
            logger.debug("[Slack] reactions.remove failed (%s): %s", emoji, e)
            return False

    def _reactions_enabled(self) -> bool:
        """Check if message reactions are enabled via config/env."""
        return os.getenv("SLACK_REACTIONS", "true").lower() not in {"false", "0", "no"}

    async def on_processing_start(self, event: MessageEvent) -> None:
        """Add an in-progress reaction when message processing begins."""
        if not self._reactions_enabled():
            return
        ts = getattr(event, "message_id", None)
        team_id = str(getattr(event.source, "scope_id", "") or "")
        marker = self._workspace_message_marker(team_id, ts) if ts else None
        if not ts or marker not in self._reacting_message_ids:
            return
        channel_id = getattr(event.source, "chat_id", None)
        if channel_id:
            await self._add_reaction(channel_id, ts, "eyes", team_id)

    async def on_processing_complete(
        self, event: MessageEvent, outcome: ProcessingOutcome
    ) -> None:
        """Swap the in-progress reaction for a final success/failure reaction."""
        if not self._reactions_enabled():
            return
        ts = getattr(event, "message_id", None)
        team_id = str(getattr(event.source, "scope_id", "") or "")
        marker = self._workspace_message_marker(team_id, ts) if ts else None
        if not ts or marker not in self._reacting_message_ids:
            return
        self._reacting_message_ids.discard(marker)
        channel_id = getattr(event.source, "chat_id", None)
        if not channel_id:
            return
        await self._remove_reaction(channel_id, ts, "eyes", team_id)
        if outcome == ProcessingOutcome.SUCCESS:
            await self._add_reaction(channel_id, ts, "white_check_mark", team_id)
        elif outcome == ProcessingOutcome.FAILURE:
            await self._add_reaction(channel_id, ts, "x", team_id)

"""Visual ack via Carbon Voice reactions.

The agent reacts to every accepted inbound message immediately, so the
user sees feedback in <100 ms even when the agent itself takes 10 s to
think. Reaction id defaults to the literal ``"acknowledged"`` (a CV
built-in); override via ``CARBONVOICE_REACTION_ID``.

On startup the service logs the workspace's available reactions so the
operator can pick one and pin it via env. Failures are non-fatal — if the
workspace has no reactions or the POST 4xxs, we log and continue.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from .api import CarbonVoiceAPI
from .constants import DEFAULT_PENDING_REACTION_ID, DEFAULT_REACTION_ID

logger = logging.getLogger(__name__)


class ReactionService:
    def __init__(
        self,
        api: CarbonVoiceAPI,
        reaction_id: Optional[str] = None,
        enabled: bool = True,
        pending_reaction_id: Optional[str] = None,
    ):
        self._api = api
        self._reaction_id = reaction_id or DEFAULT_REACTION_ID
        # Reaction used to silently acknowledge an *unauthorized* sender's
        # first message ("we saw you, you're pending approval") instead of
        # spamming the channel with a text reply. Defaults to the CV
        # built-in "confused" (⁉️).
        self._pending_reaction_id = (
            pending_reaction_id or DEFAULT_PENDING_REACTION_ID
        )
        self._enabled = enabled

    @property
    def enabled(self) -> bool:
        return self._enabled and bool(self._reaction_id)

    @property
    def reaction_id(self) -> str:
        """The reaction id used for the ack (also the server-side
        processed-marker the adapter checks for inbound dedup)."""
        return self._reaction_id

    @property
    def pending_reaction_id(self) -> str:
        """Reaction used to flag an unauthorized sender's first message."""
        return self._pending_reaction_id

    async def discover(self) -> None:
        """List the workspace's available reactions to the log.

        Informational only — we don't change ``self._reaction_id`` based on
        the result. The operator sees the list and can pin an id via the
        ``CARBONVOICE_REACTION_ID`` env var on next startup.
        """
        if not self._enabled:
            return
        try:
            reactions = await self._api.fetch_reactions()
        except Exception as exc:
            logger.warning("carbonvoice: GET /reactions failed: %s", exc)
            return
        if not reactions:
            logger.info("carbonvoice: no reactions available — ack disabled")
            self._enabled = False
            return
        logger.info("carbonvoice: %d reactions available:", len(reactions))
        for r in reactions:
            logger.info(
                "  id=%s  name=%r  code=%r",
                r.get("id"), r.get("name"), r.get("code"),
            )
        logger.info("carbonvoice: using reaction id=%s", self._reaction_id)

    def ack(self, message_id: str) -> None:
        """Fire-and-forget visual ack. Errors logged at debug, never raised."""
        if not self.enabled or not message_id:
            return
        asyncio.create_task(self._react(self._reaction_id, message_id))

    async def ack_sync(self, message_id: str) -> bool:
        """Blocking ack: await the reaction POST so the server-side dedup
        marker is GUARANTEED present before the caller proceeds.

        Used for owner allow-list commands, whose reply bumps ``updated_at``
        and re-fires the poll: without a durable ack already on the server,
        the re-fetched command re-runs and re-replies (the 298×-spam bug).
        Returns True on success. Never raises.
        """
        if not self.enabled or not message_id:
            return False
        try:
            await self._api.react(self._reaction_id, message_id)
            return True
        except Exception as exc:
            logger.debug(
                "carbonvoice: ack_sync(%s, %s) failed: %s",
                self._reaction_id, message_id, exc,
            )
            return False

    def pending(self, message_id: str) -> None:
        """Fire-and-forget "pending approval" reaction on an unauthorized
        sender's message (⁉️). Same fire-and-forget contract as :meth:`ack`."""
        if not self.enabled or not message_id:
            return
        asyncio.create_task(self._react(self._pending_reaction_id, message_id))

    async def _react(self, reaction_id: str, message_id: str) -> None:
        try:
            await self._api.react(reaction_id, message_id)
        except Exception as exc:
            logger.debug(
                "carbonvoice: react(%s, %s) failed: %s",
                reaction_id, message_id, exc,
            )

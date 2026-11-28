"""Gateway /refresh command: queue a prompt-refresh request so continuing sessions adopt
updated SOUL.md, skills and context files on their next turn.

The request is one state_meta epoch row in the serving profile's store — the same call the
CLI verb ``hermes sessions refresh-prompts`` makes — and the per-turn probe in
``agent/prompt_refresh.py`` performs the actual rebuild before the next model call. This
command adds a gateway-native trigger surface; it does not change the request semantics.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from agent.i18n import t
from gateway.platforms.event import MessageEvent

# Log-record parity with gateway/run.py and the sibling command modules.
logger = logging.getLogger("gateway.run")


class GatewayRefreshCommandsMixin:
    """Slash-command handler for prompt-refresh requests.

    Scope semantics: bare ``/refresh`` targets the session the command was typed in;
    ``/refresh <prefix>`` targets one named session (unique id or prefix); ``/refresh all``
    targets every session in the serving profile with a stored prompt and asks through the
    destructive-slash confirmation (one prompt-cache break per session is a real cost).

    History, session ids and the session's tools pin survive; ``on_session_start`` hooks do
    not fire (same contract as the CLI verb).

    Composed onto ``GatewayRunner`` through ``GatewaySlashCommandsMixin``; the runner
    surfaces used here are declared for static analysis below.
    """

    # Shared GatewayRunner surfaces (composed at runtime; typed ONLY for static analysis).
    async_session_store: Any
    _open_session_db_for_active_scope: Any
    _maybe_confirm_destructive_slash: Any

    async def _refresh_reply(self, db: Any, session_id: str | None, source_scope: str) -> str:
        """Record the request and build the reply shared by the targeted scopes."""
        count = 1 if ((await db.get_session(session_id)) or {}).get("system_prompt") else 0
        epoch = await db.request_prompt_refresh(session_id)
        logger.info("Slash /refresh queued epoch %s for scope=%s", epoch, source_scope)
        return "\n".join([
            t("gateway.refresh.queued", scope=source_scope, count=count),
            t("gateway.refresh.how"),
        ])

    async def _handle_refresh_command(self, event: MessageEvent) -> str:
        raw = (event.get_command_args() or "").strip()
        db = await asyncio.to_thread(self._open_session_db_for_active_scope)
        lowered = raw.lower()

        if lowered in ("all", "--all"):
            count_all = await db.count_sessions_with_stored_prompt()

            async def _execute() -> str:
                epoch = await db.request_prompt_refresh(None)
                logger.info("Slash /refresh all queued epoch %s (profile-wide)", epoch)
                return "\n".join([
                    t("gateway.refresh.queued", scope=t("gateway.refresh.scope_profile"), count=count_all),
                    t("gateway.refresh.how"),
                ])

            return await self._maybe_confirm_destructive_slash(
                event=event, command="refresh", title="/refresh all",
                detail=t("gateway.refresh.all_detail", count=count_all),
                execute=_execute,
            )

        if raw:
            session_id = await db.resolve_session_id(raw)
            if not session_id:
                return t("gateway.refresh.no_match", arg=raw)
            # A compressed conversation continues on its tip; a request keyed on the ended
            # root never fires (same redirect the CLI verb does).
            session_id = await db.get_compression_tip(session_id) or session_id
            return await self._refresh_reply(db, session_id, session_id)

        session_entry = await self.async_session_store.get_or_create_session(event.source)
        session_id = session_entry.session_id
        return await self._refresh_reply(db, session_id, session_id)

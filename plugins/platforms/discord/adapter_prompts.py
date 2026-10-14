"""Discord interactive-prompt lifetime: view timeouts, and pending clarify prompts across reconnects.

A Discord WebSocket drop makes ``connect()`` build a brand-new ``commands.Bot``. Views are
dispatched by the client that sent them, so buttons sent before the reconnect are orphaned: every
click fails with "This interaction failed" while the clarify entry keeps waiting until
``clarify_timeout``. ``send_clarify`` records button prompts here; ``on_ready`` re-posts the ones
still pending, with the same ``clarify_id``, so the new buttons resolve the same waiting entry.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

logger = logging.getLogger("plugins.platforms.discord.adapter")

# Default timeout for Discord button views when ``approvals.discord_prompt_timeout`` is unset;
# Discord interaction tokens expire at ~15 minutes, so 900s is the practical ceiling.
_DISCORD_PROMPT_TIMEOUT_DEFAULT = 300
_DISCORD_PROMPT_TIMEOUT_MIN = 30
_DISCORD_PROMPT_TIMEOUT_MAX = 900


def _read_discord_prompt_timeout() -> int:
    """Timeout (seconds) for Discord button views from ``approvals.discord_prompt_timeout``
    (default 300), clamped to [MIN, MAX] so a typo can't make prompts vanish or outlive tokens."""
    raw: Any = None
    try:
        from hermes_cli.config import read_raw_config
        cfg = read_raw_config() or {}
        approvals_cfg = cfg.get("approvals", {}) or {}
        raw = approvals_cfg.get("discord_prompt_timeout")
    except Exception:
        return _DISCORD_PROMPT_TIMEOUT_DEFAULT
    if raw is None or raw == "":
        return _DISCORD_PROMPT_TIMEOUT_DEFAULT
    try:
        seconds = int(raw)
    except (TypeError, ValueError):
        return _DISCORD_PROMPT_TIMEOUT_DEFAULT
    if seconds < _DISCORD_PROMPT_TIMEOUT_MIN:
        return _DISCORD_PROMPT_TIMEOUT_MIN
    if seconds > _DISCORD_PROMPT_TIMEOUT_MAX:
        return _DISCORD_PROMPT_TIMEOUT_MAX
    return seconds


class DiscordPromptLifecycleMixin:
    """Keeps pending clarify button prompts clickable across a Discord gateway reconnect."""

    _live_clarify_prompts: Optional[dict[str, dict[str, Any]]] = None
    _clarify_repost_task: Optional[asyncio.Task] = None

    def _remember_clarify_prompt(self, result, chat_id: str, question: str, choices: Optional[list],
                                 clarify_id: str, session_key: str, metadata: Optional[dict]) -> None:
        """Record a sent button prompt, tagged with the client that owns its view."""
        if not (result.success and choices):
            return  # text-only clarifies are answered through the gateway text intercept
        from plugins.platforms.discord.adapter import _prompt_target_id

        if self._live_clarify_prompts is None:
            self._live_clarify_prompts = {}
        self._live_clarify_prompts[clarify_id] = {
            "client_id": id(self._client), "chat_id": chat_id, "question": question,
            "choices": choices, "session_key": session_key, "metadata": metadata,
            "message_id": result.message_id, "channel_id": _prompt_target_id(chat_id, metadata),
        }

    def _schedule_clarify_repost(self) -> None:
        """From ``on_ready``: re-post orphaned prompts once the new client is up."""
        if self._live_clarify_prompts:
            self._clarify_repost_task = asyncio.create_task(self._repost_live_clarify_prompts())

    async def _repost_live_clarify_prompts(self) -> None:
        """Re-send button prompts whose clarify is still pending and whose client was replaced.

        The old message loses its buttons so nothing dead stays clickable. Prompts already
        answered, expired or cancelled are dropped; an ``on_ready`` re-fire on the SAME client
        (its view still listens) leaves the prompt alone.
        """
        from tools.clarify_gateway import _entries as clarify_entries

        live = self._live_clarify_prompts or {}
        for clarify_id, rec in list(live.items()):
            if rec["client_id"] == id(self._client):
                continue
            live.pop(clarify_id, None)
            entry = clarify_entries.get(clarify_id)
            if entry is None or entry.event.is_set():
                continue
            try:
                channel = await self._resolve_channel(rec["channel_id"])
                await channel.get_partial_message(int(rec["message_id"])).edit(view=None)
            except Exception:  # health: allow BLE001 -- best-effort cosmetic strip; the repost below is what matters
                logger.debug("[%s] could not strip stale clarify buttons for %s", self.name, clarify_id, exc_info=True)
            result = await self.send_clarify(
                rec["chat_id"], rec["question"], rec["choices"], clarify_id, rec["session_key"],
                metadata=rec["metadata"],
            )
            logger.info("[%s] re-posted pending clarify %s after reconnect (ok=%s)",
                        self.name, clarify_id, result.success)

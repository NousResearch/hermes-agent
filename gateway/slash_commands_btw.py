"""``/btw`` on the messaging gateway: a side question answered from a snapshot of the stored
transcript (``agent/side_question.py``), never touching the live session."""

from __future__ import annotations

import asyncio
import logging

from agent.i18n import t
from gateway.platforms.event import MessageEvent
from gateway.session_transcript import TranscriptReadError
from gateway.slash_commands_status import history_unreadable

logger = logging.getLogger("gateway.run")


def _preview(text: str, limit: int = 60) -> str:
    return text[:limit] + ("..." if len(text) > limit else "")


class GatewayBtwCommandsMixin:
    async def _handle_btw_command(self, event: MessageEvent) -> str:
        """Handle /btw <question> — one-shot auxiliary LLM call on a transcript snapshot; live history
        is never touched (alternation + prompt cache intact, current turn keeps running). Unlike /bg,
        which spawns a fresh contextless session."""
        question = event.get_command_args().strip()
        if not question:
            return t("gateway.btw.usage")
        source = event.source
        session_entry = await self.async_session_store.get_or_create_session(source)
        try:
            history = await self.async_session_store.load_transcript(session_entry.session_id)
        except TranscriptReadError:
            return history_unreadable()
        if not history:
            return t("gateway.btw.no_history")
        try:
            model, rt = self._resolve_session_agent_runtime(source=source)
        except Exception:
            model, rt = None, {}
        if not rt.get("api_key"):
            return t("gateway.btw.no_provider")
        main_runtime = {
            "model": model,
            **{k: rt.get(k) for k in ("provider", "base_url", "api_key", "api_mode")},
            "session_id": session_entry.session_id,
        }
        history_snapshot = list(history)
        # Prefer the cache-parity fork when a live cached AIAgent exists: it replays the snapshot
        # against the warm provider prefix cache, giving FULL context at cache-read prices. With no
        # cached agent the cache is cold anyway — answer_side_question's digest fallback handles it.
        try:
            parent_agent = self._cached_agent_for(self._session_key_for_source(source))
        except Exception:
            parent_agent = None
        _thread_metadata = self._reply_metadata(event)
        adapter = self._delivery_adapter_for(source)
        preview = _preview(question)

        async def _run_side_question() -> None:
            from agent.side_question import answer_side_question
            try:
                answer = await asyncio.to_thread(
                    answer_side_question, question, history_snapshot,
                    parent_agent=parent_agent, main_runtime=main_runtime)
                reply = t("gateway.btw.answer", preview=preview, answer=answer or "")
            except Exception as e:
                logger.warning("/btw side question failed: %s", e)
                reply = t("gateway.btw.failed", preview=preview, error=str(e))
            if adapter is not None:
                await adapter.send(source.chat_id, reply, metadata=_thread_metadata)

        self._track_background_task(_run_side_question())
        return t("gateway.btw.started", preview=preview)

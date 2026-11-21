"""``/btw`` on the messaging gateway.

``/btw <question>`` answers from a snapshot of the stored transcript (``agent/side_question.py``),
never touching the live session. ``/btw --explore <task>`` forks the chat's live agent into a
tool-using side thread (``agent/side_thread.py``) whose report is sent to the chat when it ends;
``--threads`` shows each thread's latest step while it runs. ``--promote`` pushes a report onto the
adapter FIFO as the chat's next turn (the ``/queue`` path), so it waits behind a running turn and
the transcript only grows by real turns.
"""

from __future__ import annotations

import asyncio
import logging

from agent.i18n import t
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session_transcript import TranscriptReadError
from gateway.slash_commands_status import history_unreadable

logger = logging.getLogger("gateway.run")


def _preview(text: str, limit: int = 60) -> str:
    return text[:limit] + ("..." if len(text) > limit else "")


class GatewayBtwCommandsMixin:
    async def _handle_btw_command(self, event: MessageEvent) -> str:
        """Handle /btw — a side question, or (with flags) a side exploration thread."""
        from agent.side_thread import parse_btw_args, run_thread_command
        request = parse_btw_args(event.get_command_args())
        if not request.text and request.action in {"ask", "explore"}:
            return f"{t('gateway.btw.usage')}\n\n{t('side_thread.usage')}"
        if request.action == "ask":
            return await self._btw_ask(event, request.text)
        session_entry = await self.async_session_store.get_or_create_session(event.source)
        if request.action == "explore":
            return await self._btw_explore(event, request.text, session_entry.session_id)
        result = run_thread_command(session_entry.session_id, request)
        if result.prompt:
            self._enqueue_btw_turn(event, result.prompt)
        return result.text

    def _enqueue_btw_turn(self, event: MessageEvent, prompt: str) -> None:
        """A promoted report as the chat's next turn: the adapter FIFO, or for a local
        ``slash.exec`` the client's own submission (``gateway/session_command_sink.py``)."""
        from gateway.session_command_sink import active_sink
        if (sink := active_sink()) is not None:
            sink.prompt, sink.display = prompt, "/btw --promote"
            return
        adapter, quick_key = self._adapter_and_key_for(event)
        if adapter is not None and quick_key:
            self._enqueue_fifo(quick_key, MessageEvent(
                text=prompt, message_type=MessageType.TEXT, source=event.source, inherit_channel_pin=True), adapter)

    async def _btw_explore(self, event: MessageEvent, task: str, session_id: str) -> str:
        """/btw --explore: fork the chat's live agent; the report is sent to the chat at the end."""
        from agent.side_thread import (
            SideThreadUnavailable, explore_refusal, format_report, register_thread, run_side_exploration,
        )
        source = event.source
        parent_agent = self._btw_live_agent(source)
        if (refusal := explore_refusal(parent_agent)) is not None:
            return refusal.user_message()
        try:
            thread = register_thread(session_id, task)
        except SideThreadUnavailable as exc:
            return exc.user_message()
        # The live agent's own conversation: the bytes its warm prefix cache was built from.
        snapshot = list(getattr(parent_agent, "_session_messages", None) or [])
        metadata, adapter, preview = self._reply_metadata(event), self._delivery_adapter_for(source), _preview(task)

        async def _run() -> None:
            try:
                await self._run_in_executor_with_context(run_side_exploration, thread, parent_agent, snapshot)
                reply = t("side_thread.report", thread_id=thread.thread_id, preview=preview,
                          report=format_report(thread))
            except Exception as exc:  # the thread's own failure, reported to the chat that started it
                logger.warning("/btw --explore %s failed", thread.thread_id, exc_info=True)
                reply = t("side_thread.failed", thread_id=thread.thread_id, preview=preview, error=str(exc))
            if adapter is not None:
                await adapter.send(source.chat_id, reply, metadata=metadata)

        self._track_background_task(_run())
        return t("side_thread.started", thread_id=thread.thread_id, preview=preview)

    def _btw_live_agent(self, source):
        """The chat's live cached agent, or None (never ran a turn, evicted, or the lookup failed:
        a question then answers from the transcript digest, an exploration is refused)."""
        try:
            return self._cached_agent_for(self._session_key_for_source(source))
        except Exception:  # a lookup failure is "no live agent", never a failed /btw
            logger.debug("/btw live agent lookup failed", exc_info=True)
            return None

    async def _btw_ask(self, event: MessageEvent, question: str) -> str:
        """/btw <question> — one-shot auxiliary LLM call on a transcript snapshot; live history is
        never touched (alternation + prompt cache intact, current turn keeps running). Unlike /bg,
        which spawns a fresh contextless session."""
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
        except Exception:  # an unresolvable runtime is the no-provider reply below
            logger.debug("/btw runtime resolution failed", exc_info=True)
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
        parent_agent = self._btw_live_agent(source)
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

"""The per-turn Slack thinking card (``platforms.slack.extra.thinking_cards``) for ``SlackAdapter``.

Narration between tool calls lands as one collapsible row per burst on a card of its own (slot
``thinking`` beside the tool card). With ``show_reasoning`` on, a reasoning block that lasts long
enough also gets a row: ``Thinking…`` while it runs, then ``Thought for Ns`` with a short tail of
its notes. Stream lifecycle helpers come from ``adapter_task_cards.SlackTaskCardMixin``.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from agent.i18n import t
from gateway.platforms.base import SendResult

try:  # sibling module; support both package and flat plugin-dir import
    from .adapter_task_cards import _NativeTaskCardStream
except ImportError:  # pragma: no cover - plugin loaded outside package context
    from adapter_task_cards import _NativeTaskCardStream  # type: ignore

logger = logging.getLogger("plugins.platforms.slack.adapter")

THINKING_SLOT = "thinking"


class SlackThinkingCardMixin:
    """Thinking-card sender, mixed into ``SlackAdapter``."""

    def thinking_cards_enabled(self) -> bool:
        """Return whether the per-turn thinking card is explicitly enabled (plan-option-a).

        Opt-in beside ``native_task_cards``; the thinking card shares the same stream
        machinery, keyed with slot "thinking" beside the tool card's "tools"."""
        extra = self.config.extra if isinstance(self.config.extra, dict) else {}
        for scope in (extra,):
            value = scope.get("thinking_cards", scope.get("thinkingCards"))
            if value is not None:
                return self._truthy_config(value)
        return False

    async def _bot_title_name(self, chat_id: str, team_id: str) -> str:
        """Display name of whichever bot serves this card (``Ji Youyan is thinking``).

        Resolved through the shared users.info path (display name, workspace-scoped);
        falls back to the auth handle, then ``Hermes``. Never raises: a title must not
        break a send."""
        bot_id = self._team_bot_user_ids.get(team_id) or self._bot_user_id
        if bot_id:
            name = await self._resolve_user_name(bot_id, chat_id, team_id)
            if name and name != bot_id:
                return name
        return (self._team_bot_names.get(team_id) or self._bot_display_name or "Hermes")

    async def send_thinking_card_progress(
        self, chat_id: str, bursts: list[str], *, title: Optional[str] = None,
        reply_to: Optional[str] = None, metadata: Optional[dict[str, Any]] = None,
        fallback_text: Optional[str] = None) -> SendResult:
        """Start or update the per-turn thinking card: one ``task_update`` row per
        narration burst (own bullet + own toggle in the plan UI). Plan mode keeps the
        card title visible after the turn seals (live-verified, probe P; timeline mode
        folds the header away) and rows expand per-row behind their own toggle.

        Delta contract: a burst's row is appended EXACTLY ONCE, carrying its
        ``details`` on that first frame only. Slack CONCATENATES ``details`` when a
        row is re-sent across frames (live-proven: the probe card's duplicated rows
        and the canary feed garble), so re-sending details is what duplicates text.
        The reopen path re-seeds the full projection into the FRESH stream, where
        there is no concat history. ``fallback_text`` is accepted for signature
        parity with the tool-card sender; the gateway owns the editable-text
        fallback rail."""
        if not self._app:
            return SendResult(success=False, error="Not connected")
        if not [b for b in bursts if str(b or "").strip()]:
            return SendResult(success=False, error="No feed")
        key, stream = self._thinking_card_stream(chat_id, reply_to, metadata)
        if key is None:
            return SendResult(success=False, error="No Slack thread target")
        async with stream.lock:
            if stream.stopped:
                return SendResult(success=False, error="Progress stream already stopped")
            try:
                client = self._get_client(chat_id, team_id=stream.team_id)
                if not stream.stream_ts:
                    await self._start_native_task_card_stream(
                        client, stream, metadata, task_display_mode="plan")
                if title is None:
                    title = t("gateway.progress.thinking_card_title",
                              name=await self._bot_title_name(chat_id, key[0]))
                plan = {"type": "plan_update", "title": str(title)[:256]}
                new_rows = [self._thinking_burst_chunk(burst, idx)
                            for idx, burst in enumerate(bursts[stream.thinking_rows_sent:], start=stream.thinking_rows_sent)
                            if str(burst or "").strip()]
                if new_rows:
                    await self._append_stream_chunks(
                        client, stream, [plan, *new_rows], chat_id, metadata,
                        reopen_chunks=[plan, *(self._thinking_burst_chunk(b, i)
                                                for i, b in enumerate(bursts) if str(b or "").strip())])
                    stream.thinking_rows_sent = len(bursts)
                    stream.thinking_rows.extend(new_rows)
                return SendResult(success=True, message_id=stream.stream_ts)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.error("[Slack] Thinking-card progress error: %s", exc, exc_info=True)
                return SendResult(success=False, error=str(exc), retryable=True)

    @staticmethod
    def _thinking_burst_chunk(burst: str, idx: int) -> dict[str, Any]:
        """One narration burst = one task_update row, appended exactly once.

        Label = the burst's first 256 characters (Slack's title cap), ellipsized
        when cut (approved mockup) -- the whole paragraph lives in ``details``
        behind the row's toggle."""
        text = str(burst)
        label = text.strip()
        if len(label) > 256:
            label = label[:255] + "…"
        return {
            "type": "task_update", "id": f"thinking-{idx}",
            "title": label,
            "status": "in_progress", "details": text,
        }

    async def stop_thinking_card_progress(
        self, chat_id: str, *, reply_to: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None) -> None:
        """Seal the thinking card. Rows arrive ``in_progress`` so the card's aggregate
        icon reads in-progress for the whole turn (fhu's nit); sealing flips every row to
        ``complete`` via a DETAILS-LESS re-send -- live-proven (status probe 3, Row B):
        a re-send without ``details`` updates the row and keeps its text, while one with
        ``details`` CONCATENATES (Row A: 'A-DETAILS-ONEA-DETAILS-ONE'). Then
        ``stopStream`` exactly once."""
        key = self._native_task_card_key(chat_id, reply_to, metadata, slot=THINKING_SLOT)
        if key is None:
            return
        stream = self._native_task_card_streams.get(key)
        if stream is None:
            return
        if self._app and stream.stream_ts and (stream.thinking_rows or stream.rows) and not stream.stopped:
            flip = [{"type": "task_update", "id": row["id"], "title": row["title"],
                     "status": "complete"} for row in [*stream.thinking_rows, *stream.rows.values()]]
            try:
                async with stream.lock:
                    if not stream.stopped and stream.stream_ts:
                        await self._get_client(stream.channel, team_id=stream.team_id).api_call(
                            "chat.appendStream", json={
                                "channel": stream.channel, "ts": stream.stream_ts, "chunks": flip})
            except Exception:  # pragma: no cover - final flip is best-effort
                logger.debug("[Slack] Thinking-card completion flip failed", exc_info=True)
        await self._stop_native_task_card_stream(key, stream)

    def _thinking_card_stream(self, chat_id, reply_to, metadata):
        key = self._native_task_card_key(chat_id, reply_to, metadata, slot=THINKING_SLOT)
        if key is None:
            return None, None
        stream = self._native_task_card_streams.get(key)
        if stream is None or stream.stopped:
            stream = _NativeTaskCardStream(team_id=key[0], channel=chat_id, thread_ts=key[2])
            self._native_task_card_streams[key] = stream
        return key, stream

    async def send_reasoning_card_row(
        self, chat_id: str, row_id: str, *, title: str, status: str, details: Optional[str] = None,
        reply_to: Optional[str] = None, metadata: Optional[dict[str, Any]] = None) -> SendResult:
        """Write one reasoning row on the thinking card (``Thinking…`` → ``Thought for Ns``).

        Title and status REPLACE on a re-send; ``details`` APPEND, so the caller sends the notes
        tail once, with the closing update. A closed row is no longer settled by the stop flip.
        """
        if not self._app:
            return SendResult(success=False, error="Not connected")
        key, stream = self._thinking_card_stream(chat_id, reply_to, metadata)
        if key is None:
            return SendResult(success=False, error="No Slack thread target")
        async with stream.lock:
            if stream.stopped:
                return SendResult(success=False, error="Progress stream already stopped")
            try:
                client = self._get_client(chat_id, team_id=stream.team_id)
                if not stream.stream_ts:
                    await self._start_native_task_card_stream(client, stream, metadata, task_display_mode="plan")
                    title_text = t("gateway.progress.thinking_card_title",
                                   name=await self._bot_title_name(chat_id, key[0]))
                    await self._append_stream_chunks(
                        client, stream, [{"type": "plan_update", "title": title_text[:256]}], chat_id, metadata)
                chunk: dict[str, Any] = {"type": "task_update", "id": row_id, "title": title[:256], "status": status}
                if details:
                    chunk["details"] = details
                await self._append_stream_chunks(client, stream, [chunk], chat_id, metadata)
                if status == "in_progress":
                    stream.rows[row_id] = {k: v for k, v in chunk.items() if k != "details"}
                else:
                    stream.rows.pop(row_id, None)
                return SendResult(success=True, message_id=stream.stream_ts)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.error("[Slack] Reasoning-card row error: %s", exc, exc_info=True)
                return SendResult(success=False, error=str(exc), retryable=True)

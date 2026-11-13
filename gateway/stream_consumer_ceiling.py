"""Long-reply file handoff for GatewayStreamConsumer (#152012).

A 415k-char reply once went out as 402 Telegram messages and the bot was flood-banned for
minutes; cutting the reply short instead loses text. So an adapter with a per-reply cap
(``MAX_SPLIT_MESSAGES``: Telegram, Discord) that can upload files never shows one reply as more
messages than that. Once a reply needs more, the consumer stops opening messages for it, waits
for the reply's complete final text, uploads the whole text as one file (``send_reply_as_file``),
and only after the upload is confirmed deletes the messages it already posted for the reply
(sealed heads, the live preview). A failed or impossible upload deletes nothing: the reply resumes
as ordinary messages from where the chat left off, so no text is lost either way.

The count spans every path that opens a message for one reply: sealed heads, the split-first
send, the final edit and fallback continuations. A reply that is not streamed is handed off by the
gateway's normal final send (``_send_with_retry`` -> ``send_long_reply_as_file``); the consumer
calls ``adapter.send`` directly, so its pieces are never filed on their own. An adapter without a
cap or without file uploads keeps today's behaviour exactly.
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from typing import Any, Optional

from gateway.platforms.helpers import reply_message_cap, send_reply_as_file

logger = logging.getLogger("gateway.stream_consumer")


class StreamCeilingMixin:
    """Counts the messages one reply opens and hands an over-cap reply off as a file."""

    def _reset_ceiling(self) -> None:
        """Per-reply state; one reply is one stream segment."""
        self._ceiling_ids: set[str] = set()
        self._ceiling_unknown = 0  # sends that may have landed without an id
        self._ceiling_hit = False  # handoff pending: nothing else opens, the final goes out as a file
        self._handoff_failed = False  # the upload failed: the reply continues as messages
        self._handoff_final: Optional[str] = None  # the authoritative final, adopted after the handoff began
        self._handoff_consumed = 0  # stream chars sealed into earlier messages when the handoff began
        self._handoff_live: tuple[Optional[str], str] = (None, "")  # (live preview id, its shown text)
        # The final's handoff runs as its own task: the gateway cancels a consumer still delivering after
        # its flush window, and a cancelled upload would leave the reply half delivered.
        self._handoff_task: Optional[asyncio.Task] = None
        self._handoff_due = False  # the handoff began at the final: a cancel may still send the file
        self._handoff_tick: Any = None

    @property
    def _reply_cap(self) -> Optional[int]:
        """The adapter's per-reply message cap (``MAX_SPLIT_MESSAGES``); None when it has none, cannot
        upload a file, or this reply's upload already failed (it then goes out as messages)."""
        return None if getattr(self, "_handoff_failed", False) else reply_message_cap(self.adapter)

    def _ceiling_opened(self) -> int:
        return len(self._ceiling_ids) + self._ceiling_unknown

    def _ceiling_note_unknown(self) -> None:
        if hasattr(self, "_ceiling_ids"):
            self._ceiling_unknown += 1

    def _ceiling_observe(self, result: Any) -> None:
        """Count the messages a successful send or edit opened (an edit's own id is already in)."""
        if not getattr(result, "success", False) or not hasattr(self, "_ceiling_ids"):
            return
        raw = getattr(result, "raw_response", None)
        ids = {str(i) for i in (getattr(result, "message_id", None),
                                *(getattr(result, "continuation_message_ids", None) or ()),
                                *((raw.get("message_ids") if isinstance(raw, dict) else None) or ()))
               if i and i != "__no_edit__"}
        if ids:
            self._ceiling_ids |= ids
        else:
            self._ceiling_unknown += 1

    def _ceiling_msgs(self, chunk: str) -> int:
        """Messages the adapter needs for ``chunk``: its formatting (MarkdownV2 escapes) can push a
        raw-sized chunk past the platform limit, and the adapter then splits it again. An adapter that
        counts its own split (``reply_message_count``) is asked, so both sides agree on the cap."""
        if callable(getattr(type(self.adapter), "reply_message_count", None)):
            return max(1, self.adapter.reply_message_count(chunk, self.metadata))
        fmt, limit = getattr(self.adapter, "format_message", None), getattr(self.adapter, "MAX_MESSAGE_LENGTH", 0)
        if not (callable(fmt) and type(limit) is int and limit > 0) or self._safe_limit > limit:
            return 1  # a rich-sized chunk is one message
        size = self._len_fn(fmt(chunk))
        return 1 if size <= limit else -(-size // (limit - 200))

    def _ceiling_need(self, text: str) -> int:
        """Messages ``text`` opens when the consumer splits it."""
        chunks = self._truncate_for_stream(text, self._safe_limit, self._len_fn) if text else []
        return sum(self._ceiling_msgs(chunk) for chunk in chunks)

    def _begin_handoff(self, *, final: bool = False) -> None:
        """Stop opening messages for this reply; what is on screen stays until the final is known.
        ``final``: the reply is already complete (its final text is in the stream ledger)."""
        logger.warning("Reply needs more than %d messages: it will be sent as a file", self._reply_cap)
        self._ceiling_hit, self._handoff_due = True, final
        self._handoff_consumed = len(self._stream_ledger) - len(self._accumulated)
        live = self._message_id if self._message_id and self._message_id != "__no_edit__" else None
        self._handoff_live = (live, self._last_sent_text if live else "")
        self._accumulated = ""
        self._message_id = None
        self._last_sent_text = ""

    def _ceiling_final_cut(self) -> bool:
        """At the final, before the last edit: hand the reply off when that edit (a rich-sized live
        message the adapter splits) would open messages past the cap."""
        cap = self._reply_cap
        if cap is None or self._ceiling_hit or not self._accumulated or self._overflows():
            return self._ceiling_hit
        if self._ceiling_opened() + self._ceiling_msgs(self._accumulated) - (self._message_id is not None) <= cap:
            return False
        self._begin_handoff(final=True)
        return True

    def _ceiling_seal_cut(self, split_at: int, *, final: bool) -> bool:
        """Before a seal: True (do not seal) once the reply is handed off. It is handed off when
        sealing would pass the cap: mid-stream the next live message, at the final the rest of the
        reply."""
        cap = self._reply_cap
        if cap is None or self._ceiling_hit:
            return self._ceiling_hit
        head, rest = self._accumulated[:split_at], self._accumulated[split_at:].lstrip("\n")
        # The seal reuses the live message; its extra parts and the messages after it are new.
        after = (self._ceiling_msgs(head) - 1) + (self._ceiling_need(rest) if final else 1)
        if self._ceiling_opened() + after <= cap:
            return False
        self._begin_handoff(final=final)
        return True

    async def _ceiling_split_first_send(self, tick: Any) -> Optional[bool]:
        """Before ``_split_first_send``: hand the reply off when its chunks would pass the cap.
        None lets the normal split run (under the cap, or the upload failed and the reply resumes);
        otherwise the split-first send's own return (True: the turn finished here)."""
        cap = self._reply_cap
        if cap is None:
            return None
        if self._ceiling_opened() + self._ceiling_need(self._accumulated) <= cap:
            return None
        self._begin_handoff(final=tick.got_done)
        if tick.got_done and not await self._handoff_at_final(tick):
            await self._finalize_turn(tick)  # the upload failed: finish the resumed messages normally
        return bool(tick.got_done)

    def _ceiling_adopt_final(self, final_raw: str) -> bool:
        """During a handoff the completed final is only recorded: it becomes the file."""
        if not getattr(self, "_ceiling_hit", False):
            return False
        self._handoff_final = final_raw
        return True

    async def _deliver_reply_file(self, text: str) -> bool:
        """Upload ``text`` whole; once it is confirmed, delete this reply's messages and record the
        file as the turn's delivery (the gateway's normal final send then does nothing)."""
        if not text.strip():
            return False
        # A later segment repeating a reply already delivered (the agent re-streams it after a tool round,
        # led by a paragraph break) is not uploaded again; its previews are only removed.
        delivered = self._shown_as(text, self._delivered_segment_texts)
        ids = [] if delivered else await send_reply_as_file(
            self.adapter, self.chat_id, text, reply_to=self._initial_reply_to_id, metadata=dict(self.metadata or {}))
        if ids is None:
            return False
        stale = sorted(self._ceiling_ids - set(ids))
        await self._delete_previews(stale, label="Long reply")  # a refused delete leaves that message
        logger.info("Long reply delivered as %d file(s); removed up to %d earlier message(s)", len(ids), len(stale))
        self._preview_message_ids -= set(stale)
        self._accumulated, self._message_id, self._last_sent_text = "", None, ""
        self._already_sent = True
        self._mark_final_delivered()
        self._delivered_final_text = self._display_payload(text)
        # Durable record: a segment reset clears the final flags, and a final equal to this file must
        # still read as delivered (else the gateway's fallback uploads it a second time).
        if not delivered:
            self._delivered_segment_texts.append(self._delivered_final_text)
        return True

    async def _handoff_at_final(self, tick: Any) -> bool:
        """The reply is complete: send it as a file, shielded from the gateway's flush-window cancel
        (``_ceiling_cancelled`` waits for it). True when the turn is settled here, False when the
        normal finalize still owns the tail (``tick.update_visible`` says whether it is on screen)."""
        if self._handoff_task is None:
            self._handoff_tick = tick
            self._handoff_task = asyncio.ensure_future(self._run_handoff(tick))
        return await asyncio.shield(self._handoff_task)

    async def _run_handoff(self, tick: Any) -> bool:
        """``_handoff_at_final``'s body. On a failed upload the reply resumes as messages from where
        the chat left off; nothing is deleted."""
        await self._notify_before_finalize()
        final = self._handoff_final if self._handoff_final is not None else self._stream_ledger
        if await self._deliver_reply_file(self._clean_for_display(final)):
            return True
        return await self._resume_as_messages(final, tick, is_turn_final=True)

    async def _resume_as_messages(self, final: str, tick: Any, *, is_turn_final: bool) -> bool:
        """The upload failed: put back what the chat shows (sealed heads, the live preview) and send
        the rest as ordinary messages, paced by the adapter's flood handling. Returns like
        ``_handoff_at_final``."""
        logger.warning("Long reply file failed; delivering the rest as messages")
        consumed, (live_id, live_text) = self._handoff_consumed, self._handoff_live
        self._ceiling_hit, self._handoff_failed = False, True
        if not final.startswith(self._stream_ledger[:consumed]):  # the final rewrote the sealed heads
            consumed, live_id, live_text = 0, None, ""
        self._stream_ledger, self._accumulated = final, final[consumed:].lstrip("\n")
        self._message_id, self._last_sent_text = live_id, live_text
        await self._seal_overflow_heads(final=True)
        if self._first_send_overflows():
            if await self._split_first_send(tick):
                return True
        if not self._accumulated:  # the sealed heads hold the whole reply
            if is_turn_final:
                self._mark_final_delivered(record=final)
            return is_turn_final
        tick.update_visible = await self._send_or_edit(self._accumulated, finalize=True, is_turn_final=is_turn_final)
        return False

    async def _ceiling_segment_end(self, tick: Any = None) -> None:
        """A tool boundary or commentary ends a reply being handed off: its text so far is complete,
        so it goes out as a file now (or, on failure, as messages) before the next segment starts."""
        if not getattr(self, "_ceiling_hit", False):
            return
        tick = tick or SimpleNamespace(got_done=False, got_segment_break=True, update_visible=False)
        if not await self._deliver_reply_file(self._clean_for_display(self._stream_ledger)):
            await self._resume_as_messages(self._stream_ledger, tick, is_turn_final=False)

    async def _ceiling_finish(self, tick: Any) -> bool:
        """At got_done: True when a pending handoff finished the turn (file delivered, or the
        message fallback already sent everything)."""
        if not getattr(self, "_ceiling_hit", False):
            return False
        return await self._handoff_at_final(tick)

    async def _ceiling_cancelled(self) -> bool:
        """Cancel path (the gateway's flush window ran out while the consumer was still delivering):
        a handoff in flight is waited for, and a reply whose cancel-time edit would pass the cap is
        handed off too, once its final is known (``finish()`` queues it before the cancel). True when
        the turn is settled here; False hands over to the normal best-effort edit."""
        cap = self._reply_cap
        if self._handoff_task is not None:
            tick, settled = self._handoff_tick, await self._handoff_task
        elif cap is None or not (self._ceiling_hit or (self._accumulated and self._message_id and (
                self._ceiling_opened() + self._ceiling_msgs(self._accumulated) - 1 > cap))):
            return False
        else:
            tick = self._drain_queue()  # a finish() queued before the cancel carries the final text
            if not (tick.got_done or self._handoff_due or self._handoff_final is not None):
                # Only a partial is known: open nothing more. The gateway's own send delivers the
                # complete reply (as a file, past the cap); the messages already shown stay.
                return True
            if not self._ceiling_hit:
                self._begin_handoff(final=True)
            settled = await self._run_handoff(tick)
        if not settled and tick.update_visible:
            self._mark_final_delivered(record=self._accumulated)
        return True

    async def _ceiling_fallback_file(self, text: str, chunks: list[str]) -> bool:
        """Fallback continuation (edits stopped working): when its chunks would pass the cap, send
        the whole reply as a file instead. False sends the chunks as before."""
        if self._reply_cap is None or self._ceiling_opened() + len(chunks) <= self._reply_cap:
            return False
        whole = self._stream_ledger if self._turn_split_delivery and self._stream_ledger else text
        if not await self._deliver_reply_file(self._clean_for_display(whole)):
            return False
        self._fallback_final_send, self._fallback_prefix = False, ""
        return True

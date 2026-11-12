"""Per-reply message ceiling for GatewayStreamConsumer (#152012).

An adapter that caps one split delivery (``MAX_SPLIT_MESSAGES``: Telegram, Discord) sees only the
messages one ``send``/``edit_message`` call opens. The consumer opens more on its own: it seals the
live preview and starts a new one every ~4,000 chars, sends the heads of a reply that overflows
before its first preview, and sends fallback continuations. This mixin counts every message one
reply (one stream segment) opens across all of them and keeps the total at or under the adapter's
cap: at most ``cap - 1`` messages carry text and the last is the adapter's own localized truncation
notice (none when warning notifications are off; the tail is dropped either way). The full text
stays in the session history; the cut delivery is recorded as the final it was cut from, so the
gateway's normal final send never repeats a head that already landed. Every send and edit the
consumer makes to a capped adapter carries ``metadata["_split_budget"]``, the messages that call may
open, so the adapter's own split of an inflated chunk stays inside the same ceiling. An adapter
without a cap keeps today's behaviour exactly.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Optional

from gateway.platforms.base import _custom_unit_to_cp

logger = logging.getLogger("gateway.stream_consumer")

_INDICATOR_RE = re.compile(r" \((\d+)/(\d+)\)$")


class StreamCeilingMixin:
    """Message budget shared by sealed heads, split-first sends, overflow edits and fallbacks."""

    def _reset_ceiling(self) -> None:
        """Per-reply state; one reply is one stream segment."""
        self._ceiling_ids: set[str] = set()
        self._ceiling_unknown = 0  # sends that may have landed without an id
        self._ceiling_hit = False  # the reply was cut: nothing else opens
        self._ceiling_notice_id: Optional[str] = None
        self._ceiling_dropped = 0  # the dropped count the notice shows
        self._ceiling_shown = 0  # stream chars on screen before the cut
        self._ceiling_final = ""  # the authoritative final, when it arrives after the cut
        self._split_inflight: Optional[str] = None  # heads a split-first send landed so far
        self._ceiling_adapter_dropped = -1  # text the adapter's own cap dropped (-1: it did not cap)

    @property
    def _reply_cap(self) -> Optional[int]:
        """The adapter's split-delivery ceiling; None (no ceiling) for adapters without one."""
        cap = getattr(self.adapter, "MAX_SPLIT_MESSAGES", None)
        return cap if type(cap) is int and cap > 1 else None

    def _ceiling_remaining(self) -> int:
        return self._reply_cap - len(self._ceiling_ids) - self._ceiling_unknown

    def _add_split_budget(self, meta: dict, *, editing: bool = False) -> dict:
        """Messages this call may open, its own notice included (an edit also reuses the message
        it edits)."""
        if self._reply_cap is not None and hasattr(self, "_ceiling_ids"):
            meta["_split_budget"] = max(1, self._ceiling_remaining() + editing)
        return meta

    def _ceiling_reserve(self) -> int:
        """Slots a cut keeps for the notice: one, or none when warnings are hidden."""
        return int(self._ceiling_notice(0) is not None)

    def _ceiling_note_unknown(self) -> None:
        if hasattr(self, "_ceiling_ids"):
            self._ceiling_unknown += 1

    def _ceiling_observe(self, result: Any) -> None:
        """Count the messages a successful send or edit opened (an edit's own id is already in)."""
        if self._reply_cap is None or not getattr(result, "success", False) or not hasattr(self, "_ceiling_ids"):
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
        if isinstance(raw, dict) and raw.get("split_capped"):
            # The adapter capped this call inside the budget and posted the notice itself, last.
            last = [*(raw.get("message_ids") or ()), *(getattr(result, "continuation_message_ids", None) or ())]
            self._ceiling_hit = True
            self._ceiling_notice_id = str(last[-1] if last else getattr(result, "message_id", "") or "") or None
            self._ceiling_adapter_dropped = int(raw.get("dropped_chars") or 0)

    def _ceiling_msgs(self, chunk: str) -> int:
        """Messages the adapter needs for ``chunk``: its formatting (MarkdownV2 escapes) can push a
        raw-sized chunk past the platform limit, and the adapter then splits it again."""
        fmt, limit = getattr(self.adapter, "format_message", None), getattr(self.adapter, "MAX_MESSAGE_LENGTH", 0)
        if not (callable(fmt) and type(limit) is int and limit > 0) or self._safe_limit > limit:
            return 1  # a rich-sized chunk is one message; the adapter's own budget covers a legacy split
        size = self._len_fn(fmt(chunk))
        return 1 if size <= limit else -(-size // (limit - 200))

    def _ceiling_head(self, text: str) -> int:
        """Where a head of ``text`` is cut (the seal rule), shrunk to fit one formatted message."""
        cp_budget = _custom_unit_to_cp(text, self._safe_limit, self._len_fn)
        line = text.rfind("\n", 0, cp_budget)
        return self._ceiling_fit(text, cp_budget if line < cp_budget // 2 or cp_budget == len(text) else line)

    def _ceiling_fit(self, text: str, cut: int) -> int:
        """Shrink ``cut`` (to a line break where one is near) until ``text[:cut]`` plus a part
        marker is one message."""
        while cut > 200 and self._ceiling_msgs(text[:cut] + " (88/88)") > 1:
            smaller = int(cut * 0.9)
            line = text.rfind("\n", 0, smaller)
            cut = line if line > smaller // 2 else smaller
        return cut

    def _ceiling_notice(self, dropped: int) -> Optional[str]:
        notice = getattr(self.adapter, "_split_cap_notice", None)
        return notice(max(0, dropped)) if callable(notice) else None

    def _ceiling_dropped_now(self) -> int:
        source = self._ceiling_final or self._stream_ledger
        return max(0, len(self._clean_for_display(source)) - self._ceiling_shown)

    def _ceiling_cut(self, shown: int, prefix_len: Optional[int] = None) -> None:
        """Stop the reply: ``shown`` stream chars are on screen and nothing else opens. The shown
        source prefix becomes the capped record a final that starts with it matches."""
        self._ceiling_hit = True
        self._ceiling_shown = shown
        self._turn_split_delivery = True
        self._accumulated = self._last_sent_text = ""
        self._message_id = None
        cut = shown if prefix_len is None else prefix_len
        self._split_capped = self._clean_for_display(self._stream_ledger[:cut]).strip()

    async def _ceiling_send_notice(self) -> None:
        """The reply's one truncation notice, in the last slot."""
        dropped = self._ceiling_dropped_now()
        notice = self._ceiling_notice(dropped)
        if not notice or self._ceiling_remaining() <= 0:
            return
        logger.warning("Reply reached the %d-message delivery ceiling; %d chars are only in the session log",
                       self._reply_cap, dropped)
        try:
            result = await self.adapter.send(chat_id=self.chat_id, content=notice,
                                             metadata=self._metadata_for_send(final=True))
        except Exception:  # health: allow BLE001 -- best-effort notice; the reply is already capped
            logger.warning("Ceiling notice send failed", exc_info=True)
            return
        self._ceiling_observe(result)
        if getattr(result, "success", False):
            self._ceiling_notice_id, self._ceiling_dropped = getattr(result, "message_id", None), dropped

    async def _ceiling_seal_cut(self, split_at: int, *, final: bool) -> bool:
        """Sealing the live preview would open a message past ``cap - 1``: seal it with the head,
        end the reply with the notice. False (seal normally) under the ceiling, or at the final
        when the rest fits the last slot."""
        if self._reply_cap is None:
            return False
        head, rest = self._accumulated[:split_at], self._accumulated[split_at:].lstrip("\n")
        # The seal reuses the live message; its extra parts and the next live message are new.
        after = self._ceiling_remaining() - (self._ceiling_msgs(head) - 1) - 1
        if after >= self._ceiling_reserve() or (
                final and after >= 0 and self._ceiling_msgs(rest) <= after + 1 - self._ceiling_reserve()):
            return False
        split_at = self._ceiling_fit(self._accumulated, split_at)
        shown = len(self._stream_ledger) - len(self._accumulated) + split_at
        if not await self._send_or_edit(self._accumulated[:split_at], finalize=True, is_turn_final=False):
            return True  # edits failed: fallback mode sends the tail, capped by _ceiling_cap_chunks
        self._ceiling_cut(shown - max(0, self._ceiling_adapter_dropped))
        if self._ceiling_notice_id is None:
            await self._ceiling_send_notice()
        return True

    async def _ceiling_split_first_send(self, tick: Any, chunks: list[str]) -> Optional[bool]:
        """``_split_first_send`` when its chunks would pass the ceiling: send the heads that fit
        (renumbered), then the notice. None when they fit (the normal split runs)."""
        if self._reply_cap is None:
            return None
        remaining, reserve = self._ceiling_remaining(), self._ceiling_reserve()
        need = sum(self._ceiling_msgs(c) for c in chunks)
        if need <= remaining - (0 if tick.got_done else reserve):
            return None
        source, base = self._accumulated, len(self._stream_ledger) - len(self._accumulated)
        self._ceiling_cut(base)
        sent, reply_to, total = "", self._initial_reply_to_id, self._reply_cap  # parts of the whole reply
        while self._ceiling_remaining() > reserve and self._ceiling_notice_id is None and len(sent) < len(source):
            rest = source[len(sent):]
            cut = self._ceiling_head(rest)
            if cut >= len(rest) and not tick.got_done:
                break  # the stream goes on: this tail is the live preview, as in a normal split
            n = self._reply_cap - self._ceiling_remaining() + 1
            new_id = await self._send_new_chunk(f"{rest[:cut]} ({n}/{total})", reply_to, final=True)
            self._message_id = None
            if new_id is None or new_id == reply_to:
                break
            reply_to = None if new_id == "__no_edit__" else new_id
            sent += rest[:cut]
            self._ceiling_cut(base + len(sent))
        if not sent or (len(sent) < len(source) and self._ceiling_remaining() > reserve):
            # No cut: nothing landed (the full text stays for the next tick or the gateway's send),
            # or the stream goes on with its tail as the live preview.
            self._ceiling_hit, self._split_capped = False, ""
            self._accumulated = source[len(sent):].lstrip("\n") if sent else source
            self._turn_split_delivery = self._turn_split_delivery or bool(sent)
            return False if sent else bool(tick.got_done)
        if self._ceiling_notice_id is None:
            await self._ceiling_send_notice()
        if tick.got_done:
            await self._ceiling_finish()
        return bool(tick.got_done)

    def _ceiling_chunk_limit(self, raw_limit: int) -> int:
        """Fallback chunk size: a capped adapter gets its streaming limit (Telegram's rich cap), so a
        rich tail is not cut into 4,096-char messages that spend the reply's budget."""
        return max(raw_limit, self._raw_message_limit()) if self._reply_cap is not None else raw_limit

    def _ceiling_spent(self, result: Any) -> bool:
        """Stop the fallback loop: the adapter capped this send, or the budget is used up."""
        raw = getattr(result, "raw_response", None)
        return self._reply_cap is not None and (
            (isinstance(raw, dict) and bool(raw.get("split_capped"))) or self._ceiling_remaining() <= 0)

    def _ceiling_cap_chunks(self, chunks: list[str]) -> list[str]:
        """Fallback continuation chunks capped to the reply's remaining slots, the notice last."""
        if self._reply_cap is None or len(chunks) <= self._ceiling_remaining():
            return chunks
        keep = max(0, self._ceiling_remaining() - 1)
        notice = self._ceiling_notice(sum(len(c) for c in chunks[keep:]))
        self._ceiling_hit = True
        logger.warning("Fallback continuation capped at the %d-message delivery ceiling", self._reply_cap)
        return chunks[:keep] + ([notice] if notice else [])

    def _ceiling_adopt_final(self, final_raw: str) -> bool:
        """After a cut, the completed final only refreshes the dropped count (nothing is shown)."""
        if not getattr(self, "_ceiling_hit", False) or self._accumulated:
            return False
        self._ceiling_final = final_raw
        return True

    async def _ceiling_finish(self) -> bool:
        """Close a consumer-cut reply as the turn's delivery: refresh the notice's dropped count
        once, then record the capped delivery so the gateway's normal final send does nothing.
        False when the reply was not cut here (the normal finalize runs)."""
        if not getattr(self, "_ceiling_hit", False) or self._accumulated:
            return False
        dropped = self._ceiling_dropped_now()
        if self._ceiling_notice_id and dropped != self._ceiling_dropped and (notice := self._ceiling_notice(dropped)):
            result = await self._edit_message(message_id=self._ceiling_notice_id, content=notice, finalize=True)
            if getattr(result, "success", False):
                self._ceiling_dropped = dropped
        await self._notify_before_finalize()
        self._already_sent = True
        self._mark_final_delivered()
        self._delivered_final_text = self._split_capped or None
        return True

    async def _ceiling_cancelled(self) -> bool:
        """Cancel-path delivery under a ceiling (the gateway cancels a consumer still delivering
        after its flush window); False hands over to the normal best-effort edit. Heads that
        already landed are never sent again: the rest goes out once, inside the budget."""
        if self._reply_cap is None:
            return False
        if await self._ceiling_finish():
            if self._ceiling_notice_id is None:
                await self._ceiling_send_notice()
            return True
        if self._split_inflight:
            # Cancelled mid split-first send: the landed heads stay; send only the rest.
            sent = _INDICATOR_RE.sub("", self._split_inflight)
            self._accumulated = self._accumulated[len(os.path.commonprefix([self._accumulated, sent])):].lstrip("\n")
            self._message_id, self._split_inflight, self._turn_split_delivery = None, None, True
            if self._accumulated.strip() and not await self._send_or_edit(
                    self._accumulated, finalize=True, is_turn_final=False):
                return False
            self._mark_final_delivered(record=self._accumulated)
            return True
        if not (self._accumulated and self._message_id):
            return False
        source = self._accumulated
        ok = bool(await self._send_or_edit(source, finalize=True, is_turn_final=False))
        if not ok and self._fallback_final_send and self._fallback_prefix:
            # A continuation of the overflow edit was refused after its head landed: send only the
            # unseen tail (capped). If that is refused too, or the tail cannot be told apart from the
            # head, the landed head is the reply, cut there: the gateway's normal send would post the
            # whole reply again beside it.
            shown = len(self._stream_ledger) - len(source) + len(self._fallback_prefix)
            if self._continuation_text(source) != source:
                await self._send_fallback_final(source)
            if not self._final_response_sent:
                self._ceiling_cut(shown)
                await self._ceiling_send_notice()
                await self._ceiling_finish()
            return True
        if ok and not self._final_response_sent:
            if self._last_edit_capped and not self._split_capped:
                self._split_capped = self._capped_source(self._accumulated)
            self._mark_final_delivered(record=self._accumulated)
        return True

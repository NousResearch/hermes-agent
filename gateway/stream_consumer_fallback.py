"""Fallback delivery for GatewayStreamConsumer: continuation sends after edits
stop working, chunking, cursor cleanup, commentary and silence retraction."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any, Callable, Optional

from gateway.platforms.base import BasePlatformAdapter as _BasePlatformAdapter
from gateway.stream_consumer_fences import ensure_closed_code_fences, open_code_fence_span

logger = logging.getLogger("gateway.stream_consumer")


class StreamFallbackMixin:
    """Non-streaming delivery paths used once progressive edits fail or the turn ends oddly."""

    async def _send_new_chunk(self, text: str, reply_to_id: Optional[str], *,
                              final: bool = False, first: bool = False) -> Optional[str]:
        """Send a sealed head. ``__no_edit__`` means it may have landed without an id;
        ``__partial__`` that the *first* head landed in part (see :meth:`_adopt_partial_head`)."""
        text = self._clean_for_display(text)
        if not text.strip():
            return reply_to_id
        try:
            result = await self.adapter.send(
                chat_id=self.chat_id, content=text, reply_to=reply_to_id,
                metadata=self._metadata_for_send(final=final, expect_edits=not final))
        except Exception as e:
            logger.error("Stream send chunk error: %s", e, exc_info=True)
            if await self._uncertain_send_fully_shown(e):
                self._already_sent = True
                self._last_sent_text = text
                return "__no_edit__"
            return reply_to_id
        if not result.success and await self._uncertain_send_fully_shown(result):
            # The head may be visible without a message id. Do not send it again
            # inside the whole answer; the split loop must send only its unseen tail.
            self._already_sent = True
            self._last_sent_text = text
            return "__no_edit__"
        if first and not result.success and self._adopt_partial_head(result, text):
            return "__partial__"
        if not (result.success and result.message_id):
            self._edit_supported = False
            return reply_to_id
        self._message_id = str(result.message_id)
        self._track_preview_ids_from_result(result)
        self._already_sent = True
        self._last_sent_text = text
        self._notify_new_message()
        return str(result.message_id)

    def _adopt_partial_head(self, result: Any, text: str) -> bool:
        """A head send whose first parts landed (``partial_overflow`` naming a ``delivered_prefix``
        of *text*): keep them as the shown prefix and enter fallback mode, so got_done sends only the
        unseen tail. Without an exact prefix the caller resends it all (a repeated head beats a lost tail)."""
        raw = getattr(result, "raw_response", None)
        prefix = raw.get("delivered_prefix") if isinstance(raw, dict) and raw.get("partial_overflow") else None
        if not (isinstance(prefix, str) and prefix.strip() and text.startswith(prefix)):
            return False
        self._message_id = str(raw.get("last_message_id") or result.message_id or "__no_edit__")
        self._track_preview_ids_from_result(result)
        self._last_sent_text = prefix
        self._fallback_preserve_partial_messages = True
        self._enter_fallback_mode(prefix)
        self._notify_new_message()
        return True

    def _visible_prefix(self) -> str:
        """Return the visible text already shown in the streamed message."""
        prefix = self._last_sent_text or ""
        cursor = self.cfg.cursor
        if cursor and prefix.endswith(cursor):
            prefix = prefix[:-len(cursor)]
        elif cursor and cursor in prefix:
            # The display fence was closed after the cursor: the user saw the raw text, closed.
            # Compared raw, a decoration never reads as unseen text that must be sent again.
            raw = prefix[:prefix.rfind(cursor)]
            if ensure_closed_code_fences(raw + cursor) == prefix:
                prefix = ensure_closed_code_fences(raw)
        return self._clean_for_display(prefix)

    def _continuation_text(self, final_text: str) -> str:
        """Return only the part of final_text the user has not already seen."""
        # A segment after a tool call streams with the separator's leading newlines; the final is
        # stripped. Leading whitespace never renders, so it must not make shown text read as unseen.
        prefix = (self._fallback_prefix or self._visible_prefix()).lstrip()
        lead = len(final_text) - len(final_text.lstrip())
        if prefix and final_text[lead:].startswith(prefix):
            cut = lead + len(prefix)
            span = open_code_fence_span(final_text, cut)
            if span is None:
                return final_text[self._prose_resume_point(final_text, cut):].lstrip()
            # The tail starts inside a code block the user saw closed: reopen it with the model's
            # own opener (a cut inside its language tag must not turn the tag's rest into code),
            # and keep the code's indentation. A cut mid-line re-sends that whole line: a code
            # line resumed from a word boundary loses its indentation and its first tokens, so
            # neither message would hold the complete line. A tail that is only the closing
            # fence adds nothing the closed preview did not already show.
            start, end = span
            if cut > end and "\n" not in (final_text[cut - 1], final_text[cut:cut + 1] or "\n"):
                cut = final_text.rfind("\n", 0, cut) + 1
            tail = final_text[max(cut, end):].lstrip("\n")
            return f"{final_text[start:end]}\n{tail}" if tail.replace("```", "").strip() else ""
        return final_text

    @staticmethod
    def _prose_resume_point(final_text: str, cut: int) -> int:
        """Back a prose cut up to the last space or newline.

        ``cut`` follows whatever the last successful edit put on screen. Edits fire on a
        throttle tick, not at a word boundary, so it can fall inside a word; re-sending the
        broken word's tail reads as an ordinary continuation. A prefix with no boundary (one
        very long token) keeps the original cut rather than re-sending the whole reply, and a
        prefix that already ends on a whole word needs no back-up (it would repeat that word).
        """
        if cut < len(final_text) and not final_text[cut].isspace() and not final_text[cut - 1].isspace():
            boundary = max(final_text.rfind(" ", 0, cut), final_text.rfind("\n", 0, cut))
            if boundary >= 0:
                return boundary + 1
        return cut

    @staticmethod
    def _split_text_chunks(text: str, limit: int, len_fn: Callable[[str], int] = len,
                           ) -> list[str]:
        """Split text for fallback sends: newline-preferred, fence-balanced across chunks."""
        from gateway.platforms.helpers import split_text_fence_aware
        return split_text_fence_aware(text, limit, len_fn, prefer_paragraphs=False,
                                      balance_fences=True)

    def _truncate_for_stream(self, text: str, limit: int, len_fn: Callable[[str], int],
                             ) -> list[str]:
        """Split via the adapter's canonical truncate_message (platform-specific rules);
        non-base test doubles / legacy adapters keep the two-argument call shape."""
        truncate = getattr(self.adapter, "truncate_message", None)
        if not callable(truncate):
            return self._split_text_chunks(text, limit, len_fn)
        if isinstance(self.adapter, _BasePlatformAdapter):
            chunks = truncate(text, limit, len_fn=len_fn)
        else:
            chunks = truncate(text, limit)
        if not isinstance(chunks, (list, tuple)) or not all(isinstance(c, str) for c in chunks):
            return self._split_text_chunks(text, limit, len_fn)
        return list(chunks)

    async def _send_fallback_final(self, text: str) -> None:
        """Send the final continuation after streaming edits stop working (one flood retry
        per chunk)."""
        if getattr(self, "_egress_declined", False):
            # The connector refused this destination earlier in the run. The
            # whole point of this path is to deliver the unseen tail as a NEW
            # message, which is exactly the re-addressing the egress guard
            # exists to stop.
            logger.warning(
                "suppressing the fallback continuation: the connector already "
                "declined this destination for this run"
            )
            return
        # Balance fences BEFORE computing the continuation so the closing fence
        # reaches the user even when only the tail is delivered.
        final_text = ensure_closed_code_fences(self._clean_for_display(text))
        continuation = self._continuation_text(final_text)
        self._fallback_final_send = False
        if not continuation.strip():
            continuation = await self._fallback_when_nothing_unseen(final_text)
            if continuation is None:
                return

        _len_fn, raw_limit = self._fallback_len_budget()
        chunks = self._split_text_chunks(continuation, max(500, raw_limit - 100), len_fn=_len_fn)

        stale_message_id = self._message_id  # partial message to clean up
        last_message_id: Optional[str] = None
        last_successful_chunk = ""
        sent_any_chunk = False
        # Thread only a FULL resend (it replaces the preview); a tail continuation
        # keeps its existing unthreaded delivery on every platform.
        anchor = self._initial_reply_to_id if continuation == final_text else None
        for chunk in chunks:
            try:
                result = await self._send_with_flood_retry(
                    content=chunk, reply_to=None if sent_any_chunk else anchor,
                    retry_log="Flood control on fallback send, retrying in %.1fs")
            except Exception as exc:
                # A RAISED timeout after the request left is the same unknown as a returned one.
                if not await self._uncertain_send_fully_shown(exc):
                    raise
                result, shown = None, True
            else:
                shown = bool(result) and not result.success and await self._uncertain_send_fully_shown(result)
            if shown:
                # This chunk may be on screen: count it, so neither it nor the whole answer is
                # sent again; later chunks still go out and must land on their own.
                sent_any_chunk = True
                last_successful_chunk = chunk
                continue
            if not result or not result.success:
                # Partial continuation landed: do NOT set _final_response_sent (the
                # gateway must still deliver the full answer); _already_sent only
                # prevents a duplicate of the partial.  Nothing landed: let the
                # gateway final send try once more.
                self._already_sent = sent_any_chunk
                self._message_id = last_message_id
                self._last_sent_text = last_successful_chunk
                self._fallback_prefix = ""
                return
            sent_any_chunk = True
            last_successful_chunk = chunk
            last_message_id = result.message_id or last_message_id
            self._notify_new_message()

        # Best-effort delete of the frozen partial — ONLY when the FULL final was
        # re-sent.  If only the missing tail went out, the partial IS the head of
        # the answer ("sent only the second half" symptom).
        if (stale_message_id and stale_message_id != last_message_id
                and not self._fallback_preserve_partial_messages and continuation == final_text):
            await self._delete_previews([stale_message_id], label="Fallback partial",
                                        skip_sentinel=False)

        self._message_id = last_message_id
        self._already_sent = True
        # Recorder substitutes the unsplit ledger on a split turn.
        self._mark_final_delivered(record=final_text)
        self._last_sent_text = chunks[-1]
        self._fallback_prefix = ""
        self._fallback_preserve_partial_messages = False

    async def _fallback_when_nothing_unseen(self, final_text: str) -> Optional[str]:
        """Fallback entered but the visible prefix already covers ``final_text``: returns the
        continuation to send (the whole final when the prefix is from a *previous* segment)
        or None when the turn is settled here."""
        visible = self._visible_prefix()
        # Leading whitespace (a post-tool segment's separator) never renders: compare without it.
        shown_whole = bool(final_text.strip()) and final_text.lstrip() == visible.lstrip()
        if self._preview_uncertain and shown_whole:
            # The whole answer went out in a preview whose send timed out: it may already be on
            # screen and, with no id, cannot be replaced, so any resend is a second copy.
            self._already_sent = True
            self._fallback_prefix = ""
            self._mark_final_delivered(record=final_text)
            return None
        # Telegram clients can lose (part of) a streamed preview after a failed
        # final edit, so opt-in adapters commit a fresh final send.
        if shown_whole and getattr(self.adapter, "RESEND_FINAL_ON_EMPTY_STREAM_FALLBACK", False) is True:
            delivery = await self._send_empty_fallback_final(final_text)
            if delivery == "delivered":
                return None
            self._already_sent = True
            self._fallback_prefix = ""
            self._fallback_preserve_partial_messages = False
            # "ambiguous" (timeout: Telegram may have accepted) and "preview" (flood:
            # the complete ACKed preview is authoritative) keep dup suppression;
            # "failed" lets the gateway perform its normal final send.
            self._final_content_delivered = delivery in {"ambiguous", "preview"}
            if delivery == "preview":
                # This branch is only reached when the ACKed preview already shows the complete final text
                # (final_text == _visible_prefix()), so record it as the turn-final payload: the gateway's
                # reconciliation then confirms delivery instead of re-sending a second bubble next to the
                # never-deleted preview (#71047 Problem B).
                self._record_turn_final_payload(final_text)
            elif delivery == "ambiguous":
                self._delivery_ambiguous = True
            else:
                self._final_response_sent = False
            return None
        # The prefix may be from a *previous* segment (before a tool boundary),
        # wrongly reading as "already shown" — send final_text as-is.
        if final_text.strip() and not shown_whole:
            return final_text
        # Best-effort strip of a cursor left stuck by the edit failure.
        if (self._message_id and self._last_sent_text and self.cfg.cursor
                and self._last_sent_text.endswith(self.cfg.cursor)):
            clean_text = self._last_sent_text[:-len(self.cfg.cursor)]
            with contextlib.suppress(Exception):
                result = await self._edit_message(message_id=self._message_id, content=clean_text)
                if result.success:
                    self._last_sent_text = clean_text
        self._already_sent = True
        # Recorder substitutes the full ledger on a split turn.
        self._mark_final_delivered(record=final_text)
        return None

    def _fallback_len_budget(self) -> tuple[Callable[[str], int], int]:
        """(len_fn, raw_limit) for fallback chunking — per-chat cap/unit on base adapters."""
        raw_limit = getattr(self.adapter, "MAX_MESSAGE_LENGTH", 4096)
        _len_fn: Callable[[str], int] = len
        if isinstance(self.adapter, _BasePlatformAdapter):
            _len_fn = self.adapter.message_len_fn
            try:  # per-chat cap/unit (relay adapter fronting N platforms)
                raw_limit = self.adapter.max_message_length_for_chat(self.chat_id)
                _len_fn = self.adapter.message_len_fn_for_chat(self.chat_id)
            except Exception as e:
                logger.debug("per-chat limit resolution failed: %s", e)
        return _len_fn, raw_limit

    async def _uncertain_send_fully_shown(self, result: Any) -> bool:
        """True when a failed send (result or raised exception) may have landed in full, so sending
        it again would duplicate it. Telegram's legacy path re-splits content escaping grew past the
        cap and reports the messages after the timed-out one as ``unattempted_chunks``: only those
        are sent now (the one in flight may already show); when they cannot be, the send is NOT
        fully shown and the caller treats it as undelivered (a repeated head beats a lost tail)."""
        shown = False
        while self._send_failure_may_have_delivered(result):
            raw = getattr(result, "raw_response", None)
            if not (isinstance(raw, dict) and raw.get("unattempted_chunks")):
                shown = True
                break
            result = await self.adapter._resume_partial_send(
                self.chat_id, result, reply_to=None, metadata=self._metadata_for_send(final=True))
            if result is None or getattr(result, "success", False):
                shown = result is not None
                break
        # Counted as shown on a guess: the gateway leaves the answer to the delivery ledger's marked
        # redelivery, so a send that never landed is not silence (``delivery_uncertain``).
        self._delivery_uncertain = self._delivery_uncertain or shown
        return shown

    async def _send_with_flood_retry(self, *, content: str, retry_log: str, reply_to=None):
        """adapter.send(final metadata) with ONE bounded flood retry; returns the last
        SendResult.  Exceptions propagate (callers decide whether a raise is "ambiguous")."""
        kwargs = dict(chat_id=self.chat_id, content=content,
                      metadata=self._metadata_for_send(final=True))
        if reply_to is not None:
            kwargs["reply_to"] = reply_to
        result = None
        for attempt in range(2):
            result = await self.adapter.send(**kwargs)
            if getattr(result, "success", False):
                break
            retry_delay = self._fallback_flood_retry_delay(result)
            if attempt or retry_delay is None:
                break  # non-flood error, long flood wait, or second failure
            raw = getattr(result, "raw_response", None)
            if isinstance(raw, dict) and raw.get("partial_overflow"):
                break  # split head already on screen: re-sending the whole content duplicates it
            logger.debug(retry_log, retry_delay)
            await asyncio.sleep(retry_delay)
        return result

    async def _send_empty_fallback_final(self, final_text: str) -> str:
        """Commit a completed answer after Telegram finalization fails: "delivered", "failed"
        (gateway may retry), "ambiguous" (a timeout may have landed) or "preview" (flood
        control; the complete preview is authoritative)."""
        # Segment-scoped only: never delete an earlier finalized preamble.
        stale_ids = self._stale_preview_ids(segment_only=True)
        try:
            result = await self._send_with_flood_retry(
                content=final_text, reply_to=self._initial_reply_to_id,
                retry_log="Flood control on empty fallback final send; retrying in %.1fs")
        except Exception as exc:
            logger.debug("Empty fallback final send failed: %s", exc)
            return "ambiguous" if await self._uncertain_send_fully_shown(exc) else "failed"
        if not getattr(result, "success", False):
            if self._is_flood_error(result):
                return "preview"
            return "ambiguous" if await self._uncertain_send_fully_shown(result) else "failed"

        new_message_id = getattr(result, "message_id", None)
        # Telegram reports delete failure by returning False; the flood window that
        # broke the finalize can reject this too — one bounded retry.
        await self._delete_previews(stale_ids, skip=new_message_id, label="Empty fallback",
                                    retry_on_false=True)
        self._segment_preview_message_ids = set()
        self._message_id = new_message_id or "__no_edit__"
        self._already_sent = True
        self._mark_final_delivered()
        # Record VERBATIM, not via _record_turn_final_payload: the sealed previews
        # were just deleted, so the ledger (still holding sealed heads) would claim
        # delivery for text this path removed.
        self._delivered_final_text = self._display_payload(final_text)
        self._last_sent_text = final_text
        self._fallback_prefix = ""
        self._fallback_preserve_partial_messages = False
        self._notify_new_message()
        return "delivered"

    @staticmethod
    def _send_failure_may_have_delivered(result_or_exc: Any) -> bool:
        """Return True for timeout failures where retrying may duplicate, or a result the adapter
        marked ``raw_response["delivery_uncertain"]`` (e.g. a connection reset after the write)."""
        if getattr(result_or_exc, "retryable", None) is True:
            return False
        raw = getattr(result_or_exc, "raw_response", None)
        if isinstance(raw, dict) and raw.get("delivery_uncertain"):
            return True
        error = str(getattr(result_or_exc, "error", None) or result_or_exc).lower()
        name = result_or_exc.__class__.__name__.lower()
        return "timeout" in error or "timed out" in error or "timeout" in name

    def _fallback_flood_retry_delay(self, result: Any) -> float | None:
        """Return a bounded retry delay for a fallback send, if safe to retry."""
        if not self._is_flood_error(result):
            return None
        try:
            delay = float(getattr(result, "retry_after", None) or 3.0)
        except (TypeError, ValueError):
            delay = 3.0
        if delay > self._max_fallback_flood_retry_seconds:
            logger.debug("Flood control requests %.1fs; leaving final delivery to the gateway",
                         delay)
            return None
        return max(0.0, delay)

    def _is_flood_error(self, result) -> bool:
        """Check if a SendResult failure is due to flood control / rate limiting."""
        err_lower = (getattr(result, "error", "") or "").lower()
        return "flood" in err_lower or "retry after" in err_lower or "rate" in err_lower

    async def _flush_segment_tail_on_edit_failure(self) -> None:
        """Before a segment reset, send the unseen tail as a new message (and best-effort
        strip the stuck cursor from the partial)."""
        if getattr(self, "_egress_declined", False):
            return  # a new message is exactly the re-addressing the egress guard refused
        if not self._fallback_final_send:
            await self._try_strip_cursor()
        visible = self._fallback_prefix or self._visible_prefix()
        # Clean the whole buffer before cutting: a trailing silence marker cut off on its own
        # would read as a bare marker and survive the cleaner.
        tail = self._clean_for_display(self._accumulated)
        if visible and tail.startswith(visible):
            tail = tail[len(visible):].lstrip()
        if not tail.strip():
            return
        try:
            # Interim: must never seal a native stream (see _send_commentary).
            _md = dict(self.metadata) if self.metadata else {}
            _md["_interim_send"] = True
            result = await self.adapter.send(chat_id=self.chat_id, content=tail, metadata=_md)
            if result.success:
                self._already_sent = True
        except Exception as e:
            logger.error("Segment-break tail flush error: %s", e)

    async def _try_strip_cursor(self) -> None:
        """Best-effort edit removing a stuck cursor when entering fallback mode."""
        prefix = self._visible_prefix()
        if not self._has_real_preview() or not prefix.strip():
            return
        with contextlib.suppress(Exception):  # never block the fallback path
            result = await self._edit_message(message_id=self._message_id, content=prefix)
            if getattr(result, "success", False):
                self._last_sent_text = prefix

    async def _send_commentary(self, text: str) -> bool:
        """Send a completed interim assistant commentary message."""
        text = self._clean_for_display(text)
        if not text.strip():
            return False
        try:
            # Interim: a stream-is-the-message adapter's seal-interception must not
            # turn this into draft(final=true), which would seal the live stream
            # with interim text and orphan the true final.
            _md = self._metadata_for_send(final=False) or {}
            _md["_interim_send"] = True
            # reply_to only for reply-anchored threading; Discord/Telegram use
            # thread_id metadata and reply_to on every commentary is spam.
            _plat = getattr(getattr(self.adapter, "platform", None), "value", None)
            _platform_name = str(_plat or getattr(self.adapter, "name", "")).lower()
            _needs_reply_anchor = _platform_name in ("buzz", "slack", "mattermost", "feishu")
            result = await self.adapter.send(
                chat_id=self.chat_id, content=text,
                reply_to=self._initial_reply_to_id if _needs_reply_anchor else None, metadata=_md)
            # Do NOT set _already_sent: commentary is interim, and the flag would
            # suppress the real final after multiple tool calls.
            if result.success:
                self._notify_new_message()
                # Lets run.py confirm whether an interim send carried the final.
                # Record the exact delivered text so run.py can confirm whether an interim "preview"
                # actually carried the final response, vs. unrelated commentary delivered during a session
                # split (#14238).
                self._delivered_commentary_texts.append(text)
            return result.success
        except Exception as e:
            logger.error("Commentary send error: %s", e)
            return False

    def _raw_message_limit(self) -> int:
        """Per-chat length budget (``message_len_fn`` units) before overflow splits; rich
        adapters may raise it via ``streaming_overflow_limit`` so a reply that fits one rich
        message isn't fragmented at the edit limit."""
        base = getattr(self.adapter, "MAX_MESSAGE_LENGTH", 4096)
        # isinstance gate keeps MagicMock adapters (mock attrs, not ints) on base.
        if isinstance(self.adapter, _BasePlatformAdapter):
            try:
                base = self.adapter.max_message_length_for_chat(self.chat_id)
            except Exception as e:
                logger.debug("max_message_length_for_chat failed: %s", e)
            try:
                cap = self.adapter.streaming_overflow_limit()
            except Exception as e:
                logger.debug("streaming_overflow_limit check failed: %s", e)
                cap = None
            if isinstance(cap, int) and cap > base:
                return cap
        return base

    # Fresh send carried exactly ``text`` — record it so the gateway can reconcile the flag against the
    # completed response (#71643/#95382 content-vs-flag contract).
    async def _suppress_silence_marker(self) -> None:
        """Retract any streamed preview when the final reply is a bare silence marker.  Flags
        stay False so the gateway's whole-response filter owns what goes out next: "" for a
        machinery turn, the visible fallback for a human one."""
        # A native-stream bubble isn't a deletable message — close an open one
        # (e.g. from an eager re-seed) with an empty finalize so it doesn't hang.
        if self._native_stream_opened:
            await self._close_empty_native_bubble("Silence-marker native stream close failed: %s")

        await self._delete_previews(self._stale_preview_ids(), label="Silence-marker")
        self._preview_message_ids = set()
        self._message_id = None
        self._accumulated = self._stream_ledger = self._last_sent_text = ""
        self._already_sent = False
        self._clear_turn_final_flags()
        logger.info("Suppressed streamed intentional-silence marker (chat=%s)", self.chat_id)

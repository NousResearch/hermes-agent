"""Telegram Bot API 10.0 guest mode: answering @mentions from chats the bot is not a member of.

Telegram delivers such a mention as ``update.guest_message`` with a one-shot ``guest_query_id``.
The bot can't ``sendMessage`` there, so the reply goes through ``answerGuestQuery``: a "thinking" stub
fires immediately, the streamed reply is buffered (or edited into the stub through its
``inline_message_id``), and the turn's final text is flushed into the stub when processing completes.
``TelegramGuestModeMixin`` holds that state machine; the adapter only calls into it from its send paths.
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import os
import random
import re
from typing import Any, Optional

from gateway.platforms.base import SendResult
from gateway.platforms.event import MessageType
from plugins.platforms.telegram.telegram_guest_media import TelegramGuestMediaMixin, guest_delivery_note

logger = logging.getLogger("plugins.platforms.telegram.adapter")


def _tg():
    """The adapter module, read at call time: it rebinds the Telegram classes on a lazy install."""
    from plugins.platforms.telegram import adapter
    return adapter


def _load_thinking_verbs() -> list:
    """The stub's progress verbs from the bundled ``thinking_verbs.py`` data file."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "thinking_verbs.py")
    namespace: dict = {}
    try:
        with open(path, encoding="utf-8-sig") as f:
            exec(f.read(), namespace)
    except Exception:
        # Deliberate catch-all: an operator-edited verb file must never break the adapter import.
        logger.warning("Could not load %s; using the default stub verb", path, exc_info=True)
    return namespace.get("THINKING_VERBS") or ["Thinking"]


_THINKING_VERBS = _load_thinking_verbs()


def guest_edit_intercept(edit_message):
    """Wrap ``edit_message``: the stream consumer's ``__no_edit__`` sentinel and guest inline edits are
    answered here, before any path that casts ``message_id`` to int or holds a chat send slot."""
    @functools.wraps(edit_message)
    async def wrapper(self, chat_id: str, message_id: str, content: str, *, finalize: bool = False,
                      metadata: Optional[dict[str, Any]] = None) -> SendResult:
        if self._bot and message_id == "__no_edit__":
            return SendResult(success=True, message_id=message_id)
        imi = self._guest_inline_message_ids.get(str(chat_id)) if self._bot else None
        if isinstance(imi, str) and message_id == imi:
            return await self._guest_inline_edit(chat_id, imi, content, finalize)
        return await edit_message(self, chat_id, message_id, content, finalize=finalize, metadata=metadata)
    return wrapper


class TelegramGuestModeMixin(TelegramGuestMediaMixin):
    """Guest-mode state and delivery for :class:`TelegramAdapter`."""

    # The guest reply is one answer, not every inter-tool segment: the stream consumer delivers only the
    # last segment and tags its first chunk ``guest_segment_start`` so the reply buffer is replaced.
    GUEST_MODE_DROPS_PRIOR_SEGMENTS: bool = True

    def _init_guest_state(self) -> None:
        self._init_guest_media_state()
        # Bot API 10.0 guest mode: chat_id → guest_query_id for answerGuestQuery
        self._pending_guest_queries: dict[str, str] = {}
        # chat IDs that are guest-mode only (bot not a member)
        self._guest_only_chats: set = set()
        # accumulated send() content for guest chats; flushed via editMessageText in on_processing_complete
        self._guest_reply_buffer: dict[str, str] = {}
        # inline_message_id returned by the stub answerGuestQuery; used for the follow-up editMessageText
        self._guest_inline_message_ids: dict[str, Optional[str]] = {}
        # Dedup set for guest_message update_ids: PTB resets its polling offset to 0 on restart,
        # so Telegram re-delivers unacknowledged updates.  We persist the last seen update_id
        # and skip updates whose ids we've already processed.
        self._seen_guest_update_ids: set = set()
        self._last_guest_update_id: int = 0

    def _telegram_guest_mode(self) -> bool:
        """Return whether non-allowlisted groups may trigger via direct @mention."""
        return self._extra_bool("guest_mode", "TELEGRAM_GUEST_MODE", "false")

    def _is_guest_chat(self, chat_id: Any) -> bool:
        """Return whether *chat_id* is currently a guest-mode (non-member) chat.

        True while a guest turn is in flight (``_pending_guest_queries``) or for
        the remainder of processing after the query has been consumed
        (``_guest_only_chats``). Shared by every send-path method that must
        suppress normal ``sendMessage``-family calls in guest chats — the bot
        isn't a member, so those calls fail with ``Forbidden``.
        """
        _cid_str = str(chat_id)
        return self._pending_guest_queries.get(_cid_str) is not None or _cid_str in self._guest_only_chats

    def buffers_stream_replies(self, chat_id: Any) -> bool:
        """True when stream sends to *chat_id* land in the guest reply buffer rather than the chat."""
        return self._is_guest_chat(chat_id)

    def exec_approval_unanswerable(self, source: Any) -> Optional[str]:
        """A guest chat can't show an approval prompt: the card's sendMessage is rejected, the guest
        send path keeps only the streamed reply, and /approve is declined there (any user, admin or not)."""
        if self._is_guest_chat(getattr(source, "chat_id", None)):
            return "this is a guest chat the bot isn't a member of, so approval can't be granted here. Tell the user you can't do that in this context."
        return None

    def _guest_drop_media_fragment(self, chat_id: Any, metadata: Optional[dict[str, Any]]) -> None:
        """Empty stream send in a guest chat: when the stream consumer strips the full MEDIA: tag the
        adapter receives empty content, but an intermediate chunk ("MEDIA" with no colon) may already be
        sitting in the guest reply buffer. Clear it so OPC doesn't edit the stub with the raw fragment."""
        _cid_str_early = str(chat_id)
        if not (bool(metadata and (metadata.get("expect_edits") or metadata.get("notify")))
                and self._is_guest_chat(_cid_str_early)):
            return
        _buf_early = self._guest_reply_buffer.get(_cid_str_early, "")
        if _buf_early:
            _buf_cleaned = re.sub(r"(?i)^MEDIA:?\s*\S*\s*", "", _buf_early).strip()
            if _buf_cleaned != _buf_early:
                self._guest_reply_buffer[_cid_str_early] = _buf_cleaned

    async def _guest_typing(self, chat_id: Any) -> None:
        """For guest-only chats fire the stub unconditionally on the first send_typing() call. No
        content classification: the stub always fires so the user sees immediate feedback, and OPC
        edits it with the final text reply."""
        _cid_str = str(chat_id)
        if _cid_str in self._guest_only_chats and self._guest_inline_message_ids.get(_cid_str) is False:
            await self._guest_fire_text_stub(_cid_str)

    async def _guest_inline_edit(self, chat_id: Any, imi: str, content: str, finalize: bool) -> SendResult:
        """Progressive stream edit of the guest stub through its ``inline_message_id``."""
        _text = content
        for _cur in (" ▉", "▉"):
            if _text.endswith(_cur):
                _text = _text[: -len(_cur)]
                break
        # Strip MEDIA: directives that the stream consumer passes through raw.
        if "MEDIA:" in _text:
            _text = re.sub(r"MEDIA:\S+", "", _text).strip()
            _text = re.sub(r"\n{3,}", "\n\n", _text)
        # Keep buffer current with the latest raw (pre-format) text so that
        # on_processing_complete can do a finalize edit if the stream consumer's
        # own finalize edit fails mid-stream (e.g. API error or truncated chunk).
        if _text.strip():
            self._guest_reply_buffer[str(chat_id)] = _text
        _text = (_tg()._strip_mdv2(self.format_message(_text)) if finalize else _text)[:4096]
        if not _text.strip():
            return SendResult(success=True, message_id=imi)
        try:
            await self._bot.edit_message_text(text=_text, inline_message_id=imi)
            return SendResult(success=True, message_id=imi)
        except Exception as _ie:  # health: allow BLE001 -- every edit failure becomes a SendResult for the stream consumer; logged redacted
            _ie_s = str(_ie).lower()
            if "not modified" in _ie_s:
                return SendResult(success=True, message_id=imi)
            logger.warning(
                "[%s] guest inline editMessageText failed (imi=%s): %s",
                self.name, imi, _tg()._redact_telegram_error_text(_ie),
            )
            return SendResult(
                success=False,
                error=_tg()._redact_telegram_error_text(_ie),
                retryable="retry after" in _ie_s or "flood" in _ie_s,
            )

    async def _guest_buffer_send(self, chat_id: Any, content: str, metadata: Optional[dict[str, Any]]) -> SendResult:
        """Bot API 10.0 guest reply: buffer content and return immediately (``sendMessage`` would be
        rejected with Forbidden, since the bot is not a member). Keys are str: ``chat_id`` may arrive as
        an int from the event source, which would silently miss every dict lookup."""
        _cid_str = str(chat_id)
        # Tool-use progress blocks (💻 terminal etc.) come through
        # send() from send_progress_messages().  The stream consumer
        # always sets expect_edits=True on the first frame and
        # notify=True on the fallback-final send; tool-progress calls
        # have neither flag.  Drop anything that isn't from the stream
        # consumer so the guest reply contains only the LLM response.
        _is_stream_send = bool(
            metadata
            and (metadata.get("expect_edits") or metadata.get("notify"))
        )
        if not _is_stream_send:
            # Tool-progress call from send_progress_messages().  Fire the
            # thinking stub on first contact so the user sees immediate
            # feedback — no content classification, the stub always fires.
            if self._guest_inline_message_ids.get(_cid_str) is False:
                await self._guest_fire_text_stub(_cid_str)
            return SendResult(success=True, message_id=None)

        # Streaming: fire the stub now if it hasn't fired yet (covers responses
        # where send_typing() was skipped and no tool-progress call ran).
        # False = slot open, stub not fired → fire now.
        # None  = stub fired but Telegram returned no imi → buffer fallback.
        # str   = real imi → live streaming edits.
        if self._guest_inline_message_ids.get(_cid_str) is False:
            await self._guest_fire_text_stub(_cid_str)

        # Stub fired (imi available or not) — buffer mode: accumulate
        # content so OPC delivers the full response at once.
        _cursor = " ▉"
        _clean = content
        if _clean.endswith(_cursor):
            _clean = _clean[:-len(_cursor)]
        elif _clean.endswith("▉"):
            _clean = _clean[:-1]
        # Streaming sends cumulative chunks; MEDIA: tags are stripped by the
        # stream consumer only once the full path (including extension) appears.
        # Intermediate chunks like "MEDIA:" or "MEDIA:/workspace/file" don't
        # match the extension-anchored cleanup regex and land in the buffer.
        # Strip any MEDIA: residuals here so they never appear as text.
        _raw_clean = _clean  # pre-strip value for cumulative-replace detection below
        if "MEDIA:" in _clean:
            _clean = re.sub(r"MEDIA:\s*\S+", "", _clean).strip()

        _existing = self._guest_reply_buffer.get(_cid_str, "")
        if metadata and metadata.get("guest_segment_start"):
            # Stream consumer had a tool-call segment break on a __no_edit__
            # platform: inter-tool commentary was cleared in the consumer and
            # this is the start of the final-answer delivery.  Replace the
            # buffer so preamble text ("searching...", failed-tool narration)
            # from earlier segments does not appear in the answerGuestQuery.
            self._guest_reply_buffer[_cid_str] = _clean
        elif _existing and _raw_clean.startswith(_existing):
            # Cumulative streaming update: the raw (pre-strip) frame
            # contains all prior content as a prefix → replace so the
            # last call wins.  Comparing against _raw_clean (not the
            # post-strip _clean) handles the case where a partial "MEDIA"
            # chunk landed in the buffer first, and the next cumulative
            # frame "MEDIA: /path.ext" strips to "" — the raw frame still
            # starts with "MEDIA" so we correctly replace (clearing the
            # partial token) rather than appending "" to "MEDIA".
            self._guest_reply_buffer[_cid_str] = _clean
        else:
            # Continuation or overflow chunk: content does NOT start
            # with what we already have → append.
            self._guest_reply_buffer[_cid_str] = _existing + _clean
        return SendResult(success=True, message_id=None)

    async def _guest_finish_turn(self, event: Any) -> None:
        """Guest mode OPC (Bot API 10.0): edit the stub with the final text reply."""
        _gc_id = str(getattr(event.source, "chat_id", None) or "")
        if not _gc_id:
            return
        _guest_qid = self._pending_guest_queries.pop(_gc_id, None)
        # Sentinel semantics for _guest_inline_message_ids:
        #   False  → stub never fired (shouldn't happen — send_typing always fires it)
        #   None   → stub fired but Telegram returned no inline_message_id
        #   str    → stub fired, real imi for editMessageText
        _guest_imi_raw = self._guest_inline_message_ids.pop(_gc_id, False)
        _guest_imi = _guest_imi_raw if isinstance(_guest_imi_raw, str) else None
        _buffered = self._guest_reply_buffer.pop(_gc_id, "")
        _turn_media = self._guest_turn_media.pop(_gc_id, None)
        self._guest_only_chats.discard(_gc_id)
        if not ((_guest_qid or _guest_imi) and self._bot):
            return
        _plain = _tg()._strip_mdv2(self.format_message(_buffered)).strip() if _buffered else ""
        # Strip any leading MEDIA artifact that escaped stream-consumer cleanup.
        _plain = re.sub(r"(?i)^MEDIA:?\s*\S*\s*", "", _plain).strip()
        # UTF-16 code units, not Python string length — Telegram's 4,096
        # limit is measured in UTF-16 units, so a naive _plain[:4096]
        # slice can pass this check while still exceeding the real cap
        # (e.g. 2,049 emoji is 2,049 chars but 4,098 UTF-16 units) and
        # fail the editMessageText call below, after guest state has
        # already been torn down above — silently dropping the turn.
        _reply_text = self._truncate_stream_overflow_preview(_plain) or \
            "⚠️ Sorry, something went wrong. Please try again."
        logger.warning("[%s] guest OPC flush (chat=%s buffered_len=%d imi=%s)",
                       self.name, _gc_id, len(_buffered), _guest_imi)
        try:
            if _turn_media and _guest_imi:
                await self._guest_edit_ready_button(_guest_imi, _plain, _turn_media, _gc_id)
            elif _guest_imi:
                await self._guest_typewriter(_guest_imi, _reply_text)
                await self._bot.edit_message_text(text=_reply_text, inline_message_id=_guest_imi)
                logger.warning("[%s] guest OPC text edit (chat=%s imi=%s)", self.name, _gc_id, _guest_imi)
            elif _guest_qid:
                # No imi (the stub call itself raised) — fall back to a fresh answerGuestQuery.
                _fallback = "⛔ Not authorized." if not _buffered else _reply_text
                await self._answer_guest_query(
                    _guest_qid,
                    _tg().InlineQueryResultArticle(
                        id="reply", title="Reply",
                        input_message_content=_tg().InputTextMessageContent(_fallback),
                    ),
                    log_label="OPC fallback reply",
                )
                logger.warning("[%s] guest OPC fallback reply (chat=%s no_imi)", self.name, _gc_id)
        except Exception as _flush_err:  # health: allow BLE001 -- end of the guest turn: nothing left to fall back to; logged redacted
            logger.error("[%s] guest OPC flush failed (chat=%s): %s",
                         self.name, _gc_id, _tg()._redact_telegram_error_text(_flush_err))

    async def _guest_typewriter(self, imi: str, reply_text: str) -> None:
        """Typewriter frames before the final edit. Animates over the already-truncated reply text:
        a longer raw text risks oversized mid-animation frames and a visible "shrink" on the final edit."""
        _tw_min, _tw_frames, _tw_delay = 80, 8, 0.4
        if len(reply_text) < _tw_min:
            return
        _tw_chunk = max(40, len(reply_text) // _tw_frames)
        for _tw_pos in range(_tw_chunk, len(reply_text), _tw_chunk):
            _tw_frame = reply_text[:_tw_pos]
            _tw_break = max(_tw_frame.rfind('\n'), _tw_frame.rfind(' '))
            if _tw_break > 0:
                _tw_frame = _tw_frame[:_tw_break]
            try:
                await self._bot.edit_message_text(text=_tw_frame, inline_message_id=imi)
            except Exception as exc:  # health: allow BLE001 -- a lost animation frame is cosmetic; the final edit follows
                logger.debug("[%s] guest typewriter frame failed: %s", self.name, _tg()._redact_telegram_error_text(exc))
            await asyncio.sleep(_tw_delay)

    async def _guest_fire_text_stub(self, chat_id: str) -> None:
        """Fire the thinking-verb stub, consuming the answerGuestQuery slot as text.

        Stores the returned inline_message_id (or None on API failure) in
        _guest_inline_message_ids so send() can drive progressive stream edits and
        on_processing_complete can update the message with the real reply.
        Should only be called when _guest_inline_message_ids[chat_id] is False
        (slot open, stub not yet fired).
        """
        _chat_id_str = str(chat_id)
        _guest_qid = self._pending_guest_queries.get(_chat_id_str)
        if not _guest_qid or not self._bot:
            return
        if self._guest_inline_message_ids.get(_chat_id_str) is not False:
            return  # already fired or not a guest chat
        # Claim the slot immediately (before the await) so a concurrent caller that
        # also passed the `is not False` check above doesn't fire a second stub.
        self._guest_inline_message_ids[_chat_id_str] = None
        _verb = random.choice(_THINKING_VERBS)
        _stub = _tg().InlineQueryResultArticle(
            id="thinking", title=f"{_verb}...",
            input_message_content=_tg().InputTextMessageContent(f"⏳ {_verb}..."),
        )
        # Wrap the API call in a Task so asyncio.shield() can protect it from
        # _keep_typing's asyncio.wait_for timeout.  When the 1.5 s timeout fires,
        # _keep_typing cancels send_typing → CancelledError reaches shield → shield
        # re-raises to us but the inner Task keeps running.  The done_callback
        # captures the inline_message_id even when the outer await was timed out.
        _fire_task: "asyncio.Task" = asyncio.ensure_future(
            self._bot.answer_guest_query(_guest_qid, _stub)
        )

        def _on_stub_done(fut: "asyncio.Future") -> None:
            try:
                # SentGuestMessage.inline_message_id is a required field — if the
                # call succeeded at all, a real id is guaranteed present.
                self._guest_inline_message_ids[_chat_id_str] = fut.result().inline_message_id
            except Exception as _e:  # health: allow BLE001 -- any stub failure falls back to the OPC reply; logged redacted (a traceback can carry the bot token)
                self._guest_inline_message_ids[_chat_id_str] = None
                logger.warning(
                    "[%s] guest stub failed (chat=%s): %s — OPC fallback will run",
                    self.name, _chat_id_str, _tg()._redact_telegram_error_text(_e),
                )

        _fire_task.add_done_callback(_on_stub_done)
        try:
            await asyncio.shield(_fire_task)
            # Happy path: shield returned normally, callback already fired or will fire.
        except asyncio.CancelledError:
            # _keep_typing's asyncio.wait_for timed out (or a genuine task cancel).
            # _fire_task continues protected by shield; _on_stub_done will set imi.
            raise
        except Exception as _stub_err:  # health: allow BLE001 -- _on_stub_done already recorded the failure; logged redacted
            # The task raised an exception (non-timeout failure).
            # _on_stub_done already set imi to None; just log for the outer context.
            logger.warning(
                "[%s] guest stub task error (chat=%s): %s",
                self.name, _chat_id_str, _tg()._redact_telegram_error_text(_stub_err),
            )

    def _persist_guest_update_id(self, update_id: int) -> None:
        """Save the latest processed guest update_id so restarts don't reprocess it."""
        from hermes_constants import get_hermes_home
        path = get_hermes_home() / "telegram_guest_update_id.json"
        payload = {"last_update_id": update_id, "seen_ids": sorted(self._seen_guest_update_ids)[-200:]}
        try:
            path.write_text(json.dumps(payload), encoding="utf-8")
        except OSError as exc:
            logger.debug("[%s] Could not persist guest update_id: %s", self.name, exc)

    def _load_guest_update_ids(self) -> None:
        """Restore the seen-update-id set from disk on startup."""
        from hermes_constants import get_hermes_home
        path = get_hermes_home() / "telegram_guest_update_id.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else {}
        except (OSError, ValueError) as exc:
            logger.debug("[%s] Could not load guest update_ids: %s", self.name, exc)
            return
        ids = data.get("seen_ids") or []
        if ids or data:
            self._seen_guest_update_ids = set(ids)
            self._last_guest_update_id = data.get("last_update_id", 0)
            logger.info("[%s] Loaded %d seen guest update_ids (last=%s)", self.name, len(ids), self._last_guest_update_id)

    async def _answer_guest_query(self, guest_query_id: str, result, *, log_label: str) -> bool:
        """Answer a guest query with *result* (an ``InlineQueryResult``); True on success.

        Never raises — a guest chat has no fallback delivery path, so a failure here
        is a dead end for the turn either way; logging and swallowing is all there is
        to do."""
        try:
            await self._bot.answer_guest_query(guest_query_id, result)
            return True
        except Exception as exc:  # health: allow BLE001 -- documented never-raise boundary; logged redacted (a traceback can carry the bot token)
            logger.warning("[%s] guest %s failed: %s", self.name, log_label, _tg()._redact_telegram_error_text(exc))
            return False

    async def _handle_guest_message_update(self, update: Any, context: Any) -> None:
        """Handle guest_message updates (Bot API 10.0 guest bot feature).

        Telegram delivers @mentions from chats the bot hasn't joined as a typed
        ``Message`` on ``update.guest_message``, carrying its own ``guest_query_id``
        (:attr:`telegram.Message.guest_query_id`). We store that id so ``send()``
        can answer via :meth:`telegram.Bot.answer_guest_query`, then route the
        message through the normal text-processing pipeline.
        """
        if not self._telegram_guest_mode():
            return
        msg = update.guest_message
        if msg is None:
            return
        guest_query_id = msg.guest_query_id
        _uid = update.update_id or 0

        # Dedup: PTB resets its polling offset to 0 on restart, causing Telegram to redeliver
        # unacknowledged updates.  Skip any update_id we've already processed.
        if _uid and _uid in self._seen_guest_update_ids:
            return
        if _uid:
            self._seen_guest_update_ids.add(_uid)
            if _uid > self._last_guest_update_id:
                self._last_guest_update_id = _uid
                self._persist_guest_update_id(_uid)
            # Trim set to last 200 entries — matches the persisted on-disk cap
            # in _persist_guest_update_id, so in-memory and reloaded-after-restart
            # dedup coverage are consistent instead of silently differing (500 vs 200).
            if len(self._seen_guest_update_ids) > 200:
                _min = min(self._seen_guest_update_ids)
                self._seen_guest_update_ids.discard(_min)

        if not guest_query_id:
            logger.warning("[%s] guest_message missing guest_query_id, skipping", self.name)
            return

        text = msg.text or getattr(msg, "caption", None) or ""
        if not text.strip():
            return

        chat_id_str = str(msg.chat.id) if msg.chat else ""
        if not chat_id_str:
            return

        # Caller authorization (fail-closed). Guest mode lets ANY chat that
        # @mentions the bot reach it — _should_process_message only gates the
        # chat/mention, never the person — so without this an unauthorized
        # stranger who knows the @handle could drive the full LLM + tools from
        # any group (cost, abuse, prompt-injection surface). Approval gating
        # only stops dangerous *commands*; it does nothing about this door.
        #
        # The caller is msg.from_user, same as any other Telegram message.
        # Message.guest_bot_caller_user is a DIFFERENT, unrelated field —
        # documented for messages the bot itself sends as a guest, not for
        # the incoming @mention. Routed through the same
        # _is_callback_user_authorized the exec-approval buttons use, so
        # there is ONE definition of "who's allowed" — the env allowlists
        # (TELEGRAM_ALLOWED_USERS / group variants / GATEWAY_ALLOWED_USERS)
        # unioned with the pairing store, with `*` as the explicit
        # open-to-everyone opt-out. Empty allowlist or unknown caller =>
        # deny. Placed before the busy-reply, guest-state registration AND
        # the deliver_<token> branch so token redemption is gated too.
        _guest_caller_id = str(getattr(msg.from_user, "id", "") or "").strip()
        _guest_chat_type = getattr(msg.chat, "type", None)
        if not self._is_callback_user_authorized(
            _guest_caller_id,
            chat_id=chat_id_str,
            chat_type=str(_guest_chat_type) if _guest_chat_type is not None else "group",
            user_name=(getattr(msg.from_user, "username", None) or getattr(msg.from_user, "first_name", None)),
        ):
            # Loud + diagnostic: if from_user is ever absent on a guest message
            # (e.g. an anonymous-admin edge case), logging that makes a silent
            # deny-all instantly traceable rather than looking like a backend outage.
            logger.warning(
                "[%s] Guest caller not authorized (caller_id=%r chat=%s); denying. has_from_user=%s",
                self.name, _guest_caller_id, chat_id_str, msg.from_user is not None,
            )
            return

        # deliver_<token> (the stub's "tap to send" button): answered at once with the staged media,
        # no LLM pass, and independent of any turn in flight for this chat. Sits behind the caller
        # gate above, so a leaked token can't be redeemed by an unauthorized caller.
        _query_text = self._clean_bot_trigger_text(text.strip()).strip()
        if _query_text.startswith("deliver_"):
            await self._guest_answer_deliver_query(guest_query_id, _query_text[len("deliver_"):], chat_id_str)
            return

        # Guest state (_pending_guest_queries, _guest_reply_buffer,
        # _guest_inline_message_ids) is keyed by chat_id, not guest_query_id —
        # a second @mention from the same chat while a turn is still in flight
        # would otherwise overwrite the first turn's query id and reset its
        # stub sentinel, orphaning the first stub and letting the two replies'
        # buffered text cross-contaminate. Reject the new query outright with
        # its own immediate answer instead of touching in-flight state.
        if chat_id_str in self._pending_guest_queries:
            await self._answer_guest_query(
                guest_query_id,
                _tg().InlineQueryResultArticle(
                    id="busy", title="Still working on the previous request",
                    input_message_content=_tg().InputTextMessageContent(
                        "⏳ Still working on a previous request in this chat — please wait for that reply, then ask again."),
                ),
                log_label="busy-reply",
            )
            return

        # Register state, fire stub, route to skill layer.
        self._pending_guest_queries[chat_id_str] = guest_query_id
        self._guest_only_chats.add(chat_id_str)

        if not self._should_process_message(msg):
            self._pending_guest_queries.pop(chat_id_str, None)
            self._guest_only_chats.discard(chat_id_str)
            return

        # Sentinel: False = slot open, stub not fired yet.
        # None = stub fired, Telegram returned no imi.  str = real imi.
        self._guest_inline_message_ids[chat_id_str] = False

        event = self._build_message_event(msg, MessageType.TEXT, update_id=update.update_id)
        event.text = self._clean_bot_trigger_text(event.text)

        # Session isolation per guest caller falls out of _build_message_event for
        # free now: it reads user_id/user_name straight off msg.from_user, which is
        # exactly _guest_caller_id's source above. build_session_key groups group
        # participants by user_id, so this alone gives guest turns the same
        # per-caller session keying ordinary group messages already get —
        # per-caller when group_sessions_per_user is on (the default), shared only
        # if the operator deliberately turned that off.

        # Inject delivery constraint so the LLM knows direct Bot API calls to this
        # chat will fail (bot is not a member) and media goes through MEDIA: staging.
        _guest_delivery_note = guest_delivery_note()
        if event.channel_prompt:
            event.channel_prompt = event.channel_prompt + "\n\n" + _guest_delivery_note
        else:
            event.channel_prompt = _guest_delivery_note

        # Slash commands aren't routed in guest context: command handlers fire
        # their own answerGuestQuery which creates a second orphaned stub, leaving
        # an unupdated "⏳" message and a separate "⚠️ Sorry..." reply.
        # Block them early and reply directly so the user knows why.
        if event.text.lstrip().startswith("/"):
            self._pending_guest_queries.pop(chat_id_str, None)
            self._guest_inline_message_ids.pop(chat_id_str, None)
            self._guest_only_chats.discard(chat_id_str)
            await self._answer_guest_query(
                guest_query_id,
                _tg().InlineQueryResultArticle(
                    id="reply", title="Commands not supported",
                    input_message_content=_tg().InputTextMessageContent(
                        "📋 Slash commands aren't supported in this context — just ask me a question!"),
                ),
                log_label="slash-block reply",
            )
            return

        event = self._apply_telegram_group_observe_attribution(event)
        self._enqueue_text_event(event)

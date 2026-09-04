"""Telegram delivery methods; runtime dependencies remain on the adapter facade."""

from typing import Any, Dict, Optional
from gateway.platforms.base import SendResult
try:
    from telegram import Message, Update
    from telegram.ext import ContextTypes
except ImportError:
    Message = Update = Any
    class ContextTypes:
        DEFAULT_TYPE = Any


class TelegramDeliveryMixin:
    @classmethod
    def _metadata_thread_id(cls, metadata: Optional[Dict[str, Any]]) -> Optional[str]:
        from . import adapter as _adapter

        thread_id = (metadata or {}).get("thread_id") or (metadata or {}).get("message_thread_id")
        return str(thread_id) if thread_id is not None else None

    @classmethod
    def _metadata_direct_messages_topic_id(cls, metadata: Optional[Dict[str, Any]]) -> Optional[str]:
        from . import adapter as _adapter

        topic_id = (metadata or {}).get("direct_messages_topic_id") or (metadata or {}).get("telegram_direct_messages_topic_id")
        return str(topic_id) if topic_id is not None else None

    @classmethod
    def _metadata_reply_to_message_id(cls, metadata: Optional[Dict[str, Any]]) -> Optional[int]:
        from . import adapter as _adapter

        reply_to = (metadata or {}).get("telegram_reply_to_message_id")
        return int(reply_to) if reply_to is not None else None

    @staticmethod
    def _dm_topic_fallback(metadata: Optional[Dict[str, Any]]) -> bool:
        """True for Hermes private-chat topic lanes (``telegram_dm_topic_reply_fallback``)."""
        from . import adapter as _adapter

        return bool(metadata and metadata.get("telegram_dm_topic_reply_fallback"))

    @classmethod
    def _is_private_dm_topic_send(cls, chat_id: str, thread_id: Optional[str], metadata: Optional[Dict[str, Any]]) -> bool:
        from . import adapter as _adapter

        if cls._metadata_direct_messages_topic_id(metadata) is not None:
            return cls._dm_topic_fallback(metadata) and cls._metadata_reply_to_message_id(metadata) is not None
        if metadata and metadata.get("telegram_dm_topic_created_for_send"):
            return False
        return bool(thread_id) and cls._dm_topic_fallback(metadata)

    @staticmethod
    def _dm_topic_missing_anchor_error() -> str:
        return "Telegram DM topic delivery requires a reply anchor; refusing to send outside the requested topic"

    @classmethod
    def _reply_to_message_id_for_send(
        cls, reply_to: Optional[str], metadata: Optional[Dict[str, Any]] = None, reply_to_mode: Optional[str] = None) -> Optional[int]:
        from . import adapter as _adapter

        if reply_to:
            return int(reply_to)
        if cls._dm_topic_fallback(metadata) and reply_to_mode != "off":
            return cls._metadata_reply_to_message_id(metadata)
        return None

    @classmethod
    def _thread_kwargs_for_send(
        cls, chat_id: str, thread_id: Optional[str], metadata: Optional[Dict[str, Any]] = None,
        reply_to_message_id: Optional[int] = None, reply_to_mode: Optional[str] = None) -> Dict[str, Any]:
        """Telegram send kwargs for forum and direct-message topic routing.

        Forum topics use ``message_thread_id``; native Bot API DM topics opt in via explicit ``direct_messages_topic_id``
        metadata; Hermes private-chat topic lanes are marked ``telegram_dm_topic_reply_fallback``. Anchor-less synthetic sends
        prefer the Hermes topic's ``message_thread_id`` (the native DM-topic id renders in a different chat lane).
        ``reply_to_mode="off"`` suppresses the anchor but keeps ``message_thread_id``.

        Live replies send the private topic thread id together with a reply anchor. Synthetic/resumed sends
        without an anchor (loop wakeups, background-process notifications, queued follow-ups after a gateway
        restart) prefer the Hermes topic's ``message_thread_id`` so they stay in the active topic lane
        (#87051); ``direct_messages_topic_id`` is only used when no topic thread resolves, since the native
        DM-topic id does not match the Hermes topic lane and can render the message in a different chat
        lane.
        """
        fallback = cls._dm_topic_fallback(metadata)
        if fallback and reply_to_mode != "off":
            if reply_to_message_id is None:
                reply_to_message_id = cls._metadata_reply_to_message_id(metadata)
            if reply_to_message_id is None:
                # Anchor-less synthetic send: prefer the Hermes topic thread id (see docstring).
                # Anchor-less synthetic sends (loop wakeups, watch notifications, restart-resumed
                # follow-ups) must stay in the active topic lane: prefer the Hermes topic thread id when it
                # resolves (#87051). Routing via direct_messages_topic_id here sent these to a different
                # lane than the topic the session runs in.
                thread_message_id = cls._message_thread_id_for_send(thread_id)
                if thread_message_id is not None:
                    return {"message_thread_id": thread_message_id}
                return cls._direct_topic_kwargs(metadata) or {}
        elif not fallback:
            direct_kwargs = cls._direct_topic_kwargs(metadata)
            if direct_kwargs is not None:
                return direct_kwargs
        return {"message_thread_id": cls._message_thread_id_for_send(thread_id)}

    @classmethod
    def _direct_topic_kwargs(cls, metadata: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Native Bot API DM-topic routing kwargs, or None when no ``direct_messages_topic_id``."""
        from . import adapter as _adapter

        direct_topic_id = cls._metadata_direct_messages_topic_id(metadata)
        if direct_topic_id is None:
            return None
        return {"message_thread_id": None, "direct_messages_topic_id": int(direct_topic_id)}

    def _thread_kwargs_for_draft(self, chat_id: str, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Routing kwargs for ``sendMessageDraft`` / ``sendRichMessageDraft`` (integer
        ``message_thread_id`` for DM topics — Telegram rejects the raw string ``thread_id``)."""
        kwargs = self._thread_kwargs_for_send(
            chat_id, self._metadata_thread_id(metadata), metadata, reply_to_message_id=self._reply_to_message_id_for_send(None, metadata),
            reply_to_mode=getattr(self, "_reply_to_mode", None))
        return {k: v for k, v in kwargs.items() if v is not None}

    @classmethod
    def _message_thread_id_for_send(cls, thread_id: Optional[str]) -> Optional[int]:
        from . import adapter as _adapter

        if not thread_id or str(thread_id) == cls._GENERAL_TOPIC_THREAD_ID:
            return None
        return int(thread_id)

    @classmethod
    def _message_thread_id_for_typing(cls, thread_id: Optional[str]) -> Optional[int]:
        # Deliberately asymmetric with _message_thread_id_for_send: sendMessage rejects message_thread_id=1
        # (forum General), but sendChatAction NEEDS it to place the typing bubble in General.
        from . import adapter as _adapter

        return int(thread_id) if thread_id else None

    @staticmethod
    def _is_thread_not_found_error(error: Exception) -> bool:
        from . import adapter as _adapter

        return "thread not found" in str(error).lower()

    def _prune_stale_dm_topic_binding(self, chat_id: Any, thread_id: Any, *, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Drop the stale ``telegram_dm_topic_bindings`` row for a topic Telegram confirmed deleted, else
        ``_recover_telegram_topic_thread_id`` keeps steering inbound to the dead thread. Best-effort.
        Rows are namespaced by profile: the send's ``hermes_profile`` wins over the adapter's stamp.

        Without this prune the recovery logic in ``gateway.run._recover_telegram_topic_thread_id`` keeps
        steering future inbound messages to the dead thread (the bug behind #31501 — tool progress,
        approvals, replies all end up in the wrong place even though the user has moved on to a fresh
        topic). Best-effort: we never raise from a send-fallback path — a failed cleanup must not turn into
        a failed user-facing send.
        Under ``gateway.profile_routes`` the transport adapter may not be the profile that wrote the
        binding, so the send's ``hermes_profile`` metadata wins over the adapter's own profile stamp;
        single-profile bots fall back to ``"default"``. See #76423.
        """
        from . import adapter as _adapter

        if chat_id is None or thread_id is None:
            return
        db = getattr(getattr(self, "_session_store", None), "_db", None)
        if db is None or not hasattr(db, "delete_telegram_topic_binding"):
            return
        try:
            profile_name = (metadata or {}).get("hermes_profile") or getattr(self, "_hermes_profile_name", None) or "default"
            removed = db.delete_telegram_topic_binding(chat_id=str(chat_id), thread_id=str(thread_id), profile_name=profile_name)
        except Exception:
            _adapter.logger.debug(
                "[%s] delete_telegram_topic_binding failed for chat=%s thread=%s — skipping prune",
                self.name, chat_id, thread_id, exc_info=True)
            return
        if removed:
            _adapter.logger.info(
                "[%s] Pruned stale Telegram DM topic binding chat=%s thread=%s (Bot API: thread not found)", self.name, chat_id, thread_id)

    @staticmethod
    def _is_bad_request_error(error: Exception) -> bool:
        from . import adapter as _adapter

        name = error.__class__.__name__.lower()
        if name == "badrequest" or name.endswith("badrequest"):
            return True
        try:
            from telegram.error import BadRequest
            return isinstance(error, BadRequest)
        except ImportError:
            return False

    @classmethod
    def _should_retry_without_dm_topic_reply_anchor(
        cls, error: Exception, metadata: Optional[Dict[str, Any]], reply_to_message_id: Optional[int]) -> bool:
        """True when a DM-topic send should be retried with routing stripped: (1) stale anchor — reply
        target deleted; (2) anchor-less synthetic send whose ``direct_messages_topic_id`` Bot API rejects.

        2. The synthetic-event case (added when #27937 introduced ``direct_messages_topic_id`` fallback for
        sends without an anchor): if Bot API rejects the topic id itself with any BadRequest that mentions
        topic/thread routing, we retry without routing rather than dropping the message.
        """
        from . import adapter as _adapter

        if not cls._dm_topic_fallback(metadata) or not cls._is_bad_request_error(error):
            return False
        err_lower = str(error).lower()
        if reply_to_message_id is not None and "message to be replied not found" in err_lower:
            return True
        if not metadata.get("direct_messages_topic_id"):  # topic id rejected → plain DM send
            return False
        topic_markers = (
            "direct_messages_topic", "message thread not found", "thread not found", "topic_closed", "topic_deleted", "topic not found")
        return any(marker in err_lower for marker in topic_markers)

    async def _send_with_dm_topic_reply_anchor_retry(
        self, send_fn: Any, send_kwargs: Dict[str, Any], metadata: Optional[Dict[str, Any]],
        reply_to_message_id: Optional[int], media_label: str, reset_media: Optional[Any] = None) -> Any:
        """Retry stale private-topic media replies once without the topic anchor."""
        from . import adapter as _adapter

        try:
            return await send_fn(**send_kwargs)
        except Exception as send_err:
            if not self._should_retry_without_dm_topic_reply_anchor(send_err, metadata, reply_to_message_id):
                raise
            _adapter.logger.warning(
                "[%s] Reply target deleted for Telegram %s, retrying without reply/topic anchor: %s",
                self.name, media_label, _adapter._redact_telegram_error_text(send_err))
            if reset_media is not None:
                reset_media()
            retry_kwargs = dict(send_kwargs)
            retry_kwargs["reply_to_message_id"] = None
            retry_kwargs.pop("message_thread_id", None)
            retry_kwargs.pop("direct_messages_topic_id", None)
            return await send_fn(**retry_kwargs)

    def _link_preview_kwargs(self) -> Dict[str, Any]:
        from . import adapter as _adapter

        if not getattr(self, "_disable_link_previews", False):
            return {}
        if _adapter.LinkPreviewOptions is not None:
            return {"link_preview_options": _adapter.LinkPreviewOptions(is_disabled=True)}
        return {"disable_web_page_preview": True}

    # --- Bot API 10.1 Rich Messages (sendRichMessage): final/new-message replies opportunistically send
    # RAW agent markdown so tables, task lists, <details>, math render natively; legacy MarkdownV2 send()
    # is the fallback. Streaming edits stay on the MarkdownV2 edit path.
    def _content_fits_rich_limits(self, content: str) -> bool:
        """Pre-check the 32,768-char cap only; other rich limits surface as BadRequest (permanent)."""
        return len(content) <= self.RICH_MESSAGE_MAX_CHARS

    def _bot_supports_rich(self) -> bool:
        """True when ``do_api_request`` is an *async* callable (real Bot or AsyncMock); plain MagicMock
        and SimpleNamespace bots resolve False → legacy path."""
        from . import adapter as _adapter

        return _adapter.inspect.iscoroutinefunction(getattr(self._bot, "do_api_request", None))

    def _has_telegram_desktop_details_math_crash_shape(self, content: str) -> bool:
        """Math inside <details> crashes Telegram Desktop 6.9.1 (tdesktop#30808); the Bot API accepts
        the payload, so rich delivery must be skipped up front."""
        if not content:
            return False
        return any(self._RICH_MATH_IN_DETAILS_RE.search(block) for block in self._RICH_DETAILS_RE.findall(content))

    def _has_telegram_desktop_cjk_rich_garble_shape(self, content: str) -> bool:
        """True for CJK content: Telegram Mac/Desktop rich rendering leaves overlapping glyphs.

        Telegram Mac/Desktop Bot API 10.1 rich-message rendering currently leaves overlapping draft/overlay
        glyph artifacts for CJK text (#47653). The legacy MarkdownV2 path renders the same text cleanly, so
        skip rich delivery up front until affected clients age out.
        """
        from . import adapter as _adapter

        return bool(content and self._RICH_CJK_RE.search(content))

    def _needs_rich_rendering(self, content: str) -> bool:
        """True for constructs MarkdownV2 degrades: pipe tables, task lists, <details>, block math.
        Ordinary replies stay on MarkdownV2 so clients render consistent font weight/spacing.

        The rich endpoint is reserved for constructs where raw markdown materially improves output: pipe
        tables (MarkdownV2 has no table syntax and rewrites them into bullet lists), GFM task lists,
        collapsible ``<details>`` blocks, and block math. Adapted from #45995 (@YonganZhang).
        """
        from . import adapter as _adapter

        if not content:
            return False
        if any(_adapter._TABLE_SEPARATOR_RE.match(line) for line in content.splitlines()):
            return True
        if _adapter.re.search(r"(?m)^\s*[-*]\s+\[[ xX]\]\s+", content):
            return True
        if _adapter.re.search(r"(?m)^<details\b|^</details>|^<summary\b|^</summary>", content):
            return True
        return "$$" in content

    def _rich_delivery_enabled(self) -> bool:
        """Whether rich delivery is allowed (``rich_messages`` opt-in)."""
        from . import adapter as _adapter

        return bool(getattr(self, "_rich_messages_enabled", True))

    def _rich_content_ok(self, content: str) -> bool:
        """Shape checks shared by rich sends and rich drafts (non-blank, no Desktop crash/garble
        shapes, under the cap, async-capable bot)."""
        from . import adapter as _adapter

        return bool(
            content and content.strip()
            and not self._has_telegram_desktop_details_math_crash_shape(content)
            and not self._has_telegram_desktop_cjk_rich_garble_shape(content)
            and self._content_fits_rich_limits(content)
            and self._bot_supports_rich())

    def _rich_eligible(self, content: str) -> bool:
        """Rich eligibility ignoring ``expect_edits`` (a streamed preview's FINAL edit still upgrades)."""
        from . import adapter as _adapter

        return bool(
            self._rich_delivery_enabled()
            and not getattr(self, "_rich_send_disabled", False)
            and content and content.strip()
            and self._needs_rich_rendering(content)
            and self._rich_content_ok(content))

    def _should_attempt_rich(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        from . import adapter as _adapter

        return bool(not (metadata or {}).get("expect_edits") and self._rich_eligible(content))

    def prefers_fresh_final_streaming(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Replace a streamed preview with a fresh rich final — DM topics only. Root DMs stay off (a live
        draft has no preview id); DM *topics* degrade to edit-in-place whose MarkdownV2 preview Telegram
        refuses to rich-edit, so a fresh sendRichMessage + delete is the only way to keep native tables.

        Root DMs keep this off (#46206 / #47048): successful draft streaming has no preview ``message_id``,
        so the hook is not consulted, and in-place ``editMessageText.rich_message`` would duplicate a live
        draft turn. Private DM *topics* often reject ``sendMessageDraft``; the consumer then degrades to
        edit-in-place. Telegram rejects a rich edit of that plain MarkdownV2 preview, and the fallback
        formatter permanently turns pipe tables into bullet lists.
        """
        metadata = metadata or {}
        if not (metadata.get("telegram_dm_topic_reply_fallback") or self._metadata_direct_messages_topic_id(metadata)):
            return False
        return self._rich_eligible(content)

    def _rich_transport_available(self) -> bool:
        from . import adapter as _adapter

        return bool(
            getattr(self, "_rich_messages_enabled", True) and not getattr(self, "_rich_send_disabled", False) and self._bot_supports_rich())

    def streaming_overflow_limit(self) -> Optional[int]:
        """Let the stream consumer accumulate up to the rich cap so a reply that fits one sendRichMessage
        isn't fragmented at 4,096. None (→ legacy limit) if rich is unavailable."""
        return self.RICH_MESSAGE_MAX_CHARS if self._rich_transport_available() else None

    def _rich_message_payload(self, content: str, *, skip_entity_detection: bool = False) -> Dict[str, Any]:
        """``InputRichMessage`` from RAW markdown — never ``format_message(content)``, whose MarkdownV2
        escaping destroys table pipes."""
        from . import adapter as _adapter

        payload: _adapter.Dict[str, _adapter.Any] = {"markdown": _adapter._rich_normalize_linebreaks(content)}
        if skip_entity_detection:
            payload["skip_entity_detection"] = True
        return payload

    def _is_rich_capability_error(self, exc: Exception) -> bool:
        """True ⇒ the rich endpoint itself is unavailable (old PTB/server); latches rich off.
        Per-message BadRequests (parser/limit) are NOT capability errors."""
        from . import adapter as _adapter

        if exc.__class__.__name__.lower() in {"endpointnotfound", "invalidtoken"}:
            return True
        if isinstance(exc, (AttributeError, TypeError, NotImplementedError)) or getattr(exc, "error_code", None) == 404:
            return True
        s = str(exc).lower()
        if ("method" in s or "endpoint" in s) and ("not found" in s or "does not exist" in s):
            return True
        return "no such method" in s

    def _is_rich_fallback_error(self, exc: Exception) -> bool:
        """True ⇒ permanent/capability error ⇒ safe to fall back to legacy. Conservative: anything not
        clearly permanent is transient — the rich request may have reached Telegram (duplicate risk)."""
        from . import adapter as _adapter

        if self._is_bad_request_error(exc) or self._is_rich_capability_error(exc):
            return True
        s = str(exc).lower()
        return "unsupported" in s or "not implemented" in s

    def _chunk_reply_routing(
        self, chat_id: str, reply_to: Optional[str], metadata: Optional[Dict[str, Any]], thread_id: Optional[str], index: int) -> tuple:
        """Reply-anchor routing for chunk ``index``: ``(private_dm_topic_send, anchor_off, reply_to_id)``.
        ``anchor_off``: reply_to_mode="off" on the DM-topic fallback path opts into "message_thread_id
        alone is enough" — don't fail loud because the anchor was suppressed by config."""
        from . import adapter as _adapter

        metadata_reply_to = self._metadata_reply_to_message_id(metadata)
        private_dm_topic_send = self._is_private_dm_topic_send(chat_id, thread_id, metadata)
        dm_topic_reply_to_off = private_dm_topic_send and self._reply_to_mode == "off" and self._dm_topic_fallback(metadata)
        reply_to_source = reply_to or (str(metadata_reply_to) if private_dm_topic_send and metadata_reply_to is not None else None)
        if private_dm_topic_send:
            should_thread = reply_to_source is not None and self._reply_to_mode != "off"
        else:
            should_thread = self._should_thread_reply(reply_to_source, index)
        reply_to_id = int(reply_to_source) if should_thread and reply_to_source else None
        return private_dm_topic_send, dm_topic_reply_to_off, reply_to_id

    def _compute_single_send_routing(
        self, chat_id: str, reply_to: Optional[str], metadata: Optional[Dict[str, Any]], thread_id: Optional[str]) -> Optional[tuple]:
        """Routing for a single (rich) send — mirrors send()'s index-0 block. Returns ``(reply_to_id,
        thread_kwargs)`` or ``None`` = skip rich, legacy owns the DM-topic fail-loud SendResult."""
        private_dm_topic_send, dm_topic_reply_to_off, reply_to_id = self._chunk_reply_routing(chat_id, reply_to, metadata, thread_id, 0)
        thread_kwargs = self._thread_kwargs_for_send(
            chat_id, thread_id, metadata, reply_to_message_id=reply_to_id, reply_to_mode=self._reply_to_mode)
        # Synthetic/resumed sends via direct_messages_topic_id need no reply anchor.
        if (
            private_dm_topic_send and reply_to_id is None and not dm_topic_reply_to_off
            and not thread_kwargs.get("direct_messages_topic_id")):
            return None
        return reply_to_id, thread_kwargs

    @staticmethod
    def _is_timed_out(exc: Exception) -> bool:
        """PTB ``TimedOut`` (when importable) or a "timed out" message."""
        from . import adapter as _adapter

        try:
            from telegram.error import TimedOut as _TimedOut
        except (ImportError, AttributeError):
            _TimedOut = None
        return bool((_TimedOut and isinstance(exc, _TimedOut)) or "timed out" in str(exc).lower())

    def _rich_transient_result(self, exc: Exception, what: str, *, retry_after: Any = None) -> SendResult:
        """SendResult for a transient/unknown rich-API failure (request may have reached Telegram, so the
        caller must NOT legacy-resend); retry semantics mirror legacy send()."""
        from . import adapter as _adapter

        safe_error = _adapter._redact_telegram_error_text(exc)
        _adapter.logger.warning("[%s] %s transient failure (no legacy resend): %s", self.name, what, safe_error)
        return _adapter.SendResult(
            success=False, error=safe_error,
            retryable=(self._looks_like_connect_timeout(exc) or not self._is_timed_out(exc)), retry_after=retry_after)

    @staticmethod
    def _record_rich_sent(chat_id: Any, message_id: Any, content: str) -> None:
        """Index rich content we sent: Telegram won't echo it back in reply_to_message."""
        from . import adapter as _adapter

        try:
            from gateway import rich_sent_store
            rich_sent_store.record(str(chat_id), str(message_id), content)
        except Exception:
            pass

    async def _try_send_rich(
        self, chat_id: str, content: str, reply_to: Optional[str], metadata: Optional[Dict[str, Any]]) -> Optional[SendResult]:
        """Attempt a single ``sendRichMessage``. Returns a SendResult (success, or a transient failure the
        caller must NOT legacy-resend), or ``None`` = fall back to legacy MarkdownV2."""
        from . import adapter as _adapter

        thread_id = self._metadata_thread_id(metadata)
        routing = self._compute_single_send_routing(chat_id, reply_to, metadata, thread_id)
        if routing is None:
            return None
        reply_to_id, thread_kwargs = routing
        payload = self._rich_payload_base(chat_id, content)
        # Only non-None routing keys: direct_messages_topic_id is paired with message_thread_id=None.
        payload.update({k: v for k, v in thread_kwargs.items() if v is not None})
        payload.update(self._notification_kwargs(metadata))
        if reply_to_id is not None:
            # sendRichMessage takes reply_parameters, NOT reply_to_message_id (silently ignored → anchor dropped).
            payload["reply_parameters"] = {"message_id": reply_to_id}
        try:
            # Raw Bot API result: return_type=Message would make PTB deserialize a 10.1 shape it doesn't
            # fully model; a post-delivery parse error ≠ send failure.
            msg = await self._bot.do_api_request("sendRichMessage", api_kwargs=payload)
        except Exception as exc:
            if self._rich_rejected(exc, "sendRichMessage", "MarkdownV2"):
                return None
            # Honor Telegram's flood-control retry_after over the base retry schedule.
            _retry_after = getattr(exc, "retry_after", None)
            if _retry_after is None:
                _m = _adapter.re.search(r"retry\s+(?:in\s+)?(\d+)", str(exc).lower(), _adapter.re.IGNORECASE)
                if _m:
                    _retry_after = float(_m.group(1))
            return self._rich_transient_result(exc, "sendRichMessage", retry_after=_retry_after)
        if isinstance(msg, dict):
            message_id = msg.get("message_id")
            if message_id is None:
                message_id = (msg.get("result") or {}).get("message_id")
        else:
            message_id = getattr(msg, "message_id", None)
        if message_id is not None:
            self._record_rich_sent(chat_id, message_id, content)
        return _adapter.SendResult(success=True, message_id=str(message_id) if message_id is not None else None)

    def _rich_payload_base(self, chat_id: str, content: str) -> Dict[str, Any]:
        from . import adapter as _adapter

        payload: _adapter.Dict[str, _adapter.Any] = {"chat_id": _adapter.normalize_telegram_chat_id(chat_id), "rich_message": self._rich_message_payload(content)}
        if getattr(self, "_disable_link_previews", False):
            payload["link_preview_options"] = {"is_disabled": True}
        return payload

    def _rich_rejected(self, exc: Exception, what: str, fallback: str) -> bool:
        """True for a permanent/capability rich-API failure (caller falls back to legacy); capability
        errors latch rich off so no doomed roundtrip repeats per send."""
        from . import adapter as _adapter

        if not self._is_rich_fallback_error(exc):
            return False
        if self._is_rich_capability_error(exc):
            self._rich_send_disabled = True
        _adapter.logger.debug("[%s] %s rejected (%s) — falling back to %s", self.name, what, _adapter._redact_telegram_error_text(exc), fallback)
        return True

    async def _try_edit_rich(
        self, chat_id: str, message_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> Optional[SendResult]:
        """Edit a message in place as rich (``editMessageText`` + ``rich_message``) so a streamed preview
        finalizes without send+delete. Same contract as :meth:`_try_send_rich`."""
        # No topic routing on edits: message_thread_id/direct_messages_topic_id make Telegram reject it.
        from . import adapter as _adapter

        payload = {**self._rich_payload_base(chat_id, content), "message_id": int(message_id)}
        try:
            await self._bot.do_api_request("editMessageText", api_kwargs=payload)
        except Exception as exc:
            # "Message is not modified" = successful no-op; skip the redundant legacy edit.
            if "not modified" in str(exc).lower():
                if self._is_rich_fallback_error(exc) and self._is_rich_capability_error(exc):
                    self._rich_send_disabled = True
                return _adapter.SendResult(success=True, message_id=message_id)
            if self._rich_rejected(exc, "rich editMessageText", "MarkdownV2 edit"):
                return None
            return self._rich_transient_result(exc, "rich editMessageText")
        # Mirror the fresh-send index: a streamed final finalized via edit is otherwise never recorded.
        self._record_rich_sent(chat_id, message_id, content)
        return _adapter.SendResult(success=True, message_id=message_id)

    def _should_attempt_rich_draft(self, content: str) -> bool:
        from . import adapter as _adapter

        return bool(
            getattr(self, "_rich_messages_enabled", True)
            and getattr(self, "_rich_drafts_enabled", False)
            and not getattr(self, "_rich_send_disabled", False)
            and not getattr(self, "_rich_draft_disabled", False)
            and self._rich_content_ok(content))

    async def _try_send_rich_draft(self, chat_id: str, draft_id: int, content: str, metadata: Optional[Dict[str, Any]]) -> bool:
        """Emit one ``sendRichMessageDraft`` frame; True on success. Frames are ephemeral, so any failure
        returns False and the caller renders the legacy draft; capability failures latch off."""
        from . import adapter as _adapter

        payload: _adapter.Dict[str, _adapter.Any] = {
            "chat_id": _adapter.normalize_telegram_chat_id(chat_id), "draft_id": int(draft_id), "rich_message": self._rich_message_payload(content)}
        payload.update(self._thread_kwargs_for_draft(chat_id, metadata))
        try:
            return bool(await self._bot.do_api_request("sendRichMessageDraft", api_kwargs=payload))
        except Exception as exc:
            if self._is_rich_capability_error(exc):
                self._rich_draft_disabled = True
                _adapter.logger.debug(
                    "[%s] sendRichMessageDraft unsupported (%s) — using legacy drafts", self.name, _adapter._redact_telegram_error_text(exc))
            else:
                _adapter.logger.debug(
                    "[%s] sendRichMessageDraft transient failure (%s) — legacy draft this frame", self.name,
                    _adapter._redact_telegram_error_text(exc))
            return False

    def _should_thread_reply(self, reply_to: Optional[str], chunk_index: int) -> bool:
        """Whether this chunk (0 = first) should reply-thread to ``reply_to``, per reply_to_mode."""
        if not reply_to:
            return False
        mode = self._reply_to_mode
        if mode == "off":
            return False
        if mode == "all":
            return True
        return chunk_index == 0  # "first" (default)

    @staticmethod
    def _telegram_error_types() -> tuple:
        """``(NetworkError, BadRequest, TimedOut)`` from PTB, with import-failure fallbacks
        (``OSError``, ``None``, ``None``) so send() still classifies without the SDK."""
        from . import adapter as _adapter

        try:
            from telegram.error import NetworkError as _NetErr
        except ImportError:
            _NetErr = OSError  # type: ignore[misc,assignment]
        try:
            from telegram.error import BadRequest as _BadReq
        except ImportError:
            _BadReq = None  # type: ignore[assignment,misc]
        try:
            from telegram.error import TimedOut as _TimedOut
        except (ImportError, AttributeError):
            _TimedOut = None  # type: ignore[assignment,misc]
        return _NetErr, _BadReq, _TimedOut

    async def _send_chunk_markdown_or_plain(self, chunk: str, send_kwargs: Dict[str, Any]):
        """MarkdownV2 first; on a parse/markdown rejection resend as stripped plain text."""
        from . import adapter as _adapter

        try:
            return await self._bot.send_message(text=chunk, parse_mode=_adapter.ParseMode.MARKDOWN_V2, **send_kwargs)
        except Exception as md_error:
            if "parse" in str(md_error).lower() or "markdown" in str(md_error).lower():
                _adapter.logger.warning("[%s] MarkdownV2 parse failed, falling back to plain text: %s", self.name, md_error)
                return await self._bot.send_message(text=_adapter._strip_mdv2(chunk), parse_mode=None, **send_kwargs)
            raise

    async def _send_chunk_with_retries(
        self, chat_id: str, chunk: str, index: int, reply_to: Optional[str], metadata: Optional[Dict[str, Any]],
        thread_id: Optional[str], used_thread_fallback: bool, error_types: tuple):
        """Deliver one chunk: routing, up to 3 attempts, thread-not-found / deleted-anchor / flood handling.

        Returns ``(msg, used_thread_fallback)`` on success or a ``SendResult`` to return verbatim (fail-loud DM-topic
        cases, flood cap); raises anything the caller's classifier should see."""
        from . import adapter as _adapter

        _NetErr, _BadReq, _TimedOut = error_types
        retried_thread_not_found = False
        private_dm_topic_send, dm_topic_reply_to_off, reply_to_id = self._chunk_reply_routing(chat_id, reply_to, metadata, thread_id, index)
        if private_dm_topic_send and reply_to_id is None and not dm_topic_reply_to_off:
            return _adapter.SendResult(success=False, error=self._dm_topic_missing_anchor_error(), retryable=False)
        thread_kwargs = self._thread_kwargs_for_send(
            chat_id, thread_id, metadata, reply_to_message_id=reply_to_id, reply_to_mode=self._reply_to_mode)
        if used_thread_fallback and thread_kwargs.get("message_thread_id") is not None:
            thread_kwargs = dict(thread_kwargs)
            thread_kwargs["message_thread_id"] = None
        effective_thread_id = thread_kwargs.get("message_thread_id")
        for _send_attempt in range(3):
            try:
                send_kwargs = {
                    "chat_id": _adapter.normalize_telegram_chat_id(chat_id), "reply_to_message_id": reply_to_id, **thread_kwargs,
                    **self._link_preview_kwargs(), **self._notification_kwargs(metadata)}
                return await self._send_chunk_markdown_or_plain(chunk, send_kwargs), used_thread_fallback
            except _NetErr as send_err:
                # BadRequest subclasses NetworkError in PTB but is permanent; handle specific cases.
                if _BadReq and isinstance(send_err, _BadReq):
                    if self._is_thread_not_found_error(send_err) and effective_thread_id is not None:
                        if private_dm_topic_send or (metadata and metadata.get("telegram_dm_topic_created_for_send")):
                            return _adapter.SendResult(success=False, error=str(send_err), retryable=False)
                        # One-off "thread not found" flakes recover on immediate retry: same thread_id once.
                        if not retried_thread_not_found:
                            retried_thread_not_found = True
                            _adapter.logger.warning("[%s] Thread %s not found, retrying once with same thread_id", self.name, effective_thread_id)
                            continue
                        # Thread is genuinely gone: retry without it and prune the stale binding.
                        _adapter.logger.warning("[%s] Thread %s not found, retrying without message_thread_id", self.name, effective_thread_id)
                        self._prune_stale_dm_topic_binding(chat_id, effective_thread_id, metadata=metadata)
                        used_thread_fallback = True
                        effective_thread_id = None
                        thread_kwargs = {"message_thread_id": None}
                        continue
                    if "message to be replied not found" in str(send_err).lower() and reply_to_id is not None:
                        safe_send_error = _adapter._redact_telegram_error_text(send_err)
                        if private_dm_topic_send:
                            return _adapter.SendResult(success=False, error=safe_send_error, retryable=False)
                        # Reply target deleted; private-topic fallback sends drop anchor + topic id together.
                        _adapter.logger.warning("[%s] Reply target deleted, retrying without reply_to: %s", self.name, safe_send_error)
                        reply_to_id = None
                        if self._dm_topic_fallback(metadata):
                            thread_kwargs = {}
                        else:
                            thread_kwargs = self._thread_kwargs_for_send(
                                chat_id, thread_id, metadata, reply_to_message_id=reply_to_id, reply_to_mode=self._reply_to_mode)
                        effective_thread_id = thread_kwargs.get("message_thread_id")
                        continue
                    raise  # other BadRequest errors are permanent
                # TimedOut also subclasses NetworkError: a generic timeout may have reached Telegram (don't
                # retry); a wrapped ConnectTimeout or an httpx pool timeout is safe to retry.
                is_pool_timeout = self._looks_like_pool_timeout(send_err)
                if (
                    _TimedOut and isinstance(send_err, _TimedOut)
                    and not self._looks_like_connect_timeout(send_err) and not is_pool_timeout):
                    raise
                if is_pool_timeout:
                    await self._drain_general_connections_after_pool_timeout()
                if _send_attempt >= 2:
                    raise
                wait = 2 ** _send_attempt
                _adapter.logger.warning("[%s] Network error on send (attempt %d/3), retrying in %ds: %s",
                               self.name, _send_attempt + 1, wait, _adapter._redact_telegram_error_text(send_err))
                await _adapter.asyncio.sleep(wait)
            except Exception as send_err:
                retry_after = getattr(send_err, "retry_after", None)
                if retry_after is not None or "retry after" in str(send_err).lower():
                    wait = float(retry_after) if retry_after is not None else 1.0
                    safe_send_error = _adapter._redact_telegram_error_text(send_err)
                    # Never sleep a long server RetryAfter verbatim — it once pinned send() for 97 minutes.
                    # Mirror the edit path: a RetryAfter past a few seconds is not something to hold this
                    # coroutine open for. Sleeping the server value verbatim pinned send() for 97 minutes in
                    # production and froze inbound on every platform when it ran on the gateway boot path
                    # (#91969).
                    if wait > _adapter._FLOOD_INLINE_WAIT_CAP_SECS:
                        _adapter.logger.warning(
                            "[%s] Telegram flood control on send (retry_after=%.1fs > %.0fs); failing closed instead of sleeping: %s",
                            self.name, wait, _adapter._FLOOD_INLINE_WAIT_CAP_SECS, safe_send_error)
                        return _adapter._flood_cap_result(wait)
                    if _send_attempt < 2:
                        _adapter.logger.warning(
                            "[%s] Telegram flood control on send (attempt %d/3), retrying in %.1fs: %s", self.name,
                            _send_attempt + 1, wait, safe_send_error)
                        await _adapter.asyncio.sleep(wait)
                        continue
                raise

    async def _retrigger_typing(self, chat_id: str, metadata: Optional[Dict[str, Any]]) -> None:
        """Re-arm typing after an intermediate send (Telegram clears it when a message lands). Skipped on
        the FINAL reply (``metadata["notify"]``): the refresh loop is gone and no API cancels the bubble."""
        from . import adapter as _adapter

        if (metadata or {}).get("notify"):
            return
        with _adapter.contextlib.suppress(Exception):
            await self.send_typing(chat_id, metadata=metadata)

    async def send(
        self, chat_id: str, content: str, reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        """Send a message to a Telegram chat."""
        from . import adapter as _adapter

        if not self._bot:
            live = self._replacement_telegram_adapter()
            if live is not None:
                return await live.send(chat_id, content, reply_to, metadata)
            if self._is_permanent_fatal() or not await self._wait_for_reconnection():
                return _adapter.SendResult(success=False, error="Not connected", retryable=not self._is_permanent_fatal())
            live = self._replacement_telegram_adapter()
            if not self._bot and live is not None:
                return await live.send(chat_id, content, reply_to, metadata)
            if not self._bot:
                return _adapter.SendResult(success=False, error="Not connected", retryable=True)
        # getattr() — tests build adapters via object.__new__() (no __init__).
        if getattr(self, "_send_path_degraded", False):
            return _adapter.SendResult(success=False, error="send_path_degraded", retryable=True)
        # Skip whitespace-only text to prevent Telegram 400 empty-text errors.
        if not content or not content.strip():
            return _adapter.SendResult(success=True, message_id=None)
        error_types = self._telegram_error_types()
        try:
            # Bot API 10.1 rich fast-path; falls through to legacy MarkdownV2 on permanent/capability
            # errors or DM-topic skips; returns directly on success or transient failure (no legacy resend).
            if self._should_attempt_rich(content, metadata=metadata):
                rich_result = await self._try_send_rich(chat_id, content, reply_to, metadata)
                if rich_result is not None:
                    if rich_result.success:
                        await self._retrigger_typing(chat_id, metadata)
                    return rich_result
            chunks = self.truncate_message(self.format_message(content), self.MAX_MESSAGE_LENGTH, len_fn=_adapter.utf16_len)
            if len(chunks) > 1:
                # truncate_message appends a raw " (1/2)" suffix; escape the MarkdownV2-special parentheses.
                chunks = [
                    _adapter._separate_chunk_indicator_from_fence(_adapter.re.sub(r" \((\d+)/(\d+)\)$", r" \\(\1/\2\\)", chunk))
                    for chunk in chunks
               ]
            message_ids = []
            thread_id = self._metadata_thread_id(metadata)
            requested_thread_id = self._message_thread_id_for_send(thread_id)
            used_thread_fallback = False
            for i, chunk in enumerate(chunks):
                outcome = await self._send_chunk_with_retries(
                    chat_id, chunk, i, reply_to, metadata, thread_id, used_thread_fallback, error_types)
                if isinstance(outcome, _adapter.SendResult):
                    return outcome
                msg, used_thread_fallback = outcome
                message_ids.append(str(msg.message_id))
            await self._retrigger_typing(chat_id, metadata)
            return _adapter.SendResult(
                success=True, message_id=message_ids[0] if message_ids else None,
                raw_response={
                    "message_ids": message_ids, "requested_thread_id": requested_thread_id, "thread_fallback": used_thread_fallback})
        except Exception as e:
            safe_error = _adapter._redact_telegram_error_text(e)
            _adapter.logger.error("[%s] Failed to send Telegram message: %s", self.name, safe_error)
            err_str = str(e).lower()
            error_kind = _adapter.classify_send_error(e)
            # Content exceeded 4096 chars: fail so the stream consumer enters fallback mode.
            if "message_too_long" in err_str or "too long" in err_str:
                _adapter.logger.debug("[%s] send() content too long, falling back to new-message continuation", self.name)
                return _adapter.SendResult(success=False, error="message_too_long", error_kind="too_long")
            # TimedOut may have reached Telegram — non-retryable so _send_with_retry() doesn't re-send,
            # except a wrapped ConnectTimeout or an httpx pool timeout (safe to re-send).
            _to = error_types[2]
            is_timeout = (_to and isinstance(e, _to)) or "timed out" in err_str
            return _adapter.SendResult(
                success=False, error=safe_error,
                retryable=(self._looks_like_connect_timeout(e) or self._looks_like_pool_timeout(e) or not is_timeout),
                error_kind=error_kind)

    async def send_or_update_status(
        self, chat_id: str, status_key: str, content: str, *, metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        """Send a status message, or edit the previous one with the same ``(chat_id, status_key)``; if the
        edit fails (deleted, too old, …) the cached id is dropped and a fresh message is sent.

        Issue #30045: progress/status callbacks (context-pressure, lifecycle, compression, etc.) used to
        append a fresh bubble on every call. With this method, the first call sends and the message id is
        remembered; subsequent calls with the same (chat_id, status_key) edit that same message in place.
        """
        from . import adapter as _adapter

        key = (str(chat_id), str(status_key))
        cached_id = self._status_message_ids.get(key)
        if cached_id is not None:
            result = await self.edit_message(chat_id, cached_id, content, finalize=True, metadata=metadata)
            if result.success:
                if result.message_id:
                    self._status_message_ids[key] = str(result.message_id)
                return result
            self._status_message_ids.pop(key, None)
        result = await self.send(chat_id, content, metadata=metadata)
        if result.success and result.message_id:
            self._status_message_ids[key] = str(result.message_id)
        return result

    async def _edit_text(self, chat_id: str, message_id: str, text: str, parse_mode: Any = None) -> None:
        """``editMessageText`` with normalized ids; ``parse_mode=None`` sends plain text."""
        from . import adapter as _adapter

        kwargs: _adapter.Dict[str, _adapter.Any] = {"chat_id": _adapter.normalize_telegram_chat_id(chat_id), "message_id": int(message_id), "text": text}
        if parse_mode is not None:
            kwargs["parse_mode"] = parse_mode
        await self._bot.edit_message_text(**kwargs)

    async def _edit_markdown_or_plain(self, chat_id: str, message_id: str, formatted: str, plain: str, warn_fmt: str) -> bool:
        """MarkdownV2 edit with plain-text fallback. Returns True on a "not modified" no-op (caller may
        skip further work); the fallback edit's exceptions propagate."""
        from . import adapter as _adapter

        try:
            await self._edit_text(chat_id, message_id, formatted, _adapter.ParseMode.MARKDOWN_V2)
        except Exception as fmt_err:
            if "not modified" in str(fmt_err).lower():
                return True
            _adapter.logger.warning(warn_fmt, self.name, _adapter._redact_telegram_error_text(fmt_err))
            await self._edit_text(chat_id, message_id, plain)
        return False

    async def edit_message(
        self, chat_id: str, message_id: str, content: str, *, finalize: bool = False, metadata: Optional[Dict[str, Any]] = None,
       ) -> SendResult:
        """Edit a previously sent Telegram message.

        Telegram caps a message at 4096 UTF-16 codeunits. Streaming replies that outgrow it must NOT be truncated
        silently nor fail (the consumer would re-send a duplicate): edit with the first chunk, send the rest as
        continuations, and return the final chunk's id as the next edit target."""
        from . import adapter as _adapter

        if not self._bot:
            return _adapter.SendResult(success=False, error="Not connected")
        # Rich finalize (Bot API 10.1): edit the preview IN PLACE via rich_message — no fresh send + delete.
        # Before the 4,096 pre-flight because the rich cap is 32,768; falls back to legacy on rejection.
        # Rich finalize (Bot API 10.1): when the completed content has constructs the legacy MarkdownV2 edit
        # degrades (tables → bullet lists, task lists, <details>, block math) and rich is available, edit
        # the preview IN PLACE via editMessageText's rich_message param. No fresh send + delete → no
        # duplicate preview (the problem #46206 reverted the fresh-final path for). Attempted before the
        # 4,096 overflow pre-flight because the rich text cap is 32,768 — a rich table that exceeds the
        # MarkdownV2 limit must not be split into legacy chunks. Falls back to the legacy edit path
        # (overflow split included) on capability/permanent rejection.
        if finalize and self._rich_eligible(content):
            rich_result = await self._try_edit_rich(chat_id, message_id, content, metadata=metadata)
            if rich_result is not None:
                return rich_result
        # Pre-flight: over-limit content is split-and-delivered on finalize; mid-stream we truncate instead
        # (splitting moves the edit target to a continuation → infinite duplication loop).
        # Pre-flight: if content already exceeds the limit, split-and-deliver without round-tripping a
        # doomed edit. During streaming (finalize=False) we truncate instead of splitting — splitting
        # creates continuation messages whose IDs become the new edit target, and on the next token chunk
        # the full accumulated text is re-edited into the continuation, triggering another split → infinite
        # duplication loop (#48648).
        _preview_key = (str(chat_id), str(message_id))
        _saturated_preview = False
        if finalize:
            self._last_overflow_preview.pop(_preview_key, None)  # the final edit always delivers full content
        if _adapter.utf16_len(content) > self.MAX_MESSAGE_LENGTH:
            if finalize:
                return await self._edit_overflow_split(chat_id, message_id, content, finalize=finalize, metadata=metadata)
            content = self._truncate_stream_overflow_preview(content)
            _saturated_preview = True
            # Saturated-preview dedup: past the cap every progressive edit truncates to the same text;
            # re-sending is a visual no-op that still burns flood budget (200s+ penalties).
            if self._last_overflow_preview.get(_preview_key) == content:
                return _adapter.SendResult(success=True, message_id=message_id)
        elif not finalize:
            # Content shrank back under the cap — clear stale saturation state so dedup can't mask an edit.
            self._last_overflow_preview.pop(_preview_key, None)
        try:
            if not finalize:
                await self._edit_text(chat_id, message_id, content)
                if _saturated_preview:
                    self._last_overflow_preview[_preview_key] = content
                return _adapter.SendResult(success=True, message_id=message_id)
            await self._edit_markdown_or_plain(
                chat_id, message_id, self.format_message(content), _adapter._strip_mdv2(content) if content else content,
                "[%s] MarkdownV2 edit failed, falling back to plain text: %s")
            return _adapter.SendResult(success=True, message_id=message_id)
        except Exception as e:
            err_str = str(e).lower()
            if "not modified" in err_str:
                return _adapter.SendResult(success=True, message_id=message_id)
            # Reactive split: MarkdownV2 escapes can inflate the payload past the limit even when raw text fit.
            if "message_too_long" in err_str or "too long" in err_str:
                _adapter.logger.debug(
                    "[%s] edit_message overflow (%d UTF-16 > %d), splitting", self.name, _adapter.utf16_len(content), self.MAX_MESSAGE_LENGTH)
                if finalize:
                    return await self._edit_overflow_split(chat_id, message_id, content, finalize=finalize, metadata=metadata)
                # Mid-stream: truncate and retry instead of splitting (saturated-preview dedup as above).
                # See #48648.
                truncated = self._truncate_stream_overflow_preview(content)
                if self._last_overflow_preview.get(_preview_key) == truncated:
                    return _adapter.SendResult(success=True, message_id=message_id)
                await self._edit_text(chat_id, message_id, truncated)
                self._last_overflow_preview[_preview_key] = truncated
                return _adapter.SendResult(success=True, message_id=message_id)
            # Flood control: short waits retry inline; long waits fail immediately so streaming falls back
            # to a normal final send instead of a clipped partial.
            retry_after = getattr(e, "retry_after", None)
            if retry_after is not None or "retry after" in err_str:
                wait = retry_after if retry_after else 1.0
                _adapter.logger.warning("[%s] Telegram flood control, waiting %.1fs", self.name, wait)
                if wait > _adapter._FLOOD_INLINE_WAIT_CAP_SECS:
                    return _adapter._flood_cap_result(wait)
                await _adapter.asyncio.sleep(wait)
                try:
                    await self._edit_text(chat_id, message_id, content)
                    return _adapter.SendResult(success=True, message_id=message_id)
                except Exception as retry_err:
                    safe_retry_error = _adapter._redact_telegram_error_text(retry_err)
                    _adapter.logger.error("[%s] Edit retry failed after flood wait: %s", self.name, safe_retry_error)
                    return _adapter.SendResult(success=False, error=safe_retry_error)
            safe_error = _adapter._redact_telegram_error_text(e)
            # Transient network errors must not permanently disable progress-message editing.
            _transient_markers = (
                "connecterror", "connect error", "connection error", "networkerror", "network error", "timed out", "readtimeout",
                "writetimeout", "server disconnected", "temporarily unavailable", "temporary failure", "httpx")
            if any(m in err_str for m in _transient_markers):
                _adapter.logger.warning("[%s] Transient network error editing message %s (will retry): %s", self.name, message_id, safe_error)
                return _adapter.SendResult(success=False, error=safe_error, retryable=True)
            _adapter.logger.error("[%s] Failed to edit Telegram message %s: %s", self.name, message_id, safe_error)
            return _adapter.SendResult(success=False, error=safe_error)

    def _truncate_stream_overflow_preview(self, content: str) -> str:
        """One-message preview for oversized streaming edits (edits must keep targeting the original id;
        final edits use ``_edit_overflow_split``).

        Splitting a mid-stream preview creates continuation messages and moves the active message id, so the
        next accumulated-token edit repeats the overflow cycle (#48648). Final edits still use
        ``_edit_overflow_split`` to deliver the complete response.
        """
        from . import adapter as _adapter

        return self.truncate_message(content, self.MAX_MESSAGE_LENGTH, len_fn=_adapter.utf16_len)[0]

    async def _send_overflow_continuation(
        self, chat_id: str, chunk: str, reply_to_id: Optional[int], thread_kwargs: Dict[str, Any],
        thread_id: Optional[str], metadata: Optional[Dict[str, Any]], finalize: bool):
        """Send one continuation chunk (MarkdownV2 then plain on finalize; raw when streaming); drops the
        reply anchor once on 'reply message not found'. Returns the sent message or None."""
        from . import adapter as _adapter

        base = {**self._link_preview_kwargs(), **self._notification_kwargs(metadata)}
        for use_markdown in (True, False) if finalize else (False,):
            try:
                if use_markdown:
                    text = _adapter._separate_chunk_indicator_from_fence(self.format_message(chunk))
                else:
                    # Degrade to stripped text on finalize (raw ** / ``` would render literally); previews stay raw.
                    text = _adapter._strip_mdv2(chunk) if finalize else chunk
                return await self._bot.send_message(
                    chat_id=_adapter.normalize_telegram_chat_id(chat_id), text=text, parse_mode=_adapter.ParseMode.MARKDOWN_V2 if use_markdown else None,
                    reply_to_message_id=reply_to_id, **thread_kwargs, **base)
            except Exception as send_err:
                if "reply message not found" in str(send_err).lower():
                    # Private DM topic fallback needs anchor + topic id together; forum topics keep thread id.
                    retry_thread_kwargs = (
                        {} if self._dm_topic_fallback(metadata)
                        else self._thread_kwargs_for_send(chat_id, thread_id, metadata, reply_to_message_id=None))
                    try:
                        return await self._bot.send_message(
                            chat_id=_adapter.normalize_telegram_chat_id(chat_id), text=_adapter._strip_mdv2(chunk) if finalize else chunk,
                            **retry_thread_kwargs, **base)
                    except Exception as _retry_err:
                        _adapter.logger.warning(
                            "[%s] Overflow continuation no-reply retry failed: %s", self.name, _adapter._redact_telegram_error_text(_retry_err))
                        return None
                if use_markdown:
                    continue  # try plain text on next loop iteration
                _adapter.logger.warning("[%s] Overflow continuation send failed: %s", self.name, _adapter._redact_telegram_error_text(send_err))
                return None
        return None

    async def _edit_overflow_split(
        self, chat_id: str, message_id: str, content: str, *, finalize: bool, metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        """Split an oversized edit across the existing message + continuations: edit ``message_id`` with
        chunk 1, send the rest as replies to the previous chunk, return ``message_id=<last-chunk-id>`` so
        the consumer keeps editing the newest message. ``success=False`` only if the first-chunk edit fails."""
        from . import adapter as _adapter

        chunks = self.truncate_message(content, self.MAX_MESSAGE_LENGTH, len_fn=_adapter.utf16_len)
        if len(chunks) <= 1:
            chunks = [content]  # defensive: a single chunk just edits normally
        first_chunk = chunks[0]
        try:
            if finalize:
                await self._edit_markdown_or_plain(
                    chat_id, message_id, _adapter._separate_chunk_indicator_from_fence(self.format_message(first_chunk)), _adapter._strip_mdv2(first_chunk),
                    "[%s] Overflow split: MarkdownV2 first-chunk edit failed, falling back to plain text: %s")
            else:
                await self._edit_text(chat_id, message_id, first_chunk)
        except Exception as e:
            if "not modified" not in str(e).lower():  # identical first chunk still sends continuations
                _adapter.logger.error("[%s] Overflow split: first-chunk edit failed: %s", self.name, _adapter._redact_telegram_error_text(e), exc_info=True)
                return _adapter.SendResult(success=False, error=_adapter._redact_telegram_error_text(e))
        # Continuations call self._bot.send_message directly to skip self.send's pre-chunking.
        continuation_ids: list[str] = []
        delivered_chunks = [first_chunk]
        prev_id = message_id
        thread_id = self._metadata_thread_id(metadata)
        for chunk in chunks[1:]:
            reply_to_id = int(prev_id) if prev_id else None
            thread_kwargs = self._thread_kwargs_for_send(chat_id, thread_id, metadata, reply_to_message_id=reply_to_id)
            sent_msg = await self._send_overflow_continuation(chat_id, chunk, reply_to_id, thread_kwargs, thread_id, metadata, finalize)
            if sent_msg is None:
                # Partial delivery: do NOT report success — the consumer would treat it as final delivery.
                _adapter.logger.warning("[%s] Overflow split: stopped at %d/%d chunks delivered", self.name, 1 + len(continuation_ids), len(chunks))
                delivered_prefix = "".join(_adapter.re.sub(r" \(\d+/\d+\)$", "", delivered) for delivered in delivered_chunks)
                return _adapter.SendResult(
                    success=False, message_id=prev_id, error="overflow_continuation_failed", retryable=True,
                    raw_response={
                        "partial_overflow": True, "delivered_chunks": 1 + len(continuation_ids),
                        "total_chunks": len(chunks), "last_message_id": prev_id, "delivered_prefix": delivered_prefix,
                        "continuation_message_ids": tuple(continuation_ids)},
                    continuation_message_ids=tuple(continuation_ids))
            new_id = str(getattr(sent_msg, "message_id", "")) or prev_id
            continuation_ids.append(new_id)
            delivered_chunks.append(chunk)
            prev_id = new_id
        last_id = continuation_ids[-1] if continuation_ids else message_id
        _adapter.logger.debug("[%s] Overflow split delivered %d chunks; last_id=%s", self.name, 1 + len(continuation_ids), last_id)
        return _adapter.SendResult(success=True, message_id=last_id, continuation_message_ids=tuple(continuation_ids))

    async def delete_message(self, chat_id: str, message_id: str) -> bool:
        """Delete a bot-posted message (Bot API allows it within 48h); failures are non-fatal.

        Used by the stream consumer's fresh-final cleanup path (ported from openclaw/openclaw#72038) to
        remove long-lived preview messages after sending the completed reply as a fresh message. Telegram's
        Bot API ``deleteMessage`` works for bot-posted messages in the last 48 hours. Failures are non-fatal
        — the caller leaves the preview in place and logs at debug level.
        """
        from . import adapter as _adapter

        if not self._bot:
            return False
        try:
            await self._bot.delete_message(chat_id=_adapter.normalize_telegram_chat_id(chat_id), message_id=int(message_id))
            return True
        except Exception as e:
            _adapter.logger.debug("[%s] Failed to delete Telegram message %s: %s", self.name, message_id, _adapter._redact_telegram_error_text(e))
            return False

    def supports_draft_streaming(self, chat_type: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """sendMessageDraft works for private chats only (Bot API 9.5) and needs PTB >= 22.6; groups and
        older installs use the edit-based path. ``rich_drafts`` controls draft *format*, not availability."""
        if not self._bot or not hasattr(self._bot, "send_message_draft"):
            return False
        return (chat_type or "").lower() in {"dm", "private"}

    async def send_draft(self, chat_id: str, draft_id: int, content: str, metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        """Stream a partial message via ``sendRichMessageDraft`` (when rich is enabled and supported) else
        ``sendMessageDraft``; reusing ``draft_id`` animates the preview. The caller sends the final text."""
        from . import adapter as _adapter

        if not self._bot:
            return _adapter.SendResult(success=False, error="not_connected")
        # Rich draft fast-path; any failure degrades to the plain draft below. Drafts have no message_id.
        if self._should_attempt_rich_draft(content) and await self._try_send_rich_draft(chat_id, draft_id, content, metadata):
            return _adapter.SendResult(success=True, message_id=None)
        if not hasattr(self._bot, "send_message_draft"):
            return _adapter.SendResult(success=False, error="api_unavailable")
        # Drafts share the regular-send UTF-16 length contract.
        text = content if len(
            content) <= self.MAX_MESSAGE_LENGTH else self.truncate_message(content, self.MAX_MESSAGE_LENGTH, len_fn=_adapter.utf16_len)[0]
        # Same MarkdownV2 conversion as ``send`` (MarkdownV2 then plain) so the draft doesn't snap at the end. Exception: a Rich
        # final with rich drafts disabled previews raw — the legacy formatter would turn pipe tables into bullets.
        plain_rich_preview = bool(
            getattr(self, "_rich_messages_enabled", False) and not getattr(self, "_rich_drafts_enabled", False)
            and self._needs_rich_rendering(text))
        draft_thread_kwargs = self._thread_kwargs_for_draft(chat_id, metadata)
        for use_markdown in ((False,) if plain_rich_preview else (True, False)):
            kwargs: _adapter.Dict[str, _adapter.Any] = {
                "chat_id": _adapter.normalize_telegram_chat_id(chat_id), "draft_id": int(draft_id),
                "text": self.format_message(text) if use_markdown else text}
            if use_markdown:
                kwargs["parse_mode"] = _adapter.ParseMode.MARKDOWN_V2
            kwargs.update(draft_thread_kwargs)
            try:
                if await self._bot.send_message_draft(**kwargs):
                    return _adapter.SendResult(success=True, message_id=None)
                return _adapter.SendResult(success=False, error="draft_rejected")
            except Exception as e:
                # MarkdownV2 parse failure → retry once as plain text; anything else returns to the caller,
                # which falls back to edit-based streaming for this response.
                if use_markdown and self._is_bad_request_error(e):
                    _adapter.logger.debug(
                        "[%s] sendMessageDraft MarkdownV2 rejected, retrying as plain text (chat=%s draft_id=%s): %s",
                        self.name, chat_id, draft_id, _adapter._redact_telegram_error_text(e))
                    continue
                _adapter.logger.debug("[%s] sendMessageDraft failed (chat=%s draft_id=%s): %s", self.name, chat_id, draft_id, e)
                return _adapter.SendResult(success=False, error=_adapter._redact_telegram_error_text(e))
        return _adapter.SendResult(success=False, error="draft_rejected")

    async def _send_message_with_thread_fallback(self, **kwargs):
        """Send a control-style message (approval prompts, pickers), retrying once without
        message_thread_id on 'Message thread not found' (stale thread_id); ``send`` has its own.

        Used for control-style sends (approval prompts, model picker, update prompts) that can carry a stale
        thread_id from a DM reply chain. The streaming send loop has its own equivalent (PR #3390) at the
        body of ``send``; this helper applies the same retry pattern to the non-streaming control paths.
        """
        from . import adapter as _adapter

        if not self._bot:
            raise RuntimeError("Not connected")
        message_thread_id = kwargs.get("message_thread_id")
        try:
            return await self._bot.send_message(**kwargs)
        except Exception as send_err:
            if (message_thread_id is not None and self._is_bad_request_error(send_err) and self._is_thread_not_found_error(send_err)):
                _adapter.logger.warning(
                    "[%s] Thread %s not found for control message, retrying without message_thread_id", self.name, message_thread_id)
                # Same prune as the streaming send path; control sends carry no gateway metadata.
                self._prune_stale_dm_topic_binding(kwargs.get("chat_id"), message_thread_id)
                retry_kwargs = dict(kwargs)
                retry_kwargs.pop("message_thread_id", None)
                return await self._bot.send_message(**retry_kwargs)
            raise

    async def _send_control_message(
        self, chat_id: str, text: str, *, parse_mode: Any, thread_id: Optional[str], metadata: Optional[Dict[str, Any]],
        reply_markup: Any = None, reply_to_mode: Optional[str] = None):
        """Send a control-style message (prompt/picker) with topic routing + thread fallback."""
        from . import adapter as _adapter

        reply_to_id = self._reply_to_message_id_for_send(None, metadata, reply_to_mode=reply_to_mode)
        kwargs: _adapter.Dict[str, _adapter.Any] = {
            "chat_id": _adapter.normalize_telegram_chat_id(chat_id), "text": text, "parse_mode": parse_mode, **self._link_preview_kwargs()}
        if reply_markup is not None:
            kwargs["reply_markup"] = reply_markup
        kwargs["reply_to_message_id"] = reply_to_id
        kwargs.update(self._thread_kwargs_for_send(
            chat_id, thread_id, metadata, reply_to_message_id=reply_to_id, reply_to_mode=reply_to_mode))
        return await self._send_message_with_thread_fallback(**kwargs)

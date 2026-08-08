"""Bot API 10.1 Rich Messages helpers for ``TelegramAdapter``.

Extracted verbatim from ``plugins/platforms/telegram/adapter.py`` as part of
the god-file decomposition campaign. Holds the rich-message cluster: eligibility
gates, payload building, and the rich send/edit attempts. Module-level
``_rich_normalize_linebreaks`` and ``_RICH_PROTECTED_REGION_RE`` moved with the
cluster; ``RICH_MESSAGE_MAX_CHARS`` and the class-level rich regexes resolve via
the adapter class through the MRO.
"""

import inspect
import logging
import re
from typing import Any, Dict, List, Optional

from gateway.platforms.base import SendResult
from gateway.platforms.helpers import TABLE_SEPARATOR_RE as _TABLE_SEPARATOR_RE

from plugins.platforms.telegram.adapter import _redact_telegram_error_text
from plugins.platforms.telegram.telegram_ids import normalize_telegram_chat_id

logger = logging.getLogger("plugins.platforms.telegram.adapter")


# ---------------------------------------------------------------------------
# Rich-message newline normalization
# ---------------------------------------------------------------------------

# Matches a protected region whose internal newlines must stay bare in the
# rich-message path: a fenced code block (```...```) OR a GFM pipe-table block
# (a header row, a delimiter row of dashes/pipes, then any pipe data rows).
# Telegram renders both natively, so injecting Markdown hard breaks inside them
# would corrupt the code block / table.

_RICH_PROTECTED_REGION_RE = re.compile(
    r'(?:```[^\n]*\n[\s\S]*?```)'                       # fenced code block
    r'|(?:^[^\n]*\|[^\n]*\n'                            # table header row (has a pipe)
    r'[ \t]*\|?[ \t]*:?-+:?[ \t]*(?:\|[ \t]*:?-+:?[ \t]*)+\|?[ \t]*'  # delimiter
    r'(?:\n[^\n]*\|[^\n]*)*)',                          # data rows (newline-led, trailing \n left for prose)
    re.MULTILINE,
)


def _rich_normalize_linebreaks(text: str) -> str:
    """Convert single ``\\n`` to Markdown hard breaks for the rich-message path.

    Standard Markdown treats a lone ``\\n`` as whitespace (soft break), so
    Bot API 10.1 ``sendRichMessage`` collapses multi-line content — e.g.
    slash-command lists joined with ``"\\n".join(lines)`` — into a single
    paragraph.  Adding two trailing spaces before each single newline
    forces a hard line break (``<br>``) in the rendered output.

    Paragraph breaks (``\\n\\n``), fenced code blocks, and GFM pipe-table
    blocks are left untouched: tables render natively in the rich path and a
    hard break injected into a row separator would corrupt the table.
    """
    if not text or '\n' not in text:
        return text

    out: list[str] = []
    # Split off protected regions (fenced code OR table blocks) and only inject
    # hard breaks in the prose between them. Boundary newlines are handled by
    # the original single-\n regex, which sees each prose run as a whole string.
    pos = 0
    for m in _RICH_PROTECTED_REGION_RE.finditer(text):
        prose = text[pos:m.start()]
        out.append(re.sub(r'(?<!\n)\n(?!\n)', '  \n', prose))
        out.append(m.group(0))  # protected region kept verbatim
        pos = m.end()
    tail = text[pos:]
    out.append(re.sub(r'(?<!\n)\n(?!\n)', '  \n', tail))
    return ''.join(out)

class RichMixin:
    """Rich-message cluster lifted verbatim from ``TelegramAdapter``."""

    def _content_fits_rich_limits(self, content: str) -> bool:
        """Cheap pre-check for the one hard rich limit we can count locally.

        Only the 32,768 UTF-8 character text cap is enforced here. Other Bot API
        rich limits (500 blocks, 16 nesting levels, 20 table columns, ...) are
        not pre-counted; if exceeded Telegram returns a BadRequest, which
        :meth:`_is_rich_fallback_error` classifies as permanent so the send
        degrades to the legacy chunking path.
        """
        return len(content) <= self.RICH_MESSAGE_MAX_CHARS

    def _bot_supports_rich(self) -> bool:
        """True when the bound bot can issue raw ``sendRichMessage`` calls.

        Gates on ``do_api_request`` being an *async* callable. The real
        ``telegram.Bot.do_api_request`` is a coroutine function; test doubles
        that opt into rich set it to an ``AsyncMock`` (also a coroutine
        function). Plain ``MagicMock`` bots expose a *sync* auto-child and
        ``SimpleNamespace`` bots lack the attribute entirely — both resolve to
        ``False`` here, so the legacy path is used unchanged.
        """
        return inspect.iscoroutinefunction(getattr(self._bot, "do_api_request", None))

    _RICH_DETAILS_RE = re.compile(r"<details\b[^>]*>.*?</details>", re.IGNORECASE | re.DOTALL)
    _RICH_MATH_IN_DETAILS_RE = re.compile(
        r"(\$\$.*?\$\$|"
        r"\\\[.*?\\\]|"
        r"\\\(.*?\\\)|"
        r"\\(?:sum|frac|alpha|beta|gamma|delta|theta|lambda|mu|pi|sigma|"
        r"int|prod|sqrt|lim|infty|begin\{(?:equation|align|matrix|cases)\}))",
        re.IGNORECASE | re.DOTALL,
    )
    _RICH_CJK_RE = re.compile(
        "["
        "\u3040-\u30ff"  # Hiragana, Katakana
        "\u3400-\u4dbf"  # CJK Extension A
        "\u4e00-\u9fff"  # CJK Unified Ideographs
        "\uac00-\ud7af"  # Hangul syllables
        "\uf900-\ufaff"  # CJK Compatibility Ideographs
        "\U00020000-\U000323af"  # CJK extensions and compatibility supplement
        "]"
    )

    def _has_telegram_desktop_details_math_crash_shape(self, content: str) -> bool:
        """Return True for rich-message details+math content that crashes TDesktop.

        Telegram Desktop 6.9.1 can crash while rendering Bot API 10.1 rich
        messages containing math inside a collapsible details block
        (telegramdesktop/tdesktop#30808). The Bot API accepts the payload, so
        Hermes must skip rich delivery up front and use the legacy MarkdownV2
        path until affected Desktop clients age out.
        """
        if not content:
            return False
        for details_block in self._RICH_DETAILS_RE.findall(content):
            if self._RICH_MATH_IN_DETAILS_RE.search(details_block):
                return True
        return False

    def _has_telegram_desktop_cjk_rich_garble_shape(self, content: str) -> bool:
        """Return True for CJK content that current TDesktop rich drafts garble.

        Telegram Mac/Desktop Bot API 10.1 rich-message rendering currently
        leaves overlapping draft/overlay glyph artifacts for CJK text (#47653).
        The legacy MarkdownV2 path renders the same text cleanly, so skip rich
        delivery up front until affected clients age out.
        """
        return bool(content and self._RICH_CJK_RE.search(content))

    def _needs_rich_rendering(self, content: str) -> bool:
        """Return True for markdown constructs that the legacy path degrades.

        Keep ordinary replies on the pre-rich MarkdownV2 path so Telegram
        clients render a consistent font weight/spacing. The rich endpoint is
        reserved for constructs where raw markdown materially improves output:
        pipe tables (MarkdownV2 has no table syntax and rewrites them into
        bullet lists), GFM task lists, collapsible ``<details>`` blocks, and
        block math.  Adapted from #45995 (@YonganZhang).
        """
        if not content:
            return False
        if any(_TABLE_SEPARATOR_RE.match(line) for line in content.splitlines()):
            return True
        if re.search(r"(?m)^\s*[-*]\s+\[[ xX]\]\s+", content):
            return True
        if re.search(r"(?m)^<details\b|^</details>|^<summary\b|^</summary>", content):
            return True
        if "$$" in content:
            return True
        return False

    def _rich_delivery_enabled(self) -> bool:
        """Whether rich delivery is allowed (``rich_messages`` opt-in)."""
        return bool(getattr(self, "_rich_messages_enabled", True))

    def _rich_eligible(self, content: str) -> bool:
        """Capability/content eligibility for rich, ignoring ``expect_edits``.

        Shared core of :meth:`_should_attempt_rich` minus the per-call
        ``expect_edits`` metadata gate.  The rich EDIT-finalize path
        (:meth:`_try_edit_rich`) needs this: a streamed preview is sent with
        ``expect_edits=True`` to stay on the editable path mid-stream, but the
        FINAL edit should still upgrade to rich when the content warrants it.
        """
        return bool(
            self._rich_delivery_enabled()
            and not getattr(self, "_rich_send_disabled", False)
            and content
            and content.strip()
            and self._needs_rich_rendering(content)
            and not self._has_telegram_desktop_details_math_crash_shape(content)
            and not self._has_telegram_desktop_cjk_rich_garble_shape(content)
            and self._content_fits_rich_limits(content)
            and self._bot_supports_rich()
        )

    def _should_attempt_rich(
        self, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        return bool(
            not (metadata or {}).get("expect_edits")
            and self._rich_eligible(content)
        )

    def prefers_fresh_final_streaming(
        self, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Whether to replace a streamed preview with a fresh rich final.

        Disabled for Telegram. The fresh-final path briefly shows two copies of
        the final answer, then deletes the streaming preview after the rich send
        succeeds — it looks like duplicate delivery at the end of every streamed
        turn (the reason #46206 reverted it).  Rich finalize is instead handled
        by editing the existing preview in place via Bot API 10.1's
        ``editMessageText`` ``rich_message`` parameter (see
        :meth:`_try_edit_rich`), so no fresh re-send / delete is needed.
        """
        return False

    def streaming_overflow_limit(self) -> Optional[int]:
        """Allow the stream consumer to accumulate up to the rich-message cap
        before splitting, so a reply that fits one ``sendRichMessage`` /
        ``sendRichMessageDraft`` isn't fragmented at the 4,096 MarkdownV2 limit.

        Gated on the same rich capability as the send path (minus the
        content-length check — raising that cap is the whole point): rich not
        latched off and the bot exposes an async ``do_api_request``.  Returns
        ``None`` (→ legacy 4,096 limit) when rich isn't available, so non-rich
        streams split exactly as before.
        """
        if (
            getattr(self, "_rich_messages_enabled", True)
            and not getattr(self, "_rich_send_disabled", False)
            and self._bot_supports_rich()
        ):
            return self.RICH_MESSAGE_MAX_CHARS
        return None

    def _rich_message_payload(
        self, content: str, *, skip_entity_detection: bool = False
    ) -> Dict[str, Any]:
        """Build the ``InputRichMessage`` object from RAW markdown.

        Never pass ``format_message(content)`` here — that converts to
        MarkdownV2 and would escape/destroy rich syntax like table pipes.

        Single newlines are normalized to Markdown hard breaks so that
        multi-line content (slash-command lists, etc.) renders correctly
        in the rich-message path.  See ``_rich_normalize_linebreaks``.
        """
        payload: Dict[str, Any] = {"markdown": _rich_normalize_linebreaks(content)}
        if skip_entity_detection:
            payload["skip_entity_detection"] = True
        return payload

    def _is_rich_capability_error(self, exc: Exception) -> bool:
        """True ⇒ the rich endpoint itself is unavailable (old PTB/server).

        These latch rich off for the rest of the adapter's life — retrying is
        pointless and would cost a failed roundtrip on every send. Per-message
        rejections (BadRequest from a parser/limit issue) are NOT capability
        errors: the next message may be fine.
        """
        name = exc.__class__.__name__.lower()
        if name in {"endpointnotfound", "invalidtoken"}:
            return True
        if isinstance(exc, (AttributeError, TypeError, NotImplementedError)):
            return True
        if getattr(exc, "error_code", None) == 404:
            return True
        s = str(exc).lower()
        if ("method" in s or "endpoint" in s) and (
            "not found" in s or "does not exist" in s
        ):
            return True
        return "no such method" in s

    def _is_rich_fallback_error(self, exc: Exception) -> bool:
        """True ⇒ permanent/capability error ⇒ safe to fall back to legacy.

        Conservative on purpose: only clearly-permanent failures (BadRequest,
        capability errors, unknown/unsupported endpoint) qualify. Everything
        else is treated as transient — the rich request may have reached
        Telegram, so we must NOT legacy-resend and risk a duplicate.
        """
        if self._is_bad_request_error(exc):
            return True
        if self._is_rich_capability_error(exc):
            return True
        s = str(exc).lower()
        return "unsupported" in s or "not implemented" in s

    def _compute_single_send_routing(
        self,
        chat_id: str,
        reply_to: Optional[str],
        metadata: Optional[Dict[str, Any]],
        thread_id: Optional[str],
    ) -> Optional[tuple]:
        """Routing for a single (rich) send — mirrors send()'s index-0 block.

        Returns ``(reply_to_id, thread_kwargs)``, or ``None`` to signal "skip
        rich, let the legacy path handle it" — used for the DM-topic fail-loud
        case so the legacy path stays the single source of the refuse result.
        """
        metadata_reply_to = self._metadata_reply_to_message_id(metadata)
        private_dm_topic_send = self._is_private_dm_topic_send(chat_id, thread_id, metadata)
        dm_topic_reply_to_off = (
            private_dm_topic_send
            and self._reply_to_mode == "off"
            and bool(metadata and metadata.get("telegram_dm_topic_reply_fallback"))
        )
        reply_to_source = reply_to or (
            str(metadata_reply_to)
            if private_dm_topic_send and metadata_reply_to is not None
            else None
        )
        if private_dm_topic_send:
            should_thread = reply_to_source is not None and self._reply_to_mode != "off"
        else:
            should_thread = self._should_thread_reply(reply_to_source, 0)
        reply_to_id = int(reply_to_source) if should_thread and reply_to_source else None
        thread_kwargs = self._thread_kwargs_for_send(
            chat_id,
            thread_id,
            metadata,
            reply_to_message_id=reply_to_id,
            reply_to_mode=self._reply_to_mode,
        )
        if private_dm_topic_send and reply_to_id is None and not dm_topic_reply_to_off:
            # Refusing to send outside the requested DM topic — defer to the
            # legacy path, which returns the canonical fail-loud SendResult.
            # Exception: synthetic/resumed topic sends that route via
            # ``direct_messages_topic_id`` do not need a reply anchor.
            if not thread_kwargs.get("direct_messages_topic_id"):
                return None
        return reply_to_id, thread_kwargs

    async def _try_send_rich(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str],
        metadata: Optional[Dict[str, Any]],
    ) -> Optional[SendResult]:
        """Attempt a single ``sendRichMessage`` send.

        Returns a :class:`SendResult` (success, or a transient failure that the
        caller must NOT legacy-resend), or ``None`` to signal "fall back to the
        legacy MarkdownV2 path" (permanent/capability error or DM-topic skip).
        """
        thread_id = self._metadata_thread_id(metadata)
        routing = self._compute_single_send_routing(chat_id, reply_to, metadata, thread_id)
        if routing is None:
            return None
        reply_to_id, thread_kwargs = routing

        payload: Dict[str, Any] = {
            "chat_id": normalize_telegram_chat_id(chat_id),
            "rich_message": self._rich_message_payload(content),
        }
        # Only forward non-None routing keys: when direct_messages_topic_id is
        # present _thread_kwargs_for_send pairs it with message_thread_id=None,
        # which must not be sent as a stray field on the raw endpoint.
        payload.update({k: v for k, v in thread_kwargs.items() if v is not None})
        payload.update(self._notification_kwargs(metadata))
        if getattr(self, "_disable_link_previews", False):
            payload["link_preview_options"] = {"is_disabled": True}
        if reply_to_id is not None:
            # Spec: sendRichMessage takes reply_parameters (ReplyParameters
            # object), NOT the legacy reply_to_message_id scalar. Unknown
            # params are silently ignored by the Bot API, so the scalar would
            # quietly drop the reply anchor instead of erroring.
            payload["reply_parameters"] = {"message_id": reply_to_id}

        try:
            # Take the raw Bot API result (dict under real PTB). Passing
            # return_type=Message would make PTB deserialize a Bot API 10.1
            # response shape it does not fully model yet; a post-delivery parse
            # error must not be mistaken for a sendable failure.
            msg = await self._bot.do_api_request(
                "sendRichMessage", api_kwargs=payload
            )
        except Exception as exc:
            if self._is_rich_fallback_error(exc):
                if self._is_rich_capability_error(exc):
                    # Endpoint missing (old PTB/server) — latch rich off so
                    # every later send doesn't pay a doomed extra roundtrip.
                    self._rich_send_disabled = True
                logger.debug(
                    "[%s] sendRichMessage rejected (%s) — falling back to MarkdownV2",
                    self.name, _redact_telegram_error_text(exc),
                )
                return None
            # Transient / network / unknown: the request may have reached
            # Telegram. Do NOT legacy-resend (duplicate risk); surface a
            # failure with retry semantics mirroring the legacy send() except.
            err_str = str(exc).lower()
            try:
                from telegram.error import TimedOut as _TimedOut
            except (ImportError, AttributeError):
                _TimedOut = None
            is_timeout = (_TimedOut and isinstance(exc, _TimedOut)) or "timed out" in err_str
            is_connect_timeout = self._looks_like_connect_timeout(exc)
            # Extract server-requested retry_after for flood control so the
            # base retry layer honors Telegram's backoff instead of its own
            # short exponential schedule.
            _retry_after = getattr(exc, "retry_after", None)
            if _retry_after is None:
                import re as _re
                _m = _re.search(r"retry\s+(?:in\s+)?(\d+)", err_str, _re.IGNORECASE)
                if _m:
                    _retry_after = float(_m.group(1))
            safe_error = _redact_telegram_error_text(exc)
            logger.warning(
                "[%s] sendRichMessage transient failure (no legacy resend): %s",
                self.name, safe_error,
            )
            return SendResult(
                success=False,
                error=safe_error,
                retryable=(is_connect_timeout or not is_timeout),
                retry_after=_retry_after,
            )

        message_id = None
        if isinstance(msg, dict):
            message_id = msg.get("message_id")
            if message_id is None:
                message_id = (msg.get("result") or {}).get("message_id")
        else:
            message_id = getattr(msg, "message_id", None)
        if message_id is not None:
            # Telegram won't echo rich content in reply_to_message, so remember
            # what we sent — replies to this message resolve via this index.
            try:
                from gateway import rich_sent_store
                rich_sent_store.record(str(chat_id), str(message_id), content)
            except Exception:
                pass
        return SendResult(
            success=True,
            message_id=str(message_id) if message_id is not None else None,
        )

    async def _try_edit_rich(
        self,
        chat_id: str,
        message_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[SendResult]:
        """Edit an existing message in place as a rich message (Bot API 10.1).

        Uses ``editMessageText`` with the ``rich_message`` parameter so a
        streamed preview can finalize as rich (tables/task lists/details/math)
        WITHOUT a fresh send + delete — no duplicate preview.  Mirrors
        :meth:`_try_send_rich`'s error contract:

        - success → ``SendResult(success=True, message_id=...)``
        - permanent / capability error → ``None`` (caller falls back to the
          legacy MarkdownV2 edit; capability errors latch rich off)
        - transient / unknown → ``SendResult(success=False)`` with retry
          semantics (the message may already be edited; do NOT legacy-resend)
        """
        payload: Dict[str, Any] = {
            "chat_id": normalize_telegram_chat_id(chat_id),
            "message_id": int(message_id),
            "rich_message": self._rich_message_payload(content),
        }
        thread_id = self._metadata_thread_id(metadata)
        thread_kwargs = self._thread_kwargs_for_send(
            chat_id,
            thread_id,
            metadata,
            reply_to_message_id=None,
            reply_to_mode=self._reply_to_mode,
        )
        payload.update({k: v for k, v in thread_kwargs.items() if v is not None})
        if getattr(self, "_disable_link_previews", False):
            payload["link_preview_options"] = {"is_disabled": True}
        try:
            # Raw Bot API result; do not request return_type=Message (PTB does
            # not fully model the 10.1 response shape yet — a post-edit parse
            # error must not be mistaken for a failed edit).
            await self._bot.do_api_request("editMessageText", api_kwargs=payload)
        except Exception as exc:
            if self._is_rich_fallback_error(exc):
                if self._is_rich_capability_error(exc):
                    self._rich_send_disabled = True
                # "Message is not modified" — content identical to the current
                # rich message; treat as a successful no-op so the caller does
                # not fall through to a redundant legacy edit.
                if "not modified" in str(exc).lower():
                    return SendResult(success=True, message_id=message_id)
                logger.debug(
                    "[%s] rich editMessageText rejected (%s) — falling back to MarkdownV2 edit",
                    self.name, _redact_telegram_error_text(exc),
                )
                return None
            if "not modified" in str(exc).lower():
                return SendResult(success=True, message_id=message_id)
            err_str = str(exc).lower()
            try:
                from telegram.error import TimedOut as _TimedOut
            except (ImportError, AttributeError):
                _TimedOut = None
            is_timeout = (_TimedOut and isinstance(exc, _TimedOut)) or "timed out" in err_str
            is_connect_timeout = self._looks_like_connect_timeout(exc)
            safe_error = _redact_telegram_error_text(exc)
            logger.warning(
                "[%s] rich editMessageText transient failure (no legacy resend): %s",
                self.name, safe_error,
            )
            return SendResult(
                success=False,
                error=safe_error,
                retryable=(is_connect_timeout or not is_timeout),
            )
        # Telegram won't echo rich content for messages that predate the bot's
        # first rich send, so mirror the fresh-send index here too: a streamed
        # final finalized via editMessageText is otherwise never recorded, and
        # replies to it would have no native echo to recover from.
        try:
            from gateway import rich_sent_store
            rich_sent_store.record(str(chat_id), str(message_id), content)
        except Exception:
            pass
        return SendResult(success=True, message_id=message_id)

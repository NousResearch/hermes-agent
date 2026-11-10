"""Bot API 10.4 HTML widgets for the Telegram adapter, opt-in via ``platforms.telegram.extra.widgets``.

python-telegram-bot 22.8 models none of it. A widget tap is an update PTB parses into
``update.api_kwargs["widget_query"]`` (``["shared_widget_query"]`` for a forwarded shared widget), with
every ``effective_*`` None, and Telegram sends those update kinds to a bot whose explicit
``allowed_updates`` names them. Sends and data edits go through ``bot.do_api_request``.

Ingress: one group-0 handler, registered after the other group-0 core handlers, matches only widget-query
updates. No core handler matches them, so it displaces nothing; group 0 keeps it among the core handlers a
late plugin re-wire hoists in front of (#87770); its normal return receipts the update like any group-0
handler (update_admission.py), while every other update never reaches its callback. It is non-blocking:
PTB runs it as a task the update's admission claim stays pinned to, so a slow widget handler never holds
one of the processor's shared update slots, and ``WIDGET_HANDLER_SLOTS`` bounds them instead. Each claimed
query is answered exactly once within Telegram's 15 s window, refusals and failures included, or the
widget shows TIMEOUT; only a dispatch cancelled by a disconnect ends unanswered. Identity is Telegram's authenticated ``from``; ``data`` is untrusted and logged only
at debug level. A query without a chat (shared widget, inline message) is authorized as the sender's own
private chat, the narrowest scope. Only plain ``data`` requests reach the prefix registry: TON wallet
outcomes carry a ``data`` label too.

Outbound: local validation mirrors Telegram's limits, sends reuse the rich-message routing, and nothing is
sent again after an ambiguous failure, since a request that timed out may already have reached Telegram.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import zlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Optional

from gateway.platforms.base import SendResult
from plugins.platforms.telegram.callback_answer import is_stale_query_error
from plugins.platforms.telegram.telegram_ids import normalize_telegram_chat_id, parse_telegram_message_id
from plugins.platforms.telegram.telegram_send_errors import maybe_delivered_result

logger = logging.getLogger("plugins.platforms.telegram.adapter")

WIDGET_UPDATE_TYPES = ("widget_query", "shared_widget_query")
HTML_MAX_BYTES = 32768
HTML_MAX_COMPRESSED_BYTES = 4096  # after zlib at level 9
# Telegram counts the compressed size with classic zlib, which CPython's usual zlib reproduces byte for
# byte; zlib-ng (python-build-standalone, so the PM-pinned Python) compresses up to a few percent larger.
# With it only HTML clearly over the limit is refused locally, and Telegram decides the band below that.
_COMPRESSED_SLACK = HTML_MAX_COMPRESSED_BYTES // 16 if "zlib-ng" in zlib.ZLIB_VERSION or hasattr(zlib, "ZLIBNG_VERSION") else 0
DATA_MAX_BYTES = 1024  # chat_data, public_data and an answer's data
CACHE_TIME_SECONDS = (5, 3600)
# Telegram refuses an answer after 15 s, counted from the tap, so queueing before dispatch eats into it too:
# the handler gets 8 s and the answer request 3.
HANDLER_TIMEOUT_SECONDS = 8.0
ANSWER_TIMEOUT_SECONDS = 3.0
WIDGET_HANDLER_SLOTS = 8  # handlers running at once per adapter; a query past them is answered "busy"
# The adapter's own answers: stable JSON a widget can branch on, never localized.
ANSWER_UNAUTHORIZED = '{"error":"unauthorized"}'
ANSWER_UNKNOWN_QUERY = '{"error":"unknown_query"}'
ANSWER_UNSUPPORTED = '{"error":"unsupported"}'
ANSWER_BUSY = '{"error":"busy"}'
ANSWER_FAILED = '{"error":"failed"}'
ANSWER_TIMEOUT = '{"error":"timeout"}'
WIDGETS_DISABLED = "Telegram widgets are disabled; set platforms.telegram.extra.widgets: true to enable them"


@dataclass(frozen=True)
class WidgetQuery:
    """One widget query, free of SDK objects. Ids are text; ``chat_id`` is None when the query has no chat."""

    kind: str  # "widget_query" or "shared_widget_query"
    query_id: str
    user_id: str  # Telegram-authenticated sender; "" when absent (always refused)
    chat_id: Optional[str]
    chat_type: Optional[str]  # chat.type, or Telegram's chat_type of an inline message
    message_id: Optional[str]
    inline_message_id: Optional[str]
    widget_id: Optional[str]  # shared widgets only
    data: str  # untrusted, 1-64 bytes from the widget
    type: str  # WidgetQueryType.type: "data", "sign_data", "sent_transaction", ...


# ``async def handler(query) -> str | None``: the answer data.
WidgetQueryHandler = Callable[[WidgetQuery], Any]


def _mapping(value: Any) -> Optional[Mapping]:
    """A raw payload: the Mapping PTB keeps in ``api_kwargs``, or ``to_dict()`` of an SDK object."""
    if isinstance(value, Mapping):
        return value
    to_dict = getattr(value, "to_dict", None)
    raw = to_dict() if callable(to_dict) else None
    return raw if isinstance(raw, Mapping) else None


def widget_query_payload(update: Any) -> Optional[tuple[str, Mapping]]:
    """``(kind, raw payload)`` when ``update`` is a widget query, else None. A first-class SDK attribute wins
    (a python-telegram-bot that models widgets); 22.8 leaves the raw payload in ``api_kwargs``."""
    api_kwargs = getattr(update, "api_kwargs", None)
    for kind in WIDGET_UPDATE_TYPES:
        raw = _mapping(getattr(update, kind, None))
        if raw is None and isinstance(api_kwargs, Mapping):
            raw = _mapping(api_kwargs.get(kind))
        if raw is not None:
            return kind, raw
    return None


def _id_text(value: Any) -> Optional[str]:
    """A Telegram id as text; None when missing, empty or not an id (a bool is not an id)."""
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        return None
    return str(value).strip() or None


def normalize_widget_query(kind: str, raw: Mapping) -> Optional[WidgetQuery]:
    """The SDK-free view of one raw query; None without an id, since such a query can never be answered."""
    query_id = _id_text(raw.get("id"))
    if query_id is None:
        return None
    sender = raw.get("from")
    chat = raw.get("chat") if isinstance(raw.get("chat"), Mapping) else {}
    chat_id = _id_text(chat.get("id"))
    chat_type = chat.get("type") if chat_id is not None else raw.get("chat_type")
    query_type = raw.get("type", {"type": "data"})  # Telegram's own example query carries no type
    if isinstance(query_type, Mapping):
        query_type = query_type.get("type")
    data = raw.get("data")
    return WidgetQuery(
        kind=kind, query_id=query_id,
        user_id=(_id_text(sender.get("id")) if isinstance(sender, Mapping) else None) or "",
        chat_id=chat_id, chat_type=chat_type if isinstance(chat_type, str) else None,
        message_id=_id_text(raw.get("message_id")), inline_message_id=_id_text(raw.get("inline_message_id")),
        widget_id=_id_text(raw.get("widget_id")), data=data if isinstance(data, str) else "",
        type=query_type if isinstance(query_type, str) else "")


def truncate_utf8(text: str, max_bytes: int) -> str:
    """``text`` cut to at most ``max_bytes`` of UTF-8 without splitting a character."""
    return text.encode("utf-8", "replace")[:max_bytes].decode("utf-8", "ignore")


def resolve_widget_handler(
        handlers: Mapping[str, WidgetQueryHandler], data: str) -> Optional[tuple[str, WidgetQueryHandler]]:
    """``(prefix, handler)`` for the longest registered prefix of ``data``."""
    prefix = max((p for p in handlers if data.startswith(p)), key=len, default=None)
    return None if prefix is None else (prefix, handlers[prefix])


def widget_update_handler(base: type, update_type: Any, callback: Callable) -> Any:
    """A non-blocking ``base`` (PTB's ``TypeHandler``) matching only widget-query updates."""

    class WidgetUpdateHandler(base):
        __slots__ = ()

        def check_update(self, update: object) -> bool:
            return widget_query_payload(update) is not None

    return WidgetUpdateHandler(update_type, callback, block=False)


def _first_error(*errors: Optional[str]) -> Optional[str]:
    return next((error for error in errors if error), None)


def _utf8(text: str) -> Optional[bytes]:
    try:
        return text.encode("utf-8")
    except UnicodeEncodeError:  # lone surrogates: not valid UTF-8 text
        return None


def _html_error(html: Any) -> Optional[str]:
    """Why Telegram would refuse ``html`` (raw and compressed size, NUL, UTF-8), or None."""
    if not isinstance(html, str):
        return "widget html must be a string"
    raw = _utf8(html)
    if raw is None or "\x00" in html:
        return "widget html must be UTF-8 text without NUL characters"
    if not 1 <= len(raw) <= HTML_MAX_BYTES:
        return f"widget html is {len(raw)} UTF-8 bytes; Telegram allows 1-{HTML_MAX_BYTES}"
    compressed = len(zlib.compress(raw, 9))
    if compressed > HTML_MAX_COMPRESSED_BYTES + _COMPRESSED_SLACK:
        return (f"widget html compresses to {compressed} bytes (zlib level 9); "
                f"Telegram allows {HTML_MAX_COMPRESSED_BYTES}")
    return None


def _data_error(name: str, value: Any) -> Optional[str]:
    """Why ``value`` can't be widget state (``chat_data`` / ``public_data``); None when fine or omitted."""
    if value is None:
        return None
    raw = _utf8(value) if isinstance(value, str) else None
    if raw is None or "\x00" in value:
        return f"widget {name} must be UTF-8 text without NUL characters"
    if len(raw) > DATA_MAX_BYTES:
        return f"widget {name} is {len(raw)} UTF-8 bytes; Telegram allows {DATA_MAX_BYTES}"
    return None


def _int_error(name: str, value: Any, low: int, high: Optional[int] = None) -> Optional[str]:
    if value is None or (isinstance(value, int) and not isinstance(value, bool)
                         and low <= value and (high is None or value <= high)):
        return None
    return f"{name} must be an integer from {low}" + (f" to {high}" if high is not None else " up")


def _widget_id_error(widget_id: Any) -> Optional[str]:
    return None if isinstance(widget_id, str) and widget_id.strip() else "widget_id must be a non-empty string"


def _present(**fields: Any) -> dict[str, Any]:
    """Only the fields a caller passed: None means "omit", and Telegram reads an omitted field as unset."""
    return {key: value for key, value in fields.items() if value is not None}


def _sent_widget_result(message: Any, widget_id: Optional[str] = None) -> SendResult:
    """The SendResult of a sent widget message (raw Bot API dict); a shared widget's id rides in
    ``raw_response["widget_id"]``, the handle for ``send_shared_widget`` and ``edit_shared_widget_data``."""
    message = message if isinstance(message, Mapping) else {}
    widget = message.get("widget")
    widget_id = widget_id or (_id_text(widget.get("id")) if isinstance(widget, Mapping) else None)
    return SendResult(success=True, message_id=_id_text(message.get("message_id")),
                      raw_response={"widget_id": widget_id} if widget_id else None)


class TelegramWidgetsMixin:
    """Widget queries and widget sends for ``TelegramAdapter`` (``extra.widgets``, read in ``__init__``).

    PTB names (``Update``, ``TypeHandler``) are passed in by the adapter, never read from the facade module
    here: production loads the adapter twice (``hermes_plugins.platforms__telegram.adapter`` and
    ``plugins.platforms.telegram.adapter``), and a lazy python-telegram-bot install rebinds them only in the
    copy whose class is live.
    """

    # --- ingress ---------------------------------------------------------------------------------------

    def _allowed_update_types(self, all_types: Any) -> Any:
        """``all_types`` (the adapter's ``Update.ALL_TYPES``) itself while widgets are off, so polling and
        webhook requests stay byte-identical; with widgets on, a new list that also names both widget update
        types once."""
        if not getattr(self, "_widgets_enabled", False):
            return all_types
        return list(dict.fromkeys((*all_types, *WIDGET_UPDATE_TYPES)))

    def _add_widget_update_handler(self, app: Any, type_handler: type, update_type: Any) -> None:
        """Register the widget-query handler (group 0, after the core group-0 handlers) when widgets are on."""
        if not getattr(self, "_widgets_enabled", False):
            return
        app.add_handler(widget_update_handler(type_handler, update_type, self._handle_widget_update))

    def register_widget_query_handler(self, prefix: str, handler: WidgetQueryHandler) -> None:
        """Route widget queries whose ``data`` starts with ``prefix`` to ``handler``; the longest registered
        prefix wins, and registering a prefix again replaces its handler. The registry belongs to this
        adapter instance and starts empty.

        ``async def handler(query: WidgetQuery) -> str | None`` returns the answer data (cut to 1024 UTF-8
        bytes) within HANDLER_TIMEOUT_SECONDS; it runs on the gateway's event loop outside any turn, so it
        must not block. It runs only for senders who pass the allowlist, but authorizing the action itself
        stays its job, and so does idempotency: ``data`` is untrusted, a tap can repeat, and a handler that
        timed out may still finish after the widget was told "timeout".
        """
        if not isinstance(prefix, str) or not prefix:
            raise ValueError("widget query prefix must be a non-empty string")
        if not (callable(handler) and (inspect.iscoroutinefunction(handler)
                                       or inspect.iscoroutinefunction(type(handler).__call__))):
            raise TypeError("widget query handler must be an async function: it runs on the gateway's event loop")
        handlers = getattr(self, "_widget_query_handlers", None)
        if handlers is None:
            handlers = self._widget_query_handlers = {}
        if handlers.get(prefix) not in (None, handler):
            logger.warning("[%s] Widget query prefix %r registered again; the newer handler replaces it", self.name, prefix)
        handlers[prefix] = handler

    async def _handle_widget_update(self, update: Any, context: Any) -> None:
        """PTB callback: claim a widget query, then answer it exactly once. Never raises into PTB."""
        try:
            payload = widget_query_payload(update)
            query = normalize_widget_query(*payload) if payload is not None else None
        except Exception:
            self._fail_update_preparation()
            logger.debug("[%s] Widget query preparation failed", self.name, exc_info=True)
            return
        if query is None:
            if payload is not None:
                logger.warning("[%s] Dropped a %s without an id: it cannot be answered", self.name, payload[0])
            return
        self._accept_update()  # claimed: every path below answers exactly once
        try:
            data = await self._widget_query_answer(query)
        except Exception:
            logger.warning("[%s] Widget query %s dispatch failed", self.name, query.query_id, exc_info=True)
            data = ANSWER_FAILED
        await self._answer_widget_query(query, data)

    def _widget_query_authorized(self, query: WidgetQuery) -> bool:
        """Fail closed through the callback gate: the sender is Telegram's ``from``; a query with no chat is
        judged as the sender's own private chat; nothing is taken from the untrusted ``data``."""
        if not query.user_id:
            return False
        in_chat = query.chat_id is not None
        try:
            return self._is_callback_user_authorized(
                query.user_id, chat_id=query.chat_id if in_chat else query.user_id,
                chat_type=query.chat_type if in_chat else "private") is True
        except Exception:
            logger.debug("[%s] Widget query authorization failed; refusing", self.name, exc_info=True)
            return False

    def _widget_handler_slots(self) -> asyncio.Semaphore:
        slots = getattr(self, "_widget_slots", None)
        if slots is None:
            slots = self._widget_slots = asyncio.Semaphore(WIDGET_HANDLER_SLOTS)
        return slots

    async def _widget_query_answer(self, query: WidgetQuery) -> Optional[str]:
        """The answer data for a claimed query: a refusal, or what its registered handler returns."""
        if not self._widget_query_authorized(query):
            logger.debug("[%s] Widget query from unauthorized user %s refused", self.name, query.user_id)
            return ANSWER_UNAUTHORIZED
        if query.type != "data":
            logger.debug("[%s] Widget query of type %r is not handled", self.name, query.type[:32])
            return ANSWER_UNSUPPORTED
        route = resolve_widget_handler(getattr(self, "_widget_query_handlers", None) or {}, query.data)
        if route is None:
            logger.debug("[%s] Widget query matched no handler (data=%r)", self.name, query.data[:64])
            return ANSWER_UNKNOWN_QUERY
        slots = self._widget_handler_slots()
        if slots.locked():
            logger.warning("[%s] %d widget query handlers already running; query %s answered busy",
                           self.name, WIDGET_HANDLER_SLOTS, query.query_id)
            return ANSWER_BUSY
        async with slots:
            return await self._run_widget_query_handler(*route, query)

    async def _run_widget_query_handler(self, prefix: str, handler: WidgetQueryHandler, query: WidgetQuery) -> Optional[str]:
        """Run ``handler`` under a wall-clock deadline that also abandons a hung or cancellation-shielded
        handler; its answer must be text or None. Warnings name the prefix, never the untrusted data."""
        from plugins.platforms.telegram import adapter as facade

        try:
            result = await facade.run_bounded_async(
                handler(query), HANDLER_TIMEOUT_SECONDS, label="telegram-widget-query", dump_on_blocked_loop=False)
        except asyncio.CancelledError:
            task = asyncio.current_task()
            if task is not None and task.cancelling():
                raise  # this dispatch itself is being cancelled (shutdown)
            logger.warning("[%s] Widget query handler for %r was cancelled (query %s)",
                           self.name, prefix, query.query_id, exc_info=True)
            return ANSWER_FAILED
        except Exception:
            logger.warning("[%s] Widget query handler for %r failed (query %s)",
                           self.name, prefix, query.query_id, exc_info=True)
            return ANSWER_FAILED
        if result.timed_out:
            logger.warning("[%s] Widget query handler for %r timed out after %ss (query %s)",
                           self.name, prefix, HANDLER_TIMEOUT_SECONDS, query.query_id)
            return ANSWER_TIMEOUT
        if result.value is None or isinstance(result.value, str):
            return result.value
        logger.warning("[%s] Widget query handler for %r returned %s instead of text",
                       self.name, prefix, type(result.value).__name__)
        return ANSWER_FAILED

    async def _answer_widget_query(self, query: WidgetQuery, data: Optional[str]) -> None:
        """The one ``answerWidgetQuery`` for ``query``. Never retried or raised: Telegram refuses a late
        answer, and a lost one leaves the widget at TIMEOUT, which a second answer cannot fix."""
        from plugins.platforms.telegram.adapter import _await_with_thread_deadline, _redact_telegram_error_text

        payload: dict[str, Any] = {"widget_query_id": query.query_id}
        if data is not None:
            payload["data"] = truncate_utf8(data, DATA_MAX_BYTES)
            if len(payload["data"]) < len(data):
                logger.warning("[%s] Widget query %s answer cut to %d UTF-8 bytes", self.name, query.query_id, DATA_MAX_BYTES)
        try:
            await _await_with_thread_deadline(
                self._bot.do_api_request("answerWidgetQuery", api_kwargs=payload),
                timeout=ANSWER_TIMEOUT_SECONDS, label="telegram-widget-answer", dump_on_blocked_loop=False)
        except Exception as exc:
            if is_stale_query_error(exc):
                logger.debug("[%s] Widget query %s expired before its answer", self.name, query.query_id, exc_info=True)
                return
            logger.warning("[%s] answerWidgetQuery failed (not retried): %s",
                           self.name, _redact_telegram_error_text(exc), exc_info=True)

    # --- outbound --------------------------------------------------------------------------------------

    def _widgets_refusal(self) -> Optional[SendResult]:
        return None if getattr(self, "_widgets_enabled", False) else SendResult(success=False, error=WIDGETS_DISABLED)

    def _widget_route(
        self, chat_id: str, reply_to: Optional[str], metadata: Optional[dict[str, Any]]) -> tuple[dict[str, Any], Optional[str]]:
        """Routing kwargs exactly as the rich send path builds them (forum or DM topic, notification mode,
        reply anchor as ``reply_parameters``), or why the requested route can't carry a widget."""
        thread_id = self._metadata_thread_id(metadata)
        try:
            routing = self._compute_single_send_routing(chat_id, reply_to, metadata, thread_id)
        except ValueError:  # a non-numeric thread id in metadata
            return {}, f"invalid Telegram thread id {thread_id!r}"
        if routing is None:
            return {}, self._dm_topic_missing_anchor_error()
        reply_to_id, thread_kwargs = routing
        if thread_kwargs.get("direct_messages_topic_id") is not None:
            return {}, "sendWidget has no direct_messages_topic_id: widgets can't go to a direct-messages topic"
        route = _present(**thread_kwargs)
        route.update(self._notification_kwargs(metadata))
        if reply_to_id is not None:
            route["reply_parameters"] = {"message_id": reply_to_id}
        return route, None

    async def _widget_request(
        self, method: str, payload: dict[str, Any], *, edit: bool = False) -> tuple[Any, Optional[SendResult]]:
        """One Bot API request, never sent twice: ``(result, None)`` or ``(None, failure)``. An edit Telegram
        calls "not modified" is done."""
        from plugins.platforms.telegram.adapter import _TEXT_SEND_DEADLINE, _await_with_thread_deadline

        bot = getattr(self, "_bot", None)
        if bot is None:
            return None, SendResult(success=False, error="Not connected", retryable=True)
        try:
            result = await _await_with_thread_deadline(
                bot.do_api_request(method, api_kwargs=payload),
                timeout=_TEXT_SEND_DEADLINE, label="telegram-send", dump_on_blocked_loop=False)
        except Exception as exc:
            logger.debug("[%s] %s failed", self.name, method, exc_info=True)
            if edit and "not modified" in str(exc).lower():
                return True, None
            return None, self._widget_failure(method, exc)
        return result, None

    def _widget_failure(self, method: str, exc: Exception) -> SendResult:
        """The rich path's classification without its legacy fallback: a refused request (BadRequest, or the
        method unavailable) fails for good; anything else may have reached Telegram and is never re-sent."""
        from plugins.platforms.telegram.adapter import _redact_telegram_error_text

        safe_error = _redact_telegram_error_text(exc)
        if self._is_bad_request_error(exc) or self._is_rich_capability_error(exc):
            logger.warning("[%s] %s refused: %s", self.name, method, safe_error)
            return SendResult(success=False, error=safe_error)
        logger.warning("[%s] %s failed (not re-sent): %s", self.name, method, safe_error)
        return maybe_delivered_result(exc, safe_error, retry_after=getattr(exc, "retry_after", None))

    async def send_widget(
        self, chat_id: str, html: str, *, reply_to: Optional[str] = None, metadata: Optional[dict[str, Any]] = None,
        caption: Optional[str] = None, chat_data: Optional[str] = None, public_data: Optional[str] = None,
        allow_queries: bool = False, is_shared: bool = False, width: Optional[int] = None,
        height: Optional[int] = None) -> SendResult:
        """Send an HTML widget (``sendWidget``); the HTML can never change afterwards, only its state. Routing
        follows ``send()`` (``metadata`` thread / DM topic, ``reply_to``). Never raises."""
        if (refusal := self._widgets_refusal()) is not None:
            return refusal
        error = _first_error(
            _html_error(html), _data_error("chat_data", chat_data), _data_error("public_data", public_data),
            _int_error("width", width, 0), _int_error("height", height, 0))
        route, route_error = self._widget_route(chat_id, reply_to, metadata) if error is None else ({}, None)
        if error or route_error:
            return SendResult(success=False, error=error or route_error)
        payload = {
            "chat_id": normalize_telegram_chat_id(chat_id), "html": html, **route,
            **_present(caption=caption, chat_data=chat_data, public_data=public_data, width=width, height=height,
                       allow_queries=True if allow_queries else None, is_shared=True if is_shared else None)}
        message, failure = await self._widget_request("sendWidget", payload)
        return failure or _sent_widget_result(message)

    async def send_shared_widget(
        self, chat_id: str, widget_id: str, *, reply_to: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None, caption: Optional[str] = None,
        chat_data: Optional[str] = None) -> SendResult:
        """Send an existing shared widget again (``sendSharedWidget``), with this message's own ``chat_data``."""
        if (refusal := self._widgets_refusal()) is not None:
            return refusal
        error = _first_error(_widget_id_error(widget_id), _data_error("chat_data", chat_data))
        route, route_error = self._widget_route(chat_id, reply_to, metadata) if error is None else ({}, None)
        if error or route_error:
            return SendResult(success=False, error=error or route_error)
        payload = {"chat_id": normalize_telegram_chat_id(chat_id), "widget_id": widget_id, **route,
                   **_present(caption=caption, chat_data=chat_data)}
        message, failure = await self._widget_request("sendSharedWidget", payload)
        return failure or _sent_widget_result(message, widget_id)

    async def edit_widget_data(
        self, chat_id: str, message_id: str, *, chat_data: Optional[str] = None,
        public_data: Optional[str] = None) -> SendResult:
        """Change a sent widget's state (``editMessageWidgetData``): None leaves a field as it is, "" clears
        it. An edit that changes nothing counts as done."""
        if (refusal := self._widgets_refusal()) is not None:
            return refusal
        parsed_id = parse_telegram_message_id(message_id)
        error = _first_error(
            None if parsed_id is not None else "message_id must be a positive integer",
            None if (chat_data, public_data) != (None, None) else "nothing to edit: pass chat_data and/or public_data",
            _data_error("chat_data", chat_data), _data_error("public_data", public_data))
        if error:
            return SendResult(success=False, error=error)
        payload = {"chat_id": normalize_telegram_chat_id(chat_id), "message_id": parsed_id,
                   **_present(chat_data=chat_data, public_data=public_data)}
        _, failure = await self._widget_request("editMessageWidgetData", payload, edit=True)
        return failure or SendResult(success=True, message_id=str(parsed_id))

    async def edit_shared_widget_data(
        self, widget_id: str, public_data: str, *, cache_time: Optional[int] = None) -> SendResult:
        """Change a shared widget's public state wherever it is shown (``editSharedWidgetData``, at most once
        per second per widget); "" clears it, ``cache_time`` (5-3600 s) bounds client-side caching."""
        if (refusal := self._widgets_refusal()) is not None:
            return refusal
        error = _first_error(
            _widget_id_error(widget_id),
            _data_error("public_data", public_data) if public_data is not None else "public_data is required",
            _int_error("cache_time", cache_time, *CACHE_TIME_SECONDS))
        if error:
            return SendResult(success=False, error=error)
        payload = {"widget_id": widget_id, "public_data": public_data, **_present(cache_time=cache_time)}
        _, failure = await self._widget_request("editSharedWidgetData", payload, edit=True)
        return failure or SendResult(success=True)

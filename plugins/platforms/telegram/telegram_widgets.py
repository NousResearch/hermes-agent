"""Bot API 10.4 Telegram HTML widgets: send/edit data and answer widget queries.

A widget is fixed HTML plus bot-written state. Telegram renders it in a sandboxed QuickJS
engine with no network access; the HTML is never executed on Hermes. This transport only
validates the documented Bot API bounds and forwards the exact HTML (no local sanitizing:
the Telegram sandbox owns that profile).

Production supports widgets only in the private chat with the bot. Sends therefore require
a positive numeric chat id and, when metadata carries it, a private/dm chat type. Widgets are
never degraded to text on a failure: that would silently drop the interactive UI. Permanent
refusals return a non-retryable ``SendResult``; timeouts and lost responses return an
ambiguous non-retryable result (the message may have landed, so it must never be resent in
another shape); flood waits use the adapter's canonical ``flood_control:<seconds>`` result.

No arbitrary Bot API proxy, no paid broadcast (``allow_paid_broadcast``), and no wallet
operations are exposed here.
"""

from __future__ import annotations

import logging
import re
import zlib
from typing import Any, Dict, Optional

from gateway.platforms.base import SendResult, utf16_len
from plugins.platforms.telegram.telegram_ids import parse_telegram_message_id

logger = logging.getLogger(__name__)

_WIDGET_SEND_DEADLINE = 30.0

# Official Bot API limits (core.telegram.org/bots/api/methods/sendwidget.md and /bots/widgets.md).
_WIDGET_MAX_HTML_BYTES = 32768
_WIDGET_MAX_COMPRESSED_BYTES = 4096
_WIDGET_MAX_DATA_BYTES = 1024
_WIDGET_MAX_CAPTION_UTF16 = 1024
_WIDGET_WIDTH_MIN = 200  # bubble width range documented in /bots/widgets.md
_WIDGET_WIDTH_MAX = 600
_WIDGET_HEIGHT_MIN = 50
_WIDGET_HEIGHT_MAX = 600
_WIDGET_CACHE_TIME_MIN = 5
_WIDGET_CACHE_TIME_MAX = 3600

# The exact spec keys this transport accepts; anything else is rejected before a network call.
_SEND_SPEC_KEYS = frozenset({
    "html", "caption", "width", "height", "public_data", "chat_data",
    "allow_queries", "is_shared", "show_caption_above_media",
})
_SHARED_SPEC_KEYS = frozenset({"widget_id", "caption", "chat_data", "show_caption_above_media"})
_EDIT_CHANGE_KEYS = frozenset({"public_data", "chat_data"})
_PRIVATE_CHAT_TYPES = frozenset({"private", "dm"})
_RETRY_AFTER_RE = re.compile(r"retry\s+(?:in\s+)?(\d+)", re.IGNORECASE)


class WidgetValidationError(ValueError):
    """A caller-supplied widget spec violates a documented Bot API bound."""


def _utf8_len(value: str, what: str = "text") -> int:
    try:
        return len(value.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise WidgetValidationError(f"{what} must be valid UTF-8 text") from exc


def _require_mapping(value: Any, what: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise WidgetValidationError(f"{what} must be a dict")
    return value


def _reject_unknown_keys(spec: Dict[str, Any], allowed: frozenset, what: str) -> None:
    unknown = sorted(set(spec) - allowed)
    if unknown:
        raise WidgetValidationError(
            f"{what} has unsupported key(s): {', '.join(unknown)}; allowed: {', '.join(sorted(allowed))}")


def _validate_chat(chat_id: Any, metadata: Optional[Dict[str, Any]]) -> str:
    """Widgets are production-private-only: a positive numeric user chat id and private/dm metadata."""
    if metadata is not None and not isinstance(metadata, dict):
        raise WidgetValidationError("metadata must be a dict or None")
    raw = str(chat_id).strip() if chat_id is not None else ""
    if not raw.isascii() or not raw.isdigit() or int(raw) <= 0:
        raise WidgetValidationError(
            "widgets are available only in private chats and require a positive numeric chat_id; "
            f"got {chat_id!r}")
    chat_type = (metadata or {}).get("chat_type")
    if chat_type is not None and str(chat_type).strip().lower() not in _PRIVATE_CHAT_TYPES:
        raise WidgetValidationError(
            "widgets are available only in private chats; "
            f"chat_type={chat_type!r} is not private/dm")
    return raw


def _validate_text(value: Any, what: str, *, max_bytes: Optional[int] = None,
                   max_utf16: Optional[int] = None, allow_empty: bool = True) -> str:
    if not isinstance(value, str):
        raise WidgetValidationError(f"{what} must be a string")
    if not allow_empty and not value.strip():
        raise WidgetValidationError(f"{what} is required and must not be empty")
    size = _utf8_len(value, what)
    if max_bytes is not None and size > max_bytes:
        raise WidgetValidationError(
            f"{what} is {size} UTF-8 bytes; Telegram allows at most {max_bytes}")
    if max_utf16 is not None and utf16_len(value) > max_utf16:
        raise WidgetValidationError(
            f"{what} is {utf16_len(value)} UTF-16 units; Telegram allows at most {max_utf16}")
    return value


def _validate_html(value: Any) -> str:
    html = _validate_text(value, "html", max_bytes=_WIDGET_MAX_HTML_BYTES, allow_empty=False)
    if "\x00" in html:
        raise WidgetValidationError("html must not contain the NUL character (U+0000)")
    compressed = zlib.compress(html.encode("utf-8"), 9)
    if len(compressed) > _WIDGET_MAX_COMPRESSED_BYTES:
        raise WidgetValidationError(
            f"html is {len(compressed)} bytes after zlib level-9 compression; "
            f"Telegram allows at most {_WIDGET_MAX_COMPRESSED_BYTES}")
    return html


def _validate_bool(value: Any, what: str) -> bool:
    if not isinstance(value, bool):
        raise WidgetValidationError(f"{what} must be a boolean")
    return value


def _validate_int(value: Any, what: str, *, allow_zero: bool,
                  minimum: int, maximum: Optional[int] = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise WidgetValidationError(f"{what} must be an integer")
    if value == 0 and allow_zero:
        return value
    if value < minimum or (maximum is not None and value > maximum):
        bound = f"{minimum}" if maximum is None else f"{minimum}-{maximum}"
        zero = " or 0 for automatic" if allow_zero else ""
        raise WidgetValidationError(f"{what} must be {bound}{zero}; got {value}")
    return value


def _validate_data(value: Any, what: str) -> str:
    return _validate_text(value, what, max_bytes=_WIDGET_MAX_DATA_BYTES)


def _validate_caption(value: Any) -> str:
    # Caption is the chat-list/notification/search fallback for clients without widgets.
    return _validate_text(
        value, "caption", max_utf16=_WIDGET_MAX_CAPTION_UTF16, allow_empty=False)


def _validate_widget_id(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WidgetValidationError("widget_id must be a non-empty string")
    return value


def _validate_message_id(value: Any) -> int:
    parsed = parse_telegram_message_id(value)
    if parsed is None:
        raise WidgetValidationError(f"message_id must be a positive integer; got {value!r}")
    return parsed


def _send_spec(spec: Any) -> Dict[str, Any]:
    spec = _require_mapping(spec, "spec")
    _reject_unknown_keys(spec, _SEND_SPEC_KEYS, "spec")
    if "html" not in spec:
        raise WidgetValidationError("spec.html is required")
    if "caption" not in spec:
        raise WidgetValidationError(
            "spec.caption is required: the caption is the fallback shown by clients "
            "without widget support")
    out: Dict[str, Any] = {"html": _validate_html(spec["html"]), "caption": _validate_caption(spec["caption"])}
    if "width" in spec:
        out["width"] = _validate_int(
            spec["width"], "width", allow_zero=True, minimum=_WIDGET_WIDTH_MIN,
            maximum=_WIDGET_WIDTH_MAX)
    if "height" in spec:
        out["height"] = _validate_int(
            spec["height"], "height", allow_zero=True, minimum=_WIDGET_HEIGHT_MIN,
            maximum=_WIDGET_HEIGHT_MAX)
    if "public_data" in spec:
        out["public_data"] = _validate_data(spec["public_data"], "public_data")
    if "chat_data" in spec:
        out["chat_data"] = _validate_data(spec["chat_data"], "chat_data")
    for flag in ("allow_queries", "is_shared", "show_caption_above_media"):
        if flag in spec:
            out[flag] = _validate_bool(spec[flag], flag)
    return out


def _shared_spec(spec: Any) -> Dict[str, Any]:
    spec = _require_mapping(spec, "spec")
    _reject_unknown_keys(spec, _SHARED_SPEC_KEYS, "spec")
    if "widget_id" not in spec:
        raise WidgetValidationError("spec.widget_id is required")
    if "caption" not in spec:
        raise WidgetValidationError(
            "spec.caption is required: the caption is the fallback shown by clients "
            "without widget support")
    out: Dict[str, Any] = {
        "widget_id": _validate_widget_id(spec["widget_id"]),
        "caption": _validate_caption(spec["caption"]),
    }
    if "chat_data" in spec:
        out["chat_data"] = _validate_data(spec["chat_data"], "chat_data")
    if "show_caption_above_media" in spec:
        out["show_caption_above_media"] = _validate_bool(
            spec["show_caption_above_media"], "show_caption_above_media")
    return out


def _edit_changes(changes: Any) -> Dict[str, str]:
    changes = _require_mapping(changes, "changes")
    _reject_unknown_keys(changes, _EDIT_CHANGE_KEYS, "changes")
    if not changes:
        raise WidgetValidationError("changes must contain public_data and/or chat_data")
    # Missing key = leave as is; present empty string = clear (the two cases are distinct).
    return {key: _validate_data(value, key) for key, value in changes.items()}


def _cache_time(value: Any) -> int:
    return _validate_int(
        value, "cache_time", allow_zero=False, minimum=_WIDGET_CACHE_TIME_MIN,
        maximum=_WIDGET_CACHE_TIME_MAX)


class TelegramWidgetsMixin:
    """Mixin for ``TelegramAdapter`` implementing the Bot API 10.4 widget methods."""

    # -- validation/payload helpers -----------------------------------------------------

    def _widget_thread_kwargs(self, chat_id: str, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Forum/DM-topic routing for a widget send, via the adapter's single route helper.

        ``sendWidget``/``sendSharedWidget`` document only ``message_thread_id``; a
        ``direct_messages_topic_id`` route is not forwarded as an unknown parameter.
        """
        metadata = metadata or {}
        thread_id = self._metadata_thread_id(metadata) if hasattr(self, "_metadata_thread_id") else None
        routing = (self._thread_kwargs_for_send(chat_id, thread_id, metadata)
                   if hasattr(self, "_thread_kwargs_for_send") else {})
        thread = (routing or {}).get("message_thread_id")
        if thread is None:
            return {}
        try:
            return {"message_thread_id": int(thread)}
        except (TypeError, ValueError) as exc:
            raise WidgetValidationError(f"invalid message_thread_id: {thread!r}") from exc

    def _widget_common_payload(self, chat_id: str, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"chat_id": chat_id, **self._widget_thread_kwargs(chat_id, metadata)}
        notify = self._notification_kwargs(metadata) if hasattr(self, "_notification_kwargs") else {}
        if "disable_notification" in (notify or {}):
            payload["disable_notification"] = bool(notify["disable_notification"])
        return payload

    def _widget_send_payload(self, chat_id: str, spec: Any,
                             metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = self._widget_common_payload(chat_id, metadata)
        payload.update(_send_spec(spec))
        return payload

    def _shared_widget_send_payload(self, chat_id: str, spec: Any,
                                    metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = self._widget_common_payload(chat_id, metadata)
        payload.update(_shared_spec(spec))
        return payload

    # -- transport ----------------------------------------------------------------------

    async def _call_widget_api(self, method: str, payload: Dict[str, Any]) -> Any:
        from plugins.platforms.telegram.adapter import _await_with_thread_deadline

        return await _await_with_thread_deadline(
            self._bot.do_api_request(method, api_kwargs=payload),
            timeout=_WIDGET_SEND_DEADLINE, label="telegram-widget", dump_on_blocked_loop=False)

    @staticmethod
    def _widget_validation_result(exc: WidgetValidationError) -> SendResult:
        logger.warning("Rejected Telegram widget request: %s", exc)
        return SendResult(success=False, error=str(exc), retryable=False,
                          raw_response={"validation_error": True})

    def _widget_failure_result(self, method: str, exc: Exception, chat_id: Optional[str] = None) -> SendResult:
        """Map a failed widget call onto the adapter's send semantics."""
        from plugins.platforms.telegram.adapter import _redact_telegram_error_text
        from plugins.platforms.telegram.telegram_send_errors import (
            LOST, REFUSED, LOCAL, send_error_kind)

        safe_error = _redact_telegram_error_text(exc)
        retry_after = getattr(exc, "retry_after", None)
        if retry_after is None:
            match = _RETRY_AFTER_RE.search(str(exc).lower())
            retry_after = float(match.group(1)) if match else None
        if retry_after is not None:
            wait = float(retry_after)
            if chat_id is not None:
                try:
                    from plugins.platforms.telegram.telegram_flood_window import note_flood
                    note_flood(self, chat_id, exc)
                except Exception:
                    logger.debug("Could not arm the Telegram flood window for a widget send", exc_info=True)
            logger.warning("[%s] %s flood control, failing closed (retry_after %.1fs)",
                           getattr(self, "name", "Telegram"), method, wait)
            return SendResult(success=False, error=f"flood_control:{wait}", retry_after=wait)

        kind = send_error_kind(exc)
        from gateway.platforms.base import classify_send_error
        generic_kind = classify_send_error(exc)
        logger.warning("[%s] %s failed (%s): %s",
                       getattr(self, "name", "Telegram"), method, kind, safe_error)
        raw: Dict[str, Any] = {"error_kind": kind}
        if kind == LOST:
            raw["delivery_uncertain"] = True
            return SendResult(success=False, error=f"{method} failed: {safe_error}",
                              retryable=False, error_kind=generic_kind, raw_response=raw)
        # A refusal/local error is permanent; an unsent request or Telegram's own 5xx is retryable.
        retryable = kind not in (REFUSED, LOCAL)
        return SendResult(success=False, error=f"{method} failed: {safe_error}",
                          retryable=retryable, error_kind=generic_kind, raw_response=raw)

    @staticmethod
    def _message_from_result(result: Any) -> Any:
        if isinstance(result, dict):
            nested = result.get("result")
            return nested if isinstance(nested, dict) else result
        return result

    @staticmethod
    def _message_field(message: Any, name: str) -> Any:
        if isinstance(message, dict):
            return message.get(name)
        return getattr(message, name, None)

    @staticmethod
    def _message_dict(message: Any) -> Any:
        """The raw Message as a dict when PTB gives us a model object; else unchanged."""
        to_dict = getattr(message, "to_dict", None)
        return to_dict() if callable(to_dict) else message

    def _widget_sent_result(self, result: Any) -> SendResult:
        message = self._message_from_result(result)
        message_id = self._message_field(message, "message_id")
        widget = self._message_field(message, "widget")
        if widget is None and isinstance(result, dict):
            widget = result.get("widget")
        message = self._message_dict(message)
        widget = self._message_dict(widget)
        # Parent UI consumers read raw_response["message"] (widget nested) or ["widget"].
        raw: Dict[str, Any] = {"message_id": message_id, "message": message}
        if widget is not None:
            raw["widget"] = widget  # keeps id/public_data/chat_data exactly as Telegram returned them
            widget_id = self._message_field(widget, "id")
            if widget_id is not None:
                raw["widget_id"] = str(widget_id)
        return SendResult(success=True,
                          message_id=str(message_id) if message_id is not None else None,
                          raw_response=raw)

    async def send_widget(self, chat_id: str, spec: dict,
                          metadata: Optional[dict] = None) -> SendResult:
        """Send a new HTML widget (``sendWidget``). Private chats only."""
        try:
            chat = _validate_chat(chat_id, metadata)
            payload = self._widget_send_payload(chat, spec, metadata)
        except WidgetValidationError as exc:
            return self._widget_validation_result(exc)
        try:
            result = await self._call_widget_api("sendWidget", payload)
        except Exception as exc:
            return self._widget_failure_result("sendWidget", exc, chat)
        # Never fall back to text: the interactive widget would be lost.
        return self._widget_sent_result(result)

    async def send_shared_widget(self, chat_id: str, spec: dict,
                                 metadata: Optional[dict] = None) -> SendResult:
        """Send an existing shared widget again by ``widget_id`` (``sendSharedWidget``)."""
        try:
            chat = _validate_chat(chat_id, metadata)
            payload = self._shared_widget_send_payload(chat, spec, metadata)
        except WidgetValidationError as exc:
            return self._widget_validation_result(exc)
        try:
            result = await self._call_widget_api("sendSharedWidget", payload)
        except Exception as exc:
            return self._widget_failure_result("sendSharedWidget", exc, chat)
        return self._widget_sent_result(result)

    async def edit_widget_data(self, chat_id: str, message_id: Any,
                               changes: dict) -> SendResult:
        """Change one message's widget state (``editMessageWidgetData``)."""
        try:
            chat = _validate_chat(chat_id, None)
            mid = _validate_message_id(message_id)
            payload = {"chat_id": chat, "message_id": mid, **_edit_changes(changes)}
        except WidgetValidationError as exc:
            return self._widget_validation_result(exc)
        try:
            result = await self._call_widget_api("editMessageWidgetData", payload)
        except Exception as exc:
            if "not modified" in str(exc).lower():
                return SendResult(success=True, message_id=str(mid))
            return self._widget_failure_result("editMessageWidgetData", exc, chat)
        return SendResult(success=True, message_id=str(mid),
                          raw_response={"result": result})

    async def edit_shared_widget_data(self, widget_id: str, public_data: str,
                                      cache_time: Optional[int] = None) -> SendResult:
        """Change a shared widget's public state (``editSharedWidgetData``)."""
        try:
            payload: Dict[str, Any] = {
                "widget_id": _validate_widget_id(widget_id),
                "public_data": _validate_data(public_data, "public_data"),
            }
            if cache_time is not None:
                payload["cache_time"] = _cache_time(cache_time)
        except WidgetValidationError as exc:
            return self._widget_validation_result(exc)
        try:
            result = await self._call_widget_api("editSharedWidgetData", payload)
        except Exception as exc:
            if "not modified" in str(exc).lower():
                return SendResult(success=True, raw_response={"result": True})
            return self._widget_failure_result("editSharedWidgetData", exc)
        return SendResult(success=True, raw_response={"result": result})

    async def answer_widget_query(self, query_id: str, data: Optional[str] = None) -> bool:
        """Answer a widget request (``answerWidgetQuery``); must be sent within 15 seconds."""
        try:
            if not isinstance(query_id, str) or not query_id.strip():
                raise WidgetValidationError("query_id must be a non-empty string")
            payload: Dict[str, Any] = {"widget_query_id": query_id}
            if data is not None:
                payload["data"] = _validate_data(data, "data")
        except WidgetValidationError as exc:
            logger.warning("Rejected Telegram widget query answer: %s", exc)
            return False
        try:
            result = await self._call_widget_api("answerWidgetQuery", payload)
        except Exception as exc:
            # A query answer has no message to resend; report the failure instead of retrying.
            self._widget_failure_result("answerWidgetQuery", exc)
            return False
        if isinstance(result, dict) and "result" in result:
            return bool(result["result"])
        return bool(result)

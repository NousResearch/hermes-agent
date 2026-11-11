"""Outgoing Bot API 10.4 rich quotes: build and send ``ReplyParameters.quote_rich_texts``.

A rich quote selects one or more *quoted texts* of a rich message (paragraphs, headings,
preformatted blocks, table cells, captions, ...), top-level one entry per selection and in the
order they appear in the source message. Formatting inside each selection is kept and nested
exactly as Telegram sent it, so the reply renders the same quotation bubble.

The transport is a single ``sendRichMessage`` request: the rich body, reply parameters, topic
routing and notification mode all travel together. A rejected or ambiguous request is never
re-sent without the quote and never downgraded to another format — the caller gets a typed
``SendResult`` and the parent may report it.

Official references:
- https://core.telegram.org/bots/api/types/replyparameters.md
- https://core.telegram.org/bots/api/types/richtext.md
- https://core.telegram.org/bots/api/types/richtextbold.md (and the other allowed types)
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Optional

from gateway.platforms.base import SendResult, classify_send_error

logger = logging.getLogger(__name__)

__all__ = [
    "MAX_QUOTE_TEXTS",
    "MAX_QUOTE_RENDERED_LENGTH",
    "RichQuoteError",
    "build_rich_reply_parameters",
    "rich_quote_plain_text",
    "send_rich_reply",
]

# ReplyParameters: "Up to 32 quoted texts with total length up to 1024 characters can be
# specified." Telegram measures message text in UTF-16 code units (the same unit as
# ``quote_position``), so the rendered length below uses that measure.
MAX_QUOTE_TEXTS = 32
MAX_QUOTE_RENDERED_LENGTH = 1024

# A hostile payload must not be able to blow the recursion limit before validation finishes.
_MAX_QUOTE_DEPTH = 32
_MAX_QUOTE_NODES = 4096

# ReplyParameters lists exactly which richer types survive in a quote; every other rich text
# type is replaced with its text by Telegram, so sending one would silently change the quote.
_TEXT_WRAPPER_TYPES = frozenset({
    "bold", "italic", "underline", "strikethrough", "spoiler", "subscript", "superscript",
})
_LEAF_QUOTE_TYPES = frozenset({
    "date_time", "custom_emoji", "mathematical_expression",
})
_ALLOWED_QUOTE_TYPES = _TEXT_WRAPPER_TYPES | _LEAF_QUOTE_TYPES


class RichQuoteError(ValueError):
    """A ``quote_rich_texts`` / reply-parameter payload Telegram would reject or silently alter."""


class _ValidationBudget:
    """Shared recursion/node counter for one ``quote_rich_texts`` array."""

    __slots__ = ("nodes",)

    def __init__(self) -> None:
        self.nodes = 0


def _validate_message_id(value: Any, *, field: str = "message_id", allow_zero: bool = False) -> int:
    """A Bot API Integer id: an int (never bool) or an ASCII digit string; ``> 0``."""
    if isinstance(value, bool):
        raise RichQuoteError(f"{field} must be an integer, not a bool")
    if isinstance(value, int):
        number = value
    elif isinstance(value, str):
        text = value.strip()
        if not text or not text.isascii() or not text.isdigit():
            raise RichQuoteError(
                f"{field} must be a {'non-negative' if allow_zero else 'positive'} integer")
        number = int(text)
    else:
        raise RichQuoteError(f"{field} must be an integer")
    if number < 0 or (number == 0 and not allow_zero):
        raise RichQuoteError(f"{field} must be {'non-negative' if allow_zero else 'positive'}")
    return number


def _validate_string(value: Any, *, field: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise RichQuoteError(f"{field} must be a string")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise RichQuoteError(f"{field} contains an unpaired surrogate") from exc
    if not allow_empty and not value:
        raise RichQuoteError(f"{field} must not be empty")
    return value


def _validate_rich_text(value: Any, *, depth: int, budget: _ValidationBudget, where: str) -> Any:
    """Validate and copy one RichText node without flattening or reordering it."""
    if depth > _MAX_QUOTE_DEPTH:
        raise RichQuoteError(f"{where} is nested too deeply (max {_MAX_QUOTE_DEPTH})")
    budget.nodes += 1
    if budget.nodes > _MAX_QUOTE_NODES:
        raise RichQuoteError(f"quote_rich_texts exceeds {_MAX_QUOTE_NODES} rich-text nodes")

    if isinstance(value, str):
        return _validate_string(value, field=where, allow_empty=True)
    if isinstance(value, list):
        if not value:
            raise RichQuoteError(f"{where} must not be an empty rich-text array")
        return [
            _validate_rich_text(node, depth=depth + 1, budget=budget, where=f"{where}[{index}]")
            for index, node in enumerate(value)
        ]
    if not isinstance(value, Mapping):
        raise RichQuoteError(f"{where} must be a string, an array, or a rich-text object")

    if any(not isinstance(key, str) for key in value):
        raise RichQuoteError(f"{where} has a non-string field name")
    rich_type = value.get("type")
    if not isinstance(rich_type, str) or not rich_type:
        raise RichQuoteError(f"{where}.type must be a string")
    if rich_type not in _ALLOWED_QUOTE_TYPES:
        allowed = ", ".join(sorted(_ALLOWED_QUOTE_TYPES))
        raise RichQuoteError(f"{where}.type {rich_type!r} cannot be quoted (allowed: plain text, arrays, {allowed})")

    if rich_type in _TEXT_WRAPPER_TYPES:
        expected = {"type", "text"}
        extra = set(value) - expected
        missing = expected - set(value)
        if missing:
            raise RichQuoteError(f"{where}.{sorted(missing)[0]} is required for {rich_type}")
        if extra:
            raise RichQuoteError(f"{where} has unsupported field(s) for {rich_type}: {', '.join(sorted(extra))}")
        return {
            "type": rich_type,
            "text": _validate_rich_text(value["text"], depth=depth + 1, budget=budget, where=f"{where}.text"),
        }

    if rich_type == "date_time":
        expected = {"type", "text", "unix_time", "date_time_format"}
        extra = set(value) - expected
        missing = expected - set(value)
        if missing:
            raise RichQuoteError(f"{where}.{sorted(missing)[0]} is required for date_time")
        if extra:
            raise RichQuoteError(f"{where} has unsupported field(s) for date_time: {', '.join(sorted(extra))}")
        unix_time = value["unix_time"]
        if isinstance(unix_time, bool) or not isinstance(unix_time, int):
            raise RichQuoteError(f"{where}.unix_time must be an integer")
        date_time_format = _validate_string(value["date_time_format"], field=f"{where}.date_time_format")
        return {
            "type": rich_type,
            "text": _validate_rich_text(value["text"], depth=depth + 1, budget=budget, where=f"{where}.text"),
            "unix_time": unix_time,
            "date_time_format": date_time_format,
        }

    if rich_type == "custom_emoji":
        expected = {"type", "custom_emoji_id", "alternative_text"}
        extra = set(value) - expected
        missing = expected - set(value)
        if missing:
            raise RichQuoteError(f"{where}.{sorted(missing)[0]} is required for custom_emoji")
        if extra:
            raise RichQuoteError(f"{where} has unsupported field(s) for custom_emoji: {', '.join(sorted(extra))}")
        emoji_id = _validate_string(value["custom_emoji_id"], field=f"{where}.custom_emoji_id")
        if not emoji_id.isascii() or not emoji_id.isdigit() or int(emoji_id) <= 0:
            raise RichQuoteError(f"{where}.custom_emoji_id must be a positive numeric string")
        return {
            "type": rich_type,
            "custom_emoji_id": emoji_id,
            "alternative_text": _validate_string(value["alternative_text"], field=f"{where}.alternative_text"),
        }

    # mathematical_expression
    expected = {"type", "expression"}
    extra = set(value) - expected
    missing = expected - set(value)
    if missing:
        raise RichQuoteError(f"{where}.{sorted(missing)[0]} is required for mathematical_expression")
    if extra:
        raise RichQuoteError(f"{where} has unsupported field(s) for mathematical_expression: {', '.join(sorted(extra))}")
    return {
        "type": rich_type,
        "expression": _validate_string(value["expression"], field=f"{where}.expression"),
    }


def _plain_text(value: Any) -> str:
    """Rendered plain text of an already-validated RichText node."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(_plain_text(node) for node in value)
    rich_type = value["type"]
    if rich_type in _TEXT_WRAPPER_TYPES or rich_type == "date_time":
        return _plain_text(value["text"])
    if rich_type == "custom_emoji":
        return value["alternative_text"]
    return value["expression"]


def _utf16_length(text: str) -> int:
    """Telegram's text measure: UTF-16 code units."""
    return len(text.encode("utf-16-le")) // 2


def _validate_quote_texts(quote_rich_texts: Any) -> list[Any]:
    """Validate the top-level quoted-parts array; returns a structure-preserving copy."""
    if not isinstance(quote_rich_texts, list):
        raise RichQuoteError("quote_rich_texts must be an array of quoted texts")
    if not 1 <= len(quote_rich_texts) <= MAX_QUOTE_TEXTS:
        raise RichQuoteError(
            f"quote_rich_texts must contain between 1 and {MAX_QUOTE_TEXTS} quoted texts")
    budget = _ValidationBudget()
    validated = [
        _validate_rich_text(node, depth=0, budget=budget, where=f"quote_rich_texts[{index}]")
        for index, node in enumerate(quote_rich_texts)
    ]
    for index, node in enumerate(validated):
        if not _plain_text(node).strip():
            raise RichQuoteError(f"quote_rich_texts[{index}] contains no visible text")
    total = sum(_utf16_length(_plain_text(node)) for node in validated)
    if total > MAX_QUOTE_RENDERED_LENGTH:
        raise RichQuoteError(
            f"quote_rich_texts renders {total} characters; the maximum is {MAX_QUOTE_RENDERED_LENGTH}")
    return validated


def build_rich_reply_parameters(
    message_id: Any,
    quote_rich_texts: Any,
    quote_position: Optional[Any] = None,
) -> dict[str, Any]:
    """ReplyParameters for a rich quote, ready to pass as ``sendRichMessage.reply_parameters``.

    ``quote_rich_texts`` is the top-level list of separately quoted texts in source order;
    each entry is a plain string, an array of RichText nodes, or one of the allowed rich-text
    objects (bold, italic, underline, strikethrough, spoiler, subscript, superscript,
    date_time, custom_emoji, mathematical_expression). Formatting and nesting are preserved
    exactly. ``quote_position`` is the UTF-16 offset in the source message when supplied.
    """
    reply: dict[str, Any] = {
        "message_id": _validate_message_id(message_id),
        "quote_rich_texts": _validate_quote_texts(quote_rich_texts),
    }
    if quote_position is not None:
        reply["quote_position"] = _validate_message_id(quote_position, field="quote_position", allow_zero=True)
    return reply


def rich_quote_plain_text(quote_rich_texts: Any) -> str:
    """Rendered plain text of validated ``quote_rich_texts`` (segments joined by line feeds).

    Telegram forms the text used for ``quote_position`` by joining all quotable texts with
    line feeds; nested arrays concatenate without a separator.
    """
    return "\n".join(_plain_text(node) for node in _validate_quote_texts(quote_rich_texts))


def _validated_reply_parameters(reply_parameters: Any) -> dict[str, Any]:
    """Re-validate the helper's own ReplyParameters shape before it reaches the Bot API."""
    if not isinstance(reply_parameters, Mapping):
        raise RichQuoteError("reply_parameters must be an object")
    unknown = set(reply_parameters) - {"message_id", "quote_rich_texts", "quote_position"}
    if unknown:
        raise RichQuoteError(f"reply_parameters has unsupported field(s): {', '.join(sorted(map(str, unknown)))}")
    if "message_id" not in reply_parameters:
        raise RichQuoteError("reply_parameters.message_id is required")
    if "quote_rich_texts" not in reply_parameters:
        raise RichQuoteError("reply_parameters.quote_rich_texts is required for a rich quote")
    return build_rich_reply_parameters(
        reply_parameters["message_id"],
        reply_parameters["quote_rich_texts"],
        reply_parameters.get("quote_position"),
    )


def _message_id_from_response(message: Any) -> Optional[int]:
    if isinstance(message, Mapping):
        message_id = message.get("message_id")
        if message_id is None:
            result = message.get("result")
            if isinstance(result, Mapping):
                message_id = result.get("message_id")
        return message_id if isinstance(message_id, int) and not isinstance(message_id, bool) else None
    message_id = getattr(message, "message_id", None)
    return message_id if isinstance(message_id, int) and not isinstance(message_id, bool) else None


def _failure(
    error: str,
    *,
    exc: Optional[BaseException] = None,
    retryable: bool = False,
    retry_after: Optional[float] = None,
    raw_response: Any = None,
) -> SendResult:
    return SendResult(
        success=False,
        error=error,
        retryable=retryable,
        retry_after=retry_after,
        error_kind=classify_send_error(exc, error),
        raw_response=raw_response,
    )


async def send_rich_reply(
    adapter: Any,
    chat_id: Any,
    content: str,
    reply_parameters: Any,
    metadata: Optional[dict[str, Any]] = None,
) -> SendResult:
    """Send one rich reply carrying ``reply_parameters`` (which MUST include the rich quote).

    The request is a single ``sendRichMessage`` with the adapter's rich payload, current topic
    routing, notification policy and flood gating. There is deliberately no plain-text or
    quote-less retry: a permanent rejection and a possibly-delivered failure both return
    ``SendResult(success=False, retryable=False)`` so the caller can never silently drop the
    quote or change the formatting on a retry.
    """
    # Lazy adapter imports: ``adapter`` imports platform mixins that may import this module.
    from plugins.platforms.telegram.adapter import (
        _TEXT_SEND_DEADLINE, _await_with_thread_deadline, _flood_cap_result,
        _redact_telegram_error_text, normalize_telegram_chat_id)
    from plugins.platforms.telegram import telegram_flood_window as _flood_window

    try:
        parameters = _validated_reply_parameters(reply_parameters)
    except RichQuoteError as exc:
        return _failure(str(exc))
    if not isinstance(content, str) or not content.strip():
        return _failure("rich reply content must not be empty")

    bot = getattr(adapter, "_bot", None)
    if bot is None:
        return _failure("Not connected")
    if isinstance(chat_id, bool) or chat_id is None or (
            isinstance(chat_id, str) and not chat_id.strip()):
        return _failure("chat_id is required")
    chat_id_norm = normalize_telegram_chat_id(chat_id)
    if not chat_id_norm:
        return _failure("chat_id is required")

    # A chat already inside Telegram's penalty gets a fail-closed result, exactly like send().
    remaining = getattr(adapter, "_send_flood_cooldown_remaining", None)
    if callable(remaining):
        cooldown = remaining(chat_id_norm)
        if cooldown is not None:
            return _flood_cap_result(float(cooldown))

    try:
        rich_message = adapter._rich_message_payload(content, metadata=metadata)
    except Exception as exc:
        return _failure(_redact_telegram_error_text(exc), exc=exc)
    if not isinstance(rich_message, Mapping) or not rich_message:
        return _failure("rich reply payload is empty")

    payload: dict[str, Any] = {"chat_id": chat_id_norm, "rich_message": dict(rich_message)}
    try:
        thread_kwargs = adapter._thread_kwargs_for_send(
            chat_id_norm, adapter._metadata_thread_id(metadata), metadata,
            reply_to_message_id=None, reply_to_mode=getattr(adapter, "_reply_to_mode", None))
    except Exception as exc:
        return _failure(_redact_telegram_error_text(exc), exc=exc)
    payload.update({key: value for key, value in (thread_kwargs or {}).items() if value is not None})
    payload.update(adapter._notification_kwargs(metadata) or {})
    if getattr(adapter, "_disable_link_previews", False):
        payload["link_preview_options"] = {"is_disabled": True}
    # sendRichMessage takes reply_parameters; reply_to_message_id is silently ignored.
    payload["reply_parameters"] = parameters

    try:
        message = await _await_with_thread_deadline(
            bot.do_api_request("sendRichMessage", api_kwargs=payload),
            timeout=_TEXT_SEND_DEADLINE, label="telegram-send", dump_on_blocked_loop=False)
    except Exception as exc:
        safe_error = _redact_telegram_error_text(exc)
        retry_after = _flood_window.flood_wait_seconds(exc)
        graded = _flood_window.note_flood(adapter, chat_id_norm, exc)
        capability = getattr(adapter, "_is_rich_capability_error", None)
        if callable(capability) and capability(exc):
            adapter._rich_send_disabled = True
        raw: dict[str, Any] = {"what": "sendRichMessage", "exc": exc}
        if graded or retry_after is not None:
            raw["flood_control"] = True
        elif callable(getattr(adapter, "_is_rich_fallback_error", None)) and adapter._is_rich_fallback_error(exc):
            raw["rejected"] = True
        else:
            raw["ambiguous"] = True
        # Never retried: a retry could drop the quote or the rich formatting.
        return _failure(safe_error, exc=exc, retry_after=retry_after, raw_response=raw)

    message_id = _message_id_from_response(message)
    raw_response: dict[str, Any] = {"message": message}
    if isinstance(message, Mapping):
        for key in ("widget", "rich_message"):
            if message.get(key) is not None:
                raw_response[key] = message[key]
    if message_id is None:
        # The request may have landed; a resend would duplicate it.
        raw_response.update({"ambiguous": True, "what": "sendRichMessage"})
        return _failure("rich reply response without message_id", raw_response=raw_response)

    record = getattr(adapter, "_record_rich_sent", None)
    if callable(record):
        try:
            await record(chat_id_norm, message_id, content)
        except Exception:
            logger.debug("[telegram] rich reply index record failed", exc_info=True)
    return SendResult(success=True, message_id=str(message_id), raw_response=raw_response)

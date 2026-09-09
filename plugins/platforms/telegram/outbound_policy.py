"""Shared Telegram outbound content and PEER_FLOOD policy helpers."""

from __future__ import annotations

import inspect
import math
import re
from collections.abc import Mapping
from copy import copy
from datetime import date, datetime, time as datetime_time, timedelta
from enum import Enum
from io import IOBase
from os import PathLike
from typing import Any, Awaitable, Callable
from urllib.parse import urlsplit

DEFAULT_PEER_FLOOD_COOLDOWN_SECONDS = 300.0
MAX_PEER_FLOOD_COOLDOWN_SECONDS = 300.0

_URL_TOKEN_RE = re.compile(
    r"(?<![a-z0-9_@.-])(?:https?:)?//[^\s<>\[\]{}()'\"`]+",
    re.IGNORECASE,
)
_BARE_COUPANG_HOST_RE = re.compile(
    r"(?<![a-z0-9_/@.=?#&-])"
    r"(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)*coupang\.com"
    r"(?:\.)?(?=(?::[0-9]+)?(?:[/\s?#)\]>,;:'\"`]|$))",
    re.IGNORECASE,
)
_PEER_FLOOD_RE = re.compile(r"\bPEER[_ ]FLOOD\b", re.IGNORECASE)


def deactivate_coupang_urls(content: str) -> str:
    """Defang Coupang URL origins without touching another URL's values."""
    if not content:
        return content

    def defang_host(value: str) -> str:
        return re.sub(
            r"coupang\.com(?=\.?(?::[0-9]+)?(?:[/\\?#]|$))",
            "coupang[.]com",
            value,
            flags=re.IGNORECASE,
        )

    def defang_url_token(match: re.Match) -> str:
        token = match.group(0)
        parsed = urlsplit(token if "://" in token else f"https:{token}")
        host = (parsed.hostname or "").rstrip(".").lower()
        if host == "coupang.com" or host.endswith(".coupang.com"):
            return defang_host(token)
        return token

    pieces = []
    cursor = 0
    for match in _URL_TOKEN_RE.finditer(content):
        pieces.append(
            _BARE_COUPANG_HOST_RE.sub(
                lambda item: defang_host(item.group(0)), content[cursor : match.start()]
            )
        )
        pieces.append(defang_url_token(match))
        cursor = match.end()
    pieces.append(
        _BARE_COUPANG_HOST_RE.sub(
            lambda item: defang_host(item.group(0)), content[cursor:]
        )
    )
    prepared = "".join(pieces)
    # Preserve str subclasses such as telegram.constants.ParseMode when the
    # policy made no change; converting them to plain str breaks caller
    # contracts and equality-by-identity assumptions in PTB integrations.
    return content if prepared == content else prepared


def prepare_caption(caption: str | None, limit: int = 1024) -> str | None:
    """Defang Coupang URLs, then truncate to Telegram's UTF-16 unit limit."""
    if not caption:
        return None
    prepared = deactivate_coupang_urls(caption)
    units = 0
    end = 0
    for end, char in enumerate(prepared, start=1):
        units += len(char.encode("utf-16-le")) // 2
        if units > limit:
            return prepared[: end - 1]
    return prepared


def is_peer_flood(value: object) -> bool:
    """Recognize PEER_FLOOD in exceptions and raw Bot API error responses."""
    if isinstance(value, dict):
        text = " ".join(
            str(value.get(key, "")) for key in ("error", "description", "message")
        )
    else:
        text = str(value or "")
    return bool(_PEER_FLOOD_RE.search(text))


def peer_flood_delay(value: object) -> float:
    """Return a finite positive cooldown, capped to the shared maximum."""
    retry_after = getattr(value, "retry_after", None)
    if retry_after is None and isinstance(value, dict):
        parameters = value.get("parameters")
        if isinstance(parameters, dict):
            retry_after = parameters.get("retry_after")
        if retry_after is None:
            retry_after = value.get("retry_after")
    try:
        delay = float(retry_after)
        if not math.isfinite(delay) or delay <= 0:
            raise ValueError
    except (TypeError, ValueError):
        delay = DEFAULT_PEER_FLOOD_COOLDOWN_SECONDS
    return min(delay, MAX_PEER_FLOOD_COOLDOWN_SECONDS)


class TelegramPeerFloodBlocked(RuntimeError):
    """Raised when the per-target policy circuit blocks a Bot API write."""

    def __init__(self, retry_after: float):
        super().__init__("Telegram PEER_FLOOD outbound circuit is open")
        self.retry_after = float(retry_after)


class TelegramUnsafeContentBlocked(RuntimeError):
    """Raised when a payload cannot be inspected safely before transport."""


def _copy_with_content_policy(
    value: Any,
    *,
    depth: int,
    max_depth: int,
    memo: dict[int, Any],
    active_tuples: set[int],
) -> Any:
    """Copy a supported outbound value and defang every reachable string value."""
    if depth > max_depth:
        raise TelegramUnsafeContentBlocked("Telegram payload exceeds safe policy depth")
    if isinstance(value, str):
        return deactivate_coupang_urls(value)
    if value is None or isinstance(value, (bytes, bytearray, int, float, bool)):
        return value

    identity = id(value)
    if identity in memo:
        return memo[identity]
    if isinstance(value, Mapping):
        copied: dict[Any, Any] = {}
        memo[identity] = copied
        for key, item in value.items():
            copied_key = _copy_with_content_policy(
                key,
                depth=depth + 1,
                max_depth=max_depth,
                memo=memo,
                active_tuples=active_tuples,
            )
            try:
                if copied_key in copied:
                    raise TelegramUnsafeContentBlocked(
                        "Telegram payload has a key collision after content policy"
                    )
            except TypeError as exc:
                raise TelegramUnsafeContentBlocked(
                    "Telegram payload key became unhashable after content policy"
                ) from exc
            copied[copied_key] = _copy_with_content_policy(
                item,
                depth=depth + 1,
                max_depth=max_depth,
                memo=memo,
                active_tuples=active_tuples,
            )
        return copied
    if isinstance(value, list):
        copied_list: list[Any] = []
        memo[identity] = copied_list
        copied_list.extend(
            _copy_with_content_policy(
                item,
                depth=depth + 1,
                max_depth=max_depth,
                memo=memo,
                active_tuples=active_tuples,
            )
            for item in value
        )
        return copied_list
    if isinstance(value, tuple):
        if identity in active_tuples:
            raise TelegramUnsafeContentBlocked("Telegram payload contains a tuple cycle")
        active_tuples.add(identity)
        try:
            copied_tuple = tuple(
                _copy_with_content_policy(
                    item,
                    depth=depth + 1,
                    max_depth=max_depth,
                    memo=memo,
                    active_tuples=active_tuples,
                )
                for item in value
            )
        finally:
            active_tuples.remove(identity)
        memo[identity] = copied_tuple
        return copied_tuple

    try:
        from telegram import TelegramObject
    except ImportError:
        TelegramObject = None

    if isinstance(TelegramObject, type) and isinstance(value, TelegramObject):
        source_bot = getattr(value, "_bot", None)
        copied_object = copy(value)
        memo[identity] = copied_object
        fields: list[str] = []
        for cls in type(value).mro():
            slots = getattr(cls, "__slots__", ())
            if isinstance(slots, str):
                slots = (slots,)
            for field in slots:
                if field.startswith("_") or field in fields or not hasattr(value, field):
                    continue
                fields.append(field)
        value_dict = getattr(value, "__dict__", None)
        if isinstance(value_dict, dict):
            for field in value_dict:
                if field.startswith("_") or field in fields:
                    continue
                fields.append(field)
        with copied_object._unfrozen():
            for field in fields:
                setattr(
                    copied_object,
                    field,
                    _copy_with_content_policy(
                        getattr(value, field),
                        depth=depth + 1,
                        max_depth=max_depth,
                        memo=memo,
                        active_tuples=active_tuples,
                    ),
                )
        if source_bot is not None:
            copied_object.set_bot(source_bot)
        return copied_object

    try:
        from telegram import InputFile
    except ImportError:
        InputFile = None
    try:
        from telegram._utils.defaultvalue import DefaultValue
    except ImportError:
        DefaultValue = None
    if isinstance(DefaultValue, type) and isinstance(value, DefaultValue):
        copied_default = type(value)(
            _copy_with_content_policy(
                value.value,
                depth=depth + 1,
                max_depth=max_depth,
                memo=memo,
                active_tuples=active_tuples,
            )
        )
        memo[identity] = copied_default
        return copied_default
    opaque_types = tuple(
        candidate
        for candidate in (
            InputFile,
            IOBase,
            PathLike,
            Enum,
            date,
            datetime,
            datetime_time,
            timedelta,
        )
        if isinstance(candidate, type)
    )
    if isinstance(value, opaque_types):
        return value
    raise TelegramUnsafeContentBlocked(
        f"Telegram payload has unsupported payload type: {type(value).__name__}"
    )


class TelegramOutboundGateway:
    """The single asynchronous policy boundary for Telegram Bot API writes."""

    _TEXT_METHODS = {
        "answer",
        "edit_message_text",
        "send_message",
        "send_message_draft",
    }
    _GLOBAL_METHODS = {
        "delete_webhook",
        "set_my_commands",
        "set_my_description",
        "set_my_short_description",
        "set_webhook",
    }
    _MAX_CONTENT_DEPTH = 32

    @staticmethod
    def _key(target_key: object) -> str:
        from plugins.platforms.telegram.outbound_circuit import chat_key

        return chat_key(target_key)

    @staticmethod
    def _apply_content_policy(method: str, args: tuple[Any, ...], kwargs: dict[str, Any]):
        del method
        memo: dict[int, Any] = {}
        active_tuples: set[int] = set()
        args_out = _copy_with_content_policy(
            args,
            depth=0,
            max_depth=TelegramOutboundGateway._MAX_CONTENT_DEPTH,
            memo=memo,
            active_tuples=active_tuples,
        )
        kwargs_out = _copy_with_content_policy(
            kwargs,
            depth=0,
            max_depth=TelegramOutboundGateway._MAX_CONTENT_DEPTH,
            memo=memo,
            active_tuples=active_tuples,
        )
        return args_out, kwargs_out

    async def write(
        self,
        target_key: object,
        method: str,
        operation: Callable[..., Awaitable[Any]],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        from plugins.platforms.telegram import outbound_circuit

        key = self._key(target_key)
        blocked = await outbound_circuit.remaining_async(key)
        if blocked is not None:
            raise TelegramPeerFloodBlocked(blocked)
        args, kwargs = self._apply_content_policy(method, args, kwargs)
        try:
            result = await operation(*args, **kwargs)
        except Exception as exc:
            if not is_peer_flood(exc):
                raise
            delay = await outbound_circuit.open_circuit_async(key, peer_flood_delay(exc))
            raise TelegramPeerFloodBlocked(delay) from None
        if is_peer_flood(result):
            delay = await outbound_circuit.open_circuit_async(key, peer_flood_delay(result))
            raise TelegramPeerFloodBlocked(delay)
        return result


class GuardedTelegramTarget:
    """Verifiably thin dynamic wrapper whose write methods all enter the gateway."""

    def __init__(
        self,
        target: object,
        gateway: TelegramOutboundGateway,
        *,
        target_key: object | None = None,
    ) -> None:
        self._target = target
        self._gateway = gateway
        self._target_key = target_key

    def __getattr__(self, method: str):
        operation = getattr(self._target, method)
        if method.startswith("get_") or method in {"initialize", "shutdown"}:
            return operation

        async def guarded(*args: Any, **kwargs: Any):
            key = self._target_key
            if key is None:
                key = kwargs.get("chat_id")
            if key is None and isinstance(kwargs.get("api_kwargs"), dict):
                key = kwargs["api_kwargs"].get("chat_id")
            if key is None and method == "set_my_commands":
                scope = kwargs.get("scope")
                if isinstance(scope, Mapping):
                    if "chat_id" in scope:
                        key = scope.get("chat_id")
                elif inspect.getattr_static(scope, "chat_id", None) is not None:
                    key = getattr(scope, "chat_id", None)
            if key is None and method in TelegramOutboundGateway._GLOBAL_METHODS:
                from plugins.platforms.telegram.outbound_circuit import GLOBAL_CIRCUIT_KEY

                key = GLOBAL_CIRCUIT_KEY
            if key is None:
                key = "telegram:user:unknown"
            return await self._gateway.write(key, method, operation, *args, **kwargs)

        return guarded

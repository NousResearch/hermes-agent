"""Zalo platform adapter backed by a local hzca serve REST/SSE backend."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import unicodedata
from dataclasses import dataclass
from ipaddress import ip_address
from typing import Any, AsyncIterator, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult
from gateway.platforms.helpers import MessageDeduplicator
from gateway.session import SessionSource

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "http://127.0.0.1:56789"
_ALLOWED_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}


@dataclass(frozen=True)
class ZaloBackendConfig:
    base_url: str = _DEFAULT_BASE_URL
    bearer_token: Optional[str] = None
    request_timeout: float = 30.0
    sse_timeout: Optional[float] = None
    allow_unsafe_remote: bool = False
    allowed_user_ids: tuple[str, ...] = ()
    allow_all_users: bool = False
    allowed_group_ids: tuple[str, ...] = ()
    require_mention: bool = True
    prefixes: tuple[str, ...] = ()
    free_response_group_ids: tuple[str, ...] = ()
    dedupe_ttl_seconds: float = 300.0
    max_message_length: int = 1800


def parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _split_csv(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple, set)):
        return tuple(str(v).strip() for v in value if str(v).strip())
    return tuple(part.strip() for part in str(value).split(",") if part.strip())


def _extra_value(extra: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in extra and extra[key] is not None:
            return extra[key]
    return None


def _env_value(*names: str) -> Optional[str]:
    for name in names:
        value = os.getenv(name)
        if value is not None:
            return value
    return None


def _policy_value(extra: dict[str, Any], keys: tuple[str, ...], env_names: tuple[str, ...], default: Any = None) -> Any:
    value = _extra_value(extra, *keys)
    if value is not None:
        return value
    value = _env_value(*env_names)
    return default if value is None else value


def _is_loopback_host(host: str) -> bool:
    host = (host or "").strip().lower().strip("[]")
    if host in _ALLOWED_LOOPBACK_HOSTS:
        return True
    try:
        return ip_address(host).is_loopback
    except ValueError:
        pass
    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except OSError:
        return False
    return bool(infos) and all(ip_address(info[4][0]).is_loopback for info in infos)


def _optional_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return None
    parsed = float(value)
    return parsed if parsed > 0 else None


def validate_local_backend_url(base_url: str, *, allow_unsafe_remote: bool = False) -> str:
    """Return normalized hzca base URL, rejecting non-loopback backends by default."""
    raw = str(base_url or "").strip() or _DEFAULT_BASE_URL
    parsed = urlparse(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("HZCA serve URL must be an absolute http(s) URL")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("HZCA serve URL must be an origin (scheme://host[:port])") from exc
    if (
        parsed.username
        or parsed.password
        or "@" in parsed.netloc
        or parsed.path not in {"", "/"}
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "HZCA serve URL must be an origin (scheme://host[:port]) without path, query, fragment, or userinfo"
        )
    if not allow_unsafe_remote and not _is_loopback_host(parsed.hostname):
        raise ValueError(
            "Refusing non-loopback HZCA serve URL; set allow_unsafe_remote=true only for an explicitly trusted backend"
        )
    host = parsed.hostname.lower()
    netloc = f"[{host}]" if ":" in host else host
    if port is not None:
        netloc = f"{netloc}:{port}"
    return f"{parsed.scheme.lower()}://{netloc}"


def config_from_platform(config: PlatformConfig) -> ZaloBackendConfig:
    extra = getattr(config, "extra", {}) or {}
    allow_unsafe = parse_bool(_policy_value(extra, ("allow_unsafe_remote",), ("ZALO_ALLOW_UNSAFE_REMOTE",), False))
    base_url = validate_local_backend_url(
        str(_policy_value(extra, ("hzca_serve_url", "base_url"), ("HZCA_SERVE_URL",), _DEFAULT_BASE_URL)),
        allow_unsafe_remote=allow_unsafe,
    )
    return ZaloBackendConfig(
        base_url=base_url,
        bearer_token=_policy_value(extra, ("hzca_bearer_token",), ("HZCA_BEARER_TOKEN",), None),
        request_timeout=float(_policy_value(extra, ("request_timeout",), ("ZALO_REQUEST_TIMEOUT",), 30.0)),
        sse_timeout=_optional_float(
            _policy_value(
                extra,
                ("sse_timeout", "sse_read_timeout", "read_timeout"),
                ("ZALO_SSE_TIMEOUT", "ZALO_SSE_READ_TIMEOUT"),
                None,
            )
        ),
        allow_unsafe_remote=allow_unsafe,
        allowed_user_ids=_split_csv(
            _policy_value(extra, ("allowed_user_ids", "allowed_users"), ("ZALO_ALLOWED_USER_IDS", "ZALO_ALLOWED_USERS"))
        ),
        allow_all_users=parse_bool(_policy_value(extra, ("allow_all_users",), ("ZALO_ALLOW_ALL_USERS",), False)),
        allowed_group_ids=_split_csv(
            _policy_value(
                extra,
                ("allowed_group_ids", "allowed_groups"),
                ("ZALO_ALLOWED_GROUP_IDS", "ZALO_ALLOWED_GROUPS"),
            )
        ),
        require_mention=parse_bool(
            _policy_value(extra, ("require_mention", "group_require_mention"), ("ZALO_GROUP_REQUIRE_MENTION",), True),
            True,
        ),
        prefixes=_split_csv(_policy_value(extra, ("prefixes", "group_prefixes"), ("ZALO_GROUP_PREFIXES",))),
        free_response_group_ids=_split_csv(
            _policy_value(extra, ("free_response_group_ids",), ("ZALO_FREE_RESPONSE_GROUP_IDS",))
        ),
        dedupe_ttl_seconds=float(_policy_value(extra, ("dedupe_ttl_seconds",), ("ZALO_DEDUPE_TTL_SECONDS",), 300.0)),
        max_message_length=int(_policy_value(extra, ("max_message_length",), ("ZALO_MAX_MESSAGE_LENGTH",), 1800)),
    )


def validate_config(config: PlatformConfig) -> bool:
    try:
        config_from_platform(config)
        return True
    except Exception as exc:
        logger.warning("[zalo] invalid config: %s", exc)
        return False


def check_requirements() -> bool:
    return True


def is_connected(config: PlatformConfig) -> bool:
    return validate_config(config)


class HzcaClient:
    def __init__(self, backend: ZaloBackendConfig):
        self.backend = backend

    def _request(self, method: str, path: str, body: Any = None) -> Any:
        data = None if body is None else json.dumps(body).encode("utf-8")
        headers = {"Accept": "application/json"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        if self.backend.bearer_token:
            headers["Authorization"] = f"Bearer {self.backend.bearer_token}"
        req = Request(f"{self.backend.base_url}{path}", data=data, headers=headers, method=method)
        with urlopen(req, timeout=self.backend.request_timeout) as resp:  # nosec - local URL is validated above
            raw = resp.read().decode("utf-8")
            if not raw:
                return None
            return json.loads(raw)

    async def request(self, method: str, path: str, body: Any = None) -> Any:
        return await asyncio.to_thread(self._request, method, path, body)

    async def get_health(self) -> Any:
        try:
            return await self.request("GET", "/api/health")
        except HTTPError as exc:
            if exc.code == 404:
                return await self.request("GET", "/health")
            raise

    async def get_me(self) -> Any:
        try:
            return await self.request("GET", "/api/me/id")
        except HTTPError as exc:
            if exc.code == 404:
                return await self.request("GET", "/api/me")
            raise

    async def send_text(self, *, thread_id: str, text: str, is_group: bool) -> Any:
        return await self.request(
            "POST",
            "/api/messages/text",
            {"threadId": thread_id, "message": text, "isGroup": is_group, "autoMarkdown": False},
        )

    async def send_typing(self, *, thread_id: str, is_group: bool) -> None:
        # HZCA serve does not expose a stable typing route in the current contract.
        # Keep this best-effort for forward compatibility if /api/typing lands later.
        try:
            await self.request("POST", "/api/typing", {"threadId": thread_id, "isGroup": is_group})
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
            logger.debug("[zalo] typing indicator unavailable/failed: %s", exc)

    async def iter_events(self, stop_event: asyncio.Event) -> AsyncIterator[dict[str, Any]]:
        """Minimal SSE parser for hzca /api/events."""
        headers = {"Accept": "text/event-stream"}
        if self.backend.bearer_token:
            headers["Authorization"] = f"Bearer {self.backend.bearer_token}"
        req = Request(f"{self.backend.base_url}/api/events", headers=headers, method="GET")

        def _open():
            return urlopen(req, timeout=self.backend.sse_timeout)  # nosec - URL validated

        resp = await asyncio.to_thread(_open)
        event_type = "message"
        data_lines: list[str] = []
        try:
            while not stop_event.is_set():
                line = await asyncio.to_thread(resp.readline)
                if not line:
                    break
                text = line.decode("utf-8").rstrip("\r\n")
                if text == "":
                    if data_lines:
                        parsed = _parse_sse_payload(data_lines, event_type)
                        if parsed is not None:
                            yield parsed
                        data_lines = []
                    event_type = "message"
                    continue
                if text.startswith(":") or text.startswith("id:") or text.startswith("retry:"):
                    continue
                if text.startswith("event:"):
                    event_type = text[6:].strip() or "message"
                elif text.startswith("data:"):
                    data_lines.append(text[5:].lstrip())
        finally:
            try:
                resp.close()
            except Exception:
                pass
        if data_lines:
            parsed = _parse_sse_payload(data_lines, event_type)
            if parsed is not None:
                yield parsed


def _parse_sse_payload(data_lines: list[str], event_type: str) -> Optional[dict[str, Any]]:
    payload = "\n".join(data_lines)
    try:
        parsed = json.loads(payload)
        if not isinstance(parsed, dict):
            return {"type": event_type, "data": parsed}
        if "type" not in parsed:
            parsed = {"type": event_type, "data": parsed}
        return parsed
    except json.JSONDecodeError:
        logger.debug("[zalo] ignoring malformed SSE payload")
        return None


def _user_id_from_me(payload: Any) -> Optional[str]:
    if not isinstance(payload, dict):
        return None
    candidates = [payload]
    for key in ("data", "user", "profile", "me"):
        if isinstance(payload.get(key), dict):
            candidates.append(payload[key])
    for item in candidates:
        for key in ("userId", "uid", "id", "zaloId"):
            value = item.get(key)
            if value is not None and str(value).strip():
                return str(value)
    return None


def _legacy_metadata_dicts(payload: Any) -> list[dict[str, Any]]:
    """Compatibility-only scan for pre-HZCA-v2 nested/raw metadata.

    HZCA v2 exposes canonical top-level SSE fields. New extraction code should
    read those first and only call this helper as a small fallback for older
    serve versions that hid mentions/quotes under raw/message/data shapes.
    TODO(zalo): remove once HZCA v2 is the minimum supported backend.
    """
    if not isinstance(payload, dict):
        return []
    found: list[dict[str, Any]] = []
    pending = [payload]
    seen: set[int] = set()
    while pending:
        item = pending.pop(0)
        marker = id(item)
        if marker in seen:
            continue
        seen.add(marker)
        found.append(item)
        for key in ("raw_message", "rawMessage", "raw", "message", "data"):
            nested = item.get(key)
            if isinstance(nested, dict):
                pending.append(nested)
    return found


def _string_or_none(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value)
    return text if text.strip() else None


def _first_string(item: dict[str, Any], *keys: str) -> Optional[str]:
    for key in keys:
        value = _string_or_none(item.get(key))
        if value is not None:
            return value
    return None


def _clean_mention(mention: Any) -> Optional[dict[str, Any]]:
    if mention is None:
        return None
    if not isinstance(mention, dict):
        text = str(mention).strip()
        return {"uid": text} if text else None

    clean: dict[str, Any] = {}
    for key in ("uid", "userId", "id", "zaloId", "user_id", "text", "name", "displayName"):
        value = mention.get(key)
        if value is not None and str(value).strip():
            clean[key] = str(value)
    for key in ("pos", "offset", "start", "len", "length"):
        value = mention.get(key)
        if value is not None and str(value).strip():
            try:
                clean[key] = int(value)
            except (TypeError, ValueError):
                clean[key] = str(value)
    return clean or None


def _clean_mentions(raw_mentions: Any) -> list[dict[str, Any]]:
    mentions: list[dict[str, Any]] = []
    if raw_mentions is None:
        return mentions
    if isinstance(raw_mentions, dict):
        if all(isinstance(value, dict) for value in raw_mentions.values()):
            raw_items = raw_mentions.values()
        else:
            raw_items = [raw_mentions]
    elif isinstance(raw_mentions, (list, tuple, set)):
        raw_items = raw_mentions
    else:
        raw_items = [raw_mentions]
    for raw_mention in raw_items:
        clean = _clean_mention(raw_mention)
        if clean:
            mentions.append(clean)
    return mentions


def _extract_legacy_mentions(payload: Any) -> list[dict[str, Any]]:
    for item in _legacy_metadata_dicts(payload):
        mentions = _clean_mentions(item.get("mentions") or item.get("mentionList") or item.get("mention_list"))
        if mentions:
            return mentions
    return []


def _extract_mentions(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("mentions"), (list, tuple)):
        return _clean_mentions(payload.get("mentions"))
    return _extract_legacy_mentions(payload)


def _extract_legacy_at_all(payload: Any) -> bool:
    for item in _legacy_metadata_dicts(payload):
        for key in ("isAtAll", "mentionAll", "atAll"):
            if key in item and parse_bool(item.get(key), False):
                return True
    return False


def _extract_at_all(payload: Any) -> bool:
    if isinstance(payload, dict) and "atAll" in payload:
        return parse_bool(payload.get("atAll"), False)
    return _extract_legacy_at_all(payload)


def _normalize_quote(quote: Any) -> Optional[dict[str, Any]]:
    if not isinstance(quote, dict):
        return None
    sender_id = _first_string(
        quote,
        "senderId",  # HZCA v2 canonical
        "uidFrom",
        "ownerId",
        "fromId",
        "userId",
        "uid",
    )
    normalized = {
        "msgId": _first_string(quote, "msgId", "id"),
        "cliMsgId": _string_or_none(quote.get("cliMsgId")),
        "senderId": sender_id,
    }
    message_id = _string_or_none(quote.get("messageId"))
    if message_id is not None:
        normalized["messageId"] = message_id
    return normalized


def _extract_legacy_quote(payload: Any) -> Optional[dict[str, Any]]:
    for item in _legacy_metadata_dicts(payload):
        for key in ("quote", "quotedMessage", "reply"):
            quote = _normalize_quote(item.get(key))
            if quote:
                return quote
    return None


def _extract_quote(payload: Any) -> Optional[dict[str, Any]]:
    if isinstance(payload, dict) and "quote" in payload:
        return _normalize_quote(payload.get("quote"))
    return _extract_legacy_quote(payload)


def _message_id(data: dict[str, Any]) -> str:
    return _first_string(data, "messageId", "id", "msgId", "cliMsgId") or ""


def _message_text(data: dict[str, Any]) -> str:
    return _first_string(data, "text", "content") or ""


def _message_kind(data: dict[str, Any]) -> str:
    return _first_string(data, "messageType", "type", "msgType") or "text"


def _quote_reply_message_id(quote: Any) -> Optional[str]:
    if not isinstance(quote, dict):
        return None
    return _first_string(quote, "messageId", "msgId", "id", "cliMsgId")


def _mention_target_id(mention: Any) -> Optional[str]:
    if not isinstance(mention, dict):
        return str(mention).strip() if mention is not None and str(mention).strip() else None
    for key in ("uid", "userId", "id", "zaloId", "user_id"):
        value = mention.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return None


def _strip_prefix(text: str, prefixes: tuple[str, ...]) -> tuple[bool, str]:
    for prefix in prefixes:
        if prefix and text.startswith(prefix):
            return True, text[len(prefix):].lstrip()
    return False, text


def _int_value(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_sse_event(event: dict[str, Any]) -> dict[str, Any]:
    """Normalize HZCA message fields for adapter routing and trigger checks."""
    data = event.get("data") if isinstance(event.get("data"), dict) else event
    quote = _extract_quote(data)
    return {
        "id": _message_id(data),
        "messageId": str(data.get("messageId") or ""),
        "msgId": str(data.get("msgId") or ""),
        "cliMsgId": str(data.get("cliMsgId") or ""),
        "threadId": str(data.get("threadId") or ""),
        "senderId": str(data.get("senderId") or ""),
        "senderName": str(data.get("senderName") or ""),
        "content": _message_text(data),
        "isGroup": bool(data.get("isGroup")),
        "type": _message_kind(data),
        "timestamp": data.get("timestamp"),
        "mentions": _extract_mentions(data),
        "atAll": _extract_at_all(data),
        "quote": quote,
    }


def _sanitize_normalized_event(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": event.get("id") or "",
        "messageId": event.get("messageId") or "",
        "msgId": event.get("msgId") or "",
        "cliMsgId": event.get("cliMsgId") or "",
        "threadId": event.get("threadId") or "",
        "senderId": event.get("senderId") or "",
        "content": event.get("content") or "",
        "isGroup": bool(event.get("isGroup")),
        "type": event.get("type") or "text",
        "quote": event.get("quote"),
    }


def sanitize_sse_event(event: dict[str, Any]) -> dict[str, Any]:
    """Keep only privacy-safe fields needed by Hermes; never retain raw event blobs."""
    return _sanitize_normalized_event(_normalize_sse_event(event))


def _split_message(text: str, limit: int) -> list[str]:
    """Split text without breaking Python Unicode code points; normalize first."""
    normalized = unicodedata.normalize("NFC", text or "")
    if len(normalized) <= limit:
        return [normalized]
    chunks: list[str] = []
    while normalized:
        chunks.append(normalized[:limit])
        normalized = normalized[limit:]
    return chunks


def _message_type(kind: str) -> MessageType:
    kind = (kind or "text").lower()
    if kind in {"photo", "image"}:
        return MessageType.PHOTO
    if kind in {"voice"}:
        return MessageType.VOICE
    if kind in {"audio"}:
        return MessageType.AUDIO
    if kind in {"video"}:
        return MessageType.VIDEO
    if kind in {"file", "document"}:
        return MessageType.DOCUMENT
    return MessageType.TEXT


class ZaloAdapter(BasePlatformAdapter):
    MAX_MESSAGE_LENGTH = 1800

    def __init__(self, config: PlatformConfig, client: Optional[HzcaClient] = None):
        super().__init__(config, Platform("zalo"))
        self.backend = config_from_platform(config)
        self.MAX_MESSAGE_LENGTH = self.backend.max_message_length
        self._client = client or HzcaClient(self.backend)
        self._stop_event = asyncio.Event()
        self._listen_task: Optional[asyncio.Task] = None
        self._dedup = MessageDeduplicator(ttl_seconds=self.backend.dedupe_ttl_seconds)
        self._self_user_id: Optional[str] = None
        self._thread_is_group: dict[str, bool] = {}

    @property
    def name(self) -> str:
        return "Zalo"

    async def connect(self) -> bool:
        try:
            await self._client.get_health()
            self._self_user_id = _user_id_from_me(await self._client.get_me())
        except Exception as exc:
            self._set_fatal_error(
                "zalo_connect_failed",
                f"Failed to connect to local hzca serve: {exc}",
                retryable=True,
            )
            return False
        self._stop_event.clear()
        self._listen_task = asyncio.create_task(self._listen_events())
        self._mark_connected()
        return True

    async def disconnect(self) -> None:
        self._stop_event.set()
        if self._listen_task and not self._listen_task.done():
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass
        self._mark_disconnected()

    async def _listen_events(self) -> None:
        while not self._stop_event.is_set():
            try:
                async for raw_event in self._client.iter_events(self._stop_event):
                    await self._handle_sse_event(raw_event)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if not self._stop_event.is_set():
                    logger.warning("[zalo] SSE listener error; reconnecting: %s", exc)
                    await asyncio.sleep(2)

    def _is_source_allowed(self, *, chat_type: str, chat_id: str, user_id: Optional[str]) -> bool:
        if chat_type == "group":
            return bool(self.backend.allowed_group_ids and chat_id in self.backend.allowed_group_ids)
        if self.backend.allow_all_users:
            return True
        return bool(user_id and self.backend.allowed_user_ids and user_id in self.backend.allowed_user_ids)

    def _is_authorized(self, event: dict[str, Any]) -> bool:
        return self._is_source_allowed(
            chat_type="group" if event.get("isGroup") else "dm",
            chat_id=str(event.get("threadId") or ""),
            user_id=str(event.get("senderId") or "") or None,
        )

    def is_source_authorized(self, source: SessionSource) -> bool:
        """Authorize sources that already passed Zalo's adapter allowlist."""
        if source.platform != self.platform:
            return False
        return self._is_source_allowed(
            chat_type=source.chat_type,
            chat_id=str(source.chat_id or ""),
            user_id=source.user_id,
        )

    def _mentions_self(self, event: dict[str, Any]) -> bool:
        if not self._self_user_id:
            return False
        return any(_mention_target_id(mention) == self._self_user_id for mention in event.get("mentions") or ())

    def _is_verified_reply_to_self(self, event: dict[str, Any]) -> bool:
        if not self._self_user_id:
            return False
        quote = event.get("quote") if isinstance(event.get("quote"), dict) else None
        if not quote:
            return False
        sender_id = quote.get("senderId")
        return bool(sender_id and str(sender_id) == self._self_user_id)

    def _should_process_group_trigger(self, event: dict[str, Any]) -> bool:
        thread_id = str(event.get("threadId") or "")
        if thread_id in self.backend.free_response_group_ids:
            return True
        if not self.backend.require_mention:
            return True
        if _strip_prefix(str(event.get("content") or ""), self.backend.prefixes)[0]:
            return True
        if self._mentions_self(event):
            return True
        return self._is_verified_reply_to_self(event)

    def _should_process_inbound(self, event: dict[str, Any]) -> bool:
        if not self._is_authorized(event):
            return False
        if not event.get("isGroup"):
            return True
        return self._should_process_group_trigger(event)

    def _strip_leading_self_mention(self, text: str, mentions: list[dict[str, Any]]) -> str:
        if not self._self_user_id:
            return text
        for mention in mentions:
            if _mention_target_id(mention) != self._self_user_id:
                continue
            pos = _int_value(mention.get("pos", mention.get("offset", mention.get("start"))))
            length = _int_value(mention.get("len", mention.get("length")))
            if pos == 0 and length and 0 < length <= len(text):
                return text[length:].lstrip()
            for key in ("text", "name", "displayName"):
                label = str(mention.get(key) or "").strip()
                if not label:
                    continue
                candidates = (label, label if label.startswith("@") else f"@{label}")
                for candidate in candidates:
                    if candidate and text.startswith(candidate):
                        return text[len(candidate):].lstrip()
        return text

    def _message_text_for_event(self, event: dict[str, Any]) -> str:
        text = str(event.get("content") or "")
        if not event.get("isGroup"):
            return text
        prefix_matched, stripped = _strip_prefix(text, self.backend.prefixes)
        if prefix_matched:
            return stripped
        if self._mentions_self(event):
            return self._strip_leading_self_mention(text, event.get("mentions") or [])
        return text

    def _dedupe_key(self, event: dict[str, Any]) -> str:
        return "|".join(str(event.get(k) or "") for k in ("id", "msgId", "cliMsgId", "threadId", "senderId"))

    async def _handle_sse_event(self, raw_event: dict[str, Any]) -> None:
        if (raw_event.get("type") or "message") != "message":
            return
        event = _normalize_sse_event(raw_event)
        if not event["threadId"] or not event["senderId"]:
            return
        if self._self_user_id and event["senderId"] == self._self_user_id:
            return
        if self._dedup.is_duplicate(self._dedupe_key(event)):
            return
        if not self._should_process_inbound(event):
            return

        self._thread_is_group[event["threadId"]] = bool(event["isGroup"])
        source = SessionSource(
            platform=Platform("zalo"),
            chat_id=event["threadId"],
            chat_name=event["threadId"],
            chat_type="group" if event["isGroup"] else "dm",
            user_id=event["senderId"],
            user_name=event.get("senderName") or event["senderId"],
            thread_id=event["threadId"],
            message_id=event.get("id") or event.get("msgId"),
        )
        message = MessageEvent(
            text=self._message_text_for_event(event),
            message_type=_message_type(event.get("type")),
            source=source,
            raw_message=_sanitize_normalized_event(event),
            message_id=event.get("id") or event.get("msgId"),
            reply_to_message_id=_quote_reply_message_id(event.get("quote")),
        )
        await self.handle_message(message)

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> SendResult:
        is_group = self._resolve_is_group(chat_id, metadata)
        if is_group is None:
            return SendResult(success=False, error="Zalo send requires isGroup metadata or cached inbound thread type")
        responses: list[Any] = []
        try:
            for chunk in _split_message(content, self.backend.max_message_length):
                responses.append(await self._client.send_text(thread_id=chat_id, text=chunk, is_group=is_group))
            last = responses[-1] if responses else None
            message_id = None
            if isinstance(last, dict):
                message_id = str(last.get("messageId") or last.get("msgId") or last.get("id") or "") or None
            return SendResult(
                success=True,
                message_id=message_id,
                raw_response=responses if len(responses) != 1 else last,
            )
        except Exception as exc:
            return SendResult(success=False, error=str(exc), retryable=True)

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        is_group = self._resolve_is_group(chat_id, metadata)
        if is_group is None:
            return
        await self._client.send_typing(thread_id=chat_id, is_group=is_group)

    async def get_chat_info(self, chat_id: str) -> dict[str, Any]:
        is_group = self._thread_is_group.get(str(chat_id))
        return {
            "id": str(chat_id),
            "name": str(chat_id),
            "type": "group" if is_group else "dm" if is_group is False else "unknown",
            "isGroup": is_group,
        }

    def _resolve_is_group(self, chat_id: str, metadata: Optional[dict[str, Any]]) -> Optional[bool]:
        if metadata and "isGroup" in metadata:
            return bool(metadata["isGroup"])
        if metadata and "is_group" in metadata:
            return bool(metadata["is_group"])
        return self._thread_is_group.get(str(chat_id))


def register(ctx) -> None:
    ctx.register_platform(
        name="zalo",
        label="Zalo",
        adapter_factory=lambda cfg: ZaloAdapter(cfg),
        check_fn=check_requirements,
        validate_config=validate_config,
        is_connected=is_connected,
        emoji="💬",
        install_hint=(
            "Run `hzca serve --host 127.0.0.1 --token-file ~/.hermes/hzca-token` "
            "and configure platforms.zalo.extra.hzca_serve_url."
        ),
    )

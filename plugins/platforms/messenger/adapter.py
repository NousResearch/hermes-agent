"""Facebook Messenger platform adapter for Hermes Agent.

This is a bundled platform plugin, not a core platform. It follows Meta's
Messenger webhook contract:

* ``GET /messenger/webhook`` handles ``hub.challenge`` verification.
* ``POST /messenger/webhook`` requires ``X-Hub-Signature-256`` HMAC-SHA256
  over the raw request body using the Facebook app secret.
* outbound messages use the Page Send API at ``/{version}/me/messages``.

Access control is intentionally delegated to the gateway-level platform
authorization layer. Configure ``MESSENGER_ALLOWED_USERS`` /
``MESSENGER_ALLOW_ALL_USERS`` or use the default pairing flow.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import re
import time
import uuid
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit

try:
    from aiohttp import web

    AIOHTTP_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on optional runtime deps
    web = None  # type: ignore[assignment]
    AIOHTTP_AVAILABLE = False

try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on optional runtime deps
    httpx = None  # type: ignore[assignment]
    HTTPX_AVAILABLE = False

from gateway.config import Platform
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    _ssrf_redirect_guard,
    cache_media_bytes,
    safe_url_for_log,
    validate_inbound_media_size,
)
from tools.url_safety import is_safe_url

logger = logging.getLogger(__name__)


GRAPH_API_BASE = "https://graph.facebook.com"
DEFAULT_API_VERSION = "v21.0"
DEFAULT_WEBHOOK_HOST = "0.0.0.0"
DEFAULT_WEBHOOK_PORT = 8650
DEFAULT_WEBHOOK_PATH = "/messenger/webhook"
MAX_MESSAGE_LENGTH = 2000
WEBHOOK_BODY_MAX_BYTES = 1_048_576
DEDUP_MAX_SIZE = 2048

_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^\s)]+)\)")
_MARKDOWN_CODE_BLOCK_RE = re.compile(r"```[a-zA-Z0-9_+-]*\n?(.*?)```", re.DOTALL)
_MARKDOWN_CODE_INLINE_RE = re.compile(r"`([^`]+)`")
_MARKDOWN_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_MARKDOWN_ITALIC_RE = re.compile(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)")
_MARKDOWN_HEADING_RE = re.compile(r"^#{1,6}\s+", re.MULTILINE)
_MARKDOWN_BULLET_RE = re.compile(r"^[\s]*[-*+]\s+", re.MULTILINE)


def _env_value(key: str) -> str:
    """Read env values the Hermes way, including ``~/.hermes/.env``."""
    try:
        from hermes_cli.config import get_env_value

        return (get_env_value(key) or "").strip()
    except Exception:
        return os.getenv(key, "").strip()


class MessengerGraphError(RuntimeError):
    """Raised for failed Meta Graph API calls."""

    def __init__(self, status_code: int, detail: Any):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"Meta Graph API returned {status_code}: {detail}")

    @property
    def retryable(self) -> bool:
        return self.status_code == 429 or self.status_code >= 500


@dataclass(frozen=True)
class MessengerAccount:
    """One Messenger Page/App credential set."""

    account_id: str
    name: str
    page_access_token: str
    app_secret: str
    verify_token: str
    enabled: bool = True
    webhook_path: str = DEFAULT_WEBHOOK_PATH
    api_version: str = DEFAULT_API_VERSION

    @property
    def complete(self) -> bool:
        return bool(
            self.enabled
            and self.page_access_token
            and self.app_secret
            and self.verify_token
        )


def _truthy(value: Any, default: bool = False) -> bool:
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


def _normalise_webhook_path(path: Any) -> str:
    text = str(path or DEFAULT_WEBHOOK_PATH).strip() or DEFAULT_WEBHOOK_PATH
    return text if text.startswith("/") else f"/{text}"


def _normalise_api_version(value: Any) -> str:
    text = str(value or DEFAULT_API_VERSION).strip() or DEFAULT_API_VERSION
    return text if text.startswith("v") else f"v{text}"


def _read_secret_file(path_value: Any) -> str:
    path = str(path_value or "").strip()
    if not path:
        return ""
    try:
        return Path(path).expanduser().read_text(encoding="utf-8").strip()
    except OSError as exc:
        logger.warning("Failed to read Messenger secret file %s: %s", path, exc)
        return ""


def _get_cfg_value(data: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in data and data[key] not in (None, ""):
            return data[key]
    return None


def _get_secret(
    data: Dict[str, Any],
    snake_key: str,
    camel_key: str,
    file_snake_key: str,
    file_camel_key: str,
    env_key: str = "",
) -> str:
    env_value = _env_value(env_key) if env_key else ""
    direct = _get_cfg_value(data, snake_key, camel_key)
    file_value = _get_cfg_value(data, file_snake_key, file_camel_key)
    return env_value or str(direct or "").strip() or _read_secret_file(file_value)


def verify_messenger_signature(body: bytes, signature_header: str, app_secret: str) -> bool:
    """Return True when ``signature_header`` matches ``body`` for ``app_secret``."""
    if not body or not signature_header or not app_secret:
        return False
    header = signature_header.strip()
    if not header.startswith("sha256="):
        return False
    expected = "sha256=" + hmac.new(
        app_secret.encode("utf-8"),
        body,
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(header, expected)


def strip_markdown_for_messenger(text: str) -> str:
    """Convert common Markdown to Messenger-friendly plain text."""
    if not text:
        return ""

    text = _MARKDOWN_CODE_BLOCK_RE.sub(lambda m: m.group(1).rstrip("\n"), text)
    text = _MARKDOWN_CODE_INLINE_RE.sub(r"\1", text)
    text = _MARKDOWN_LINK_RE.sub(lambda m: f"{m.group(1)} ({m.group(2)})", text)
    text = _MARKDOWN_BOLD_RE.sub(r"\1", text)
    text = _MARKDOWN_ITALIC_RE.sub(r"\1", text)
    text = _MARKDOWN_HEADING_RE.sub("", text)
    text = _MARKDOWN_BULLET_RE.sub("• ", text)
    return text


def split_for_messenger(text: str, max_chars: int = MAX_MESSAGE_LENGTH) -> List[str]:
    """Split text into Messenger's 2000-character Send API chunks."""
    if text is None:
        return [""]
    text = str(text)
    if len(text) <= max_chars:
        return [text]

    chunks: List[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= max_chars:
            chunks.append(remaining)
            break

        window = remaining[:max_chars]
        split_at = max(window.rfind("\n\n"), window.rfind("\n"), window.rfind(" "))
        if split_at < int(max_chars * 0.65):
            split_at = max_chars
        chunk = remaining[:split_at].rstrip()
        chunks.append(chunk)
        remaining = remaining[split_at:].lstrip()
    return chunks or [""]


def _coerce_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, Iterable):
        return [str(part).strip() for part in value if str(part).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _default_account_from_extra(extra: Dict[str, Any]) -> Optional[MessengerAccount]:
    page_token = _get_secret(
        extra,
        "page_access_token",
        "pageAccessToken",
        "token_file",
        "tokenFile",
        "MESSENGER_PAGE_ACCESS_TOKEN",
    )
    app_secret = _get_secret(
        extra,
        "app_secret",
        "appSecret",
        "secret_file",
        "secretFile",
        "MESSENGER_APP_SECRET",
    )
    verify_token = _get_secret(
        extra,
        "verify_token",
        "verifyToken",
        "verify_token_file",
        "verifyTokenFile",
        "MESSENGER_VERIFY_TOKEN",
    )
    if not any((page_token, app_secret, verify_token)):
        return None

    return MessengerAccount(
        account_id=str(extra.get("account_id") or extra.get("accountId") or "default"),
        name=str(extra.get("name") or "Messenger"),
        enabled=_truthy(extra.get("enabled"), True),
        page_access_token=page_token,
        app_secret=app_secret,
        verify_token=verify_token,
        webhook_path=_normalise_webhook_path(
            _env_value("MESSENGER_WEBHOOK_PATH") or extra.get("webhook_path") or extra.get("webhookPath")
        ),
        api_version=_normalise_api_version(
            _env_value("MESSENGER_API_VERSION") or extra.get("api_version") or extra.get("apiVersion")
        ),
    )


def _account_from_mapping(account_id: str, data: Dict[str, Any], defaults: Dict[str, Any]) -> MessengerAccount:
    merged = dict(defaults)
    merged.update(data)
    return MessengerAccount(
        account_id=str(data.get("id") or data.get("account_id") or data.get("accountId") or account_id),
        name=str(data.get("name") or account_id),
        enabled=_truthy(data.get("enabled"), True),
        page_access_token=_get_secret(merged, "page_access_token", "pageAccessToken", "token_file", "tokenFile"),
        app_secret=_get_secret(merged, "app_secret", "appSecret", "secret_file", "secretFile"),
        verify_token=_get_secret(merged, "verify_token", "verifyToken", "verify_token_file", "verifyTokenFile"),
        webhook_path=_normalise_webhook_path(
            data.get("webhook_path") or data.get("webhookPath") or defaults.get("webhook_path")
        ),
        api_version=_normalise_api_version(
            data.get("api_version") or data.get("apiVersion") or defaults.get("api_version")
        ),
    )


def parse_accounts(extra: Optional[Dict[str, Any]]) -> Dict[str, MessengerAccount]:
    """Parse single- or multi-page Messenger account config."""
    extra = dict(extra or {})
    defaults = {
        "webhook_path": _env_value("MESSENGER_WEBHOOK_PATH")
        or extra.get("webhook_path")
        or extra.get("webhookPath")
        or DEFAULT_WEBHOOK_PATH,
        "api_version": _env_value("MESSENGER_API_VERSION")
        or extra.get("api_version")
        or extra.get("apiVersion")
        or DEFAULT_API_VERSION,
    }

    accounts: Dict[str, MessengerAccount] = {}
    configured = extra.get("accounts")
    if isinstance(configured, dict):
        for key, value in configured.items():
            if isinstance(value, dict):
                account = _account_from_mapping(str(key), value, defaults)
                accounts[account.account_id] = account
    elif isinstance(configured, list):
        for index, value in enumerate(configured, start=1):
            if isinstance(value, dict):
                account = _account_from_mapping(str(value.get("id") or index), value, defaults)
                accounts[account.account_id] = account

    default_account = _default_account_from_extra(extra)
    if default_account is not None:
        accounts.setdefault(default_account.account_id, default_account)

    return accounts


def _has_complete_account(extra: Optional[Dict[str, Any]]) -> bool:
    return any(account.complete for account in parse_accounts(extra).values())


def _message_type_for_attachment(kind: str) -> MessageType:
    return {
        "image": MessageType.PHOTO,
        "video": MessageType.VIDEO,
        "audio": MessageType.VOICE,
        "file": MessageType.DOCUMENT,
        "fallback": MessageType.TEXT,
    }.get(kind, MessageType.TEXT)


def _extension_from_url(url: str) -> str:
    try:
        path = urlsplit(url).path
    except Exception:
        return ""
    suffix = Path(path).suffix.lower()
    return suffix if suffix and len(suffix) <= 12 else ""


def _filename_from_url(url: str, fallback: str) -> str:
    try:
        name = Path(urlsplit(url).path).name
    except Exception:
        name = ""
    return name or fallback


class _MessageDeduplicator:
    """Bounded inbound message/postback de-duplicator."""

    def __init__(self, max_size: int = DEDUP_MAX_SIZE):
        self._max_size = max_size
        self._seen: OrderedDict[str, float] = OrderedDict()

    def is_duplicate(self, key: str) -> bool:
        if not key:
            return False
        now = time.time()
        if key in self._seen:
            self._seen.move_to_end(key)
            return True
        self._seen[key] = now
        while len(self._seen) > self._max_size:
            self._seen.popitem(last=False)
        return False


class MessengerAdapter(BasePlatformAdapter):
    """Facebook Messenger webhook adapter."""

    splits_long_messages = True
    supports_code_blocks = False

    def __init__(self, config):
        super().__init__(config, Platform("messenger"))
        extra = getattr(config, "extra", {}) or {}
        self.webhook_host = _env_value("MESSENGER_HOST") or extra.get("host") or DEFAULT_WEBHOOK_HOST
        try:
            self.webhook_port = int(
                _env_value("MESSENGER_PORT") or extra.get("port") or DEFAULT_WEBHOOK_PORT
            )
        except (TypeError, ValueError):
            self.webhook_port = DEFAULT_WEBHOOK_PORT
        self._dm_policy = str(
            _env_value("MESSENGER_DM_POLICY")
            or extra.get("dm_policy")
            or extra.get("dmPolicy")
            or "pairing"
        ).strip().lower()
        self.dm_policy = self._dm_policy
        self.accounts = parse_accounts(extra)
        self._complete_accounts = {
            account_id: account
            for account_id, account in self.accounts.items()
            if account.complete
        }
        self._runner = None
        self._site = None
        self._http = None
        self._dedup = _MessageDeduplicator()

    @property
    def name(self) -> str:
        return "messenger"

    async def connect(self) -> bool:
        if not AIOHTTP_AVAILABLE or not HTTPX_AVAILABLE:
            logger.error("Messenger requires aiohttp and httpx")
            return False
        if not self._complete_accounts:
            logger.error("Messenger is missing page access token, app secret, or verify token")
            return False

        app = web.Application(client_max_size=WEBHOOK_BODY_MAX_BYTES)
        route_paths: Dict[str, MessengerAccount] = {}
        for account in self._complete_accounts.values():
            previous = route_paths.get(account.webhook_path)
            if previous is not None:
                logger.error(
                    "Duplicate Messenger webhook path %s for accounts %s and %s",
                    account.webhook_path,
                    previous.account_id,
                    account.account_id,
                )
                return False
            route_paths[account.webhook_path] = account
            app.router.add_get(account.webhook_path, self._build_verify_handler(account))
            app.router.add_post(account.webhook_path, self._build_webhook_handler(account))

        app.router.add_get("/messenger/health", self._health)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        try:
            self._site = web.TCPSite(self._runner, self.webhook_host, self.webhook_port)
            await self._site.start()
        except OSError as exc:
            await self._runner.cleanup()
            self._runner = None
            logger.error("Failed to start Messenger webhook on %s:%s: %s", self.webhook_host, self.webhook_port, exc)
            return False

        self._http = httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=True,
            event_hooks={"response": [_ssrf_redirect_guard]},
        )
        self._mark_connected()
        logger.info(
            "Messenger webhook listening on %s:%s for %d account(s)",
            self.webhook_host,
            self.webhook_port,
            len(self._complete_accounts),
        )
        return True

    async def disconnect(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
            self._site = None
        self._mark_disconnected()

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        account, recipient_id = self._resolve_outbound_account(chat_id, metadata)
        if account is None or not recipient_id:
            return SendResult(success=False, error="Missing Messenger account or recipient ID")
        return await self._send_text_chunks(account, recipient_id, self.format_message(content or ""))

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        account, recipient_id = self._resolve_outbound_account(chat_id, metadata)
        if account is None or not recipient_id:
            return
        try:
            await self._send_sender_action(account, recipient_id, "typing_on")
        except Exception:
            logger.debug("Failed to send Messenger typing indicator", exc_info=True)

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        account, recipient_id = self._resolve_outbound_account(chat_id, metadata)
        if account is None or not recipient_id:
            return SendResult(success=False, error="Missing Messenger account or recipient ID")
        if not str(image_url).startswith(("http://", "https://")):
            return SendResult(success=False, error="Messenger image sending requires a public http(s) URL")
        image_result = await self._send_attachment_url(account, recipient_id, "image", image_url)
        if not image_result.success or not caption:
            return image_result
        caption_result = await self.send(chat_id, caption, reply_to=reply_to, metadata=metadata)
        return caption_result if not caption_result.success else image_result

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        account, recipient_id = self._resolve_outbound_account(chat_id, None)
        return {
            "name": recipient_id or chat_id,
            "type": "dm",
            "chat_id": chat_id,
            "account_id": account.account_id if account else None,
        }

    def format_message(self, content: str) -> str:
        return strip_markdown_for_messenger(content or "")

    def _build_verify_handler(self, account: MessengerAccount):
        async def handler(request):
            mode = request.query.get("hub.mode")
            token = request.query.get("hub.verify_token")
            challenge = request.query.get("hub.challenge")
            if mode == "subscribe" and token == account.verify_token and challenge is not None:
                return web.Response(text=challenge)
            return web.Response(status=403, text="Verification failed")

        return handler

    def _build_webhook_handler(self, account: MessengerAccount):
        async def handler(request):
            body = await request.read()
            if len(body) > WEBHOOK_BODY_MAX_BYTES:
                return web.Response(status=413, text="Webhook body too large")
            signature = request.headers.get("X-Hub-Signature-256", "")
            if not signature:
                return web.Response(status=400, text="Missing signature")
            if not verify_messenger_signature(body, signature, account.app_secret):
                return web.Response(status=401, text="Invalid signature")
            try:
                payload = json.loads(body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                return web.Response(status=400, text="Invalid JSON")
            asyncio.create_task(self._process_webhook_payload(account, payload))
            return web.json_response({"status": "ok"})

        return handler

    async def _health(self, request):
        return web.json_response({"ok": True, "platform": "messenger"})

    async def _process_webhook_payload(self, account: MessengerAccount, payload: Dict[str, Any]) -> None:
        if payload.get("object") != "page":
            logger.debug("Ignoring non-page Messenger webhook object: %r", payload.get("object"))
            return
        for entry in payload.get("entry") or []:
            messaging_events = entry.get("messaging") or []
            for event in messaging_events:
                try:
                    await self._process_messaging_event(account, event)
                except Exception:
                    logger.exception("Failed to process Messenger webhook event")

    async def _process_messaging_event(self, account: MessengerAccount, event: Dict[str, Any]) -> None:
        message = event.get("message") or {}
        if message.get("is_echo"):
            return
        if event.get("delivery") or event.get("read"):
            return
        if not (message or event.get("postback")):
            return

        dedup_key = self._dedup_key(account, event)
        if self._dedup.is_duplicate(dedup_key):
            return

        sender_id = str((event.get("sender") or {}).get("id") or "")
        if self._dm_policy == "disabled":
            logger.debug("Messenger DM policy disabled; dropping message from %s", sender_id or "?")
            return
        if sender_id:
            asyncio.create_task(self._best_effort_sender_action(account, sender_id, "mark_seen"))
            asyncio.create_task(self._best_effort_sender_action(account, sender_id, "typing_on"))

        event_obj = await self._to_message_event(account, event)
        if event_obj is not None:
            await self.handle_message(event_obj)

    def _dedup_key(self, account: MessengerAccount, event: Dict[str, Any]) -> str:
        message = event.get("message") or {}
        postback = event.get("postback") or {}
        sender_id = str((event.get("sender") or {}).get("id") or "")
        if message.get("mid"):
            return f"{account.account_id}:mid:{message['mid']}"
        if postback.get("mid"):
            return f"{account.account_id}:mid:{postback['mid']}"
        return f"{account.account_id}:evt:{sender_id}:{event.get('timestamp')}:{postback.get('payload') or postback.get('title')}"

    async def _to_message_event(self, account: MessengerAccount, event: Dict[str, Any]) -> Optional[MessageEvent]:
        sender_id = str((event.get("sender") or {}).get("id") or "")
        if not sender_id:
            return None
        message = event.get("message") or {}
        postback = event.get("postback") or {}
        message_id = str(message.get("mid") or postback.get("mid") or uuid.uuid4().hex)

        text_parts: List[str] = []
        message_type = MessageType.TEXT
        media_urls: List[str] = []
        media_types: List[str] = []

        if message.get("text"):
            text_parts.append(str(message.get("text") or ""))
        if postback:
            title = str(postback.get("title") or "").strip()
            payload = str(postback.get("payload") or "").strip()
            text_parts.append(payload or title or "[postback]")

        for attachment in message.get("attachments") or []:
            kind = str(attachment.get("type") or "file").lower()
            payload = attachment.get("payload") or {}
            if kind == "location":
                coordinates = payload.get("coordinates") or {}
                lat = coordinates.get("lat")
                long = coordinates.get("long")
                title = attachment.get("title") or "location"
                text_parts.append(f"[location: {title} {lat},{long}]")
                message_type = MessageType.LOCATION
                continue

            url = str(payload.get("url") or "").strip()
            if not url:
                text_parts.append(f"[{kind} attachment]")
                if message_type == MessageType.TEXT:
                    message_type = _message_type_for_attachment(kind)
                continue

            cached = await self._download_attachment(url, kind)
            if cached is not None:
                media_urls.append(cached.path)
                media_types.append(cached.media_type)
                text_parts.append(cached.context_note())
            else:
                text_parts.append(f"[{kind} attachment: {url}]")
            if message_type == MessageType.TEXT:
                message_type = _message_type_for_attachment(kind)

        chat_id = self._scoped_chat_id(account, sender_id)
        source = self.build_source(
            chat_id=chat_id,
            chat_name=sender_id,
            chat_type="dm",
            user_id=sender_id,
            user_name=sender_id,
            message_id=message_id,
        )
        text = "\n".join(part for part in text_parts if part).strip() or "[message]"
        return MessageEvent(
            text=text,
            message_type=message_type,
            source=source,
            raw_message=event,
            message_id=message_id,
            media_urls=media_urls,
            media_types=media_types,
        )

    async def _download_attachment(self, url: str, kind: str):
        if not HTTPX_AVAILABLE or not self._http:
            return None
        if not is_safe_url(url):
            logger.warning("Blocked unsafe Messenger attachment URL: %s", safe_url_for_log(url))
            return None
        media_type = "media"
        if kind in {"image", "video", "audio", "file"}:
            media_type = kind
        try:
            async with self._http.stream(
                "GET",
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; HermesAgent/1.0)",
                    "Accept": "*/*",
                },
            ) as response:
                response.raise_for_status()
                content_type = response.headers.get("content-type", "").split(";", 1)[0].strip()
                content_length = response.headers.get("content-length")
                if content_length:
                    try:
                        validate_inbound_media_size(int(content_length), media_type=media_type)
                    except ValueError:
                        raise
                    except Exception:
                        pass
                chunks: List[bytes] = []
                total = 0
                async for chunk in response.aiter_bytes():
                    total += len(chunk)
                    validate_inbound_media_size(total, media_type=media_type)
                    chunks.append(chunk)
            filename = _filename_from_url(url, f"messenger-{kind}{_extension_from_url(url)}")
            return cache_media_bytes(
                b"".join(chunks),
                filename=filename,
                mime_type=content_type,
                default_kind=kind if kind in {"image", "video", "audio"} else "document",
            )
        except Exception as exc:
            logger.warning("Failed to cache Messenger attachment %s: %s", safe_url_for_log(url), exc)
            return None

    def _resolve_outbound_account(
        self,
        chat_id: str,
        metadata: Optional[Dict[str, Any]],
    ) -> Tuple[Optional[MessengerAccount], str]:
        metadata = metadata or {}
        account_id = str(metadata.get("account_id") or metadata.get("messenger_account_id") or "").strip()
        recipient_id = str(metadata.get("recipient_id") or metadata.get("messenger_recipient_id") or chat_id or "").strip()

        if account_id:
            account = self._complete_accounts.get(account_id)
            if account and recipient_id.startswith(f"{account_id}:"):
                recipient_id = recipient_id.split(":", 1)[1]
            return account, recipient_id

        if ":" in recipient_id:
            prefix, rest = recipient_id.split(":", 1)
            account = self._complete_accounts.get(prefix)
            if account is not None:
                return account, rest

        account = self._complete_accounts.get("default")
        if account is not None:
            return account, recipient_id
        if len(self._complete_accounts) == 1:
            only_account = next(iter(self._complete_accounts.values()))
            return only_account, recipient_id
        return None, recipient_id

    def _scoped_chat_id(self, account: MessengerAccount, sender_id: str) -> str:
        if account.account_id == "default":
            return sender_id
        return f"{account.account_id}:{sender_id}"

    async def _send_text_chunks(self, account: MessengerAccount, recipient_id: str, text: str) -> SendResult:
        chunks = split_for_messenger(text) or [""]
        message_ids: List[str] = []
        responses: List[Any] = []
        for chunk in chunks:
            payload = {
                "recipient": {"id": recipient_id},
                "messaging_type": "RESPONSE",
                "message": {"text": chunk},
            }
            try:
                data = await self._post_graph(account, "/me/messages", payload)
            except MessengerGraphError as exc:
                return SendResult(
                    success=False,
                    error=str(exc),
                    raw_response=exc.detail,
                    retryable=exc.retryable,
                )
            except Exception as exc:
                return SendResult(success=False, error=str(exc), retryable=True)
            responses.append(data)
            mid = data.get("message_id") if isinstance(data, dict) else None
            if mid:
                message_ids.append(str(mid))
        return SendResult(
            success=True,
            message_id=message_ids[-1] if message_ids else None,
            raw_response=responses,
            continuation_message_ids=tuple(message_ids[:-1]),
        )

    async def _send_attachment_url(
        self,
        account: MessengerAccount,
        recipient_id: str,
        attachment_type: str,
        url: str,
    ) -> SendResult:
        payload = {
            "recipient": {"id": recipient_id},
            "messaging_type": "RESPONSE",
            "message": {
                "attachment": {
                    "type": attachment_type,
                    "payload": {"url": url, "is_reusable": True},
                },
            },
        }
        try:
            data = await self._post_graph(account, "/me/messages", payload)
        except MessengerGraphError as exc:
            return SendResult(
                success=False,
                error=str(exc),
                raw_response=exc.detail,
                retryable=exc.retryable,
            )
        except Exception as exc:
            return SendResult(success=False, error=str(exc), retryable=True)
        return SendResult(
            success=True,
            message_id=str(data.get("message_id") or "") if isinstance(data, dict) else None,
            raw_response=data,
        )

    async def _send_sender_action(
        self,
        account: MessengerAccount,
        recipient_id: str,
        action: str,
    ) -> None:
        payload = {
            "recipient": {"id": recipient_id},
            "sender_action": action,
        }
        await self._post_graph(account, "/me/messages", payload)

    async def _best_effort_sender_action(self, account: MessengerAccount, recipient_id: str, action: str) -> None:
        try:
            await self._send_sender_action(account, recipient_id, action)
        except Exception:
            logger.debug("Messenger sender action failed: %s", action, exc_info=True)

    async def _post_graph(self, account: MessengerAccount, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self._http is None:
            raise RuntimeError("Messenger HTTP client is not connected")
        url = f"{GRAPH_API_BASE}/{account.api_version}{path}"
        response = await self._http.post(
            url,
            headers={
                "Authorization": f"Bearer {account.page_access_token}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        try:
            data = response.json()
        except Exception:
            data = response.text
        if response.status_code >= 400:
            raise MessengerGraphError(response.status_code, data)
        return data if isinstance(data, dict) else {"response": data}


def check_requirements() -> bool:
    return AIOHTTP_AVAILABLE and HTTPX_AVAILABLE


def validate_config(config) -> bool:
    return _has_complete_account(getattr(config, "extra", {}) or {})


def is_connected(config) -> bool:
    return validate_config(config)


def _env_enablement() -> Optional[Dict[str, Any]]:
    if not (
        _env_value("MESSENGER_PAGE_ACCESS_TOKEN")
        and _env_value("MESSENGER_APP_SECRET")
        and _env_value("MESSENGER_VERIFY_TOKEN")
    ):
        return None
    seeded: Dict[str, Any] = {
        "dm_policy": _env_value("MESSENGER_DM_POLICY") or "pairing",
    }
    host = _env_value("MESSENGER_HOST")
    port = _env_value("MESSENGER_PORT")
    webhook_path = _env_value("MESSENGER_WEBHOOK_PATH")
    api_version = _env_value("MESSENGER_API_VERSION")
    home_channel = _env_value("MESSENGER_HOME_CHANNEL")
    if host:
        seeded["host"] = host
    if port:
        try:
            seeded["port"] = int(port)
        except ValueError:
            pass
    if webhook_path:
        seeded["webhook_path"] = _normalise_webhook_path(webhook_path)
    if api_version:
        seeded["api_version"] = _normalise_api_version(api_version)
    if home_channel:
        seeded["home_channel"] = {"chat_id": home_channel, "name": "Messenger"}
    return seeded


def _apply_yaml_config(yaml_cfg: dict, messenger_cfg: dict) -> dict | None:
    """Translate ``messenger:`` / ``platforms.messenger:`` YAML into extras.

    Env vars remain higher priority because ``parse_accounts`` reads env first.
    """
    seeded: Dict[str, Any] = {}
    extra = messenger_cfg.get("extra") if isinstance(messenger_cfg.get("extra"), dict) else {}
    for source in (messenger_cfg, extra):
        for key in (
            "page_access_token",
            "pageAccessToken",
            "token_file",
            "tokenFile",
            "app_secret",
            "appSecret",
            "secret_file",
            "secretFile",
            "verify_token",
            "verifyToken",
            "verify_token_file",
            "verifyTokenFile",
            "account_id",
            "accountId",
            "name",
            "webhook_path",
            "webhookPath",
            "api_version",
            "apiVersion",
            "dm_policy",
            "dmPolicy",
            "accounts",
        ):
            if key in source and source[key] not in (None, ""):
                seeded[key] = source[key]
        if "host" in source:
            seeded["host"] = source["host"]
        if "port" in source:
            try:
                seeded["port"] = int(source["port"])
            except (TypeError, ValueError):
                pass
    if "allow_from" in messenger_cfg and not _env_value("MESSENGER_ALLOWED_USERS"):
        value = messenger_cfg["allow_from"]
        os.environ["MESSENGER_ALLOWED_USERS"] = ",".join(_coerce_list(value))
    if "allow_all_users" in messenger_cfg and not _env_value("MESSENGER_ALLOW_ALL_USERS"):
        os.environ["MESSENGER_ALLOW_ALL_USERS"] = str(messenger_cfg["allow_all_users"]).lower()
    return seeded or None


async def _standalone_send(
    pconfig,
    chat_id: str,
    message: str,
    *,
    thread_id: Optional[str] = None,
    media_files: Optional[List[str]] = None,
    force_document: bool = False,
) -> Dict[str, Any]:
    extra = getattr(pconfig, "extra", {}) or {}
    accounts = {
        account_id: account
        for account_id, account in parse_accounts(extra).items()
        if account.complete
    }
    if not accounts:
        return {"error": "Messenger standalone send: missing credentials"}

    adapter = MessengerAdapter(pconfig)
    adapter._complete_accounts = accounts
    adapter._http = httpx.AsyncClient(timeout=30.0) if HTTPX_AVAILABLE else None
    try:
        if adapter._http is None:
            return {"error": "Messenger standalone send: httpx is not installed"}
        result = await adapter.send(chat_id, message)
        if not result.success:
            return {"error": result.error or "Messenger send failed"}
        if media_files:
            return {
                "success": True,
                "message_id": result.message_id,
                "warning": "Messenger standalone media delivery requires public URLs; sent text only",
            }
        return {"success": True, "message_id": result.message_id}
    finally:
        if adapter._http is not None:
            await adapter._http.aclose()
            adapter._http = None


def interactive_setup() -> None:
    print()
    print("Facebook Messenger setup")
    print("------------------------")
    print("Use a Meta app with Messenger enabled, then paste the Page token, app secret,")
    print("and webhook verify token. Set the Meta webhook URL to:")
    print("  https://<public-host>/messenger/webhook")
    print()

    try:
        from hermes_cli.config import get_env_value, save_env_value
        from hermes_cli.secret_prompt import masked_secret_prompt
    except ImportError:
        print("hermes_cli.config not available; set MESSENGER_* vars manually in ~/.hermes/.env")
        return

    def prompt_secret(var: str, label: str) -> None:
        existing = get_env_value(var)
        suffix = " [keep current]" if existing else ""
        try:
            value = masked_secret_prompt(f"{label}{suffix}: ")
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if value:
            save_env_value(var, value)

    prompt_secret("MESSENGER_PAGE_ACCESS_TOKEN", "Page access token")
    prompt_secret("MESSENGER_APP_SECRET", "App secret")
    prompt_secret("MESSENGER_VERIFY_TOKEN", "Webhook verify token")
    print("Done.")


def register(ctx) -> None:
    ctx.register_platform(
        name="messenger",
        label="Messenger",
        adapter_factory=lambda cfg: MessengerAdapter(cfg),
        check_fn=check_requirements,
        validate_config=validate_config,
        is_connected=is_connected,
        required_env=[
            "MESSENGER_PAGE_ACCESS_TOKEN",
            "MESSENGER_APP_SECRET",
            "MESSENGER_VERIFY_TOKEN",
        ],
        install_hint="pip install aiohttp httpx",
        setup_fn=interactive_setup,
        env_enablement_fn=_env_enablement,
        apply_yaml_config_fn=_apply_yaml_config,
        cron_deliver_env_var="MESSENGER_HOME_CHANNEL",
        standalone_sender_fn=_standalone_send,
        allowed_users_env="MESSENGER_ALLOWED_USERS",
        allow_all_env="MESSENGER_ALLOW_ALL_USERS",
        max_message_length=MAX_MESSAGE_LENGTH,
        emoji="💬",
        pii_safe=False,
        allow_update_command=True,
        platform_hint=(
            "You are chatting through Facebook Messenger Page DMs. Messenger "
            "does not render Markdown reliably, so send concise plain text with "
            "bare URLs. Each text message is capped at 2000 characters; the "
            "adapter chunks longer replies. Image sending requires a public "
            "http(s) URL."
        ),
    )

"""Text-only Microsoft Teams Bot Framework gateway adapter.

This is intentionally a narrow first slice:
- receive normal Bot Framework ``message`` activities over an aiohttp webhook
- validate inbound Bot Framework JWTs
- normalize text messages into Hermes ``MessageEvent`` / ``SessionSource``
- require mentions in non-DM conversations by default
- send outbound text replies through the Bot Framework conversation endpoint

It does not implement Graph, files, images, Adaptive Cards, channel history, or
standalone cron/send_message delivery.
"""

from __future__ import annotations

import asyncio
import contextlib
import html
import json
import logging
import re
from typing import Any, Dict, Iterable, Optional
from urllib.parse import quote, urlparse

import httpx

try:
    from aiohttp import web

    AIOHTTP_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised by requirement probe
    web = None  # type: ignore[assignment]
    AIOHTTP_AVAILABLE = False

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult
from gateway.platforms.msteams.auth import (
    BOT_FRAMEWORK_SCOPE,
    AuthError,
    BotFrameworkJWTValidator,
    BotFrameworkTokenProvider,
)

logger = logging.getLogger(__name__)

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 3978
DEFAULT_PATH = "/api/messages"
DEFAULT_MAX_BODY_BYTES = 1_048_576
MAX_MESSAGE_LENGTH = 28_000

_AT_TAG_RE = re.compile(r"<at>(.*?)</at>\s*", re.IGNORECASE | re.DOTALL)
_TRUSTED_SERVICE_URL_SUFFIXES = (
    ".trafficmanager.net",
    ".botframework.com",
    ".botframework.us",
    ".cloud.microsoft",
)


def check_msteams_requirements() -> bool:
    """Return True when the webhook dependency needed by Teams is present."""
    return AIOHTTP_AVAILABLE


def _coerce_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
        return default
    return bool(value)


def _coerce_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(part).strip() for part in value if str(part).strip()]
    return []


def _get_case_insensitive(mapping: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    lowered = {str(key).lower(): value for key, value in mapping.items()}
    for name in names:
        if name.lower() in lowered:
            return lowered[name.lower()]
    return None


def _entity_value(entity: Any, key: str) -> Any:
    if isinstance(entity, dict):
        return entity.get(key)
    return getattr(entity, key, None)


def _identity_value(identity: Any, key: str) -> Any:
    if isinstance(identity, dict):
        return _get_case_insensitive(identity, key)
    return getattr(identity, key, None)


def _normalize_token(value: str) -> str:
    return str(value or "").strip().lower()


def _matches_any(candidate: str, values: Iterable[str]) -> bool:
    normalized = _normalize_token(candidate)
    return bool(normalized and normalized in {_normalize_token(value) for value in values if value})


def strip_bot_mention(
    text: str,
    *,
    bot_ids: Iterable[str] = (),
    bot_names: Iterable[str] = (),
    entities: Iterable[Any] = (),
) -> tuple[str, bool]:
    """Strip Teams bot mentions from activity text.

    Teams usually sends bot mentions as ``<at>Bot Name</at>`` tags and also
    includes a ``mention`` entity.  The entity is authoritative when present;
    the tag/name fallback keeps local emulator and simplified tests usable.
    """
    if not text:
        return "", False

    bot_ids_set = {str(value) for value in bot_ids if value}
    bot_names_set = {str(value) for value in bot_names if value}
    mention_texts: set[str] = set()
    mentioned = False

    for entity in entities or []:
        entity_type = str(_entity_value(entity, "type") or "").lower()
        if entity_type != "mention":
            continue
        mentioned_identity = _entity_value(entity, "mentioned") or {}
        mentioned_id = str(_identity_value(mentioned_identity, "id") or "")
        mentioned_name = str(_identity_value(mentioned_identity, "name") or "")
        entity_text = str(_entity_value(entity, "text") or "")
        if _matches_any(mentioned_id, bot_ids_set) or _matches_any(mentioned_name, bot_names_set):
            mentioned = True
            if entity_text:
                mention_texts.add(entity_text)

    cleaned = str(text)
    for mention_text in sorted(mention_texts, key=len, reverse=True):
        cleaned = cleaned.replace(mention_text, "")

    def _strip_at(match: re.Match) -> str:
        nonlocal mentioned
        inner = html.unescape(match.group(1) or "").strip()
        if _matches_any(inner, bot_names_set) or _matches_any(inner, bot_ids_set):
            mentioned = True
            return ""
        # Preserve non-bot mentions as readable plain text.
        return f"@{inner} " if inner else ""

    cleaned = _AT_TAG_RE.sub(_strip_at, cleaned)
    cleaned = html.unescape(cleaned).strip()

    for bot_name in sorted(bot_names_set, key=len, reverse=True):
        if not bot_name:
            continue
        pattern = re.compile(rf"^@{re.escape(bot_name)}\b[\s,:-]*", re.IGNORECASE)
        cleaned, count = pattern.subn("", cleaned, count=1)
        if count:
            mentioned = True
            break

    return cleaned.strip(), mentioned


def _activities_url(service_url: str, conversation_id: str) -> str:
    """Build the Bot Framework activities endpoint for a conversation."""
    base = str(service_url or "").rstrip("/")
    parsed = urlparse(base)
    segments = [segment for segment in parsed.path.split("/") if segment]
    if "v3" not in segments:
        base = f"{base}/v3"
    return f"{base}/conversations/{quote(str(conversation_id), safe='')}/activities"


def _is_trusted_service_url(
    service_url: str,
    *,
    extra_trusted_hosts: Iterable[str] = (),
    allow_untrusted: bool = False,
) -> bool:
    if allow_untrusted:
        return True
    parsed = urlparse(str(service_url or ""))
    if parsed.scheme != "https":
        return False
    hostname = (parsed.hostname or "").lower()
    if not hostname:
        return False
    trusted = tuple(_TRUSTED_SERVICE_URL_SUFFIXES) + tuple(
        str(host).lower() for host in extra_trusted_hosts if str(host).strip()
    )
    return any(hostname == suffix.lstrip(".") or hostname.endswith(suffix) for suffix in trusted)


class MsTeamsAdapter(BasePlatformAdapter):
    """Microsoft Teams text adapter backed by Bot Framework activities."""

    MAX_MESSAGE_LENGTH = MAX_MESSAGE_LENGTH

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.MSTEAMS)
        extra = dict(config.extra or {})

        self._app_id = str(extra.get("app_id") or config.token or "").strip()
        self._app_password = str(extra.get("app_password") or config.api_key or "")
        self._tenant_id = str(extra.get("tenant_id") or "botframework.com").strip()
        self._bot_display_name = str(extra.get("bot_display_name") or "").strip()

        self._host = str(extra.get("host") or DEFAULT_HOST)
        self._port = int(extra.get("port") or DEFAULT_PORT)
        self._path = str(extra.get("path") or DEFAULT_PATH)
        if not self._path.startswith("/"):
            self._path = f"/{self._path}"
        self._max_body_bytes = int(extra.get("max_body_bytes") or DEFAULT_MAX_BODY_BYTES)

        self._require_mention = self._config_bool("require_mention", True)
        self._free_response_conversations = self._load_free_response_conversations()
        self._mention_patterns = self._compile_mention_patterns()
        self._trusted_service_url_hosts = _coerce_str_list(
            extra.get("trusted_service_url_hosts")
        )
        self._allow_untrusted_service_urls = _coerce_bool(
            extra.get("allow_untrusted_service_urls"), False
        )
        self._insecure_skip_auth = _coerce_bool(extra.get("insecure_skip_auth"), False)

        self._http_client: Optional[httpx.AsyncClient] = None
        self._token_provider: Optional[BotFrameworkTokenProvider] = None
        self._jwt_validator: Optional[BotFrameworkJWTValidator] = None
        self._runner = None
        self._site = None
        self._service_urls: dict[str, str] = {}
        self._save_lock = asyncio.Lock()

    @property
    def name(self) -> str:
        return "msteams"

    def _config_bool(self, key: str, default: bool) -> bool:
        if key in self.config.extra:
            return _coerce_bool(self.config.extra.get(key), default)
        env_key = f"MSTEAMS_{key.upper()}"
        import os

        return _coerce_bool(os.getenv(env_key), default)

    def _load_free_response_conversations(self) -> set[str]:
        import os

        raw = (
            self.config.extra.get("free_response_conversations")
            or self.config.extra.get("free_response_channels")
            or self.config.extra.get("free_response_chats")
            or os.getenv("MSTEAMS_FREE_RESPONSE_CONVERSATIONS")
            or os.getenv("MSTEAMS_FREE_RESPONSE_CHANNELS")
            or ""
        )
        return set(_coerce_str_list(raw))

    def _compile_mention_patterns(self) -> list[re.Pattern]:
        import os

        patterns = self.config.extra.get("mention_patterns")
        if patterns is None:
            raw = os.getenv("MSTEAMS_MENTION_PATTERNS", "").strip()
            if raw:
                try:
                    patterns = json.loads(raw)
                except Exception:
                    patterns = [part.strip() for part in raw.splitlines() if part.strip()]
                    if not patterns:
                        patterns = [part.strip() for part in raw.split(",") if part.strip()]
        if patterns is None:
            return []
        if isinstance(patterns, str):
            patterns = [patterns]
        if not isinstance(patterns, list):
            logger.warning(
                "[msteams] mention_patterns must be a list or string; got %s",
                type(patterns).__name__,
            )
            return []

        compiled: list[re.Pattern] = []
        for pattern in patterns:
            if not isinstance(pattern, str) or not pattern.strip():
                continue
            try:
                compiled.append(re.compile(pattern, re.IGNORECASE))
            except re.error as exc:
                logger.warning("[msteams] invalid mention pattern %r: %s", pattern, exc)
        return compiled

    async def connect(self) -> bool:
        if not AIOHTTP_AVAILABLE:
            self._set_fatal_error("msteams_aiohttp", "aiohttp is required for Microsoft Teams", retryable=False)
            return False
        if not self._app_id:
            self._set_fatal_error("msteams_config", "MSTEAMS_APP_ID is required", retryable=False)
            return False
        if not self._app_password:
            self._set_fatal_error("msteams_config", "MSTEAMS_APP_PASSWORD is required", retryable=False)
            return False

        self._http_client = httpx.AsyncClient(timeout=30.0, follow_redirects=True, trust_env=True)
        try:
            self._token_provider = BotFrameworkTokenProvider(
                app_id=self._app_id,
                app_password=self._app_password,
                tenant_id=self._tenant_id,
                http_client=self._http_client,
            )
            self._jwt_validator = BotFrameworkJWTValidator(
                self._app_id,
                self._http_client,
                cache_ttl_seconds=int(
                    self.config.extra.get("auth_cache_ttl_seconds")
                    or self.config.extra.get("auth_cache_ttl")
                    or 3600
                ),
            )
        except AuthError as exc:
            self._set_fatal_error("msteams_auth", str(exc), retryable=False)
            await self._close_http_client()
            return False

        if not self._acquire_platform_lock(
            "msteams-endpoint",
            f"{self._host}:{self._port}",
            f"Microsoft Teams endpoint {self._host}:{self._port}",
        ):
            await self._close_http_client()
            return False

        app = web.Application(client_max_size=self._max_body_bytes)
        app.router.add_get("/health", self._handle_health)
        app.router.add_post(self._path, self._handle_activity)

        self._runner = web.AppRunner(app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self._host, self._port)
        try:
            await self._site.start()
        except OSError as exc:
            self._set_fatal_error(
                "msteams_bind",
                f"Cannot bind {self._host}:{self._port}: {exc}",
                retryable=False,
            )
            await self.disconnect()
            return False

        self._mark_connected()
        logger.info("[msteams] listening on %s:%d%s", self._host, self._port, self._path)
        return True

    async def disconnect(self) -> None:
        if self._site is not None:
            with contextlib.suppress(Exception):
                await self._site.stop()
            self._site = None
        if self._runner is not None:
            with contextlib.suppress(Exception):
                await self._runner.cleanup()
            self._runner = None
        await self._close_http_client()
        self._release_platform_lock()
        self._mark_disconnected()

    async def _close_http_client(self) -> None:
        if self._http_client is not None:
            with contextlib.suppress(Exception):
                await self._http_client.aclose()
            self._http_client = None

    async def _handle_health(self, request: "web.Request") -> "web.Response":
        return web.json_response(
            {
                "status": "ok" if self._running else "starting",
                "platform": "msteams",
                "path": self._path,
            }
        )

    async def _handle_activity(self, request: "web.Request") -> "web.Response":
        content_length = request.content_length or 0
        if content_length > self._max_body_bytes:
            return web.json_response({"error": "Payload too large"}, status=413)

        try:
            raw = await request.read()
            body = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            return web.json_response({"error": "Invalid JSON"}, status=400)
        except Exception:
            logger.warning("[msteams] failed to read request body", exc_info=True)
            return web.json_response({"error": "Bad request"}, status=400)

        if not isinstance(body, dict):
            return web.json_response({"error": "Activity must be a JSON object"}, status=400)

        service_url = str(body.get("serviceUrl") or "")
        if service_url and not self._service_url_is_trusted(service_url):
            return web.json_response({"error": "Untrusted serviceUrl"}, status=400)

        if not self._insecure_skip_auth:
            if self._jwt_validator is None:
                return web.json_response({"error": "Auth not initialized"}, status=500)
            auth_header = request.headers.get("Authorization", "")
            ok = await self._jwt_validator.validate_authorization_header(
                auth_header,
                service_url=service_url or None,
            )
            if not ok:
                return web.json_response({"error": "Unauthorized"}, status=401)

        activity_type = str(body.get("type") or "").lower()
        if activity_type != "message":
            return web.json_response({"status": "ignored", "type": activity_type or "unknown"})

        chat_id = str((body.get("conversation") or {}).get("id") or "")
        if chat_id and service_url:
            await self._remember_service_url(chat_id, service_url)

        event = self._build_event(body)
        if event is None:
            return web.json_response({"status": "ignored"})

        try:
            await self.handle_message(event)
        except Exception:
            logger.exception("[msteams] handle_message failed")
            return web.json_response({"error": "Dispatch failed"}, status=500)

        return web.json_response({"status": "accepted"})

    def _service_url_is_trusted(self, service_url: str) -> bool:
        return _is_trusted_service_url(
            service_url,
            extra_trusted_hosts=self._trusted_service_url_hosts,
            allow_untrusted=self._allow_untrusted_service_urls or self._insecure_skip_auth,
        )

    async def _remember_service_url(self, chat_id: str, service_url: str) -> None:
        if not chat_id or not service_url:
            return
        if not self._service_url_is_trusted(service_url):
            logger.warning("[msteams] refusing to store untrusted serviceUrl for %s", chat_id)
            return
        if self._service_urls.get(chat_id) == service_url:
            return
        async with self._save_lock:
            self._service_urls[chat_id] = service_url

    def _build_event(self, activity: dict[str, Any]) -> Optional[MessageEvent]:
        conversation = activity.get("conversation") or {}
        sender = activity.get("from") or activity.get("from_property") or {}
        recipient = activity.get("recipient") or {}
        if not isinstance(conversation, dict) or not isinstance(sender, dict):
            return None

        chat_id = str(_get_case_insensitive(conversation, "id") or "")
        if not chat_id:
            return None
        raw_conversation_type = str(
            _get_case_insensitive(conversation, "conversationType", "conversation_type")
            or "personal"
        )
        chat_type = self._conversation_type_to_chat_type(raw_conversation_type)

        sender_id = str(
            _get_case_insensitive(sender, "aadObjectId", "aad_object_id")
            or _get_case_insensitive(sender, "id")
            or ""
        )
        sender_name = str(_get_case_insensitive(sender, "name") or "") or None
        if self._is_self_message(sender, recipient):
            return None

        channel_data = activity.get("channelData") or activity.get("channel_data") or {}
        if not isinstance(channel_data, dict):
            channel_data = {}
        team = channel_data.get("team") if isinstance(channel_data.get("team"), dict) else {}
        channel = (
            channel_data.get("channel")
            if isinstance(channel_data.get("channel"), dict)
            else {}
        )
        tenant = (
            channel_data.get("tenant")
            if isinstance(channel_data.get("tenant"), dict)
            else {}
        )
        team_id = str(team.get("id") or "") or None
        channel_id = str(channel.get("id") or "") or None
        tenant_id = str(tenant.get("id") or conversation.get("tenantId") or "") or None

        raw_text = str(activity.get("text") or "")
        bot_ids, bot_names = self._bot_identity(activity)
        cleaned_text, mentioned = strip_bot_mention(
            raw_text,
            bot_ids=bot_ids,
            bot_names=bot_names,
            entities=activity.get("entities") or [],
        )

        if not self._should_dispatch(
            chat_type=chat_type,
            chat_id=chat_id,
            channel_id=channel_id,
            team_id=team_id,
            text=cleaned_text,
            mentioned=mentioned,
        ):
            return None
        if not cleaned_text:
            return None

        chat_name = (
            str(channel.get("name") or "")
            or str(conversation.get("name") or "")
            or str(team.get("name") or "")
            or None
        )
        message_id = str(activity.get("id") or "")
        thread_id = channel_id if chat_type == "channel" and channel_id else None
        source = self.build_source(
            chat_id=chat_id,
            chat_name=chat_name,
            chat_type=chat_type,
            user_id=sender_id or None,
            user_name=sender_name,
            thread_id=thread_id,
            chat_id_alt=team_id,
            guild_id=team_id or tenant_id,
            parent_chat_id=team_id,
            message_id=message_id or None,
        )

        return MessageEvent(
            text=cleaned_text,
            message_type=MessageType.TEXT,
            source=source,
            raw_message=activity,
            message_id=message_id or None,
            reply_to_message_id=str(activity.get("replyToId") or "") or None,
        )

    @staticmethod
    def _conversation_type_to_chat_type(value: str) -> str:
        normalized = str(value or "").lower()
        if normalized == "personal":
            return "dm"
        if normalized == "groupchat":
            return "group"
        if normalized == "channel":
            return "channel"
        return "dm"

    def _bot_identity(self, activity: dict[str, Any]) -> tuple[set[str], set[str]]:
        recipient = activity.get("recipient") or {}
        bot_ids = {self._app_id, f"28:{self._app_id}" if self._app_id else ""}
        bot_names = {self._bot_display_name}
        if isinstance(recipient, dict):
            bot_ids.add(str(_get_case_insensitive(recipient, "id") or ""))
            bot_names.add(str(_get_case_insensitive(recipient, "name") or ""))
        return ({value for value in bot_ids if value}, {value for value in bot_names if value})

    def _is_self_message(self, sender: dict[str, Any], recipient: dict[str, Any]) -> bool:
        sender_id = str(_get_case_insensitive(sender, "id") or "")
        recipient_id = str(_get_case_insensitive(recipient, "id") or "")
        if sender_id and recipient_id and sender_id == recipient_id:
            return True
        return bool(
            self._app_id
            and sender_id
            and sender_id.lower() in {self._app_id.lower(), f"28:{self._app_id}".lower()}
        )

    def _should_dispatch(
        self,
        *,
        chat_type: str,
        chat_id: str,
        channel_id: str | None,
        team_id: str | None,
        text: str,
        mentioned: bool,
    ) -> bool:
        if chat_type == "dm":
            return True
        free_response_ids = {chat_id}
        if channel_id:
            free_response_ids.add(channel_id)
        if team_id:
            free_response_ids.add(team_id)
        if free_response_ids & self._free_response_conversations:
            return True
        if not self._require_mention:
            return True
        if mentioned:
            return True
        if text and any(pattern.search(text) for pattern in self._mention_patterns):
            return True
        return False

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        chunks = self.truncate_message(content, self.MAX_MESSAGE_LENGTH)
        last_result = SendResult(success=True)
        for index, chunk in enumerate(chunks):
            payload: dict[str, Any] = {"type": "message", "text": chunk}
            if reply_to and index == 0:
                payload["replyToId"] = reply_to
            last_result = await self._post_activity(chat_id, payload, metadata=metadata)
            if not last_result.success:
                return last_result
        return last_result

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        await self._post_activity(chat_id, {"type": "typing"}, metadata=metadata)

    async def _post_activity(
        self,
        chat_id: str,
        payload: dict[str, Any],
        *,
        metadata: Optional[dict[str, Any]] = None,
    ) -> SendResult:
        service_url = (
            (metadata or {}).get("service_url")
            or self._service_urls.get(str(chat_id))
        )
        if not service_url:
            return SendResult(
                success=False,
                error="unknown Teams serviceUrl for conversation",
                retryable=False,
            )
        if not self._service_url_is_trusted(str(service_url)):
            return SendResult(
                success=False,
                error="refusing to send to untrusted Teams serviceUrl",
                retryable=False,
            )
        if self._token_provider is None:
            return SendResult(
                success=False,
                error="Bot Framework token provider is not initialized",
                retryable=False,
            )
        if self._http_client is None:
            return SendResult(
                success=False,
                error="HTTP client is not initialized",
                retryable=True,
            )

        try:
            token = await self._token_provider.get_token(BOT_FRAMEWORK_SCOPE)
            response = await self._http_client.post(
                _activities_url(str(service_url), str(chat_id)),
                json=payload,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
            )
        except AuthError as exc:
            return SendResult(success=False, error=str(exc), retryable=False)
        except httpx.HTTPError as exc:
            return SendResult(success=False, error=str(exc), retryable=True)

        try:
            data = response.json()
        except Exception:
            data = {}
        if 200 <= response.status_code < 300:
            message_id = None
            if isinstance(data, dict):
                message_id = str(data.get("id") or "") or None
            return SendResult(success=True, message_id=message_id, raw_response=data)

        error_text = ""
        try:
            error_text = response.text[:500]
        except Exception:
            error_text = f"HTTP {response.status_code}"
        return SendResult(
            success=False,
            error=f"Bot Framework send failed ({response.status_code}): {error_text}",
            raw_response=data,
            retryable=response.status_code in {408, 429} or response.status_code >= 500,
        )

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {"name": str(chat_id), "type": "msteams", "chat_id": str(chat_id)}

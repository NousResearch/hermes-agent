from __future__ import annotations

import asyncio
import base64
import html
import json
import logging
import mimetypes
import os
import re
import socket as _socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
from urllib.parse import urlparse

import jwt

try:
    from aiohttp import web, ClientSession
    AIOHTTP_AVAILABLE = True
except ImportError:
    web = None  # type: ignore[assignment]
    ClientSession = Any  # type: ignore[misc,assignment]
    AIOHTTP_AVAILABLE = False

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    cache_document_from_bytes,
    cache_image_from_bytes,
)
from gateway.platforms.msteams_graph import BOTFRAMEWORK_SCOPE, GRAPH_SCOPE, MSTeamsBotClient, MSTeamsGraphClient
from gateway.platforms.msteams_mentions import (
    build_adaptive_card_attachment,
    build_mention_text_and_entities,
    build_poll_card,
    extract_activity_text,
    strip_leading_teams_mentions,
)
from gateway.platforms.msteams_state import ConversationRef, ConversationRegistry, default_msteams_state_path

logger = logging.getLogger(__name__)

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 3978
DEFAULT_PATH = "/api/messages"
DEFAULT_DM_POLICY = "pairing"
DEFAULT_GROUP_POLICY = "allowlist"
DEFAULT_REPLY_STYLE = "thread"
DEFAULT_MAX_BODY_BYTES = 1_048_576
DEFAULT_IDEMPOTENCY_TTL_SECONDS = 3600
DEFAULT_PENDING_UPLOAD_TTL_SECONDS = 300
DEFAULT_MEDIA_ALLOW_HOSTS = [
    "graph.microsoft.com", "graph.microsoft.us", "graph.microsoft.de", "graph.microsoft.cn",
    "sharepoint.com", "sharepoint.us", "sharepoint.de", "sharepoint.cn", "sharepoint-df.com",
    "1drv.ms", "onedrive.com", "teams.microsoft.com", "teams.cdn.office.net",
    "statics.teams.cdn.office.net", "office.com", "office.net", "asm.skype.com",
    "ams.skype.com", "media.ams.skype.com", "trafficmanager.net", "blob.core.windows.net",
    "azureedge.net", "microsoft.com",
]
DEFAULT_MEDIA_AUTH_ALLOW_HOSTS = [
    "api.botframework.com", "botframework.com", "smba.trafficmanager.net",
    "graph.microsoft.com", "graph.microsoft.us", "graph.microsoft.de", "graph.microsoft.cn",
]
VALID_DM_POLICIES = {"pairing", "allowlist", "open", "disabled"}
VALID_GROUP_POLICIES = {"allowlist", "open", "disabled"}
VALID_REPLY_STYLES = {"thread", "top-level"}
VALID_CHUNK_MODES = {"length", "newline"}
VALID_REACTION_TYPES = {"like", "heart", "laugh", "surprised", "sad", "angry"}
TRUSTED_SERVICE_URL_HOST_SUFFIXES = (
    ".trafficmanager.net",
    ".botframework.com",
    ".botframework.us",
    ".cloud.microsoft",
    ".azure.com",
)
BOTFRAMEWORK_OPENID_METADATA_URL = "https://login.botframework.com/v1/.well-known/openidconfiguration"
BOTFRAMEWORK_VALID_ISSUERS = {
    "https://api.botframework.com",
    "https://api.botframework.com/",
}
DEFAULT_AUTH_CACHE_TTL_SECONDS = 3600


class BotFrameworkJWTValidator:
    def __init__(self, app_id: str, session: ClientSession, cache_ttl_seconds: int = DEFAULT_AUTH_CACHE_TTL_SECONDS):
        self._app_id = app_id
        self._session = session
        self._cache_ttl_seconds = max(300, cache_ttl_seconds)
        self._openid_config: Optional[dict[str, Any]] = None
        self._openid_config_expiry = 0.0
        self._jwks: dict[str, Any] = {}
        self._jwks_expiry = 0.0
        self._lock = asyncio.Lock()

    async def validate(self, token: str, service_url: str | None = None) -> dict[str, Any]:
        header = jwt.get_unverified_header(token)
        key_id = str(header.get("kid") or "").strip()
        openid_config = await self._get_openid_config()
        jwks = await self._get_jwks(openid_config)
        signing_key = self._resolve_signing_key(key_id, jwks)
        payload = jwt.decode(
            token,
            signing_key,
            algorithms=[str(header.get("alg") or "RS256")],
            audience=self._app_id,
            issuer=list(BOTFRAMEWORK_VALID_ISSUERS),
            options={"require": ["exp", "iss", "aud"]},
        )
        if service_url:
            token_service_url = str(payload.get("serviceurl") or payload.get("serviceUrl") or "").strip()
            if token_service_url and token_service_url.rstrip("/") != service_url.rstrip("/"):
                raise jwt.InvalidTokenError("Bot Framework token serviceUrl mismatch")
        return payload

    async def _get_openid_config(self) -> dict[str, Any]:
        now = asyncio.get_running_loop().time()
        async with self._lock:
            if self._openid_config and now < self._openid_config_expiry:
                return self._openid_config
            async with self._session.get(BOTFRAMEWORK_OPENID_METADATA_URL) as resp:
                payload = await resp.json(content_type=None)
                if resp.status >= 400 or not isinstance(payload, dict):
                    raise RuntimeError(f"Failed to fetch Bot Framework OpenID config ({resp.status}): {payload}")
            self._openid_config = payload
            self._openid_config_expiry = now + self._cache_ttl_seconds
            return payload

    async def _get_jwks(self, openid_config: dict[str, Any]) -> dict[str, Any]:
        now = asyncio.get_running_loop().time()
        async with self._lock:
            if self._jwks and now < self._jwks_expiry:
                return self._jwks
            jwks_uri = str(openid_config.get("jwks_uri") or "").strip()
            if not jwks_uri:
                raise RuntimeError("Bot Framework OpenID config missing jwks_uri")
            async with self._session.get(jwks_uri) as resp:
                payload = await resp.json(content_type=None)
                if resp.status >= 400 or not isinstance(payload, dict):
                    raise RuntimeError(f"Failed to fetch Bot Framework JWKS ({resp.status}): {payload}")
            self._jwks = payload
            self._jwks_expiry = now + self._cache_ttl_seconds
            return payload

    @staticmethod
    def _resolve_signing_key(key_id: str, jwks: dict[str, Any]) -> Any:
        keys = jwks.get("keys") if isinstance(jwks, dict) else None
        if not isinstance(keys, list):
            raise RuntimeError("Bot Framework JWKS payload missing keys list")
        for key_payload in keys:
            if not isinstance(key_payload, dict):
                continue
            if key_id and str(key_payload.get("kid") or "") != key_id:
                continue
            return jwt.algorithms.RSAAlgorithm.from_jwk(json.dumps(key_payload))
        raise RuntimeError(f"Bot Framework signing key not found for kid={key_id!r}")


@dataclass
class EffectiveConversationPolicy:
    require_mention: bool
    reply_style: str
    dm_policy: str
    group_policy: str
    sender_allowed: bool
    route_allowed: bool

    @property
    def permitted(self) -> bool:
        return self.sender_allowed and self.route_allowed


@dataclass
class PendingUpload:
    upload_id: str
    conversation_id: str
    file_name: str
    file_bytes: bytes
    content_type: str
    caption: str = ""
    reply_to: Optional[str] = None
    created_at: float = 0.0


def check_msteams_requirements() -> bool:
    return AIOHTTP_AVAILABLE


def _normalize_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _normalize_bool(value: Any, default: bool) -> bool:
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
    return bool(value)


def _normalize_reply_style(value: Any, default: str = DEFAULT_REPLY_STYLE) -> str:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in VALID_REPLY_STYLES:
            return normalized
    return default


def _normalize_chunk_mode(value: Any, default: str = "length") -> str:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in VALID_CHUNK_MODES:
            return normalized
    return default


def _normalize_dm_policy(value: Any, default: str = DEFAULT_DM_POLICY) -> str:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in VALID_DM_POLICIES:
            return normalized
    return default


def _normalize_group_policy(value: Any, default: str = DEFAULT_GROUP_POLICY) -> str:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in VALID_GROUP_POLICIES:
            return normalized
    return default


def _is_trusted_service_url(service_url: str) -> bool:
    if not service_url:
        return False
    parsed = urlparse(service_url)
    if parsed.scheme != "https":
        return False
    hostname = (parsed.hostname or "").lower()
    if not hostname:
        return False
    return hostname.endswith(TRUSTED_SERVICE_URL_HOST_SUFFIXES)


def _hostname_matches_allowlist(hostname: str, allowlist: list[str]) -> bool:
    normalized_host = str(hostname or "").strip().lower()
    if not normalized_host:
        return False
    for entry in allowlist:
        normalized = str(entry or "").strip().lower()
        if not normalized:
            continue
        if normalized == "*":
            return True
        if normalized_host == normalized or normalized_host.endswith(f".{normalized}"):
            return True
    return False


class MSTeamsAdapter(BasePlatformAdapter):
    """Native Hermes Microsoft Teams adapter."""

    MAX_MESSAGE_LENGTH = 4000

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.MSTEAMS)
        extra = config.extra or {}
        self._host = str(extra.get("host") or DEFAULT_HOST)
        self._port = int(extra.get("port") or DEFAULT_PORT)
        self._path = str(extra.get("path") or DEFAULT_PATH)
        self._app_id = str(extra.get("app_id") or "")
        self._app_password = str(extra.get("app_password") or "")
        self._tenant_id = str(extra.get("tenant_id") or "")
        self._require_mention = _normalize_bool(extra.get("require_mention"), True)
        self._reply_style = _normalize_reply_style(extra.get("reply_style"), DEFAULT_REPLY_STYLE)
        self._dm_policy = _normalize_dm_policy(extra.get("dm_policy"), DEFAULT_DM_POLICY)
        self._group_policy = _normalize_group_policy(extra.get("group_policy"), DEFAULT_GROUP_POLICY)
        self._allow_from = _normalize_list(extra.get("allow_from"))
        self._group_allow_from = _normalize_list(extra.get("group_allow_from"))
        self._dangerously_allow_name_matching = _normalize_bool(extra.get("dangerously_allow_name_matching"), False)
        self._teams_config = extra.get("teams") if isinstance(extra.get("teams"), dict) else {}
        self._chunk_mode = _normalize_chunk_mode(extra.get("chunk_mode"), "length")
        self._text_chunk_limit = min(self.MAX_MESSAGE_LENGTH, max(1, int(extra.get("text_chunk_limit") or self.MAX_MESSAGE_LENGTH)))
        self._history_limit = max(0, int(extra.get("history_limit") or extra.get("historyLimit") or 50))
        self._dm_history_limit = max(0, int(extra.get("dm_history_limit") or extra.get("dmHistoryLimit") or self._history_limit))
        self._max_body_bytes = max(1024, int(extra.get("max_body_bytes") or DEFAULT_MAX_BODY_BYTES))
        self._idempotency_ttl = max(60, int(extra.get("idempotency_ttl_seconds") or DEFAULT_IDEMPOTENCY_TTL_SECONDS))
        self._state_path = Path(extra.get("state_path") or default_msteams_state_path())
        self._auth_cache_ttl_seconds = max(300, int(extra.get("auth_cache_ttl_seconds") or DEFAULT_AUTH_CACHE_TTL_SECONDS))
        self._pending_upload_ttl_seconds = max(60, int(extra.get("pending_upload_ttl_seconds") or DEFAULT_PENDING_UPLOAD_TTL_SECONDS))
        self._sharepoint_site_id = str(extra.get("share_point_site_id") or extra.get("sharePointSiteId") or "").strip()
        configured_media_allow_hosts = _normalize_list(extra.get("media_allow_hosts"))
        configured_media_auth_allow_hosts = _normalize_list(extra.get("media_auth_allow_hosts"))
        self._media_allow_hosts = configured_media_allow_hosts or list(DEFAULT_MEDIA_ALLOW_HOSTS)
        self._media_auth_allow_hosts = configured_media_auth_allow_hosts or list(DEFAULT_MEDIA_AUTH_ALLOW_HOSTS)
        self._seen_activities: dict[str, float] = {}
        self._pending_uploads: dict[str, PendingUpload] = {}
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None
        self._app: Optional[web.Application] = None
        self._http: Optional[ClientSession] = None
        self._bot: Optional[MSTeamsBotClient] = None
        self._graph: Optional[MSTeamsGraphClient] = None
        self._auth_validator: Optional[BotFrameworkJWTValidator] = None
        self._queue: asyncio.Queue[MessageEvent] = asyncio.Queue()
        self._poll_task: Optional[asyncio.Task] = None
        self._inflight_tasks: set[asyncio.Task] = set()
        self._conversation_locks: dict[str, asyncio.Lock] = {}
        self._conversations = ConversationRegistry.load_from_path(self._state_path)

    async def connect(self) -> bool:
        if not check_msteams_requirements():
            logger.warning("[MSTeams] aiohttp not installed")
            return False
        if not (self._app_id and self._app_password and self._tenant_id):
            logger.warning("[MSTeams] Missing app_id/app_password/tenant_id")
            return False
        try:
            with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                sock.connect(("127.0.0.1", self._port))
            logger.error("[MSTeams] Port %d already in use", self._port)
            return False
        except (ConnectionRefusedError, OSError):
            pass

        try:
            self._http = ClientSession()
            self._bot = MSTeamsBotClient(self._app_id, self._app_password, self._tenant_id, self._http)
            self._graph = MSTeamsGraphClient(self._app_id, self._app_password, self._tenant_id, self._http)
            self._auth_validator = BotFrameworkJWTValidator(self._app_id, self._http, self._auth_cache_ttl_seconds)
            self._app = web.Application()
            self._app.router.add_get("/health", self._handle_health)
            self._app.router.add_post(self._path, self._handle_activity)
            if self._path != DEFAULT_PATH:
                self._app.router.add_post(DEFAULT_PATH, self._handle_activity)
            self._runner = web.AppRunner(self._app)
            await self._runner.setup()
            self._site = web.TCPSite(self._runner, self._host, self._port)
            await self._site.start()
            self._poll_task = asyncio.create_task(self._poll_loop())
            self._mark_connected()
            logger.info("[MSTeams] HTTP server listening on %s:%s%s", self._host, self._port, self._path)
            return True
        except Exception:
            await self._cleanup()
            logger.exception("[MSTeams] Failed to start")
            return False

    async def disconnect(self) -> None:
        self._running = False
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None
        if self._inflight_tasks:
            for task in list(self._inflight_tasks):
                task.cancel()
            await asyncio.gather(*list(self._inflight_tasks), return_exceptions=True)
            self._inflight_tasks.clear()
        await self._cleanup()
        self._mark_disconnected()

    async def _cleanup(self) -> None:
        self._site = None
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
        self._app = None
        if self._http:
            await self._http.close()
            self._http = None
        self._bot = None
        self._graph = None
        self._auth_validator = None

    async def _handle_health(self, request: web.Request) -> web.Response:
        return web.json_response({"ok": True, "platform": "msteams"})

    async def _handle_activity(self, request: web.Request) -> web.Response:
        content_length = request.content_length or 0
        if content_length > self._max_body_bytes:
            return web.json_response({"error": "Payload too large"}, status=413)
        auth_error = await self._check_bearer_auth(request)
        if auth_error is not None:
            return auth_error
        raw = await request.read()
        if len(raw) > self._max_body_bytes:
            return web.json_response({"error": "Payload too large"}, status=413)
        try:
            payload = json.loads(raw.decode("utf-8") or "{}")
        except Exception:
            return web.json_response({"error": "Invalid JSON"}, status=400)
        if not isinstance(payload, dict):
            return web.json_response({"error": "Invalid payload"}, status=400)
        auth_claims = getattr(request, "_hermes_botframework_claims", None)
        if isinstance(auth_claims, dict):
            service_url_error = self._validate_service_url_claim(auth_claims, payload)
            if service_url_error is not None:
                return service_url_error
        duplicate_id = self._get_delivery_id(payload)
        if duplicate_id and self._is_duplicate_delivery(duplicate_id):
            return web.json_response({"status": "duplicate", "delivery_id": duplicate_id}, status=200)
        invoke_response = await self._handle_invoke_activity(payload)
        if invoke_response is not None:
            return invoke_response
        event = self._build_event(payload)
        if event is not None:
            self._debug_log_inbound_attachments(payload, event)
            event = await self._enrich_event_media(event, payload)
            event = await self._enrich_event_history(event, payload)
            await self._queue.put(event)
        return web.json_response({"status": "accepted"}, status=202)

    async def _check_bearer_auth(self, request: web.Request) -> Optional[web.Response]:
        auth_header = str(request.headers.get("Authorization") or "").strip()
        if not auth_header.startswith("Bearer "):
            return web.json_response({"error": "Unauthorized"}, status=401)
        token = auth_header[7:].strip()
        if not token:
            return web.json_response({"error": "Unauthorized"}, status=401)
        if self._auth_validator is None:
            return web.json_response({"error": "Microsoft Teams auth validator unavailable"}, status=503)
        try:
            claims = await self._auth_validator.validate(token)
        except Exception as exc:
            logger.warning("[MSTeams] Bot Framework auth validation failed: %s", exc)
            return web.json_response({"error": "Unauthorized"}, status=401)
        setattr(request, "_hermes_botframework_claims", claims)
        return None

    def _validate_service_url_claim(self, claims: Dict[str, Any], payload: Dict[str, Any]) -> Optional[web.Response]:
        token_service_url = str(claims.get("serviceurl") or claims.get("serviceUrl") or "").strip()
        payload_service_url = str(payload.get("serviceUrl") or "").strip()
        if token_service_url and payload_service_url and token_service_url.rstrip("/") != payload_service_url.rstrip("/"):
            logger.warning("[MSTeams] Bot Framework token serviceUrl mismatch: token=%s payload=%s", token_service_url, payload_service_url)
            return web.json_response({"error": "Unauthorized"}, status=401)
        return None

    def _get_delivery_id(self, payload: Dict[str, Any]) -> Optional[str]:
        activity_id = str(payload.get("id") or "").strip()
        conversation_id = str((payload.get("conversation") or {}).get("id") or "").strip()
        if not activity_id:
            return None
        return f"{conversation_id}:{activity_id}" if conversation_id else activity_id

    def _debug_log_inbound_attachments(self, activity: Dict[str, Any], event: MessageEvent) -> None:
        try:
            attachments = activity.get("attachments") if isinstance(activity.get("attachments"), list) else []
            if not attachments:
                return
            if getattr(event.source, "chat_type", "") != "dm":
                return
            summary = []
            for attachment in attachments:
                if not isinstance(attachment, dict):
                    continue
                content = attachment.get("content") if isinstance(attachment.get("content"), dict) else {}
                item = {
                    "contentType": attachment.get("contentType"),
                    "hasContentUrl": bool(attachment.get("contentUrl")),
                    "hasDownloadUrl": bool(content.get("downloadUrl")),
                    "fileType": content.get("fileType"),
                    "hasContentText": bool(content.get("text") or content.get("body") or content.get("content")),
                }
                summary.append(item)
            logger.debug("[MSTeams][debug] DM inbound attachments activity_id=%s summary=%s", activity.get("id"), json.dumps(summary, ensure_ascii=False))
        except Exception:
            logger.debug("[MSTeams] Failed to emit inbound attachment debug summary", exc_info=True)

    def _is_duplicate_delivery(self, delivery_id: str) -> bool:
        now = time.time()
        self._seen_activities = {
            key: seen_at
            for key, seen_at in self._seen_activities.items()
            if now - seen_at < self._idempotency_ttl
        }
        if delivery_id in self._seen_activities:
            return True
        self._seen_activities[delivery_id] = now
        return False

    def _persist_conversations(self) -> None:
        try:
            self._conversations.save_to_path(self._state_path)
        except Exception:
            logger.exception("[MSTeams] Failed to persist conversation state to %s", self._state_path)

    def _prune_pending_uploads(self) -> None:
        now = time.time()
        self._pending_uploads = {
            upload_id: pending
            for upload_id, pending in self._pending_uploads.items()
            if now - pending.created_at < self._pending_upload_ttl_seconds
        }

    async def _handle_invoke_activity(self, activity: Dict[str, Any]) -> Optional[web.Response]:
        if str(activity.get("type") or "").lower() != "invoke":
            return None
        if str(activity.get("name") or "").strip() != "fileConsent/invoke":
            return web.json_response({"status": 200}, status=200)
        return await self._handle_file_consent_invoke(activity)

    async def _handle_file_consent_invoke(self, activity: Dict[str, Any]) -> web.Response:
        self._prune_pending_uploads()
        value = activity.get("value") if isinstance(activity.get("value"), dict) else {}
        context = value.get("context") if isinstance(value.get("context"), dict) else {}
        upload_id = str(context.get("uploadId") or context.get("upload_id") or "").strip()
        if not upload_id:
            return web.json_response({"status": 400, "body": {"error": "Missing uploadId"}}, status=200)
        pending = self._pending_uploads.get(upload_id)
        if pending is None:
            return web.json_response({"status": 404, "body": {"error": "Upload request expired"}}, status=200)

        conversation_id = str((activity.get("conversation") or {}).get("id") or "").strip()
        if conversation_id and conversation_id != pending.conversation_id:
            return web.json_response({"status": 403, "body": {"error": "Upload conversation mismatch"}}, status=200)

        action = str(value.get("action") or "").strip().lower()
        if action == "decline":
            self._pending_uploads.pop(upload_id, None)
            return web.json_response({"status": 200}, status=200)
        if action != "accept":
            return web.json_response({"status": 400, "body": {"error": f"Unsupported file consent action: {action or 'unknown'}"}}, status=200)

        upload_info = value.get("uploadInfo") if isinstance(value.get("uploadInfo"), dict) else {}
        upload_url = str(upload_info.get("uploadUrl") or "").strip()
        content_url = str(upload_info.get("contentUrl") or "").strip()
        unique_id = str(upload_info.get("uniqueId") or "").strip()
        file_type = str(upload_info.get("fileType") or Path(pending.file_name).suffix.lstrip(".") or "bin").strip()
        upload_name = str(upload_info.get("name") or pending.file_name).strip() or pending.file_name
        if not upload_url or not content_url:
            return web.json_response({"status": 400, "body": {"error": "Missing uploadInfo.uploadUrl/contentUrl"}}, status=200)
        if not self._http or not self._bot:
            return web.json_response({"status": 503, "body": {"error": "MSTeams upload client unavailable"}}, status=200)

        try:
            async with self._http.put(upload_url, data=pending.file_bytes, headers={"Content-Type": pending.content_type or "application/octet-stream"}) as resp:
                if resp.status >= 400:
                    payload = await resp.text()
                    raise RuntimeError(f"File upload failed ({resp.status}): {payload}")

            ref = self._conversations.get(pending.conversation_id)
            if ref is None:
                return web.json_response({"status": 404, "body": {"error": "Conversation not found"}}, status=200)
            attachment = {
                "contentType": "application/vnd.microsoft.teams.card.file.info",
                "contentUrl": content_url,
                "name": upload_name,
                "content": {
                    "uniqueId": unique_id or upload_name,
                    "fileType": file_type or "bin",
                },
            }
            result = await self._bot.send_message(
                ref,
                pending.caption or "",
                reply_to=pending.reply_to,
                attachments=[attachment],
            )
            message_id = str(result.get("id") or result.get("activityId") or "") or None
            if message_id:
                self._conversations.register_sent_message(pending.conversation_id, message_id)
                self._persist_conversations()
            self._pending_uploads.pop(upload_id, None)
            return web.json_response({"status": 200}, status=200)
        except Exception as exc:
            logger.exception("[MSTeams] Failed to finalize file consent upload")
            return web.json_response({"status": 500, "body": {"error": str(exc)}}, status=200)

    @staticmethod
    def _parse_attachment_json(value: Any) -> Any:
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith("{") or stripped.startswith("["):
                try:
                    return json.loads(stripped)
                except Exception:
                    return value
        return value

    @classmethod
    def _extract_message_reference(cls, activity: Dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
        for attachment in activity.get("attachments") or []:
            if not isinstance(attachment, dict):
                continue
            if str(attachment.get("contentType") or "").strip().lower() != "messagereference":
                continue
            content = cls._parse_attachment_json(attachment.get("content"))
            if not isinstance(content, dict):
                continue
            reference = content.get("messageReference") if isinstance(content.get("messageReference"), dict) else content
            message_id = str(reference.get("messageId") or content.get("messageId") or "").strip() or None
            message_preview = str(reference.get("messagePreview") or content.get("messagePreview") or "").strip() or None
            return message_id, message_preview
        return None, None

    @staticmethod
    def _html_to_plain_text(value: str) -> str:
        text = re.sub(r"<br\s*/?>", "\n", value, flags=re.IGNORECASE)
        text = re.sub(r"</p\s*>", "\n", text, flags=re.IGNORECASE)
        text = re.sub(r"<[^>]+>", "", text)
        text = html.unescape(text)
        text = re.sub(r"\s+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @classmethod
    def _extract_reply_html_quote(cls, activity: Dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
        for attachment in activity.get("attachments") or []:
            if not isinstance(attachment, dict):
                continue
            content = attachment.get("content")
            if isinstance(content, dict):
                content = content.get("text") or content.get("body") or ""
            if not isinstance(content, str) or "http://schema.skype.com/Reply" not in content:
                continue
            sender_match = re.search(r'<strong[^>]*itemprop=["\']mri["\'][^>]*>(.*?)</strong>', content, flags=re.IGNORECASE | re.DOTALL)
            body_match = re.search(r'<p[^>]*itemprop=["\']copy["\'][^>]*>(.*?)</p>', content, flags=re.IGNORECASE | re.DOTALL)
            sender = cls._html_to_plain_text(sender_match.group(1)) if sender_match else None
            body = cls._html_to_plain_text(body_match.group(1)) if body_match else None
            if body:
                return sender, body
        return None, None

    @staticmethod
    def _extract_attachment_text(activity: Dict[str, Any]) -> Optional[str]:
        _, reply_html_body = MSTeamsAdapter._extract_reply_html_quote(activity)
        if reply_html_body:
            return reply_html_body
        attachments = activity.get("attachments") or []
        for attachment in attachments:
            if not isinstance(attachment, dict):
                continue
            if str(attachment.get("contentType") or "").strip().lower() == "messagereference":
                _, message_preview = MSTeamsAdapter._extract_message_reference({"attachments": [attachment]})
                if message_preview:
                    return message_preview
            for key in ("text", "contentText", "summary", "name"):
                value = attachment.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            content = MSTeamsAdapter._parse_attachment_json(attachment.get("content"))
            if isinstance(content, dict):
                for key in ("text", "content", "body", "subtitle", "title"):
                    value = content.get(key)
                    if isinstance(value, str) and value.strip():
                        return value.strip()
        return None

    @staticmethod
    def _extract_attachment_placeholders(activity: Dict[str, Any]) -> list[str]:
        placeholders: list[str] = []
        for attachment in activity.get("attachments") or []:
            if not isinstance(attachment, dict):
                continue
            if MSTeamsAdapter._attachment_is_image(attachment):
                placeholders.append("<media:image>")
            elif str(attachment.get("contentType") or "").strip():
                placeholders.append("<media:document>")
        return placeholders

    @staticmethod
    def _attachment_download_url(attachment: Dict[str, Any]) -> str:
        content_type = str(attachment.get("contentType") or "").strip().lower()
        if content_type == "application/vnd.microsoft.teams.file.download.info":
            content = attachment.get("content") if isinstance(attachment.get("content"), dict) else {}
            return str(content.get("downloadUrl") or attachment.get("contentUrl") or "").strip()
        return str(attachment.get("contentUrl") or "").strip()

    @staticmethod
    def _attachment_file_name(attachment: Dict[str, Any]) -> str:
        content = attachment.get("content") if isinstance(attachment.get("content"), dict) else {}
        return str(content.get("fileName") or attachment.get("name") or "attachment").strip() or "attachment"

    @staticmethod
    def _attachment_media_type(attachment: Dict[str, Any]) -> str:
        content_type = str(attachment.get("contentType") or "").strip().lower()
        if content_type == "application/vnd.microsoft.teams.file.download.info":
            content = attachment.get("content") if isinstance(attachment.get("content"), dict) else {}
            file_type = str(content.get("fileType") or "").strip().lower()
            file_name = str(content.get("fileName") or attachment.get("name") or "").strip().lower()
            guessed, _ = mimetypes.guess_type(file_name or (f"file.{file_type}" if file_type else ""))
            return (guessed or content_type or "application/octet-stream").lower()
        return content_type or "application/octet-stream"

    @classmethod
    def _attachment_is_image(cls, attachment: Dict[str, Any]) -> bool:
        media_type = cls._attachment_media_type(attachment)
        if media_type.startswith("image/"):
            return True
        file_name = cls._attachment_file_name(attachment).lower()
        guessed, _ = mimetypes.guess_type(file_name)
        return bool(guessed and guessed.startswith("image/"))

    def _is_allowed_media_url(self, url: str) -> bool:
        parsed = urlparse(str(url or "").strip())
        if parsed.scheme != "https":
            return False
        return _hostname_matches_allowlist(parsed.hostname or "", self._media_allow_hosts)

    def _is_auth_allowed_media_url(self, url: str) -> bool:
        parsed = urlparse(str(url or "").strip())
        if parsed.scheme != "https":
            return False
        return _hostname_matches_allowlist(parsed.hostname or "", self._media_auth_allow_hosts)

    async def _download_media_bytes(self, url: str) -> Optional[bytes]:
        if not self._http:
            return None
        async with self._http.get(url) as resp:
            if resp.status < 400:
                return await resp.read()
            if resp.status not in {401, 403} or not self._is_auth_allowed_media_url(url):
                logger.warning("[MSTeams] Failed to download attachment %s (%s)", url, resp.status)
                return None

        parsed = urlparse(url)
        hostname = (parsed.hostname or "").lower()
        token_scopes = [BOTFRAMEWORK_SCOPE, GRAPH_SCOPE]
        if hostname.endswith("graph.microsoft.com") or hostname.endswith("graph.microsoft.us") or hostname.endswith("graph.microsoft.de") or hostname.endswith("graph.microsoft.cn") or "sharepoint" in hostname:
            token_scopes = [GRAPH_SCOPE, BOTFRAMEWORK_SCOPE]
        for scope in token_scopes:
            try:
                token_provider = self._graph if scope == GRAPH_SCOPE else self._bot
                if token_provider is None:
                    continue
                token = await token_provider._get_token_for_scope(scope)
                async with self._http.get(url, headers={"Authorization": f"Bearer {token}"}) as retry_resp:
                    if retry_resp.status < 400:
                        return await retry_resp.read()
            except Exception:
                logger.debug("[MSTeams] Attachment auth retry failed for %s", url, exc_info=True)
        logger.warning("[MSTeams] Failed to download attachment %s after auth retry", url)
        return None

    async def _collect_media_from_graph_fallback(self, activity: Dict[str, Any]) -> tuple[list[str], list[str]]:
        if not self._graph:
            return [], []
        attachments = activity.get("attachments") if isinstance(activity.get("attachments"), list) else []
        if not any(str((attachment or {}).get("contentType") or "").strip().lower() == "text/html" for attachment in attachments if isinstance(attachment, dict)):
            return [], []

        media_urls: list[str] = []
        media_types: list[str] = []
        for message_path in self._graph.build_message_url_candidates(activity):
            try:
                recovered = await self._graph.download_message_media(message_path)
            except Exception:
                logger.debug("[MSTeams] Graph media fallback failed for %s", message_path, exc_info=True)
                continue
            for item in recovered:
                try:
                    graph_path = str(item.get("graph_path") or "").strip()
                    content_url = str(item.get("content_url") or "").strip()
                    content_type = str(item.get("content_type") or "application/octet-stream").strip().lower()
                    name = str(item.get("name") or "attachment")
                    if graph_path and content_url:
                        if not self._is_allowed_media_url(content_url):
                            continue
                        data = await self._graph._graph_get_bytes(graph_path)
                        if content_type.startswith("image/"):
                            ext = mimetypes.guess_extension(content_type if content_type != "image/*" else "image/png") or ".png"
                            local_path = cache_image_from_bytes(data, ext)
                        else:
                            local_path = cache_document_from_bytes(data, name)
                    else:
                        data = item.get("data")
                        if not isinstance(data, (bytes, bytearray)):
                            continue
                        if content_type.startswith("image/"):
                            ext = mimetypes.guess_extension(content_type if content_type != "image/*" else "image/png") or ".png"
                            local_path = cache_image_from_bytes(bytes(data), ext)
                        else:
                            local_path = cache_document_from_bytes(bytes(data), name)
                    media_urls.append(local_path)
                    media_types.append(content_type or "application/octet-stream")
                except Exception:
                    logger.debug("[MSTeams] Failed to cache Graph fallback media", exc_info=True)
            if media_urls:
                break
        return media_urls, media_types

    async def _collect_media_from_activity(self, activity: Dict[str, Any]) -> tuple[list[str], list[str]]:
        media_urls: list[str] = []
        media_types: list[str] = []
        for attachment in activity.get("attachments") or []:
            if not isinstance(attachment, dict):
                continue
            content_url = self._attachment_download_url(attachment)
            if not content_url or not self._is_allowed_media_url(content_url):
                continue
            content_type = self._attachment_media_type(attachment)
            name = self._attachment_file_name(attachment)
            try:
                data = await self._download_media_bytes(content_url)
                if data is None:
                    continue
                if content_type.startswith("image/"):
                    ext = mimetypes.guess_extension(content_type if content_type != "image/*" else "image/png") or ".png"
                    local_path = cache_image_from_bytes(data, ext)
                else:
                    local_path = cache_document_from_bytes(data, name)
                media_urls.append(local_path)
                media_types.append(content_type or "application/octet-stream")
            except Exception:
                logger.debug("[MSTeams] Failed to cache attachment from activity", exc_info=True)
        if not media_urls:
            return await self._collect_media_from_graph_fallback(activity)
        return media_urls, media_types

    def _build_inline_file_attachment(self, file_path: str, *, file_name: Optional[str] = None) -> dict[str, Any]:
        mime_type = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
        with open(file_path, "rb") as handle:
            encoded = base64.b64encode(handle.read()).decode("ascii")
        return {
            "contentType": mime_type,
            "contentUrl": f"data:{mime_type};base64,{encoded}",
            "name": file_name or os.path.basename(file_path),
        }

    def _build_inline_image_attachment(self, image_path: str) -> dict[str, Any]:
        return self._build_inline_file_attachment(image_path)

    def _build_file_info_attachment(self, content_url: str, file_name: str, *, unique_id: Optional[str] = None, file_type: Optional[str] = None) -> dict[str, Any]:
        ext = (file_type or Path(file_name).suffix.lstrip(".") or "bin").strip() or "bin"
        return {
            "contentType": "application/vnd.microsoft.teams.card.file.info",
            "contentUrl": content_url,
            "name": file_name,
            "content": {
                "uniqueId": unique_id or file_name,
                "fileType": ext,
            },
        }

    def _build_file_consent_attachment(self, file_name: str, file_size: int, upload_id: str) -> dict[str, Any]:
        context = {"filename": file_name, "uploadId": upload_id}
        return {
            "contentType": "application/vnd.microsoft.teams.card.file.consent",
            "name": file_name,
            "content": {
                "description": file_name,
                "sizeInBytes": int(file_size),
                "acceptContext": context,
                "declineContext": context,
            },
        }

    def _derive_reply_target(self, ref: ConversationRef, explicit_reply_to: Optional[str] = None) -> Optional[str]:
        if explicit_reply_to:
            return explicit_reply_to
        if (ref.reply_style or self._reply_style) == "top-level":
            return None
        return ref.activity_id

    async def _send_dm_file_via_consent(
        self,
        ref: ConversationRef,
        file_path: str,
        *,
        caption: Optional[str] = None,
        file_name: Optional[str] = None,
        reply_to: Optional[str] = None,
    ) -> SendResult:
        if not self._bot:
            return SendResult(success=False, error="MSTeams adapter is not connected")
        effective_name = file_name or os.path.basename(file_path)
        file_bytes = Path(file_path).read_bytes()
        content_type = mimetypes.guess_type(effective_name)[0] or "application/octet-stream"
        upload_id = f"upload-{int(time.time() * 1000)}-{os.urandom(4).hex()}"
        attachment = self._build_file_consent_attachment(effective_name, len(file_bytes), upload_id)
        effective_reply_to = self._derive_reply_target(ref, reply_to)
        self._pending_uploads[upload_id] = PendingUpload(
            upload_id=upload_id,
            conversation_id=ref.conversation_id,
            file_name=effective_name,
            file_bytes=file_bytes,
            content_type=content_type,
            caption=caption or "",
            reply_to=effective_reply_to,
            created_at=time.time(),
        )
        result = await self._bot.send_message(
            ref,
            caption or "",
            reply_to=effective_reply_to,
            attachments=[attachment],
        )
        message_id = str(result.get("id") or result.get("activityId") or "") or None
        if message_id:
            self._conversations.register_sent_message(ref.conversation_id, message_id)
            self._persist_conversations()
        return SendResult(success=True, message_id=message_id, raw_response={"pending_upload_id": upload_id, **(result if isinstance(result, dict) else {"raw": result})})

    async def _send_file_via_sharepoint(
        self,
        ref: ConversationRef,
        file_path: str,
        *,
        caption: Optional[str] = None,
        file_name: Optional[str] = None,
        reply_to: Optional[str] = None,
    ) -> SendResult:
        if not self._bot or not self._graph:
            return SendResult(success=False, error="MSTeams adapter is not connected")
        if not self._sharepoint_site_id:
            return SendResult(success=False, error="SharePoint site ID is not configured for Microsoft Teams file upload")
        effective_name = file_name or os.path.basename(file_path)
        file_bytes = Path(file_path).read_bytes()
        content_type = mimetypes.guess_type(effective_name)[0] or "application/octet-stream"
        try:
            upload = await self._graph.upload_file_to_sharepoint(
                self._sharepoint_site_id,
                effective_name,
                file_bytes,
                content_type=content_type,
            )
            item_id = str(upload.get("id") or "").strip()
            link = await self._graph.create_sharepoint_link(self._sharepoint_site_id, item_id) if item_id else {}
            content_url = str(upload.get("webDavUrl") or upload.get("webUrl") or ((link.get("link") or {}).get("webUrl") or "")).strip()
            if not content_url:
                raise RuntimeError("SharePoint upload did not return a usable content URL")
            etag = str(upload.get("eTag") or "").strip().strip('"')
            unique_id = etag.split(",")[-1] if etag else item_id or effective_name
            attachment = self._build_file_info_attachment(
                content_url,
                str(upload.get("name") or effective_name),
                unique_id=unique_id,
                file_type=Path(effective_name).suffix.lstrip("."),
            )
            effective_reply_to = self._derive_reply_target(ref, reply_to)
            result = await self._bot.send_message(
                ref,
                caption or "",
                reply_to=effective_reply_to,
                attachments=[attachment],
            )
            message_id = str(result.get("id") or result.get("activityId") or "") or None
            if message_id:
                self._conversations.register_sent_message(ref.conversation_id, message_id)
                self._persist_conversations()
            return SendResult(success=True, message_id=message_id, raw_response={"upload": upload, "share_link": link, "attachment": attachment})
        except Exception as exc:
            logger.exception("[MSTeams] Failed to send SharePoint file")
            return SendResult(success=False, error=str(exc), retryable=True)

    async def _enrich_event_media(self, event: MessageEvent, activity: Dict[str, Any]) -> MessageEvent:
        media_urls, media_types = await self._collect_media_from_activity(activity)
        event.media_urls = media_urls
        event.media_types = media_types
        if media_urls and event.message_type == MessageType.TEXT:
            primary_type = media_types[0] if media_types else ""
            if primary_type.startswith("image/"):
                event.message_type = MessageType.PHOTO
            elif primary_type.startswith("audio/"):
                event.message_type = MessageType.AUDIO
            else:
                event.message_type = MessageType.DOCUMENT
        return event

    async def _enrich_event_history(self, event: MessageEvent, activity: Dict[str, Any]) -> MessageEvent:
        if not self._graph:
            return event
        reply_to_id = str(activity.get("replyToId") or "").strip()
        if not reply_to_id:
            return event
        channel_data = activity.get("channelData") or {}
        team_id = str((channel_data.get("team") or {}).get("id") or "").strip()
        channel_id = str((channel_data.get("channel") or {}).get("id") or "").strip()
        if not team_id or not channel_id:
            return event
        try:
            thread_context = await self._graph.build_thread_context(team_id, channel_id, reply_to_id)
        except Exception:
            logger.debug("[MSTeams] Failed to fetch thread context", exc_info=True)
            return event
        if thread_context and thread_context not in event.text:
            event.text = f"{thread_context}{event.text}" if event.text else thread_context.rstrip()
        return event

    def _history_allowed_sender_ids(self, activity: Dict[str, Any], policy: EffectiveConversationPolicy) -> set[str] | None:
        channel_data = activity.get("channelData") or {}
        team_cfg, channel_cfg, teams_present = self._resolve_team_and_channel_config(activity)
        sender_ids: set[str] = set()
        if self._group_allow_from:
            if any(str(item).strip() == "*" for item in self._group_allow_from):
                return None
            sender_ids.update(str(item).strip() for item in self._group_allow_from if str(item).strip())
        elif self._allow_from:
            if any(str(item).strip() == "*" for item in self._allow_from):
                return None
            sender_ids.update(str(item).strip() for item in self._allow_from if str(item).strip())
        elif teams_present and (team_cfg or channel_cfg):
            return None
        sender_id = str((activity.get("from") or {}).get("aadObjectId") or (activity.get("from") or {}).get("id") or "").strip()
        if policy.sender_allowed and sender_id:
            sender_ids.add(sender_id)
        return sender_ids or None

    async def enrich_new_session_history(self, event: MessageEvent) -> MessageEvent:
        if not self._graph or event.is_command() or event.reply_to_message_id:
            return event
        raw_activity = event.raw_message if isinstance(event.raw_message, dict) else None
        if not raw_activity:
            return event
        chat_type = getattr(event.source, "chat_type", "")
        try:
            if chat_type == "dm":
                if self._dm_history_limit <= 0:
                    return event
                history_context = await self._graph.build_recent_chat_context(
                    event.source.chat_id,
                    current_message_id=event.message_id,
                    limit=self._dm_history_limit,
                    user_turns_only=True,
                )
            elif chat_type == "group":
                if self._history_limit <= 0:
                    return event
                policy = self._resolve_policy(raw_activity, chat_type)
                allowed_sender_ids = self._history_allowed_sender_ids(raw_activity, policy)
                history_context = await self._graph.build_recent_chat_context(
                    event.source.chat_id,
                    current_message_id=event.message_id,
                    limit=self._history_limit,
                    allowed_sender_ids=allowed_sender_ids,
                )
            elif chat_type == "channel":
                if self._history_limit <= 0:
                    return event
                channel_data = raw_activity.get("channelData") or {}
                team_id = str((channel_data.get("team") or {}).get("id") or "").strip()
                channel_id = str((channel_data.get("channel") or {}).get("id") or "").strip()
                if not team_id or not channel_id:
                    return event
                policy = self._resolve_policy(raw_activity, chat_type)
                allowed_sender_ids = self._history_allowed_sender_ids(raw_activity, policy)
                history_context = await self._graph.build_recent_channel_context(
                    team_id,
                    channel_id,
                    current_message_id=event.message_id,
                    limit=self._history_limit,
                    allowed_sender_ids=allowed_sender_ids,
                )
            else:
                return event
        except Exception:
            logger.debug("[MSTeams] Failed to fetch new-session history context", exc_info=True)
            return event
        if history_context and history_context not in event.text:
            event.text = f"{history_context}{event.text}" if event.text else history_context.rstrip()
        return event

    def _chunk_text_for_send(self, content: str) -> list[str]:
        if not content.strip():
            return [content]
        if self._chunk_mode == "newline":
            chunks = self._chunk_text_by_paragraph(content, self._text_chunk_limit)
        else:
            chunks = self._chunk_text_by_length(content, self._text_chunk_limit)
        cleaned = [chunk.strip() for chunk in chunks if chunk and chunk.strip()]
        return cleaned or [content.strip()]

    def _chunk_text_by_paragraph(self, content: str, max_length: int) -> list[str]:
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", content) if part and part.strip()]
        if not paragraphs:
            return self._chunk_text_by_length(content, max_length)
        chunks: list[str] = []
        for paragraph in paragraphs:
            chunks.extend(self._chunk_text_by_length(paragraph, max_length))
        return chunks

    def _chunk_text_by_length(self, content: str, max_length: int) -> list[str]:
        if len(content) <= max_length:
            return [content]

        fence_close = "\n```"
        chunks: list[str] = []
        remaining = content
        carry_lang: Optional[str] = None

        while remaining:
            prefix = f"```{carry_lang}\n" if carry_lang is not None else ""
            headroom = max_length - len(prefix) - len(fence_close)
            if headroom < 1:
                headroom = max_length // 2
            if len(prefix) + len(remaining) <= max_length:
                chunks.append(prefix + remaining)
                break

            region = remaining[:headroom]
            split_at = region.rfind("\n")
            if split_at < headroom // 2:
                split_at = region.rfind(" ")
            if split_at < 1:
                split_at = headroom

            candidate = remaining[:split_at]
            backtick_count = candidate.count("`") - candidate.count("\\`")
            if backtick_count % 2 == 1:
                last_bt = candidate.rfind("`")
                while last_bt > 0 and candidate[last_bt - 1] == "\\":
                    last_bt = candidate.rfind("`", 0, last_bt)
                if last_bt > 0:
                    safe_split = max(candidate.rfind(" ", 0, last_bt), candidate.rfind("\n", 0, last_bt))
                    if safe_split > headroom // 4:
                        split_at = safe_split

            chunk_body = remaining[:split_at]
            remaining = remaining[split_at:].lstrip()
            full_chunk = prefix + chunk_body

            in_code = carry_lang is not None
            lang = carry_lang or ""
            for line in chunk_body.split("\n"):
                stripped = line.strip()
                if stripped.startswith("```"):
                    if in_code:
                        in_code = False
                        lang = ""
                    else:
                        in_code = True
                        tag = stripped[3:].strip()
                        lang = tag.split()[0] if tag else ""

            if in_code:
                full_chunk += fence_close
                carry_lang = lang
            else:
                carry_lang = None

            chunks.append(full_chunk)

        return chunks

    async def _poll_loop(self) -> None:
        while self._running:
            event = await self._queue.get()
            task = asyncio.create_task(self._process_event(event))
            self._inflight_tasks.add(task)
            task.add_done_callback(self._inflight_tasks.discard)

    async def _process_event(self, event: MessageEvent) -> None:
        lock = self._conversation_locks.setdefault(event.source.chat_id, asyncio.Lock())
        async with lock:
            try:
                await self.handle_message(event)
            except Exception:
                logger.exception("[MSTeams] Failed to process inbound event")

    def _build_event(self, activity: Dict[str, Any]) -> Optional[MessageEvent]:
        if str(activity.get("type") or "").lower() != "message":
            return None
        conversation = activity.get("conversation") or {}
        conversation_id = str(conversation.get("id") or "").strip()
        if not conversation_id:
            return None
        service_url = str(activity.get("serviceUrl") or "").strip()
        if not _is_trusted_service_url(service_url):
            logger.warning("[MSTeams] Ignoring activity with untrusted serviceUrl: %s", service_url)
            return None
        channel_data = activity.get("channelData") or {}
        tenant = channel_data.get("tenant") or {}
        from_user = activity.get("from") or {}
        chat_type = self._chat_type(activity)
        chat_name = self._chat_name(activity)
        reply_to_id = str(activity.get("replyToId") or "") or None
        if not reply_to_id:
            reply_to_id, _ = self._extract_message_reference(activity)
        policy = self._resolve_policy(activity, chat_type)
        self._conversations.remember(
            ConversationRef(
                conversation_id=conversation_id,
                service_url=service_url,
                conversation_type=str(conversation.get("conversationType") or ""),
                chat_type=chat_type,
                tenant_id=str(tenant.get("id") or "") or None,
                team_id=str((channel_data.get("team") or {}).get("id") or "") or None,
                channel_id=str((channel_data.get("channel") or {}).get("id") or "") or None,
                activity_id=reply_to_id or str(activity.get("id") or "") or None,
                user_id=str(from_user.get("aadObjectId") or from_user.get("id") or "") or None,
                user_name=str(from_user.get("name") or "") or None,
                chat_name=chat_name,
                raw=activity,
                last_inbound_activity_id=str(activity.get("id") or "") or None,
                reply_style=policy.reply_style,
                require_mention=policy.require_mention,
            )
        )
        self._persist_conversations()
        if not self._should_process_activity(activity, chat_type, policy):
            return None

        source = self.build_source(
            chat_id=conversation_id,
            chat_name=chat_name,
            chat_type=chat_type,
            user_id=str(from_user.get("aadObjectId") or from_user.get("id") or "") or None,
            user_name=str(from_user.get("name") or "") or None,
            thread_id=reply_to_id,
        )
        text = extract_activity_text(activity)
        raw_text = str(activity.get("text") or "")
        if raw_text:
            normalized_activity_text = re.sub(
                r'<blockquote[^>]*itemtype=["\']http://schema\.skype\.com/Reply["\'][^>]*>.*?</blockquote>',
                " ",
                raw_text,
                flags=re.IGNORECASE | re.DOTALL,
            )
            normalized_activity_text = strip_leading_teams_mentions(normalized_activity_text)
            if normalized_activity_text != raw_text:
                text = extract_activity_text({**activity, "text": normalized_activity_text})
                if re.match(r"^\s*(?:<at>.*?</at>\s*)+/", raw_text, flags=re.IGNORECASE | re.DOTALL):
                    slash_index = text.find("/")
                    if slash_index >= 0:
                        text = text[slash_index:]
        _, reply_html_body = self._extract_reply_html_quote(activity)
        reply_context = self._extract_attachment_text(activity)
        if not reply_context:
            reply_context = reply_html_body
        if not reply_context:
            _, message_preview = self._extract_message_reference(activity)
            reply_context = message_preview
        if not reply_context and reply_to_id:
            placeholders = self._extract_attachment_placeholders(activity)
            if placeholders:
                reply_context = " ".join(placeholders)
        return MessageEvent(
            text=text,
            message_type=MessageType.TEXT,
            source=source,
            raw_message=activity,
            message_id=str(activity.get("id") or "") or None,
            reply_to_message_id=reply_to_id,
            reply_to_text=reply_context,
        )

    @staticmethod
    def _chat_type(activity: Dict[str, Any]) -> str:
        conversation = activity.get("conversation") or {}
        ctype = str(conversation.get("conversationType") or "").lower()
        if ctype == "personal":
            return "dm"
        if ctype in {"groupchat", "group"}:
            return "group"
        if activity.get("channelId") == "msteams":
            return "channel"
        return "group"

    @staticmethod
    def _chat_name(activity: Dict[str, Any]) -> str:
        conversation = activity.get("conversation") or {}
        if conversation.get("name"):
            return str(conversation.get("name"))
        channel_data = activity.get("channelData") or {}
        team = channel_data.get("team") or {}
        channel = channel_data.get("channel") or {}
        if team.get("name") and channel.get("name"):
            return f"{team['name']} / {channel['name']}"
        return str(activity.get("recipient", {}).get("name") or "Microsoft Teams")

    def _resolve_policy(self, activity: Dict[str, Any], chat_type: str) -> EffectiveConversationPolicy:
        sender_id = str((activity.get("from") or {}).get("aadObjectId") or (activity.get("from") or {}).get("id") or "")
        sender_name = str((activity.get("from") or {}).get("name") or "")
        team_cfg, channel_cfg, teams_present = self._resolve_team_and_channel_config(activity)
        require_mention = self._effective_bool(team_cfg, channel_cfg, "requireMention", self._require_mention)
        reply_style = self._effective_reply_style(team_cfg, channel_cfg)
        dm_policy = self._dm_policy
        group_policy = self._group_policy

        if chat_type == "dm":
            if dm_policy == "disabled":
                return EffectiveConversationPolicy(False, reply_style, dm_policy, group_policy, False, False)
            if dm_policy == "allowlist":
                allowed = self._matches_allowlist(sender_id, sender_name, self._allow_from)
                return EffectiveConversationPolicy(False, reply_style, dm_policy, group_policy, allowed, True)
            return EffectiveConversationPolicy(False, reply_style, dm_policy, group_policy, True, True)

        route_allowed = self._group_route_allowed(activity, team_cfg, channel_cfg, teams_present)
        sender_allowed = self._group_sender_allowed(sender_id, sender_name, group_policy, team_cfg, channel_cfg, teams_present)
        return EffectiveConversationPolicy(
            require_mention=require_mention,
            reply_style=reply_style,
            dm_policy=dm_policy,
            group_policy=group_policy,
            sender_allowed=sender_allowed,
            route_allowed=route_allowed,
        )

    def _resolve_team_and_channel_config(self, activity: Dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], bool]:
        channel_data = activity.get("channelData") or {}
        team = channel_data.get("team") or {}
        channel = channel_data.get("channel") or {}
        team_id = str(team.get("id") or "")
        channel_id = str(channel.get("id") or "")
        team_name = str(team.get("name") or "")
        channel_name = str(channel.get("name") or "")
        team_cfg = self._match_mapping(self._teams_config, team_id, team_name)
        if not isinstance(team_cfg, dict):
            team_cfg = {}
        channel_cfg: dict[str, Any] = {}
        channels_map = team_cfg.get("channels") if isinstance(team_cfg.get("channels"), dict) else {}
        if isinstance(channels_map, dict):
            matched = self._match_mapping(channels_map, channel_id, channel_name)
            if isinstance(matched, dict):
                channel_cfg = matched
        return team_cfg, channel_cfg, bool(self._teams_config)

    def _effective_bool(self, team_cfg: dict[str, Any], channel_cfg: dict[str, Any], key: str, default: bool) -> bool:
        if key in channel_cfg:
            return _normalize_bool(channel_cfg.get(key), default)
        if key in team_cfg:
            return _normalize_bool(team_cfg.get(key), default)
        return default

    def _effective_reply_style(self, team_cfg: dict[str, Any], channel_cfg: dict[str, Any]) -> str:
        if "replyStyle" in channel_cfg:
            return _normalize_reply_style(channel_cfg.get("replyStyle"), self._reply_style)
        if "replyStyle" in team_cfg:
            return _normalize_reply_style(team_cfg.get("replyStyle"), self._reply_style)
        return self._reply_style

    def _match_mapping(self, mapping: Any, stable_id: str, mutable_name: str) -> Optional[dict[str, Any]]:
        if not isinstance(mapping, dict):
            return None
        if stable_id and stable_id in mapping and isinstance(mapping[stable_id], dict):
            return mapping[stable_id]
        if self._dangerously_allow_name_matching and mutable_name and mutable_name in mapping and isinstance(mapping[mutable_name], dict):
            return mapping[mutable_name]
        return None

    def _matches_allowlist(self, sender_id: str, sender_name: str, allowed_values: Iterable[str]) -> bool:
        for entry in allowed_values:
            normalized = str(entry).strip()
            if not normalized:
                continue
            if normalized == "*":
                return True
            if sender_id and normalized == sender_id:
                return True
            if self._dangerously_allow_name_matching and sender_name and normalized.lower() == sender_name.lower():
                return True
        return False

    def _group_route_allowed(self, activity: Dict[str, Any], team_cfg: dict[str, Any], channel_cfg: dict[str, Any], teams_present: bool) -> bool:
        if self._group_policy == "disabled":
            return False
        if self._group_policy == "open":
            return True
        if teams_present:
            if channel_cfg:
                return True
            return bool(team_cfg and (not isinstance(team_cfg.get("channels"), dict) or not team_cfg.get("channels")))
        return bool(self._group_allow_from or self._allow_from)

    def _group_sender_allowed(self, sender_id: str, sender_name: str, group_policy: str, team_cfg: dict[str, Any], channel_cfg: dict[str, Any], teams_present: bool) -> bool:
        if group_policy == "disabled":
            return False
        if group_policy == "open":
            return True
        if self._group_allow_from:
            return self._matches_allowlist(sender_id, sender_name, self._group_allow_from)
        if self._allow_from:
            return self._matches_allowlist(sender_id, sender_name, self._allow_from)
        if teams_present:
            return bool(team_cfg or channel_cfg)
        return False

    def _is_bot_reply(self, activity: Dict[str, Any]) -> bool:
        conversation_id = str((activity.get("conversation") or {}).get("id") or "")
        reply_to_id = str(activity.get("replyToId") or "") or None
        return self._conversations.has_sent_activity(conversation_id, reply_to_id)

    async def _resolve_mentions_metadata(self, mentions: Optional[list[dict[str, Any]]]) -> Optional[list[dict[str, Any]]]:
        if not mentions:
            return mentions
        resolved: list[dict[str, Any]] = []
        for mention in mentions:
            if not isinstance(mention, dict):
                continue
            mention_id = str(mention.get("id") or "").strip()
            mention_name = str(mention.get("name") or "").strip()
            if mention_id and mention_name:
                resolved.append({"id": mention_id, "name": mention_name})
                continue
            if self._graph is None:
                continue
            query = (
                str(mention.get("query") or "").strip()
                or str(mention.get("email") or "").strip()
                or str(mention.get("userPrincipalName") or "").strip()
                or mention_name
            )
            if not query and mention_id:
                try:
                    member = await self._graph.get_member_info(mention_id)
                except Exception:
                    logger.debug("[MSTeams] Failed to resolve mention by member id", exc_info=True)
                    continue
                user = member.get("user") if isinstance(member, dict) else None
                if isinstance(user, dict) and user.get("id") and user.get("displayName"):
                    resolved.append({"id": user["id"], "name": user["displayName"]})
                continue
            if not query:
                continue
            try:
                users = await self._graph.search_users(query, limit=1)
            except Exception:
                logger.debug("[MSTeams] Failed to resolve mention query=%r", query, exc_info=True)
                continue
            if not users:
                continue
            user = users[0]
            user_id = str(user.get("id") or "").strip()
            user_name = mention_name or str(user.get("displayName") or "").strip() or query
            if user_id and user_name:
                resolved.append({"id": user_id, "name": user_name})
        return resolved

    @classmethod
    def resolve_persisted_conversation_ref(cls, config: PlatformConfig, chat_id: str) -> tuple[Optional[dict[str, Any]], str]:
        extra = config.extra or {}
        ref_map = extra.get("conversation_refs", {})
        ref_payload = ref_map.get(chat_id) if isinstance(ref_map, dict) else None
        state_path = str(extra.get("state_path") or default_msteams_state_path())
        if isinstance(ref_payload, dict) and ref_payload.get("service_url"):
            return ref_payload, state_path
        persisted_registry = ConversationRegistry.load_from_path(state_path)
        persisted_ref = persisted_registry.get(chat_id)
        if persisted_ref is not None:
            return persisted_ref.to_dict(), state_path
        return None, state_path

    def remember_conversation_ref(self, chat_id: str, ref_payload: dict[str, Any], *, thread_id: Optional[str] = None) -> None:
        self._conversations.remember(
            ConversationRef(
                conversation_id=chat_id,
                service_url=str(ref_payload.get("service_url")),
                conversation_type=str(ref_payload.get("conversation_type") or ""),
                chat_type=str(ref_payload.get("chat_type") or "group"),
                tenant_id=ref_payload.get("tenant_id"),
                team_id=ref_payload.get("team_id"),
                channel_id=ref_payload.get("channel_id"),
                activity_id=thread_id or ref_payload.get("activity_id"),
                chat_name=ref_payload.get("chat_name"),
            )
        )

    async def send_standalone(self, chat_id: str, message: str, media_files=None, *, thread_id: Optional[str] = None) -> dict[str, Any]:
        media_files = media_files or []
        ref_payload, state_path = self.resolve_persisted_conversation_ref(self.config, chat_id)
        if not isinstance(ref_payload, dict) or not ref_payload.get("service_url"):
            return {
                "error": (
                    "Microsoft Teams direct send requires a persisted conversation reference for this chat_id. "
                    f"Checked config.extra['conversation_refs'] and state file at {state_path}."
                )
            }

        self.remember_conversation_ref(chat_id, ref_payload, thread_id=thread_id)
        async with __import__("aiohttp").ClientSession() as session:
            from gateway.platforms import msteams_graph as msteams_graph_module

            self._http = session
            self._bot = msteams_graph_module.MSTeamsBotClient(self._app_id, self._app_password, self._tenant_id, session)
            self._graph = msteams_graph_module.MSTeamsGraphClient(self._app_id, self._app_password, self._tenant_id, session)
            try:
                result = None
                pending_upload_ids = []
                if message.strip():
                    result = await self.send(chat_id, message, reply_to=thread_id)
                    if not result.success:
                        return {"error": result.error or "Microsoft Teams send failed"}
                for media_path, is_voice in media_files:
                    ext = os.path.splitext(media_path)[1].lower()
                    if ext in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}:
                        result = await self.send_image_file(chat_id, media_path, caption="" if message.strip() else None)
                        if not result.success:
                            return {"error": result.error or "Microsoft Teams image send failed"}
                    elif ext in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
                        result = await self.send_video(chat_id, media_path, caption="" if message.strip() else None)
                        if not result.success:
                            return {"error": result.error or "Microsoft Teams video send failed"}
                    elif is_voice or ext in {".ogg", ".opus", ".wav", ".mp3", ".m4a", ".aac", ".flac"}:
                        result = await self.send_voice(chat_id, media_path, caption="" if message.strip() else None)
                        if not result.success:
                            return {"error": result.error or "Microsoft Teams audio send failed"}
                    else:
                        result = await self.send_document(chat_id, media_path, caption="" if message.strip() else None, file_name=os.path.basename(media_path))
                        if not result.success:
                            return {"error": result.error or "Microsoft Teams document send failed"}
                    if result and isinstance(result.raw_response, dict):
                        pending_upload_id = str(result.raw_response.get("pending_upload_id") or "").strip()
                        if pending_upload_id:
                            pending_upload_ids.append(pending_upload_id)
                if result and result.success:
                    payload = {"success": True, "message_id": result.message_id}
                    if pending_upload_ids:
                        payload["pending_upload_ids"] = pending_upload_ids
                    return payload
                return {"error": "Microsoft Teams send failed"}
            finally:
                self._http = None
                self._bot = None
                self._graph = None

    async def get_member_info(self, user_id: str) -> Dict[str, Any]:
        if not self._graph:
            raise RuntimeError("MSTeams Graph client is not connected")
        return await self._graph.get_member_info(user_id)

    async def search_users(self, query: str, *, limit: int = 5) -> list[Dict[str, Any]]:
        if not self._graph:
            raise RuntimeError("MSTeams Graph client is not connected")
        return await self._graph.search_users(query, limit=limit)

    def _should_process_activity(self, activity: Dict[str, Any], chat_type: str, policy: EffectiveConversationPolicy) -> bool:
        if not policy.permitted:
            return False
        if chat_type == "dm" or not policy.require_mention:
            return True
        if self._is_bot_reply(activity):
            return True
        entities = activity.get("entities") or []
        recipient = activity.get("recipient") or {}
        recipient_id = str(recipient.get("id") or "")
        recipient_name = str(recipient.get("name") or "").strip().lower()
        for entity in entities:
            if not isinstance(entity, dict) or str(entity.get("type") or "").lower() != "mention":
                continue
            mentioned = entity.get("mentioned") or {}
            mentioned_id = str(mentioned.get("id") or "")
            mentioned_name = str(mentioned.get("name") or "").strip().lower()
            if recipient_id and mentioned_id == recipient_id:
                return True
            if recipient_name and mentioned_name == recipient_name:
                return True
        raw_text = str(activity.get("text") or "")
        if recipient_name and recipient.get("name") and f"<at>{recipient.get('name')}</at>".lower() in raw_text.lower():
            return True
        return False

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        ref = self._conversations.get(chat_id)
        if ref is None:
            return SendResult(success=False, error=f"Unknown Teams conversation: {chat_id}")
        if not self._bot:
            return SendResult(success=False, error="MSTeams adapter is not connected")
        metadata = metadata or {}
        reply_style = _normalize_reply_style(metadata.get("reply_style"), ref.reply_style or self._reply_style)
        mentions = metadata.get("mentions") if isinstance(metadata.get("mentions"), list) else None
        mentions = await self._resolve_mentions_metadata(mentions)
        adaptive_card = metadata.get("adaptive_card") if isinstance(metadata.get("adaptive_card"), dict) else None
        poll = metadata.get("poll") if isinstance(metadata.get("poll"), dict) else None
        if poll and adaptive_card is None:
            adaptive_card = build_poll_card(
                str(poll.get("question") or ""),
                list(poll.get("options") or []),
                int(poll.get("max_selections") or 1),
                str(poll.get("poll_id") or "").strip() or None,
            )
        attachments = [build_adaptive_card_attachment(adaptive_card)] if adaptive_card else None
        prepared_content, entities = build_mention_text_and_entities(content, mentions)
        if adaptive_card:
            entities = None
            chunks = [""]
        else:
            chunks = self._chunk_text_for_send(prepared_content)
        last_message_id = None
        try:
            current_reply_to = reply_to
            if current_reply_to is None and reply_style != "top-level":
                current_reply_to = ref.activity_id
            for index, chunk in enumerate(chunks):
                result = await self._bot.send_message(
                    ref,
                    chunk,
                    reply_to=current_reply_to,
                    entities=entities if index == 0 else None,
                    attachments=attachments if index == 0 else None,
                )
                message_id = str(result.get("id") or result.get("activityId") or "") or None
                if message_id:
                    self._conversations.register_sent_message(chat_id, message_id)
                    self._persist_conversations()
                    last_message_id = message_id
            return SendResult(success=True, message_id=last_message_id, raw_response={"chunks": len(chunks), "reply_style": reply_style, "has_mentions": bool(mentions), "has_adaptive_card": bool(adaptive_card)})
        except Exception as exc:
            logger.exception("[MSTeams] Failed to send message")
            return SendResult(success=False, error=str(exc), retryable=True)

    async def send_image_file(self, chat_id: str, image_path: str, caption: Optional[str] = None, metadata=None) -> SendResult:
        if not os.path.exists(image_path):
            return SendResult(success=False, error=f"Image file not found: {image_path}")
        ref = self._conversations.get(chat_id)
        if ref is None:
            return SendResult(success=False, error=f"Unknown Teams conversation: {chat_id}")
        if not self._bot:
            return SendResult(success=False, error="MSTeams adapter is not connected")
        try:
            attachment = self._build_inline_image_attachment(image_path)
            result = await self._bot.send_message(
                ref,
                caption or "",
                reply_to=ref.activity_id if (ref.reply_style or self._reply_style) != "top-level" else None,
                attachments=[attachment],
            )
            message_id = str(result.get("id") or result.get("activityId") or "") or None
            if message_id:
                self._conversations.register_sent_message(chat_id, message_id)
                self._persist_conversations()
            return SendResult(success=True, message_id=message_id, raw_response=result)
        except Exception as exc:
            logger.exception("[MSTeams] Failed to send image file")
            return SendResult(success=False, error=str(exc), retryable=True)

    async def send_document(
        self,
        chat_id: str,
        file_path: str,
        caption: Optional[str] = None,
        file_name: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata=None,
    ) -> SendResult:
        if not os.path.exists(file_path):
            return SendResult(success=False, error=f"File not found: {file_path}")
        ref = self._conversations.get(chat_id)
        if ref is None:
            return SendResult(success=False, error=f"Unknown Teams conversation: {chat_id}")
        if not self._bot:
            return SendResult(success=False, error="MSTeams adapter is not connected")
        try:
            if ref.chat_type == "dm":
                return await self._send_dm_file_via_consent(
                    ref,
                    file_path,
                    caption=caption,
                    file_name=file_name,
                    reply_to=reply_to,
                )
            if self._sharepoint_site_id and self._graph:
                return await self._send_file_via_sharepoint(
                    ref,
                    file_path,
                    caption=caption,
                    file_name=file_name,
                    reply_to=reply_to,
                )
            attachment = self._build_inline_file_attachment(file_path, file_name=file_name)
            effective_reply_to = self._derive_reply_target(ref, reply_to)
            result = await self._bot.send_message(
                ref,
                caption or "",
                reply_to=effective_reply_to,
                attachments=[attachment],
            )
            message_id = str(result.get("id") or result.get("activityId") or "") or None
            if message_id:
                self._conversations.register_sent_message(chat_id, message_id)
                self._persist_conversations()
            return SendResult(success=True, message_id=message_id, raw_response=result)
        except Exception as exc:
            logger.exception("[MSTeams] Failed to send document")
            return SendResult(success=False, error=str(exc), retryable=True)

    async def edit_message(
        self,
        chat_id: str,
        message_id: str,
        content: str,
    ) -> SendResult:
        ref = self._conversations.get(chat_id)
        if ref is None:
            return SendResult(success=False, error=f"Unknown Teams conversation: {chat_id}")
        if not self._bot:
            return SendResult(success=False, error="MSTeams adapter is not connected")
        try:
            result = await self._bot.update_message(ref, message_id, content)
            return SendResult(success=True, message_id=message_id, raw_response=result)
        except Exception as exc:
            logger.exception("[MSTeams] Failed to edit message")
            return SendResult(success=False, error=str(exc), retryable=True)

    async def send_voice(self, chat_id: str, audio_path: str, caption: Optional[str] = None, reply_to: Optional[str] = None, **kwargs) -> SendResult:
        return await self.send_document(chat_id, audio_path, caption=caption, file_name=os.path.basename(audio_path), reply_to=reply_to)

    async def send_video(self, chat_id: str, video_path: str, caption: Optional[str] = None, reply_to: Optional[str] = None, **kwargs) -> SendResult:
        return await self.send_document(chat_id, video_path, caption=caption, file_name=os.path.basename(video_path), reply_to=reply_to)

    async def delete_message(self, chat_id: str, message_id: str) -> SendResult:
        ref = self._conversations.get(chat_id)
        if ref is None:
            return SendResult(success=False, error=f"Unknown Teams conversation: {chat_id}")
        if not self._bot:
            return SendResult(success=False, error="MSTeams adapter is not connected")
        try:
            await self._bot.delete_message(ref, message_id)
            return SendResult(success=True, message_id=message_id)
        except Exception as exc:
            logger.exception("[MSTeams] Failed to delete message")
            return SendResult(success=False, error=str(exc), retryable=True)

    async def get_message(self, chat_id: str, message_id: str) -> Dict[str, Any]:
        ref = self._conversations.get(chat_id)
        if ref is None or not self._graph:
            raise RuntimeError(f"Unknown Teams conversation: {chat_id}")
        return await self._graph.get_message(ref, message_id)

    async def pin_message(self, chat_id: str, message_id: str) -> Dict[str, Any]:
        ref = self._conversations.get(chat_id)
        if ref is None or not self._graph:
            raise RuntimeError(f"Unknown Teams conversation: {chat_id}")
        return await self._graph.pin_message(ref, message_id)

    async def unpin_message(self, chat_id: str, pinned_message_id: str) -> Dict[str, Any]:
        ref = self._conversations.get(chat_id)
        if ref is None or not self._graph:
            raise RuntimeError(f"Unknown Teams conversation: {chat_id}")
        return await self._graph.unpin_message(ref, pinned_message_id)

    async def list_pins(self, chat_id: str) -> Dict[str, Any]:
        ref = self._conversations.get(chat_id)
        if ref is None or not self._graph:
            raise RuntimeError(f"Unknown Teams conversation: {chat_id}")
        return await self._graph.list_pins(ref)

    async def add_reaction(self, chat_id: str, message_id: str, reaction_type: str) -> Dict[str, Any]:
        normalized = str(reaction_type or "").strip().lower()
        if normalized not in VALID_REACTION_TYPES:
            raise RuntimeError(f"Invalid reaction type: {reaction_type}")
        ref = self._conversations.get(chat_id)
        if ref is None or not self._graph:
            raise RuntimeError(f"Unknown Teams conversation: {chat_id}")
        return await self._graph.set_reaction(ref, message_id, normalized)

    async def remove_reaction(self, chat_id: str, message_id: str, reaction_type: str) -> Dict[str, Any]:
        normalized = str(reaction_type or "").strip().lower()
        if normalized not in VALID_REACTION_TYPES:
            raise RuntimeError(f"Invalid reaction type: {reaction_type}")
        ref = self._conversations.get(chat_id)
        if ref is None or not self._graph:
            raise RuntimeError(f"Unknown Teams conversation: {chat_id}")
        return await self._graph.unset_reaction(ref, message_id, normalized)

    async def list_reactions(self, chat_id: str, message_id: str) -> Dict[str, Any]:
        ref = self._conversations.get(chat_id)
        if ref is None or not self._graph:
            raise RuntimeError(f"Unknown Teams conversation: {chat_id}")
        return await self._graph.list_reactions(ref, message_id)

    async def search_messages(self, chat_id: str, query: str, *, from_display_name: Optional[str] = None, limit: int = 25) -> Dict[str, Any]:
        ref = self._conversations.get(chat_id)
        if ref is None or not self._graph:
            raise RuntimeError(f"Unknown Teams conversation: {chat_id}")
        return await self._graph.search_messages(ref, query, from_display_name=from_display_name, limit=limit)

    async def list_channels(self, team_id: str) -> Dict[str, Any]:
        if not self._graph:
            raise RuntimeError("MSTeams Graph client is not connected")
        return await self._graph.list_channels(team_id)

    async def get_channel_info(self, team_id: str, channel_id: str) -> Dict[str, Any]:
        if not self._graph:
            raise RuntimeError("MSTeams Graph client is not connected")
        return await self._graph.get_channel_info(team_id, channel_id)

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        ref = self._conversations.get(chat_id)
        if ref is None or not self._bot:
            return None
        try:
            await self._bot.send_typing(ref)
        except Exception:
            logger.debug("[MSTeams] Failed to send typing indicator", exc_info=True)
        return None

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        ref = self._conversations.get(chat_id)
        if ref is None:
            return {"name": chat_id, "type": "unknown", "chat_id": chat_id}
        return {"name": ref.chat_name or chat_id, "type": ref.chat_type, "chat_id": chat_id}

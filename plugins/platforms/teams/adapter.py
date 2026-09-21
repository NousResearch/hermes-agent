"""Microsoft Teams adapter: microsoft-teams-apps SDK for auth/activity processing, an aiohttp
webhook server for inbound, ``App.send()`` for proactive sends.

Requires the ``teams`` extra (auto-installed by the gateway on first start, or
``<hermes-venv>/bin/pip install microsoft-teams-apps aiohttp``) and credentials via env
(TEAMS_CLIENT_ID / TEAMS_CLIENT_SECRET / TEAMS_TENANT_ID, optional TEAMS_PORT) or
``platforms.teams.extra`` in config.yaml (``client_id`` / ``client_secret`` / ``tenant_id`` / ``port``).
"""

from __future__ import annotations

import asyncio
# microsoft-teams-apps calls ``load_dotenv(find_dotenv(usecwd=True))`` at ``microsoft_teams.apps.app``
# import time. Importing it during plugin discovery / ``TeamsSummaryWriter`` imports would pollute process
# ``os.environ`` from a cwd-discovered ``.env`` (#62935). Detect presence via find_spec only; bind symbols
# in ``check_teams_requirements()`` behind a dotenv no-op.
import importlib.util
import dataclasses
import inspect
import json
import logging
import os
import re
import sys
import uuid
from collections import deque
from contextlib import contextmanager, suppress
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, Optional
from urllib.parse import quote, urlparse

try:
    from aiohttp import web
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    web = None  # type: ignore[assignment]


def _probe_teams_sdk_available() -> bool:
    """True when ``microsoft_teams.apps`` is on sys.path, without importing it: the SDK loads a cwd
    ``.env`` at import, so ``check_teams_requirements()`` binds symbols behind a dotenv no-op.
    Sibling packages share the namespace, so probe the parent first — ``find_spec`` of the child
    raises on 3.11+ if the parent is absent."""
    try:
        find_spec = importlib.util.find_spec
        return find_spec("microsoft_teams") is not None and find_spec("microsoft_teams.apps") is not None
    except (ValueError, ModuleNotFoundError, ImportError):
        return "microsoft_teams.apps" in sys.modules  # test stubs may lack ``__spec__``


TEAMS_SDK_AVAILABLE = _probe_teams_sdk_available()
# SDK symbols stay None until check_teams_requirements() binds them (via _SDK_IMPORTS below).
ClientOptions = App = ActivityContext = MessageActivity = ConversationReference = None  # type: ignore[assignment,misc]
TypingActivityInput = AdaptiveCardInvokeActivity = AdaptiveCardActionCardResponse = None  # type: ignore[assignment,misc]
AdaptiveCardActionMessageResponse = AdaptiveCardInvokeResponse = InvokeResponse = None  # type: ignore[assignment,misc]
HttpRequest = HttpResponse = HttpRouteHandler = AdaptiveCard = ExecuteAction = TextBlock = None  # type: ignore[assignment,misc]
HttpMethod = str  # type: ignore[assignment,misc]

from gateway.config import Platform, PlatformConfig
from gateway.platforms.helpers import MessageDeduplicator
from gateway.platforms.base import (
    gateway_trust_env, BasePlatformAdapter, ExecApprovalPrompt, SendResult, cache_image_from_url, cache_media_bytes_async,
)
from gateway.platforms.base_exec_approval import (
    EA_HEADER_TEXT, EA_REASON_LABEL_TEXT, approval_timeout_seconds, format_approval_deadline_line)
from gateway.platforms.event import MessageEvent, MessageType
from gateway.platforms._shared import (
    coerce_port, extra_or_secret as _extra_or_secret, get_scoped_secret as _get_scoped_secret,
    seed_extra_from_env as _seed_extra_from_env, send_error
)

logger = logging.getLogger(__name__)

_DEFAULT_PORT = 3978
_MAX_BODY_BYTES = 1_048_576  # Bot Framework activities are JSON well under 1 MiB
# ``None`` host → aiohttp binds IPv4 + IPv6 ("0.0.0.0" was unreachable on IPv6-only
# networks such as Fly.io 6PN). Pin via TEAMS_HOST or extra.host.
_DEFAULT_HOST = None
_WEBHOOK_PATH = "/api/messages"
# Regional/government tenants override via ``TEAMS_SERVICE_URL`` / ``extra['service_url']``.
_DEFAULT_TEAMS_SERVICE_URL = "https://smba.trafficmanager.net/teams/"
# Hosts that may receive a freshly minted bearer token (blocks SSRF / token exfiltration via a
# tampered env var). Exact match only: any Azure customer can register ``<name>.trafficmanager.net``.
_ALLOWED_TEAMS_SERVICE_HOSTS = frozenset({"smba.trafficmanager.net", "smba.infra.gov.teams.microsoft.us"})
# Conservative conversation-ID charset (``thread.skype`` / ``thread.tacv2`` suffixes included) so a
# hostile value cannot path-traverse out of ``/v3/conversations/<id>/activities``.
_TEAMS_CONV_ID_RE = re.compile(r"^[A-Za-z0-9:@\-_.]+$")
_BF_TOKEN_SCOPE = "https://api.botframework.com/.default"
# File-consent cards (personal chats) — Bot Framework content types, not SDK imports.
_CONTENT_TYPE_FILE_CONSENT = "application/vnd.microsoft.teams.card.file.consent"
_CONTENT_TYPE_FILE_INFO = "application/vnd.microsoft.teams.card.file.info"
_MAX_FILE_SEND_BYTES = 20 * 1024 * 1024
_PENDING_UPLOAD_MAX = 32
# Channel/group chats reject Bot Framework document attachments (400). Small text
# files are inlined; anything else uploads via Graph into the team's SharePoint
# folder (or FileConsent in a 1:1 DM).
_INLINE_CHANNEL_TEXT_MAX_BYTES = 48 * 1024
_INLINE_CHANNEL_TEXT_EXTS = frozenset({
    ".txt", ".md", ".markdown", ".rst", ".csv", ".tsv",
    ".json", ".jsonl", ".log",
    ".yaml", ".yml", ".toml", ".ini",
    ".py", ".rs", ".js", ".ts", ".tsx", ".jsx",
    ".go", ".rb", ".sh",
})
_CHANNEL_FILE_GRAPH_NOT_CONFIGURED = (
    "Can't attach this file in a Teams channel or group chat. FileConsent cards "
    "work only in a 1:1 chat with the bot. Binary channel/group files upload via "
    "Microsoft Graph to the team's SharePoint folder, but Graph is not configured. "
    "Set MSGRAPH_TENANT_ID, MSGRAPH_CLIENT_ID, and MSGRAPH_CLIENT_SECRET (or grant "
    "Files.ReadWrite.All to the Teams bot app and reuse TEAMS_*), then grant admin "
    "consent. Meanwhile send the file in a 1:1 DM."
)
_CHANNEL_FILE_GRAPH_NO_TARGET = (
    "Can't attach this file in a Teams channel or group chat: Hermes does not yet "
    "know this channel's team id (needed for SharePoint). Send a message in the "
    "channel first, or set TEAMS_TEAM_ID. FileConsent works in a 1:1 DM."
)
_CHANNEL_FILE_GRAPH_PERMISSIONS = (
    "Can't attach this file in a Teams channel or group chat: Microsoft Graph "
    "returned {status} ({detail}). The app needs admin-consented application "
    "permission Files.ReadWrite.All. FileConsent still works in a 1:1 DM."
)
_CHANNEL_FILE_GRAPH_FAILED = (
    "Can't attach this file in a Teams channel or group chat: {detail}. "
    "FileConsent cards work only in a 1:1 chat with the bot."
)
# OneDrive upload session hosts for file-consent PUT (exact suffix; blocks lookalikes).
_ONEDRIVE_UPLOAD_HOSTS = frozenset({
    "sharepoint.com", "onedrive.com", "1drv.com", "office.com", "office365.com",
})
_ONEDRIVE_UPLOAD_SUFFIXES = (
    ".sharepoint.com", ".sharepoint-df.com", ".onedrive.com", ".1drv.com",
    ".office.com", ".office365.com",
)
# Teams reaction IDs the Bot Framework connector accepts. Unicode / Slack-style
# aliases map here so send_message(action="react") and lifecycle 👀/✅/❌ work.
_REACTION_TYPE_BY_ALIAS = {
    "👍": "like", "+1": "like", "thumbsup": "like", "like": "like",
    "❤️": "heart", "❤": "heart", "♥️": "heart", "heart": "heart",
    "👀": "1f440_eyes", "eyes": "1f440_eyes", "1f440_eyes": "1f440_eyes",
    "✅": "2705_whiteheavycheckmark", "white_check_mark": "2705_whiteheavycheckmark",
    "2705_whiteheavycheckmark": "2705_whiteheavycheckmark",
    "🚀": "launch", "rocket": "launch", "launch": "launch",
    "📌": "1f4cc_pushpin", "pushpin": "1f4cc_pushpin", "1f4cc_pushpin": "1f4cc_pushpin",
    "😆": "laugh", "😂": "laugh", "laugh": "laugh",
    "😮": "surprised", "surprised": "surprised",
    "😢": "sad", "sad": "sad",
    "😠": "angry", "😡": "angry", "angry": "angry",
    "❌": "angry", "x": "angry",
}
_REACTION_EMOJI_BY_TYPE = {
    "like": "👍", "heart": "❤️", "1f440_eyes": "👀",
    "2705_whiteheavycheckmark": "✅", "launch": "🚀", "1f4cc_pushpin": "📌",
    "laugh": "😆", "surprised": "😮", "sad": "😢", "angry": "😠",
}
_REACTION_TYPE_RE = re.compile(r"^[A-Za-z0-9_\-]+$")
# Inline wait for HTTP 429 on activity updates; longer Retry-After fails the edit so
# the stream consumer falls back to a single non-streaming send (Telegram flood pattern).
_EDIT_FLOOD_INLINE_WAIT_CAP_SECS = 2.0
_EDIT_CACHE_MAX = 64


def _bf_token_request(tenant_id: str, client_id: str, client_secret: str) -> tuple[str, dict]:
    """(token URL, client-credentials form) for a Bot Framework bearer token."""
    return (
        f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token",
        {"grant_type": "client_credentials", "client_id": client_id, "client_secret": client_secret, "scope": _BF_TOKEN_SCOPE},
    )


def _is_allowed_https_host(url: str, *, check_port: bool = False) -> bool:
    """https + host in ``_ALLOWED_TEAMS_SERVICE_HOSTS`` (+ default port when asked)."""
    try:
        parsed = urlparse(url)
        if parsed.scheme != "https" or (check_port and parsed.port not in (None, 443)):
            return False
        return parsed.hostname in _ALLOWED_TEAMS_SERVICE_HOSTS
    except Exception:
        return False


def _is_botframework_attachment_url(url: str) -> bool:
    """True if ``url`` is a Bot Framework connector attachment host (may carry the bot token)."""
    return _is_allowed_https_host(url, check_port=True)


def _validate_teams_service_url(raw: str) -> Optional[str]:
    """Normalized (trailing-slash) service URL, or ``None`` if not on the allowlist."""
    if not raw or not _is_allowed_https_host(raw):
        return None
    return raw if raw.endswith("/") else raw + "/"


def _to_teams_reaction_type(emoji: Optional[str]) -> Optional[str]:
    """Map unicode / Slack-style alias / Teams reaction id → connector reaction type."""
    raw = (emoji or "").strip().strip(":")
    if not raw:
        return None
    mapped = _REACTION_TYPE_BY_ALIAS.get(raw) or _REACTION_TYPE_BY_ALIAS.get(raw.lower())
    if mapped:
        return mapped
    if _REACTION_TYPE_RE.match(raw):
        return raw
    return None


def _flat_conversation_id(chat_id: str) -> str:
    """Strip Teams ``;messageid=`` so Bot Framework REST uses a flat conversation id.

    Channel thread activities often arrive as ``19:…@thread.tacv2;messageid=123``.
    ``conversations.activities.update`` / ``delete`` reject that suffix.
    """
    raw = str(chat_id or "").strip()
    marker = ";messageid="
    idx = raw.lower().find(marker)
    return raw[:idx] if idx != -1 else raw


def _bf_activity_url(service_url: str, conversation_id: str, activity_id: str) -> str:
    """``/v3/conversations/{id}/activities/{id}`` on an allowlisted service URL."""
    return (
        f"{service_url}v3/conversations/{quote(conversation_id, safe=':@-_.')}"
        f"/activities/{quote(activity_id, safe=':@-_.')}"
    )


def _http_status_from_exc(exc: BaseException) -> Optional[int]:
    """Best-effort HTTP status on SDK / httpx errors."""
    for attr in ("status_code", "status"):
        val = getattr(exc, attr, None)
        if isinstance(val, int):
            return val
    resp = getattr(exc, "response", None)
    if resp is not None:
        val = getattr(resp, "status_code", None) or getattr(resp, "status", None)
        if isinstance(val, int):
            return val
    return None


def _retry_after_seconds(source: Any) -> Optional[float]:
    """Parse ``Retry-After`` from a response or exception, if present."""
    headers = getattr(source, "headers", None)
    if headers is None:
        resp = getattr(source, "response", None)
        headers = getattr(resp, "headers", None) if resp is not None else None
    if not headers:
        return None
    try:
        raw = headers.get("Retry-After") or headers.get("retry-after")
    except Exception:
        return None
    if raw is None or raw == "":
        return None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return None


def _activity_update_unsupported(exc: BaseException, status: Optional[int]) -> bool:
    """True when the connector cannot update this activity (fallback to a plain send)."""
    if status in (404, 405, 501):
        return True
    blob = str(exc).lower()
    return any(s in blob for s in (
        "method not allowed", "not supported", "cannot be updated",
        "activity not found", "message not found",
    ))


def _reaction_to_emoji(reaction_type: Optional[str]) -> str:
    """Teams reaction id → unicode (unknown ids pass through)."""
    raw = (reaction_type or "").strip()
    return _REACTION_EMOJI_BY_TYPE.get(raw, raw)


def _is_allowed_onedrive_upload_url(url: str) -> bool:
    """True if ``url`` is an https OneDrive/SharePoint upload session (file-consent PUT)."""
    try:
        parsed = urlparse(url)
        if parsed.scheme != "https" or parsed.port not in (None, 443):
            return False
        host = (parsed.hostname or "").lower()
        if host in _ONEDRIVE_UPLOAD_HOSTS:
            return True
        return any(host.endswith(suffix) for suffix in _ONEDRIVE_UPLOAD_SUFFIXES)
    except Exception:
        return False


def _is_mock_object(value: Any) -> bool:
    """True for unittest.mock stand-ins — never treat auto-attrs as real IDs/URLs."""
    return type(value).__module__.startswith("unittest.mock")


def _invoke_field(value: Any, *names: str) -> Any:
    """Read a field from an SDK model or a dict (file-consent / attachments use both).

    Accepts snake_case and camelCase names. Skips ``None``, blank strings, and
    ``MagicMock`` auto-attributes so a dict payload or a test double cannot
    shadow a real sibling field.
    """
    if isinstance(value, dict):
        for name in names:
            if name not in value:
                continue
            got = value[name]
            if _usable_invoke_value(got):
                return got
        return None
    if value is None:
        return None
    for name in names:
        got = getattr(value, name, None)
        if _usable_invoke_value(got):
            return got
    return None


def _usable_invoke_value(got: Any) -> bool:
    if got is None or _is_mock_object(got):
        return False
    if isinstance(got, str) and not got.strip():
        return False
    return True


def _field_text(value: Any, *names: str) -> str:
    got = _invoke_field(value, *names)
    return got.strip() if isinstance(got, str) else ""


def _activity_attachments(activity: Any) -> list:
    raw = _invoke_field(activity, "attachments")
    if isinstance(raw, list):
        return raw
    if isinstance(raw, tuple):
        return list(raw)
    return []


def _attachment_content_dict(content: Any) -> dict:
    """Normalize ``attachment.content`` (dict, SDK model, or mock) to a mapping."""
    if isinstance(content, dict):
        return content
    if content is None or _is_mock_object(content):
        return {}
    dump = getattr(content, "model_dump", None)
    if callable(dump):
        for kwargs in ({"by_alias": True}, {}):
            try:
                dumped = dump(**kwargs) if kwargs else dump()
            except TypeError:
                continue
            except Exception:
                dumped = None
            if isinstance(dumped, dict):
                return dumped
            break
    raw = getattr(content, "__dict__", None)
    return raw if isinstance(raw, dict) else {}


def _is_anonymous_body_mirror(content_type: str, content_url: str, att_name: str) -> bool:
    """Teams mirrors the message body as an unnamed text/html|text/plain with no URL.

    A *named* ``text/plain`` (for example ``notes.txt``) is a real file, even
    when the activity omitted ``contentUrl``.
    """
    if content_type.startswith("application/vnd.microsoft.card"):
        return True
    return content_type in ("text/html", "text/plain") and not content_url and not att_name


def _attachments_are_html_only(attachments: list) -> bool:
    """True when every Bot Framework attachment is a body/card mirror.

    Channel/group file drops often look like this: caption in ``activity.text`` plus
    a single unnamed ``text/html`` attachment — no ``file.download.info``.
    """
    if not attachments:
        return False
    for att in attachments:
        content_url = _field_text(att, "content_url", "contentUrl")
        content_type_raw = _invoke_field(att, "content_type", "contentType")
        content_type = (
            content_type_raw.lower().split(";")[0].strip()
            if isinstance(content_type_raw, str) else ""
        )
        att_name = _field_text(att, "name")
        if _is_anonymous_body_mirror(content_type, content_url, att_name):
            continue
        return False
    return True


def _normalize_consent_action(raw: Any) -> str:
    """Normalize FileConsent action to accept/decline.

    Prefer Enum.value; on Python 3.11 str(Action.ACCEPT) is 'Action.ACCEPT'.
    """
    if raw is None:
        return ""
    if hasattr(raw, "value"):
        raw = raw.value
    text = str(raw).strip().lower()
    if text.startswith("action."):
        text = text.split(".", 1)[-1]
    return text


def _consent_card_activity_id(activity: Any) -> Optional[str]:
    """FileConsent invoke ``replyToId`` / ``reply_to_id`` is the message that holds the card."""
    raw = _invoke_field(activity, "reply_to_id", "replyToId")
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if not text or not _TEAMS_CONV_ID_RE.match(text):
        return None
    return text


def _is_inlineable_channel_document(path: str, file_name: Optional[str] = None) -> bool:
    """True when a channel/group send can inline the file as a text message."""
    import mimetypes
    name = (file_name or os.path.basename(path) or "").lower()
    if os.path.splitext(name)[1] in _INLINE_CHANNEL_TEXT_EXTS:
        return True
    mime, _ = mimetypes.guess_type(name or path)
    return bool(mime and mime.split(";", 1)[0].strip().startswith("text/"))


def _read_inline_channel_text(path: str, *, max_bytes: int = _INLINE_CHANNEL_TEXT_MAX_BYTES) -> Optional[str]:
    """UTF-8 text at or under ``max_bytes``, else None (binary / too large / unreadable)."""
    try:
        if os.path.getsize(path) > max_bytes:
            return None
        with open(path, "rb") as fh:
            data = fh.read(max_bytes + 1)
    except OSError:
        return None
    if len(data) > max_bytes or b"\x00" in data:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _fence_channel_text(text: str, *, language: str = "") -> str:
    """Wrap ``text`` in a markdown fence that cannot collide with its contents."""
    fence = "```"
    while fence in text:
        fence += "`"
    info = language if language and all(ch.isalnum() or ch in "-_+" for ch in language) else ""
    return f"{fence}{info}\n{text}\n{fence}"


class _AiohttpBridgeAdapter:
    """HttpServerAdapter bridging SDK route registrations into our aiohttp app; without it
    ``App()`` unconditionally imports fastapi/uvicorn and allocates a ``FastAPI()``."""

    def __init__(self, aiohttp_app: "web.Application"):
        self._aiohttp_app = aiohttp_app

    def register_route(self, method: "HttpMethod", path: str, handler: "HttpRouteHandler") -> None:
        async def _aiohttp_handler(request: "web.Request") -> "web.Response":
            result: "HttpResponse" = await handler(HttpRequest(body=await request.json(), headers=dict(request.headers)))
            status = result.get("status", 200)
            resp_body = result.get("body")
            if resp_body is not None:
                return web.Response(status=status, body=json.dumps(resp_body), content_type="application/json")
            return web.Response(status=status)

        self._aiohttp_app.router.add_route(method, path, _aiohttp_handler)

    def serve_static(self, path: str, directory: str) -> None:
        pass

    async def start(self, port: int) -> None:
        raise NotImplementedError("aiohttp server is managed by the adapter")

    async def stop(self) -> None:
        pass


def check_requirements() -> bool:
    """PASSIVE probe (registry ``check_fn``): SDK + aiohttp importable? Never installs."""
    return TEAMS_SDK_AVAILABLE and AIOHTTP_AVAILABLE


def _credentials(config) -> tuple[str, str, str]:
    """(client_id, client_secret, tenant_id): ``config.extra`` first, then the profile-scoped env.

    client_id/tenant_id are read through the same scoped reader as the secret: under multiplex,
    ``os.environ`` holds the DEFAULT profile's app identity, and pairing it with a secondary's
    secret requests a Bot Framework token for the wrong app. ``extra`` wins so a per-profile
    config.yaml identity is never overridden by the process env.
    """
    extra = getattr(config, "extra", {}) or {}
    return (
        extra.get("client_id") or _get_scoped_secret("TEAMS_CLIENT_ID", ""),
        extra.get("client_secret") or _get_scoped_secret("TEAMS_CLIENT_SECRET", ""),
        extra.get("tenant_id") or _get_scoped_secret("TEAMS_TENANT_ID", ""))


def validate_config(config) -> bool:
    return bool(all(_credentials(config)))


def is_connected(config) -> bool:
    return validate_config(config)


def _env_enablement() -> dict | None:
    """``env_enablement_fn``: seed ``PlatformConfig.extra`` from the profile's env before adapter construction
    so ``gateway status`` reflects env-only setups without the SDK; ``None`` when not minimally configured.
    Every identity/endpoint is per-profile (the app the secret belongs to, its regional service URL, the
    cron home conversation), so a secondary is never seeded with the default profile's Teams app."""
    seed = _seed_extra_from_env((
        ("TEAMS_CLIENT_ID", "client_id", None), ("TEAMS_CLIENT_SECRET", "client_secret", None),
        ("TEAMS_TENANT_ID", "tenant_id", None), ("TEAMS_PORT", "port", int), ("TEAMS_SERVICE_URL", "service_url", None),
    ), home_env="TEAMS_HOME_CHANNEL")
    if not all(seed.get(k) for k in ("client_id", "client_secret", "tenant_id")):
        return None
    return seed



async def _standalone_send(
    pconfig, chat_id: str, message: str, *,
    thread_id: Optional[str] = None, media_files: Optional[list] = None, force_document: bool = False,
) -> Dict[str, Any]:
    """Acquire a Bot Framework bearer token and POST a single message activity; used by
    ``send_message_tool._send_via_adapter`` when the gateway runner is not in this process
    (``hermes cron``). ``TEAMS_SERVICE_URL`` is allowlisted and ``chat_id`` charset-checked
    (SSRF/path traversal). ``media_files`` / ``force_document`` are signature parity only — text-only."""
    extra = getattr(pconfig, "extra", {}) or {}
    client_id, client_secret, tenant_id = _credentials(pconfig)
    if not (client_id and client_secret and tenant_id):
        return send_error("Teams standalone send: TEAMS_CLIENT_ID, TEAMS_CLIENT_SECRET, and TEAMS_TENANT_ID are all required")
    raw_service_url = extra.get("service_url") or _get_scoped_secret("TEAMS_SERVICE_URL", "") or _DEFAULT_TEAMS_SERVICE_URL
    service_url = _validate_teams_service_url(raw_service_url)
    for failed, error in (
        (service_url is None, f"TEAMS_SERVICE_URL host is not on the Bot Framework allowlist; "
                              f"expected one of {sorted(_ALLOWED_TEAMS_SERVICE_HOSTS)}"),
        (not chat_id, "chat_id (conversation ID) is required"),
        (not _TEAMS_CONV_ID_RE.match(_flat_conversation_id(chat_id or "")),
         "chat_id contains characters outside the Bot Framework conversation ID set"),
        (not _TEAMS_CONV_ID_RE.match(tenant_id), "TEAMS_TENANT_ID contains characters outside the expected set"),
        (not AIOHTTP_AVAILABLE, "aiohttp not installed")):
        if failed:
            return send_error(f"Teams standalone send: {error}")
    token_url, token_form = _bf_token_request(tenant_id, client_id, client_secret)
    conv_id = _flat_conversation_id(chat_id or "")
    activities_url = f"{service_url}v3/conversations/{quote(conv_id, safe=':@-_.')}/activities"
    try:
        import aiohttp as _aiohttp
        # Per-request timeouts so a slow STS endpoint cannot starve the activity POST.
        per_request_timeout = _aiohttp.ClientTimeout(total=15.0)
        async with _aiohttp.ClientSession(trust_env=gateway_trust_env()) as session:
            async with session.post(
                token_url, data=token_form, headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=per_request_timeout,
            ) as token_resp:
                if token_resp.status >= 400:
                    body = await token_resp.text()
                    return send_error(f"Teams standalone send: token request failed ({token_resp.status}): {body[:300]}")
                token_payload = await token_resp.json()
            access_token = token_payload.get("access_token")
            if not access_token:
                return send_error("Teams standalone send: token response missing access_token")
            async with session.post(
                activities_url, json={"type": "message", "text": message, "textFormat": "markdown"},
                headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"},
                timeout=per_request_timeout,
            ) as send_resp:
                if send_resp.status >= 400:
                    body = await send_resp.text()
                    return send_error(f"Teams standalone send: activity post failed ({send_resp.status}): {body[:300]}")
                send_payload = await send_resp.json()
        return {"success": True, "message_id": send_payload.get("id")}
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.debug("Teams standalone send raised", exc_info=True)
        return send_error(f"Teams standalone send failed: {e}")


# SDK module → names rebound into this module's globals by check_teams_requirements().
_SDK_IMPORTS = {
    "microsoft_teams.apps": ("App", "ActivityContext"),
    "microsoft_teams.common.http.client": ("ClientOptions",),
    "microsoft_teams.api": ("MessageActivity", "ConversationReference"),
    "microsoft_teams.api.activities.typing": ("TypingActivityInput",),
    "microsoft_teams.api.activities.invoke.adaptive_card": ("AdaptiveCardInvokeActivity",),
    "microsoft_teams.api.models.adaptive_card": ("AdaptiveCardActionCardResponse", "AdaptiveCardActionMessageResponse"),
    "microsoft_teams.api.models.invoke_response": ("InvokeResponse", "AdaptiveCardInvokeResponse"),
    "microsoft_teams.apps.http.adapter": ("HttpMethod", "HttpRequest", "HttpResponse", "HttpRouteHandler"),
    "microsoft_teams.cards": ("AdaptiveCard", "ExecuteAction", "TextBlock")}


# NOTE: ``check_requirements`` is the
# PASSIVE probe (registry ``check_fn``, status / unit tests) — it must never trigger a pip install.
# ``check_teams_requirements`` is the ACTIVE lazy-installer, registered as ``ensure_deps_fn``: the
# registry's ``create_adapter()`` runs it when the passive probe fails, right before the gateway connects
# Teams (#79812). ``connect()`` re-checks defensively.
@contextmanager
def _suppress_third_party_dotenv() -> Iterator[None]:
    """No-op ``dotenv.load_dotenv`` while importing the Teams SDK: ``microsoft_teams.apps.app`` loads a
    cwd-discovered ``.env`` at import, mutating process-global ``os.environ``. Hermes owns dotenv loading.

    See #62935.
    """
    try:
        import dotenv as _dotenv
    except ImportError:
        _dotenv = None
    original = getattr(_dotenv, "load_dotenv", None)
    if original is None:
        yield
        return
    _dotenv.load_dotenv = lambda *args, **kwargs: False  # type: ignore[assignment]
    try:
        yield
    finally:
        _dotenv.load_dotenv = original  # type: ignore[assignment]


def check_teams_requirements() -> bool:
    """ACTIVE lazy-installer (registry ``ensure_deps_fn``): install the SDK on first use and rebind
    the module-level SDK globals. Gate on ``App is not None`` — ``TEAMS_SDK_AVAILABLE`` is only a
    find_spec probe and can be True before any import ran."""
    if App is not None and AIOHTTP_AVAILABLE:
        return True

    def _import() -> dict:
        from aiohttp import web as _web
        bindings: dict = {"web": _web, "AIOHTTP_AVAILABLE": True}
        with _suppress_third_party_dotenv():
            for module_name, names in _SDK_IMPORTS.items():
                module = importlib.import_module(module_name)
                for name in names:
                    try:
                        bindings[name] = getattr(module, name)
                    except AttributeError as exc:  # same failure class as ``from X import Y``
                        raise ImportError(f"cannot import name {name!r} from {module_name!r}") from exc
        bindings["TEAMS_SDK_AVAILABLE"] = True
        return bindings

    from tools.lazy_deps import ensure_and_bind
    return ensure_and_bind("platform.teams", _import, globals(), prompt=False)


_CHAT_TYPES = {"personal": "dm", "groupChat": "group", "channel": "channel"}
# DOCUMENT wins over PHOTO/VIDEO/AUDIO for mixed attachments: document-context
# injection gates strictly on MessageType.DOCUMENT (same precedence as Email/Signal).
_MEDIA_KIND_PRECEDENCE = (
    ("document", MessageType.DOCUMENT), ("image", MessageType.PHOTO),
    ("video", MessageType.VIDEO), ("audio", MessageType.AUDIO))
_APPROVAL_CHOICES = {"approve_once": "once", "approve_session": "session", "approve_always": "always", "deny": "deny"}
_APPROVAL_LABELS = {
    "once": "✅ Allowed (once)", "session": "✅ Allowed (session)", "always": "✅ Always allowed", "deny": "❌ Denied",
}


def _truncate(text: str, limit: int) -> str:
    return text[:limit] + "..." if len(text) > limit else text


def _approval_body(cmd: str, desc: str, *, always: bool = False) -> list:
    """Adaptive Card body blocks for an approval prompt; unless ``always``, empty ``cmd``/``desc`` omit their blocks."""
    body = []
    if cmd or always:
        body.append(TextBlock(text=f"⚠️ {EA_HEADER_TEXT}", wrap=True, weight="Bolder"))
        body.append(TextBlock(text=f"```\n{cmd}\n```", wrap=True))
    if desc or always:
        body.append(TextBlock(text=f"{EA_REASON_LABEL_TEXT}: {desc}", wrap=True, isSubtle=True))
    return body


class TeamsAdapter(BasePlatformAdapter):
    """Microsoft Teams adapter using the microsoft-teams-apps SDK."""
    # Answers /p/<profile>/... on the default listener for a served secondary (shared_ingress).
    serves_profile_prefix: bool = True

    MAX_MESSAGE_LENGTH = 28000  # Teams text message limit (~28 KB)
    splits_long_messages = True  # send() chunks via truncate_message()
    # Edit-based streaming (send then conversations.activities.update). Not Slack-style
    # native stream-is-the-message; do not flip this without a distinct Teams stream object.
    draft_stream_is_message = False
    # Processing-lifecycle reactions (👀 while working, ✅/❌ on complete). Unicode maps to
    # Teams reaction ids in _REACTION_TYPE_BY_ALIAS.
    _ACK_EMOJI = "👀"
    _OK_EMOJI = "✅"
    _FAIL_EMOJI = "❌"

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform("teams"))
        # Kept on the instance: ``platforms.teams.extra.*`` keys are read after construction too.
        self._extra: Dict[str, Any] = config.extra or {}
        self._client_id, self._client_secret, self._tenant_id = _credentials(config)
        # (token, expiry monotonic ts) for connector attachment auth; refreshed under
        # _bf_token_lock so concurrent attachments can't stampede the STS.
        self._bf_token_cache: Optional[tuple] = None
        self._bf_token_lock: Optional[asyncio.Lock] = None
        self._port = coerce_port(self._extra.get("port") or _get_scoped_secret("TEAMS_PORT", str(_DEFAULT_PORT)), _DEFAULT_PORT)
        _raw_host = self._extra.get("host") or _get_scoped_secret("TEAMS_HOST", "") or _DEFAULT_HOST  # falsy → dual-stack None
        self._host: Optional[str] = str(_raw_host) if _raw_host else None
        self._app: Optional["App"] = None
        self._runner: Optional["web.AppRunner"] = None
        self._dedup = MessageDeduplicator(max_size=1000)
        # chat_id → ConversationReference so proactive cards use the right conversation type.
        self._conv_refs: Dict[str, Any] = {}
        # chat_id → GraphFileTarget (team aadGroupId / channel id / group chat id) from inbound
        # channelData. Tests inject ``_graph_client`` (client or False to disable).
        self._graph_file_targets: Dict[str, Any] = {}
        self._graph_client: Any = None
        self._require_mention: bool = self._parse_require_mention(config)
        self._observe_unmentioned: bool = self._parse_observe_unmentioned(config)
        # Outbound activity ids (bounded) so require_mention can exempt replies to our own messages.
        self._sent_ids: deque = deque(maxlen=500)
        # chat_id → last inbound activity id (send_message react default target).
        self._last_inbound_by_chat: Dict[str, str] = {}
        # (chat_id, message_id) → last reaction type this bot set (unreact without emoji).
        self._bot_reactions: Dict[tuple, str] = {}
        # file-consent acceptContext id → {name, bytes, mime} (bounded).
        self._pending_uploads: Dict[str, Dict[str, Any]] = {}
        self._pending_upload_ids: deque = deque()
        # (chat_id, activity_id) → last edited text / saturated mid-stream preview.
        self._last_edit_text: Dict[tuple, str] = {}
        self._last_overflow_preview: Dict[tuple, str] = {}

    @staticmethod
    def _parse_require_mention(config) -> bool:
        """TEAMS_REQUIRE_MENTION (scoped) → ``require_mention`` in config.extra → false (opt-in, same
        default as TELEGRAM_REQUIRE_MENTION). Without RSC Teams only delivers mention activities to a
        group bot anyway, so the gate changes nothing until the app gains ChannelMessage.Read.Group /
        ChatMessage.Read.Chat and starts receiving every conversation message."""
        configured = _extra_or_secret(config.extra, "require_mention", "TEAMS_REQUIRE_MENTION", False)
        if isinstance(configured, bool):
            return configured
        return str(configured).strip().lower() not in {"false", "0", "no", "off"}

    @staticmethod
    def _parse_observe_unmentioned(config) -> bool:
        """TEAMS_OBSERVE_UNMENTIONED → ``observe_unmentioned`` in extra → true.

        Only runs when ``require_mention`` would drop a channel/group message (RSC
        delivers every post). Default on: without RSC those messages never arrive,
        so the flag is a no-op until the app has ChannelMessage.Read.Group /
        ChatMessage.Read.Chat. Set false to keep the old silent-drop behavior.
        """
        configured = _extra_or_secret(config.extra, "observe_unmentioned", "TEAMS_OBSERVE_UNMENTIONED", True)
        if isinstance(configured, bool):
            return configured
        return str(configured).strip().lower() not in {"false", "0", "no", "off"}

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        # Reconnect paths reach here without create_adapter()'s installer — re-run to bind SDK globals.
        check_teams_requirements()
        pip = f"{sys.executable} -m pip install"
        for failed, code, message in (
            (App is None or ClientOptions is None, "MISSING_SDK",
             f"microsoft-teams-apps could not be installed. Run: {pip} microsoft-teams-apps"),
            (not AIOHTTP_AVAILABLE, "MISSING_SDK", f"aiohttp not installed. Run: {pip} aiohttp"),
            (not self._client_id or not self._client_secret or not self._tenant_id, "MISSING_CREDENTIALS",
             "TEAMS_CLIENT_ID, TEAMS_CLIENT_SECRET, and TEAMS_TENANT_ID are all required")):
            if failed:
                self._set_fatal_error(code, message, retryable=False)
                return False
        try:
            # aiohttp app first — the bridge adapter wires SDK routes into it.
            # Set up aiohttp app first — the bridge adapter wires SDK routes into it. client_max_size: Bot
            # Framework activities are JSON (caps out well under 1 MiB); an explicit cap keeps
            # oversized/chunked bodies from being buffered unbounded on a 0.0.0.0 bind (same pattern as
            # webhook.py / raft, #58536/#58902).
            aiohttp_app = web.Application(client_max_size=_MAX_BODY_BYTES)
            aiohttp_app.router.add_get("/health", lambda _: web.Response(text="ok"))
            self._app = App(
                client_id=self._client_id, client_secret=self._client_secret, tenant_id=self._tenant_id,
                http_server_adapter=_AiohttpBridgeAdapter(aiohttp_app),
                client=ClientOptions(headers={"User-Agent": "Hermes"}))
            # Handlers (ours, then plugin on_* decorators) must be wired before initialize(),
            # which registers POST /api/messages on aiohttp_app via the bridge's register_route().
            @self._app.on_message
            async def _handle_message(ctx: ActivityContext[MessageActivity]):
                await self._on_message(ctx)

            @self._app.on_card_action
            async def _handle_card_action(
                ctx: ActivityContext[AdaptiveCardInvokeActivity],
            ) -> InvokeResponse[AdaptiveCardActionMessageResponse]:
                return await self._on_card_action(ctx)

            on_reaction = getattr(self._app, "on_message_reaction", None)
            if callable(on_reaction):
                @on_reaction
                async def _handle_reaction(ctx):
                    await self._on_message_reaction(ctx)

            on_file_consent = getattr(self._app, "on_file_consent", None)
            if callable(on_file_consent):
                @on_file_consent
                async def _handle_file_consent(ctx):
                    await self._on_file_consent(ctx)

            self._wire_plugin_handlers(self._app)
            await self._app.initialize()
            # Shared-listener mode (multiplex secondary): no bind; served at /p/<profile>/api/messages.
            from gateway.platforms.shared_ingress import bind_listener
            self._runner = await bind_listener(self, aiohttp_app, self._host, self._port, _WEBHOOK_PATH)
            self._running = True
            self._mark_connected()
            if self._runner is not None:
                logger.info(
                    "[teams] Webhook server listening on %s:%d%s",
                    self._host or "* (all interfaces, IPv4+IPv6)", self._port, _WEBHOOK_PATH)
            return True
        except Exception as e:
            self._set_fatal_error("CONNECT_FAILED", f"Teams connection failed: {e}", retryable=True)
            logger.error("[teams] Failed to connect: %s", e, exc_info=True)
            return False

    async def disconnect(self) -> None:
        self._running = False
        if self._runner:
            await self._runner.cleanup()
        self._runner = self._app = None
        self._mark_disconnected()
        logger.info("[teams] Disconnected")

    async def _get_botframework_token(self) -> str:
        """Bot Framework bearer token (client credentials), cached until ~5 min before expiry; connector
        attachments are NOT pre-authenticated, unlike SharePoint downloadUrls. The lock is created lazily
        because ``asyncio.Lock()`` in __init__ may bind the wrong loop."""
        import time
        import httpx
        if self._bf_token_lock is None:
            self._bf_token_lock = asyncio.Lock()
        async with self._bf_token_lock:
            cached = self._bf_token_cache
            if cached and cached[1] > time.monotonic() + 300:
                return cached[0]
            if not (self._client_id and self._client_secret and self._tenant_id):
                raise ValueError("Missing TEAMS_CLIENT_ID/SECRET/TENANT_ID for attachment auth")
            token_url, token_form = _bf_token_request(self._tenant_id, self._client_id, self._client_secret)
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(token_url, data=token_form)
                resp.raise_for_status()
                payload = resp.json()
            expires_in = float(payload.get("expires_in", 3600) or 3600)
            self._bf_token_cache = (payload["access_token"], time.monotonic() + expires_in)
            return self._bf_token_cache[0]

    async def _fetch_attachment_bytes(self, url: str, timeout: float = 30.0) -> bytes:
        """Download attachment bytes with SSRF protection. Connector URLs get the bot's bearer token;
        redirects and body size go through the shared guards (as the cache_*_from_url helpers)."""
        from tools.url_safety import create_ssrf_safe_async_client, is_safe_url
        from gateway.platforms.base import _ssrf_redirect_guard, _read_httpx_body_with_limit
        if not is_safe_url(url):
            raise ValueError("Blocked unsafe attachment URL (SSRF protection)")
        headers = {"User-Agent": "Mozilla/5.0 (compatible; HermesAgent/1.0)"}
        if _is_botframework_attachment_url(url):
            try:
                headers["Authorization"] = f"Bearer {await self._get_botframework_token()}"
            except Exception as e:
                logger.warning("[teams] Could not acquire Bot Framework token for attachment: %s", e)
        async with create_ssrf_safe_async_client(
            timeout=timeout, follow_redirects=True, event_hooks={"response": [_ssrf_redirect_guard]}) as client:
            async with client.stream("GET", url, headers=headers) as response:
                response.raise_for_status()
                # Never buffer .content — a lying Content-Length must not OOM the gateway.
                return await _read_httpx_body_with_limit(response, media_type="attachment")

    async def _on_message(self, ctx: ActivityContext[MessageActivity]) -> None:
        activity = ctx.activity
        # Teams writes the bot's conversation identity as ``28:<app id>`` (activity.recipient) while
        # App.id is the bare app id — accept both when deciding "is this us".
        recipient_id = getattr(getattr(activity, "recipient", None), "id", None)
        bot_ids = {i for i in (self._app.id if self._app else None, recipient_id) if isinstance(i, str) and i}
        bot_ids |= {f"28:{i}" for i in tuple(bot_ids) if not i.startswith("28:")}
        if getattr(activity.from_, "id", None) in bot_ids:
            return
        msg_id = getattr(activity, "id", None)
        if msg_id and self._dedup.is_duplicate(msg_id):
            return
        conv = activity.conversation
        conv_id = getattr(conv, "id", None)
        if conv_id:  # cache the conversation reference for proactive sends (approval cards, etc.)
            self._remember_conv_ref(str(conv_id), ctx.conversation_ref)
            self._remember_graph_file_target(str(conv_id), activity)
        text = activity.text if hasattr(activity, "text") and activity.text else ""
        conv_type = getattr(conv, "conversation_type", None)
        addressed = True
        if self._require_mention and conv_type != "personal":
            # RSC-delivered history: every channel/groupChat message arrives. Keep the ones that
            # @mention the bot or reply to one of its own messages; observe or drop the rest
            # BEFORE the attachment loop so a gated post never downloads anything onto the host.
            addressed = (
                self._activity_mentions_bot(activity, bot_ids, text)
                or getattr(activity, "reply_to_id", None) in self._sent_ids
            )
            if not addressed:
                self._observe_unmentioned_activity(activity, text)
                logger.debug(
                    "[teams] %s non-personal message without a bot mention (chat=%s, msg=%s)",
                    "Observed" if self._observe_unmentioned else "Dropping", conv_id, msg_id)
                return
        if conv_id and msg_id:
            self._last_inbound_by_chat[str(conv_id)] = str(msg_id)
        if "<at>" in text:  # strip the <at>BotName</at> tags Teams prepends for @mentions
            text = re.sub(r"<at>[^<]*</at>\s*", "", text).strip()
        from_account = activity.from_
        user_id = getattr(from_account, "aad_object_id", None) or getattr(from_account, "id", "")
        source = self.build_source(
            chat_id=conv.id,
            chat_name=getattr(conv, "name", None) or "",
            chat_type=_CHAT_TYPES.get(conv_type or "", "dm"),
            user_id=str(user_id),
            user_name=getattr(from_account, "name", None) or "",
            guild_id=getattr(conv, "tenant_id", None) or self._tenant_id,
            message_id=msg_id)
        graph_target = self._graph_file_target_for(str(conv_id)) if conv_id else None
        raw_atts = _activity_attachments(activity)
        media: list = [
            m for m in [
                await self._cache_attachment(a, graph_target=graph_target)
                for a in raw_atts
            ] if m
        ]
        if (
            not media
            and graph_target is not None
            and msg_id
            and _attachments_are_html_only(raw_atts)
        ):
            media = await self._cache_files_from_graph_message(graph_target, str(msg_id))
        media_kinds = [kind for _, _, kind in media]  # media items are (path, media_type, kind)
        msg_type = next((t for kind, t in _MEDIA_KIND_PRECEDENCE if kind in media_kinds), MessageType.TEXT)
        event = MessageEvent(
            text=text, source=source, message_type=msg_type, message_id=msg_id,
            media_urls=[path for path, _, _ in media], media_types=[mt for _, mt, _ in media])
        if addressed and self._require_mention and conv_type != "personal" and self._observe_unmentioned:
            event = self._apply_teams_observe_attribution(event)
        await self.handle_message(event)

    @staticmethod
    def _activity_mentions_bot(activity: Any, bot_ids: set, text: str) -> bool:
        """True when a ``mention`` entity points at the bot (``mentioned.id`` is ``28:<app id>`` on the
        wire; ``bot_ids`` carries both spellings). A payload with no mention entities at all falls back
        to the rendered ``<at>`` tag; one that mentions only other people does not."""
        mentions = [e for e in getattr(activity, "entities", None) or [] if getattr(e, "type", None) == "mention"]
        if not mentions:
            return "<at>" in text
        return any(str(getattr(getattr(e, "mentioned", None), "id", "")) in bot_ids for e in mentions)

    _TEAMS_OBSERVED_CONTEXT_PROMPT = (
        "You are handling a Microsoft Teams channel or group-chat message.\n"
        "- observed Teams channel context may be provided in a separate context-only block "
        "before the current message; it is not necessarily addressed to you.\n"
        "- Treat only the current new message as a request explicitly directed at you, "
        "and use observed context only when the current message asks for it."
    )

    def _observe_unmentioned_activity(self, activity: Any, text: str) -> None:
        """Append gated channel/group chatter to the shared session; do not dispatch."""
        if not self._observe_unmentioned:
            return
        store = getattr(self, "_session_store", None)
        if not store:
            return
        from_account = getattr(activity, "from_", None)
        user_name = getattr(from_account, "name", None) or ""
        user_id = getattr(from_account, "aad_object_id", None) or getattr(from_account, "id", "") or "unknown"
        body = re.sub(r"<at>[^<]*</at>\s*", "", text).strip() if "<at>" in (text or "") else (text or "")
        attributed = f"[{user_name or user_id}] {body}".strip()
        conv = getattr(activity, "conversation", None)
        msg_id = getattr(activity, "id", None)
        source = self.build_source(
            chat_id=getattr(conv, "id", "") or "",
            chat_name=getattr(conv, "name", None) or "",
            chat_type=_CHAT_TYPES.get(getattr(conv, "conversation_type", None) or "", "group"),
            user_id=None,
            user_name=None,
            guild_id=getattr(conv, "tenant_id", None) or self._tenant_id,
            message_id=msg_id)
        try:
            session_entry = store.get_or_create_session(source)
            entry = {
                "role": "user",
                "content": attributed,
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                "observed": True,
            }
            if msg_id:
                entry["message_id"] = str(msg_id)
            store.append_to_transcript(session_entry.session_id, entry)
            logger.info(
                "[teams] Channel message observed (no bot trigger): chat=%s from=%s",
                getattr(conv, "id", "unknown"), user_id)
        except Exception as exc:
            logger.warning("[teams] Failed to observe unmentioned message: %s", exc)

    def _apply_teams_observe_attribution(self, event: MessageEvent) -> MessageEvent:
        """Shared session + channel_prompt marker so run.py wraps observed rows."""
        observe_prompt = self._TEAMS_OBSERVED_CONTEXT_PROMPT
        channel_prompt = (
            f"{event.channel_prompt}\n\n{observe_prompt}" if event.channel_prompt else observe_prompt
        )
        if (event.text or "").startswith("/"):
            return dataclasses.replace(event, channel_prompt=channel_prompt)
        user_name = event.source.user_name or event.source.user_id or "unknown"
        attributed = f"[{user_name}] {event.text or ''}".strip()
        source = dataclasses.replace(event.source, user_id=None, user_name=None)
        return dataclasses.replace(event, text=attributed, source=source, channel_prompt=channel_prompt)

    async def _cache_attachment(self, att: Any, *, graph_target: Any = None) -> Optional[tuple]:
        """Download + cache one inbound attachment → ``(path, media_type, kind)`` or ``None``."""
        content_url = _field_text(att, "content_url", "contentUrl")
        content_type_raw = _invoke_field(att, "content_type", "contentType")
        content_type = (
            content_type_raw.lower().split(";")[0].strip()
            if isinstance(content_type_raw, str) else ""
        )
        att_name = _field_text(att, "name")
        content = _invoke_field(att, "content")
        content_map = _attachment_content_dict(content)
        if content_map:
            content_keys: list[str] = sorted(str(k) for k in content_map.keys())
        elif isinstance(content, str):
            content_keys = ["<inline>"]
        else:
            content_keys = []
        logger.info(
            "[teams] Inbound attachment name=%r contentType=%s hasUrl=%s contentKeys=%s",
            att_name, content_type or "-", bool(content_url), content_keys,
        )
        if _is_anonymous_body_mirror(content_type, content_url, att_name):
            return None
        is_file_info = content_type == "application/vnd.microsoft.teams.file.download.info"
        download_url = _field_text(content_map, "downloadUrl", "download_url")
        file_type = _field_text(content_map, "fileType", "file_type").lstrip(".")
        filename = att_name or (f"document.{file_type}" if file_type else "document")
        if is_file_info or (not content_url and content_map):
            if is_file_info and not download_url:
                logger.warning(
                    "[teams] file.download.info attachment %r has no downloadUrl "
                    "(contentKeys=%s)",
                    filename, content_keys,
                )
                download_url = await self._resolve_inbound_download_url_via_graph(
                    content_map, content_url=content_url, filename=filename,
                    graph_target=graph_target,
                )
            if download_url:
                try:
                    data = await self._fetch_attachment_bytes(download_url)
                    cached = await cache_media_bytes_async(data, filename=filename, mime_type="")
                    if not cached:
                        logger.warning(
                            "[teams] Unsupported document type for attachment '%s', skipping",
                            filename)
                        return None
                    return cached.path, cached.media_type, cached.kind
                except Exception as e:
                    logger.warning("[teams] Failed to cache file attachment '%s': %s", filename, e)
                    return None
            if is_file_info:
                return None
        if not content_url and att_name and isinstance(content, str) and content:
            try:
                cached = await cache_media_bytes_async(
                    content.encode("utf-8"), filename=att_name, mime_type=content_type)
                return (cached.path, cached.media_type, cached.kind) if cached else None
            except Exception as e:
                logger.warning(
                    "[teams] Failed to cache inline attachment '%s' (%s): %s",
                    att_name, content_type, e)
                return None
        if not content_url and att_name:
            download_url = await self._resolve_inbound_download_url_via_graph(
                content_map, content_url=content_url, filename=filename,
                graph_target=graph_target,
            )
            if download_url:
                try:
                    data = await self._fetch_attachment_bytes(download_url)
                    cached = await cache_media_bytes_async(
                        data, filename=filename, mime_type=content_type)
                    return (cached.path, cached.media_type, cached.kind) if cached else None
                except Exception as e:
                    logger.warning("[teams] Failed to cache attachment '%s' (%s): %s", filename, content_type, e)
                    return None
            logger.warning(
                "[teams] Named attachment %r (%s) has no URL and Graph fallback did not resolve one",
                filename, content_type or "-",
            )
            return None
        if content_url and content_type.startswith("image/"):
            try:
                if _is_botframework_attachment_url(content_url):
                    # Connector URL needs the bot's bearer token; the generic cache helper sends none.
                    data = await self._fetch_attachment_bytes(content_url)
                    ext = content_type.split("/")[-1].split(";")[0] or "png"
                    cached = await cache_media_bytes_async(data, filename=att_name or f"image.{ext}", mime_type=content_type)
                    if not cached:
                        logger.warning(
                            "[teams] Bot Framework attachment '%s' returned data that failed image validation, skipping",
                            att_name or content_url)
                        return None
                    return cached.path, cached.media_type, "image"
                path = await cache_image_from_url(content_url)
                return (path, content_type, "image") if path else None
            except Exception as e:
                logger.warning("[teams] Failed to cache image attachment: %s", e)
            return None
        if content_url:  # direct-URL non-image attachment (video/audio/document)
            try:
                data = await self._fetch_attachment_bytes(content_url)
                cached = await cache_media_bytes_async(data, filename=att_name, mime_type=content_type)
                return (cached.path, cached.media_type, cached.kind) if cached else None
            except Exception as e:
                logger.warning("[teams] Failed to cache attachment '%s' (%s): %s", att_name or content_url, content_type, e)
        return None

    async def _resolve_inbound_download_url_via_graph(
        self,
        content_map: dict,
        *,
        content_url: str,
        filename: str,
        graph_target: Any,
    ) -> str:
        """When channel activities omit downloadUrl, resolve one via Graph filesFolder/shares."""
        from plugins.platforms.teams.graph_files import resolve_inbound_file_download_url
        client = self._graph_client_for_files()
        if client is None:
            logger.warning(
                "[teams] Inbound file %r has no downloadUrl and Graph is not configured; "
                "the agent will not see this attachment. Grant Files.ReadWrite.All "
                "(same MSGRAPH_*/TEAMS_* app as outbound channel uploads).",
                filename,
            )
            return ""
        try:
            url = await resolve_inbound_file_download_url(
                client,
                target=graph_target,
                content=content_map,
                content_url=content_url,
                filename=filename,
            )
        except Exception as e:
            logger.warning("[teams] Graph inbound file lookup failed for %r: %s", filename, e)
            return ""
        if not url:
            logger.warning(
                "[teams] Graph inbound file lookup for %r returned no downloadUrl "
                "(uniqueId/site info missing or filesFolder lookup failed)",
                filename,
            )
        return url or ""

    async def _cache_files_from_graph_message(self, graph_target: Any, message_id: str) -> list:
        """Channel/group file drop: Bot Framework sent only text/html — GET the Graph message."""
        from plugins.platforms.teams.graph_files import (
            GRAPH_CHANNEL_MESSAGE_PERMISSION, GRAPH_CHANNEL_MESSAGE_RSC,
            GRAPH_CHAT_MESSAGE_PERMISSION, GRAPH_CHAT_MESSAGE_RSC,
            list_graph_message_file_refs,
        )
        from tools.microsoft_graph_client import MicrosoftGraphAPIError

        client = self._graph_client_for_files()
        if client is None:
            logger.warning(
                "[teams] Channel/group activity had only text/html attachments; Graph is not "
                "configured so the agent cannot read the SharePoint file. Grant %s plus %s "
                "(RSC) or %s (same MSGRAPH_*/TEAMS_* app as outbound uploads).",
                "Files.ReadWrite.All", GRAPH_CHANNEL_MESSAGE_RSC, GRAPH_CHANNEL_MESSAGE_PERMISSION,
            )
            return []
        try:
            refs = await list_graph_message_file_refs(client, graph_target, message_id)
        except MicrosoftGraphAPIError as e:
            needed = (
                f"{GRAPH_CHANNEL_MESSAGE_RSC} (RSC) or {GRAPH_CHANNEL_MESSAGE_PERMISSION}"
                if getattr(graph_target, "team_id", "")
                else f"{GRAPH_CHAT_MESSAGE_RSC} (RSC) or {GRAPH_CHAT_MESSAGE_PERMISSION}"
            )
            logger.warning(
                "[teams] Graph GET message %s failed (%s). Inbound channel files need %s "
                "and Files.ReadWrite.All to download. %s",
                message_id, getattr(e, "status_code", "?"), needed, e,
            )
            return []
        except Exception as e:
            logger.warning("[teams] Graph GET message %s failed: %s", message_id, e)
            return []
        if not refs:
            logger.info("[teams] Graph message %s has no file attachments", message_id)
            return []
        media: list = []
        for ref in refs:
            filename = ref.get("name") or "document"
            download_url = await self._resolve_inbound_download_url_via_graph(
                {"uniqueId": ref.get("uniqueId") or ""},
                content_url=ref.get("contentUrl") or "",
                filename=filename,
                graph_target=graph_target,
            )
            if not download_url:
                continue
            try:
                data = await self._fetch_attachment_bytes(download_url)
                cached = await cache_media_bytes_async(data, filename=filename, mime_type="")
                if cached:
                    media.append((cached.path, cached.media_type, cached.kind))
            except Exception as e:
                logger.warning("[teams] Failed to cache Graph message file '%s': %s", filename, e)
        return media

    async def _send_card(self, chat_id: str, card: "AdaptiveCard") -> "Any":
        """Send an AdaptiveCard, using a stored ConversationReference when available."""
        from microsoft_teams.api import MessageActivityInput
        if not self._app:
            return None
        return await self._send_via_conv_ref(chat_id, MessageActivityInput().add_card(card), card)

    async def _send_via_conv_ref(self, chat_id: str, activity: Any, fallback: Any) -> Any:
        """Send ``activity`` through the cached ConversationReference, else ``App.send(fallback)``."""
        conv_ref = self._conv_ref_for(chat_id)
        if conv_ref:
            result = await self._app.activity_sender.send(activity, conv_ref)
        else:
            result = await self._app.send(chat_id, fallback)
        self._remember_sent(result)
        return result

    def _remember_conv_ref(self, chat_id: str, conv_ref: Any) -> None:
        """Cache a conversation reference under the wire id and the flat Bot Framework id."""
        if not chat_id:
            return
        self._conv_refs[chat_id] = conv_ref
        flat = _flat_conversation_id(chat_id)
        if flat != chat_id:
            self._conv_refs[flat] = conv_ref

    def _conv_ref_for(self, chat_id: str) -> Any:
        return self._conv_refs.get(chat_id) or self._conv_refs.get(_flat_conversation_id(chat_id))

    def _remember_sent(self, result: Any) -> None:
        """Track an outbound activity id (bounded deque) for the require_mention reply exemption."""
        sent_id = getattr(result, "id", None)
        if isinstance(sent_id, str) and sent_id:
            self._sent_ids.append(sent_id)

    @staticmethod
    def _invoke_message(text: str) -> "InvokeResponse[AdaptiveCardActionMessageResponse]":
        return InvokeResponse(status=200, body=AdaptiveCardActionMessageResponse(value=text))

    @staticmethod
    def _invoke_card(body: list) -> "InvokeResponse[AdaptiveCardActionMessageResponse]":
        card = AdaptiveCard().with_version("1.4").with_body(body)
        return InvokeResponse(status=200, body=AdaptiveCardActionCardResponse(value=card))

    async def _on_card_action(
        self, ctx: "ActivityContext[AdaptiveCardInvokeActivity]"
    ) -> "InvokeResponse[AdaptiveCardActionMessageResponse]":
        from tools.approval import resolve_gateway_approval, has_blocking_approval

        data = ctx.activity.value.action.data or {}
        hermes_action = data.get("hermes_action", "")
        session_key = data.get("session_key", "")
        if not hermes_action or not session_key:
            return self._invoke_message("Unknown action.")
        denied = self._card_action_denied(ctx.activity.from_)
        if denied:
            return self._invoke_message(denied)
        choice = _APPROVAL_CHOICES.get(hermes_action)
        if not choice:
            return self._invoke_message("Unknown action.")
        if not has_blocking_approval(session_key):
            return self._invoke_card([TextBlock(text="⚠️ Approval already resolved or expired.", wrap=True)])
        resolve_gateway_approval(session_key, choice)
        body = _approval_body(data.get("cmd", ""), data.get("desc", ""))
        body.append(TextBlock(text=_APPROVAL_LABELS[choice], wrap=True, weight="Bolder"))
        return self._invoke_card(body)

    @staticmethod
    def _card_action_denied(from_account: Any) -> Optional[str]:
        """Default-deny gate for approval clicks: require TEAMS_ALLOWED_USERS or an explicit
        TEAMS_ALLOW_ALL_USERS=true opt-in, else anyone who can message the bot could approve.
        Returns the user-facing denial text, or ``None`` when allowed."""
        # Scoped reads: under multiplex os.environ is the DEFAULT profile's allow-all/allowlist.
        if _get_scoped_secret("TEAMS_ALLOW_ALL_USERS", "").strip().lower() in {"1", "true", "yes"}:
            return None
        allowed_csv = _get_scoped_secret("TEAMS_ALLOWED_USERS", "").strip()
        if not allowed_csv:
            logger.warning(
                "[teams] card action rejected: TEAMS_ALLOWED_USERS not configured "
                "and TEAMS_ALLOW_ALL_USERS not set — default deny")
            return "⛔ Approval buttons require TEAMS_ALLOWED_USERS to be configured."
        clicker_id = getattr(from_account, "aad_object_id", None) or getattr(from_account, "id", "")
        allowed_ids = {uid.strip() for uid in allowed_csv.split(",") if uid.strip()}
        if "*" not in allowed_ids and clicker_id not in allowed_ids:
            logger.warning("[teams] Unauthorized card action by %s — ignoring", clicker_id)
            return "⛔ Not authorized."
        return None

    _EA_CMD_BUDGET = 2000
    _EA_CARD_ACTIONS = {"once": "approve_once", "session": "approve_session", "always": "approve_always", "deny": "deny"}
    _EA_CARD_STYLES = {"primary": "positive", "danger": "destructive"}

    async def _send_exec_approval_prompt(self, prompt: ExecApprovalPrompt) -> SendResult:
        """Adaptive Card: the shared text is split into its header / fenced command / reason blocks."""
        if not self._app:
            return SendResult(success=False, error="Teams app not initialized")
        # Button data carries a truncated cmd — just enough to reconstruct the card body.
        btn_data_base = {"session_key": prompt.session_key, "cmd": _truncate(prompt.command, 200), "desc": prompt.description}
        actions = []
        for label, choice, style in prompt.actions:
            kw = {"style": self._EA_CARD_STYLES[style]} if style else {}
            actions.append(ExecuteAction(
                title=label, verb="hermes_approve",
                data={**btn_data_base, "hermes_action": self._EA_CARD_ACTIONS[choice]}, **kw))
        body = _approval_body(self._truncate_preview(prompt.command, self._EA_CMD_BUDGET), prompt.description, always=True)
        body.append(TextBlock(text=format_approval_deadline_line(approval_timeout_seconds()), wrap=True))
        if prompt.smart_denied:
            body.append(TextBlock(text=self._EA_SMART_DENY_LINE.strip(), wrap=True))
        card = AdaptiveCard().with_version("1.4").with_body(body).with_actions(actions)
        try:
            result = await self._send_card(prompt.chat_id, card)
            return SendResult(success=True, message_id=getattr(result, "id", None) if result else None)
        except Exception as e:
            logger.error("[teams] send_exec_approval failed: %s", e, exc_info=True)
            return SendResult(success=False, error=str(e), retryable=True)

    async def send(
        self, chat_id: str, content: str, reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None
    ) -> SendResult:
        if not self._app:
            return SendResult(success=False, error="Teams app not initialized")
        last_message_id = None
        for chunk in self.truncate_message(self.format_message(content)):
            try:
                if reply_to and reply_to.isdigit() and reply_to != "0":
                    try:
                        result = await self._app.reply(chat_id, reply_to, chunk)
                    except Exception as reply_err:
                        # Group chats 400 on threaded sends; the SDK has no typed HTTP errors → fall back on any.
                        logger.debug("Teams reply() failed, falling back to flat send: %s", reply_err)
                        result = await self._app.send(chat_id, chunk)
                else:
                    result = await self._app.send(chat_id, chunk)
                last_message_id = getattr(result, "id", None)
                self._remember_sent(result)
            except Exception as e:
                return SendResult(success=False, error=str(e), retryable=True)
        return SendResult(success=True, message_id=last_message_id)

    async def edit_message(
        self, chat_id: str, message_id: str, content: str, *, finalize: bool = False,
    ) -> SendResult:
        """Progressively update a bot activity (gateway send-then-edit streaming).

        Mid-stream (``finalize=False``) truncates oversize text in place so the
        edit target stays this activity. Identical payloads are skipped. HTTP 429
        with a short Retry-After is waited inline; a longer wait or 404/405
        returns ``success=False`` so the stream consumer falls back to a plain send.
        ``draft_stream_is_message`` stays False: Teams has no native stream object.
        """
        if not self._app:
            return SendResult(success=False, error="Teams app not initialized")
        if not chat_id or not message_id:
            return SendResult(success=False, error="missing conversation or activity id")
        formatted = self.format_message(content)
        key = (str(chat_id), str(message_id))
        oversize = len(formatted) > self.MAX_MESSAGE_LENGTH
        if oversize:
            formatted = self.truncate_message(formatted, self.MAX_MESSAGE_LENGTH)[0]
            if not finalize and self._last_overflow_preview.get(key) == formatted:
                return SendResult(success=True, message_id=str(message_id))
            if not finalize:
                self._remember_edit_cache(self._last_overflow_preview, key, formatted)
        elif not finalize:
            self._last_overflow_preview.pop(key, None)
        if finalize:
            self._last_overflow_preview.pop(key, None)
        if not finalize and self._last_edit_text.get(key) == formatted:
            return SendResult(success=True, message_id=str(message_id))
        try:
            await self._update_activity(str(chat_id), str(message_id), formatted)
        except Exception as e:
            return await self._on_edit_activity_error(
                e, chat_id=str(chat_id), message_id=str(message_id), text=formatted)
        self._remember_edit_cache(self._last_edit_text, key, formatted)
        return SendResult(success=True, message_id=str(message_id))

    def _remember_edit_cache(self, cache: Dict[tuple, str], key: tuple, text: str) -> None:
        if key not in cache and len(cache) >= _EDIT_CACHE_MAX:
            cache.pop(next(iter(cache)))
        cache[key] = text

    async def _on_edit_activity_error(
        self, exc: BaseException, *, chat_id: str, message_id: str, text: str,
    ) -> SendResult:
        """Classify an activity-update failure: short 429 retry, unsupported → fallback send."""
        status = _http_status_from_exc(exc)
        retry_after = _retry_after_seconds(exc)
        if status == 429 or retry_after is not None:
            wait = float(retry_after) if retry_after is not None else 1.0
            if wait <= _EDIT_FLOOD_INLINE_WAIT_CAP_SECS:
                logger.debug("[teams] activity update 429, waiting %.1fs", wait)
                await asyncio.sleep(wait)
                try:
                    await self._update_activity(chat_id, message_id, text)
                    self._remember_edit_cache(
                        self._last_edit_text, (chat_id, message_id), text)
                    return SendResult(success=True, message_id=message_id)
                except Exception as retry_err:
                    logger.warning("[teams] activity update retry failed: %s", retry_err)
                    return SendResult(
                        success=False, error=str(retry_err), retryable=True,
                        error_kind="rate_limited")
            return SendResult(
                success=False, error=str(exc), retryable=True, retry_after=wait,
                error_kind="rate_limited")
        if _activity_update_unsupported(exc, status):
            logger.info("[teams] activity update unsupported (%s); stream will fall back to send",
                        status or exc)
            return SendResult(
                success=False, error=str(exc),
                error_kind="not_found" if status == 404 else None)
        logger.warning("[teams] edit_message failed: %s", exc)
        return SendResult(
            success=False, error=str(exc),
            retryable=status is None or status >= 500)

    async def send_typing(self, chat_id: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        if self._app:
            with suppress(Exception):
                await self._app.send(chat_id, TypingActivityInput())

    async def _send_media_attachment(
        self, chat_id: str, source: str, default_mime: str, caption: Optional[str] = None,
        media_label: str = "media", file_name: Optional[str] = None,
    ) -> SendResult:
        """Send any media file/URL as a Teams attachment (shared by send_image/video/voice/document).
        Remote ``http(s)://`` URLs are attached by reference; local paths (optional ``file://`` prefix)
        are base64-encoded into a data URI. MIME is guessed from the path, else ``default_mime``."""
        if not self._app:
            return SendResult(success=False, error="Teams app not initialized")
        try:
            import base64
            import mimetypes
            from microsoft_teams.api import Attachment, MessageActivityInput

            if source.startswith(("http://", "https://")):
                content_url = source
                mime_type = mimetypes.guess_type(source.split("?")[0])[0] or default_mime
                name = file_name or os.path.basename(source.split("?")[0]) or None
            else:
                path = source.removeprefix("file://")
                mime_type = mimetypes.guess_type(path)[0] or default_mime
                name = file_name or os.path.basename(path) or None
                with open(path, "rb") as f:
                    content_url = f"data:{mime_type};base64,{base64.b64encode(f.read()).decode()}"
            activity = MessageActivityInput().add_attachments(
                Attachment(content_type=mime_type, content_url=content_url, name=name))
            if caption:
                activity = activity.add_text(caption)
            result = await self._send_via_conv_ref(chat_id, activity, activity)
            return SendResult(success=True, message_id=getattr(result, "id", None))
        except Exception as e:
            logger.error("[teams] send_%s failed: %s", media_label, e, exc_info=True)
            return SendResult(success=False, error=str(e), retryable=True)

    async def send_image(self, chat_id: str, image_url: str, caption: Optional[str] = None, reply_to: Optional[str] = None,
                         metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        return await self._send_media_attachment(chat_id, image_url, "image/png", caption=caption, media_label="image")

    async def send_image_file(self, chat_id: str, image_path: str, caption: Optional[str] = None,
                              reply_to: Optional[str] = None, **kwargs) -> SendResult:
        return await self.send_image(chat_id=chat_id, image_url=image_path, caption=caption, reply_to=reply_to)

    async def send_video(self, chat_id: str, video_path: str, caption: Optional[str] = None, reply_to: Optional[str] = None,
                         metadata: Optional[Dict[str, Any]] = None, **kwargs) -> SendResult:
        return await self._send_media_attachment(chat_id, video_path, "video/mp4", caption=caption, media_label="video")

    async def send_voice(self, chat_id: str, audio_path: str, caption: Optional[str] = None, reply_to: Optional[str] = None,
                         metadata: Optional[Dict[str, Any]] = None, **kwargs) -> SendResult:
        return await self._send_media_attachment(chat_id, audio_path, "audio/mpeg", caption=caption, media_label="voice")

    async def send_document(self, chat_id: str, file_path: str, caption: Optional[str] = None, file_name: Optional[str] = None,
                            reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None, **kwargs) -> SendResult:
        """Send a file. Personal chats use FileConsent; channel/group local files are
        inlined when they are small text, otherwise uploaded via Graph/SharePoint
        (Bot Framework document attachments 400 in channels). Remote URLs stay attachments."""
        if file_path.startswith(("http://", "https://")):
            return await self._send_media_attachment(
                chat_id, file_path, "application/octet-stream", caption=caption,
                media_label="document", file_name=file_name)
        conv_type = self._conversation_type(chat_id)
        if conv_type and conv_type != "personal":
            return await self._send_channel_document(
                chat_id, file_path, caption=caption, file_name=file_name)
        return await self._send_file_consent(
            chat_id, file_path, caption=caption, file_name=file_name)

    async def _send_channel_document(
        self, chat_id: str, file_path: str, *, caption: Optional[str] = None, file_name: Optional[str] = None,
    ) -> SendResult:
        """Channel/group file send: inline small text, else Graph → SharePoint link.

        Never base64 document attachments (Bot Framework returns 400 in channels).
        """
        path = file_path.removeprefix("file://")
        name = file_name or os.path.basename(path) or "file"
        if _is_inlineable_channel_document(path, name):
            text = _read_inline_channel_text(path)
            if text is not None:
                language = os.path.splitext(name)[1].lstrip(".").lower()
                body = f"**{name}**\n\n{_fence_channel_text(text, language=language)}"
                if caption:
                    body = f"{caption}\n\n{body}"
                return await self.send(chat_id, body)
        return await self._send_channel_document_via_graph(
            chat_id, path, caption=caption, file_name=name)

    def _remember_graph_file_target(self, chat_id: str, activity: Any) -> None:
        """Stash Graph team/channel/chat ids from inbound channelData for later uploads."""
        from plugins.platforms.teams.graph_files import extract_graph_file_target
        target = extract_graph_file_target(activity)
        if target is not None:
            self._graph_file_targets[chat_id] = target

    def _graph_client_for_files(self) -> Any:
        """Injected client, ``False`` to disable, or app-only Graph credentials."""
        injected = self._graph_client
        if injected is not None:
            return None if injected is False else injected
        from plugins.platforms.teams.graph_files import resolve_graph_credentials
        from tools.microsoft_graph_auth import MicrosoftGraphTokenProvider
        from tools.microsoft_graph_client import MicrosoftGraphClient
        creds = resolve_graph_credentials(
            teams_tenant_id=self._tenant_id or "",
            teams_client_id=self._client_id or "",
            teams_client_secret=self._client_secret or "",
        )
        if creds is None:
            return None
        return MicrosoftGraphClient(MicrosoftGraphTokenProvider(creds))

    def _graph_file_target_for(self, chat_id: str) -> Any:
        from plugins.platforms.teams.graph_files import resolve_graph_file_target
        extra_team = str(
            self._extra.get("team_id") or _get_scoped_secret("TEAMS_TEAM_ID", "") or ""
        ).strip()
        extra_channel = str(
            self._extra.get("channel_id") or _get_scoped_secret("TEAMS_CHANNEL_ID", "") or ""
        ).strip()
        return resolve_graph_file_target(
            chat_id,
            conv_type=self._conversation_type(chat_id),
            cached=self._graph_file_targets.get(chat_id),
            extra_team_id=extra_team,
            extra_channel_id=extra_channel,
        )

    async def _channel_file_failure(self, chat_id: str, name: str, note: str) -> SendResult:
        text = f"`{name}` — {note}"
        with suppress(Exception):
            await self.send(chat_id, text)
        return SendResult(success=False, error=text)

    async def _send_channel_document_via_graph(
        self, chat_id: str, path: str, *, caption: Optional[str] = None, file_name: str,
    ) -> SendResult:
        """Upload a binary (or oversized text) file via Graph and post a SharePoint link."""
        import mimetypes
        from plugins.platforms.teams.graph_files import (
            GraphFileTargetError, GraphFileUploadError, markdown_file_link, upload_conversation_file,
        )
        from tools.microsoft_graph_auth import MicrosoftGraphAuthError, MicrosoftGraphConfigError
        from tools.microsoft_graph_client import MicrosoftGraphAPIError, MicrosoftGraphClientError

        try:
            size = os.path.getsize(path)
        except OSError as e:
            return SendResult(success=False, error=f"Cannot read file: {e}")
        if size > _MAX_FILE_SEND_BYTES:
            return await self._channel_file_failure(
                chat_id, file_name,
                f"File exceeds Teams send limit ({_MAX_FILE_SEND_BYTES // (1024 * 1024)} MB). "
                "FileConsent cards work only in a 1:1 chat with the bot.")
        if size == 0:
            return await self._channel_file_failure(
                chat_id, file_name, "File is empty. FileConsent cards work only in a 1:1 chat with the bot.")
        try:
            with open(path, "rb") as fh:
                data = fh.read()
        except OSError as e:
            return SendResult(success=False, error=f"Cannot read file: {e}")

        client = self._graph_client_for_files()
        if client is None:
            return await self._channel_file_failure(chat_id, file_name, _CHANNEL_FILE_GRAPH_NOT_CONFIGURED)
        target = self._graph_file_target_for(chat_id)
        if target is None:
            return await self._channel_file_failure(chat_id, file_name, _CHANNEL_FILE_GRAPH_NO_TARGET)
        mime_type = mimetypes.guess_type(file_name or path)[0] or "application/octet-stream"
        try:
            uploaded = await upload_conversation_file(
                client, target, file_name=file_name, data=data, content_type=mime_type)
        except MicrosoftGraphConfigError:
            return await self._channel_file_failure(chat_id, file_name, _CHANNEL_FILE_GRAPH_NOT_CONFIGURED)
        except MicrosoftGraphAPIError as e:
            detail = str(e)
            if e.status_code in (401, 403):
                note = _CHANNEL_FILE_GRAPH_PERMISSIONS.format(status=e.status_code, detail=detail)
            else:
                note = _CHANNEL_FILE_GRAPH_FAILED.format(detail=detail)
            logger.warning("[teams] Graph channel file upload failed: %s", e)
            return await self._channel_file_failure(chat_id, file_name, note)
        except (MicrosoftGraphAuthError, MicrosoftGraphClientError, GraphFileUploadError, GraphFileTargetError) as e:
            logger.warning("[teams] Graph channel file upload failed: %s", e)
            return await self._channel_file_failure(
                chat_id, file_name, _CHANNEL_FILE_GRAPH_FAILED.format(detail=str(e)))
        except Exception as e:
            logger.error("[teams] Graph channel file upload failed: %s", e, exc_info=True)
            return await self._channel_file_failure(
                chat_id, file_name, _CHANNEL_FILE_GRAPH_FAILED.format(detail=str(e)))

        url = uploaded.link
        body = f"**{uploaded.name}**\n{markdown_file_link(uploaded.name, url)}"
        if caption:
            body = f"{caption}\n\n{body}"
        return await self.send(chat_id, body)

    def _conversation_type(self, chat_id: str) -> Optional[str]:
        """Cached conversation_type for ``chat_id``, or ``None`` when unseen this process."""
        ref = self._conv_ref_for(chat_id)
        conv = getattr(ref, "conversation", None)
        cached_type = getattr(conv, "conversation_type", None) or getattr(conv, "conversationType", None)
        if cached_type:
            return cached_type
        target = self._graph_file_targets.get(chat_id)
        return getattr(target, "conversation_type", None)

    async def get_chat_info(self, chat_id: str) -> dict:
        return {"name": chat_id, "type": "unknown", "chat_id": chat_id}

    # -- File consent (personal-chat native file send) --

    def _remember_pending_upload(self, file_id: str, payload: Dict[str, Any]) -> None:
        while len(self._pending_upload_ids) >= _PENDING_UPLOAD_MAX:
            old = self._pending_upload_ids.popleft()
            self._pending_uploads.pop(old, None)
        self._pending_uploads[file_id] = payload
        self._pending_upload_ids.append(file_id)

    async def _send_file_consent(
        self, chat_id: str, file_path: str, *, caption: Optional[str] = None, file_name: Optional[str] = None,
    ) -> SendResult:
        """Offer a FileConsentCard; the file lands in OneDrive after the user taps Accept."""
        if not self._app:
            return SendResult(success=False, error="Teams app not initialized")
        path = file_path.removeprefix("file://")
        try:
            size = os.path.getsize(path)
        except OSError as e:
            return SendResult(success=False, error=f"Cannot read file: {e}")
        if size > _MAX_FILE_SEND_BYTES:
            return SendResult(
                success=False,
                error=f"File exceeds Teams send limit ({_MAX_FILE_SEND_BYTES // (1024 * 1024)} MB)")
        try:
            with open(path, "rb") as fh:
                data = fh.read()
        except OSError as e:
            return SendResult(success=False, error=f"Cannot read file: {e}")
        name = file_name or os.path.basename(path) or "file"
        file_id = uuid.uuid4().hex
        self._remember_pending_upload(file_id, {"name": name, "bytes": data})
        try:
            from microsoft_teams.api import Attachment, MessageActivityInput
            activity = MessageActivityInput().add_attachments(Attachment(
                content_type=_CONTENT_TYPE_FILE_CONSENT,
                name=name,
                content={
                    "description": caption or name,
                    "sizeInBytes": size,
                    "acceptContext": {"file_id": file_id},
                    "declineContext": {"file_id": file_id},
                },
            ))
            if caption:
                activity = activity.add_text(caption)
            result = await self._send_via_conv_ref(chat_id, activity, activity)
            return SendResult(success=True, message_id=getattr(result, "id", None))
        except Exception as e:
            self._pending_uploads.pop(file_id, None)
            logger.error("[teams] send_document (file consent) failed: %s", e, exc_info=True)
            return SendResult(success=False, error=str(e), retryable=True)

    async def _on_file_consent(self, ctx) -> None:
        """Handle ``fileConsent/invoke`` accept/decline from a FileConsentCard."""
        activity = ctx.activity
        value = getattr(activity, "value", None)
        action = _normalize_consent_action(_invoke_field(value, "action"))
        logger.info("[teams] file consent invoke action=%s", action or "(empty)")
        context = _invoke_field(value, "context") or {}
        file_id = _invoke_field(context, "file_id", "fileId") if context is not None else None
        chat_id = getattr(getattr(activity, "conversation", None), "id", None)
        card_id = _consent_card_activity_id(activity)
        denied = self._card_action_denied(getattr(activity, "from_", None))
        if denied:
            logger.warning("[teams] file consent rejected: %s", denied)
            await self._dismiss_consent_card(chat_id, card_id)
            return
        if action == "decline":
            if file_id:
                self._pending_uploads.pop(str(file_id), None)
            if chat_id:
                with suppress(Exception):
                    await self.send(str(chat_id), "File upload declined.")
            await self._dismiss_consent_card(chat_id, card_id)
            return
        if action != "accept":
            return
        upload_info = _invoke_field(value, "upload_info", "uploadInfo")
        upload_url = _invoke_field(upload_info, "upload_url", "uploadUrl") if upload_info is not None else None
        if not upload_url or not _is_allowed_onedrive_upload_url(str(upload_url)):
            logger.warning("[teams] file consent accept with missing/unsafe upload URL")
            return
        from tools.url_safety import is_safe_url
        if not is_safe_url(str(upload_url)):
            logger.warning("[teams] file consent upload URL failed SSRF check")
            return
        pending = self._pending_uploads.pop(str(file_id), None) if file_id else None
        if not pending:
            if chat_id:
                with suppress(Exception):
                    await self.send(str(chat_id), "That file is no longer available to upload.")
            await self._dismiss_consent_card(chat_id, card_id)
            return
        try:
            await self._upload_consented_file(str(upload_url), pending["bytes"])
            await self._send_file_info_card(str(chat_id), upload_info, pending["name"])
        except Exception as e:
            logger.error("[teams] file consent upload failed: %s", e, exc_info=True)
            if chat_id:
                with suppress(Exception):
                    await self.send(str(chat_id), "File upload failed.")
            await self._dismiss_consent_card(chat_id, card_id)
            return
        await self._dismiss_consent_card(chat_id, card_id)

    async def _dismiss_consent_card(self, chat_id: Optional[str], activity_id: Optional[str]) -> None:
        """Delete the FileConsentCard so Accept/Decline cannot be clicked again.

        Uses ``api.conversations.activities(chat_id).delete`` — the same client as
        streaming ``edit_message`` (``activities.update``). Failures are logged and
        never fail the upload path.
        """
        if not chat_id or not activity_id:
            return
        try:
            conv_id, ops = self._conversation_activity_ops(chat_id)
            delete_fn = getattr(ops, "delete", None) if ops is not None else None
            if callable(delete_fn):
                result = delete_fn(str(activity_id))
                if inspect.isawaitable(result):
                    await result
                return
            await self._delete_activity_via_rest(conv_id, str(activity_id))
        except Exception as e:
            logger.debug("[teams] file consent card dismiss failed: %s", e)

    def _conversation_activity_ops(self, chat_id: str) -> tuple[str, Any]:
        """``(flat conversation id, SDK activities client or None)``."""
        conv_id = _flat_conversation_id(chat_id)
        api = getattr(self._app, "api", None) if self._app else None
        conversations = getattr(api, "conversations", None) if api is not None else None
        activities_fn = getattr(conversations, "activities", None) if conversations is not None else None
        if not callable(activities_fn):
            return conv_id, None
        return conv_id, activities_fn(str(conv_id))

    async def _update_activity(self, chat_id: str, activity_id: str, text: str) -> None:
        """PUT the activity via the SDK client, else Bot Framework REST."""
        conv_id, ops = self._conversation_activity_ops(chat_id)
        if not _TEAMS_CONV_ID_RE.match(conv_id) or not _TEAMS_CONV_ID_RE.match(activity_id):
            raise ValueError("conversation/activity id outside the Bot Framework charset")
        update_fn = getattr(ops, "update", None) if ops is not None else None
        if callable(update_fn):
            from microsoft_teams.api import MessageActivityInput
            activity = MessageActivityInput()
            adder = getattr(activity, "add_text", None) or getattr(activity, "with_text", None)
            if callable(adder):
                activity = adder(text) or activity
            with suppress(Exception):
                activity.id = str(activity_id)
            result = update_fn(str(activity_id), activity)
            if inspect.isawaitable(result):
                await result
            return
        await self._update_activity_via_rest(conv_id, activity_id, text)

    async def _update_activity_via_rest(self, conversation_id: str, activity_id: str, text: str) -> None:
        """PUT ``/v3/conversations/{id}/activities/{id}`` (ConversationActivityClient.update)."""
        import httpx
        conv_id = _flat_conversation_id(conversation_id)
        if not _TEAMS_CONV_ID_RE.match(conv_id) or not _TEAMS_CONV_ID_RE.match(activity_id):
            raise ValueError("conversation/activity id outside the Bot Framework charset")
        token = await self._get_botframework_token()
        url = _bf_activity_url(self._service_url_for(conversation_id), conv_id, activity_id)
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        payload = {"type": "message", "id": activity_id, "text": text, "textFormat": "markdown"}
        async with httpx.AsyncClient(timeout=15.0, trust_env=gateway_trust_env()) as client:
            response = await client.put(url, json=payload, headers=headers)
            if response.status_code == 429:
                raise httpx.HTTPStatusError(
                    f"Teams activity update rate-limited ({response.status_code})",
                    request=response.request, response=response)
            response.raise_for_status()

    async def _delete_activity_via_rest(self, chat_id: str, activity_id: str) -> None:
        """DELETE ``/v3/conversations/{id}/activities/{id}`` (ConversationActivityClient.delete)."""
        import httpx
        conv_id = _flat_conversation_id(chat_id)
        if not _TEAMS_CONV_ID_RE.match(conv_id) or not _TEAMS_CONV_ID_RE.match(activity_id):
            raise ValueError("conversation/activity id outside the Bot Framework charset")
        token = await self._get_botframework_token()
        url = _bf_activity_url(self._service_url_for(chat_id), conv_id, activity_id)
        headers = {"Authorization": f"Bearer {token}"}
        async with httpx.AsyncClient(timeout=15.0, trust_env=gateway_trust_env()) as client:
            response = await client.delete(url, headers=headers)
            response.raise_for_status()

    async def _upload_consented_file(self, upload_url: str, data: bytes) -> None:
        """PUT file bytes into the OneDrive upload session Teams returned on accept."""
        from tools.url_safety import create_ssrf_safe_async_client
        from gateway.platforms.base import _ssrf_redirect_guard
        size = len(data)
        headers = {
            "Content-Type": "application/octet-stream",
            "Content-Length": str(size),
            "Content-Range": f"bytes 0-{size - 1}/{size}",
        }
        async with create_ssrf_safe_async_client(
            timeout=60.0, follow_redirects=True, event_hooks={"response": [_ssrf_redirect_guard]},
        ) as client:
            response = await client.put(upload_url, content=data, headers=headers)
            response.raise_for_status()

    async def _send_file_info_card(self, chat_id: str, upload_info: Any, fallback_name: str) -> None:
        """Notify the user with a FileInfoCard after a successful consent upload."""
        from microsoft_teams.api import Attachment, MessageActivityInput
        name = _invoke_field(upload_info, "name") or fallback_name
        content_url = _invoke_field(upload_info, "content_url", "contentUrl")
        unique_id = _invoke_field(upload_info, "unique_id", "uniqueId")
        file_type = _invoke_field(upload_info, "file_type", "fileType")
        activity = MessageActivityInput().add_attachments(Attachment(
            content_type=_CONTENT_TYPE_FILE_INFO,
            name=name,
            content_url=content_url,
            content={"uniqueId": unique_id, "fileType": file_type},
        ))
        if name:
            activity = activity.add_text(f"**{name}** uploaded.")
        await self._send_via_conv_ref(chat_id, activity, activity)

    # -- Reactions --

    def _reactions_enabled(self) -> bool:
        """Processing-lifecycle reactions: scoped ``TEAMS_REACTIONS`` → ``extra.reactions`` → on.

        Agent-facing ``add_reaction`` / ``remove_reaction`` (send_message action=react) are
        NOT gated — they are deliberate intents, same as Photon.
        """
        configured = _extra_or_secret(self.config.extra, "reactions", "TEAMS_REACTIONS", True)
        if isinstance(configured, bool):
            return configured
        return str(configured).strip().lower() not in {"false", "0", "no", "off"}

    def _service_url_for(self, chat_id: str) -> str:
        """Bot Framework service URL for this conversation (conv-ref, else the allowlisted default)."""
        ref = self._conv_ref_for(chat_id)
        raw = getattr(ref, "service_url", None) or _DEFAULT_TEAMS_SERVICE_URL
        return _validate_teams_service_url(str(raw)) or _DEFAULT_TEAMS_SERVICE_URL

    async def _react(self, chat_id: str, message_id: str, reaction_type: str, *, remove: bool) -> bool:
        """Add or remove a Teams reaction via the SDK client, else Bot Framework REST."""
        conv_id = _flat_conversation_id(chat_id)
        if not self._app or not conv_id or not message_id or not _REACTION_TYPE_RE.match(reaction_type):
            return False
        api = getattr(self._app, "api", None)
        reactions = getattr(api, "reactions", None) if api is not None else None
        method_name = "delete" if remove else "add"
        sdk_fn = getattr(reactions, method_name, None)
        try:
            if callable(sdk_fn):
                await sdk_fn(conv_id, message_id, reaction_type)
            else:
                await self._react_via_rest(conv_id, message_id, reaction_type, remove=remove)
        except Exception as e:
            logger.debug("[teams] reaction %s failed (%s): %s", method_name, reaction_type, e)
            return False
        key = (str(chat_id), str(message_id))
        if remove:
            if self._bot_reactions.get(key) == reaction_type:
                self._bot_reactions.pop(key, None)
        else:
            self._bot_reactions[key] = reaction_type
        return True

    async def _react_via_rest(self, chat_id: str, message_id: str, reaction_type: str, *, remove: bool) -> None:
        """PUT/DELETE ``/v3/conversations/{id}/activities/{id}/reactions/{type}`` (SDK ReactionClient)."""
        import httpx
        conv_id = _flat_conversation_id(chat_id)
        if not _TEAMS_CONV_ID_RE.match(conv_id) or not _TEAMS_CONV_ID_RE.match(message_id):
            raise ValueError("conversation/activity id outside the Bot Framework charset")
        token = await self._get_botframework_token()
        url = (
            f"{_bf_activity_url(self._service_url_for(chat_id), conv_id, message_id)}"
            f"/reactions/{quote(reaction_type, safe='')}"
        )
        headers = {"Authorization": f"Bearer {token}"}
        async with httpx.AsyncClient(timeout=15.0, trust_env=gateway_trust_env()) as client:
            response = await (client.delete(url, headers=headers) if remove else client.put(url, headers=headers))
            response.raise_for_status()

    async def _add_reaction(self, chat_id: str, message_id: str, emoji: str) -> bool:
        """Lifecycle hook: add ``emoji`` (unicode or Teams id) on a message."""
        rtype = _to_teams_reaction_type(emoji)
        if not rtype:
            return False
        return await self._react(chat_id, message_id, rtype, remove=False)

    async def _remove_reaction(self, chat_id: str, message_id: str, emoji: Optional[str] = None) -> bool:
        """Lifecycle hook: remove a reaction. Bare call removes the in-progress 👀 (or last bot set)."""
        rtype = _to_teams_reaction_type(emoji) if emoji else (
            self._bot_reactions.get((str(chat_id), str(message_id)))
            or _to_teams_reaction_type(self._ACK_EMOJI)
        )
        if not rtype:
            return False
        return await self._react(chat_id, message_id, rtype, remove=True)

    async def on_processing_start(self, event: MessageEvent) -> None:
        """👀 while the agent works (gated by TEAMS_REACTIONS)."""
        if not self._reactions_enabled():
            return
        chat_id = getattr(event.source, "chat_id", None)
        message_id = getattr(event, "message_id", None)
        if chat_id and message_id:
            await self._add_reaction(str(chat_id), str(message_id), self._ACK_EMOJI)

    async def add_reaction(self, chat_id: str, emoji: str, message_id: Optional[str] = None) -> Dict[str, Any]:
        """Agent-facing react (send_message action='react'); not gated by TEAMS_REACTIONS."""
        target = message_id or self._last_inbound_by_chat.get(str(chat_id))
        if not target:
            return {"success": False, "error": "no message to react to — pass message_id (no "
                    "inbound message seen in this chat since the gateway started)"}
        rtype = _to_teams_reaction_type(emoji)
        if not rtype:
            return {"success": False, "error": f"unsupported Teams reaction {emoji!r}"}
        if not await self._react(chat_id, target, rtype, remove=False):
            return {"success": False, "error": "reaction failed (see gateway debug log)"}
        return {"success": True, "message_id": target, "reaction": rtype}

    async def remove_reaction(self, chat_id: str, message_id: Optional[str] = None,
                              emoji: Optional[str] = None) -> Dict[str, Any]:
        """Agent-facing unreact (send_message action='unreact')."""
        target = message_id or self._last_inbound_by_chat.get(str(chat_id))
        if not target:
            return {"success": False, "error": "no message to unreact — pass message_id"}
        if not await self._remove_reaction(chat_id, target, emoji):
            return {"success": False, "error": "unreact failed (see gateway debug log)"}
        return {"success": True, "message_id": target}

    async def _on_message_reaction(self, ctx) -> None:
        """Inbound ``messageReaction`` → gateway reaction hooks + platform-event envelope."""
        activity = ctx.activity
        recipient_id = getattr(getattr(activity, "recipient", None), "id", None)
        bot_ids = {i for i in (self._app.id if self._app else None, recipient_id) if isinstance(i, str) and i}
        bot_ids |= {f"28:{i}" for i in tuple(bot_ids) if not i.startswith("28:")}
        from_account = getattr(activity, "from_", None)
        user_id = getattr(from_account, "aad_object_id", None) or getattr(from_account, "id", "")
        if str(user_id) in bot_ids or getattr(from_account, "id", None) in bot_ids:
            return
        conv = getattr(activity, "conversation", None)
        conv_id = getattr(conv, "id", None)
        message_id = getattr(activity, "reply_to_id", None) or getattr(activity, "id", None)
        if not conv_id or not message_id:
            return
        added = list(getattr(activity, "reactions_added", None) or [])
        removed = list(getattr(activity, "reactions_removed", None) or [])
        source = self.build_source(
            chat_id=str(conv_id),
            chat_name=getattr(conv, "name", None) or "",
            chat_type=_CHAT_TYPES.get(getattr(conv, "conversation_type", None) or "", "dm"),
            user_id=str(user_id),
            user_name=getattr(from_account, "name", None) or "",
            guild_id=getattr(conv, "tenant_id", None) or self._tenant_id,
            message_id=str(message_id))
        for reaction, event_name in (
            *((r, "reaction:added") for r in added),
            *((r, "reaction:removed") for r in removed),
        ):
            rtype = getattr(reaction, "type", None) if not isinstance(reaction, dict) else reaction.get("type")
            emoji = _reaction_to_emoji(str(rtype) if rtype else "")
            if not emoji:
                continue
            await self._emit_reaction_event(
                event_name, emoji, str(rtype or emoji), source, activity)

    async def _emit_reaction_event(
        self, event_name: str, emoji: str, reaction_type: str, source, raw_activity: Any,
    ) -> None:
        """Fan a reaction out to the gateway hook (Slack shape) and platform-event plugins."""
        handler = getattr(self, "_reaction_handler", None)
        if handler is not None:
            try:
                await handler({
                    "platform": "teams", "event_name": event_name, "reaction": emoji,
                    "user_id": source.user_id, "item_user_id": None,
                    "channel_id": source.chat_id, "message_ts": source.message_id,
                    "event_ts": getattr(raw_activity, "id", None), "raw_event": raw_activity,
                    "reaction_type": reaction_type,
                })
            except Exception:
                logger.debug("[teams] reaction hook forwarding failed", exc_info=True)
        platform_handler = getattr(self, "_platform_event_handler", None)
        if platform_handler is None:
            return
        try:
            from hermes_cli.lifecycle import has_hook
            if not has_hook("gateway_platform_event"):
                return
            envelope = {
                "platform": "teams",
                "event_type": "reaction",
                "payload": {
                    "emojis": [emoji], "chat_id": source.chat_id,
                    "message_id": str(source.message_id or ""), "thread_id": None,
                    "event_name": event_name, "reaction_type": reaction_type,
                },
            }
            await platform_handler(envelope, source)
        except Exception:
            logger.debug("[teams] gateway_platform_event reaction dispatch failed", exc_info=True)


_SETUP_CREDENTIALS = (
    ("Client ID", "TEAMS_CLIENT_ID", {}),
    ("Client secret", "TEAMS_CLIENT_SECRET", {"password": True}),
    ("Tenant ID", "TEAMS_TENANT_ID", {}))
_SETUP_INTRO = (  # "" → blank line
    "You'll need the Teams CLI. If you haven't already:", "  npm install -g @microsoft/teams.cli@preview",
    "  teams login", "", "Then expose port 3978 publicly (devtunnel / ngrok / cloudflared),", "and create your bot:",
    '  teams app create --name "Hermes" --endpoint "https://<tunnel>/api/messages"', "",
    "The CLI will print CLIENT_ID, CLIENT_SECRET, and TENANT_ID. Paste them below.", "")


def interactive_setup() -> None:
    from hermes_cli.config import get_env_value, save_env_value
    from hermes_cli.cli_output import prompt, prompt_yes_no, print_info, print_success, print_warning
    from hermes_cli.setup_platforms import declines_reconfigure
    if declines_reconfigure("Teams", "Reconfigure Teams?", "TEAMS_CLIENT_ID"):
        return
    for line in _SETUP_INTRO:
        print_info(line) if line else print()
    for label, env_key, prompt_kwargs in _SETUP_CREDENTIALS:
        value = prompt(label, default=get_env_value(env_key) or "", **prompt_kwargs)
        if not value:
            print_warning(f"{label} is required — skipping Teams setup")
            return
        save_env_value(env_key, value.strip())
    print()
    print_info("To find your AAD object ID for the allowlist: teams status --verbose")
    if prompt_yes_no("Restrict access to specific users? (recommended)", True):
        allowed = prompt("Allowed AAD object IDs (comma-separated)", default=get_env_value("TEAMS_ALLOWED_USERS") or "")
        if allowed:
            save_env_value("TEAMS_ALLOWED_USERS", allowed.replace(" ", ""))
            print_success("Allowlist configured")
        else:
            save_env_value("TEAMS_ALLOWED_USERS", "")
    else:
        save_env_value("TEAMS_ALLOW_ALL_USERS", "true")
        print_warning("⚠️  Open access — anyone who can message the bot can command it.")
    print()
    print_success("Teams configuration saved to ~/.hermes/.env")
    print_info("Install the app in Teams:  teams app install --id <teamsAppId>")
    print_info("Restart the gateway:       hermes gateway restart")


def _install_hint() -> str:
    """Install hint derived from the LAZY_DEPS pins (aiohttp is CVE-pinned, so bumps happen);
    ``venv_pip=True`` targets the real Hermes venv, sidestepping PEP 668 on Ubuntu 24.04."""
    try:
        from tools.lazy_deps import feature_install_command
        cmd = feature_install_command("platform.teams", venv_pip=True)
    except Exception:  # pragma: no cover — defensive
        cmd = None
    if not cmd:
        cmd = f"{sys.executable} -m pip install microsoft-teams-apps aiohttp"
    return f"Teams SDK missing — restart the gateway to auto-install, or run: {cmd}"


def register(ctx) -> None:
    ctx.register_platform(
        name="teams", label="Microsoft Teams", adapter_factory=lambda cfg: TeamsAdapter(cfg),
        check_fn=check_requirements,  # PASSIVE probe — never installs
        ensure_deps_fn=check_teams_requirements,  # ACTIVE lazy-installer, run by create_adapter()
        validate_config=validate_config, is_connected=is_connected,
        required_env=["TEAMS_CLIENT_ID", "TEAMS_CLIENT_SECRET", "TEAMS_TENANT_ID"],
        install_hint=_install_hint(), setup_fn=interactive_setup,
        env_enablement_fn=_env_enablement,  # env-only setups show up in gateway status
        cron_deliver_env_var="TEAMS_HOME_CHANNEL",  # deliver=teams cron home-channel routing
        standalone_sender_fn=_standalone_send,  # out-of-process cron delivery via Bot Framework REST
        allowed_users_env="TEAMS_ALLOWED_USERS", allow_all_env="TEAMS_ALLOW_ALL_USERS",
        max_message_length=28000,  # Teams supports up to ~28 KB per message
        emoji="💼", allow_update_command=True,
        platform_hint=(
            "You are chatting via Microsoft Teams. Teams renders a subset of "
            "markdown — bold (**text**), italic (*text*), and inline code "
            "(`code`) work, but complex tables or raw HTML do not. Keep "
            "responses clear and professional."))


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import html  # noqa: F401,E402
from urllib.parse import quote  # noqa: F401,E402


_PLUGIN_COMPAT_LAZY = {
    'TeamsSummaryWriter': ('plugins.platforms.teams.summary_writer', 'TeamsSummaryWriter'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----

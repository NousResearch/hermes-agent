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
import json
import logging
import os
import re
import sys
from contextlib import contextmanager, suppress
from typing import Any, Dict, Iterator, Optional
from urllib.parse import urlparse

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
    gateway_trust_env, BasePlatformAdapter, MessageEvent, MessageType, SendResult, cache_image_from_url, cache_media_bytes_async,
)
from gateway.platforms._shared import coerce_port, get_scoped_secret as _get_scoped_secret

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
    """(client_id, client_secret, tenant_id): env first, then ``config.extra``."""
    extra = getattr(config, "extra", {}) or {}
    return (
        os.getenv("TEAMS_CLIENT_ID") or extra.get("client_id", ""),
        _get_scoped_secret("TEAMS_CLIENT_SECRET") or extra.get("client_secret", ""),
        os.getenv("TEAMS_TENANT_ID") or extra.get("tenant_id", ""))


def validate_config(config) -> bool:
    return bool(all(_credentials(config)))


def is_connected(config) -> bool:
    return validate_config(config)


def _env_enablement() -> dict | None:
    """Seed ``PlatformConfig.extra`` from env before adapter construction so ``gateway status`` reflects
    env-only setups without the SDK. ``None`` when not minimally configured; ``home_channel`` becomes a
    ``HomeChannel`` via the core hook."""
    client_id = os.getenv("TEAMS_CLIENT_ID", "").strip()
    client_secret = _get_scoped_secret("TEAMS_CLIENT_SECRET", "").strip()
    tenant_id = os.getenv("TEAMS_TENANT_ID", "").strip()
    if not (client_id and client_secret and tenant_id):
        return None
    seed: dict = {"client_id": client_id, "client_secret": client_secret, "tenant_id": tenant_id}
    port = coerce_port(os.getenv("TEAMS_PORT", "").strip(), None)
    if port is not None:
        seed["port"] = port
    if service_url := os.getenv("TEAMS_SERVICE_URL", "").strip():
        seed["service_url"] = service_url
    if home := os.getenv("TEAMS_HOME_CHANNEL", "").strip():
        seed["home_channel"] = {"chat_id": home, "name": os.getenv("TEAMS_HOME_CHANNEL_NAME", "Home")}
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
        return {"error": "Teams standalone send: TEAMS_CLIENT_ID, TEAMS_CLIENT_SECRET, and TEAMS_TENANT_ID are all required"}
    raw_service_url = os.getenv("TEAMS_SERVICE_URL") or extra.get("service_url", "") or _DEFAULT_TEAMS_SERVICE_URL
    service_url = _validate_teams_service_url(raw_service_url)
    for failed, error in (
        (service_url is None, f"TEAMS_SERVICE_URL host is not on the Bot Framework allowlist; "
                              f"expected one of {sorted(_ALLOWED_TEAMS_SERVICE_HOSTS)}"),
        (not chat_id, "chat_id (conversation ID) is required"),
        (not _TEAMS_CONV_ID_RE.match(chat_id or ""), "chat_id contains characters outside the Bot Framework conversation ID set"),
        (not _TEAMS_CONV_ID_RE.match(tenant_id), "TEAMS_TENANT_ID contains characters outside the expected set"),
        (not AIOHTTP_AVAILABLE, "aiohttp not installed")):
        if failed:
            return {"error": f"Teams standalone send: {error}"}
    token_url, token_form = _bf_token_request(tenant_id, client_id, client_secret)
    activities_url = f"{service_url}v3/conversations/{chat_id}/activities"
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
                    return {"error": f"Teams standalone send: token request failed ({token_resp.status}): {body[:300]}"}
                token_payload = await token_resp.json()
            access_token = token_payload.get("access_token")
            if not access_token:
                return {"error": "Teams standalone send: token response missing access_token"}
            async with session.post(
                activities_url, json={"type": "message", "text": message, "textFormat": "markdown"},
                headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"},
                timeout=per_request_timeout,
            ) as send_resp:
                if send_resp.status >= 400:
                    body = await send_resp.text()
                    return {"error": f"Teams standalone send: activity post failed ({send_resp.status}): {body[:300]}"}
                send_payload = await send_resp.json()
        return {"success": True, "message_id": send_payload.get("id")}
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.debug("Teams standalone send raised", exc_info=True)
        return {"error": f"Teams standalone send failed: {e}"}


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
        body.append(TextBlock(text="⚠️ Command Approval Required", wrap=True, weight="Bolder"))
        body.append(TextBlock(text=f"```\n{cmd}\n```", wrap=True))
    if desc or always:
        body.append(TextBlock(text=f"Reason: {desc}", wrap=True, isSubtle=True))
    return body


class TeamsAdapter(BasePlatformAdapter):
    """Microsoft Teams adapter using the microsoft-teams-apps SDK."""

    MAX_MESSAGE_LENGTH = 28000  # Teams text message limit (~28 KB)
    splits_long_messages = True  # send() chunks via truncate_message()

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform("teams"))
        extra = config.extra or {}
        self._client_id = extra.get("client_id") or os.getenv("TEAMS_CLIENT_ID", "")
        self._client_secret = extra.get("client_secret") or _get_scoped_secret("TEAMS_CLIENT_SECRET", "")
        self._tenant_id = extra.get("tenant_id") or os.getenv("TEAMS_TENANT_ID", "")
        # (token, expiry monotonic ts) for connector attachment auth; refreshed under
        # _bf_token_lock so concurrent attachments can't stampede the STS.
        self._bf_token_cache: Optional[tuple] = None
        self._bf_token_lock: Optional[asyncio.Lock] = None
        self._port = coerce_port(extra.get("port") or os.getenv("TEAMS_PORT", str(_DEFAULT_PORT)), _DEFAULT_PORT)
        _raw_host = extra.get("host") or os.getenv("TEAMS_HOST", "") or _DEFAULT_HOST  # falsy → dual-stack None
        self._host: Optional[str] = str(_raw_host) if _raw_host else None
        self._app: Optional["App"] = None
        self._runner: Optional["web.AppRunner"] = None
        self._dedup = MessageDeduplicator(max_size=1000)
        # chat_id → ConversationReference so proactive cards use the right conversation type.
        self._conv_refs: Dict[str, Any] = {}

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

            self._wire_plugin_handlers(self._app)
            await self._app.initialize()
            self._runner = web.AppRunner(aiohttp_app)
            await self._runner.setup()
            site = web.TCPSite(self._runner, self._host, self._port)
            await site.start()
            self._running = True
            self._mark_connected()
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
        bot_id = self._app.id if self._app else None
        if bot_id and getattr(activity.from_, "id", None) == bot_id:
            return
        msg_id = getattr(activity, "id", None)
        if msg_id and self._dedup.is_duplicate(msg_id):
            return
        conv = activity.conversation
        conv_id = getattr(conv, "id", None)
        if conv_id:  # cache the conversation reference for proactive sends (approval cards, etc.)
            self._conv_refs[conv_id] = ctx.conversation_ref
        text = activity.text if hasattr(activity, "text") and activity.text else ""
        if "<at>" in text:  # strip the <at>BotName</at> tags Teams prepends for @mentions
            text = re.sub(r"<at>[^<]*</at>\s*", "", text).strip()
        from_account = activity.from_
        user_id = getattr(from_account, "aad_object_id", None) or getattr(from_account, "id", "")
        source = self.build_source(
            chat_id=conv.id,
            chat_name=getattr(conv, "name", None) or "",
            chat_type=_CHAT_TYPES.get(getattr(conv, "conversation_type", None) or "", "dm"),
            user_id=str(user_id),
            user_name=getattr(from_account, "name", None) or "",
            guild_id=getattr(conv, "tenant_id", None) or self._tenant_id)
        media: list = [m for m in [await self._cache_attachment(a) for a in getattr(activity, "attachments", None) or []] if m]
        media_kinds = [kind for _, _, kind in media]  # media items are (path, media_type, kind)
        msg_type = next((t for kind, t in _MEDIA_KIND_PRECEDENCE if kind in media_kinds), MessageType.TEXT)
        await self.handle_message(MessageEvent(
            text=text, source=source, message_type=msg_type, message_id=msg_id,
            media_urls=[path for path, _, _ in media], media_types=[mt for _, mt, _ in media]))

    async def _cache_attachment(self, att: Any) -> Optional[tuple]:
        """Download + cache one inbound attachment → ``(path, media_type, kind)`` or ``None``."""
        content_url = getattr(att, "content_url", None)
        content_type = (getattr(att, "content_type", None) or "").lower()
        att_name = getattr(att, "name", None) or ""
        # Skip non-file payloads: Teams mirrors the message body as a text/html attachment,
        # and cards arrive as application/vnd.microsoft.card.*
        if (content_type in ("text/html", "text/plain") and not content_url) or content_type.startswith("application/vnd.microsoft.card"):
            return None
        if content_type == "application/vnd.microsoft.teams.file.download.info":
            # Consent-free download: content carries a pre-authed SharePoint downloadUrl + file type.
            content = getattr(att, "content", None)
            if not isinstance(content, dict):
                content = getattr(content, "__dict__", None) or {}
            download_url = content.get("downloadUrl") or content.get("download_url")
            file_type = (content.get("fileType") or content.get("file_type") or "").lstrip(".")
            if not download_url:
                return None
            filename = att_name or (f"document.{file_type}" if file_type else "document")
            try:
                data = await self._fetch_attachment_bytes(download_url)
                cached = await cache_media_bytes_async(data, filename=filename, mime_type="")
                if not cached:
                    logger.warning("[teams] Unsupported document type for attachment '%s', skipping", filename)
                    return None
                return cached.path, cached.media_type, cached.kind
            except Exception as e:
                logger.warning("[teams] Failed to cache file attachment '%s': %s", filename, e)
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

    async def _send_card(self, chat_id: str, card: "AdaptiveCard") -> "Any":
        """Send an AdaptiveCard, using a stored ConversationReference when available."""
        from microsoft_teams.api import MessageActivityInput
        if not self._app:
            return None
        return await self._send_via_conv_ref(chat_id, MessageActivityInput().add_card(card), card)

    async def _send_via_conv_ref(self, chat_id: str, activity: Any, fallback: Any) -> Any:
        """Send ``activity`` through the cached ConversationReference, else ``App.send(fallback)``."""
        conv_ref = self._conv_refs.get(chat_id)
        if conv_ref:
            return await self._app.activity_sender.send(activity, conv_ref)
        return await self._app.send(chat_id, fallback)

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
        if os.getenv("TEAMS_ALLOW_ALL_USERS", "").strip().lower() in {"1", "true", "yes"}:
            return None
        allowed_csv = os.getenv("TEAMS_ALLOWED_USERS", "").strip()
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

    async def send_exec_approval(
        self, chat_id: str, command: str, session_key: str, description: str = "dangerous command",
        metadata: Optional[Dict[str, Any]] = None, allow_permanent: bool = True, allow_session: bool = True,
        smart_denied: bool = False) -> SendResult:
        if not self._app:
            return SendResult(success=False, error="Teams app not initialized")
        # Button data carries a truncated cmd — just enough to reconstruct the card body.
        btn_data_base = {"session_key": session_key, "cmd": _truncate(command, 200), "desc": description}

        def _action(title: str, hermes_action: str, **kw) -> "ExecuteAction":
            return ExecuteAction(
                title=title, verb="hermes_approve", data={**btn_data_base, "hermes_action": hermes_action}, **kw)

        actions = [_action("Allow Once", "approve_once", style="positive")]
        if not smart_denied and allow_session:
            actions.append(_action("Allow Session", "approve_session"))
            if allow_permanent:
                actions.append(_action("Always Allow", "approve_always"))
        actions.append(_action("Deny", "deny", style="destructive"))
        body = _approval_body(_truncate(command, 2000), description, always=True)
        if smart_denied:
            body.append(TextBlock(text="Smart DENY: owner override applies to this one operation only.", wrap=True))
        card = AdaptiveCard().with_version("1.4").with_body(body).with_actions(actions)
        try:
            result = await self._send_card(chat_id, card)
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
            except Exception as e:
                return SendResult(success=False, error=str(e), retryable=True)
        return SendResult(success=True, message_id=last_message_id)

    async def send_typing(self, chat_id: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        if self._app:
            with suppress(Exception):
                await self._app.send(chat_id, TypingActivityInput())

    async def _send_media_attachment(
        self, chat_id: str, source: str, default_mime: str, caption: Optional[str] = None, media_label: str = "media"
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
            else:
                path = source.removeprefix("file://")
                mime_type = mimetypes.guess_type(path)[0] or default_mime
                with open(path, "rb") as f:
                    content_url = f"data:{mime_type};base64,{base64.b64encode(f.read()).decode()}"
            activity = MessageActivityInput().add_attachments(Attachment(content_type=mime_type, content_url=content_url))
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
        return await self._send_media_attachment(
            chat_id, file_path, "application/octet-stream", caption=caption, media_label="document")

    async def get_chat_info(self, chat_id: str) -> dict:
        return {"name": chat_id, "type": "unknown", "chat_id": chat_id}


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
    existing_id = get_env_value("TEAMS_CLIENT_ID")
    if existing_id:
        print_info(f"Teams: already configured (app ID: {existing_id})")
        if not prompt_yes_no("Reconfigure Teams?", False):
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

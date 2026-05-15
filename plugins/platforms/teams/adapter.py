"""
Microsoft Teams platform adapter for Hermes Agent.

Uses the microsoft-teams-apps SDK for authentication and activity processing.
Runs an aiohttp webhook server to receive messages from Teams.
Proactive messaging (send, typing) uses the SDK's App.send() method.

Requires:
    pip install microsoft-teams-apps aiohttp
    TEAMS_CLIENT_ID, TEAMS_CLIENT_SECRET, and TEAMS_TENANT_ID env vars

Configuration in config.yaml:
    platforms:
      teams:
        enabled: true
        extra:
          client_id: "your-client-id"      # or TEAMS_CLIENT_ID env var
          client_secret: "your-secret"      # or TEAMS_CLIENT_SECRET env var
          tenant_id: "your-tenant-id"       # or TEAMS_TENANT_ID env var
          port: 3978                        # or TEAMS_PORT env var
"""

from __future__ import annotations

import asyncio
import hashlib
import html
import json
import logging
import os
import re
import time
from collections import OrderedDict
from typing import Any, Dict, Optional
from urllib.parse import quote

# httpx is imported lazily — only the ``_write_summary_via_incoming_webhook``
# code path actually constructs an ``AsyncClient``. Top-level import here
# pulled in the entire httpx + httpcore stack (~37 ms, ~15 MB) on every
# process that triggered plugin discovery, even ones that never instantiate
# the Teams adapter. ``from __future__ import annotations`` above keeps the
# ``httpx.AsyncBaseTransport`` parameter annotation valid as a string at
# runtime; nothing in the codebase calls ``typing.get_type_hints()`` on
# this class so the annotation never has to resolve to a real symbol.

try:
    from aiohttp import web

    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    web = None  # type: ignore[assignment]

try:
    from microsoft_teams.apps import App, ActivityContext
    from microsoft_teams.common.http.client import ClientOptions
    from microsoft_teams.api import MessageActivity, ConversationReference
    from microsoft_teams.api.activities.typing import TypingActivityInput
    from microsoft_teams.api.activities.invoke.adaptive_card import AdaptiveCardInvokeActivity
    from microsoft_teams.api.activities.invoke.file_consent import FileConsentInvokeActivity
    from microsoft_teams.api.models import FileUploadInfo
    from microsoft_teams.api.models.adaptive_card import (
        AdaptiveCardActionCardResponse,
        AdaptiveCardActionMessageResponse,
    )
    from microsoft_teams.api.models.invoke_response import InvokeResponse, AdaptiveCardInvokeResponse
    from microsoft_teams.apps.http.adapter import (
        HttpMethod,
        HttpRequest,
        HttpResponse,
        HttpRouteHandler,
    )
    from microsoft_teams.cards import AdaptiveCard, ExecuteAction, TextBlock

    TEAMS_SDK_AVAILABLE = True
except ImportError:
    TEAMS_SDK_AVAILABLE = False
    ClientOptions = None  # type: ignore[assignment,misc]
    App = None  # type: ignore[assignment,misc]
    ActivityContext = None  # type: ignore[assignment,misc]
    MessageActivity = None  # type: ignore[assignment,misc]
    ConversationReference = None  # type: ignore[assignment,misc]
    TypingActivityInput = None  # type: ignore[assignment,misc]
    AdaptiveCardInvokeActivity = None  # type: ignore[assignment,misc]
    FileConsentInvokeActivity = None  # type: ignore[assignment,misc]
    FileUploadInfo = None  # type: ignore[assignment,misc]
    AdaptiveCardActionCardResponse = None  # type: ignore[assignment,misc]
    AdaptiveCardActionMessageResponse = None  # type: ignore[assignment,misc]
    AdaptiveCardInvokeResponse = None  # type: ignore[assignment,misc,union-attr]
    InvokeResponse = None  # type: ignore[assignment,misc]
    HttpMethod = str  # type: ignore[assignment,misc]
    HttpRequest = None  # type: ignore[assignment,misc]
    HttpResponse = None  # type: ignore[assignment,misc]
    HttpRouteHandler = None  # type: ignore[assignment,misc]
    AdaptiveCard = None  # type: ignore[assignment,misc]
    ExecuteAction = None  # type: ignore[assignment,misc]
    TextBlock = None  # type: ignore[assignment,misc]

from gateway.config import Platform, PlatformConfig
from gateway.platforms.helpers import MessageDeduplicator
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    SUPPORTED_DOCUMENT_TYPES,
    cache_audio_from_url,
    cache_document_from_bytes,
    cache_image_from_url,
    cache_video_from_url,
)

logger = logging.getLogger(__name__)

_DEFAULT_PORT = 3978
_WEBHOOK_PATH = "/api/messages"


def _parse_bool(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


class _StaticAccessTokenProvider:
    """Minimal token-provider shim so outbound Graph delivery can reuse the shared client."""

    def __init__(self, access_token: str):
        self._access_token = str(access_token or "").strip()

    async def get_access_token(self, *, force_refresh: bool = False) -> str:
        del force_refresh
        if not self._access_token:
            raise ValueError("TEAMS_GRAPH_ACCESS_TOKEN is required for graph delivery mode.")
        return self._access_token

    def clear_cache(self) -> None:
        return None


class TeamsSummaryWriter:
    """Pipeline-facing Teams outbound delivery surface.

    This stays inside the existing Teams platform plugin so the meeting-pipeline
    PR can reuse one Teams integration surface instead of introducing a second
    adapter elsewhere in the gateway core.
    """

    def __init__(
        self,
        platform_config: PlatformConfig | None = None,
        *,
        graph_client: Any | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._platform_config = platform_config
        self._graph_client = graph_client
        self._transport = transport

    async def write_summary(
        self,
        payload: Any,
        config: dict[str, Any] | None,
        existing_record: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        merged = self._resolve_delivery_config(config)
        if existing_record and not _parse_bool(merged.get("force_resend"), default=False):
            return dict(existing_record)

        mode = str(merged.get("delivery_mode") or merged.get("mode") or "").strip().lower()
        if not mode:
            if merged.get("incoming_webhook_url"):
                mode = "incoming_webhook"
            elif merged.get("chat_id") or (
                merged.get("team_id") and merged.get("channel_id")
            ):
                mode = "graph"
        if mode == "incoming_webhook":
            return await self._write_summary_via_incoming_webhook(payload, merged)
        if mode == "graph":
            return await self._write_summary_via_graph(payload, merged)
        raise ValueError(
            "Teams delivery_mode must be 'incoming_webhook' or 'graph'."
        )

    def _resolve_delivery_config(self, config: dict[str, Any] | None) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        platform_cfg = self._platform_config
        if platform_cfg is not None:
            merged.update(dict(platform_cfg.extra or {}))
            if platform_cfg.token and "access_token" not in merged:
                merged["access_token"] = platform_cfg.token
            if platform_cfg.home_channel:
                merged.setdefault("channel_id", platform_cfg.home_channel.chat_id)
        merged.update(dict(config or {}))

        env_defaults = {
            "delivery_mode": os.getenv("TEAMS_DELIVERY_MODE", ""),
            "incoming_webhook_url": os.getenv("TEAMS_INCOMING_WEBHOOK_URL", ""),
            "access_token": os.getenv("TEAMS_GRAPH_ACCESS_TOKEN", ""),
            "team_id": os.getenv("TEAMS_TEAM_ID", ""),
            "channel_id": os.getenv("TEAMS_CHANNEL_ID", ""),
            "chat_id": os.getenv("TEAMS_CHAT_ID", ""),
        }
        for key, value in env_defaults.items():
            if value and not merged.get(key):
                merged[key] = value
        return merged

    async def _write_summary_via_incoming_webhook(
        self,
        payload: Any,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        # Lazy import — see module-level note. The teams plugin loads on
        # every CLI invocation as a side effect of plugin discovery, but
        # 99% of those processes never reach this method.
        import httpx
        webhook_url = str(config.get("incoming_webhook_url") or "").strip()
        if not webhook_url:
            raise ValueError("TEAMS_INCOMING_WEBHOOK_URL is required for incoming_webhook mode.")
        body = {"text": self._render_summary_markdown(payload)}
        async with httpx.AsyncClient(timeout=20.0, transport=self._transport) as client:
            response = await client.post(webhook_url, json=body)
            response.raise_for_status()
        return {
            "delivery_mode": "incoming_webhook",
            "webhook_url": webhook_url,
            "status_code": response.status_code,
            "delivered": True,
        }

    async def _write_summary_via_graph(
        self,
        payload: Any,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        graph_client = self._build_graph_client(config)
        chat_id = str(config.get("chat_id") or "").strip()
        if chat_id:
            path = f"/chats/{quote(chat_id, safe='')}/messages"
            response = await graph_client.post_json(
                path,
                json_body={"body": {"contentType": "html", "content": self._render_summary_html(payload)}},
            )
            return {
                "delivery_mode": "graph",
                "target_type": "chat",
                "chat_id": chat_id,
                "message_id": (response or {}).get("id"),
                "web_url": (response or {}).get("webUrl"),
            }

        team_id = str(config.get("team_id") or "").strip()
        channel_id = str(config.get("channel_id") or "").strip()
        if not team_id or not channel_id:
            raise ValueError(
                "Graph delivery mode requires chat_id, or both team_id and channel_id."
            )
        path = (
            f"/teams/{quote(team_id, safe='')}/channels/"
            f"{quote(channel_id, safe='')}/messages"
        )
        response = await graph_client.post_json(
            path,
            json_body={"body": {"contentType": "html", "content": self._render_summary_html(payload)}},
        )
        return {
            "delivery_mode": "graph",
            "target_type": "channel",
            "team_id": team_id,
            "channel_id": channel_id,
            "message_id": (response or {}).get("id"),
            "web_url": (response or {}).get("webUrl"),
        }

    def _build_graph_client(self, config: dict[str, Any]) -> Any:
        if self._graph_client is not None:
            return self._graph_client

        from tools.microsoft_graph_auth import MicrosoftGraphTokenProvider
        from tools.microsoft_graph_client import MicrosoftGraphClient

        access_token = str(config.get("access_token") or "").strip()
        if access_token:
            return MicrosoftGraphClient(
                _StaticAccessTokenProvider(access_token),
                transport=self._transport,
            )
        return MicrosoftGraphClient(
            MicrosoftGraphTokenProvider.from_env(),
            transport=self._transport,
        )

    def _render_summary_markdown(self, payload: Any) -> str:
        lines = [
            f"**{self._title(payload)}**",
            "",
            f"Summary: {self._text(getattr(payload, 'summary', None), 'No summary available.')}",
            "",
            "Key decisions:",
            *self._bullet_lines(getattr(payload, "key_decisions", None)),
            "",
            "Action items:",
            *self._bullet_lines(getattr(payload, "action_items", None)),
            "",
            "Risks:",
            *self._bullet_lines(getattr(payload, "risks", None)),
        ]
        return "\n".join(lines)

    def _render_summary_html(self, payload: Any) -> str:
        sections = [
            ("Summary", [self._text(getattr(payload, "summary", None), "No summary available.")]),
            ("Key decisions", list(getattr(payload, "key_decisions", None) or [])),
            ("Action items", list(getattr(payload, "action_items", None) or [])),
            ("Risks", list(getattr(payload, "risks", None) or [])),
        ]
        blocks = [f"<h2>{html.escape(self._title(payload))}</h2>"]
        for heading, items in sections:
            blocks.append(f"<h3>{html.escape(heading)}</h3>")
            if len(items) == 1 and heading == "Summary":
                blocks.append(f"<p>{html.escape(str(items[0]))}</p>")
                continue
            if items:
                rendered = "".join(f"<li>{html.escape(str(item))}</li>" for item in items if str(item).strip())
                blocks.append(rendered and f"<ul>{rendered}</ul>" or "<p>None</p>")
            else:
                blocks.append("<p>None</p>")
        return "".join(blocks)

    @staticmethod
    def _title(payload: Any) -> str:
        title = getattr(payload, "title", None)
        if title:
            return str(title)
        meeting_ref = getattr(payload, "meeting_ref", None)
        meeting_id = getattr(meeting_ref, "meeting_id", None) if meeting_ref else None
        return f"Meeting {meeting_id or 'summary'}"

    @staticmethod
    def _text(value: Any, default: str) -> str:
        text = str(value or "").strip()
        return text or default

    @classmethod
    def _bullet_lines(cls, values: Any) -> list[str]:
        items = [str(item).strip() for item in (values or []) if str(item).strip()]
        return [f"- {item}" for item in items] or ["- None"]


class _AiohttpBridgeAdapter:
    """HttpServerAdapter that bridges the Teams SDK into an aiohttp server.

    Without a custom adapter, ``App()`` unconditionally imports fastapi/uvicorn
    and allocates a ``FastAPI()`` instance.  This bridge captures the SDK's
    route registrations and wires them into our own aiohttp ``Application``.
    """

    def __init__(self, aiohttp_app: "web.Application"):
        self._aiohttp_app = aiohttp_app

    def register_route(self, method: "HttpMethod", path: str, handler: "HttpRouteHandler") -> None:
        """Register an SDK route handler as an aiohttp route."""

        async def _aiohttp_handler(request: "web.Request") -> "web.Response":
            body = await request.json()
            headers = dict(request.headers)
            result: "HttpResponse" = await handler(HttpRequest(body=body, headers=headers))
            status = result.get("status", 200)
            resp_body = result.get("body")
            if resp_body is not None:
                return web.Response(
                    status=status,
                    body=json.dumps(resp_body),
                    content_type="application/json",
                )
            return web.Response(status=status)

        self._aiohttp_app.router.add_route(method, path, _aiohttp_handler)

    def serve_static(self, path: str, directory: str) -> None:
        pass

    async def start(self, port: int) -> None:
        raise NotImplementedError("aiohttp server is managed by the adapter")

    async def stop(self) -> None:
        pass


def check_requirements() -> bool:
    """Return True when all Teams dependencies and credentials are present.

    Lazy-installs microsoft-teams-apps via ``tools.lazy_deps.ensure("platform.teams")``
    on first call if not present, following the same pattern as check_slack_requirements().
    """
    global TEAMS_SDK_AVAILABLE, AIOHTTP_AVAILABLE, web
    global App, ActivityContext, ClientOptions, MessageActivity, ConversationReference
    global TypingActivityInput, AdaptiveCardInvokeActivity
    global FileConsentInvokeActivity, FileUploadInfo
    global AdaptiveCardActionCardResponse, AdaptiveCardActionMessageResponse
    global InvokeResponse, AdaptiveCardInvokeResponse
    global HttpMethod, HttpRequest, HttpResponse, HttpRouteHandler
    global AdaptiveCard, ExecuteAction, TextBlock

    if TEAMS_SDK_AVAILABLE and AIOHTTP_AVAILABLE:
        return True
    try:
        from tools.lazy_deps import ensure as _lazy_ensure
        _lazy_ensure("platform.teams", prompt=False)
    except Exception:
        return False
    try:
        from aiohttp import web as _web
        from microsoft_teams.apps import App as _App, ActivityContext as _ActivityContext
        from microsoft_teams.common.http.client import ClientOptions as _ClientOptions
        from microsoft_teams.api import MessageActivity as _MessageActivity, ConversationReference as _ConversationReference
        from microsoft_teams.api.activities.typing import TypingActivityInput as _TypingActivityInput
        from microsoft_teams.api.activities.invoke.adaptive_card import AdaptiveCardInvokeActivity as _AdaptiveCardInvokeActivity
        from microsoft_teams.api.activities.invoke.file_consent import FileConsentInvokeActivity as _FileConsentInvokeActivity
        from microsoft_teams.api.models import FileUploadInfo as _FileUploadInfo
        from microsoft_teams.api.models.adaptive_card import (
            AdaptiveCardActionCardResponse as _AdaptiveCardActionCardResponse,
            AdaptiveCardActionMessageResponse as _AdaptiveCardActionMessageResponse,
        )
        from microsoft_teams.api.models.invoke_response import InvokeResponse as _InvokeResponse, AdaptiveCardInvokeResponse as _AdaptiveCardInvokeResponse
        from microsoft_teams.apps.http.adapter import (
            HttpMethod as _HttpMethod,
            HttpRequest as _HttpRequest,
            HttpResponse as _HttpResponse,
            HttpRouteHandler as _HttpRouteHandler,
        )
        from microsoft_teams.cards import AdaptiveCard as _AdaptiveCard, ExecuteAction as _ExecuteAction, TextBlock as _TextBlock
    except ImportError:
        return False
    web = _web
    App, ActivityContext, ClientOptions = _App, _ActivityContext, _ClientOptions
    MessageActivity, ConversationReference = _MessageActivity, _ConversationReference
    TypingActivityInput = _TypingActivityInput
    AdaptiveCardInvokeActivity = _AdaptiveCardInvokeActivity
    FileConsentInvokeActivity = _FileConsentInvokeActivity
    FileUploadInfo = _FileUploadInfo
    AdaptiveCardActionCardResponse = _AdaptiveCardActionCardResponse
    AdaptiveCardActionMessageResponse = _AdaptiveCardActionMessageResponse
    InvokeResponse, AdaptiveCardInvokeResponse = _InvokeResponse, _AdaptiveCardInvokeResponse
    HttpMethod, HttpRequest = _HttpMethod, _HttpRequest
    HttpResponse, HttpRouteHandler = _HttpResponse, _HttpRouteHandler
    AdaptiveCard, ExecuteAction, TextBlock = _AdaptiveCard, _ExecuteAction, _TextBlock
    AIOHTTP_AVAILABLE = True
    TEAMS_SDK_AVAILABLE = True
    return True


def validate_config(config) -> bool:
    """Return True when the config has the minimum required credentials."""
    extra = getattr(config, "extra", {}) or {}
    client_id = os.getenv("TEAMS_CLIENT_ID") or extra.get("client_id", "")
    client_secret = os.getenv("TEAMS_CLIENT_SECRET") or extra.get("client_secret", "")
    tenant_id = os.getenv("TEAMS_TENANT_ID") or extra.get("tenant_id", "")
    return bool(client_id and client_secret and tenant_id)


def is_connected(config) -> bool:
    """Check whether Teams is configured (env or config.yaml)."""
    return validate_config(config)


def _env_enablement() -> dict | None:
    """Seed ``PlatformConfig.extra`` from env vars during gateway config load.

    Called by the platform registry's env-enablement hook BEFORE adapter
    construction, so ``gateway status`` and ``get_connected_platforms()``
    reflect env-only configuration without instantiating the Teams SDK.
    Returns ``None`` when Teams isn't minimally configured.

    The special ``home_channel`` key in the returned dict becomes a proper
    ``HomeChannel`` dataclass on the ``PlatformConfig`` via the core hook.
    """
    client_id = os.getenv("TEAMS_CLIENT_ID", "").strip()
    client_secret = os.getenv("TEAMS_CLIENT_SECRET", "").strip()
    tenant_id = os.getenv("TEAMS_TENANT_ID", "").strip()
    if not (client_id and client_secret and tenant_id):
        return None
    seed: dict = {
        "client_id": client_id,
        "client_secret": client_secret,
        "tenant_id": tenant_id,
    }
    port = os.getenv("TEAMS_PORT", "").strip()
    if port:
        try:
            seed["port"] = int(port)
        except ValueError:
            pass
    service_url = os.getenv("TEAMS_SERVICE_URL", "").strip()
    if service_url:
        seed["service_url"] = service_url
    home = os.getenv("TEAMS_HOME_CHANNEL", "").strip()
    if home:
        seed["home_channel"] = {
            "chat_id": home,
            "name": os.getenv("TEAMS_HOME_CHANNEL_NAME", "Home"),
        }
    return seed


# Bot Framework default service URL for the global Teams endpoint.  Some
# regional/government tenants need a different host (e.g.
# ``https://smba.infra.gov.teams.microsoft.us/``) which can be supplied via
# ``TEAMS_SERVICE_URL`` or ``extra['service_url']``.
_DEFAULT_TEAMS_SERVICE_URL = "https://smba.trafficmanager.net/teams/"

# Allowlist of Bot Framework service hosts that may receive a freshly
# minted bearer token.  Operator-supplied URLs are matched against this
# allowlist to block SSRF / token-exfiltration via a tampered env var.
_ALLOWED_TEAMS_SERVICE_HOSTS = frozenset({
    "smba.trafficmanager.net",
    "smba.infra.gov.teams.microsoft.us",
})

# Conservative pattern for Bot Framework conversation IDs.  Real values
# combine digits, colons, hyphens, dots, '@', and the ``thread.skype`` /
# ``thread.tacv2`` suffixes; reject anything outside this set so a hostile
# value cannot path-traverse out of ``/v3/conversations/<id>/activities``.
import re as _re_teams
_TEAMS_CONV_ID_RE = _re_teams.compile(r"^[A-Za-z0-9:@\-_.]+$")


def _validate_teams_service_url(raw: str) -> Optional[str]:
    """Return a normalized service URL or ``None`` if it is not allowed.

    Requires ``https://`` and a host in ``_ALLOWED_TEAMS_SERVICE_HOSTS``.
    The trailing slash is added if absent so callers can append
    ``v3/conversations/...`` without double slashes.
    """
    if not raw:
        return None
    try:
        from urllib.parse import urlparse

        parsed = urlparse(raw)
    except Exception:
        return None
    if parsed.scheme != "https":
        return None
    if parsed.hostname not in _ALLOWED_TEAMS_SERVICE_HOSTS:
        return None
    normalized = raw if raw.endswith("/") else raw + "/"
    return normalized


async def _standalone_send(
    pconfig,
    chat_id: str,
    message: str,
    *,
    thread_id: Optional[str] = None,
    media_files: Optional[list] = None,
    force_document: bool = False,
) -> Dict[str, Any]:
    """Acquire a Bot Framework bearer token and POST a single message activity.

    Used by ``tools/send_message_tool._send_via_adapter`` when the gateway
    runner is not in this process (e.g. ``hermes cron`` running as a
    separate process from ``hermes gateway``).  Without this hook,
    ``deliver=teams`` cron jobs fail with ``No live adapter for platform``.

    Configuration: requires ``TEAMS_CLIENT_ID``, ``TEAMS_CLIENT_SECRET``,
    ``TEAMS_TENANT_ID``, ``TEAMS_HOME_CHANNEL`` (the conversation ID), and
    optionally ``TEAMS_SERVICE_URL`` (Bot Framework service host; must be
    a known Bot Framework endpoint, see ``_ALLOWED_TEAMS_SERVICE_HOSTS``).

    Security: ``service_url`` is validated against an allowlist of known
    Bot Framework hosts to block SSRF / token-exfiltration via a tampered
    env var.  ``chat_id`` is validated to match the documented Bot
    Framework ID character set so it cannot escape the URL path.

    ``media_files`` and ``force_document`` are accepted for signature
    parity but not implemented for the standalone path; messages with
    attachments will send as text-only.  The live adapter handles
    attachments via the SDK.
    """
    extra = getattr(pconfig, "extra", {}) or {}
    client_id = os.getenv("TEAMS_CLIENT_ID") or extra.get("client_id", "")
    client_secret = os.getenv("TEAMS_CLIENT_SECRET") or extra.get("client_secret", "")
    tenant_id = os.getenv("TEAMS_TENANT_ID") or extra.get("tenant_id", "")
    if not (client_id and client_secret and tenant_id):
        return {"error": "Teams standalone send: TEAMS_CLIENT_ID, TEAMS_CLIENT_SECRET, and TEAMS_TENANT_ID are all required"}

    raw_service_url = (
        os.getenv("TEAMS_SERVICE_URL")
        or extra.get("service_url", "")
        or _DEFAULT_TEAMS_SERVICE_URL
    )
    service_url = _validate_teams_service_url(raw_service_url)
    if service_url is None:
        return {"error": (
            f"Teams standalone send: TEAMS_SERVICE_URL host is not on the "
            f"Bot Framework allowlist; expected one of "
            f"{sorted(_ALLOWED_TEAMS_SERVICE_HOSTS)}"
        )}

    # Bot Framework conversation IDs are restricted to a known character
    # set; anything else means a tampered chat_id trying to break out of
    # the URL path.
    if not chat_id:
        return {"error": "Teams standalone send: chat_id (conversation ID) is required"}
    if not _TEAMS_CONV_ID_RE.match(chat_id):
        return {"error": "Teams standalone send: chat_id contains characters outside the Bot Framework conversation ID set"}
    if not _TEAMS_CONV_ID_RE.match(tenant_id):
        return {"error": "Teams standalone send: TEAMS_TENANT_ID contains characters outside the expected set"}

    token_url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
    activities_url = f"{service_url}v3/conversations/{chat_id}/activities"

    if not AIOHTTP_AVAILABLE:
        return {"error": "Teams standalone send: aiohttp not installed"}

    try:
        import aiohttp as _aiohttp

        # Per-request timeouts so a slow STS endpoint cannot starve the
        # subsequent activity POST of its budget.
        per_request_timeout = _aiohttp.ClientTimeout(total=15.0)
        async with _aiohttp.ClientSession() as session:
            async with session.post(
                token_url,
                data={
                    "grant_type": "client_credentials",
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "scope": "https://api.botframework.com/.default",
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=per_request_timeout,
            ) as token_resp:
                if token_resp.status >= 400:
                    body = await token_resp.text()
                    return {"error": f"Teams standalone send: token request failed ({token_resp.status}): {body[:300]}"}
                token_payload = await token_resp.json()
            access_token = token_payload.get("access_token")
            if not access_token:
                return {"error": "Teams standalone send: token response missing access_token"}

            activity = {
                "type": "message",
                "text": message,
                "textFormat": "markdown",
            }
            async with session.post(
                activities_url,
                json=activity,
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json",
                },
                timeout=per_request_timeout,
            ) as send_resp:
                if send_resp.status >= 400:
                    body = await send_resp.text()
                    return {"error": f"Teams standalone send: activity post failed ({send_resp.status}): {body[:300]}"}
                send_payload = await send_resp.json()
        return {
            "success": True,
            "message_id": send_payload.get("id"),
        }
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.debug("Teams standalone send raised", exc_info=True)
        return {"error": f"Teams standalone send failed: {e}"}


# Keep the old name as an alias so existing test imports don't break.
check_teams_requirements = check_requirements


_HOSTED_CONTENT_RE = re.compile(
    r"/hostedContents/([^/?#]+)(?:/\$value)?(?:[?#]|$)",
    re.IGNORECASE,
)


# Bot Framework hosts that require an Authorization: Bearer *** on GETs.
# When an inbound activity references a content_url under one of these hosts
# (typical shape: https://smba.trafficmanager.net/<region>/<tenant>/v3/
# attachments/<id>/views/original) the shared cache_*_from_url helpers will
# 401 because they don't carry per-platform auth. We fetch the bytes here
# instead, using the SDK-managed bot token, then route to cache_*_from_bytes.
_BF_ATTACHMENT_HOSTS = {"smba.trafficmanager.net"}


def _url_fingerprint(url: Optional[str], *, path_chars: int = 24) -> str:
    """Return a log-safe fingerprint of ``url`` — host + truncated path + sha8.

    Teams/SharePoint download URLs frequently embed bearer or ``tempauth``
    material in the query string or path (and sometimes deeper-path tokens
    like ``/_api/v2.0/drives/<id>/items/<id>/content?tempauth=...``). Logging
    the raw URL leaks file-access credentials into gateway logs, where a
    log-aggregator subscriber or anyone reading ``gateway.log`` can replay
    them until they expire (minutes-to-hours window).

    The fingerprint preserves enough signal to debug routing/dispatch
    (which host, roughly which path) while stripping query strings, fragments,
    userinfo, and most of the path. The trailing ``#<sha8>`` lets two log
    lines referencing the *same* URL be correlated without exposing the URL
    itself. Output for a malformed/empty URL is the literal ``"<no-url>"``
    rather than ``None`` so format-string callers can use ``%s`` safely.
    """
    if not url or not isinstance(url, str):
        return "<no-url>"
    try:
        from urllib.parse import urlparse

        parsed = urlparse(url)
        host = (parsed.hostname or "?").lower()
        path = parsed.path or ""
        if len(path) > path_chars:
            path = path[:path_chars] + "..."
        # SHA-256 over the *full* original URL so the same URL always
        # produces the same 8-char tag — useful for grepping log
        # correlations without exposing token material.
        digest = hashlib.sha256(url.encode("utf-8", errors="replace")).hexdigest()[:8]
        return f"{host}{path}#{digest}"
    except Exception:  # noqa: BLE001 — never raise from a logging helper
        return "<unparseable-url>"


def _is_bf_attachment_url(url: Optional[str]) -> bool:
    """True if ``url`` points at a Bot Framework attachment endpoint that
    needs a Bearer token."""
    if not url or not isinstance(url, str):
        return False
    try:
        from urllib.parse import urlparse

        host = (urlparse(url).hostname or "").lower()
    except Exception:
        return False
    return host in _BF_ATTACHMENT_HOSTS


# Magic-byte prefixes used when a Teams attachment arrives with a wildcard
# or missing MIME subtype (e.g. ``image/*`` from inline-pasted images). The
# subtype must never be passed straight through into a cache filename or it
# leaks as a literal ``.*`` extension and breaks downstream tooling that
# resolves files by extension.
_IMAGE_DEFAULT_EXT = ".jpg"
_AUDIO_DEFAULT_EXT = ".ogg"
_VIDEO_DEFAULT_EXT = ".mp4"


def _sniff_image_ext(data: bytes) -> Optional[str]:
    if not data:
        return None
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if data.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        return ".gif"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    if data.startswith(b"BM"):
        return ".bmp"
    return None


def _sniff_audio_ext(data: bytes) -> Optional[str]:
    if not data:
        return None
    if data.startswith(b"OggS"):
        return ".ogg"
    if data.startswith(b"ID3") or data.startswith(b"\xff\xfb") or data.startswith(b"\xff\xf3") or data.startswith(b"\xff\xf2"):
        return ".mp3"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return ".wav"
    if data.startswith(b"fLaC"):
        return ".flac"
    # ISO BMFF (m4a/aac in mp4 container) — ftyp box at offset 4
    if len(data) >= 12 and data[4:8] == b"ftyp":
        brand = data[8:12]
        if brand in (b"M4A ", b"M4B ", b"mp42", b"isom"):
            return ".m4a"
    return None


def _sniff_video_ext(data: bytes) -> Optional[str]:
    if not data:
        return None
    # ISO BMFF — common brands for mp4/mov
    if len(data) >= 12 and data[4:8] == b"ftyp":
        brand = data[8:12]
        if brand.startswith(b"qt"):
            return ".mov"
        # isom / mp42 / iso2 / dash / etc. all map to .mp4 for our purposes
        return ".mp4"
    # WebM / Matroska EBML header
    if data.startswith(b"\x1aE\xdf\xa3"):
        return ".webm"
    # AVI
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"AVI ":
        return ".avi"
    return None


def _resolve_media_ext(subtype: str, data: bytes, kind: str) -> str:
    """Resolve a safe file extension for a Teams media attachment.

    Teams sometimes posts attachments with ``content_type="image/*"`` (literal
    asterisk wildcard) — particularly inline-pasted images. Splitting that on
    ``"/"`` would yield ``"*"`` and produce a cache filename with a literal
    ``.*`` extension, breaking every downstream consumer that opens files by
    extension. Instead, when the subtype is missing, ``"*"``, or otherwise
    meaningless, sniff the bytes via magic numbers and fall back to a sane
    per-kind default.

    Args:
        subtype: The MIME subtype (already lowercased; e.g. ``"png"``,
            ``"jpeg"``, ``"*"``, or ``""``).
        data: The fetched bytes (may be empty if the fetch failed).
        kind: ``"image"``, ``"audio"``, or ``"video"``.

    Returns:
        Extension including the leading dot, e.g. ``".png"``. Never returns
        ``".*"`` or an empty string.
    """
    sub = (subtype or "").strip().lower()
    # Strip any codec params left in the subtype (defence against caller bugs)
    sub = sub.split(";", 1)[0].strip()

    # jpeg → jpg normalisation, regardless of source
    if sub == "jpeg":
        return ".jpg"

    # Explicit, non-wildcard subtype: trust it. Codec/container variants are
    # too broad to whitelist and the caller already validated the kind via
    # the ``image/`` / ``audio/`` / ``video/`` prefix.
    if sub and sub != "*":
        return "." + sub

    # Wildcard or empty — sniff bytes.
    if kind == "image":
        sniffed = _sniff_image_ext(data)
        return sniffed or _IMAGE_DEFAULT_EXT
    if kind == "audio":
        sniffed = _sniff_audio_ext(data)
        return sniffed or _AUDIO_DEFAULT_EXT
    if kind == "video":
        sniffed = _sniff_video_ext(data)
        return sniffed or _VIDEO_DEFAULT_EXT
    return _IMAGE_DEFAULT_EXT


def _parse_hosted_content_id(url: str) -> Optional[str]:
    """Extract the hostedContents/{id} fragment from a Teams URL.

    Returns the opaque id that Graph's
    /teams/{team}/channels/{channel}/messages/{msg}/hostedContents/{id}
    endpoint expects, or None when the URL isn't shaped that way
    (e.g. a SharePoint file-upload URL — those have no Graph fallback).
    """
    if not url:
        return None
    match = _HOSTED_CONTENT_RE.search(url)
    return match.group(1) if match else None


class TeamsAdapter(BasePlatformAdapter):
    """Microsoft Teams adapter using the microsoft-teams-apps SDK."""

    MAX_MESSAGE_LENGTH = 28000  # Teams text message limit (~28 KB)

    # Bound _pending_uploads memory: drop oldest beyond this many entries
    # and sweep entries older than the TTL on every DM send. Without
    # this, a long-running gateway holding many users' un-consented DM
    # file sends grows unboundedly in RAM (especially when users send
    # videos but never click Allow / Decline).
    _PENDING_UPLOAD_MAX = 64
    _PENDING_UPLOAD_TTL_SECONDS = 3600  # 1 hour

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform("teams"))
        extra = config.extra or {}
        self._client_id = extra.get("client_id") or os.getenv("TEAMS_CLIENT_ID", "")
        self._client_secret = extra.get("client_secret") or os.getenv("TEAMS_CLIENT_SECRET", "")
        self._tenant_id = extra.get("tenant_id") or os.getenv("TEAMS_TENANT_ID", "")
        self._port = int(extra.get("port") or os.getenv("TEAMS_PORT", str(_DEFAULT_PORT)))
        self._app: Optional["App"] = None
        self._runner: Optional["web.AppRunner"] = None
        # Captured in connect() so tools/send_message_tool._send_teams can
        # bridge cross-loop calls back to the gateway loop the SDK App was
        # built on. Stays None until connect() succeeds; cleared on
        # disconnect(). See _send_teams' "Cross-loop bridge" docstring.
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._dedup = MessageDeduplicator(max_size=1000)
        # Maps chat_id → ConversationReference captured from incoming messages.
        # Used to send cards with the correct conversation type (personal/group/channel).
        self._conv_refs: Dict[str, Any] = {}

        # SharePoint target for outbound channel/group file uploads.
        # Defaults: site_id="" disables channel uploads; folder defaults to
        # "hermes" so DM-only deployments without SharePoint config still
        # construct cleanly (channel sends will return a clean error).
        self._sharepoint_site_id: str = (
            extra.get("sharepoint_site_id")
            or os.getenv("TEAMS_SHAREPOINT_SITE_ID", "")
        )
        self._sharepoint_folder: str = (
            extra.get("sharepoint_folder")
            or os.getenv("TEAMS_SHAREPOINT_FOLDER", "hermes")
        )
        # Files awaiting FileConsent acceptance from a DM user — the
        # fileConsent/invoke handler drains this dict; the DM send path
        # primes it. Keyed by the upload_id seeded into the FileConsent
        # acceptContext.
        # OrderedDict + size cap + TTL bound memory; see
        # _register_pending_upload / _evict_stale_pending_uploads.
        self._pending_uploads: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        # Lazy Graph client + token provider; built on first channel send
        # so DM-only deployments don't pay the msgraph-sdk import cost.
        self._graph: Any = None
        self._token_provider: Any = None

    async def connect(self) -> bool:
        if not TEAMS_SDK_AVAILABLE:
            self._set_fatal_error(
                "MISSING_SDK",
                "microsoft-teams-apps not installed. Run: pip install microsoft-teams-apps",
                retryable=False,
            )
            return False

        if not AIOHTTP_AVAILABLE:
            self._set_fatal_error(
                "MISSING_SDK",
                "aiohttp not installed. Run: pip install aiohttp",
                retryable=False,
            )
            return False

        if not self._client_id or not self._client_secret or not self._tenant_id:
            self._set_fatal_error(
                "MISSING_CREDENTIALS",
                "TEAMS_CLIENT_ID, TEAMS_CLIENT_SECRET, and TEAMS_TENANT_ID are all required",
                retryable=False,
            )
            return False

        try:
            # Set up aiohttp app first — the bridge adapter wires SDK routes into it
            aiohttp_app = web.Application()
            aiohttp_app.router.add_get("/health", lambda _: web.Response(text="ok"))

            self._app = App(
                client_id=self._client_id,
                client_secret=self._client_secret,
                tenant_id=self._tenant_id,
                http_server_adapter=_AiohttpBridgeAdapter(aiohttp_app),
                client=ClientOptions(headers={"User-Agent": "Hermes"}),
            )

            # Register message handler before initialize()
            @self._app.on_message
            async def _handle_message(ctx: ActivityContext[MessageActivity]):
                await self._on_message(ctx)

            @self._app.on_card_action
            async def _handle_card_action(
                ctx: ActivityContext[AdaptiveCardInvokeActivity],
            ) -> InvokeResponse[AdaptiveCardActionMessageResponse]:
                return await self._on_card_action(ctx)

            @self._app.on_file_consent
            async def _handle_file_consent(
                ctx: ActivityContext[FileConsentInvokeActivity],
            ) -> Optional[InvokeResponse[None]]:
                return await self._handle_file_consent_invoke(ctx)

            # initialize() calls register_route() on the bridge, which adds
            # POST /api/messages to aiohttp_app automatically
            await self._app.initialize()

            self._runner = web.AppRunner(aiohttp_app)
            await self._runner.setup()
            site = web.TCPSite(self._runner, "0.0.0.0", self._port)
            await site.start()

            self._running = True
            # Capture the gateway loop so cross-loop callers (e.g.
            # tools/send_message_tool._send_teams invoked from the agent's
            # worker loop) can hop here via run_coroutine_threadsafe. Done
            # last — only after _app + aiohttp are fully wired so a partial
            # init never publishes a half-baked loop.
            self._loop = asyncio.get_running_loop()
            self._mark_connected()
            logger.info(
                "[teams] Webhook server listening on 0.0.0.0:%d%s",
                self._port,
                _WEBHOOK_PATH,
            )
            return True

        except Exception as e:
            self._set_fatal_error(
                "CONNECT_FAILED",
                f"Teams connection failed: {e}",
                retryable=True,
            )
            logger.error("[teams] Failed to connect: %s", e)
            return False

    async def disconnect(self) -> None:
        self._running = False
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
        self._app = None
        # Clear the captured loop reference so a stale send_message tool
        # call doesn't try to hop to a loop that's about to close.
        self._loop = None
        self._mark_disconnected()
        logger.info("[teams] Disconnected")

    async def _fetch_bf_attachment_bytes(self, url: str) -> Optional[bytes]:
        """GET a Bot Framework attachment URL with an SDK-minted bearer token.

        BF attachment endpoints (``smba.trafficmanager.net/.../v3/attachments/
        .../views/original``) reject anonymous GETs with 401. The SDK already
        manages an MSAL-cached bot token via ``self._app._get_bot_token``;
        we stringify that token (``JsonWebToken.__str__`` returns the raw JWT)
        and attach it as ``Authorization: Bearer <jwt>``.

        Returns the response body on 2xx, or ``None`` on any failure (no
        token, non-200, network error). Logs at WARNING for diagnosable
        failures so the same ``[teams][attach]`` log breadcrumbs the rest
        of the attachment loop emits remain greppable.
        """
        if not url:
            return None
        if self._app is None:
            logger.warning("[teams][attach] BF fetch skipped — no SDK app available")
            return None
        try:
            token = await self._app._get_bot_token()
        except Exception as exc:  # noqa: BLE001
            logger.warning("[teams][attach] BF token fetch failed: %s", exc)
            return None
        if token is None:
            logger.warning("[teams][attach] BF token fetch returned None")
            return None
        bearer = str(token)
        headers = {"Authorization": f"Bearer {bearer}"}

        try:
            import aiohttp as _aiohttp

            timeout = _aiohttp.ClientTimeout(total=60)
            async with _aiohttp.ClientSession(timeout=timeout) as sess:
                async with sess.get(url, headers=headers) as resp:
                    if resp.status != 200:
                        logger.warning(
                            "[teams][attach] BF GET %s -> status=%s",
                            _url_fingerprint(url), resp.status,
                        )
                        return None
                    return await resp.read()
        except Exception as exc:  # noqa: BLE001
            logger.warning("[teams][attach] BF GET %s raised: %s", _url_fingerprint(url), exc)
            return None

    async def _on_message(self, ctx: ActivityContext[MessageActivity]) -> None:
        """Process an incoming Teams message and dispatch to the gateway."""
        activity = ctx.activity

        # Self-message filter
        bot_id = self._app.id if self._app else None
        if bot_id and getattr(activity.from_, "id", None) == bot_id:
            return

        # Deduplication
        msg_id = getattr(activity, "id", None)
        if msg_id and self._dedup.is_duplicate(msg_id):
            return

        # Cache the conversation reference for proactive sends (approval cards, etc.)
        conv_id = getattr(activity.conversation, "id", None)
        if conv_id:
            self._conv_refs[conv_id] = ctx.conversation_ref

        # Extract text — strip bot @mentions
        text = ""
        if hasattr(activity, "text") and activity.text:
            text = activity.text
        # Strip <at>BotName</at> HTML tags that Teams prepends for @mentions
        if "<at>" in text:
            import re
            text = re.sub(r"<at>[^<]*</at>\s*", "", text).strip()

        # Determine chat type from conversation
        conv = activity.conversation
        conv_type = getattr(conv, "conversation_type", None) or ""
        if conv_type == "personal":
            chat_type = "dm"
        elif conv_type == "groupChat":
            chat_type = "group"
        elif conv_type == "channel":
            chat_type = "channel"
        else:
            chat_type = "dm"

        # Build source
        from_account = activity.from_
        user_id = getattr(from_account, "aad_object_id", None) or getattr(from_account, "id", "")
        user_name = getattr(from_account, "name", None) or ""

        source = self.build_source(
            chat_id=conv.id,
            chat_name=getattr(conv, "name", None) or "",
            chat_type=chat_type,
            user_id=str(user_id),
            user_name=user_name,
            guild_id=getattr(conv, "tenant_id", None) or self._tenant_id,
        )

        # Handle inbound attachments — images, audio, video, and documents.
        #
        # Teams delivers attachments in two shapes:
        #   1. ``content_type: "image/<sub>"`` (or "audio/...", "video/...") with
        #      a Bot Framework URL in ``content_url``. These can be cached
        #      directly from the URL.
        #   2. ``content_type: "application/vnd.microsoft.teams.file.download.info"``
        #      — file uploads from the Teams web/desktop client. The real
        #      filename and download URL live in ``content`` (a SharePoint
        #      tempauth URL — Authorization header MUST be omitted on the GET
        #      or it returns 401). We fetch the bytes ourselves and route to
        #      the appropriate cache_* helper based on file extension.
        media_urls = []
        media_types = []
        msg_type_override: Optional[MessageType] = None  # set by first non-image/non-text attachment
        _IMAGE_EXTS = {"png", "jpg", "jpeg", "gif", "webp"}
        _AUDIO_EXTS = {"mp3", "ogg", "wav", "m4a", "aac", "flac"}
        _VIDEO_EXTS = {"mp4", "mov", "webm", "mkv", "avi"}
        _DOC_EXT_TO_MIME = {
            ext.lstrip("."): mime for ext, mime in SUPPORTED_DOCUMENT_TYPES.items()
        }

        attachments = getattr(activity, "attachments", None) or []
        logger.info("[teams][attach] received %d attachment(s)", len(attachments))

        # Context for Graph hostedContents fallback. Channel attachments
        # (not DM uploads) carry team_id + channel_id under channel_data;
        # without all three of (team, channel, activity_id) the fallback
        # cannot address the blob and we just take the direct-failure path.
        channel_data = getattr(activity, "channel_data", None) or {}
        graph_team_id: Optional[str] = None
        graph_channel_id: Optional[str] = None
        graph_activity_id: str = str(getattr(activity, "id", "") or "")
        if isinstance(channel_data, dict):
            team_block = channel_data.get("team") or {}
            channel_block = channel_data.get("channel") or {}
            graph_team_id = team_block.get("id")
            graph_channel_id = channel_block.get("id")

        for idx, att in enumerate(attachments):
            content_url = getattr(att, "content_url", None) or getattr(att, "contentUrl", None)
            content_type = (getattr(att, "content_type", None) or getattr(att, "contentType", None) or "").lower()
            att_name = getattr(att, "name", None)
            logger.info(
                "[teams][attach][%d] content_type=%r name=%r content_url=%s has_content=%s",
                idx, content_type, att_name, _url_fingerprint(content_url), getattr(att, "content", None) is not None,
            )

            try:
                # ── Shape 1: classic content_url with image/audio/video MIME ─
                if content_url and content_type.startswith("image/"):
                    if _is_bf_attachment_url(content_url):
                        logger.info("[teams][attach][%d] dispatch=image_bf_url", idx)
                        data = await self._fetch_bf_attachment_bytes(content_url)
                        cached = None
                        if data is not None:
                            from gateway.platforms.base import cache_image_from_bytes
                            sub = content_type.split("/", 1)[1].split(";")[0].strip()
                            ext = _resolve_media_ext(sub, data, "image")
                            try:
                                cached = cache_image_from_bytes(data, ext=ext)
                            except Exception as exc:  # noqa: BLE001
                                logger.exception(
                                    "[teams][attach][%d] cache_image_from_bytes failed: %s",
                                    idx, exc,
                                )
                                cached = None
                            logger.info("[teams][attach][%d] cache_image_from_bytes -> %r", idx, cached)
                        if cached is None:
                            cached = await self._try_graph_hosted_fallback(
                                idx=idx,
                                url=content_url,
                                team_id=graph_team_id,
                                channel_id=graph_channel_id,
                                activity_id=graph_activity_id,
                                kind="image",
                                ext=".jpg",
                                filename=att_name,
                            )
                        if cached:
                            media_urls.append(cached)
                            media_types.append(content_type)
                        continue
                    logger.info("[teams][attach][%d] dispatch=image_url", idx)
                    cached = await cache_image_from_url(content_url)
                    logger.info("[teams][attach][%d] cache_image_from_url -> %r", idx, cached)
                    if cached is None:
                        cached = await self._try_graph_hosted_fallback(
                            idx=idx,
                            url=content_url,
                            team_id=graph_team_id,
                            channel_id=graph_channel_id,
                            activity_id=graph_activity_id,
                            kind="image",
                            ext=".jpg",  # default — most hosted contents are jpegs
                            filename=att_name,
                        )
                    if cached:
                        media_urls.append(cached)
                        media_types.append(content_type)
                    continue
                if content_url and content_type.startswith("audio/"):
                    sub = content_type.split("/", 1)[1].split(";")[0].strip()
                    if _is_bf_attachment_url(content_url):
                        logger.info("[teams][attach][%d] dispatch=audio_bf_url subtype=%s", idx, sub or "*")
                        data = await self._fetch_bf_attachment_bytes(content_url)
                        cached = None
                        if data is not None:
                            from gateway.platforms.base import cache_audio_from_bytes
                            ext = _resolve_media_ext(sub, data, "audio")
                            try:
                                cached = cache_audio_from_bytes(data, ext=ext)
                            except Exception as exc:  # noqa: BLE001
                                logger.exception(
                                    "[teams][attach][%d] cache_audio_from_bytes failed: %s",
                                    idx, exc,
                                )
                                cached = None
                            logger.info("[teams][attach][%d] cache_audio_from_bytes -> %r (ext=%s)", idx, cached, ext)
                        if cached:
                            media_urls.append(cached)
                            media_types.append(content_type)
                            msg_type_override = msg_type_override or MessageType.VOICE
                        continue
                    ext = _resolve_media_ext(sub, b"", "audio")
                    logger.info("[teams][attach][%d] dispatch=audio_url ext=%s", idx, ext)
                    cached = await cache_audio_from_url(content_url, ext=ext)
                    logger.info("[teams][attach][%d] cache_audio_from_url -> %r", idx, cached)
                    if cached:
                        media_urls.append(cached)
                        media_types.append(content_type)
                        msg_type_override = msg_type_override or MessageType.VOICE
                    continue
                if content_url and content_type.startswith("video/"):
                    sub = content_type.split("/", 1)[1].split(";")[0].strip()
                    if _is_bf_attachment_url(content_url):
                        logger.info("[teams][attach][%d] dispatch=video_bf_url subtype=%s", idx, sub or "*")
                        data = await self._fetch_bf_attachment_bytes(content_url)
                        cached = None
                        if data is not None:
                            from gateway.platforms.base import cache_video_from_bytes
                            ext = _resolve_media_ext(sub, data, "video")
                            try:
                                cached = cache_video_from_bytes(data, ext=ext)
                            except Exception as exc:  # noqa: BLE001
                                logger.exception(
                                    "[teams][attach][%d] cache_video_from_bytes failed: %s",
                                    idx, exc,
                                )
                                cached = None
                            logger.info("[teams][attach][%d] cache_video_from_bytes -> %r (ext=%s)", idx, cached, ext)
                        if cached:
                            media_urls.append(cached)
                            media_types.append(content_type)
                            msg_type_override = msg_type_override or MessageType.VIDEO
                        continue
                    ext = _resolve_media_ext(sub, b"", "video")
                    logger.info("[teams][attach][%d] dispatch=video_url ext=%s", idx, ext)
                    cached = await cache_video_from_url(content_url, ext=ext)
                    logger.info("[teams][attach][%d] cache_video_from_url -> %r", idx, cached)
                    if cached:
                        media_urls.append(cached)
                        media_types.append(content_type)
                        msg_type_override = msg_type_override or MessageType.VIDEO
                    continue

                # ── Shape 2: file.download.info ─────────────────────────────
                if content_type == "application/vnd.microsoft.teams.file.download.info":
                    logger.info("[teams][attach][%d] dispatch=file.download.info", idx)
                    content_block = getattr(att, "content", None) or {}
                    # Bot Framework SDK delivers `content` as either a dict or
                    # a typed object — handle both.
                    def _field(obj, *names):
                        for n in names:
                            if isinstance(obj, dict):
                                v = obj.get(n)
                            else:
                                v = getattr(obj, n, None)
                            if v is not None:
                                return v
                        return None

                    file_type = str(_field(content_block, "file_type", "fileType") or "").lower().lstrip(".")
                    download_url = str(_field(content_block, "download_url", "downloadUrl") or "")
                    filename = str(getattr(att, "name", None) or "").strip() or f"attachment.{file_type or 'bin'}"
                    logger.info(
                        "[teams][attach][%d] file.download.info parsed: filename=%r file_type=%r download_url=%s",
                        idx, filename, file_type, _url_fingerprint(download_url),
                    )

                    if not download_url or not file_type:
                        logger.info(
                            "[teams][attach][%d] DROP missing fields (name=%r, file_type=%r, has_url=%s)",
                            idx, filename, file_type, bool(download_url),
                        )
                        continue

                    # SharePoint tempauth URLs reject the Authorization header — fetch raw.
                    import aiohttp as _aiohttp
                    timeout = _aiohttp.ClientTimeout(total=60)
                    async with _aiohttp.ClientSession(timeout=timeout) as sess:
                        async with sess.get(download_url) as resp:
                            logger.info(
                                "[teams][attach][%d] tempauth GET %s -> status=%s",
                                idx, filename, resp.status,
                            )
                            if resp.status != 200:
                                logger.warning(
                                    "[teams][attach][%d] tempauth GET %s returned %s — trying Graph fallback",
                                    idx, filename, resp.status,
                                )
                                data = await self._try_graph_hosted_bytes(
                                    url=download_url,
                                    team_id=graph_team_id,
                                    channel_id=graph_channel_id,
                                    activity_id=graph_activity_id,
                                )
                                if data is None:
                                    logger.warning(
                                        "[teams][attach][%d] Graph fallback also failed for %s",
                                        idx, filename,
                                    )
                                    continue
                            else:
                                data = await resp.read()
                    logger.info("[teams][attach][%d] tempauth body len=%d bytes", idx, len(data))

                    if file_type in _IMAGE_EXTS:
                        logger.info("[teams][attach][%d] dispatch=image_bytes ext=%s", idx, file_type)
                        # Treat as image — no _from_url helper needed since we have bytes.
                        from gateway.platforms.base import cache_image_from_bytes
                        cached = cache_image_from_bytes(data, ext="." + file_type)
                        logger.info("[teams][attach][%d] cache_image_from_bytes -> %r", idx, cached)
                        media_urls.append(cached)
                        media_types.append(f"image/{file_type if file_type != 'jpg' else 'jpeg'}")
                    elif file_type in _AUDIO_EXTS:
                        logger.info("[teams][attach][%d] dispatch=audio_bytes ext=%s", idx, file_type)
                        from gateway.platforms.base import cache_audio_from_bytes
                        cached = cache_audio_from_bytes(data, ext="." + file_type)
                        logger.info("[teams][attach][%d] cache_audio_from_bytes -> %r", idx, cached)
                        media_urls.append(cached)
                        media_types.append(f"audio/{file_type}")
                        msg_type_override = msg_type_override or MessageType.VOICE
                    elif file_type in _VIDEO_EXTS:
                        logger.info("[teams][attach][%d] dispatch=video_bytes ext=%s", idx, file_type)
                        from gateway.platforms.base import cache_video_from_bytes
                        cached = cache_video_from_bytes(data, ext="." + file_type)
                        logger.info("[teams][attach][%d] cache_video_from_bytes -> %r", idx, cached)
                        media_urls.append(cached)
                        media_types.append(f"video/{file_type}")
                        msg_type_override = msg_type_override or MessageType.VIDEO
                    elif file_type in _DOC_EXT_TO_MIME:
                        logger.info("[teams][attach][%d] dispatch=document_bytes ext=%s mime=%s", idx, file_type, _DOC_EXT_TO_MIME[file_type])
                        cached = cache_document_from_bytes(data, filename)
                        logger.info("[teams][attach][%d] cache_document_from_bytes -> %r", idx, cached)
                        media_urls.append(cached)
                        media_types.append(_DOC_EXT_TO_MIME[file_type])
                        msg_type_override = msg_type_override or MessageType.DOCUMENT
                    else:
                        logger.info(
                            "[teams][attach][%d] DROP unsupported file_type (file_type=%r, name=%r)",
                            idx, file_type, filename,
                        )
                    continue

                # ── Fell through every branch ────────────────────────────────
                # FORENSICS: dump the full content payload for text/html and
                # other dropped attachments so we can see whether Teams is
                # shipping inline <img src=".../hostedContents/.../$value">
                # references in there. DEBUG-level so it's off in production
                # but available when troubleshooting the channel-message inline
                # image flow (PR #2's Graph hostedContents fallback path).
                # Truncate to 8KB to keep logs sane.
                if content_type == "text/html":
                    raw_content = getattr(att, "content", None)
                    if isinstance(raw_content, str):
                        html_text = raw_content
                    elif isinstance(raw_content, dict):
                        html_text = raw_content.get("text") or raw_content.get("html") or repr(raw_content)
                    else:
                        html_text = repr(raw_content)
                    # Pull out any hostedContents URLs explicitly so they're greppable
                    import re as _re
                    hc_urls = _re.findall(
                        r"https?://[^\"'\s>]*?hostedContents/[^\"'\s>]+",
                        html_text,
                    )
                    # The HTML body itself can carry token-bearing URLs (Teams
                    # often inlines hostedContents/SharePoint refs as <img src>
                    # or <a href>), so we deliberately don't log a raw snippet.
                    # Length + URL count is enough to debug routing; individual
                    # URL fingerprints are emitted in the loop below.
                    logger.debug(
                        "[teams][attach][%d] dropped html payload (%d chars, %d hostedContents refs)",
                        idx, len(html_text), len(hc_urls),
                    )
                    for hc_idx, hc_url in enumerate(hc_urls):
                        logger.debug(
                            "[teams][attach][%d] dropped hostedContents[%d]: %s",
                            idx, hc_idx, _url_fingerprint(hc_url),
                        )
                logger.info(
                    "[teams][attach][%d] DROP unhandled (content_type=%r, has_url=%s)",
                    idx, content_type, bool(content_url),
                )
            except Exception as e:
                logger.warning("[teams][attach][%d] EXCEPTION (content_type=%s): %s", idx, content_type, e)

        logger.info(
            "[teams][attach] done: %d cached, types=%r, msg_override=%r",
            len(media_urls), media_types, msg_type_override,
        )

        # Image always wins over other media for downstream classification —
        # the vision pipeline is the most useful interpretation when the user
        # sent both an image and (e.g.) a document in the same message.
        if any(t.startswith("image/") for t in media_types):
            msg_type = MessageType.PHOTO
        elif msg_type_override is not None:
            msg_type = msg_type_override
        else:
            msg_type = MessageType.TEXT

        event = MessageEvent(
            text=text,
            source=source,
            message_type=msg_type,
            media_urls=media_urls,
            media_types=media_types,
            message_id=msg_id,
        )
        await self.handle_message(event)

    async def _send_card(self, chat_id: str, card: "AdaptiveCard") -> "Any":
        """Send an AdaptiveCard, using a stored ConversationReference when available."""
        from microsoft_teams.api import MessageActivityInput

        conv_ref = self._conv_refs.get(chat_id)
        if conv_ref and self._app:
            activity = MessageActivityInput().add_card(card)
            return await self._app.activity_sender.send(activity, conv_ref)
        elif self._app:
            return await self._app.send(chat_id, card)
        return None

    async def _on_card_action(
        self, ctx: "ActivityContext[AdaptiveCardInvokeActivity]"
    ) -> "InvokeResponse[AdaptiveCardActionMessageResponse]":
        """Handle an Adaptive Card Action.Execute button click."""
        from tools.approval import resolve_gateway_approval, has_blocking_approval

        action = ctx.activity.value.action
        data = action.data or {}
        hermes_action = data.get("hermes_action", "")
        session_key = data.get("session_key", "")

        if not hermes_action or not session_key:
            return InvokeResponse(
                status=200,
                body=AdaptiveCardActionMessageResponse(value="Unknown action."),
            )

        # Only authorized users may click approval buttons.
        # Default-deny: require either TEAMS_ALLOWED_USERS or an explicit
        # TEAMS_ALLOW_ALL_USERS=true opt-in. Without one of these set, the
        # bot silently treated every clicker as authorized — meaning any
        # Teams user who could message the bot could approve dangerous commands.
        allowed_csv = os.getenv("TEAMS_ALLOWED_USERS", "").strip()
        allow_all = os.getenv("TEAMS_ALLOW_ALL_USERS", "").strip().lower() in ("1", "true", "yes")

        if not allow_all:
            if not allowed_csv:
                logger.warning(
                    "[teams] card action rejected: TEAMS_ALLOWED_USERS not configured "
                    "and TEAMS_ALLOW_ALL_USERS not set — default deny"
                )
                return InvokeResponse(
                    status=200,
                    body=AdaptiveCardActionMessageResponse(
                        value="⛔ Approval buttons require TEAMS_ALLOWED_USERS to be configured."
                    ),
                )
            from_account = ctx.activity.from_
            clicker_id = getattr(from_account, "aad_object_id", None) or getattr(from_account, "id", "")
            allowed_ids = {uid.strip() for uid in allowed_csv.split(",") if uid.strip()}
            if "*" not in allowed_ids and clicker_id not in allowed_ids:
                logger.warning("[teams] Unauthorized card action by %s — ignoring", clicker_id)
                return InvokeResponse(
                    status=200,
                    body=AdaptiveCardActionMessageResponse(value="⛔ Not authorized."),
                )

        choice_map = {
            "approve_once": "once",
            "approve_session": "session",
            "approve_always": "always",
            "deny": "deny",
        }
        choice = choice_map.get(hermes_action)
        if not choice:
            return InvokeResponse(
                status=200,
                body=AdaptiveCardActionMessageResponse(value="Unknown action."),
            )

        if not has_blocking_approval(session_key):
            return InvokeResponse(
                status=200,
                body=AdaptiveCardActionCardResponse(
                    value=AdaptiveCard()
                    .with_version("1.4")
                    .with_body([TextBlock(text="⚠️ Approval already resolved or expired.", wrap=True)])
                ),
            )

        resolve_gateway_approval(session_key, choice)

        label_map = {
            "once": "✅ Allowed (once)",
            "session": "✅ Allowed (session)",
            "always": "✅ Always allowed",
            "deny": "❌ Denied",
        }
        cmd = data.get("cmd", "")
        desc = data.get("desc", "")
        body = []
        if cmd:
            body.append(TextBlock(text="⚠️ Command Approval Required", wrap=True, weight="Bolder"))
            body.append(TextBlock(text=f"```\n{cmd}\n```", wrap=True))
        if desc:
            body.append(TextBlock(text=f"Reason: {desc}", wrap=True, isSubtle=True))
        body.append(TextBlock(text=label_map[choice], wrap=True, weight="Bolder"))

        return InvokeResponse(
            status=200,
            body=AdaptiveCardActionCardResponse(
                value=AdaptiveCard().with_version("1.4").with_body(body)
            ),
        )

    async def send_exec_approval(
        self,
        chat_id: str,
        command: str,
        session_key: str,
        description: str = "dangerous command",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send an Adaptive Card approval prompt with Allow/Deny buttons."""
        if not self._app:
            return SendResult(success=False, error="Teams app not initialized")

        cmd_preview = command[:2000] + "..." if len(command) > 2000 else command
        # Truncated for button data payload — just enough to reconstruct the card body.
        btn_data_base = {
            "session_key": session_key,
            "cmd": command[:200] + "..." if len(command) > 200 else command,
            "desc": description,
        }

        card = (
            AdaptiveCard()
            .with_version("1.4")
            .with_body([
                TextBlock(text="⚠️ Command Approval Required", wrap=True, weight="Bolder"),
                TextBlock(text=f"```\n{cmd_preview}\n```", wrap=True),
                TextBlock(text=f"Reason: {description}", wrap=True, isSubtle=True),
            ])
            .with_actions([
                ExecuteAction(
                    title="Allow Once",
                    verb="hermes_approve",
                    data={**btn_data_base, "hermes_action": "approve_once"},
                    style="positive",
                ),
                ExecuteAction(
                    title="Allow Session",
                    verb="hermes_approve",
                    data={**btn_data_base, "hermes_action": "approve_session"},
                ),
                ExecuteAction(
                    title="Always Allow",
                    verb="hermes_approve",
                    data={**btn_data_base, "hermes_action": "approve_always"},
                ),
                ExecuteAction(
                    title="Deny",
                    verb="hermes_approve",
                    data={**btn_data_base, "hermes_action": "deny"},
                    style="destructive",
                ),
            ])
        )

        try:
            result = await self._send_card(chat_id, card)
            message_id = getattr(result, "id", None) if result else None
            return SendResult(success=True, message_id=message_id)
        except Exception as e:
            logger.error("[teams] send_exec_approval failed: %s", e, exc_info=True)
            return SendResult(success=False, error=str(e), retryable=True)

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        if not self._app:
            return SendResult(success=False, error="Teams app not initialized")

        formatted = self.format_message(content)
        chunks = self.truncate_message(formatted)
        last_message_id = None

        for chunk in chunks:
            try:
                if reply_to and reply_to.isdigit() and reply_to != "0":
                    try:
                        result = await self._app.reply(chat_id, reply_to, chunk)
                    except Exception as reply_err:
                        # Group chats 400 on threaded sends; the Teams SDK
                        # doesn't expose typed HTTP errors, so fall back on
                        # any exception and log for diagnostics.
                        logger.debug(
                            "Teams reply() failed, falling back to flat send: %s",
                            reply_err,
                        )
                        result = await self._app.send(chat_id, chunk)
                else:
                    result = await self._app.send(chat_id, chunk)
                last_message_id = getattr(result, "id", None)
            except Exception as e:
                return SendResult(success=False, error=str(e), retryable=True)

        return SendResult(success=True, message_id=last_message_id)

    async def send_typing(self, chat_id: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        if not self._app:
            return
        try:
            await self._app.send(chat_id, TypingActivityInput())
        except Exception:
            pass

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        if not self._app:
            return SendResult(success=False, error="Teams app not initialized")

        try:
            import base64
            import mimetypes
            from microsoft_teams.api import Attachment, MessageActivityInput

            if image_url.startswith("http://") or image_url.startswith("https://"):
                content_url = image_url
                mime_type = "image/png"
            else:
                # Local path — encode as base64 data URI
                path = image_url.removeprefix("file://")
                mime_type = mimetypes.guess_type(path)[0] or "image/png"
                with open(path, "rb") as f:
                    content_url = f"data:{mime_type};base64,{base64.b64encode(f.read()).decode()}"

            attachment = Attachment(content_type=mime_type, content_url=content_url)
            activity = MessageActivityInput().add_attachments(attachment)
            if caption:
                activity = activity.add_text(caption)

            conv_ref = self._conv_refs.get(chat_id)
            if conv_ref:
                result = await self._app.activity_sender.send(activity, conv_ref)
            else:
                result = await self._app.send(chat_id, activity)

            return SendResult(success=True, message_id=getattr(result, "id", None))
        except Exception as e:
            logger.error("[teams] send_image failed: %s", e, exc_info=True)
            return SendResult(success=False, error=str(e), retryable=True)

    async def send_image_file(
        self,
        chat_id: str,
        image_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        return await self.send_image(
            chat_id=chat_id,
            image_url=image_path,
            caption=caption,
            reply_to=reply_to,
        )

    # ------------------------------------------------------------------
    # Outbound files — documents, video, voice
    #
    # Teams' wire protocol for file delivery is split:
    #   • DMs: the bot sends a FileConsentCard, the user clicks accept,
    #     Teams fires a fileConsent/invoke with a OneDrive upload URL,
    #     the bot PUTs the bytes there. The DM send path primes
    #     _pending_uploads; the fileConsent/invoke handler drains it.
    #   • Channels / group chats: the bot uploads to a SharePoint
    #     document library via Microsoft Graph, then sends a
    #     file.download.info attachment pointing at the resulting
    #     drive item's webUrl.
    # ------------------------------------------------------------------

    async def send_document(
        self,
        chat_id: str,
        file_path: str,
        caption: Optional[str] = None,
        file_name: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        # ``file_name`` and ``metadata`` are accepted for parity with the base
        # class signature; Teams cards always render the on-disk basename and
        # Teams has no per-message metadata channel like Telegram's.
        del file_name, metadata
        return await self._send_local_file(chat_id, file_path, caption, reply_to)

    async def send_video(
        self,
        chat_id: str,
        video_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        del metadata
        return await self._send_local_file(chat_id, video_path, caption, reply_to)

    async def send_voice(
        self,
        chat_id: str,
        audio_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        del metadata
        return await self._send_local_file(chat_id, audio_path, caption, reply_to)

    async def _send_local_file(
        self,
        chat_id: str,
        path: str,
        caption: Optional[str],
        reply_to: Optional[str],
    ) -> SendResult:
        """Read *path* from disk and dispatch via the right Teams flow."""
        if not os.path.isfile(path):
            return SendResult(
                success=False,
                error=f"file not found: {path}",
                retryable=False,
            )
        try:
            with open(path, "rb") as f:
                data = f.read()
        except OSError as e:
            return SendResult(
                success=False,
                error=f"could not read file {path}: {e}",
                retryable=False,
            )
        filename = os.path.basename(path)
        if self._is_channel_chat(chat_id):
            return await self._send_channel_file(
                chat_id, filename, data, caption, reply_to
            )
        return await self._send_dm_file_consent(
            chat_id, filename, data, caption, reply_to
        )

    def _is_channel_chat(self, chat_id: str) -> bool:
        """Return True if *chat_id* refers to a Teams channel conversation.

        Channel ids are shaped ``19:<groupId>@thread.tacv2`` whereas DMs
        use ``a:...`` or ``19:<userId>@unq.gbl.spaces``. The id-shape
        heuristic is necessary for cold-path sends (no inbound activity
        seen yet) but a stored conversation reference is authoritative
        when present — Teams tells us the conversation type explicitly.
        """
        ref = self._conv_refs.get(chat_id)
        if ref is not None:
            convo = getattr(ref, "conversation", None)
            convo_type = getattr(convo, "conversation_type", None)
            if convo_type:
                return convo_type == "channel"
        return chat_id.startswith("19:") and "@thread." in chat_id

    async def _handle_file_consent_invoke(
        self,
        ctx: "ActivityContext[FileConsentInvokeActivity]",
    ) -> "Optional[InvokeResponse[None]]":
        """Resolve a fileConsent/invoke (Allow/Decline) activity.

        Looks up the pending upload by ``context.upload_id``, declines
        silently or PUTs the bytes to OneDrive on accept, then posts a
        FileInfoCard so the attachment renders natively in the DM. The
        pending entry is popped under every exit path so a Teams retry
        cannot double-handle the upload.

        Always returns None (the SDK auto-acks 200) — fileConsent retries
        are noisy, and we'd rather log + drop a flaky upload than
        retry-loop on it.
        """
        # Sweep stale pending entries before lookup (mirrors send-side eviction).
        self._evict_stale_pending_uploads()

        value = ctx.activity.value
        if value is None:
            logger.warning("[teams] fileConsent/invoke without value")
            return None

        # Defensive: context may arrive as a dict, an arbitrary pydantic-
        # serialised object, or a plain string. Only the dict shape carries
        # the upload_id we seeded in build_file_consent_card.
        context = value.context or {}
        if isinstance(context, dict):
            upload_id = str(context.get("upload_id") or "")
        else:
            upload_id = ""

        pending = self._pending_uploads.pop(upload_id, None) if upload_id else None
        if pending is None:
            logger.info(
                "[teams] fileConsent invoke for unknown upload_id=%r "
                "(stale card from a previous gateway run, restart between "
                "send and click, or eviction)",
                upload_id,
            )
            return None

        action = str(getattr(value, "action", "") or "").lower()
        # Action enum stringifies as "Action.ACCEPT" — fall back to .value.
        if action.startswith("action."):
            action = action.split(".", 1)[1]
        if action != "accept":
            logger.info(
                "[teams] fileConsent declined for %s", pending["filename"]
            )
            await self._delete_consent_card(ctx, pending)
            return None

        upload_info = value.upload_info
        if upload_info is None or not upload_info.upload_url:
            logger.warning(
                "[teams] fileConsent/invoke missing upload_info.upload_url for %s",
                pending["filename"],
            )
            return None

        # PUT the bytes to the OneDrive upload session.
        success = await self._put_consent_bytes(
            upload_info.upload_url, pending["bytes"]
        )
        if not success:
            return None

        # Delete the consent card so the buttons don't sit grey-then-re-enabled.
        # Done before the FileInfoCard so the user sees the card disappear
        # then the file appear, rather than two cards stacked briefly.
        await self._delete_consent_card(ctx, pending)

        # Post the FileInfoCard so the file renders as a native attachment.
        await self._post_file_info_card(
            chat_id=pending["chat_id"],
            filename=pending["filename"],
            upload_info=upload_info,
            caption=pending.get("caption"),
            reply_to=pending.get("reply_to"),
        )
        return None

    async def _delete_consent_card(
        self,
        ctx: "ActivityContext[FileConsentInvokeActivity]",
        pending: Dict[str, Any],
    ) -> None:
        """Delete the FileConsentCard activity that triggered this invoke.

        Without this, Teams shows the card buttons greying out for a moment
        then re-enabling — the card never reaches a resolved state in the
        UI. Failures are swallowed and logged; consent-card cleanup must
        never break the invoke handler (the upload itself already succeeded
        / was declined).
        """
        activity_id = pending.get("activity_id")
        if not activity_id:
            # Older entries (or send_failed paths) won't have one.
            return
        try:
            conversation_id = ctx.activity.conversation.id
        except AttributeError:
            logger.warning(
                "[teams] consent-card delete skipped — no conversation.id on ctx"
            )
            return
        try:
            await ctx.api.conversations.activities(conversation_id).delete(
                activity_id
            )
        except Exception as e:
            logger.warning(
                "[teams] consent-card delete failed for activity_id=%s: %s",
                activity_id,
                e,
            )

    async def _put_consent_bytes(self, upload_url: str, data: bytes) -> bool:
        """PUT *data* to a OneDrive upload-session URL using the
        single-shot content-range protocol. Returns ``True`` on a 2xx
        response.

        A fresh ``aiohttp.ClientSession`` is created per call. File
        uploads are rare and transient so the per-call cost is fine; if
        throughput becomes a concern we can plumb a shared session
        through the adapter (upstream caches one on a ``_get_http_session``
        helper).
        """
        try:
            import aiohttp
        except ImportError:
            logger.error("[teams] aiohttp missing; cannot PUT FileConsent bytes")
            return False

        size = len(data)
        headers = {
            "Content-Type": "application/octet-stream",
            "Content-Length": str(size),
            "Content-Range": f"bytes 0-{max(size - 1, 0)}/{size}",
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.put(upload_url, data=data, headers=headers) as resp:
                    if resp.status not in (200, 201):
                        body = await resp.text()
                        logger.warning(
                            "[teams] FileConsent PUT failed status=%d body=%s",
                            resp.status,
                            body[:200],
                        )
                        return False
        except aiohttp.ClientError as exc:
            logger.warning("[teams] FileConsent PUT transport error: %s", exc)
            return False
        except Exception:
            logger.exception("[teams] FileConsent PUT raised")
            return False
        return True

    async def _post_file_info_card(
        self,
        *,
        chat_id: str,
        filename: str,
        upload_info: "FileUploadInfo",
        caption: Optional[str],
        reply_to: Optional[str],
    ) -> None:
        """Send the FileInfoCard follow-up after a successful PUT."""
        from plugins.platforms.teams.cards import build_file_info_card

        card = build_file_info_card(
            filename=filename,
            content_url=upload_info.content_url or "",
            unique_id=upload_info.unique_id,
            file_type=upload_info.file_type,
        )
        result = await self._send_attachment(
            chat_id, card, caption=caption, reply_to=reply_to
        )
        if not result.success:
            logger.warning(
                "[teams] FileInfoCard follow-up failed for %s: %s",
                filename,
                result.error,
            )

    async def _send_dm_file_consent(
        self,
        chat_id: str,
        filename: str,
        data: bytes,
        caption: Optional[str],
        reply_to: Optional[str],
    ) -> SendResult:
        """Send a FileConsentCard and remember the bytes for the invoke handler."""
        from plugins.platforms.teams.cards import build_file_consent_card

        # Sweep TTL-expired entries on every DM send so a long-running
        # gateway with users who never click Allow / Decline doesn't grow
        # unboundedly.
        self._evict_stale_pending_uploads()

        size = len(data)
        accept_ctx = {"filename": filename, "service_url_chat_id": chat_id}
        card = build_file_consent_card(
            filename=filename,
            size_bytes=size,
            description=caption or f"Hermes wants to send you {filename}",
            accept_context=accept_ctx,
        )
        upload_id = card["content"]["acceptContext"]["upload_id"]
        # Stash the bytes + routing info; the fileConsent/invoke handler
        # reads this back when the user clicks Accept.
        self._register_pending_upload(
            upload_id,
            {
                "filename": filename,
                "bytes": data,
                "chat_id": chat_id,
                "caption": caption,
                "reply_to": reply_to,
            },
        )
        result = await self._send_attachment(
            chat_id, card, caption=caption, reply_to=reply_to
        )
        if not result.success:
            # Send failed — drop the stashed bytes so we don't leak
            # ~one-file-of-RAM per failed FileConsent send. The user
            # never got a card to click, so the entry is dead weight.
            self._pending_uploads.pop(upload_id, None)
            logger.warning(
                "[teams] FileConsent send failed for upload_id=%s — "
                "bytes dropped, user got no card",
                upload_id,
            )
        else:
            # Stash the consent card's activity_id so the invoke handler
            # can delete the card on Accept/Decline. Without this, Teams
            # leaves the card buttons grey-then-re-enabled (no resolution
            # state) — the user reports the card "doesn't go away".
            entry = self._pending_uploads.get(upload_id)
            if entry is not None:
                entry["activity_id"] = result.message_id
        return result

    def _register_pending_upload(
        self, upload_id: str, entry: Dict[str, Any]
    ) -> None:
        """Insert a pending-upload entry, stamping it and enforcing the size cap.

        Entries are stored with a monotonic timestamp so
        ``_evict_stale_pending_uploads`` can sweep stale ones. When the cap
        is reached we drop the oldest (FIFO via OrderedDict insertion order)
        and emit a warning so saturation is visible in logs.
        """
        self._evict_stale_pending_uploads()
        entry["ts"] = time.monotonic()
        self._pending_uploads[upload_id] = entry
        while len(self._pending_uploads) > self._PENDING_UPLOAD_MAX:
            evicted_id, _evicted = self._pending_uploads.popitem(last=False)
            logger.warning(
                "[teams] _pending_uploads at cap (%d) — evicted oldest upload_id=%s",
                self._PENDING_UPLOAD_MAX,
                evicted_id,
            )

    def _evict_stale_pending_uploads(self) -> None:
        """Drop pending-upload entries older than the TTL."""
        now = time.monotonic()
        ttl = self._PENDING_UPLOAD_TTL_SECONDS
        stale = [
            uid
            for uid, entry in self._pending_uploads.items()
            if now - entry.get("ts", now) > ttl
        ]
        for uid in stale:
            self._pending_uploads.pop(uid, None)
            logger.info(
                "[teams] _pending_uploads evicted stale entry upload_id=%s (TTL %ds)",
                uid,
                ttl,
            )

    async def _send_channel_file(
        self,
        chat_id: str,
        filename: str,
        data: bytes,
        caption: Optional[str],
        reply_to: Optional[str],
    ) -> SendResult:
        """Upload to SharePoint via Graph and post a FileDownload card."""
        if not self._sharepoint_site_id:
            return SendResult(
                success=False,
                error=(
                    "TEAMS_SHAREPOINT_SITE_ID not configured — "
                    "channel uploads disabled"
                ),
                retryable=False,
            )
        graph = await self._ensure_graph()
        # Sanitize the chat_id for use as a folder name. Teams ids contain
        # ':' and '@' which work in URLs but make for ugly SharePoint
        # paths. The mapping is one-way; we never reverse-engineer the
        # chat_id from the folder.
        safe_chat_id = chat_id.replace(":", "_").replace("@", "_at_")
        folder = f"{self._sharepoint_folder}/{safe_chat_id}"
        url = await graph.upload_to_sharepoint(
            site_id=self._sharepoint_site_id,
            folder_path=folder,
            filename=filename,
            content=data,
        )
        if not url:
            # Graph errors are logged inside the client; surface a
            # retryable failure so the gateway can re-queue.
            return SendResult(
                success=False,
                error="SharePoint upload failed",
                retryable=True,
            )
        from plugins.platforms.teams.cards import build_file_download_card

        # Channel uploads don't have a Graph drive-item id readily on hand
        # here (upload_to_sharepoint returns the webUrl, not the item id).
        # Pass the filename + URL; cards.py infers fileType from the
        # extension. unique_id is omitted — the card still renders.
        card = build_file_download_card(
            filename=filename,
            content_url=url,
        )
        return await self._send_attachment(
            chat_id, card, caption=caption, reply_to=reply_to
        )

    async def _ensure_graph(self):
        """Lazily build the GraphClient + GraphTokenProvider."""
        # Lazy single-init: there's no await between the check and the
        # assignments, so asyncio's cooperative scheduler guarantees this
        # block is atomic. Even on the (impossible) double-init path,
        # GraphTokenProvider's per-scope locks make duplicate construction
        # harmless.
        if self._graph is None:
            from plugins.platforms.teams.auth_graph import GraphTokenProvider
            from plugins.platforms.teams.graph import GraphClient

            self._token_provider = GraphTokenProvider(
                client_id=self._client_id,
                tenant_id=self._tenant_id,
                client_secret=self._client_secret,
            )
            self._graph = GraphClient(self._token_provider)
        return self._graph

    async def _try_graph_hosted_bytes(
        self,
        *,
        url: str,
        team_id: Optional[str],
        channel_id: Optional[str],
        activity_id: str,
    ) -> Optional[bytes]:
        """Best-effort Graph hostedContents retrieval.

        Returns the bytes on success, None on every failure path
        (missing context, no hostedContents id in the URL, Graph
        SDK not installed, Graph call raised, or Graph returned no
        data). Callers fall through to whatever the direct path
        produced (which may also be None — that's fine, the
        attachment is just dropped at the higher level).
        """
        hosted_id = _parse_hosted_content_id(url)
        if not (hosted_id and team_id and channel_id and activity_id):
            return None
        try:
            graph = await self._ensure_graph()
        except Exception as exc:
            logger.warning(
                "[teams][graph-fallback] Graph not available: %s", exc,
            )
            return None
        if graph is None:
            return None
        try:
            data = await graph.download_hosted_content(
                team_id=team_id,
                channel_id=channel_id,
                message_id=activity_id,
                hosted_content_id=hosted_id,
            )
        except Exception:
            logger.exception(
                "[teams][graph-fallback] download_hosted_content raised "
                "for msg=%s hc=%s", activity_id, hosted_id,
            )
            return None
        if data is not None:
            logger.info(
                "[teams][graph-fallback] recovered %d bytes via Graph "
                "hostedContents (msg=%s, hc=%s)",
                len(data), activity_id, hosted_id,
            )
        return data

    async def _try_graph_hosted_fallback(
        self,
        *,
        idx: int,
        url: str,
        team_id: Optional[str],
        channel_id: Optional[str],
        activity_id: str,
        kind: str,                 # "image" — only image path uses this v1
        ext: str,
        filename: Optional[str] = None,
    ) -> Optional[str]:
        """Attempt Graph hostedContents fallback then cache the bytes.

        Image-path entry point. Returns a cached file path on
        success, None on failure. Documents from the file.download.info
        branch use _try_graph_hosted_bytes directly (they need to
        fan out to image / audio / video / doc caches based on
        file_type, which the caller already has).
        """
        data = await self._try_graph_hosted_bytes(
            url=url,
            team_id=team_id,
            channel_id=channel_id,
            activity_id=activity_id,
        )
        if data is None:
            return None
        try:
            from gateway.platforms.base import cache_image_from_bytes
            return cache_image_from_bytes(data, ext=ext)
        except Exception:
            logger.exception(
                "[teams][graph-fallback][%d] cache failed for %r",
                idx, filename,
            )
            return None

    async def _send_attachment(
        self,
        chat_id: str,
        attachment_dict: Dict[str, Any],
        *,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
    ) -> SendResult:
        """Send a Bot Framework Attachment dict via the Teams SDK.

        The card builders in :mod:`plugins.platforms.teams.cards` produce
        dicts in Bot Framework wire shape. ``microsoft_teams.api.Attachment``
        is a pydantic model whose ``content_type`` / ``content`` / ``name``
        fields map cleanly. We wrap the dict, attach it to a fresh
        ``MessageActivityInput``, and send via the same conv_ref / direct
        path the existing ``send_image`` uses.

        ``reply_to`` is currently ignored — matching ``send_image``'s
        behaviour. Threading file deliveries to a parent message can be
        added later without changing the public surface.
        """
        # reply_to is a parity arg for the public send_* methods; threading
        # behaviour is a future improvement.
        del reply_to

        if not self._app:
            return SendResult(success=False, error="Teams app not initialized")
        try:
            from microsoft_teams.api import Attachment, MessageActivityInput

            att = Attachment(
                content_type=attachment_dict["contentType"],
                content_url=attachment_dict.get("contentUrl"),
                content=attachment_dict.get("content"),
                name=attachment_dict.get("name"),
            )
            activity = MessageActivityInput().add_attachments(att)
            if caption:
                activity = activity.add_text(caption)

            conv_ref = self._conv_refs.get(chat_id)
            if conv_ref:
                result = await self._app.activity_sender.send(activity, conv_ref)
            else:
                result = await self._app.send(chat_id, activity)
            return SendResult(success=True, message_id=getattr(result, "id", None))
        except Exception as e:
            logger.error("[teams] _send_attachment failed: %s", e, exc_info=True)
            return SendResult(success=False, error=str(e), retryable=True)

    async def get_chat_info(self, chat_id: str) -> dict:
        return {"name": chat_id, "type": "unknown", "chat_id": chat_id}


# ── Interactive setup ─────────────────────────────────────────────────────────

def interactive_setup() -> None:
    """Guide the user through Teams setup using the Teams CLI."""
    from hermes_cli.config import (
        get_env_value,
        save_env_value,
    )
    from hermes_cli.cli_output import (
        prompt,
        prompt_yes_no,
        print_info,
        print_success,
        print_warning,
    )

    existing_id = get_env_value("TEAMS_CLIENT_ID")
    if existing_id:
        print_info(f"Teams: already configured (app ID: {existing_id})")
        if not prompt_yes_no("Reconfigure Teams?", False):
            return

    print_info("You'll need the Teams CLI. If you haven't already:")
    print_info("  npm install -g @microsoft/teams.cli@preview")
    print_info("  teams login")
    print()
    print_info("Then expose port 3978 publicly (devtunnel / ngrok / cloudflared),")
    print_info("and create your bot:")
    print_info("  teams app create --name \"Hermes\" --endpoint \"https://<tunnel>/api/messages\"")
    print()
    print_info("The CLI will print CLIENT_ID, CLIENT_SECRET, and TENANT_ID. Paste them below.")
    print()

    client_id = prompt("Client ID", default=existing_id or "")
    if not client_id:
        print_warning("Client ID is required — skipping Teams setup")
        return
    save_env_value("TEAMS_CLIENT_ID", client_id.strip())

    client_secret = prompt("Client secret", default=get_env_value("TEAMS_CLIENT_SECRET") or "", password=True)
    if not client_secret:
        print_warning("Client secret is required — skipping Teams setup")
        return
    save_env_value("TEAMS_CLIENT_SECRET", client_secret.strip())

    tenant_id = prompt("Tenant ID", default=get_env_value("TEAMS_TENANT_ID") or "")
    if not tenant_id:
        print_warning("Tenant ID is required — skipping Teams setup")
        return
    save_env_value("TEAMS_TENANT_ID", tenant_id.strip())

    print()
    print_info("To find your AAD object ID for the allowlist: teams status --verbose")
    if prompt_yes_no("Restrict access to specific users? (recommended)", True):
        allowed = prompt(
            "Allowed AAD object IDs (comma-separated)",
            default=get_env_value("TEAMS_ALLOWED_USERS") or "",
        )
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


# ── Plugin entry point ────────────────────────────────────────────────────────

def register(ctx) -> None:
    """Plugin entry point — called by the Hermes plugin system."""
    ctx.register_platform(
        name="teams",
        label="Microsoft Teams",
        adapter_factory=lambda cfg: TeamsAdapter(cfg),
        check_fn=check_requirements,
        validate_config=validate_config,
        is_connected=is_connected,
        required_env=["TEAMS_CLIENT_ID", "TEAMS_CLIENT_SECRET", "TEAMS_TENANT_ID"],
        install_hint="pip install microsoft-teams-apps aiohttp",
        setup_fn=interactive_setup,
        # Env-driven auto-configuration — seeds PlatformConfig.extra with
        # client_id/secret/tenant + port + home_channel so env-only setups
        # show up in gateway status without instantiating the Teams SDK.
        env_enablement_fn=_env_enablement,
        # Cron home-channel delivery support.  Lets deliver=teams cron
        # jobs route to the configured Teams chat/channel without editing
        # cron/scheduler.py's hardcoded sets.
        cron_deliver_env_var="TEAMS_HOME_CHANNEL",
        # Out-of-process cron delivery via Bot Framework REST.  Without
        # this hook, deliver=teams cron jobs fail with "No live adapter"
        # when cron runs separately from the gateway.
        standalone_sender_fn=_standalone_send,
        # Auth env vars for _is_user_authorized() integration
        allowed_users_env="TEAMS_ALLOWED_USERS",
        allow_all_env="TEAMS_ALLOW_ALL_USERS",
        # Teams supports up to ~28 KB per message
        max_message_length=28000,
        # Display
        emoji="💼",
        allow_update_command=True,
        # LLM guidance
        platform_hint=(
            "You are chatting via Microsoft Teams. Teams renders a subset of "
            "markdown — bold (**text**), italic (*text*), and inline code "
            "(`code`) work, but complex tables or raw HTML do not. Keep "
            "responses clear and professional."
        ),
    )

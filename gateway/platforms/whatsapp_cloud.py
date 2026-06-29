"""
WhatsApp Cloud API adapter — official Meta WhatsApp Business Platform.

This adapter is a *complement* to ``whatsapp.py`` (the Baileys bridge), not
a replacement. The two are independent:

- ``whatsapp.py``      — unofficial Baileys bridge, personal accounts, no
                         public URL needed, account-ban risk.
- ``whatsapp_cloud.py`` (this file) — official Meta Cloud API, Business
                         account required, public webhook URL required,
                         token-based auth.

Both share gating / mention / formatting behavior via ``WhatsAppBehaviorMixin``.

Phase scope (this file evolves across phases):
- Phase 2 — outbound text via Graph API + webhook server with verify-token
            handshake.
- Phase 3 — X-Hub-Signature-256 HMAC verification (raw body, constant-time)
            + wamid replay protection + dispatch via handle_message. Phase 3
            adapter is end-to-end usable for text DMs.
- Phase 4 — media upload + send (image/video/audio/document), inbound
            media download via the Graph media endpoint, voice-note opus
            conversion via ffmpeg with graceful MP3 fallback when ffmpeg
            isn't on PATH. Document text injection for readable types.
- Phase 5 — 24-hour conversation window + template fallback.

Required env vars to enable the adapter:
- WHATSAPP_CLOUD_PHONE_NUMBER_ID  (the Graph URL path component)
- WHATSAPP_CLOUD_ACCESS_TOKEN     (System User permanent token)

Optional / Phase-3+:
- WHATSAPP_CLOUD_APP_ID
- WHATSAPP_CLOUD_APP_SECRET       (HMAC key for X-Hub-Signature-256)
- WHATSAPP_CLOUD_WABA_ID          (analytics / future use)
- WHATSAPP_CLOUD_VERIFY_TOKEN     (hub.verify_token shared secret)
- WHATSAPP_CLOUD_WEBHOOK_HOST     (default 0.0.0.0)
- WHATSAPP_CLOUD_WEBHOOK_PORT     (default 8090)
- WHATSAPP_CLOUD_WEBHOOK_PATH     (default /whatsapp/webhook)
- WHATSAPP_CLOUD_API_VERSION      (default v20.0)
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import mimetypes
import os
import re
import shutil
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from aiohttp import web

    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    web = None  # type: ignore[assignment]

try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    httpx = None  # type: ignore[assignment]

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    SUPPORTED_DOCUMENT_TYPES,
)
from gateway.platforms.whatsapp_common import WhatsAppBehaviorMixin
from gateway import rich_sent_store
from hermes_constants import get_hermes_dir

logger = logging.getLogger(__name__)


DEFAULT_API_VERSION = "v20.0"
DEFAULT_WEBHOOK_HOST = "0.0.0.0"
DEFAULT_WEBHOOK_PORT = 8090
DEFAULT_WEBHOOK_PATH = "/whatsapp/webhook"
GRAPH_API_BASE = "https://graph.facebook.com"
WEBHOOK_MAX_BODY_BYTES = 3 * 1024 * 1024
# Meta retries failed webhooks for up to 7 days. We don't need to remember
# every wamid for the full retry window — the practical risk is duplicate
# delivery within minutes, not days. 5000 entries with FIFO eviction is
# plenty for normal traffic and bounds memory.
WAMID_DEDUP_CACHE_SIZE = 5000
# Cap for the interactive-button state dicts and the per-chat last-wamid
# cache. Generous for any realistic number of in-flight prompts / chats.
INTERACTIVE_STATE_CACHE_SIZE = 1000

# Per-type size caps documented by Meta for the Cloud API /media endpoint.
# These are the hard limits; we refuse uploads above them with a clean
# error instead of round-tripping to Graph just to be rejected.
# https://developers.facebook.com/docs/whatsapp/cloud-api/reference/media
_MEDIA_SIZE_LIMITS = {
    "image": 5 * 1024 * 1024,        # 5 MB (JPEG, PNG)
    "video": 16 * 1024 * 1024,       # 16 MB
    "audio": 16 * 1024 * 1024,       # 16 MB (MP3, AAC, AMR, OGG opus)
    "document": 100 * 1024 * 1024,   # 100 MB
    "sticker": 100 * 1024,           # 100 KB animated, 500 KB static
}

# Default mime types when we can't guess from the path's extension.
_DEFAULT_MIME = {
    "image": "image/jpeg",
    "video": "video/mp4",
    "audio": "audio/mpeg",
    "document": "application/octet-stream",
    "sticker": "image/webp",
}

# ffmpeg location at import time. ``shutil.which`` honours PATHEXT on
# Windows so a user's ``ffmpeg.exe`` is picked up. None means MP3 voice
# falls back to "audio file attachment" rendering in WhatsApp.
_FFMPEG_PATH = shutil.which("ffmpeg")

# Python's mimetypes module returns RFC-correct but real-world-uncommon
# extensions for some types (audio/ogg → .oga since RFC 5334; audio/mp4
# → .mp4 instead of the de-facto .m4a for voice notes). Our downstream
# STT pipeline whitelists the common-in-the-wild extensions, so override
# the few Meta sends that don't match those defaults.
_WHATSAPP_MIME_EXTENSION_OVERRIDES: Dict[str, str] = {
    # WhatsApp voice notes — opus codec inside an Ogg container.
    "audio/ogg": ".ogg",
    "audio/x-opus+ogg": ".ogg",
    "audio/opus": ".ogg",
    # iOS voice memos — AAC inside an MP4 container; STT tools expect .m4a.
    "audio/mp4": ".m4a",
    "audio/x-m4a": ".m4a",
    # Image — mimetypes occasionally returns .jpe (legacy IANA) instead
    # of .jpg, which trips up tools that switch on extension.
    "image/jpeg": ".jpg",
}


async def _read_limited_request_body(request: Any, max_bytes: int) -> bytes:
    """Read at most ``max_bytes`` from an aiohttp request body."""
    try:
        body = await request.content.readexactly(max_bytes + 1)
    except asyncio.IncompleteReadError as exc:
        body = exc.partial
    if len(body) > max_bytes:
        raise ValueError("payload too large")
    return body


def _ext_for_mime(mime: str) -> Optional[str]:
    """Resolve a mime type to the file extension we want on disk.

    Consults the override map first so types like ``audio/ogg`` produce
    the extension downstream tools actually accept (``.ogg``, not the
    technically-correct-but-broken ``.oga``). Falls back to Python's
    ``mimetypes.guess_extension`` for anything we haven't pinned.
    """
    if not mime:
        return None
    primary = mime.split(";")[0].strip().lower()
    override = _WHATSAPP_MIME_EXTENSION_OVERRIDES.get(primary)
    if override:
        return override
    return mimetypes.guess_extension(primary) or None


# Inbound media cache lives under the user's hermes dir so it survives
# restarts and gateway reloads — same convention the Baileys bridge uses.
_INBOUND_MEDIA_CACHE = Path(get_hermes_dir("platforms/whatsapp_cloud/media", "whatsapp_cloud/media"))


def check_whatsapp_cloud_requirements() -> bool:
    """Return whether transport dependencies are available.

    aiohttp is needed for the webhook server (inbound). httpx is needed
    for Graph API calls (outbound). Both ship with hermes-agent's default
    dependency set, so this should always be True in normal installs.
    """
    return AIOHTTP_AVAILABLE and HTTPX_AVAILABLE


class WhatsAppCloudAdapter(WhatsAppBehaviorMixin, BasePlatformAdapter):
    """WhatsApp Business Cloud API adapter.

    Outbound: HTTPS POST to ``graph.facebook.com/<api_version>/<phone_id>/messages``.
    Inbound: aiohttp server accepting Meta's webhook payloads.

    The mixin must come first in the bases list so its ``format_message``
    overrides ``BasePlatformAdapter.format_message`` (the base provides a
    generic implementation that does not convert markdown to WhatsApp
    syntax). The Baileys adapter does the same.
    """

    splits_long_messages = True  # send() chunks via truncate_message()

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.WHATSAPP_CLOUD)
        extra = config.extra or {}

        # Required
        self._phone_number_id: str = str(extra.get("phone_number_id", "")).strip()
        self._access_token: str = str(extra.get("access_token", "")).strip()

        # Optional / used in later phases
        self._app_id: str = str(extra.get("app_id", "")).strip()
        self._app_secret: str = str(extra.get("app_secret", "")).strip()
        self._waba_id: str = str(extra.get("waba_id", "")).strip()
        self._verify_token: str = str(extra.get("verify_token", "")).strip()

        # Webhook server config
        self._webhook_host: str = str(extra.get("webhook_host", DEFAULT_WEBHOOK_HOST))
        self._webhook_port: int = int(extra.get("webhook_port", DEFAULT_WEBHOOK_PORT))
        self._webhook_path: str = self._normalize_path(
            extra.get("webhook_path", DEFAULT_WEBHOOK_PATH)
        )
        self._health_path: str = self._normalize_path(
            extra.get("health_path", "/health")
        )

        # Graph API
        self._api_version: str = str(extra.get("api_version", DEFAULT_API_VERSION))

        # Behavior-mixin contract: these names are read by the mixin's
        # gating methods. WHATSAPP_CLOUD_* env vars take precedence so the
        # two adapters can run in parallel with independent policies; the
        # shared WHATSAPP_* names remain as fallback for single-adapter
        # setups.
        import os

        self._reply_prefix: Optional[str] = extra.get("reply_prefix")
        # Allowlist: honor the *documented* WHATSAPP_CLOUD_ALLOWED_USERS (the
        # var the setup wizard writes) in addition to WHATSAPP_CLOUD_ALLOW_FROM.
        # The adapter historically read only ALLOW_FROM, so an allowlist
        # configured via the documented var silently dropped every inbound.
        self._allow_from: set[str] = self._normalize_allow_ids(
            self._coerce_allow_list(
                extra.get("allow_from")
                or extra.get("allowFrom")
                or os.getenv("WHATSAPP_CLOUD_ALLOW_FROM")
                or os.getenv("WHATSAPP_CLOUD_ALLOWED_USERS")
            )
        )
        # DM policy: explicit config wins; otherwise choose a safe, working
        # default -- "open" if the operator opted into allow-all, else
        # "allowlist" when an allowlist is configured (so it is actually
        # enforced instead of silently dropping), else "open".
        _allow_all_optin = str(
            os.getenv("WHATSAPP_CLOUD_ALLOW_ALL_USERS", "")
        ).strip().lower() in {"true", "1", "yes"}
        _default_dm_policy = (
            "open" if _allow_all_optin
            else ("allowlist" if self._allow_from else "open")
        )
        self._dm_policy: str = str(
            extra.get("dm_policy")
            or os.getenv("WHATSAPP_CLOUD_DM_POLICY")
            or os.getenv("WHATSAPP_DM_POLICY")
            or _default_dm_policy
        ).strip().lower()
        self._group_policy: str = str(
            extra.get("group_policy")
            or os.getenv("WHATSAPP_CLOUD_GROUP_POLICY")
            or os.getenv("WHATSAPP_GROUP_POLICY", "open")
        ).strip().lower()
        self._group_allow_from: set[str] = self._normalize_allow_ids(
            self._coerce_allow_list(
                extra.get("group_allow_from")
                or extra.get("groupAllowFrom")
                or os.getenv("WHATSAPP_CLOUD_GROUP_ALLOW_FROM")
            )
        )
        self._mention_patterns = self._compile_mention_patterns()

        # Webhook dedup state — wamid → True. OrderedDict gives O(1) FIFO
        # eviction. In-memory only; Phase 5 may promote to SessionDB if we
        # decide we need replay protection across gateway restarts.
        self._seen_wamids: "OrderedDict[str, bool]" = OrderedDict()
        self._duplicate_count: int = 0
        self._accepted_count: int = 0
        self._rejected_signature_count: int = 0

        # One-shot flags for warnings that would otherwise spam the log.
        self._warned_no_ffmpeg: bool = False

        # Per-chat cache of the latest inbound wamid. Meta's typing
        # indicator + read-receipt API requires a specific message_id
        # to attach to (typically "the latest message in the
        # conversation"). We refresh this on every accepted inbound
        # message so ``send_typing`` always has a valid target without
        # threading an extra kwarg through the gateway's base contract.
        # In-memory only; on gateway restart the next inbound message
        # repopulates it.
        self._last_inbound_wamid_by_chat: "OrderedDict[str, str]" = OrderedDict()

        # Interactive-button state. Each maps a short id (embedded in the
        # outbound button payload) → the session/correlation key needed
        # by the gateway's resolver. See ``_handle_interactive_reply`` for
        # the dispatch table. Entries are popped when the user taps a
        # button; ignored prompts would otherwise accumulate forever, so
        # each dict is FIFO-capped via _bounded_put (oldest pending prompt
        # evicted first — an evicted button tap degrades to the plain-text
        # fallback path, same as after a gateway restart).
        #   _clarify_state:        clarify_id → session_key (resolves via
        #                          tools.clarify_gateway.resolve_gateway_clarify)
        #   _exec_approval_state:  approval_id → session_key (resolves via
        #                          tools.approval.resolve_gateway_approval)
        #   _slash_confirm_state:  confirm_id → session_key (resolves via
        #                          tools.slash_confirm.resolve)
        self._clarify_state: "OrderedDict[str, str]" = OrderedDict()
        self._exec_approval_state: "OrderedDict[str, str]" = OrderedDict()
        self._slash_confirm_state: "OrderedDict[str, str]" = OrderedDict()

        # Runtime
        self._runner = None
        self._http_client: Optional["httpx.AsyncClient"] = None

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _normalize_path(path: Any) -> str:
        raw = str(path or "").strip() or "/"
        return raw if raw.startswith("/") else f"/{raw}"

    def _graph_url(self, path: str) -> str:
        """Build a Graph API URL for this adapter's phone-number scope."""
        if path.startswith("/"):
            path = path[1:]
        return f"{GRAPH_API_BASE}/{self._api_version}/{self._phone_number_id}/{path}"

    @staticmethod
    def _bounded_put(cache: "OrderedDict[str, str]", key: str, value: str) -> None:
        """Insert into a FIFO-capped OrderedDict, evicting oldest entries."""
        cache[key] = value
        while len(cache) > INTERACTIVE_STATE_CACHE_SIZE:
            cache.popitem(last=False)

    def _effective_reply_prefix(self) -> str:
        """Cloud API has no self-chat concept — never prepend a reply prefix.

        Override the mixin default which keys off WHATSAPP_MODE=self-chat
        (a Baileys-only setting).
        """
        if self._reply_prefix is not None:
            return self._reply_prefix.replace("\\n", "\n")
        return ""

    @staticmethod
    def _normalize_allow_ids(ids: set[str]) -> set[str]:
        """Normalize allowlist entries to bare wa_id form.

        The Cloud API identifies users by bare wa_id (digits, no JID
        suffix), while Baileys uses ``<digits>@s.whatsapp.net`` JIDs.
        Users sharing an allowlist between both adapters (or pasting a
        JID/phone number with ``+`` or separators) should still match,
        so strip any ``@...`` suffix and non-digit characters.
        """
        normalized: set[str] = set()
        for entry in ids:
            bare = entry.split("@", 1)[0]
            digits = re.sub(r"\D", "", bare)
            normalized.add(digits or entry)
        return normalized

    def _is_dm_allowed(self, sender_id: str) -> bool:
        """Allowlist check against the normalized bare wa_id."""
        if self._dm_policy == "allowlist":
            bare = re.sub(r"\D", "", str(sender_id).split("@", 1)[0])
            return (bare or sender_id) in self._allow_from
        return super()._is_dm_allowed(sender_id)

    def _open_dm_opted_in(self) -> bool:
        """Also honor the documented WHATSAPP_CLOUD_ALLOW_ALL_USERS opt-in.

        The shared mixin only checks GATEWAY_ALLOW_ALL_USERS /
        WHATSAPP_ALLOW_ALL_USERS; the Cloud adapter's documented open-access
        opt-in is WHATSAPP_CLOUD_ALLOW_ALL_USERS, so honor it here too.
        """
        if str(os.getenv("WHATSAPP_CLOUD_ALLOW_ALL_USERS", "")).strip().lower() in {"true", "1", "yes"}:
            return True
        return super()._open_dm_opted_in()

    # ------------------------------------------------------------------ lifecycle
    async def connect(self, *, is_reconnect: bool = False) -> bool:
        if not check_whatsapp_cloud_requirements():
            self._set_fatal_error(
                "whatsapp_cloud_deps_missing",
                "aiohttp and httpx are required for whatsapp_cloud — "
                "reinstall hermes-agent.",
                retryable=False,
            )
            return False
        if not self._phone_number_id or not self._access_token:
            self._set_fatal_error(
                "whatsapp_cloud_unconfigured",
                "WHATSAPP_CLOUD_PHONE_NUMBER_ID and WHATSAPP_CLOUD_ACCESS_TOKEN "
                "are required.",
                retryable=False,
            )
            return False

        # Outbound HTTP client. Tighter keepalive matches other platform
        # adapters so idle CLOSE_WAIT drains promptly (#18451).
        from gateway.platforms._http_client_limits import platform_httpx_limits

        self._http_client = httpx.AsyncClient(
            timeout=30.0, limits=platform_httpx_limits()
        )

        # Inbound webhook server.
        app = web.Application()
        app.router.add_get(self._health_path, self._handle_health)
        app.router.add_get(self._webhook_path, self._handle_verify)
        app.router.add_post(self._webhook_path, self._handle_webhook)

        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self._webhook_host, self._webhook_port)
        await site.start()

        self._mark_connected()
        logger.info(
            "[whatsapp_cloud] Listening on %s:%d%s (Graph %s, phone_id=%s)",
            self._webhook_host,
            self._webhook_port,
            self._webhook_path,
            self._api_version,
            self._phone_number_id,
        )
        if not self._verify_token:
            logger.warning(
                "[whatsapp_cloud] WHATSAPP_CLOUD_VERIFY_TOKEN is not set — "
                "the GET subscription handshake will fail until it is."
            )
        if not self._app_secret:
            logger.warning(
                "[whatsapp_cloud] WHATSAPP_CLOUD_APP_SECRET is not set — "
                "incoming webhook POSTs will be refused with 503. Set "
                "the app secret to enable inbound message delivery."
            )
        return True

    async def disconnect(self) -> None:
        if self._runner is not None:
            try:
                await self._runner.cleanup()
            except Exception:
                logger.exception("[whatsapp_cloud] webhook server cleanup failed")
            self._runner = None
        if self._http_client is not None:
            try:
                await self._http_client.aclose()
            except Exception:
                logger.exception("[whatsapp_cloud] http client close failed")
            self._http_client = None
        self._mark_disconnected()

    # ------------------------------------------------------------------ outbound
    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a text message via Graph API.

        ``chat_id`` is the recipient's WhatsApp ID (``wa_id``) — typically
        their phone number with country code, no plus sign.
        """
        if self._http_client is None:
            return SendResult(success=False, error="Not connected")
        if not content or not content.strip():
            return SendResult(success=True, message_id=None)

        formatted = self.format_message(content)
        chunks = self.truncate_message(formatted, self._outgoing_chunk_limit())

        url = self._graph_url("messages")
        headers = {
            "Authorization": f"Bearer {self._access_token}",
            "Content-Type": "application/json",
        }

        last_message_id: Optional[str] = None
        for idx, chunk in enumerate(chunks):
            payload: Dict[str, Any] = {
                "messaging_product": "whatsapp",
                "recipient_type": "individual",
                "to": chat_id,
                "type": "text",
                "text": {"body": chunk, "preview_url": True},
            }
            if reply_to and idx == 0:
                # Quote the user's message on the first chunk only.
                payload["context"] = {"message_id": reply_to}
            try:
                resp = await self._http_client.post(url, headers=headers, json=payload)
            except Exception as exc:
                logger.exception("[whatsapp_cloud] send failed")
                return SendResult(success=False, error=str(exc))

            if resp.status_code != 200:
                # Meta returns structured errors in the body — surface them
                # to the caller so log lines have actionable context.
                try:
                    body = resp.json()
                except Exception:
                    body = {"raw": resp.text[:500]}
                error_msg = self._format_graph_error(body, resp.status_code)
                logger.warning(
                    "[whatsapp_cloud] send rejected (status=%d): %s",
                    resp.status_code,
                    error_msg,
                )
                return SendResult(success=False, error=error_msg)

            try:
                data = resp.json()
                ids = data.get("messages") or []
                if ids:
                    last_message_id = ids[0].get("id")
            except Exception:
                pass

        # Remember (chat_id, wamid) -> text so that when the user replies to
        # one of our messages, _build_message_event_from_cloud can resolve the
        # quoted text. Meta's inbound webhook ``context`` object carries only
        # the quoted message's id, never its text, so without this index the
        # agent would never learn what the user was replying to. Best-effort;
        # rich_sent_store swallows all errors.
        if last_message_id:
            rich_sent_store.record(chat_id, last_message_id, formatted)

        return SendResult(success=True, message_id=last_message_id)

    # ------------------------------------------------------------------ typing indicator + read receipts
    #
    # Meta couples these into a single API call: a POST to /messages
    # with ``status: "read"`` marks the message read (blue double
    # checkmarks), and the optional ``typing_indicator`` field
    # additionally shows the user a "typing..." pip in their chat UI.
    # The indicator auto-dismisses when we respond OR after 25 seconds,
    # whichever comes first — so this matches "I see your message and
    # I'm working on a reply" UX exactly.
    #
    # The API requires a specific message_id to attach to. We cache the
    # latest inbound wamid per chat in _last_inbound_wamid_by_chat
    # (refreshed in _build_message_event_from_cloud) so this method can
    # look it up without needing the gateway base contract to plumb
    # event.message_id into send_typing's signature.

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """Mark the latest inbound message as read AND show a typing
        indicator in the user's chat UI.

        Best-effort: any error (no inbound wamid yet, network failure,
        stale token, message older than 30 days) is swallowed silently
        so the agent's main reply path isn't blocked by UX polish.
        """
        if self._http_client is None:
            return
        wamid = self._last_inbound_wamid_by_chat.get(chat_id)
        if not wamid:
            # No inbound message yet for this chat (or cache cleared on
            # restart) — skip. The next inbound message will repopulate.
            return

        url = self._graph_url("messages")
        headers = {
            "Authorization": f"Bearer {self._access_token}",
            "Content-Type": "application/json",
        }
        payload = {
            "messaging_product": "whatsapp",
            "status": "read",
            "message_id": wamid,
            "typing_indicator": {"type": "text"},
        }
        try:
            resp = await self._http_client.post(url, headers=headers, json=payload)
        except Exception:
            # Network / connection error — silent fail. Typing UX must
            # never block message dispatch.
            return
        # Best-effort: surface 4xx for ops visibility but don't raise.
        # Code 131009 = "Parameter value is not valid" (typically wamid
        # > 30 days old) — common after a long-quiet conversation, log
        # at info not warning.
        if resp.status_code != 200:
            try:
                body = resp.json()
                code = ((body or {}).get("error") or {}).get("code")
            except Exception:
                code = None
            if code == 131009:
                logger.info(
                    "[whatsapp_cloud] typing/read indicator rejected: "
                    "wamid %s likely older than 30 days", wamid,
                )
            else:
                logger.debug(
                    "[whatsapp_cloud] typing/read indicator returned %d (%s)",
                    resp.status_code, code,
                )

    # ------------------------------------------------------------------ interactive messages
    #
    # WhatsApp Cloud supports two interactive primitives we use here:
    #   * ``interactive.type=button`` — up to 3 quick-reply buttons. Each
    #     button has an ``id`` (≤256 chars, returned verbatim on tap) and
    #     a ``title`` (≤20 chars, the label shown). Used for clarify with
    #     ≤3 choices, exec_approval, and slash_confirm.
    #   * ``interactive.type=list``   — a single "Tap to choose" button
    #     that opens a sheet with up to 10 rows. Used for clarify with
    #     >3 choices and the model picker.
    #
    # Unlike utility templates these are FREE-FORM and need no Meta-side
    # approval. They only work *inside* the 24-hour conversation window —
    # which is fine because all five senders below fire in direct response
    # to a user message (clarify mid-conversation, approval mid-tool-call,
    # etc.) so we're always inside the window when they're invoked.

    async def _post_interactive(
        self,
        chat_id: str,
        interactive_body: Dict[str, Any],
        reply_to: Optional[str] = None,
    ) -> SendResult:
        """Low-level POST for an ``interactive`` message payload.

        ``interactive_body`` is the inner ``interactive: {...}`` dict —
        the caller supplies ``type``, ``body``, and ``action``. This
        wrapper handles auth, error mapping, and message_id extraction so
        each send_* method stays focused on its own button shape.
        """
        if self._http_client is None:
            return SendResult(success=False, error="Not connected")

        url = self._graph_url("messages")
        headers = {
            "Authorization": f"Bearer {self._access_token}",
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": chat_id,
            "type": "interactive",
            "interactive": interactive_body,
        }
        if reply_to:
            payload["context"] = {"message_id": reply_to}

        try:
            resp = await self._http_client.post(url, headers=headers, json=payload)
        except Exception as exc:
            logger.exception("[whatsapp_cloud] interactive send failed")
            return SendResult(success=False, error=str(exc))

        if resp.status_code != 200:
            try:
                body = resp.json()
            except Exception:
                body = {"raw": resp.text[:500]}
            error_msg = self._format_graph_error(body, resp.status_code)
            logger.warning(
                "[whatsapp_cloud] interactive rejected (status=%d): %s",
                resp.status_code, error_msg,
            )
            return SendResult(success=False, error=error_msg)

        last_message_id: Optional[str] = None
        try:
            data = resp.json()
            ids = data.get("messages") or []
            if ids:
                last_message_id = ids[0].get("id")
        except Exception:
            pass
        return SendResult(success=True, message_id=last_message_id)

    @staticmethod
    def _truncate_button_label(text: str, limit: int = 20) -> str:
        """WhatsApp caps quick-reply button titles at 20 chars and list-row
        titles at 24. Truncate with an ellipsis so we surface as much of
        the choice as fits."""
        text = str(text or "").strip()
        if len(text) <= limit:
            return text
        # Reserve 1 char for the ellipsis. WhatsApp counts the ellipsis
        # toward the limit.
        return text[: max(1, limit - 1)] + "…"

    @staticmethod
    def _truncate_body(text: str, limit: int = 1024) -> str:
        """``interactive.body.text`` caps at 1024 chars."""
        text = str(text or "")
        if len(text) <= limit:
            return text
        return text[: limit - 3] + "..."

    async def send_clarify(
        self,
        chat_id: str,
        question: str,
        choices: Optional[list],
        clarify_id: str,
        session_key: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Render a clarify prompt as native WhatsApp interactive buttons.

        - 1–3 choices → ``interactive.type=button`` (inline pill buttons).
        - 4+ choices → ``interactive.type=list`` (tap-to-open sheet with
          up to 10 rows). Telegram's "Other (type answer)" escape hatch
          is appended as the final row, picking it flips the entry into
          text-capture mode handled by the gateway's text intercept.
        - 0 choices (open-ended) → plain text question; the next message
          in the session is captured by the gateway and resolves clarify.

        The button ``id`` field carries ``cl:<clarify_id>:<idx>`` (or
        ``:other``); inbound webhook parsing dispatches on the prefix.
        """
        if self._http_client is None:
            return SendResult(success=False, error="Not connected")

        question = (question or "").strip()
        reply_to = (metadata or {}).get("reply_to_message_id") if metadata else None

        # Open-ended → just send the question, gateway captures next msg.
        if not choices:
            return await self.send(chat_id, f"❓ {question}", reply_to=reply_to)

        # Mirror Telegram: render full choice text in body so long
        # options aren't truncated to the 20-char button label cap.
        # Truncate choices to MAX_CHOICES (4) — the tool layer enforces
        # this already, but be defensive.
        choices_list = [str(c).strip() for c in choices[:10] if str(c).strip()]
        option_lines = "\n".join(
            f"{i + 1}. {c}" for i, c in enumerate(choices_list)
        )
        body_text = self._truncate_body(f"❓ {question}\n\n{option_lines}")

        if len(choices_list) <= 3:
            buttons = [
                {
                    "type": "reply",
                    "reply": {
                        "id": f"cl:{clarify_id}:{idx}",
                        "title": self._truncate_button_label(str(idx + 1)),
                    },
                }
                for idx in range(len(choices_list))
            ]
            interactive: Dict[str, Any] = {
                "type": "button",
                "body": {"text": body_text},
                "action": {"buttons": buttons},
            }
        else:
            # List mode: rows must each have id + title (≤24 chars).
            # Description (≤72 chars) renders below the title — we put
            # the truncated choice text there for skimmability.
            rows = []
            for idx, choice_text in enumerate(choices_list):
                rows.append({
                    "id": f"cl:{clarify_id}:{idx}",
                    "title": self._truncate_button_label(f"{idx + 1}", limit=24),
                    "description": self._truncate_button_label(choice_text, limit=72),
                })
            rows.append({
                "id": f"cl:{clarify_id}:other",
                "title": "✏️ Other",
                "description": "Type your own answer",
            })
            interactive = {
                "type": "list",
                "body": {"text": body_text},
                "action": {
                    "button": "Choose",
                    "sections": [{"title": "Options", "rows": rows}],
                },
            }

        result = await self._post_interactive(chat_id, interactive, reply_to=reply_to)
        if result.success:
            self._bounded_put(self._clarify_state, clarify_id, session_key)
        return result

    async def send_exec_approval(
        self,
        chat_id: str,
        command: str,
        session_key: str,
        description: str = "dangerous command",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Render a dangerous-command approval prompt with native buttons.

        Two quick-reply buttons (Approve / Deny). Tapping resolves the
        waiting agent via ``tools.approval.resolve_gateway_approval`` —
        same mechanism as the text ``/approve`` flow. The agent thread
        is blocked until the user taps or types a response.
        """
        if self._http_client is None:
            return SendResult(success=False, error="Not connected")

        # WhatsApp body caps at 1024 chars; reserve room for the
        # framing prose around the command.
        cmd = command or ""
        cmd_preview = cmd if len(cmd) <= 800 else cmd[:800] + "..."
        body_text = self._truncate_body(
            f"⚠️ *Command Approval Required*\n\n"
            f"```\n{cmd_preview}\n```\n\n"
            f"Reason: {description}"
        )

        approval_id = uuid.uuid4().hex[:12]
        reply_to = (metadata or {}).get("reply_to_message_id") if metadata else None

        interactive = {
            "type": "button",
            "body": {"text": body_text},
            "action": {
                "buttons": [
                    {
                        "type": "reply",
                        "reply": {"id": f"appr:{approval_id}:approve", "title": "✅ Approve"},
                    },
                    {
                        "type": "reply",
                        "reply": {"id": f"appr:{approval_id}:deny", "title": "❌ Deny"},
                    },
                ],
            },
        }

        result = await self._post_interactive(chat_id, interactive, reply_to=reply_to)
        if result.success:
            self._bounded_put(self._exec_approval_state, approval_id, session_key)
        return result

    async def send_slash_confirm(
        self,
        chat_id: str,
        title: str,
        message: str,
        session_key: str,
        confirm_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Render a 3-button slash-command confirmation prompt.

        Mirrors Telegram's send_slash_confirm: Approve Once / Always /
        Cancel. The confirm_id is supplied by the caller (slash command
        handler) — we just store the session_key mapping for the inbound
        resolver to look up.
        """
        if self._http_client is None:
            return SendResult(success=False, error="Not connected")

        body_text = self._truncate_body(f"*{title}*\n\n{message}")
        reply_to = (metadata or {}).get("reply_to_message_id") if metadata else None

        interactive = {
            "type": "button",
            "body": {"text": body_text},
            "action": {
                "buttons": [
                    {
                        "type": "reply",
                        "reply": {"id": f"sc:once:{confirm_id}", "title": "✅ Approve Once"},
                    },
                    {
                        "type": "reply",
                        "reply": {"id": f"sc:always:{confirm_id}", "title": "🔒 Always"},
                    },
                    {
                        "type": "reply",
                        "reply": {"id": f"sc:cancel:{confirm_id}", "title": "❌ Cancel"},
                    },
                ],
            },
        }

        result = await self._post_interactive(chat_id, interactive, reply_to=reply_to)
        if result.success:
            self._bounded_put(self._slash_confirm_state, confirm_id, session_key)
        return result

    @staticmethod
    def _format_graph_error(body: Dict[str, Any], status_code: int) -> str:
        err = (body or {}).get("error") or {}
        # Graph API error shape:
        # {"error": {"message": "...", "type": "...", "code": ..., "fbtrace_id": "..."}}
        message = err.get("message") or body.get("raw") or "unknown error"
        code = err.get("code")
        if code is not None:
            return f"graph error {code} (HTTP {status_code}): {message}"
        return f"HTTP {status_code}: {message}"

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        # Cloud API doesn't expose a direct "chat info" endpoint the way
        # Slack/Discord do — we just echo the wa_id. Profile name (when
        # known) flows in via webhook ``contacts[].profile.name`` and is
        # cached on the MessageEvent, not here.
        return {"name": chat_id, "type": "dm"}

    # ------------------------------------------------------------------ outbound media
    async def _upload_media(
        self,
        file_path: str,
        media_kind: str,
        mime_type: Optional[str] = None,
    ) -> tuple[Optional[str], Optional[str]]:
        """Upload a local file to the Graph /media endpoint.

        Returns ``(media_id, None)`` on success, ``(None, error_string)``
        on failure. Two-step send: this gets the id, then ``_send_media``
        references it. Used when we have a local file and no public URL.

        ``media_kind`` is one of "image", "video", "audio", "document",
        "sticker" — selects size cap + default mime fallback.
        """
        if self._http_client is None:
            return None, "Not connected"
        if not os.path.exists(file_path):
            return None, f"File not found: {file_path}"

        size = os.path.getsize(file_path)
        cap = _MEDIA_SIZE_LIMITS.get(media_kind, _MEDIA_SIZE_LIMITS["document"])
        if size > cap:
            return None, (
                f"File {os.path.basename(file_path)} is {size} bytes; "
                f"Cloud API {media_kind} cap is {cap} bytes"
            )

        if not mime_type:
            mime_type, _ = mimetypes.guess_type(file_path)
        if not mime_type:
            mime_type = _DEFAULT_MIME.get(media_kind, "application/octet-stream")

        url = self._graph_url("media")
        headers = {"Authorization": f"Bearer {self._access_token}"}
        try:
            with open(file_path, "rb") as fh:
                files = {
                    "file": (os.path.basename(file_path), fh, mime_type),
                    "messaging_product": (None, "whatsapp"),
                    "type": (None, mime_type),
                }
                resp = await self._http_client.post(url, headers=headers, files=files)
        except Exception as exc:
            logger.exception("[whatsapp_cloud] media upload failed")
            return None, str(exc)

        if resp.status_code != 200:
            try:
                body = resp.json()
            except Exception:
                body = {"raw": resp.text[:500]}
            return None, self._format_graph_error(body, resp.status_code)

        try:
            data = resp.json()
            media_id = data.get("id")
        except Exception:
            media_id = None
        if not media_id:
            return None, "Upload response missing 'id'"
        return media_id, None

    async def _send_media(
        self,
        chat_id: str,
        media_kind: str,
        *,
        media_id: Optional[str] = None,
        media_link: Optional[str] = None,
        caption: Optional[str] = None,
        filename: Optional[str] = None,
        reply_to: Optional[str] = None,
    ) -> SendResult:
        """POST a media message referencing either an uploaded media_id or
        a public ``link``.

        Exactly one of ``media_id`` or ``media_link`` must be set. Captions
        and filenames are passed through where Meta accepts them (caption
        on image/video/document; filename on document only).
        """
        if self._http_client is None:
            return SendResult(success=False, error="Not connected")
        if bool(media_id) == bool(media_link):
            return SendResult(
                success=False,
                error="Exactly one of media_id or media_link must be set",
            )

        url = self._graph_url("messages")
        headers = {
            "Authorization": f"Bearer {self._access_token}",
            "Content-Type": "application/json",
        }

        media_block: Dict[str, Any] = {}
        if media_id:
            media_block["id"] = media_id
        else:
            media_block["link"] = media_link
        if caption and media_kind in {"image", "video", "document"}:
            media_block["caption"] = caption
        if filename and media_kind == "document":
            media_block["filename"] = filename

        payload: Dict[str, Any] = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": chat_id,
            "type": media_kind,
            media_kind: media_block,
        }
        if reply_to:
            payload["context"] = {"message_id": reply_to}

        try:
            resp = await self._http_client.post(url, headers=headers, json=payload)
        except Exception as exc:
            logger.exception("[whatsapp_cloud] media send failed")
            return SendResult(success=False, error=str(exc))

        if resp.status_code != 200:
            try:
                body = resp.json()
            except Exception:
                body = {"raw": resp.text[:500]}
            error_msg = self._format_graph_error(body, resp.status_code)
            logger.warning(
                "[whatsapp_cloud] media send rejected (status=%d, kind=%s): %s",
                resp.status_code, media_kind, error_msg,
            )
            return SendResult(success=False, error=error_msg)

        try:
            data = resp.json()
            ids = data.get("messages") or []
            wamid = ids[0].get("id") if ids else None
        except Exception:
            wamid = None
        return SendResult(success=True, message_id=wamid)

    async def _send_media_from_path_or_link(
        self,
        chat_id: str,
        source: str,
        media_kind: str,
        *,
        caption: Optional[str] = None,
        filename: Optional[str] = None,
        reply_to: Optional[str] = None,
        mime_type: Optional[str] = None,
    ) -> SendResult:
        """Smart dispatcher: HTTPS URL → ``link`` send; local path → upload + ``id`` send.

        Prefers the ``link`` path when possible (one fewer Graph round
        trip). Meta fetches from the URL themselves. Used as the common
        backend for ``send_image`` / ``send_video`` / etc. — keeps the
        public method bodies thin.
        """
        if source.startswith(("http://", "https://")):
            return await self._send_media(
                chat_id,
                media_kind,
                media_link=source,
                caption=caption,
                filename=filename,
                reply_to=reply_to,
            )
        media_id, err = await self._upload_media(source, media_kind, mime_type)
        if err:
            return SendResult(success=False, error=err)
        return await self._send_media(
            chat_id,
            media_kind,
            media_id=media_id,
            caption=caption,
            filename=filename,
            reply_to=reply_to,
        )

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        """Send an image by public URL. Prefers Meta's ``link`` mode.

        ``**kwargs`` absorbs platform-agnostic args the base class passes
        (e.g. ``metadata``) that the Cloud API doesn't have a use for.
        Mirrors send_image_file / send_video / send_voice / send_document.
        """
        return await self._send_media_from_path_or_link(
            chat_id, image_url, "image", caption=caption, reply_to=reply_to
        )

    async def send_image_file(
        self,
        chat_id: str,
        image_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        """Send a local image file via two-step upload + id."""
        return await self._send_media_from_path_or_link(
            chat_id, image_path, "image", caption=caption, reply_to=reply_to
        )

    async def send_video(
        self,
        chat_id: str,
        video_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        """Send a video. Local path → upload; HTTPS URL → link mode."""
        return await self._send_media_from_path_or_link(
            chat_id, video_path, "video", caption=caption, reply_to=reply_to
        )

    async def send_voice(
        self,
        chat_id: str,
        audio_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        """Send an audio file as a WhatsApp voice message.

        WhatsApp renders ``audio/ogg; codecs=opus`` as the green
        voice-note bubble; other audio types (MP3, AAC, etc.) appear as
        a generic audio attachment. Hermes TTS produces MP3, so we try
        ffmpeg conversion to opus first and fall back to sending the
        MP3 as-is when ffmpeg is unavailable.
        """
        source = audio_path
        mime_type: Optional[str] = None

        is_local_mp3 = (
            not audio_path.startswith(("http://", "https://"))
            and audio_path.lower().endswith(".mp3")
            and os.path.exists(audio_path)
        )
        if is_local_mp3:
            opus_path = await self._convert_to_opus(audio_path)
            if opus_path:
                try:
                    result = await self._send_media_from_path_or_link(
                        chat_id, opus_path, "audio",
                        caption=caption, reply_to=reply_to,
                        mime_type="audio/ogg; codecs=opus",
                    )
                finally:
                    # The .ogg is a transient conversion artifact next to
                    # the source MP3 — clean it up after upload so voice
                    # sends don't leak a file per message.
                    try:
                        os.unlink(opus_path)
                    except OSError:
                        pass
                return result
            # Will deliver as MP3 attachment, not voice bubble.
            # Warn-once is logged inside _convert_to_opus.
            mime_type = "audio/mpeg"

        return await self._send_media_from_path_or_link(
            chat_id, source, "audio",
            caption=caption, reply_to=reply_to, mime_type=mime_type,
        )

    async def send_document(
        self,
        chat_id: str,
        file_path: str,
        caption: Optional[str] = None,
        file_name: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        """Send a document attachment with optional filename + caption."""
        return await self._send_media_from_path_or_link(
            chat_id, file_path, "document",
            caption=caption,
            filename=file_name or os.path.basename(file_path),
            reply_to=reply_to,
        )

    # ------------------------------------------------------------------ opus conversion
    async def _convert_to_opus(self, mp3_path: str) -> Optional[str]:
        """Convert an MP3 to ``audio/ogg; codecs=opus`` for voice bubbles.

        Returns the path to the converted file, or None if ffmpeg is
        missing / conversion fails (caller falls back to sending the
        original MP3 as an audio file).

        ``-application voip`` tunes the opus encoder for speech.
        ``-b:a 32k -vbr on`` matches the bitrate WhatsApp produces for
        native voice notes (small files, good intelligibility).
        """
        if not _FFMPEG_PATH:
            self._warn_once_no_ffmpeg()
            return None

        out_path = mp3_path.rsplit(".", 1)[0] + ".ogg"
        try:
            proc = await asyncio.create_subprocess_exec(
                _FFMPEG_PATH, "-y", "-i", mp3_path,
                "-c:a", "libopus", "-b:a", "32k", "-vbr", "on",
                "-application", "voip", out_path,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            if proc.returncode != 0 or not Path(out_path).exists():
                logger.error(
                    "[whatsapp_cloud] ffmpeg opus conversion failed "
                    "(returncode=%s): %s",
                    proc.returncode,
                    (stderr or b"").decode("utf-8", errors="replace")[:500],
                )
                return None
            return out_path
        except Exception:
            logger.exception("[whatsapp_cloud] ffmpeg subprocess raised")
            return None

    def _warn_once_no_ffmpeg(self) -> None:
        if self._warned_no_ffmpeg:
            return
        self._warned_no_ffmpeg = True
        logger.warning(
            "[whatsapp_cloud] ffmpeg not found on PATH — voice messages will "
            "be delivered as MP3 audio attachments instead of native voice "
            "notes (green waveform bubble). Install ffmpeg to enable: "
            "Windows `winget install Gyan.FFmpeg`, macOS `brew install ffmpeg`, "
            "Linux package manager."
        )

    # ------------------------------------------------------------------ inbound media
    async def _download_media_to_cache(
        self,
        media_id: str,
        *,
        ext_hint: Optional[str] = None,
    ) -> tuple[Optional[str], Optional[str]]:
        """Two-step Graph media download: ``GET /<id>`` → temp URL → bytes.

        Returns ``(local_path, mime_type)`` on success. ``mime_type``
        falls back to what Graph reports in the metadata response.
        Returns ``(None, None)`` on any failure (logged).

        The temporary URL from step 1 is signed and expires in ~5
        minutes; we download immediately and never persist the URL.
        """
        if self._http_client is None:
            return None, None
        # Defense in depth: media_id comes from the (signature-verified)
        # webhook payload, but it's interpolated into both a Graph URL and
        # a cache filename below — refuse anything that isn't a plain
        # Meta-style media id so a hostile payload can't traverse paths.
        media_id = str(media_id).strip()
        if not re.fullmatch(r"[A-Za-z0-9._-]+", media_id):
            logger.warning(
                "[whatsapp_cloud] refusing malformed media id %r", media_id[:64]
            )
            return None, None
        headers = {"Authorization": f"Bearer {self._access_token}"}

        # Step 1 — metadata (gives us a temporary signed URL + mime)
        try:
            meta_resp = await self._http_client.get(
                f"{GRAPH_API_BASE}/{self._api_version}/{media_id}",
                headers=headers,
            )
        except Exception:
            logger.exception(
                "[whatsapp_cloud] media metadata fetch raised (id=%s)", media_id
            )
            return None, None
        if meta_resp.status_code != 200:
            logger.warning(
                "[whatsapp_cloud] media metadata fetch failed (id=%s, status=%d)",
                media_id, meta_resp.status_code,
            )
            return None, None

        try:
            meta = meta_resp.json()
        except Exception:
            return None, None
        temp_url = meta.get("url")
        mime = meta.get("mime_type") or ""
        if not temp_url:
            return None, None

        # Step 2 — bytes (auth required even though URL is signed; Meta
        # documents this explicitly — the URL alone is not enough).
        try:
            blob_resp = await self._http_client.get(temp_url, headers=headers)
        except Exception:
            logger.exception(
                "[whatsapp_cloud] media bytes fetch raised (id=%s)", media_id
            )
            return None, None
        if blob_resp.status_code != 200:
            logger.warning(
                "[whatsapp_cloud] media bytes fetch failed (id=%s, status=%d)",
                media_id, blob_resp.status_code,
            )
            return None, None

        # Decide the extension. Prefer the override map so audio/ogg
        # produces .ogg (not the technically-correct-but-broken .oga
        # mimetypes returns by default). Fall back to ext_hint then
        # ``.bin`` for unknown types.
        ext = ext_hint
        if not ext and mime:
            ext = _ext_for_mime(mime)
        if not ext:
            ext = ".bin"

        _INBOUND_MEDIA_CACHE.mkdir(parents=True, exist_ok=True)
        out_path = _INBOUND_MEDIA_CACHE / f"{media_id}{ext}"
        try:
            out_path.write_bytes(blob_resp.content)
        except OSError:
            logger.exception(
                "[whatsapp_cloud] failed to write cached media (id=%s)", media_id
            )
            return None, None

        return str(out_path), mime or None


    # ------------------------------------------------------------------ inbound
    async def _handle_health(self, request: "web.Request") -> "web.Response":
        return web.json_response(
            {
                "status": "ok",
                "platform": self.platform.value,
                "phone_number_id": self._phone_number_id,
                "webhook_path": self._webhook_path,
                "verify_token_configured": bool(self._verify_token),
                "app_secret_configured": bool(self._app_secret),
                "ffmpeg_present": _FFMPEG_PATH is not None,
                "accepted": self._accepted_count,
                "duplicates": self._duplicate_count,
                "rejected_signature": self._rejected_signature_count,
            }
        )

    async def _handle_verify(self, request: "web.Request") -> "web.Response":
        """Meta subscription verification handshake.

        Meta calls GET ``<webhook>?hub.mode=subscribe&hub.verify_token=...
        &hub.challenge=...``. We must echo the challenge as plain text iff
        ``hub.mode == "subscribe"`` AND ``hub.verify_token`` matches the
        shared secret. Constant-time comparison.
        """
        if not self._verify_token:
            # Misconfigured server — refuse rather than silently accepting
            # any verify_token, which would let an attacker subscribe.
            return web.Response(status=503, text="verify_token not configured")

        mode = request.query.get("hub.mode", "")
        token = request.query.get("hub.verify_token", "")
        challenge = request.query.get("hub.challenge", "")

        if mode != "subscribe":
            return web.Response(status=400, text="bad mode")

        # Constant-time compare to avoid token-length / token-content leaks
        # via timing. ``hmac.compare_digest`` works on str.
        import hmac as _hmac

        if not _hmac.compare_digest(token, self._verify_token):
            return web.Response(status=403, text="verify_token mismatch")
        if not challenge:
            return web.Response(status=400, text="missing challenge")
        return web.Response(text=challenge, content_type="text/plain")

    async def _handle_webhook(self, request: "web.Request") -> "web.Response":
        """Inbound webhook POST handler.

        Lifecycle:
          1. Read raw bytes (signature is over the raw body — JSON parsing
             must NOT happen first, or the bytes change).
          2. Verify ``X-Hub-Signature-256`` HMAC against ``app_secret``.
          3. Parse JSON.
          4. Walk ``entry[].changes[].value.{messages, statuses, contacts}``.
          5. Per-message: dedup by wamid, build MessageEvent, dispatch via
             ``handle_message`` (which runs the mixin's gating).
          6. Always respond 200 once we've ack'd a valid request — Meta
             retries on non-200 for up to 7 days, and we don't want to
             multiply downstream agent work because of a transient bug
             during dispatch.
        """
        # Meta's documented max payload is 3MB. Read one byte past the limit
        # so oversized chunked bodies are rejected before buffering the rest.
        try:
            raw = await _read_limited_request_body(
                request,
                WEBHOOK_MAX_BODY_BYTES,
            )
        except ValueError:
            return web.Response(status=413)
        except Exception:
            return web.Response(status=400)

        # Refuse to accept anything if app_secret isn't configured. Without
        # it we can't authenticate the sender, and the handler would be a
        # data-injection point. Same defensive posture as the GET verify
        # handshake refusing when verify_token is empty.
        if not self._app_secret:
            logger.error(
                "[whatsapp_cloud] webhook POST refused: app_secret unset. "
                "Set WHATSAPP_CLOUD_APP_SECRET to enable inbound delivery."
            )
            return web.Response(status=503, text="app_secret not configured")

        signature_header = request.headers.get("X-Hub-Signature-256", "")
        if not self._verify_signature(raw, signature_header):
            self._rejected_signature_count += 1
            logger.warning(
                "[whatsapp_cloud] rejected webhook: invalid X-Hub-Signature-256 "
                "(header=%r, body_len=%d)",
                signature_header,
                len(raw),
            )
            return web.Response(status=401)

        # Parse only AFTER signature passes — bad JSON from an attacker is
        # already filtered out, this just guards against Meta sending
        # something malformed.
        import json as _json

        try:
            payload = _json.loads(raw)
        except Exception:
            logger.warning("[whatsapp_cloud] webhook body is not valid JSON")
            return web.Response(status=400)

        if not isinstance(payload, dict):
            return web.Response(status=400)

        await self._dispatch_payload(payload)
        return web.Response(status=200)

    # ------------------------------------------------------------------ signature
    def _verify_signature(self, raw_body: bytes, header: str) -> bool:
        """Verify the X-Hub-Signature-256 HMAC.

        Meta sends ``sha256=<hex>``; we compute the same HMAC with
        ``app_secret`` as the key and ``raw_body`` (UTF-8 bytes, not
        re-serialized JSON) as the message. Constant-time compare.
        """
        if not self._app_secret or not header:
            return False
        if not header.startswith("sha256="):
            return False
        expected_hex = header[len("sha256="):].strip()
        if not expected_hex:
            return False
        computed = hmac.new(
            self._app_secret.encode("utf-8"),
            raw_body,
            hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(computed.lower(), expected_hex.lower())

    # ------------------------------------------------------------------ dispatch
    def _dedup_wamid(self, wamid: str) -> bool:
        """Return True if this wamid is being seen for the first time.

        Returns False (and increments duplicate counter) if the wamid is
        already in the in-memory cache. Cache is FIFO-evicted at
        ``WAMID_DEDUP_CACHE_SIZE``.
        """
        if not wamid:
            # No wamid means we can't dedup — let it through. Meta should
            # always populate ``id``, but be defensive.
            return True
        if wamid in self._seen_wamids:
            self._duplicate_count += 1
            return False
        self._seen_wamids[wamid] = True
        # Trim oldest entries to stay under the cap.
        while len(self._seen_wamids) > WAMID_DEDUP_CACHE_SIZE:
            self._seen_wamids.popitem(last=False)
        return True

    async def _dispatch_payload(self, payload: Dict[str, Any]) -> None:
        """Walk a verified Meta webhook payload and dispatch each message.

        Payload shape (truncated):
          {object, entry: [{id, changes: [{value: {messages, contacts,
          statuses, metadata}, field: "messages"}]}]}

        We surface ``messages`` events as MessageEvents; ``statuses``
        events (sent/delivered/read/failed) are logged but not dispatched
        — the agent doesn't currently consume delivery receipts and
        forwarding them would create noisy synthetic events.
        """
        if payload.get("object") != "whatsapp_business_account":
            logger.debug(
                "[whatsapp_cloud] ignoring non-WABA payload (object=%r)",
                payload.get("object"),
            )
            return
        for entry in payload.get("entry") or []:
            if not isinstance(entry, dict):
                continue
            for change in entry.get("changes") or []:
                if not isinstance(change, dict):
                    continue
                if change.get("field") != "messages":
                    # Other fields (account_alerts, template_status_update,
                    # etc.) are subscription-dependent and not message
                    # ingress. Silent skip.
                    continue
                value = change.get("value") or {}
                contacts = value.get("contacts") or []
                metadata = value.get("metadata") or {}
                # Build a wa_id → profile-name index for the messages we're
                # about to surface.
                contacts_by_waid: Dict[str, str] = {}
                for contact in contacts:
                    if not isinstance(contact, dict):
                        continue
                    wa_id = str(contact.get("wa_id") or "").strip()
                    profile = contact.get("profile") or {}
                    name = str(profile.get("name") or "").strip()
                    if wa_id:
                        contacts_by_waid[wa_id] = name

                for raw_message in value.get("messages") or []:
                    if not isinstance(raw_message, dict):
                        continue
                    wamid = str(raw_message.get("id") or "").strip()
                    if not self._dedup_wamid(wamid):
                        logger.debug(
                            "[whatsapp_cloud] duplicate wamid %s, skipping",
                            wamid,
                        )
                        continue
                    try:
                        event = await self._build_message_event_from_cloud(
                            raw_message, contacts_by_waid, metadata
                        )
                    except Exception:
                        # Build errors must not bubble out either: the wamid
                        # is already dedup-marked above, so a 500 here would
                        # make Meta retry the batch and every message in it
                        # (including this one) would be silently dropped as
                        # a duplicate. Log and move on to the next message.
                        logger.exception(
                            "[whatsapp_cloud] failed to build event for wamid %s",
                            wamid,
                        )
                        continue
                    if event is None:
                        continue
                    self._accepted_count += 1
                    try:
                        await self.handle_message(event)
                    except Exception:
                        # Dispatch errors must not bubble out — Meta would
                        # retry the whole batch, multiplying the bug.
                        logger.exception(
                            "[whatsapp_cloud] handle_message raised for wamid %s",
                            wamid,
                        )

                # Log status updates at debug level — useful for diagnosing
                # "did Meta accept my outbound" without flooding INFO logs.
                for status in value.get("statuses") or []:
                    if isinstance(status, dict):
                        logger.debug(
                            "[whatsapp_cloud] status %s for %s",
                            status.get("status"),
                            status.get("id"),
                        )

    async def _dispatch_interactive_reply(
        self,
        raw_message: Dict[str, Any],
        contacts_by_waid: Dict[str, str],
    ) -> bool:
        """Route an inbound interactive reply to the matching resolver.

        Returns True if the tap was claimed (caller should drop the
        webhook entry without dispatching a fresh conversation turn).
        Returns False when the id has no recognized prefix, no live
        state entry, or the resolver itself reports no waiter — in
        those cases the caller falls back to standard text-event
        dispatch, which treats the button title as a normal user
        message. That graceful fallback covers stale-tap and
        cross-process-restart scenarios.

        Dispatch table:
          ``cl:<clarify_id>:<idx|other>``  → resolve_gateway_clarify
          ``appr:<approval_id>:approve|deny`` → resolve_gateway_approval
          ``sc:<once|always|cancel>:<confirm_id>`` → slash_confirm.resolve
        """
        inter = raw_message.get("interactive") or {}
        # button_reply (interactive.type=button) and list_reply
        # (interactive.type=list) carry id+title in different sub-objects.
        inner = inter.get("button_reply") or inter.get("list_reply") or {}
        button_id = str(inner.get("id") or "").strip()
        if not button_id:
            return False

        # Clarify: cl:<clarify_id>:<idx|other>
        if button_id.startswith("cl:"):
            parts = button_id.split(":", 2)
            if len(parts) != 3:
                return False
            _, clarify_id, choice = parts
            session_key = self._clarify_state.pop(clarify_id, None)
            if not session_key:
                logger.info(
                    "[whatsapp_cloud] clarify tap with no matching state "
                    "(clarify_id=%s) — likely stale; falling back to text",
                    clarify_id,
                )
                return False
            try:
                from tools.clarify_gateway import resolve_gateway_clarify
            except ImportError:
                logger.warning(
                    "[whatsapp_cloud] clarify resolver unavailable; "
                    "falling back to text dispatch"
                )
                return False
            if choice == "other":
                # User wants to type a free-form answer. Flip the entry
                # into text-capture mode so the gateway's text-intercept
                # (in _handle_message) picks up their next message and
                # resolves the clarify. Without this flip,
                # ``get_pending_for_session`` won't return the entry —
                # the next text would fall through to the regular agent
                # path, which collides with the agent thread still
                # blocked in clarify and produces an "Interrupting
                # current task" loop.
                try:
                    from tools.clarify_gateway import mark_awaiting_text
                    flipped = mark_awaiting_text(clarify_id)
                except Exception:
                    logger.exception(
                        "[whatsapp_cloud] mark_awaiting_text failed for %s",
                        clarify_id,
                    )
                    flipped = False
                if not flipped:
                    # Entry vanished between the user tap and our handler
                    # (timeout, /new, gateway restart). Drop the stale
                    # state and fall through to text dispatch so the
                    # user's tap isn't completely ignored.
                    logger.info(
                        "[whatsapp_cloud] clarify 'Other' tap but entry "
                        "missing (clarify_id=%s); falling back to text",
                        clarify_id,
                    )
                    return False
                # Put state back since we popped it earlier — keep the
                # clarify_id → session_key mapping live in case future
                # taps land on the same prompt.
                self._clarify_state[clarify_id] = session_key
                try:
                    await self.send(
                        str(raw_message.get("from") or ""),
                        "✏️ Type your answer:",
                    )
                except Exception:
                    logger.exception("[whatsapp_cloud] clarify other-prompt failed")
                return True  # claim so we don't also dispatch the tap as text
            try:
                idx = int(choice)
            except ValueError:
                logger.warning(
                    "[whatsapp_cloud] clarify tap had non-int choice: %r",
                    choice,
                )
                # Put state back so a follow-up text can still resolve.
                self._clarify_state[clarify_id] = session_key
                return False
            # Use the title text as the resolved response so the agent
            # sees the human-readable answer, not the index. Title is
            # the numeric label ("1", "2", ...) so we look up the
            # full choice from the original prompt — but we didn't
            # persist that. Fall back to passing the index; the agent
            # has the prompt in context and can interpret it.
            response_text = str(inner.get("title") or str(idx + 1))
            resolved = resolve_gateway_clarify(clarify_id, response_text)
            if not resolved:
                # Resolver couldn't find a waiter (e.g. agent already
                # timed out). Fall through to text dispatch.
                logger.info(
                    "[whatsapp_cloud] clarify resolver reported no waiter "
                    "(clarify_id=%s) — falling back to text", clarify_id,
                )
                return False
            return True

        # Exec approval: appr:<approval_id>:approve|deny
        if button_id.startswith("appr:"):
            parts = button_id.split(":", 2)
            if len(parts) != 3:
                return False
            _, approval_id, choice = parts
            session_key = self._exec_approval_state.pop(approval_id, None)
            if not session_key:
                logger.info(
                    "[whatsapp_cloud] approval tap with no matching state "
                    "(approval_id=%s) — likely stale; falling back to text",
                    approval_id,
                )
                return False
            if choice not in ("approve", "deny"):
                self._exec_approval_state[approval_id] = session_key
                return False
            try:
                from tools.approval import resolve_gateway_approval
            except ImportError:
                logger.warning(
                    "[whatsapp_cloud] approval resolver unavailable"
                )
                return False
            count = resolve_gateway_approval(session_key, choice)
            if not count:
                logger.info(
                    "[whatsapp_cloud] approval resolver reported no waiter "
                    "(session_key=%s) — likely already resolved",
                    session_key,
                )
            # Send confirmation message — paralleling Telegram's UX.
            try:
                confirm_text = (
                    "✅ Approved." if choice == "approve" else "❌ Denied."
                )
                await self.send(str(raw_message.get("from") or ""), confirm_text)
            except Exception:
                logger.exception("[whatsapp_cloud] approval confirm failed")
            return True

        # Slash confirm: sc:<once|always|cancel>:<confirm_id>
        if button_id.startswith("sc:"):
            parts = button_id.split(":", 2)
            if len(parts) != 3:
                return False
            _, choice, confirm_id = parts
            session_key = self._slash_confirm_state.pop(confirm_id, None)
            if not session_key:
                logger.info(
                    "[whatsapp_cloud] slash_confirm tap with no matching state "
                    "(confirm_id=%s) — likely stale", confirm_id,
                )
                return False
            if choice not in ("once", "always", "cancel"):
                self._slash_confirm_state[confirm_id] = session_key
                return False
            try:
                from tools import slash_confirm as _slash_confirm_mod
            except ImportError:
                logger.warning(
                    "[whatsapp_cloud] slash_confirm resolver unavailable"
                )
                return False
            try:
                result_text = await _slash_confirm_mod.resolve(
                    session_key, confirm_id, choice
                )
            except Exception:
                logger.exception("[whatsapp_cloud] slash_confirm.resolve failed")
                return True  # still claim the tap; surfacing it as text wouldn't help
            if result_text:
                try:
                    await self.send(str(raw_message.get("from") or ""), result_text)
                except Exception:
                    logger.exception("[whatsapp_cloud] slash_confirm reply failed")
            return True

        # Unknown prefix — let text dispatch handle the title as a
        # regular message. Could be a tap from a plugin-defined adapter
        # we don't know about; treating it as text is the safe default.
        return False

    async def _build_message_event_from_cloud(
        self,
        raw_message: Dict[str, Any],
        contacts_by_waid: Dict[str, str],
        metadata: Dict[str, Any],
    ) -> Optional[MessageEvent]:
        """Convert a Cloud-API message object into a Hermes MessageEvent.

        Phase 4 expands beyond text to download inbound media (image,
        video, audio/voice, document, sticker) by ``media_id`` via the
        two-step Graph endpoint. Cached files are populated into
        ``media_urls`` / ``media_types`` so the agent's vision and STT
        layers see them. Text-readable documents (.txt, .md, .json,
        source code, etc.) are read and prepended to the message body
        up to 100KB — same heuristic the Baileys adapter uses.

        Returns None if the message is filtered out by the mixin's
        gating (broadcast filter, allow-list, mention requirements).
        """
        msg_type_str = str(raw_message.get("type") or "text").lower()

        # Interactive replies (button taps, list selections) carry an ``id``
        # we set when sending the prompt. Route those to the appropriate
        # gateway resolver BEFORE falling through to text dispatch — the
        # resolver unblocks the waiting agent thread, so we don't want to
        # also kick a fresh conversation turn off the same tap.
        if msg_type_str == "interactive":
            handled = await self._dispatch_interactive_reply(
                raw_message, contacts_by_waid
            )
            if handled:
                return None

        body = ""
        if msg_type_str == "text":
            text = raw_message.get("text") or {}
            body = str(text.get("body") or "")
        elif msg_type_str in {"button", "interactive"}:
            # Quick-reply buttons. Treat the button payload as text so the
            # agent can reason about the user's choice.
            if msg_type_str == "button":
                body = str((raw_message.get("button") or {}).get("text") or "")
            else:
                inter = raw_message.get("interactive") or {}
                # button_reply / list_reply both expose ``title``
                inner = inter.get("button_reply") or inter.get("list_reply") or {}
                body = str(inner.get("title") or "")
        elif msg_type_str in {"image", "video", "audio", "voice", "document", "sticker"}:
            # Captions live on image / video / document. Other media types
            # don't carry a caption in Meta's spec, but be defensive.
            inner = raw_message.get(msg_type_str) or {}
            body = str(inner.get("caption") or "")

        message_type = {
            "text": MessageType.TEXT,
            "image": MessageType.PHOTO,
            "video": MessageType.VIDEO,
            "audio": MessageType.VOICE,
            "voice": MessageType.VOICE,
            "document": MessageType.DOCUMENT,
            "sticker": MessageType.PHOTO,
            "button": MessageType.TEXT,
            "interactive": MessageType.TEXT,
            "location": MessageType.TEXT,
            "contacts": MessageType.TEXT,
        }.get(msg_type_str, MessageType.TEXT)

        sender_id = str(raw_message.get("from") or "").strip()
        sender_name = contacts_by_waid.get(sender_id, "")

        # Cloud API doesn't have a separate "chat" entity for DMs — chat_id
        # equals the sender's wa_id. Group support is deferred to v2.
        #
        # Defensive guard: if Meta ever delivers a group-shaped payload
        # (group support is capability-tier gated by Meta; some WABAs
        # have it enabled), refuse rather than silently treating it as
        # a DM. Group messages carry a ``chat`` field on the message
        # object identifying the group JID — its absence signals DM.
        chat_field = raw_message.get("chat")
        if chat_field:
            logger.warning(
                "[whatsapp_cloud] received group-shaped message (chat=%s, "
                "wamid=%s) — group support is not yet implemented; dropping. "
                "Use the Baileys whatsapp adapter for group chats.",
                chat_field, raw_message.get("id"),
            )
            return None

        chat_id = sender_id

        # Build the data dict the mixin's _should_process_message expects.
        # Cloud API uses different field names from Baileys, so we adapt.
        gating_data = {
            "chatId": chat_id,
            "senderId": sender_id,
            "isGroup": False,  # Phase 3 = DM only
            "body": body,
        }
        if not self._should_process_message(gating_data):
            return None

        # Download media if this is a non-text message type. Inbound media
        # arrives as ``{type: "image", image: {id, mime_type, sha256, ...}}``.
        media_urls: list[str] = []
        media_types: list[str] = []
        if msg_type_str in {"image", "video", "audio", "voice", "document", "sticker"}:
            inner = raw_message.get(msg_type_str) or {}
            media_id = str(inner.get("id") or "").strip()
            inbound_mime = str(inner.get("mime_type") or "").strip()
            if media_id:
                ext_hint = None
                if inbound_mime:
                    ext_hint = _ext_for_mime(inbound_mime)
                local_path, dl_mime = await self._download_media_to_cache(
                    media_id, ext_hint=ext_hint
                )
                if local_path:
                    media_urls.append(local_path)
                    media_types.append(dl_mime or inbound_mime or "application/octet-stream")
                    logger.info(
                        "[whatsapp_cloud] cached inbound %s media: %s",
                        msg_type_str, local_path,
                    )
                else:
                    logger.warning(
                        "[whatsapp_cloud] failed to download inbound %s (id=%s) — "
                        "agent will see message metadata but not the binary",
                        msg_type_str, media_id,
                    )
                # Document: original filename for the agent's UX.
                if msg_type_str == "document":
                    fname = str(inner.get("filename") or "").strip()
                    if fname and not body:
                        body = f"[Document: {fname}]"

        # For text-readable documents, inject the file content directly into
        # the message body so the agent can reason about it without a
        # separate read_file call. Same heuristic the Baileys adapter uses.
        # 100KB cap matches Telegram/Discord/Slack.
        MAX_TEXT_INJECT_BYTES = 100 * 1024
        if msg_type_str == "document" and media_urls:
            for doc_path in media_urls:
                ext = Path(doc_path).suffix.lower()
                if ext in {
                    ".txt", ".md", ".csv", ".json", ".xml", ".yaml", ".yml",
                    ".log", ".py", ".js", ".ts", ".html", ".css",
                }:
                    try:
                        file_size = Path(doc_path).stat().st_size
                        if file_size > MAX_TEXT_INJECT_BYTES:
                            logger.info(
                                "[whatsapp_cloud] skipping text injection for %s "
                                "(%d bytes > %d)",
                                doc_path, file_size, MAX_TEXT_INJECT_BYTES,
                            )
                            continue
                        content = Path(doc_path).read_text(
                            encoding="utf-8", errors="replace"
                        )
                        display_name = Path(doc_path).name
                        injection = f"[Content of {display_name}]:\n{content}"
                        body = f"{injection}\n\n{body}" if body else injection
                    except OSError:
                        logger.exception(
                            "[whatsapp_cloud] failed to read document text: %s",
                            doc_path,
                        )

        # context.id is set when the user replied to a prior message. Meta's
        # webhook only gives us the quoted message's id (and its author in
        # context.from) — never the quoted text. We resolve the text from
        # rich_sent_store, which we populate on every inbound message (below)
        # and every outbound send. Without this the agent receives a bare
        # reply_to_message_id and run.py can't inject the "[Replying to: ...]"
        # disambiguation prefix (it gates on reply_to_text being present).
        context = raw_message.get("context") or {}
        reply_to_id = str(context.get("id") or "").strip() or None
        reply_to_text: Optional[str] = None
        reply_to_is_own = False
        if reply_to_id:
            reply_to_text = rich_sent_store.lookup(chat_id, reply_to_id)
            # context.from is the wa_id of the quoted message's author. When it
            # matches our business number the user replied to the bot's own
            # message; otherwise they replied to one of their own messages.
            quoted_from = str(context.get("from") or "").strip()
            our_number = str(metadata.get("display_phone_number") or "").strip()
            if quoted_from and our_number:
                reply_to_is_own = quoted_from == our_number

        source = self.build_source(
            chat_id=chat_id,
            chat_name=sender_name or chat_id,
            chat_type="dm",
            user_id=sender_id,
            user_name=sender_name or None,
        )

        # Cloud API timestamps are unix seconds (string). MessageEvent
        # doesn't enforce a type but downstream code formats with it.
        wamid = str(raw_message.get("id") or "") or None
        if wamid and chat_id:
            # Refresh the per-chat latest-wamid cache so a subsequent
            # send_typing call can attach the indicator + read receipt
            # to this message. Done HERE (after _should_process_message
            # gating) so filtered messages don't leak typing on
            # unwanted inbound traffic.
            self._bounded_put(self._last_inbound_wamid_by_chat, chat_id, wamid)
            # Index this message's text by wamid so a later reply to it can
            # resolve the quoted text (Meta's webhook context carries only
            # the id). Mirrors the outbound record in send(). Best-effort.
            if body:
                rich_sent_store.record(chat_id, wamid, body)

        return MessageEvent(
            text=body,
            message_type=message_type,
            source=source,
            raw_message=raw_message,
            message_id=wamid,
            reply_to_message_id=reply_to_id,
            reply_to_text=reply_to_text,
            reply_to_is_own_message=reply_to_is_own,
            media_urls=media_urls,
            media_types=media_types,
        )

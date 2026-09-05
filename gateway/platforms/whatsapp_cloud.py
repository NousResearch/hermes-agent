"""WhatsApp Cloud API adapter — official Meta WhatsApp Business Platform (Graph API +
aiohttp webhook). Complements the Baileys bridge plugin; both share gating / mention /
formatting behavior via ``WhatsAppBehaviorMixin``.

Required env: WHATSAPP_CLOUD_PHONE_NUMBER_ID, WHATSAPP_CLOUD_ACCESS_TOKEN. Optional:
WHATSAPP_CLOUD_APP_ID, _APP_SECRET (HMAC key for X-Hub-Signature-256), _WABA_ID,
_VERIFY_TOKEN (hub.verify_token), _WEBHOOK_HOST (unset → dual-stack all interfaces),
_WEBHOOK_PORT (8090), _WEBHOOK_PATH (/whatsapp/webhook), _API_VERSION (v20.0)."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import hmac
import importlib
import json
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
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult
from gateway.platforms.whatsapp_common import _OPTIN_TRUTHY, WhatsAppBehaviorMixin, _get_wsecret
from gateway.platforms.media_cache import ext_for_mime
from gateway import rich_sent_store
from hermes_constants import get_hermes_dir

logger = logging.getLogger(__name__)


DEFAULT_API_VERSION = "v20.0"
# ``None`` → aiohttp binds one socket per address family; "0.0.0.0" was unreachable on IPv6-only hosts.
DEFAULT_WEBHOOK_HOST = None
DEFAULT_WEBHOOK_PORT = 8090
DEFAULT_WEBHOOK_PATH = "/whatsapp/webhook"
GRAPH_API_BASE = "https://graph.facebook.com"
WEBHOOK_MAX_BODY_BYTES = 3 * 1024 * 1024
# Meta retries failed webhooks for up to 7 days, but the practical duplicate risk is
# within minutes — 5000 FIFO entries bounds memory and covers that.
WAMID_DEDUP_CACHE_SIZE = 5000
INTERACTIVE_STATE_CACHE_SIZE = 1000  # interactive-button state dicts + per-chat last-wamid cache

# Meta's hard per-type caps for /media (developers.facebook.com/docs/whatsapp/cloud-api/reference/media);
# refuse locally instead of round-tripping to Graph just to be rejected.
_MEDIA_SIZE_LIMITS = {
    "image": 5 * 1024 * 1024, "video": 16 * 1024 * 1024, "audio": 16 * 1024 * 1024,
    "document": 100 * 1024 * 1024, "sticker": 100 * 1024,
}
# Default mime types when we can't guess from the path's extension.
_DEFAULT_MIME = {
    "image": "image/jpeg", "video": "video/mp4", "audio": "audio/mpeg",
    "document": "application/octet-stream", "sticker": "image/webp",
}

# None → MP3 voice falls back to an "audio file attachment" rendering in WhatsApp.
_FFMPEG_PATH = shutil.which("ffmpeg")

# mimetypes returns RFC-correct but uncommon extensions (audio/ogg → .oga, image/jpeg → .jpe);
# downstream STT/vision whitelists the in-the-wild forms, so pin the few Meta sends.
_WHATSAPP_MIME_EXTENSION_OVERRIDES: Dict[str, str] = {
    "audio/ogg": ".ogg", "audio/x-opus+ogg": ".ogg", "audio/opus": ".ogg",
    "audio/mp4": ".m4a", "audio/x-m4a": ".m4a", "image/jpeg": ".jpg",
}

_INBOUND_MEDIA_KINDS = {"image", "video", "audio", "voice", "document", "sticker"}
_TEXT_INJECT_EXTS = {".txt", ".md", ".csv", ".json", ".xml", ".yaml", ".yml", ".log", ".py", ".js", ".ts", ".html", ".css"}
_MAX_TEXT_INJECT_BYTES = 100 * 1024  # matches Telegram/Discord/Slack
_MESSAGE_TYPE_BY_KIND = {
    "text": MessageType.TEXT, "image": MessageType.PHOTO, "video": MessageType.VIDEO,
    "audio": MessageType.VOICE, "voice": MessageType.VOICE, "document": MessageType.DOCUMENT,
    "sticker": MessageType.PHOTO, "button": MessageType.TEXT, "interactive": MessageType.TEXT,
    "location": MessageType.TEXT, "contacts": MessageType.TEXT, "reaction": MessageType.TEXT,
}
_HTTP_PREFIXES = ("http://", "https://")


def _interactive_inner(raw_message: Dict[str, Any]) -> Dict[str, Any]:
    """``button_reply`` (type=button) or ``list_reply`` (type=list); both carry id+title."""
    inter = raw_message.get("interactive") or {}
    return inter.get("button_reply") or inter.get("list_reply") or {}


def _location_body(raw_message: Dict[str, Any]) -> str:
    """Cloud location payload → bracketed text, mirroring the Baileys bridge's formatLocationText."""
    loc = raw_message.get("location") or {}
    name = str(loc.get("name") or loc.get("address") or "").strip()
    lat, lng = loc.get("latitude"), loc.get("longitude")
    coords = f"{lat},{lng}" if lat is not None and lng is not None else ""
    parts = " ".join(part for part in (name, coords) if part)
    return f"[Location: {parts}]" if parts else ""


def _reaction_body(raw_message: Dict[str, Any]) -> str:
    """Cloud reaction payload → bracketed text, mirroring the Baileys bridge's formatReactionText."""
    reaction = raw_message.get("reaction") or {}
    emoji = str(reaction.get("emoji") or "").strip()
    target = str(reaction.get("message_id") or "").strip()
    out = f"[Reaction: {emoji}"
    if target:
        out += f" to {target}"
    return out + "]" if emoji else ""


# Inbound message type → body text extractor; media kinds use the caption.
_BODY_BY_KIND = {
    "text": lambda m: (m.get("text") or {}).get("body"),
    "button": lambda m: (m.get("button") or {}).get("text"),
    "interactive": lambda m: _interactive_inner(m).get("title"),
    "location": _location_body,
    "reaction": _reaction_body,
    **{kind: (lambda m, k=kind: (m.get(k) or {}).get("caption")) for kind in _INBOUND_MEDIA_KINDS},
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
    """Mime → on-disk extension: WhatsApp overrides → mimetypes → None (never the shared default table)."""
    return ext_for_mime(mime, overrides=_WHATSAPP_MIME_EXTENSION_OVERRIDES, use_defaults=False, use_mimetypes=True, fallback=None) if mime else None


def _cloud_allow_all_opted_in() -> bool:
    return str(_get_wsecret("WHATSAPP_CLOUD_ALLOW_ALL_USERS", default="") or "").strip().lower() in _OPTIN_TRUTHY


def _reply_to_from(metadata: Optional[Dict[str, Any]]) -> Optional[str]:
    return (metadata or {}).get("reply_to_message_id")


def _optional_module(name: str, unavailable_log: str) -> Any:
    """Lazy-import a resolver module (tools.* may be absent in slim installs); None + warning if missing."""
    try:
        return importlib.import_module(name)
    except ImportError:
        logger.warning(unavailable_log)
        return None


# Under the hermes dir so it survives restarts/reloads — same as the Baileys bridge.
_INBOUND_MEDIA_CACHE = Path(get_hermes_dir("platforms/whatsapp_cloud/media", "whatsapp_cloud/media"))


def check_whatsapp_cloud_requirements() -> bool:
    """aiohttp (webhook server) + httpx (Graph API) — both default deps."""
    return AIOHTTP_AVAILABLE and HTTPX_AVAILABLE


class WhatsAppCloudAdapter(WhatsAppBehaviorMixin, BasePlatformAdapter):
    """Outbound: Graph ``/<api_version>/<phone_id>/messages``; inbound: aiohttp webhook
    server. The mixin comes first so its ``format_message`` overrides the base one."""

    splits_long_messages = True  # send() chunks via truncate_message()

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.WHATSAPP_CLOUD)
        extra = config.extra or {}
        self._phone_number_id, self._access_token, self._app_id, self._app_secret, self._waba_id, self._verify_token = (
            str(extra.get(key, "")).strip()
            for key in ("phone_number_id", "access_token", "app_id", "app_secret", "waba_id", "verify_token")
        )
        # Falsy host (None/"") collapses to the dual-stack default.
        self._webhook_host: Optional[str] = str(extra.get("webhook_host") or DEFAULT_WEBHOOK_HOST or "") or None
        self._webhook_port: int = int(extra.get("webhook_port", DEFAULT_WEBHOOK_PORT))
        self._webhook_path: str = self._normalize_path(extra.get("webhook_path", DEFAULT_WEBHOOK_PATH))
        self._health_path: str = self._normalize_path(extra.get("health_path", "/health"))
        self._api_version: str = str(extra.get("api_version", DEFAULT_API_VERSION))
        # Behavior-mixin contract attributes. WHATSAPP_CLOUD_* env vars take precedence so
        # both adapters can run in parallel with independent policies; shared WHATSAPP_*
        # names remain the fallback. Allowlist: config, then legacy ALLOW_FROM, then ALLOWED_USERS.
        self._reply_prefix: Optional[str] = extra.get("reply_prefix")
        allow_raw = self._select_dm_allowlist(extra, ("WHATSAPP_CLOUD_ALLOW_FROM", "WHATSAPP_CLOUD_ALLOWED_USERS"), _get_wsecret)
        self._allow_from: set[str] = self._normalize_allow_ids(self._coerce_allow_list(allow_raw))
        # DM policy default: "open" if the operator opted into allow-all, else
        # "allowlist" when one is configured (so it is enforced), else "open".
        default_dm_policy = "allowlist" if self._allow_from and not _cloud_allow_all_opted_in() else "open"
        self._dm_policy: str = str(
            extra.get("dm_policy") or _get_wsecret("WHATSAPP_CLOUD_DM_POLICY")
            or _get_wsecret("WHATSAPP_DM_POLICY") or default_dm_policy
        ).strip().lower()
        self._group_policy: str = str(
            extra.get("group_policy") or _get_wsecret("WHATSAPP_CLOUD_GROUP_POLICY")
            or _get_wsecret("WHATSAPP_GROUP_POLICY", default="open") or "open"
        ).strip().lower()
        self._group_allow_from: set[str] = self._normalize_allow_ids(self._coerce_allow_list(
            extra.get("group_allow_from") or extra.get("groupAllowFrom") or _get_wsecret("WHATSAPP_CLOUD_GROUP_ALLOW_FROM")
        ))
        self._mention_patterns = self._compile_mention_patterns()
        # Webhook dedup state (in-memory, FIFO-evicted) and counters.
        self._seen_wamids: "OrderedDict[str, bool]" = OrderedDict()
        self._duplicate_count = self._accepted_count = self._rejected_signature_count = 0
        self._warned_no_ffmpeg: bool = False
        # Latest inbound wamid per chat: Meta's typing/read-receipt API needs a
        # message_id to attach to, and the base send_typing contract has none.
        self._last_inbound_wamid_by_chat: "OrderedDict[str, str]" = OrderedDict()
        # Interactive-button state: short id (in the button payload) → session_key for
        # the gateway resolver. Popped on tap; FIFO-capped via _bounded_put so ignored
        # prompts don't accumulate (an evicted tap degrades to text fallback).
        self._clarify_state: "OrderedDict[str, str]" = OrderedDict()
        self._exec_approval_state: "OrderedDict[str, str]" = OrderedDict()
        self._slash_confirm_state: "OrderedDict[str, str]" = OrderedDict()
        self._runner = self._http_client = None

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _normalize_path(path: Any) -> str:
        raw = str(path or "").strip() or "/"
        return raw if raw.startswith("/") else f"/{raw}"

    def _graph_url(self, path: str) -> str:
        """Build a Graph API URL for this adapter's phone-number scope."""
        return f"{GRAPH_API_BASE}/{self._api_version}/{self._phone_number_id}/{path.removeprefix('/')}"

    def _auth_headers(self, *, json_body: bool = True) -> Dict[str, str]:
        headers = {"Authorization": f"Bearer {self._access_token}"}
        return {**headers, "Content-Type": "application/json"} if json_body else headers

    @staticmethod
    def _bounded_put(cache: "OrderedDict[str, Any]", key: str, value: Any, cap: int = INTERACTIVE_STATE_CACHE_SIZE) -> None:
        """Insert into a FIFO-capped OrderedDict, evicting oldest entries."""
        cache[key] = value
        while len(cache) > cap:
            cache.popitem(last=False)

    def _effective_reply_prefix(self) -> str:
        """Cloud API has no self-chat concept (a Baileys-only setting) — no default prefix."""
        return self._reply_prefix.replace("\\n", "\n") if self._reply_prefix is not None else ""

    @staticmethod
    def _normalize_allow_ids(ids: set[str]) -> set[str]:
        """Normalize allowlist entries to bare wa_id (digits): strip ``@...`` JID suffixes and non-digits."""
        return {re.sub(r"\D", "", entry.split("@", 1)[0]) or entry for entry in ids}

    def _is_dm_allowed(self, sender_id: str) -> bool:
        """Allowlist check against the normalized bare wa_id."""
        if self._dm_policy == "allowlist":
            bare = re.sub(r"\D", "", str(sender_id).split("@", 1)[0])
            return (bare or sender_id) in self._normalize_allow_ids(self._live_dm_allow_from())
        return super()._is_dm_allowed(sender_id)

    def _open_dm_opted_in(self) -> bool:
        """Also honor the documented WHATSAPP_CLOUD_ALLOW_ALL_USERS opt-in."""
        return _cloud_allow_all_opted_in() or super()._open_dm_opted_in()

    # ------------------------------------------------------------------ lifecycle
    async def connect(self, *, is_reconnect: bool = False) -> bool:
        for ok, code, message in (
            (check_whatsapp_cloud_requirements(), "whatsapp_cloud_deps_missing",
             "aiohttp and httpx are required for whatsapp_cloud — reinstall hermes-agent."),
            (self._phone_number_id and self._access_token, "whatsapp_cloud_unconfigured",
             "WHATSAPP_CLOUD_PHONE_NUMBER_ID and WHATSAPP_CLOUD_ACCESS_TOKEN are required."),
        ):
            if not ok:
                self._set_fatal_error(code, message, retryable=False)
                return False
        # Tighter keepalive so idle CLOSE_WAIT drains promptly.
        # Outbound HTTP client. See #18451.
        from gateway.platforms._http_client_limits import platform_httpx_limits
        self._http_client = httpx.AsyncClient(timeout=30.0, limits=platform_httpx_limits())
        # client_max_size backstops the bounded reader in _handle_webhook.
        # Inbound webhook server. client_max_size backstops the bounded reader in _handle_webhook — aiohttp
        # enforces the cap on request.read()/post() paths too (#58536/#58902/#59180 pattern).
        app = web.Application(client_max_size=WEBHOOK_MAX_BODY_BYTES)
        app.router.add_get(self._health_path, self._handle_health)
        app.router.add_get(self._webhook_path, self._handle_verify)
        app.router.add_post(self._webhook_path, self._handle_webhook)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        await web.TCPSite(self._runner, self._webhook_host, self._webhook_port).start()
        self._mark_connected()
        logger.info(
            "[whatsapp_cloud] Listening on %s:%d%s (Graph %s, phone_id=%s)",
            self._webhook_host, self._webhook_port, self._webhook_path, self._api_version, self._phone_number_id,
        )
        if not self._verify_token:
            logger.warning("[whatsapp_cloud] WHATSAPP_CLOUD_VERIFY_TOKEN is not set — the GET subscription handshake will fail until it is.")
        if not self._app_secret:
            logger.warning(
                "[whatsapp_cloud] WHATSAPP_CLOUD_APP_SECRET is not set — incoming webhook POSTs will be refused "
                "with 503. Set the app secret to enable inbound message delivery."
            )
        self._wire_plugin_handlers(None)
        return True

    async def disconnect(self) -> None:
        for attr, close, what in (("_runner", "cleanup", "webhook server cleanup"), ("_http_client", "aclose", "http client close")):
            obj = getattr(self, attr)
            if obj is None:
                continue
            try:
                await getattr(obj, close)()
            except Exception:
                logger.exception("[whatsapp_cloud] %s failed", what)
            setattr(self, attr, None)
        self._mark_disconnected()

    # ------------------------------------------------------------------ outbound
    @staticmethod
    def _response_error(resp: Any) -> str:
        """Map a non-200 Graph response to an actionable error string."""
        try:
            body = resp.json()
        except Exception:
            body = {"raw": resp.text[:500]}
        # Graph shape: {"error": {"message", "type", "code", "fbtrace_id"}}
        err = (body or {}).get("error") or {}
        message = err.get("message") or body.get("raw") or "unknown error"
        code = err.get("code")
        return f"graph error {code} (HTTP {resp.status_code}): {message}" if code is not None else f"HTTP {resp.status_code}: {message}"

    async def _post_messages(self, payload: Dict[str, Any], *, fail_log: str, reject_log: str, reject_args: tuple = ()) -> tuple[list, Optional[str]]:
        """POST one /messages payload. Returns ``(response messages[], None)`` or ``([], error)``."""
        try:
            resp = await self._http_client.post(self._graph_url("messages"), headers=self._auth_headers(), json=payload)
        except Exception as exc:
            logger.exception(fail_log)
            return [], str(exc) or type(exc).__name__
        if resp.status_code != 200:
            error_msg = self._response_error(resp)
            logger.warning(reject_log, resp.status_code, *reject_args, error_msg)
            return [], error_msg
        try:
            return resp.json().get("messages") or [], None
        except Exception:
            return [], None

    async def _post_message_result(self, payload: Dict[str, Any], **log_kwargs) -> SendResult:
        """``_post_messages`` → SendResult with the first returned message id; guards disconnected state."""
        if self._http_client is None:
            return SendResult(success=False, error="Not connected")
        ids, err = await self._post_messages(payload, **log_kwargs)
        return SendResult(success=False, error=err) if err is not None else SendResult(success=True, message_id=ids[0].get("id") if ids else None)

    @staticmethod
    def _outbound_payload(chat_id: str, kind: str, block: Any, reply_to: Optional[str]) -> Dict[str, Any]:
        """Common ``/messages`` envelope; ``context`` quotes ``reply_to`` when given."""
        payload: Dict[str, Any] = {"messaging_product": "whatsapp", "recipient_type": "individual", "to": chat_id, "type": kind, kind: block}
        if reply_to:
            payload["context"] = {"message_id": reply_to}
        return payload

    async def send(self, chat_id: str, content: str, reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        """Send a text message via Graph API. ``chat_id`` is the recipient's ``wa_id``."""
        if self._http_client is None:
            return SendResult(success=False, error="Not connected")
        if not content or not content.strip():
            return SendResult(success=True, message_id=None)
        formatted = self.format_message(content)
        last_message_id: Optional[str] = None
        for idx, chunk in enumerate(self.truncate_message(formatted, self._outgoing_chunk_limit())):
            # Quote the user's message on the first chunk only.
            payload = self._outbound_payload(
                chat_id, "text", {"body": chunk, "preview_url": True}, reply_to if idx == 0 else None
            )
            ids, err = await self._post_messages(
                payload,
                fail_log="[whatsapp_cloud] send failed",
                reject_log="[whatsapp_cloud] send rejected (status=%d): %s",
            )
            if err is not None:
                return SendResult(success=False, error=err)
            last_message_id = ids[0].get("id") if ids else last_message_id
        # Index (chat_id, wamid) → text: Meta's inbound ``context`` carries only the
        # quoted message's id, so this is how replies to our messages resolve text.
        if last_message_id:
            rich_sent_store.record(chat_id, last_message_id, formatted)
        return SendResult(success=True, message_id=last_message_id)

    # ------------------------------------------------------------------ typing indicator + read receipts
    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """Mark the latest inbound message read AND show a typing indicator. Meta couples
        both into one POST; the indicator auto-dismisses on reply or after 25s. Best-effort:
        every error is swallowed so the main reply path is never blocked."""
        wamid = self._last_inbound_wamid_by_chat.get(chat_id)
        if self._http_client is None or not wamid:
            return  # not connected, or no inbound yet for this chat (cache cleared on restart)
        payload = {"messaging_product": "whatsapp", "status": "read", "message_id": wamid, "typing_indicator": {"type": "text"}}
        try:
            resp = await self._http_client.post(self._graph_url("messages"), headers=self._auth_headers(), json=payload)
        except Exception:
            return
        if resp.status_code == 200:
            return
        try:
            code = ((resp.json() or {}).get("error") or {}).get("code")
        except Exception:
            code = None
        # 131009 = "Parameter value is not valid" (typically wamid > 30 days old) — common
        # after a long-quiet conversation, so log at info.
        if code == 131009:
            logger.info("[whatsapp_cloud] typing/read indicator rejected: wamid %s likely older than 30 days", wamid)
        else:
            logger.debug("[whatsapp_cloud] typing/read indicator returned %d (%s)", resp.status_code, code)

    # ------------------------------------------------------------------ interactive messages
    async def _send_interactive(
        self, chat_id: str, interactive: Dict[str, Any], metadata: Optional[Dict[str, Any]],
        state: "OrderedDict[str, str]", state_id: str, session_key: str,
    ) -> SendResult:
        """POST an ``interactive`` message (caller supplies ``type``/``body``/``action``) and, on
        success, remember ``state_id → session_key`` for the tap. Free-form interactives need no
        Meta approval but are only valid inside the 24h window — fine, all senders here reply to a user."""
        result = await self._post_message_result(
            self._outbound_payload(chat_id, "interactive", interactive, _reply_to_from(metadata)),
            fail_log="[whatsapp_cloud] interactive send failed",
            reject_log="[whatsapp_cloud] interactive rejected (status=%d): %s",
        )
        if result.success:
            self._bounded_put(state, state_id, session_key)
        return result

    @staticmethod
    def _truncate_button_label(text: str, limit: int = 20) -> str:
        """Button titles cap at 20 chars, list-row titles at 24; the ellipsis counts."""
        text = str(text or "").strip()
        return text if len(text) <= limit else text[: max(1, limit - 1)] + "…"

    @staticmethod
    def _truncate_body(text: str, limit: int = 1024) -> str:
        """``interactive.body.text`` caps at 1024 chars."""
        text = str(text or "")
        return text if len(text) <= limit else text[: limit - 3] + "..."

    @staticmethod
    def _reply_buttons(*buttons: tuple[str, str]) -> list[Dict[str, Any]]:
        return [{"type": "reply", "reply": {"id": bid, "title": title}} for bid, title in buttons]

    @classmethod
    def _button_interactive(cls, body_text: str, *buttons: tuple[str, str]) -> Dict[str, Any]:
        """``interactive.type=button`` body: ≤3 quick-reply buttons (id ≤256 chars, title ≤20)."""
        return {"type": "button", "body": {"text": body_text}, "action": {"buttons": cls._reply_buttons(*buttons)}}

    async def send_clarify(
        self, chat_id: str, question: str, choices: Optional[list], clarify_id: str,
        session_key: str, metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Clarify as native buttons: 1–3 choices → buttons, 4+ → list (+ "Other" row that flips
        the entry into text-capture mode), 0 → plain text question. Button ``id`` carries
        ``cl:<clarify_id>:<idx|other>``; inbound dispatches on it."""
        question = (question or "").strip()
        if not choices:
            return await self.send(chat_id, f"❓ {question}", reply_to=_reply_to_from(metadata))
        # Full choice text goes in the body so long options aren't lost to the
        # 20-char label cap; labels are just the option number.
        choices_list = [str(c).strip() for c in choices[:10] if str(c).strip()]
        body_text = self._truncate_body(f"❓ {question}\n\n" + "\n".join(f"{i + 1}. {c}" for i, c in enumerate(choices_list)))
        if len(choices_list) <= 3:
            interactive = self._button_interactive(
                body_text, *((f"cl:{clarify_id}:{idx}", self._truncate_button_label(str(idx + 1))) for idx in range(len(choices_list)))
            )
        else:
            # List rows: id + title (≤24) + description (≤72) with the choice text.
            rows = [
                {"id": f"cl:{clarify_id}:{idx}", "title": self._truncate_button_label(f"{idx + 1}", limit=24),
                 "description": self._truncate_button_label(choice_text, limit=72)}
                for idx, choice_text in enumerate(choices_list)
            ]
            rows.append({"id": f"cl:{clarify_id}:other", "title": "✏️ Other", "description": "Type your own answer"})
            interactive = {"type": "list", "body": {"text": body_text}, "action": {"button": "Choose", "sections": [{"title": "Options", "rows": rows}]}}
        return await self._send_interactive(chat_id, interactive, metadata, self._clarify_state, clarify_id, session_key)

    async def send_exec_approval(
        self, chat_id: str, command: str, session_key: str, description: str = "dangerous command",
        metadata: Optional[Dict[str, Any]] = None, allow_permanent: bool = True,
        allow_session: bool = True, smart_denied: bool = False,
    ) -> SendResult:
        """Approve / Deny buttons; a tap resolves via ``tools.approval.resolve_gateway_approval``."""
        del allow_permanent, allow_session  # This adapter already offers one-shot Approve / Deny only.
        # Body caps at 1024; reserve room for the framing prose.
        cmd_preview = (command or "")[:800] + ("..." if len(command or "") > 800 else "")
        body_text = self._truncate_body(
            f"⚠️ *Command Approval Required*\n\n```\n{cmd_preview}\n```\n\nReason: {description}"
            + ("\n\nSmart DENY: owner override applies to this one operation only." if smart_denied else "")
        )
        approval_id = uuid.uuid4().hex[:12]
        interactive = self._button_interactive(body_text, (f"appr:{approval_id}:approve", "✅ Approve"), (f"appr:{approval_id}:deny", "❌ Deny"))
        return await self._send_interactive(chat_id, interactive, metadata, self._exec_approval_state, approval_id, session_key)

    async def send_slash_confirm(
        self, chat_id: str, title: str, message: str, session_key: str, confirm_id: str, metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Approve Once / Always / Cancel buttons; ``confirm_id`` is caller-supplied."""
        interactive = self._button_interactive(
            self._truncate_body(f"*{title}*\n\n{message}"), (f"sc:once:{confirm_id}", "✅ Approve Once"),
            (f"sc:always:{confirm_id}", "🔒 Always"), (f"sc:cancel:{confirm_id}", "❌ Cancel"),
        )
        return await self._send_interactive(chat_id, interactive, metadata, self._slash_confirm_state, confirm_id, session_key)

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        # No chat-info endpoint; profile name arrives via webhook contacts[].
        return {"name": chat_id, "type": "dm"}

    # ------------------------------------------------------------------ outbound media
    async def _upload_media(self, file_path: str, media_kind: str, mime_type: Optional[str] = None) -> tuple[Optional[str], Optional[str]]:
        """Upload a local file to Graph /media. Returns ``(media_id, None)`` or ``(None, error)``."""
        if self._http_client is None:
            return None, "Not connected"
        if not os.path.exists(file_path):
            return None, f"File not found: {file_path}"
        size, cap = os.path.getsize(file_path), _MEDIA_SIZE_LIMITS.get(media_kind, _MEDIA_SIZE_LIMITS["document"])
        if size > cap:
            return None, f"File {os.path.basename(file_path)} is {size} bytes; Cloud API {media_kind} cap is {cap} bytes"
        mime_type = mime_type or mimetypes.guess_type(file_path)[0] or _DEFAULT_MIME.get(media_kind, "application/octet-stream")
        try:
            with open(file_path, "rb") as fh:
                files = {"file": (os.path.basename(file_path), fh, mime_type), "messaging_product": (None, "whatsapp"), "type": (None, mime_type)}
                resp = await self._http_client.post(self._graph_url("media"), headers=self._auth_headers(json_body=False), files=files)
        except Exception as exc:
            logger.exception("[whatsapp_cloud] media upload failed")
            return None, str(exc)
        if resp.status_code != 200:
            return None, self._response_error(resp)
        try:
            media_id = resp.json().get("id")
        except Exception:
            media_id = None
        return (media_id, None) if media_id else (None, "Upload response missing 'id'")

    async def _send_media(
        self, chat_id: str, media_kind: str, *, media_id: Optional[str] = None,
        media_link: Optional[str] = None, caption: Optional[str] = None,
        filename: Optional[str] = None, reply_to: Optional[str] = None,
    ) -> SendResult:
        """POST a media message referencing exactly one of an uploaded ``media_id`` or a public
        ``link``. Caption is accepted on image/video/document; filename on document only."""
        if self._http_client is None:
            return SendResult(success=False, error="Not connected")
        if bool(media_id) == bool(media_link):
            return SendResult(success=False, error="Exactly one of media_id or media_link must be set")
        media_block: Dict[str, Any] = {"id": media_id} if media_id else {"link": media_link}
        if caption and media_kind in {"image", "video", "document"}:
            media_block["caption"] = caption
        if filename and media_kind == "document":
            media_block["filename"] = filename
        return await self._post_message_result(
            self._outbound_payload(chat_id, media_kind, media_block, reply_to),
            fail_log="[whatsapp_cloud] media send failed",
            reject_log="[whatsapp_cloud] media send rejected (status=%d, kind=%s): %s",
            reject_args=(media_kind,),
        )

    async def _send_media_from_path_or_link(
        self, chat_id: str, source: str, media_kind: str, *, caption: Optional[str] = None,
        filename: Optional[str] = None, reply_to: Optional[str] = None,
        mime_type: Optional[str] = None,
    ) -> SendResult:
        """HTTPS URL → ``link`` send (one fewer round trip); local path → upload + ``id`` send."""
        ref: Dict[str, Optional[str]] = {"media_link": source}
        if not source.startswith(_HTTP_PREFIXES):
            media_id, err = await self._upload_media(source, media_kind, mime_type)
            if err:
                return SendResult(success=False, error=err)
            ref = {"media_id": media_id}
        return await self._send_media(chat_id, media_kind, caption=caption, filename=filename, reply_to=reply_to, **ref)

    # ``**kwargs`` absorbs base-class args (e.g. ``metadata``) the Cloud API has no use for.
    async def send_image(self, chat_id: str, image_url: str, caption: Optional[str] = None, reply_to: Optional[str] = None, **kwargs) -> SendResult:
        return await self._send_media_from_path_or_link(chat_id, image_url, "image", caption=caption, reply_to=reply_to)

    async def send_image_file(self, chat_id: str, image_path: str, caption: Optional[str] = None, reply_to: Optional[str] = None, **kwargs) -> SendResult:
        return await self._send_media_from_path_or_link(chat_id, image_path, "image", caption=caption, reply_to=reply_to)

    async def send_video(self, chat_id: str, video_path: str, caption: Optional[str] = None, reply_to: Optional[str] = None, **kwargs) -> SendResult:
        return await self._send_media_from_path_or_link(chat_id, video_path, "video", caption=caption, reply_to=reply_to)

    async def send_document(
        self, chat_id: str, file_path: str, caption: Optional[str] = None, file_name: Optional[str] = None, reply_to: Optional[str] = None, **kwargs,
    ) -> SendResult:
        return await self._send_media_from_path_or_link(
            chat_id, file_path, "document", caption=caption, filename=file_name or os.path.basename(file_path), reply_to=reply_to,
        )

    async def send_voice(self, chat_id: str, audio_path: str, caption: Optional[str] = None, reply_to: Optional[str] = None, **kwargs) -> SendResult:
        """Voice message: ``audio/ogg; codecs=opus`` renders as a voice bubble, so a
        local MP3 (Hermes TTS output) is converted via ffmpeg first; other audio is sent as-is."""
        mime_type: Optional[str] = None
        if not audio_path.startswith(_HTTP_PREFIXES) and audio_path.lower().endswith(".mp3") and os.path.exists(audio_path):
            opus_path = await self._convert_to_opus(audio_path)
            if opus_path:
                try:
                    return await self._send_media_from_path_or_link(
                        chat_id, opus_path, "audio", caption=caption, reply_to=reply_to, mime_type="audio/ogg; codecs=opus",
                    )
                finally:
                    with contextlib.suppress(OSError):  # the .ogg is a transient artifact next to the source MP3
                        os.unlink(opus_path)
            # No ffmpeg (warn-once logged in _convert_to_opus) → MP3 attachment.
            mime_type = "audio/mpeg"
        return await self._send_media_from_path_or_link(chat_id, audio_path, "audio", caption=caption, reply_to=reply_to, mime_type=mime_type)

    async def _convert_to_opus(self, mp3_path: str) -> Optional[str]:
        """MP3 → ``audio/ogg; codecs=opus``; None if ffmpeg is missing or fails. ``-application voip``
        tunes for speech; ``-b:a 32k -vbr on`` matches WhatsApp's native voice-note bitrate."""
        if not _FFMPEG_PATH:
            if not self._warned_no_ffmpeg:
                self._warned_no_ffmpeg = True
                logger.warning(
                    "[whatsapp_cloud] ffmpeg not found on PATH — voice messages will be delivered as MP3 audio "
                    "attachments instead of native voice notes (green waveform bubble). Install ffmpeg to enable: "
                    "Windows `winget install Gyan.FFmpeg`, macOS `brew install ffmpeg`, Linux package manager."
                )
            return None
        out_path = mp3_path.rsplit(".", 1)[0] + ".ogg"
        try:
            proc = await asyncio.create_subprocess_exec(
                _FFMPEG_PATH, "-y", "-i", mp3_path, "-c:a", "libopus", "-b:a", "32k", "-vbr", "on", "-application", "voip",
                out_path, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
        except Exception:
            logger.exception("[whatsapp_cloud] ffmpeg subprocess raised")
            return None
        if proc.returncode == 0 and Path(out_path).exists():
            return out_path
        logger.error(
            "[whatsapp_cloud] ffmpeg opus conversion failed (returncode=%s): %s",
            proc.returncode, (stderr or b"").decode("utf-8", errors="replace")[:500],
        )
        return None

    # ------------------------------------------------------------------ inbound media
    async def _graph_get(self, url: str, headers: Dict[str, str], what: str, media_id: str) -> Any:
        """GET with the download path's uniform failure logging; None on exception or non-200."""
        try:
            resp = await self._http_client.get(url, headers=headers)
        except Exception:
            logger.exception("[whatsapp_cloud] media %s fetch raised (id=%s)", what, media_id)
            return None
        if resp.status_code != 200:
            logger.warning("[whatsapp_cloud] media %s fetch failed (id=%s, status=%d)", what, media_id, resp.status_code)
            return None
        return resp

    async def _download_media_to_cache(self, media_id: str, *, ext_hint: Optional[str] = None) -> tuple[Optional[str], Optional[str]]:
        """Two-step Graph download: ``GET /<id>`` → signed temp URL (~5 min) → bytes.
        Returns ``(local_path, mime_type)`` or ``(None, None)`` on any failure (logged)."""
        if self._http_client is None:
            return None, None
        # media_id is interpolated into a Graph URL and a cache filename — refuse
        # anything that isn't a plain Meta-style id so a hostile payload can't traverse.
        media_id = str(media_id).strip()
        if not re.fullmatch(r"[A-Za-z0-9._-]+", media_id):
            logger.warning("[whatsapp_cloud] refusing malformed media id %r", media_id[:64])
            return None, None
        headers = self._auth_headers(json_body=False)
        meta_resp = await self._graph_get(f"{GRAPH_API_BASE}/{self._api_version}/{media_id}", headers, "metadata", media_id)
        if meta_resp is None:
            return None, None
        try:
            meta = meta_resp.json()
        except Exception:
            meta = {}
        temp_url, mime = meta.get("url"), meta.get("mime_type") or ""
        if not temp_url:
            return None, None
        # Auth is required even though the URL is signed (Meta documents this).
        blob_resp = await self._graph_get(temp_url, headers, "bytes", media_id)
        if blob_resp is None:
            return None, None
        _INBOUND_MEDIA_CACHE.mkdir(parents=True, exist_ok=True)
        out_path = _INBOUND_MEDIA_CACHE / f"{media_id}{ext_hint or _ext_for_mime(mime) or '.bin'}"
        try:
            out_path.write_bytes(blob_resp.content)
        except OSError:
            logger.exception("[whatsapp_cloud] failed to write cached media (id=%s)", media_id)
            return None, None
        return str(out_path), mime or None

    # ------------------------------------------------------------------ inbound
    async def _handle_health(self, request: "web.Request") -> "web.Response":
        return web.json_response({
            "status": "ok", "platform": self.platform.value, "phone_number_id": self._phone_number_id,
            "webhook_path": self._webhook_path, "verify_token_configured": bool(self._verify_token),
            "app_secret_configured": bool(self._app_secret), "ffmpeg_present": _FFMPEG_PATH is not None,
            "accepted": self._accepted_count, "duplicates": self._duplicate_count,
            "rejected_signature": self._rejected_signature_count,
        })

    async def _handle_verify(self, request: "web.Request") -> "web.Response":
        """Meta subscription handshake: echo ``hub.challenge`` iff mode is
        ``subscribe`` and ``hub.verify_token`` matches (constant-time)."""
        if not self._verify_token:
            # Refuse rather than accept any token, which would let an attacker subscribe.
            return web.Response(status=503, text="verify_token not configured")
        q = request.query
        if q.get("hub.mode", "") != "subscribe":
            return web.Response(status=400, text="bad mode")
        # Compare as bytes: compare_digest raises TypeError on non-ASCII str.
        if not hmac.compare_digest(q.get("hub.verify_token", "").encode(), self._verify_token.encode()):
            return web.Response(status=403, text="verify_token mismatch")
        if not q.get("hub.challenge", ""):
            return web.Response(status=400, text="missing challenge")
        return web.Response(text=q["hub.challenge"], content_type="text/plain")

    async def _handle_webhook(self, request: "web.Request") -> "web.Response":
        """Inbound webhook POST: raw bytes → HMAC verify → JSON → dispatch. Signature is over
        the raw body, so JSON parsing must come after verification. Always 200 once a valid
        request is ack'd — Meta retries non-200 for up to 7 days and would multiply agent work."""
        # Read one byte past Meta's 3MB max so oversized chunked bodies are rejected before buffering.
        try:
            raw = await _read_limited_request_body(request, WEBHOOK_MAX_BODY_BYTES)
        except ValueError:
            return web.Response(status=413)
        except Exception:
            return web.Response(status=400)
        # Without app_secret the sender can't be authenticated → refuse (same
        # posture as the GET handshake refusing without verify_token).
        if not self._app_secret:
            logger.error("[whatsapp_cloud] webhook POST refused: app_secret unset. Set WHATSAPP_CLOUD_APP_SECRET to enable inbound delivery.")
            return web.Response(status=503, text="app_secret not configured")
        signature_header = request.headers.get("X-Hub-Signature-256", "")
        if not self._verify_signature(raw, signature_header):
            self._rejected_signature_count += 1
            logger.warning("[whatsapp_cloud] rejected webhook: invalid X-Hub-Signature-256 (header=%r, body_len=%d)", signature_header, len(raw))
            return web.Response(status=401)
        try:
            payload = json.loads(raw)
        except Exception:
            logger.warning("[whatsapp_cloud] webhook body is not valid JSON")
            payload = None
        if not isinstance(payload, dict):
            return web.Response(status=400)
        await self._dispatch_payload(payload)
        return web.Response(status=200)

    def _verify_signature(self, raw_body: bytes, header: str) -> bool:
        """Verify ``sha256=<hex>`` HMAC of the raw body keyed by ``app_secret`` (constant-time)."""
        if not (self._app_secret and header and header.startswith("sha256=") and (expected_hex := header[7:].strip())):
            return False
        computed = hmac.new(self._app_secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()
        # Compare as bytes: compare_digest raises TypeError on non-ASCII str.
        return hmac.compare_digest(computed.lower().encode(), expected_hex.lower().encode())

    # ------------------------------------------------------------------ dispatch
    def _dedup_wamid(self, wamid: str) -> bool:
        """True if this wamid is new; False (and count a duplicate) if already seen."""
        if wamid in self._seen_wamids:
            self._duplicate_count += 1
            return False
        if wamid:  # can't dedup without an id — let it through
            self._bounded_put(self._seen_wamids, wamid, True, cap=WAMID_DEDUP_CACHE_SIZE)
        return True

    async def _dispatch_payload(self, payload: Dict[str, Any]) -> None:
        """Walk ``entry[].changes[].value.{messages, contacts, statuses}`` and dispatch each message.
        ``statuses`` (sent/delivered/read/failed) are only logged — the agent doesn't consume receipts."""
        if payload.get("object") != "whatsapp_business_account":
            logger.debug("[whatsapp_cloud] ignoring non-WABA payload (object=%r)", payload.get("object"))
            return
        for entry in payload.get("entry") or []:
            for change in (entry.get("changes") or []) if isinstance(entry, dict) else []:
                if not isinstance(change, dict) or change.get("field") != "messages":
                    continue  # account_alerts, template_status_update, … — not message ingress
                value = change.get("value") or {}
                contacts_by_waid = {
                    wa_id: str((contact.get("profile") or {}).get("name") or "").strip()
                    for contact in value.get("contacts") or []
                    if isinstance(contact, dict) and (wa_id := str(contact.get("wa_id") or "").strip())
                }
                for raw_message in value.get("messages") or []:
                    if isinstance(raw_message, dict):
                        await self._ingest_message(raw_message, contacts_by_waid, value.get("metadata") or {})
                for status in value.get("statuses") or []:
                    if isinstance(status, dict):
                        logger.debug("[whatsapp_cloud] status %s for %s", status.get("status"), status.get("id"))

    async def _ingest_message(self, raw_message: Dict[str, Any], contacts_by_waid: Dict[str, str], metadata: Dict[str, Any]) -> None:
        """Dedup → build event → handle_message. Neither build nor dispatch errors may
        bubble out: the wamid is already dedup-marked, so a 500 would make Meta retry
        the batch and every message in it would then be dropped as a duplicate."""
        wamid = str(raw_message.get("id") or "").strip()
        if not self._dedup_wamid(wamid):
            logger.debug("[whatsapp_cloud] duplicate wamid %s, skipping", wamid)
            return
        try:
            event = await self._build_message_event_from_cloud(raw_message, contacts_by_waid, metadata)
        except Exception:
            logger.exception("[whatsapp_cloud] failed to build event for wamid %s", wamid)
            return
        if event is None:
            return
        self._accepted_count += 1
        try:
            await self.handle_message(event)
        except Exception:
            logger.exception("[whatsapp_cloud] handle_message raised for wamid %s", wamid)

    async def _dispatch_interactive_reply(self, raw_message: Dict[str, Any], contacts_by_waid: Dict[str, str]) -> bool:
        """Route an inbound button tap to its resolver (see ``_INTERACTIVE_HANDLERS``). True = claimed.
        False (unknown prefix, no live state, or no waiter) makes the caller fall back to text
        dispatch with the button title — covers stale taps."""
        inner = _interactive_inner(raw_message)
        button_id = str(inner.get("id") or "").strip()
        sender_id = str(raw_message.get("from") or "").strip()
        if not button_id:
            return False
        # Taps bypass ``_should_process_message``; re-check the strict DM gate so a stale
        # prompt can't be answered after the sender leaves the allowlist.
        if not (sender_id and self._is_dm_allowed(sender_id)):
            logger.warning("[whatsapp_cloud] Rejected unauthorized interactive tap from %s (button_id=%r)", sender_id or "<unknown>", button_id)
            return True  # claim so the tap isn't re-dispatched as plain text
        parts = button_id.split(":", 2)
        handler = next((h for prefix, h in self._INTERACTIVE_HANDLERS.items() if button_id.startswith(prefix)), None)
        # Unknown prefix (maybe a plugin-defined adapter's) — text dispatch is the safe default.
        if handler is None or len(parts) != 3:
            return False
        return await handler(self, str(raw_message.get("from") or ""), inner, parts)

    async def _reply_best_effort(self, to: str, text: str, fail_log: str) -> None:
        try:
            await self.send(to, text)
        except Exception:
            logger.exception(fail_log)

    @staticmethod
    def _pop_tap_state(
        state: "OrderedDict[str, str]", key: str, stale_log: str, choice: str = "", valid: tuple = (),
    ) -> Optional[str]:
        """Pop the session_key for a tapped prompt. None (info-logged) when nothing is live — likely
        a stale tap; an unrecognised ``choice`` keeps the prompt live and also yields None."""
        session_key = state.pop(key, None)
        if not session_key:
            logger.info(stale_log, key)
        elif valid and choice not in valid:
            state[key] = session_key
            return None
        return session_key

    async def _handle_clarify_tap(self, to: str, inner: Dict[str, Any], parts: list) -> bool:
        _, clarify_id, choice = parts
        session_key = self._pop_tap_state(
            self._clarify_state, clarify_id,
            "[whatsapp_cloud] clarify tap with no matching state (clarify_id=%s) — likely stale; falling back to text",
        )
        if not session_key:
            return False
        clarify_gateway = _optional_module(
            "tools.clarify_gateway", "[whatsapp_cloud] clarify resolver unavailable; falling back to text dispatch"
        )
        if clarify_gateway is None:
            return False
        if choice == "other":
            # Flip the entry into text-capture mode so the gateway's text intercept resolves the
            # clarify with the next message; otherwise that text would hit the agent path while
            # it's still blocked in clarify ("Interrupting current task" loop).
            try:
                flipped = clarify_gateway.mark_awaiting_text(clarify_id)
            except Exception:
                logger.exception("[whatsapp_cloud] mark_awaiting_text failed for %s", clarify_id)
                flipped = False
            if not flipped:
                # Entry vanished (timeout, /new, restart) — fall through to text.
                logger.info("[whatsapp_cloud] clarify 'Other' tap but entry missing (clarify_id=%s); falling back to text", clarify_id)
                return False
            # Keep the mapping live for further taps on the same prompt.
            self._clarify_state[clarify_id] = session_key
            await self._reply_best_effort(to, "✏️ Type your answer:", "[whatsapp_cloud] clarify other-prompt failed")
            return True
        try:
            idx = int(choice)
        except ValueError:
            logger.warning("[whatsapp_cloud] clarify tap had non-int choice: %r", choice)
            self._clarify_state[clarify_id] = session_key  # a follow-up text can still resolve
            return False
        # Title is the numeric label; the agent has the prompt in context to interpret it.
        if not clarify_gateway.resolve_gateway_clarify(clarify_id, str(inner.get("title") or str(idx + 1))):
            logger.info("[whatsapp_cloud] clarify resolver reported no waiter (clarify_id=%s) — falling back to text", clarify_id)
            return False
        return True

    async def _handle_approval_tap(self, to: str, inner: Dict[str, Any], parts: list) -> bool:
        _, approval_id, choice = parts
        session_key = self._pop_tap_state(
            self._exec_approval_state, approval_id,
            "[whatsapp_cloud] approval tap with no matching state (approval_id=%s) — likely stale; falling back to text",
            choice, ("approve", "deny"),
        )
        if not session_key:
            return False
        approval = _optional_module("tools.approval", "[whatsapp_cloud] approval resolver unavailable")
        if approval is None:
            return False
        count = approval.resolve_gateway_approval(session_key, choice)
        # A tap after the wait timed out (count == 0) must not claim approval:
        # the command was already denied fail-closed.
        if count:
            confirm_text = "✅ Approved." if choice == "approve" else "❌ Denied."
        else:
            logger.info("[whatsapp_cloud] approval resolver reported no waiter (session_key=%s) — likely already resolved", session_key)
            confirm_text = "⌛ Approval expired — command was not run (already timed out or resolved elsewhere)."
        await self._reply_best_effort(to, confirm_text, "[whatsapp_cloud] approval confirm failed")
        return True

    async def _handle_slash_confirm_tap(self, to: str, inner: Dict[str, Any], parts: list) -> bool:
        _, choice, confirm_id = parts
        session_key = self._pop_tap_state(
            self._slash_confirm_state, confirm_id,
            "[whatsapp_cloud] slash_confirm tap with no matching state (confirm_id=%s) — likely stale",
            choice, ("once", "always", "cancel"),
        )
        if not session_key:
            return False
        slash_confirm = _optional_module("tools.slash_confirm", "[whatsapp_cloud] slash_confirm resolver unavailable")
        if slash_confirm is None:
            return False
        try:
            result_text = await slash_confirm.resolve(session_key, confirm_id, choice)
        except Exception:
            logger.exception("[whatsapp_cloud] slash_confirm.resolve failed")
            return True  # still claim the tap; surfacing it as text wouldn't help
        if result_text:
            await self._reply_best_effort(to, result_text, "[whatsapp_cloud] slash_confirm reply failed")
        return True

    _INTERACTIVE_HANDLERS = {"cl:": _handle_clarify_tap, "appr:": _handle_approval_tap, "sc:": _handle_slash_confirm_tap}

    async def _collect_inbound_media(self, msg_type_str: str, raw_message: Dict[str, Any], body: str) -> tuple[list[str], list[str], str]:
        """Download inbound media by ``media_id``; returns ``(media_urls, media_types, body)``."""
        inner = raw_message.get(msg_type_str) or {}
        media_id, inbound_mime = str(inner.get("id") or "").strip(), str(inner.get("mime_type") or "").strip()
        if not media_id:
            return [], [], body
        local_path, dl_mime = await self._download_media_to_cache(media_id, ext_hint=_ext_for_mime(inbound_mime))
        if local_path:
            logger.info("[whatsapp_cloud] cached inbound %s media: %s", msg_type_str, local_path)
        else:
            logger.warning(
                "[whatsapp_cloud] failed to download inbound %s (id=%s) — agent will see message metadata but not the binary",
                msg_type_str, media_id,
            )
        fname = str(inner.get("filename") or "").strip() if msg_type_str == "document" else ""
        body = body or (f"[Document: {fname}]" if fname else body)
        if not local_path:
            return [], [], body
        return [local_path], [dl_mime or inbound_mime or "application/octet-stream"], body

    @staticmethod
    def _inject_document_text(media_urls: list[str], body: str) -> str:
        """Prepend text-readable document contents (≤100KB) to the body."""
        for doc in map(Path, media_urls):
            if doc.suffix.lower() not in _TEXT_INJECT_EXTS:
                continue
            try:
                file_size = doc.stat().st_size
                if file_size > _MAX_TEXT_INJECT_BYTES:
                    logger.info("[whatsapp_cloud] skipping text injection for %s (%d bytes > %d)", doc, file_size, _MAX_TEXT_INJECT_BYTES)
                    continue
                injection = f"[Content of {doc.name}]:\n{doc.read_text(encoding='utf-8', errors='replace')}"
                body = f"{injection}\n\n{body}" if body else injection
            except OSError:
                logger.exception("[whatsapp_cloud] failed to read document text: %s", doc)
        return body

    async def _build_message_event_from_cloud(
        self, raw_message: Dict[str, Any], contacts_by_waid: Dict[str, str], metadata: Dict[str, Any],
    ) -> Optional[MessageEvent]:
        """Convert a Cloud-API message object into a MessageEvent, or None if gated out."""
        msg_type_str = str(raw_message.get("type") or "text").lower()
        # Button taps route to the gateway resolver BEFORE text dispatch — the
        # resolver unblocks the waiting agent, so don't also start a fresh turn.
        if msg_type_str == "interactive" and await self._dispatch_interactive_reply(raw_message, contacts_by_waid):
            return None
        extract = _BODY_BY_KIND.get(msg_type_str)
        body = str(extract(raw_message) or "") if extract else ""
        chat_id = sender_id = str(raw_message.get("from") or "").strip()
        sender_name = contacts_by_waid.get(sender_id, "")
        # DMs only: chat_id == sender wa_id. A ``chat`` field marks a group-shaped
        # payload (capability-gated by Meta) — refuse rather than treat as a DM.
        if raw_message.get("chat"):
            logger.warning(
                "[whatsapp_cloud] received group-shaped message (chat=%s, wamid=%s) — group support is not yet "
                "implemented; dropping. Use the Baileys whatsapp adapter for group chats.",
                raw_message.get("chat"), raw_message.get("id"),
            )
            return None
        if not self._should_process_message({"chatId": chat_id, "senderId": sender_id, "isGroup": False, "body": body}):
            return None
        media_urls, media_types = [], []
        if msg_type_str in _INBOUND_MEDIA_KINDS:
            media_urls, media_types, body = await self._collect_inbound_media(msg_type_str, raw_message, body)
            if msg_type_str == "document" and media_urls:
                body = self._inject_document_text(media_urls, body)
        # Meta's ``context`` gives only the quoted message's id (+ author), never its text;
        # resolve from rich_sent_store so run.py can build "[Replying to: ...]".
        context = raw_message.get("context") or {}
        reply_to_id = str(context.get("id") or "").strip() or None
        reply_to_text = rich_sent_store.lookup(chat_id, reply_to_id) if reply_to_id else None
        # context.from == our business number → user replied to the bot.
        quoted_from = str(context.get("from") or "").strip()
        our_number = str(metadata.get("display_phone_number") or "").strip()
        reply_to_is_own = bool(reply_to_id and quoted_from and our_number) and quoted_from == our_number
        wamid = str(raw_message.get("id") or "") or None
        if wamid and chat_id:
            # Done AFTER gating so filtered messages don't leak typing/read receipts.
            self._bounded_put(self._last_inbound_wamid_by_chat, chat_id, wamid)
            if body:
                rich_sent_store.record(chat_id, wamid, body)
        return MessageEvent(
            text=body, message_type=_MESSAGE_TYPE_BY_KIND.get(msg_type_str, MessageType.TEXT),
            source=self.build_source(
                chat_id=chat_id, chat_name=sender_name or chat_id, chat_type="dm",
                user_id=sender_id, user_name=sender_name or None,
            ),
            raw_message=raw_message, message_id=wamid, reply_to_message_id=reply_to_id,
            reply_to_text=reply_to_text, reply_to_is_own_message=reply_to_is_own,
            media_urls=media_urls, media_types=media_types,
        )

"""Chatwoot Agent Bot gateway adapter for Hermes.

Implements the integration described in ``docs/plans/CHATWOOT_INTEGRATION.md``:

- Inbound: Chatwoot POSTs webhook events to this adapter's **own** aiohttp
  listener; ``message_created`` events from the contact are converted to a
  gateway ``MessageEvent`` and dispatched via ``handle_message``.
- Outbound: replies are posted to Chatwoot's Application API, authenticating
  with the Agent Bot token in the ``api_access_token`` header.
- Reasoning / tool + skill activity can be surfaced as agent-only **private
  notes** (``private: true``) when ``CHATWOOT_PRIVATE_NOTE_TRACE`` is on.

The adapter is a plugin: ``register(ctx)`` wires it into Hermes with zero core
edits (aside from the shared non-conversational metadata marker used by the
private-note trace).
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

try:
    import aiohttp
    from aiohttp import web

    AIOHTTP_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dep
    AIOHTTP_AVAILABLE = False
    aiohttp = None  # type: ignore[assignment]
    web = None  # type: ignore[assignment]

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    cache_audio_from_bytes,
    cache_document_from_bytes,
    cache_image_from_bytes,
    is_network_accessible,
)

logger = logging.getLogger(__name__)

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8647
DEFAULT_WEBHOOK_PATH = "/chatwoot/webhook"
DEFAULT_HEALTH_PATH = "/health"
DEFAULT_MAX_BODY_BYTES = 1_000_000  # 1 MB webhook body cap (DoS guard)
DEFAULT_MAX_SEEN_IDS = 5000

# Chatwoot has no hard message length, but very long single messages render
# poorly in the agent inbox — chunk beyond this.
MAX_MESSAGE_LENGTH = 10_000

# conversation.status values; only ``pending`` is the bot's to answer.
_BOT_OWNED_STATUS = "pending"


# ── module-level helpers ─────────────────────────────────────────────────────


def check_chatwoot_requirements() -> bool:
    """Return whether required dependencies are available (only aiohttp)."""
    return AIOHTTP_AVAILABLE


def _bool_env(value: Any) -> bool:
    return str(value or "").strip().lower() in ("1", "true", "yes", "on")


def _redact(token: Optional[str]) -> str:
    """Mask a token for safe logging (keep a short suffix for correlation)."""
    if not token:
        return "<none>"
    tok = str(token)
    if len(tok) <= 6:
        return "***"
    return f"***{tok[-4:]}"


def _format_chat_id(account_id: Any, conversation_id: Any) -> str:
    """Encode account + conversation into a single opaque chat id."""
    return f"{account_id}:{conversation_id}"


def _parse_chat_id(chat_id: str, default_account: Optional[str] = None) -> Tuple[str, str]:
    """Parse ``account:conversation`` into ``(account, conversation)``.

    A bare conversation id (no colon) falls back to ``default_account``.
    Raises ``ValueError`` if no account can be resolved.
    """
    raw = str(chat_id or "").strip()
    if ":" in raw:
        account, _, conversation = raw.partition(":")
        account = account.strip()
        conversation = conversation.strip()
        if account and conversation:
            return account, conversation
        # e.g. ":42" — fall through to default account with the tail as conv
        conversation = conversation or account
    else:
        conversation = raw
    account = (str(default_account).strip() if default_account else "")
    if not account or not conversation:
        raise ValueError(f"Cannot resolve account/conversation from chat_id {chat_id!r}")
    return account, conversation


def _resolve_conversation_id(payload: Dict[str, Any]) -> Optional[str]:
    """Resolve the reply conversation id from several candidates (doc §3.2).

    Priority: ``conversation.id`` → ``conversation.display_id`` → top-level
    ``conversation_id``/``display_id``. Chatwoot's reply path expects the
    **display id**; if a deployment routes by the internal id instead, flip the
    order here.
    """
    conv = payload.get("conversation")
    if isinstance(conv, dict):
        for key in ("id", "display_id"):
            val = conv.get(key)
            if val is not None and str(val).strip():
                return str(val)
    for key in ("conversation_id", "display_id"):
        val = payload.get(key)
        if val is not None and str(val).strip():
            return str(val)
    return None


def _is_incoming(message_type: Any) -> bool:
    """Direction filter — accept string ``"incoming"`` OR integer enum ``0``.

    Chatwoot's message read-model uses ``integer [0,1,2,3]`` (0=incoming,
    1=outgoing, 2=activity, 3=template); the message *create* body uses the
    strings ``["incoming","outgoing"]``. Filtering strictly to incoming also
    excludes activity/template messages.
    """
    if isinstance(message_type, bool):
        return False
    if isinstance(message_type, int):
        return message_type == 0
    if isinstance(message_type, str):
        return message_type.strip().lower() == "incoming"
    return False


def validate_config(config) -> bool:
    """Return True when Chatwoot has the minimum credentials (base URL + token)."""
    extra = getattr(config, "extra", {}) or {}
    base_url = (getattr(config, "extra", {}) or {}).get("base_url") or os.getenv("CHATWOOT_BASE_URL", "")
    token = getattr(config, "token", None) or extra.get("token") or os.getenv("CHATWOOT_TOKEN", "")
    return bool(str(base_url).strip() and str(token).strip())


def is_connected(config) -> bool:
    """A Chatwoot platform is connected when it has both a base URL and a token."""
    return validate_config(config)


def _env_enablement() -> Optional[dict]:
    """Seed ``PlatformConfig.extra`` from env vars before adapter construction.

    Returns ``None`` when Chatwoot isn't minimally configured (base URL + bot
    token). The special ``home_channel`` key becomes a ``HomeChannel`` on the
    ``PlatformConfig`` via the core hook.
    """
    base_url = os.getenv("CHATWOOT_BASE_URL", "").strip()
    token = os.getenv("CHATWOOT_TOKEN", "").strip()
    if not (base_url and token):
        return None
    seed: dict = {
        "base_url": base_url.rstrip("/"),
        "token": token,
    }
    agent_token = os.getenv("CHATWOOT_AGENT_TOKEN", "").strip()
    if agent_token:
        seed["agent_token"] = agent_token
    account_id = os.getenv("CHATWOOT_ACCOUNT_ID", "").strip()
    if account_id:
        seed["account_id"] = account_id
    secret = os.getenv("CHATWOOT_WEBHOOK_SECRET", "").strip()
    if secret:
        seed["webhook_secret"] = secret
    host = os.getenv("CHATWOOT_HOST", "").strip()
    if host:
        seed["host"] = host
    port = os.getenv("CHATWOOT_PORT", "").strip()
    if port:
        try:
            seed["port"] = int(port)
        except ValueError:
            pass
    path = os.getenv("CHATWOOT_WEBHOOK_PATH", "").strip()
    if path:
        seed["webhook_path"] = path
    if _bool_env(os.getenv("CHATWOOT_PRIVATE_NOTE_TRACE")):
        seed["private_note_trace"] = True
    home = os.getenv("CHATWOOT_HOME_CHANNEL", "").strip()
    if home:
        seed["home_channel"] = {"chat_id": home, "name": "Chatwoot"}
    return seed


# ── adapter ──────────────────────────────────────────────────────────────────


class ChatwootAdapter(BasePlatformAdapter):
    """Receive Chatwoot webhook events and reply via the Application API."""

    MAX_MESSAGE_LENGTH = MAX_MESSAGE_LENGTH

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform("chatwoot"))
        extra = config.extra or {}
        self._base_url: str = str(
            extra.get("base_url") or os.getenv("CHATWOOT_BASE_URL", "")
        ).strip().rstrip("/")
        self._token: str = str(
            getattr(config, "token", None) or extra.get("token") or os.getenv("CHATWOOT_TOKEN", "")
        ).strip()
        self._agent_token: str = str(
            extra.get("agent_token") or os.getenv("CHATWOOT_AGENT_TOKEN", "")
        ).strip()
        self._account_id: str = str(
            extra.get("account_id") or os.getenv("CHATWOOT_ACCOUNT_ID", "")
        ).strip()
        self._webhook_secret: Optional[str] = (
            str(extra.get("webhook_secret") or os.getenv("CHATWOOT_WEBHOOK_SECRET", "")).strip()
            or None
        )
        self._host: str = str(extra.get("host", DEFAULT_HOST) or DEFAULT_HOST)
        self._port: int = int(extra.get("port", DEFAULT_PORT) or DEFAULT_PORT)
        self._webhook_path: str = self._normalize_path(
            extra.get("webhook_path", DEFAULT_WEBHOOK_PATH)
        )
        self._health_path: str = self._normalize_path(
            extra.get("health_path", DEFAULT_HEALTH_PATH)
        )
        self._max_body_bytes: int = int(extra.get("max_body_bytes", DEFAULT_MAX_BODY_BYTES))
        self._private_note_trace: bool = bool(
            extra.get("private_note_trace")
            or _bool_env(os.getenv("CHATWOOT_PRIVATE_NOTE_TRACE"))
        )

        self._runner = None  # aiohttp AppRunner
        self._session: Optional["aiohttp.ClientSession"] = None
        self._max_seen_ids = max(1, int(extra.get("max_seen_ids", DEFAULT_MAX_SEEN_IDS)))
        self._seen_ids: Set[str] = set()
        self._seen_id_order: Deque[str] = deque()
        self._private_note_warned = False

    # -- small utilities ------------------------------------------------------

    @staticmethod
    def _normalize_path(path: Any) -> str:
        raw = str(path or "").strip() or "/"
        return raw if raw.startswith("/") else f"/{raw}"

    def _parse_chat_id(self, chat_id: str) -> Tuple[str, str]:
        return _parse_chat_id(chat_id, default_account=self._account_id or None)

    def webhook_url(self) -> str:
        """Return the exact URL to paste into Chatwoot's Agent Bot config."""
        url = f"http://{self._host}:{self._port}{self._webhook_path}"
        if self._webhook_secret:
            url += f"?token={self._webhook_secret}"
        return url

    def _headers(self, *, private: bool = False, use_agent_token: bool = False) -> Dict[str, str]:
        """Auth headers. Private notes and typing use the agent token when configured."""
        token = (
            self._agent_token
            if ((private or use_agent_token) and self._agent_token)
            else self._token
        )
        return {"api_access_token": token}

    # -- idempotency ----------------------------------------------------------

    def _seen(self, message_id: str) -> bool:
        return message_id in self._seen_ids

    def _remember(self, message_id: str) -> None:
        self._seen_ids.add(message_id)
        self._seen_id_order.append(message_id)
        while len(self._seen_id_order) > self._max_seen_ids:
            oldest = self._seen_id_order.popleft()
            self._seen_ids.discard(oldest)

    # -- connection lifecycle -------------------------------------------------

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        if not AIOHTTP_AVAILABLE:
            logger.error("[chatwoot] aiohttp not installed. Run: pip install aiohttp")
            return False
        if not self._base_url or not self._token:
            logger.error("[chatwoot] Refusing to start without CHATWOOT_BASE_URL + CHATWOOT_TOKEN")
            return False

        if self._webhook_secret is None and is_network_accessible(self._host):
            logger.warning(
                "[chatwoot] Webhook bound to a network-accessible host (%s) without "
                "CHATWOOT_WEBHOOK_SECRET — anyone who can reach %s can POST events. "
                "Set a shared secret and append ?token=… to the Agent Bot outgoing URL.",
                self._host,
                self.webhook_url(),
            )

        self._session = aiohttp.ClientSession()

        app = web.Application(client_max_size=self._max_body_bytes + 1024)
        app.router.add_get(self._health_path, self._handle_health)
        app.router.add_post(self._webhook_path, self._handle_webhook)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self._host, self._port)
        await site.start()
        self._mark_connected()
        logger.info(
            "[chatwoot] Listening on %s:%d%s (token %s, agent-token %s, trace=%s)",
            self._host,
            self._port,
            self._webhook_path,
            _redact(self._token),
            _redact(self._agent_token) if self._agent_token else "<unset>",
            self._private_note_trace,
        )
        return True

    async def disconnect(self) -> None:
        if self._runner is not None:
            try:
                await self._runner.cleanup()
            except Exception:
                logger.debug("[chatwoot] runner cleanup failed", exc_info=True)
            self._runner = None
        if self._session is not None:
            try:
                await self._session.close()
            except Exception:
                logger.debug("[chatwoot] session close failed", exc_info=True)
            self._session = None
        self._mark_disconnected()

    # -- inbound: webhook handler --------------------------------------------

    async def _handle_health(self, request: "web.Request") -> "web.Response":
        return web.json_response({"status": "ok", "platform": self.platform.value})

    def _secret_ok(self, request: "web.Request") -> bool:
        if self._webhook_secret is None:
            return True
        provided = request.query.get("token", "")
        return hmac.compare_digest(provided, self._webhook_secret)

    async def _handle_webhook(self, request: "web.Request") -> "web.Response":
        """Receive a Chatwoot webhook POST, ack fast, dispatch async.

        Response contract (doc §3.5):
          - not running / creds missing → 404
          - body exceeds size limit     → 413
          - shared-secret mismatch      → 403
          - body isn't valid JSON       → 400
          - duplicate message id        → 200 (no-op)
          - non-actionable / unusable   → 200 (dropped, no retry)
          - accepted and dispatched     → 200
        """
        if not self._running:
            return web.Response(status=404)

        # Body-size guard before reading the payload.
        if request.content_length is not None and request.content_length > self._max_body_bytes:
            return web.Response(status=413)

        if not self._secret_ok(request):
            return web.Response(status=403)

        try:
            raw = await request.read()
        except Exception:
            # aiohttp raises when the streamed body exceeds client_max_size.
            return web.Response(status=413)
        if len(raw) > self._max_body_bytes:
            return web.Response(status=413)

        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            return web.Response(status=400)
        if not isinstance(payload, dict):
            return web.Response(status=400)

        # Idempotency: drop repeated message ids (Chatwoot retries webhooks).
        msg_id = self._message_id(payload)
        if msg_id is not None:
            if self._seen(msg_id):
                return web.Response(status=200)
            self._remember(msg_id)

        try:
            event = self._convert(payload)
        except Exception:
            logger.debug("[chatwoot] converter raised; acking to avoid retry", exc_info=True)
            return web.Response(status=200)

        if event is None:
            return web.Response(status=200)

        # Dispatch without blocking the webhook response.
        task = asyncio.create_task(self.handle_message(event))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return web.Response(status=200)

    @staticmethod
    def _message_id(payload: Dict[str, Any]) -> Optional[str]:
        val = payload.get("id")
        if val is not None and str(val).strip():
            return f"msg:{val}"
        return None

    # -- inbound: payload → MessageEvent -------------------------------------

    def _convert(self, payload: Dict[str, Any]) -> Optional[MessageEvent]:
        """Apply the §3 filters and build the internal message, or return None."""
        # 1. Event type — only new inbound messages.
        if payload.get("event") != "message_created":
            return None
        # 2. Direction — contact messages only (accept string or int enum).
        if not _is_incoming(payload.get("message_type")):
            return None
        # 3. Private notes — internal agent notes, not customer content.
        if payload.get("private") is True:
            return None
        # 4. Human handoff — only ``pending`` is the bot's to answer.
        conv = payload.get("conversation") if isinstance(payload.get("conversation"), dict) else {}
        status = str(conv.get("status") or payload.get("status") or "").strip().lower()
        if status and status != _BOT_OWNED_STATUS:
            return None

        conversation_id = _resolve_conversation_id(payload)
        if not conversation_id:
            return None

        account = payload.get("account") if isinstance(payload.get("account"), dict) else {}
        account_id = str(account.get("id") or self._account_id or "").strip()
        if not account_id:
            return None

        text = payload.get("content")
        text = str(text).strip() if text is not None else ""

        sender = payload.get("sender") if isinstance(payload.get("sender"), dict) else {}
        sender_id = str(sender.get("id") or "").strip() or None
        sender_name = sender.get("name") or sender.get("email") or None

        media_urls: List[str] = []
        media_types: List[str] = []
        self._collect_attachments(payload, media_urls, media_types)

        # 5. Empty content with no usable attachment — ignore.
        if not text and not media_urls:
            return None

        chat_id = _format_chat_id(account_id, conversation_id)
        source = self.build_source(
            chat_id=chat_id,
            chat_name=sender_name or f"conversation {conversation_id}",
            chat_type="direct",
            user_id=sender_id,
            user_name=sender_name,
        )
        mtype = MessageType.PHOTO if media_urls else MessageType.TEXT
        return MessageEvent(
            text=text,
            message_type=mtype,
            source=source,
            raw_message=payload,
            message_id=self._message_id(payload),
            media_urls=media_urls,
            media_types=media_types,
        )

    def _collect_attachments(
        self,
        payload: Dict[str, Any],
        media_urls: List[str],
        media_types: List[str],
    ) -> None:
        """Best-effort cache of inbound images/audio; failures never drop the msg."""
        attachments = payload.get("attachments")
        if not isinstance(attachments, list):
            return
        for att in attachments:
            if not isinstance(att, dict):
                continue
            url = att.get("data_url") or att.get("file_url") or att.get("thumb_url")
            file_type = str(att.get("file_type") or "").strip().lower()
            if not url:
                continue
            try:
                path, media_type = self._download_and_cache(str(url), file_type)
            except Exception:
                logger.debug("[chatwoot] attachment fetch failed: %s", _redact(str(url)), exc_info=True)
                continue
            if path:
                media_urls.append(path)
                media_types.append(media_type)

    def _download_and_cache(self, url: str, file_type: str) -> Tuple[Optional[str], str]:
        """Synchronously fetch an attachment URL and cache it locally.

        Uses a short-lived urllib fetch to keep the converter synchronous and
        testable; swap for the shared client if you need auth on media URLs.
        """
        import urllib.request

        with urllib.request.urlopen(url, timeout=15) as resp:  # noqa: S310 (trusted Chatwoot URLs)
            data = resp.read()
        if file_type == "image":
            return cache_image_from_bytes(data, ".jpg"), "image/jpeg"
        if file_type in ("audio", "voice"):
            return cache_audio_from_bytes(data, ".ogg"), "audio/ogg"
        filename = url.rsplit("/", 1)[-1] or "attachment"
        return cache_document_from_bytes(data, filename), "application/octet-stream"

    # -- outbound: replies ----------------------------------------------------

    def _messages_endpoint(self, account_id: str, conversation_id: str) -> str:
        return (
            f"{self._base_url}/api/v1/accounts/{account_id}"
            f"/conversations/{conversation_id}/messages"
        )

    def _split(self, content: str) -> List[str]:
        """Split content into <= MAX_MESSAGE_LENGTH chunks on newline boundaries."""
        if len(content) <= self.MAX_MESSAGE_LENGTH:
            return [content] if content else []
        chunks: List[str] = []
        remaining = content
        while len(remaining) > self.MAX_MESSAGE_LENGTH:
            window = remaining[: self.MAX_MESSAGE_LENGTH]
            split_at = window.rfind("\n")
            if split_at <= 0:
                split_at = self.MAX_MESSAGE_LENGTH
            chunks.append(remaining[:split_at])
            remaining = remaining[split_at:].lstrip("\n")
        if remaining:
            chunks.append(remaining)
        return chunks

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        # Private-note trace: reasoning / tool progress arrives marked
        # ``non_conversational`` (see gateway/run.py). When the trace is on,
        # route those to an agent-only private note instead of a customer reply.
        private = bool(metadata and metadata.get("non_conversational")) and self._private_note_trace
        if private and not self._agent_token and not self._private_note_warned:
            logger.warning(
                "[chatwoot] CHATWOOT_PRIVATE_NOTE_TRACE is on but CHATWOOT_AGENT_TOKEN "
                "is unset; private notes will be attempted with the bot token and may "
                "be rejected. Customer replies are unaffected."
            )
            self._private_note_warned = True

        try:
            account_id, conversation_id = self._parse_chat_id(chat_id)
        except ValueError as exc:
            return SendResult(success=False, error=str(exc))

        chunks = self._split(content)
        if not chunks:
            return SendResult(success=True)

        last: Optional[SendResult] = None
        for chunk in chunks:
            last = await self._post_message(
                account_id, conversation_id, chunk, private=private
            )
            if not last.success:
                return last
        return last or SendResult(success=True)

    async def _post_message(
        self,
        account_id: str,
        conversation_id: str,
        content: str,
        *,
        private: bool = False,
    ) -> SendResult:
        if self._session is None:
            return SendResult(success=False, error="adapter not connected")
        url = self._messages_endpoint(account_id, conversation_id)
        body = {
            "content": content,
            "message_type": "outgoing",
            "private": private,
        }
        try:
            async with self._session.post(
                url, json=body, headers=self._headers(private=private)
            ) as resp:
                if 200 <= resp.status < 300:
                    try:
                        data = await resp.json()
                    except Exception:
                        data = None
                    mid = str(data.get("id")) if isinstance(data, dict) and data.get("id") else None
                    return SendResult(success=True, message_id=mid, raw_response=data)
                detail = await resp.text()
                retryable = resp.status >= 500
                if private and resp.status in (401, 403):
                    # Degrade gracefully — a private note that can't post must
                    # never look like a hard failure that blocks the pipeline.
                    logger.warning(
                        "[chatwoot] private note rejected (%s) — check CHATWOOT_AGENT_TOKEN. "
                        "Skipping note; customer replies unaffected.",
                        resp.status,
                    )
                return SendResult(
                    success=False,
                    error=f"HTTP {resp.status}: {detail[:200]}",
                    retryable=retryable,
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return SendResult(success=False, error=str(exc), retryable=True)

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        if self._session is None:
            return
        try:
            account_id, conversation_id = self._parse_chat_id(chat_id)
        except ValueError:
            return
        url = (
            f"{self._base_url}/api/v1/accounts/{account_id}"
            f"/conversations/{conversation_id}/toggle_typing_status"
        )
        try:
            async with self._session.post(
                url,
                json={"typing_status": "on"},
                headers=self._headers(use_agent_token=True),
            ):
                pass
        except Exception:
            logger.debug("[chatwoot] typing toggle failed (cosmetic)", exc_info=True)

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: str = "",
    ) -> SendResult:
        return await self._post_attachment(chat_id, image_url, caption)

    async def send_document(
        self,
        chat_id: str,
        path: str,
        caption: str = "",
    ) -> SendResult:
        return await self._post_attachment(chat_id, path, caption)

    async def _post_attachment(
        self,
        chat_id: str,
        path_or_url: str,
        caption: str = "",
    ) -> SendResult:
        """Upload a local file (or remote URL) as a multipart attachment."""
        if self._session is None:
            return SendResult(success=False, error="adapter not connected")
        try:
            account_id, conversation_id = self._parse_chat_id(chat_id)
        except ValueError as exc:
            return SendResult(success=False, error=str(exc))

        local_path, cleanup = await self._ensure_local_file(path_or_url)
        try:
            data = aiohttp.FormData()
            data.add_field("message_type", "outgoing")
            if caption:
                data.add_field("content", caption)
            with open(local_path, "rb") as fh:
                data.add_field(
                    "attachments[]",
                    fh.read(),
                    filename=os.path.basename(local_path) or "attachment",
                )
            url = self._messages_endpoint(account_id, conversation_id)
            # Multipart: send only the auth header; let aiohttp set the
            # multipart content type + boundary.
            async with self._session.post(url, data=data, headers=self._headers()) as resp:
                if 200 <= resp.status < 300:
                    return SendResult(success=True)
                detail = await resp.text()
                return SendResult(
                    success=False,
                    error=f"HTTP {resp.status}: {detail[:200]}",
                    retryable=resp.status >= 500,
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return SendResult(success=False, error=str(exc), retryable=True)
        finally:
            if cleanup:
                try:
                    os.unlink(local_path)
                except OSError:
                    pass

    async def _ensure_local_file(self, path_or_url: str) -> Tuple[str, bool]:
        """Return (local_path, needs_cleanup). Downloads remote URLs to temp."""
        if os.path.exists(path_or_url):
            return path_or_url, False
        # Remote URL → download to a temp file.
        import tempfile

        assert self._session is not None
        async with self._session.get(path_or_url) as resp:
            payload = await resp.read()
        suffix = os.path.splitext(path_or_url.split("?", 1)[0])[1] or ""
        fd, tmp = tempfile.mkstemp(suffix=suffix)
        with os.fdopen(fd, "wb") as fh:
            fh.write(payload)
        return tmp, True

    async def handle_message(self, event: "MessageEvent") -> None:
        """Fire CRWD contact/Honcho enrichment, then run the normal pipeline.

        Enrichment is spawned as a background task so it runs concurrently with
        the agent turn and never delays the reply. It self-gates (idempotent)
        and swallows its own errors.
        """
        try:
            from plugins.platforms.chatwoot import enrichment
            task = asyncio.create_task(enrichment.enrich(self, event))
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
        except Exception:
            logger.debug("[chatwoot] failed to spawn enrichment task", exc_info=True)
        await super().handle_message(event)

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {"name": chat_id, "type": "direct", "chat_id": chat_id}

    # -- contact enrichment (CRWD Mongo → Chatwoot) --------------------------
    #
    # Read/write the Chatwoot Contacts API so the enrichment path can hydrate
    # a contact from the CRWD database.  All three helpers reuse the adapter's
    # live aiohttp session, base URL, account id, and admin token; they return
    # plain dicts (never raise) so enrichment stays best-effort.

    def _contacts_endpoint(self, account_id: str, contact_id: str = "") -> str:
        base = f"{self._base_url}/api/v1/accounts/{account_id}/contacts"
        return f"{base}/{contact_id}" if contact_id else base

    async def get_contact(self, account_id: str, contact_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a contact record. Returns the contact payload dict or None."""
        if self._session is None:
            return None
        url = self._contacts_endpoint(account_id, contact_id)
        try:
            # Contacts API is not authorized for Agent Bots — use the agent/user token.
            async with self._session.get(url, headers=self._headers(use_agent_token=True)) as resp:
                if 200 <= resp.status < 300:
                    data = await resp.json()
                    if isinstance(data, dict):
                        # Show responses wrap the record under "payload".
                        return data.get("payload", data)
                    return None
                logger.debug("[chatwoot] get_contact %s → HTTP %s", contact_id, resp.status)
                return None
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug("[chatwoot] get_contact %s failed: %s", contact_id, exc)
            return None

    async def update_contact(
        self, account_id: str, contact_id: str, fields: Dict[str, Any]
    ) -> bool:
        """PUT contact fields (name/email/phone_number/avatar_url/attributes)."""
        if self._session is None or not fields:
            return False
        url = self._contacts_endpoint(account_id, contact_id)
        try:
            async with self._session.put(
                url, json=fields, headers=self._headers(use_agent_token=True)
            ) as resp:
                if 200 <= resp.status < 300:
                    return True
                detail = await resp.text()
                logger.warning(
                    "[chatwoot] update_contact %s → HTTP %s: %s",
                    contact_id, resp.status, detail[:200],
                )
                return False
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("[chatwoot] update_contact %s failed: %s", contact_id, exc)
            return False

    async def url_is_image(self, url: str) -> bool:
        """Best-effort check that ``url`` actually serves an image.

        The CRWD staging asset server returns its admin HTML page with HTTP
        200 for missing files, so a status check alone isn't enough — we must
        confirm the ``Content-Type`` is an image before pinning it as a
        contact avatar (a broken avatar is worse than none). Headers are read
        without consuming the body.
        """
        if self._session is None or not url:
            return False
        try:
            async with self._session.get(url, allow_redirects=True) as resp:
                ctype = (resp.headers.get("Content-Type") or "").lower()
                return resp.status < 400 and ctype.startswith("image/")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug("[chatwoot] url_is_image %s failed: %s", url, exc)
            return False

    async def add_contact_labels(
        self, account_id: str, contact_id: str, labels: List[str]
    ) -> bool:
        """POST the contact's label set (Chatwoot replaces the full list)."""
        if self._session is None or not labels:
            return False
        url = f"{self._contacts_endpoint(account_id, contact_id)}/labels"
        try:
            async with self._session.post(
                url, json={"labels": labels}, headers=self._headers(use_agent_token=True)
            ) as resp:
                if 200 <= resp.status < 300:
                    return True
                detail = await resp.text()
                logger.warning(
                    "[chatwoot] add_contact_labels %s → HTTP %s: %s",
                    contact_id, resp.status, detail[:200],
                )
                return False
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("[chatwoot] add_contact_labels %s failed: %s", contact_id, exc)
            return False


# ── out-of-process cron delivery ─────────────────────────────────────────────


async def _standalone_send(
    pconfig,
    chat_id: str,
    message: str,
    *,
    thread_id: Optional[str] = None,
    media_files: Optional[list] = None,
    force_document: bool = False,
) -> Dict[str, Any]:
    """POST a single reply without a live adapter (cron in a separate process).

    Matches the ``standalone_sender_fn`` contract consumed by
    ``tools/send_message_tool.py``: returns ``{"success": True, ...}`` or
    ``{"error": "..."}``.
    """
    if not AIOHTTP_AVAILABLE:
        return {"error": "aiohttp not installed"}

    extra = getattr(pconfig, "extra", {}) or {}
    base_url = str(
        extra.get("base_url") or os.getenv("CHATWOOT_BASE_URL", "")
    ).strip().rstrip("/")
    token = str(
        getattr(pconfig, "token", None) or extra.get("token") or os.getenv("CHATWOOT_TOKEN", "")
    ).strip()
    default_account = str(extra.get("account_id") or os.getenv("CHATWOOT_ACCOUNT_ID", "")).strip()
    if not base_url or not token:
        return {"error": "Chatwoot not configured (CHATWOOT_BASE_URL + CHATWOOT_TOKEN)"}

    try:
        account_id, conversation_id = _parse_chat_id(chat_id, default_account or None)
    except ValueError as exc:
        return {"error": str(exc)}

    url = (
        f"{base_url}/api/v1/accounts/{account_id}"
        f"/conversations/{conversation_id}/messages"
    )
    body = {"content": message, "message_type": "outgoing", "private": False}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=body, headers={"api_access_token": token}) as resp:
                if 200 <= resp.status < 300:
                    return {"success": True}
                detail = await resp.text()
                return {"error": f"HTTP {resp.status}: {detail[:200]}"}
    except Exception as exc:
        return {"error": f"Chatwoot standalone send failed: {exc}"}


# ── interactive setup ────────────────────────────────────────────────────────


def interactive_setup() -> None:
    """Minimal setup flow — surfaces the webhook URL to paste into Chatwoot."""
    try:
        from hermes_cli.config import get_env_value, save_env_value
        from hermes_cli.cli_output import (
            prompt,
            prompt_yes_no,
            print_info,
            print_success,
            print_warning,
        )
    except Exception:
        # Setup helpers unavailable (e.g. non-interactive context) — no-op.
        print("Set CHATWOOT_BASE_URL, CHATWOOT_TOKEN, and point the Agent Bot "
              "webhook at http://<host>:%d%s" % (DEFAULT_PORT, DEFAULT_WEBHOOK_PATH))
        return

    base_url = prompt("Chatwoot base URL", default=get_env_value("CHATWOOT_BASE_URL") or "")
    if base_url:
        save_env_value("CHATWOOT_BASE_URL", base_url.rstrip("/"))
    token = prompt("Agent Bot access token", default=get_env_value("CHATWOOT_TOKEN") or "")
    if token:
        save_env_value("CHATWOOT_TOKEN", token)
    if prompt_yes_no("Post reasoning/tool activity as private notes?", False):
        save_env_value("CHATWOOT_PRIVATE_NOTE_TRACE", "true")
        agent_token = prompt("Agent (user) token for private notes", default=get_env_value("CHATWOOT_AGENT_TOKEN") or "")
        if agent_token:
            save_env_value("CHATWOOT_AGENT_TOKEN", agent_token)

    host = get_env_value("CHATWOOT_HOST") or DEFAULT_HOST
    port = get_env_value("CHATWOOT_PORT") or str(DEFAULT_PORT)
    path = get_env_value("CHATWOOT_WEBHOOK_PATH") or DEFAULT_WEBHOOK_PATH
    secret = get_env_value("CHATWOOT_WEBHOOK_SECRET")
    url = f"http://{host}:{port}{path}"
    if secret:
        url += f"?token={secret}"
    print_success("Chatwoot configuration saved to ~/.hermes/.env")
    print_info(f"Point the Agent Bot 'Outgoing URL' at:  {url}")
    print_warning("Set CHATWOOT_ALLOWED_USERS (or CHATWOOT_ALLOW_ALL_USERS=true) to authorize contacts.")


# ── plugin entry point ───────────────────────────────────────────────────────


def register(ctx) -> None:
    """Plugin entry point — called by the Hermes plugin system."""
    ctx.register_platform(
        name="chatwoot",
        label="Chatwoot",
        adapter_factory=lambda cfg: ChatwootAdapter(cfg),
        check_fn=check_chatwoot_requirements,
        validate_config=validate_config,
        is_connected=is_connected,
        required_env=["CHATWOOT_BASE_URL", "CHATWOOT_TOKEN"],
        install_hint="pip install aiohttp",
        setup_fn=interactive_setup,
        # Seed PlatformConfig.extra from env before adapter construction so
        # env-only setups surface in `hermes gateway status`.
        env_enablement_fn=_env_enablement,
        # Cron / proactive delivery.
        cron_deliver_env_var="CHATWOOT_HOME_CHANNEL",
        standalone_sender_fn=_standalone_send,
        # Authorization integration for _is_user_authorized().
        allowed_users_env="CHATWOOT_ALLOWED_USERS",
        allow_all_env="CHATWOOT_ALLOW_ALL_USERS",
        max_message_length=MAX_MESSAGE_LENGTH,
        emoji="💬",
        platform_hint=(
            "You are a Chatwoot support agent bot talking to a customer in a "
            "helpdesk conversation. Markdown renders. Keep replies clear and "
            "professional. To send an image or file, use the media tools — they "
            "upload as Chatwoot attachments. A human agent may take over at any "
            "time, after which you stay silent until handed control back."
        ),
    )

    # Inject the current CRWD member's user_id into each turn so the coach can
    # call crwd_db user lookups directly (no get_user round-trip). Best-effort;
    # no-ops off Chatwoot or when the id can't be resolved.
    from plugins.platforms.chatwoot import coach_context

    ctx.register_hook("pre_llm_call", coach_context.member_context_hook)

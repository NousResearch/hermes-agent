"""Native Facebook Messenger platform adapter.

The adapter owns the Meta Page webhook and Graph API edge.  It deliberately
keeps comment handling separate from normal Messenger DMs: comments can only
produce one fixed private reply, while DMs are normalized into the ordinary
Hermes ``MessageEvent`` path.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import re
import tempfile
import threading
import time
import unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    import aiohttp
    from aiohttp import web
except ImportError:  # pragma: no cover - the registry's passive probe handles this
    aiohttp = None  # type: ignore[assignment]
    web = None  # type: ignore[assignment]

try:
    import httpx
except ImportError:  # pragma: no cover - the registry's passive probe handles this
    httpx = None  # type: ignore[assignment]

from agent.secret_scope import UnscopedSecretError, get_secret
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult
from hermes_constants import get_hermes_home


logger = logging.getLogger(__name__)

GRAPH_BASE_URL = "https://graph.facebook.com"
DEFAULT_API_VERSION = "v23.0"
DEFAULT_WEBHOOK_HOST = "0.0.0.0"
DEFAULT_WEBHOOK_PORT = 8647
DEFAULT_WEBHOOK_PATH = "/messenger/webhook"
WEBHOOK_BODY_MAX_BYTES = 1_048_576
DEFAULT_COMMENT_TRIGGER = "hotel"
MAX_LOOKBACK_DAYS = 7
MAX_LOOKBACK_COMMENTS = 100
DEFAULT_MAX_COMMENTS = 25
DEFAULT_HISTORY_LIMIT = 20
MAX_HISTORY_LIMIT = 20
MAX_HISTORY_CHARS = 8_000

PRIVATE_REPLY_OPENING = (
    "Hola, te saluda el asistente automático de VIAJA CON CARLOS. "
    "Estoy aquí para resolver tus dudas y, si lo prefieres, te conecto con una persona. "
    "¿Cuál propiedad te interesó?"
)


class GraphAPIError(RuntimeError):
    """Raised when Meta rejects a Graph API request."""


def _scoped_secret(name: str, default: str = "") -> str:
    """Read a credential in the active profile, with default-profile fallback."""
    try:
        value = get_secret(name)
    except UnscopedSecretError:
        value = os.getenv(name)
    return str(value if value is not None else default).strip()


def _credential(extra: Dict[str, Any], key: str, env_name: str) -> str:
    value = extra.get(key)
    return str(value).strip() if value is not None and str(value).strip() else _scoped_secret(env_name)


def _coerce_int(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def bound_lookback_days(value: Any) -> int:
    """Return a non-negative retrospective window, capped at seven days."""
    return max(0, min(MAX_LOOKBACK_DAYS, _coerce_int(value, 0)))


def bound_max_comments(value: Any) -> int:
    """Return a bounded maximum number of retrospective comments."""
    return max(1, min(MAX_LOOKBACK_COMMENTS, _coerce_int(value, DEFAULT_MAX_COMMENTS)))


def normalize_comment_text(value: Any) -> str:
    """Normalize text before applying a whole-word trigger."""
    return unicodedata.normalize("NFKC", str(value or "")).casefold().strip()


def comment_matches_trigger(text: Any, trigger: Any = DEFAULT_COMMENT_TRIGGER) -> bool:
    normalized_text = normalize_comment_text(text)
    normalized_trigger = normalize_comment_text(trigger)
    if not normalized_text or not normalized_trigger:
        return False
    return re.search(
        rf"(?<!\w){re.escape(normalized_trigger)}(?!\w)",
        normalized_text,
        flags=re.UNICODE,
    ) is not None


def verify_messenger_signature(body: bytes, header: str, app_secret: str) -> bool:
    """Verify ``X-Hub-Signature-256`` over the exact raw request bytes."""
    if body is None or not header or not app_secret:
        return False
    supplied = str(header).strip()
    if not supplied.lower().startswith("sha256="):
        return False
    supplied_digest = supplied[7:]
    if not supplied_digest:
        return False
    expected = hmac.new(
        app_secret.encode("utf-8"), body, hashlib.sha256
    ).hexdigest()
    try:
        return hmac.compare_digest(expected.encode("ascii"), supplied_digest.encode("ascii"))
    except UnicodeEncodeError:
        return False


def verify_webhook_challenge(mode: Any, token: Any, expected_token: str) -> bool:
    """Constant-time comparison for Meta's GET webhook verification."""
    if str(mode or "") != "subscribe" or not token or not expected_token:
        return False
    try:
        return hmac.compare_digest(
            str(token).encode("utf-8"), str(expected_token).encode("utf-8")
        )
    except UnicodeEncodeError:
        return False


class ProcessedCommentLedger:
    """Small durable receipt ledger for accepted private comment replies."""

    MAX_RECEIPTS = 5_000

    def __init__(self, path: Optional[str | Path] = None) -> None:
        self.path = Path(path) if path else get_hermes_home() / "platforms" / "messenger" / "processed_comments.json"
        self._lock = threading.Lock()
        self._receipts: Dict[str, float] = {}
        self._load()

    def _load(self) -> None:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            values = raw.get("receipts", {}) if isinstance(raw, dict) else raw
            if isinstance(values, dict):
                self._receipts = {str(k): float(v) for k, v in values.items()}
            elif isinstance(values, list):
                self._receipts = {str(k): 0.0 for k in values}
        except (OSError, ValueError, TypeError):
            self._receipts = {}
        if len(self._receipts) > self.MAX_RECEIPTS:
            self._receipts = dict(
                sorted(self._receipts.items(), key=lambda item: item[1])[-self.MAX_RECEIPTS :]
            )

    def contains(self, comment_id: str) -> bool:
        with self._lock:
            return str(comment_id) in self._receipts

    def mark(self, comment_id: str) -> bool:
        """Persist a receipt; callers invoke this only after Meta accepts."""
        key = str(comment_id).strip()
        if not key:
            return False
        with self._lock:
            self._receipts[key] = time.time()
            if len(self._receipts) > self.MAX_RECEIPTS:
                oldest = min(self._receipts, key=self._receipts.get)
                self._receipts.pop(oldest, None)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps({"receipts": self._receipts}, sort_keys=True)
            fd, temp_name = tempfile.mkstemp(
                prefix="processed_comments.", suffix=".tmp", dir=str(self.path.parent)
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temp_name, self.path)
            except OSError:
                try:
                    os.unlink(temp_name)
                except OSError:
                    pass
                self._receipts.pop(key, None)
                return False
            return True


class MessengerAdapter(BasePlatformAdapter):
    """Facebook Messenger Page webhook + Graph API adapter."""

    MAX_MESSAGE_LENGTH = 2_000

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform("messenger"))
        extra = dict(getattr(config, "extra", {}) or {})
        self.page_id = _credential(extra, "page_id", "MESSENGER_PAGE_ID")
        self.page_access_token = _credential(
            extra, "page_access_token", "MESSENGER_PAGE_ACCESS_TOKEN"
        )
        self.app_secret = _credential(extra, "app_secret", "MESSENGER_APP_SECRET")
        self.verify_token = _credential(extra, "verify_token", "MESSENGER_VERIFY_TOKEN")
        self.api_version = str(
            extra.get("api_version") or os.getenv("MESSENGER_API_VERSION", DEFAULT_API_VERSION)
        ).strip() or DEFAULT_API_VERSION
        if not self.api_version.startswith("v"):
            self.api_version = f"v{self.api_version}"
        self.webhook_host = str(
            extra.get("host") or os.getenv("MESSENGER_HOST", DEFAULT_WEBHOOK_HOST)
        ).strip() or DEFAULT_WEBHOOK_HOST
        self.webhook_port = max(
            0,
            _coerce_int(extra.get("port") or os.getenv("MESSENGER_PORT"), DEFAULT_WEBHOOK_PORT),
        )
        self.webhook_path = str(
            extra.get("path") or os.getenv("MESSENGER_WEBHOOK_PATH", DEFAULT_WEBHOOK_PATH)
        ).strip() or DEFAULT_WEBHOOK_PATH
        if not self.webhook_path.startswith("/"):
            self.webhook_path = "/" + self.webhook_path
        self.comment_trigger = normalize_comment_text(
            extra.get("comment_trigger")
            or os.getenv("MESSENGER_COMMENT_TRIGGER", DEFAULT_COMMENT_TRIGGER)
        )
        self.lookback_days = bound_lookback_days(
            extra.get("lookback_days") or os.getenv("MESSENGER_LOOKBACK_DAYS", 0)
        )
        self.max_comments = bound_max_comments(
            extra.get("max_comments")
            or os.getenv("MESSENGER_MAX_COMMENTS", DEFAULT_MAX_COMMENTS)
        )
        history_limit = _coerce_int(extra.get("history_limit"), DEFAULT_HISTORY_LIMIT)
        self.history_limit = max(1, min(MAX_HISTORY_LIMIT, history_limit))
        receipt_path = extra.get("receipt_path")
        self._ledger = ProcessedCommentLedger(receipt_path)
        self._http_client: Optional["httpx.AsyncClient"] = None
        self._runner: Optional["web.AppRunner"] = None
        self._site: Optional["web.TCPSite"] = None
        self._app: Optional["web.Application"] = None

    # ------------------------------------------------------------------
    # Lifecycle / Graph transport
    # ------------------------------------------------------------------

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        if not check_requirements() or not self._credentials_present():
            self._set_fatal_error(
                "messenger_not_configured",
                "Messenger requires Page ID, Page access token, App Secret, and Verify Token.",
                retryable=False,
            )
            return False
        assert aiohttp is not None and httpx is not None
        self._app = web.Application(client_max_size=WEBHOOK_BODY_MAX_BYTES)
        self._app.router.add_get(self.webhook_path, self._handle_verify)
        self._app.router.add_post(self.webhook_path, self._handle_webhook)
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        try:
            self._site = web.TCPSite(self._runner, self.webhook_host, self.webhook_port)
            await self._site.start()
        except OSError as exc:
            await self._runner.cleanup()
            self._runner = None
            self._set_fatal_error("messenger_bind_failed", str(exc), retryable=True)
            return False
        self._http_client = httpx.AsyncClient(timeout=20.0)
        self._mark_connected()
        if self.lookback_days:
            try:
                await self._process_lookback()
            except Exception:
                logger.exception("Messenger retrospective comment lookback failed")
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()
        if self._site is not None:
            await self._site.stop()
            self._site = None
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None
        self._app = None

    def _credentials_present(self) -> bool:
        return all(
            (self.page_id, self.page_access_token, self.app_secret, self.verify_token)
        )

    def _graph_url(self, path: str) -> str:
        return f"{GRAPH_BASE_URL}/{self.api_version}/{path.lstrip('/')}"

    async def _graph_post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self._http_client is None:
            raise GraphAPIError("Messenger adapter is not connected")
        response = await self._http_client.post(
            self._graph_url(path),
            params={"access_token": self.page_access_token},
            json=payload,
        )
        if not 200 <= response.status_code < 300:
            raise GraphAPIError(f"Meta Graph API returned HTTP {response.status_code}: {response.text}")
        try:
            data = response.json()
        except ValueError:
            data = {}
        return data if isinstance(data, dict) else {}

    async def _graph_get(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if self._http_client is None:
            raise GraphAPIError("Messenger adapter is not connected")
        query = dict(params)
        query["access_token"] = self.page_access_token
        response = await self._http_client.get(self._graph_url(path), params=query)
        if not 200 <= response.status_code < 300:
            raise GraphAPIError(f"Meta Graph API returned HTTP {response.status_code}: {response.text}")
        try:
            data = response.json()
        except ValueError:
            data = {}
        return data if isinstance(data, dict) else {}

    # ------------------------------------------------------------------
    # Webhook verification and dispatch
    # ------------------------------------------------------------------

    async def _handle_verify(self, request: Any) -> Any:
        mode = request.query.get("hub.mode")
        token = request.query.get("hub.verify_token")
        challenge = request.query.get("hub.challenge", "")
        if verify_webhook_challenge(mode, token, self.verify_token):
            return web.Response(status=200, text=str(challenge))
        return web.Response(status=403, text="forbidden")

    async def _handle_webhook(self, request: Any) -> Any:
        content_length = request.headers.get("Content-Length")
        if content_length:
            try:
                if int(content_length) > WEBHOOK_BODY_MAX_BYTES:
                    return web.Response(status=413, text="payload too large")
            except (TypeError, ValueError):
                pass
        try:
            body = await request.read()
        except Exception:
            return web.Response(status=400, text="bad request")
        if len(body) > WEBHOOK_BODY_MAX_BYTES:
            return web.Response(status=413, text="payload too large")
        signature = request.headers.get("X-Hub-Signature-256", "")
        if not verify_messenger_signature(body, signature, self.app_secret):
            return web.Response(status=401, text="invalid signature")
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return web.Response(status=400, text="bad json")
        if not isinstance(payload, dict) or payload.get("object") not in (None, "page"):
            return web.Response(status=400, text="unsupported object")
        for entry in payload.get("entry", []) or []:
            if isinstance(entry, dict):
                try:
                    await self._dispatch_entry(entry)
                except Exception:
                    logger.exception("Messenger webhook entry processing failed")
        return web.Response(status=200, text="EVENT_RECEIVED")

    async def _dispatch_entry(self, entry: Dict[str, Any]) -> None:
        entry_page_id = str(entry.get("id") or "")
        if entry_page_id and self.page_id and entry_page_id != self.page_id:
            return
        for event in entry.get("messaging", []) or []:
            if isinstance(event, dict):
                await self._handle_messaging_event(event)
        for change in entry.get("changes", []) or []:
            if not isinstance(change, dict) or change.get("field") not in (None, "feed"):
                continue
            value = change.get("value") or {}
            if isinstance(value, dict):
                await self._handle_comment(value)

    async def _handle_messaging_event(self, event: Dict[str, Any]) -> None:
        sender = str((event.get("sender") or {}).get("id") or "").strip()
        if not sender or sender == self.page_id:
            return
        message = event.get("message") or {}
        postback = event.get("postback") or message.get("quick_reply")
        if postback:
            text = str(postback.get("title") or postback.get("payload") or "").strip()
        else:
            text = str(message.get("text") or "").strip()
        if not text:
            return
        history_lines = await self._fetch_psid_history(sender)
        source = self.build_source(
            chat_id=sender,
            chat_type="dm",
            user_id=sender,
            user_name=sender,
            chat_name=sender,
        )
        event_metadata: Dict[str, Any] = {}
        if postback:
            event_metadata["messenger_postback"] = postback
        if history_lines:
            event_metadata["messenger_history"] = history_lines
        normalized_event = MessageEvent(
            text=text,
            message_type=MessageType.TEXT,
            source=source,
            raw_message=event,
            message_id=str(message.get("mid") or event.get("message_id") or "") or None,
            user_id=sender,
            user_name=sender,
            channel_context="\n".join(history_lines) if history_lines else None,
            metadata=event_metadata,
        )
        await self.handle_message(normalized_event)

    async def _fetch_psid_history(self, psid: str) -> List[str]:
        """Fetch only bounded Page-scoped history when a PSID messages again."""
        if not psid or not self.page_id or not self.page_access_token:
            return []
        try:
            data = await self._graph_get(
                f"/{self.page_id}/conversations",
                {
                    "user_id": psid,
                    "limit": str(self.history_limit),
                    "fields": (
                        f"messages.limit({self.history_limit})"
                        "{message,from,created_time}"
                    ),
                },
            )
        except GraphAPIError:
            logger.info("Messenger PSID history unavailable for %s", psid)
            return []
        lines: List[str] = []
        for conversation in data.get("data", []) or []:
            messages = (conversation or {}).get("messages", {}) if isinstance(conversation, dict) else {}
            for item in messages.get("data", []) if isinstance(messages, dict) else []:
                if not isinstance(item, dict) or not item.get("message"):
                    continue
                author = (item.get("from") or {}).get("name") or (item.get("from") or {}).get("id") or "Messenger"
                lines.append(f"{author}: {str(item['message']).strip()}")
                if len(lines) >= self.history_limit:
                    break
            if len(lines) >= self.history_limit:
                break
        return _cap_history(lines)

    # ------------------------------------------------------------------
    # Page feed comments and retrospective lookback
    # ------------------------------------------------------------------

    async def _handle_comment(self, comment: Dict[str, Any]) -> bool:
        if comment.get("item") not in (None, "comment"):
            return False
        if comment.get("verb") not in (None, "add"):
            return False
        comment_id = str(comment.get("comment_id") or comment.get("id") or "").strip()
        if not comment_id or self._ledger.contains(comment_id):
            return False
        if not comment_matches_trigger(comment.get("message"), self.comment_trigger):
            return False
        accepted = await self._send_private_comment_reply(comment_id)
        if accepted:
            # The receipt is durable and intentionally written only after Meta
            # accepted the private reply (HTTP 2xx from Graph).
            self._ledger.mark(comment_id)
        return accepted

    async def _send_private_comment_reply(self, comment_id: str) -> bool:
        if not comment_id:
            return False
        try:
            await self._graph_post(
                f"/{self.page_id}/messages",
                {
                    "recipient": {"comment_id": comment_id},
                    "message": {"text": PRIVATE_REPLY_OPENING},
                },
            )
            return True
        except GraphAPIError:
            logger.exception("Messenger private comment reply failed for %s", comment_id)
            return False

    def _lookback_since(self, now: Optional[datetime] = None) -> int:
        current = now or datetime.now(timezone.utc)
        return int((current - timedelta(days=self.lookback_days)).timestamp())

    async def _process_lookback(self) -> None:
        if not self.lookback_days:
            return
        data = await self._graph_get(
            f"/{self.page_id}/feed",
            {
                "since": str(self._lookback_since()),
                "limit": str(self.max_comments),
                "fields": (
                    f"comments.limit({self.max_comments})"
                    "{id,message,from,created_time}"
                ),
            },
        )
        count = 0
        for post in data.get("data", []) or []:
            if not isinstance(post, dict):
                continue
            comments = post.get("comments", {}).get("data", []) if isinstance(post.get("comments"), dict) else []
            for comment in comments or []:
                if count >= self.max_comments:
                    return
                if isinstance(comment, dict):
                    await self._handle_comment({"item": "comment", **comment})
                    count += 1

    # ------------------------------------------------------------------
    # Base adapter outbound surface
    # ------------------------------------------------------------------

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        if not chat_id or not content or not content.strip():
            return SendResult(success=True, message_id=None)
        try:
            data = await self._graph_post(
                f"/{self.page_id}/messages",
                {
                    "recipient": {"id": str(chat_id)},
                    "message": {"text": str(content)},
                    "messaging_type": "RESPONSE",
                },
            )
            return SendResult(success=True, message_id=data.get("message_id"))
        except GraphAPIError as exc:
            return SendResult(success=False, error=str(exc))

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {"name": str(chat_id), "type": "dm", "platform": "messenger"}


# ----------------------------------------------------------------------
# Plugin registration
# ----------------------------------------------------------------------


def check_requirements() -> bool:
    if aiohttp is None or httpx is None:
        return False
    return bool(
        _scoped_secret("MESSENGER_PAGE_ID")
        and _scoped_secret("MESSENGER_PAGE_ACCESS_TOKEN")
        and _scoped_secret("MESSENGER_APP_SECRET")
        and _scoped_secret("MESSENGER_VERIFY_TOKEN")
    )


def validate_config(config: PlatformConfig) -> bool:
    extra = dict(getattr(config, "extra", {}) or {})
    return all(
        (
            _credential(extra, "page_id", "MESSENGER_PAGE_ID"),
            _credential(extra, "page_access_token", "MESSENGER_PAGE_ACCESS_TOKEN"),
            _credential(extra, "app_secret", "MESSENGER_APP_SECRET"),
            _credential(extra, "verify_token", "MESSENGER_VERIFY_TOKEN"),
        )
    )


def is_connected(config: PlatformConfig) -> bool:
    return validate_config(config)


def _env_enablement() -> Optional[Dict[str, Any]]:
    required = (
        "MESSENGER_PAGE_ID",
        "MESSENGER_PAGE_ACCESS_TOKEN",
        "MESSENGER_APP_SECRET",
        "MESSENGER_VERIFY_TOKEN",
    )
    if not all(_scoped_secret(name) for name in required):
        return None
    seeded: Dict[str, Any] = {}
    for env_name, key in (
        ("MESSENGER_HOST", "host"),
        ("MESSENGER_WEBHOOK_PATH", "path"),
        ("MESSENGER_API_VERSION", "api_version"),
        ("MESSENGER_COMMENT_TRIGGER", "comment_trigger"),
    ):
        if os.getenv(env_name):
            seeded[key] = os.environ[env_name]
    for env_name, key in (
        ("MESSENGER_PORT", "port"),
        ("MESSENGER_LOOKBACK_DAYS", "lookback_days"),
        ("MESSENGER_MAX_COMMENTS", "max_comments"),
    ):
        if os.getenv(env_name):
            seeded[key] = _coerce_int(os.environ[env_name], 0)
    return seeded


async def _standalone_send(
    pconfig: PlatformConfig,
    chat_id: str,
    message: str,
    *,
    thread_id: Optional[str] = None,
    media_files: Optional[List[str]] = None,
    force_document: bool = False,
) -> Dict[str, Any]:
    """Send a text-only cron notification without a live gateway adapter."""
    adapter = MessengerAdapter(pconfig)
    if httpx is None or not adapter.page_id or not adapter.page_access_token:
        return {"error": "Messenger standalone send: missing credentials"}
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            adapter._http_client = client
            result = await adapter.send(chat_id, message)
            return {"success": True, "message_id": result.message_id} if result.success else {"error": result.error}
    except Exception as exc:
        return {"error": str(exc)}


def register(ctx) -> None:
    ctx.register_platform(
        name="messenger",
        label="Facebook Messenger",
        adapter_factory=lambda cfg: MessengerAdapter(cfg),
        check_fn=check_requirements,
        validate_config=validate_config,
        is_connected=is_connected,
        required_env=[
            "MESSENGER_PAGE_ID",
            "MESSENGER_PAGE_ACCESS_TOKEN",
            "MESSENGER_APP_SECRET",
            "MESSENGER_VERIFY_TOKEN",
        ],
        install_hint="pip install aiohttp httpx",
        env_enablement_fn=_env_enablement,
        cron_deliver_env_var="MESSENGER_HOME_CHANNEL",
        standalone_sender_fn=_standalone_send,
        max_message_length=MessengerAdapter.MAX_MESSAGE_LENGTH,
        emoji="📨",
        platform_hint=(
            "You are chatting via Facebook Messenger. Keep replies concise and "
            "plain text; comments are handled by the Messenger page webhook."
        ),
    )

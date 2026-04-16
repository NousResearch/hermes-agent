"""Native Gmail Push gateway adapter.

Receives authenticated Cloud Pub/Sub push notifications for Gmail mailbox
changes, reconciles them through ``users.history.list`` + ``messages.get``,
and dispatches Hermes ``MessageEvent`` objects directly into the gateway.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import tempfile
import time
import uuid
from datetime import datetime, timezone
from email.utils import parseaddr
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from hermes_constants import get_hermes_home

try:
    from aiohttp import web

    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    web = None  # type: ignore[assignment]

try:
    from google.auth.transport.requests import Request as GoogleRequest
    from google.oauth2 import id_token as google_id_token
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build as google_build

    GOOGLE_CLIENT_AVAILABLE = True
except ImportError:
    GoogleRequest = None  # type: ignore[assignment]
    google_id_token = None  # type: ignore[assignment]
    Credentials = None  # type: ignore[assignment]
    google_build = None  # type: ignore[assignment]
    GOOGLE_CLIENT_AVAILABLE = False

try:
    from googleapiclient.errors import HttpError
except ImportError:
    HttpError = Exception  # type: ignore[misc,assignment]

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult

logger = logging.getLogger(__name__)

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8645
DEFAULT_PATH = "/gmail-push"
DEFAULT_HISTORY_TYPES = ["messageAdded"]
DEFAULT_INCLUDE_HEADERS = [
    "From",
    "To",
    "Subject",
    "Date",
    "List-Unsubscribe",
    "List-Id",
    "Precedence",
    "Auto-Submitted",
]
DEFAULT_FETCH_FORMAT = "full"
DEFAULT_MAX_BODY_CHARS = 20_000
DEFAULT_RENEW_EVERY_HOURS = 24
RECENT_DELIVERY_TTL_SECONDS = 24 * 3600
GMAIL_PUSH_OAUTH_SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


def check_gmail_push_requirements() -> bool:
    """Check whether the Gmail push adapter dependencies are available."""
    return AIOHTTP_AVAILABLE and GOOGLE_CLIENT_AVAILABLE


def gmail_push_account_slug(account: str) -> str:
    """Return a filesystem-safe slug for a Gmail account."""
    account = (account or "").strip().lower()
    if not account:
        return "default"
    slug = re.sub(r"[^a-z0-9._-]+", "-", account).strip("-")
    return slug or "default"


def gmail_push_account_paths(account: str) -> Dict[str, Path]:
    """Return the default profile-scoped storage paths for an account."""
    base_dir = get_hermes_home() / "integrations" / "gmail_push" / gmail_push_account_slug(account)
    return {
        "base_dir": base_dir,
        "token_path": base_dir / "token.json",
        "state_path": base_dir / "state.json",
        "recent_delivery_ids_path": base_dir / "recent_delivery_ids.json",
        "client_secret_path": base_dir / "client_secret.json",
    }


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _expand_path(value: str | None, default: Path) -> Path:
    if not value:
        return default
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return (get_hermes_home() / path).resolve()


def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=str(path.parent),
        delete=False,
    ) as tmp:
        json.dump(data, tmp, indent=2, ensure_ascii=False)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def _decode_base64url(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


def _strip_html(value: str) -> str:
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", value or "")
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</p\s*>", "\n\n", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = (
        text.replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
    )
    return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", text)).strip()


def _history_404(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if status == 404:
        return True
    resp = getattr(exc, "resp", None)
    if getattr(resp, "status", None) == 404:
        return True
    return "404" in str(exc)


class GmailPushAdapter(BasePlatformAdapter):
    """First-class Gmail push event-source adapter."""

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.GMAIL_PUSH)

        extra = config.extra or {}
        endpoint_cfg = extra.get("endpoint") or {}
        oauth_cfg = extra.get("oauth") or {}
        watch_cfg = extra.get("watch") or {}
        push_auth_cfg = extra.get("push_auth") or {}
        processing_cfg = extra.get("processing") or {}
        state_cfg = extra.get("state") or {}

        self._account: str = str(extra.get("account") or "").strip()
        self._topic: str = str(extra.get("topic") or "").strip()
        self._subscription: str = str(extra.get("subscription") or "").strip()

        self._host: str = str(endpoint_cfg.get("host") or DEFAULT_HOST)
        self._port: int = int(endpoint_cfg.get("port") or DEFAULT_PORT)
        raw_path = str(endpoint_cfg.get("path") or DEFAULT_PATH).strip() or DEFAULT_PATH
        self._path: str = raw_path if raw_path.startswith("/") else f"/{raw_path}"
        self._public_url: str = str(endpoint_cfg.get("public_url") or "").strip()

        paths = gmail_push_account_paths(self._account)
        self._client_secret_path = _expand_path(
            oauth_cfg.get("client_secret_path"),
            paths["client_secret_path"],
        )
        self._token_path = _expand_path(oauth_cfg.get("token_path"), paths["token_path"])
        self._state_path = _expand_path(
            state_cfg.get("path"),
            paths["state_path"],
        )
        self._recent_delivery_ids_path = paths["recent_delivery_ids_path"]
        if not self._recent_delivery_ids_path.is_absolute():
            self._recent_delivery_ids_path = (self._state_path.parent / self._recent_delivery_ids_path).resolve()
        else:
            self._recent_delivery_ids_path = self._recent_delivery_ids_path.resolve()

        self._label_ids: list[str] = list(watch_cfg.get("label_ids") or ["INBOX"])
        self._label_filter_behavior: str = str(
            watch_cfg.get("label_filter_behavior") or "INCLUDE"
        ).upper()
        self._renew_every_hours: int = int(
            watch_cfg.get("renew_every_hours") or DEFAULT_RENEW_EVERY_HOURS
        )

        self._push_service_account_email: str = str(
            push_auth_cfg.get("service_account_email") or ""
        ).strip()
        self._push_audience: str = str(
            push_auth_cfg.get("audience") or self._public_url or ""
        ).strip()

        self._history_types: list[str] = list(
            processing_cfg.get("history_types") or DEFAULT_HISTORY_TYPES
        )
        self._fetch_format: str = str(
            processing_cfg.get("fetch_format") or DEFAULT_FETCH_FORMAT
        ).lower()
        self._include_headers: list[str] = list(
            processing_cfg.get("include_headers") or DEFAULT_INCLUDE_HEADERS
        )
        self._include_html: bool = bool(processing_cfg.get("include_html", False))
        self._max_body_chars: int = int(
            processing_cfg.get("max_body_chars") or DEFAULT_MAX_BODY_CHARS
        )

        self._runner = None
        self._renewal_task: Optional[asyncio.Task] = None
        self._gmail_service = None
        self._recent_delivery_ids: Dict[str, float] = {}
        self._response_log: list[dict[str, Any]] = []
        self._state: Dict[str, Any] = {
            "account": self._account,
            "last_history_id": None,
            "watch_expiration_ms": None,
            "last_watch_renewed_at": None,
            "last_notification_at": None,
            "last_error": None,
            "last_successful_pubsub_message_id": None,
            "degraded": False,
        }
        self._load_state()
        self._load_recent_delivery_ids()

    async def connect(self) -> bool:
        """Start the HTTP listener and establish/renew the Gmail watch."""
        if not check_gmail_push_requirements():
            self._set_fatal_error(
                "missing_deps",
                "Gmail push requires aiohttp and google-api-python-client/google-auth packages.",
                retryable=False,
            )
            return False

        config_error = self._validate_required_config()
        if config_error:
            self._set_fatal_error("config", config_error, retryable=False)
            return False

        try:
            self._load_credentials()
        except Exception as exc:
            self._set_fatal_error("auth", f"Gmail OAuth credentials unavailable: {exc}", retryable=False)
            return False

        if self._port and self._port > 0:
            import socket as _socket

            try:
                with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as sock:
                    sock.settimeout(1)
                    sock.connect(("127.0.0.1", self._port))
                self._set_fatal_error(
                    "port_in_use",
                    f"Gmail push port {self._port} is already in use.",
                    retryable=False,
                )
                return False
            except (ConnectionRefusedError, OSError):
                pass

        app = web.Application()
        app.router.add_get("/health", self._handle_health)
        app.router.add_post(self._path, self._handle_push)

        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self._host, self._port)
        await site.start()

        try:
            await self.refresh_watch_now()
        except Exception as exc:
            await self._cleanup_server()
            self._set_fatal_error("watch", f"Failed to establish Gmail watch: {exc}", retryable=True)
            return False

        self._renewal_task = asyncio.create_task(self._renewal_loop())
        self._mark_connected()
        logger.info(
            "[gmail_push] Listening on %s:%s%s for %s",
            self._host,
            self._port,
            self._path,
            self._account,
        )
        return True

    async def disconnect(self) -> None:
        """Stop the HTTP listener and flush durable state."""
        self._running = False
        if self._renewal_task:
            self._renewal_task.cancel()
            try:
                await self._renewal_task
            except asyncio.CancelledError:
                pass
            self._renewal_task = None

        await self._cleanup_server()
        self._save_state()
        self._save_recent_delivery_ids()
        self._mark_disconnected()
        logger.info("[gmail_push] Disconnected")

    async def _cleanup_server(self) -> None:
        if self._runner:
            await self._runner.cleanup()
            self._runner = None

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Log/store the response instead of sending email."""
        self._response_log.append(
            {
                "chat_id": chat_id,
                "content": content,
                "reply_to": reply_to,
                "metadata": metadata or {},
                "timestamp": _utc_now_iso(),
            }
        )
        logger.info("[gmail_push] Response for %s: %s", chat_id, content[:200])
        return SendResult(success=True, message_id=uuid.uuid4().hex[:12])

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """Gmail push is non-interactive and does not support typing."""

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {
            "name": f"Gmail Push ({self._account})",
            "type": "channel",
            "path": self._path,
        }

    async def refresh_watch_now(self, *, clear_health_state: bool = True) -> Dict[str, Any]:
        """Force a Gmail watch refresh and persist the new baseline."""
        watch_result = await asyncio.to_thread(self._watch_mailbox)
        self._apply_watch_result(
            watch_result,
            clear_health_state=clear_health_state,
        )
        return watch_result

    async def rebaseline(self) -> Dict[str, Any]:
        """Reset degraded state and establish a fresh watch baseline."""
        self._state["degraded"] = False
        self._state["last_error"] = None
        return await self.refresh_watch_now(clear_health_state=True)

    async def run_health_check(self) -> Dict[str, Any]:
        """Return a health snapshot without starting the HTTP server."""
        issues = []
        if not AIOHTTP_AVAILABLE:
            issues.append("aiohttp is not installed")
        if not GOOGLE_CLIENT_AVAILABLE:
            issues.append("Google API client libraries are not installed")
        config_error = self._validate_required_config()
        if config_error:
            issues.append(config_error)
        if not self._token_path.exists():
            issues.append(f"OAuth token not found at {self._token_path}")
        else:
            try:
                self._load_credentials()
            except Exception as exc:
                issues.append(f"OAuth token is invalid: {exc}")
        return {
            "ok": not issues,
            "issues": issues,
            "account": self._account,
            "state": dict(self._state),
            "endpoint": {
                "host": self._host,
                "port": self._port,
                "path": self._path,
                "public_url": self._public_url,
            },
        }

    async def _handle_health(self, request: "web.Request") -> "web.Response":
        return web.json_response({"status": "ok", "platform": "gmail_push"})

    async def _handle_push(self, request: "web.Request") -> "web.Response":
        try:
            bearer = self._extract_bearer_token(request.headers.get("Authorization", ""))
            await asyncio.to_thread(self._verify_pubsub_bearer_token, bearer)
        except PermissionError as exc:
            return web.json_response({"error": str(exc)}, status=401)
        except Exception as exc:
            logger.warning("[gmail_push] JWT verification failed: %s", exc)
            return web.json_response({"error": str(exc)}, status=401)

        try:
            envelope = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON payload"}, status=400)

        try:
            notification = self._parse_pubsub_envelope(envelope)
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=400)

        if notification["email_address"].lower() != self._account.lower():
            return web.json_response(
                {"error": f"Unexpected Gmail account: {notification['email_address']}"},
                status=400,
            )

        message_id = notification["pubsub_message_id"]
        if self._recent_delivery_ids.get(message_id):
            logger.info("[gmail_push] Skipping duplicate Pub/Sub delivery %s", message_id)
            return web.Response(status=204)

        try:
            events = await self._reconcile_notification(notification)
        except Exception as exc:
            logger.exception("[gmail_push] Failed to process push notification: %s", exc)
            self._state["last_error"] = str(exc)
            self._save_state()
            return web.json_response({"error": "Failed to process notification"}, status=500)

        for event in events:
            task = asyncio.create_task(self.handle_message(event))
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

        self._recent_delivery_ids[message_id] = time.time()
        self._prune_recent_delivery_ids()
        self._save_recent_delivery_ids()
        self._state["last_successful_pubsub_message_id"] = message_id
        self._save_state()
        return web.Response(status=204)

    async def _reconcile_notification(self, notification: Dict[str, Any]) -> list[MessageEvent]:
        old_history_id = self._state.get("last_history_id")
        notification_history_id = str(notification.get("history_id") or "")
        self._state["last_notification_at"] = _utc_now_iso()

        if not old_history_id:
            self._state["last_history_id"] = notification_history_id or old_history_id
            self._state["last_error"] = None
            self._save_state()
            return []

        try:
            history_pages = await asyncio.to_thread(
                self._fetch_history_pages,
                str(old_history_id),
            )
        except Exception as exc:
            if _history_404(exc):
                logger.warning("[gmail_push] Stale history cursor for %s — rebaselining", self._account)
                await self.refresh_watch_now(clear_health_state=False)
                self._state["degraded"] = True
                self._state["last_error"] = (
                    "Stale Gmail history cursor — watch rebaselined; historical backfill skipped"
                )
                self._save_state()
                return []
            raise

        message_ids: list[str] = []
        seen_ids: set[str] = set()
        high_water = notification_history_id or str(old_history_id)
        for page in history_pages:
            high_water = str(page.get("historyId") or high_water)
            for history in page.get("history", []) or []:
                for added in history.get("messagesAdded", []) or []:
                    message = added.get("message") or {}
                    message_id = str(message.get("id") or "").strip()
                    if message_id and message_id not in seen_ids:
                        seen_ids.add(message_id)
                        message_ids.append(message_id)

        events: list[MessageEvent] = []
        for gmail_message_id in message_ids:
            gmail_message = await asyncio.to_thread(self._get_message, gmail_message_id)
            normalized = self._normalize_message(
                gmail_message,
                history_id=high_water,
                pubsub_message_id=notification["pubsub_message_id"],
            )
            events.append(self._build_message_event(normalized, gmail_message))

        self._state["last_history_id"] = high_water
        self._state["degraded"] = False
        self._state["last_error"] = None
        self._save_state()
        return events

    def _fetch_history_pages(self, start_history_id: str) -> list[Dict[str, Any]]:
        pages = []
        page_token = None
        while True:
            page = self._list_history(start_history_id=start_history_id, page_token=page_token)
            pages.append(page)
            page_token = page.get("nextPageToken")
            if not page_token:
                break
        return pages

    def _watch_mailbox(self) -> Dict[str, Any]:
        service = self._get_gmail_service()
        body: Dict[str, Any] = {"topicName": self._topic}
        if self._label_ids:
            body["labelIds"] = list(self._label_ids)
            body["labelFilterBehavior"] = self._label_filter_behavior
        return (
            service.users()
            .watch(userId="me", body=body)
            .execute()
        )

    def _list_history(self, start_history_id: str, page_token: str | None = None) -> Dict[str, Any]:
        service = self._get_gmail_service()
        params: Dict[str, Any] = {
            "userId": "me",
            "startHistoryId": str(start_history_id),
            "historyTypes": list(self._history_types),
        }
        if page_token:
            params["pageToken"] = page_token
        return service.users().history().list(**params).execute()

    def _get_message(self, message_id: str) -> Dict[str, Any]:
        service = self._get_gmail_service()
        params: Dict[str, Any] = {
            "userId": "me",
            "id": message_id,
            "format": self._fetch_format,
        }
        if self._fetch_format == "metadata":
            params["metadataHeaders"] = list(self._include_headers)
        return service.users().messages().get(**params).execute()

    def _get_gmail_service(self):
        if self._gmail_service is None:
            creds = self._load_credentials()
            self._gmail_service = google_build("gmail", "v1", credentials=creds, cache_discovery=False)
        return self._gmail_service

    def _load_credentials(self):
        if not GOOGLE_CLIENT_AVAILABLE:
            raise RuntimeError("Google API client libraries are not installed")
        if not self._token_path.exists():
            raise FileNotFoundError(self._token_path)
        creds = Credentials.from_authorized_user_file(str(self._token_path))
        if creds.expired and creds.refresh_token:
            creds.refresh(GoogleRequest())
            self._token_path.write_text(creds.to_json(), encoding="utf-8")
        if not creds.valid:
            raise RuntimeError("OAuth token is invalid or expired")
        return creds

    def _apply_watch_result(
        self,
        watch_result: Dict[str, Any],
        *,
        clear_health_state: bool = True,
    ) -> None:
        history_id = str(watch_result.get("historyId") or self._state.get("last_history_id") or "")
        expiration = watch_result.get("expiration")
        state_update = {
            "account": self._account,
            "last_history_id": history_id or None,
            "watch_expiration_ms": int(expiration) if expiration else None,
            "last_watch_renewed_at": _utc_now_iso(),
        }
        if clear_health_state:
            state_update["last_error"] = None
            state_update["degraded"] = False
        self._state.update(state_update)
        self._save_state()

    def _validate_required_config(self) -> str | None:
        missing = []
        if not self._account:
            missing.append("account")
        if not self._topic:
            missing.append("topic")
        if not self._subscription:
            missing.append("subscription")
        if not self._push_service_account_email:
            missing.append("push_auth.service_account_email")
        if not self._push_audience:
            missing.append("push_auth.audience or endpoint.public_url")
        if missing:
            return f"Missing Gmail push config: {', '.join(missing)}"
        return None

    def _extract_bearer_token(self, authorization_header: str) -> str:
        header = (authorization_header or "").strip()
        if not header.lower().startswith("bearer "):
            raise PermissionError("Missing Bearer token")
        token = header.split(" ", 1)[1].strip()
        if not token:
            raise PermissionError("Empty Bearer token")
        return token

    def _verify_pubsub_bearer_token(self, bearer_token: str) -> Dict[str, Any]:
        if not GOOGLE_CLIENT_AVAILABLE:
            raise RuntimeError("Google auth libraries are not installed")
        verified = google_id_token.verify_oauth2_token(
            bearer_token,
            GoogleRequest(),
            audience=self._push_audience,
        )
        if not verified.get("email_verified"):
            raise PermissionError("Pub/Sub JWT email is not verified")
        expected_email = self._push_service_account_email
        if expected_email and verified.get("email") != expected_email:
            raise PermissionError(
                f"Unexpected Pub/Sub service account: {verified.get('email')}"
            )
        return verified

    def _parse_pubsub_envelope(self, envelope: Dict[str, Any]) -> Dict[str, Any]:
        message = envelope.get("message")
        if not isinstance(message, dict):
            raise ValueError("Missing Pub/Sub message envelope")

        encoded_data = message.get("data")
        if not encoded_data:
            raise ValueError("Pub/Sub message missing data")

        try:
            payload = json.loads(_decode_base64url(str(encoded_data)).decode("utf-8"))
        except Exception as exc:
            raise ValueError(f"Invalid Pub/Sub data payload: {exc}") from exc

        email_address = str(payload.get("emailAddress") or "").strip()
        history_id = str(payload.get("historyId") or "").strip()
        if not email_address or not history_id:
            raise ValueError("Pub/Sub Gmail payload missing emailAddress/historyId")

        pubsub_message_id = str(
            message.get("messageId")
            or message.get("message_id")
            or envelope.get("messageId")
            or ""
        ).strip()
        if not pubsub_message_id:
            raise ValueError("Pub/Sub message missing messageId")

        return {
            "email_address": email_address,
            "history_id": history_id,
            "pubsub_message_id": pubsub_message_id,
            "publish_time": message.get("publishTime") or message.get("publish_time"),
            "raw_envelope": envelope,
        }

    def _normalize_message(
        self,
        gmail_message: Dict[str, Any],
        *,
        history_id: str,
        pubsub_message_id: str,
    ) -> Dict[str, Any]:
        payload = gmail_message.get("payload") or {}
        headers = {
            str(item.get("name")): str(item.get("value"))
            for item in (payload.get("headers") or [])
            if item.get("name")
        }
        selected_headers = {
            name: headers[name]
            for name in self._include_headers
            if name in headers
        }
        plain_text_parts, html_parts = self._extract_body_parts(payload)
        plain_text = "\n\n".join(part for part in plain_text_parts if part).strip()
        html_body = "\n\n".join(part for part in html_parts if part).strip()
        if plain_text:
            body_text = plain_text
        elif html_body:
            body_text = _strip_html(html_body)
        else:
            body_text = str(gmail_message.get("snippet") or "").strip()

        if self._max_body_chars > 0:
            body_text = body_text[: self._max_body_chars]
            if html_body:
                html_body = html_body[: self._max_body_chars]

        from_value = headers.get("From", "")
        _, from_email = parseaddr(from_value)
        return {
            "source": "gmail_push",
            "account": self._account,
            "pubsub_message_id": pubsub_message_id,
            "history_id": history_id,
            "gmail_message": {
                "id": gmail_message.get("id"),
                "thread_id": gmail_message.get("threadId"),
                "label_ids": list(gmail_message.get("labelIds") or []),
                "internal_date_ms": int(gmail_message.get("internalDate") or 0),
                "snippet": gmail_message.get("snippet"),
                "subject": headers.get("Subject", ""),
                "from": from_value,
                "from_email": from_email,
                "headers": selected_headers,
                "body_text": body_text,
                "body_html": html_body if self._include_html else None,
            },
        }

    def _build_message_event(self, normalized: Dict[str, Any], gmail_message: Dict[str, Any]) -> MessageEvent:
        gm = normalized["gmail_message"]
        internal_ms = int(gm.get("internal_date_ms") or 0)
        if internal_ms > 0:
            when = datetime.fromtimestamp(internal_ms / 1000, tz=timezone.utc)
            when_text = when.replace(microsecond=0).isoformat().replace("+00:00", "Z")
        else:
            when_text = ""
        label_text = ", ".join(gm.get("label_ids") or []) or "(none)"
        body_text = gm.get("body_text") or gm.get("snippet") or "(no body text available)"
        prompt = (
            f"New Gmail message for {normalized['account']}\n\n"
            f"From: {gm.get('from') or '(unknown)'}\n"
            f"Subject: {gm.get('subject') or '(no subject)'}\n"
            f"Labels: {label_text}\n"
            f"Date: {when_text or '(unknown)'}\n"
            f"Snippet: {gm.get('snippet') or ''}\n\n"
            f"Body:\n{body_text}"
        ).strip()
        source = self.build_source(
            chat_id=f"gmail_push:{self._account}:{gm.get('id')}",
            chat_name=f"Gmail Push {self._account}",
            chat_type="channel",
            user_id=self._account,
            user_name=self._account,
        )
        return MessageEvent(
            text=prompt,
            message_type=MessageType.TEXT,
            source=source,
            raw_message={
                "normalized": normalized,
                "gmail_raw": gmail_message,
            },
            message_id=str(gm.get("id") or ""),
        )

    def _extract_body_parts(self, payload: Dict[str, Any]) -> tuple[list[str], list[str]]:
        plain_parts: list[str] = []
        html_parts: list[str] = []

        def _walk(part: Dict[str, Any]) -> None:
            mime_type = str(part.get("mimeType") or "").lower()
            body = part.get("body") or {}
            data = body.get("data")
            if data:
                try:
                    decoded = _decode_base64url(str(data)).decode("utf-8", errors="replace")
                except Exception:
                    decoded = ""
                if mime_type == "text/plain":
                    plain_parts.append(decoded)
                elif mime_type == "text/html":
                    html_parts.append(decoded)
            for child in part.get("parts") or []:
                if isinstance(child, dict):
                    _walk(child)

        _walk(payload)
        return plain_parts, html_parts

    def _load_state(self) -> None:
        if not self._state_path.exists():
            return
        try:
            data = json.loads(self._state_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                self._state.update(data)
        except Exception as exc:
            logger.warning("[gmail_push] Failed to load state %s: %s", self._state_path, exc)

    def _save_state(self) -> None:
        self._state["account"] = self._account
        _atomic_write_json(self._state_path, self._state)

    def _load_recent_delivery_ids(self) -> None:
        if not self._recent_delivery_ids_path.exists():
            return
        try:
            data = json.loads(self._recent_delivery_ids_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                self._recent_delivery_ids = {
                    str(key): float(value)
                    for key, value in data.items()
                    if isinstance(key, str)
                }
                self._prune_recent_delivery_ids()
        except Exception as exc:
            logger.warning(
                "[gmail_push] Failed to load delivery id cache %s: %s",
                self._recent_delivery_ids_path,
                exc,
            )

    def _save_recent_delivery_ids(self) -> None:
        self._prune_recent_delivery_ids()
        _atomic_write_json(
            self._recent_delivery_ids_path,
            {key: value for key, value in self._recent_delivery_ids.items()},
        )

    def _prune_recent_delivery_ids(self) -> None:
        cutoff = time.time() - RECENT_DELIVERY_TTL_SECONDS
        self._recent_delivery_ids = {
            key: value
            for key, value in self._recent_delivery_ids.items()
            if value >= cutoff
        }

    async def _renewal_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(self._seconds_until_next_renewal())
                await self.refresh_watch_now()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("[gmail_push] Watch renewal failed: %s", exc)
                self._state["last_error"] = f"Watch renewal failed: {exc}"
                self._save_state()

    def _seconds_until_next_renewal(self) -> float:
        expiration_ms = self._state.get("watch_expiration_ms")
        renew_interval = max(self._renew_every_hours, 1) * 3600
        if not expiration_ms:
            return float(renew_interval)
        expiration_seconds = max((int(expiration_ms) / 1000) - time.time(), 300)
        return float(min(renew_interval, max(expiration_seconds - 300, 300)))

"""BlueBubbles iMessage platform adapter.

Uses the local BlueBubbles macOS server for outbound REST sends and inbound
webhooks.  Supports text messaging, media attachments (images, voice, video,
documents), tapback reactions, typing indicators, and read receipts.

Architecture based on PR #5869 (benjaminsehl) with inbound attachment
downloading from PR #4588 (YuhangLin).
"""

import asyncio
import json
import logging
import os
import re
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import httpx

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    cache_image_from_bytes,
    cache_audio_from_bytes,
    cache_document_from_bytes,
)
from gateway.platforms.helpers import strip_markdown

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_WEBHOOK_HOST = "127.0.0.1"
DEFAULT_WEBHOOK_PORT = 8645
DEFAULT_WEBHOOK_PATH = "/bluebubbles-webhook"
MAX_TEXT_LENGTH = 4000
DEFAULT_POLL_INTERVAL_SECONDS = 3.0

# Tapback reaction codes (BlueBubbles associatedMessageType values)
_TAPBACK_ADDED = {
    2000: "love", 2001: "like", 2002: "dislike",
    2003: "laugh", 2004: "emphasize", 2005: "question",
}
_TAPBACK_REMOVED = {
    3000: "love", 3001: "like", 3002: "dislike",
    3003: "laugh", 3004: "emphasize", 3005: "question",
}

# Webhook event types that carry user messages
_MESSAGE_EVENTS = {"new-message", "message", "updated-message"}

# Log redaction patterns
_PHONE_RE = re.compile(r"\+?\d{7,15}")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")


def _redact(text: str) -> str:
    """Redact phone numbers and emails from log output."""
    text = _PHONE_RE.sub("[REDACTED]", text)
    text = _EMAIL_RE.sub("[REDACTED]", text)
    return text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def check_bluebubbles_requirements() -> bool:
    try:
        import aiohttp  # noqa: F401
        import httpx  # noqa: F401
    except ImportError:
        return False
    return True


def _normalize_server_url(raw: str) -> str:
    value = (raw or "").strip()
    if not value:
        return ""
    if not re.match(r"^https?://", value, flags=re.I):
        value = f"http://{value}"
    return value.rstrip("/")





# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class BlueBubblesAdapter(BasePlatformAdapter):
    platform = Platform.BLUEBUBBLES
    SUPPORTS_MESSAGE_EDITING = False
    MAX_MESSAGE_LENGTH = MAX_TEXT_LENGTH

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.BLUEBUBBLES)
        extra = config.extra or {}
        self.server_url = _normalize_server_url(
            extra.get("server_url") or os.getenv("BLUEBUBBLES_SERVER_URL", "")
        )
        self.password = extra.get("password") or os.getenv("BLUEBUBBLES_PASSWORD", "")
        self.webhook_host = (
            extra.get("webhook_host")
            or os.getenv("BLUEBUBBLES_WEBHOOK_HOST", DEFAULT_WEBHOOK_HOST)
        )
        self.webhook_port = int(
            extra.get("webhook_port")
            or os.getenv("BLUEBUBBLES_WEBHOOK_PORT", str(DEFAULT_WEBHOOK_PORT))
        )
        self.webhook_path = (
            extra.get("webhook_path")
            or os.getenv("BLUEBUBBLES_WEBHOOK_PATH", DEFAULT_WEBHOOK_PATH)
        )
        if not str(self.webhook_path).startswith("/"):
            self.webhook_path = f"/{self.webhook_path}"
        self.send_read_receipts = bool(extra.get("send_read_receipts", True))
        self.client: Optional[httpx.AsyncClient] = None
        self._runner = None
        self._private_api_enabled: Optional[bool] = None
        self._helper_connected: bool = False
        self._guid_cache: Dict[str, str] = {}
        self._poll_task: Optional[asyncio.Task] = None
        self._poll_checkpoint: int = 0
        self._seen_inbound_message_ids: set[str] = set()
        self._poll_interval = float(
            extra.get("poll_interval")
            or os.getenv("BLUEBUBBLES_POLL_INTERVAL", str(DEFAULT_POLL_INTERVAL_SECONDS))
        )
        self._poll_enabled = str(
            extra.get("poll_messages")
            or os.getenv("BLUEBUBBLES_POLL_MESSAGES", "true")
        ).lower() in ("true", "1", "yes")
        self._poll_allowed_users = self._parse_allowed_users(
            str(extra.get("allowed_users") or os.getenv("BLUEBUBBLES_ALLOWED_USERS", ""))
        )
        self._messages_db_path = Path.home() / "Library" / "Messages" / "chat.db"

    # ------------------------------------------------------------------
    # API helpers
    # ------------------------------------------------------------------

    def _api_url(self, path: str) -> str:
        sep = "&" if "?" in path else "?"
        return f"{self.server_url}{path}{sep}password={quote(self.password, safe='')}"

    async def _api_get(self, path: str) -> Dict[str, Any]:
        assert self.client is not None
        res = await self.client.get(self._api_url(path))
        res.raise_for_status()
        return res.json()

    async def _api_post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        assert self.client is not None
        res = await self.client.post(self._api_url(path), json=payload)
        res.raise_for_status()
        return res.json()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        if not self.server_url or not self.password:
            logger.error(
                "[bluebubbles] BLUEBUBBLES_SERVER_URL and BLUEBUBBLES_PASSWORD are required"
            )
            return False
        from aiohttp import web

        # Tighter keepalive so idle CLOSE_WAIT drains promptly (#18451).
        from gateway.platforms._http_client_limits import platform_httpx_limits
        self.client = httpx.AsyncClient(timeout=30.0, limits=platform_httpx_limits())
        try:
            await self._api_get("/api/v1/ping")
            info = await self._api_get("/api/v1/server/info")
            server_data = (info or {}).get("data", {})
            self._private_api_enabled = bool(server_data.get("private_api"))
            self._helper_connected = bool(server_data.get("helper_connected"))
            logger.info(
                "[bluebubbles] connected to %s (private_api=%s, helper=%s)",
                self.server_url,
                self._private_api_enabled,
                self._helper_connected,
            )
        except Exception as exc:
            logger.error(
                "[bluebubbles] cannot reach server at %s: %s", self.server_url, exc
            )
            if self.client:
                await self.client.aclose()
                self.client = None
            return False

        app = web.Application()
        app.router.add_get("/health", lambda _: web.Response(text="ok"))
        app.router.add_post(self.webhook_path, self._handle_webhook)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.webhook_host, self.webhook_port)
        await site.start()
        self._mark_connected()
        logger.info(
            "[bluebubbles] webhook listening on http://%s:%s%s",
            self.webhook_host,
            self.webhook_port,
            self.webhook_path,
        )

        # Register webhook with BlueBubbles server
        # This is required for the server to know where to send events
        await self._register_webhook()
        await self._start_message_polling()

        return True

    async def disconnect(self) -> None:
        await self._stop_message_polling()

        # Unregister webhook before cleaning up
        await self._unregister_webhook()

        if self.client:
            await self.client.aclose()
            self.client = None
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
        self._mark_disconnected()

    @property
    def _webhook_url(self) -> str:
        """Compute the external webhook URL for BlueBubbles registration."""
        host = self.webhook_host
        if host in {"0.0.0.0", "127.0.0.1", "localhost", "::"}:
            host = "localhost"
        return f"http://{host}:{self.webhook_port}{self.webhook_path}"

    @property
    def _webhook_register_url(self) -> str:
        """Webhook URL registered with BlueBubbles, including the password as
        a query param so inbound webhook POSTs carry credentials.

        BlueBubbles posts events to the exact URL registered via
        ``/api/v1/webhook``. Its webhook registration API does not support
        custom headers, so embedding the password in the URL is the only
        way to authenticate inbound webhooks without disabling auth.
        """
        base = self._webhook_url
        if self.password:
            return f"{base}?password={quote(self.password, safe='')}"
        return base

    async def _find_registered_webhooks(self, url: str) -> list:
        """Return list of BB webhook entries matching *url*."""
        try:
            res = await self._api_get("/api/v1/webhook")
            data = res.get("data")
            if isinstance(data, list):
                return [wh for wh in data if wh.get("url") == url]
        except Exception:
            pass
        return []

    async def _register_webhook(self) -> bool:
        """Register this webhook URL with the BlueBubbles server.

        BlueBubbles requires webhooks to be registered via API before
        it will send events.  Checks for an existing registration first
        to avoid duplicates (e.g. after a crash without clean shutdown).
        """
        if not self.client:
            return False

        webhook_url = self._webhook_register_url

        # Crash resilience — reuse an existing registration if present
        existing = await self._find_registered_webhooks(webhook_url)
        if existing:
            logger.info(
                "[bluebubbles] webhook already registered: %s", self._webhook_url
            )
            return True

        payload = {
            "url": webhook_url,
            "events": ["new-message", "updated-message"],
        }

        try:
            res = await self._api_post("/api/v1/webhook", payload)
            status = res.get("status", 0)
            if 200 <= status < 300:
                logger.info(
                    "[bluebubbles] webhook registered with server: %s",
                    self._webhook_url,
                )
                return True
            else:
                logger.warning(
                    "[bluebubbles] webhook registration returned status %s: %s",
                    status,
                    res.get("message"),
                )
                return False
        except Exception as exc:
            logger.warning(
                "[bluebubbles] failed to register webhook with server: %s",
                exc,
            )
            return False

    async def _unregister_webhook(self) -> bool:
        """Unregister this webhook URL from the BlueBubbles server.

        Removes *all* matching registrations to clean up any duplicates
        left by prior crashes.
        """
        if not self.client:
            return False

        webhook_url = self._webhook_register_url
        removed = False

        try:
            for wh in await self._find_registered_webhooks(webhook_url):
                wh_id = wh.get("id")
                if wh_id:
                    res = await self.client.delete(
                        self._api_url(f"/api/v1/webhook/{wh_id}")
                    )
                    res.raise_for_status()
                    removed = True
            if removed:
                logger.info(
                    "[bluebubbles] webhook unregistered: %s", self._webhook_url
                )
        except Exception as exc:
            logger.debug(
                "[bluebubbles] failed to unregister webhook (non-critical): %s",
                exc,
            )
        return removed

    # ------------------------------------------------------------------
    # Local Messages DB polling fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_allowed_users(raw: str) -> set[str]:
        return {item.strip().lower() for item in raw.split(",") if item.strip()}

    @staticmethod
    def _user_matches_allowed(value: Optional[str], allowed: set[str]) -> bool:
        if not value:
            return False
        raw = value.strip().lower()
        phoneish = re.sub(r"\D", "", raw)
        for item in allowed:
            if raw == item:
                return True
            item_phoneish = re.sub(r"\D", "", item)
            if phoneish and item_phoneish and phoneish == item_phoneish:
                return True
        return False

    def _latest_messages_date(self) -> int:
        try:
            db_uri = f"file:{quote(str(self._messages_db_path), safe='/')}?mode=ro"
            conn = sqlite3.connect(db_uri, uri=True)
            try:
                row = conn.execute("SELECT COALESCE(MAX(date), 0) FROM message").fetchone()
                return int(row[0] or 0)
            finally:
                conn.close()
        except Exception as exc:
            logger.debug("[bluebubbles] message poll checkpoint unavailable: %s", exc)
            return 0

    def _poll_messages_db(self, since_date: int) -> tuple[int, List[Dict[str, Any]]]:
        db_uri = f"file:{quote(str(self._messages_db_path), safe='/')}?mode=ro"
        conn = sqlite3.connect(db_uri, uri=True)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                """
                SELECT
                    message.ROWID AS rowid,
                    message.date AS date,
                    message.guid AS message_guid,
                    message.text AS text,
                    handle.id AS sender,
                    chat.guid AS chat_guid,
                    chat.chat_identifier AS chat_identifier
                FROM message
                LEFT JOIN handle ON message.handle_id = handle.ROWID
                LEFT JOIN chat_message_join cmj ON cmj.message_id = message.ROWID
                LEFT JOIN chat ON cmj.chat_id = chat.ROWID
                WHERE message.date > ?
                  AND message.is_from_me = 0
                  AND message.text IS NOT NULL
                  AND TRIM(message.text) != ''
                ORDER BY message.date ASC
                LIMIT 100
                """,
                (since_date,),
            ).fetchall()
        finally:
            conn.close()

        newest = since_date
        records: List[Dict[str, Any]] = []
        for row in rows:
            newest = max(newest, int(row["date"] or 0))
            sender = row["sender"] or ""
            chat_identifier = row["chat_identifier"] or ""
            if self._poll_allowed_users and not (
                self._user_matches_allowed(sender, self._poll_allowed_users)
                or self._user_matches_allowed(chat_identifier, self._poll_allowed_users)
            ):
                continue
            records.append(
                {
                    "rowid": str(row["rowid"]),
                    "date": int(row["date"] or 0),
                    "guid": row["message_guid"] or f"chatdb-{row['rowid']}",
                    "text": row["text"] or "",
                    "sender": sender,
                    "chatGuid": row["chat_guid"] or chat_identifier,
                    "chatIdentifier": chat_identifier,
                    "source": "messages-db-poll",
                }
            )
        return newest, records

    async def _start_message_polling(self) -> None:
        if not self._poll_enabled:
            logger.info("[bluebubbles] local Messages polling disabled")
            return
        if not self._poll_allowed_users:
            logger.info(
                "[bluebubbles] local Messages polling disabled: BLUEBUBBLES_ALLOWED_USERS is empty"
            )
            return
        self._poll_checkpoint = await self._latest_message_api_date()
        self._poll_task = asyncio.create_task(self._poll_messages_loop())
        logger.info(
            "[bluebubbles] BlueBubbles message polling enabled for %d allowed sender(s)",
            len(self._poll_allowed_users),
        )

    async def _stop_message_polling(self) -> None:
        if not self._poll_task:
            return
        self._poll_task.cancel()
        try:
            await self._poll_task
        except asyncio.CancelledError:
            pass
        self._poll_task = None

    async def _poll_messages_loop(self) -> None:
        while True:
            try:
                newest, records = await self._poll_messages_api(self._poll_checkpoint)
                self._poll_checkpoint = max(self._poll_checkpoint, newest)
                for record in records:
                    await self._handle_inbound_record(record, {"source": "bluebubbles-api-poll"})
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("[bluebubbles] BlueBubbles message poll failed: %s", exc)
            await asyncio.sleep(self._poll_interval)

    async def _latest_message_api_date(self) -> int:
        try:
            payload = await self._api_post(
                "/api/v1/message/query",
                {"limit": 1, "offset": 0, "with": ["handle", "chats"]},
            )
            rows = payload.get("data") or []
            if rows:
                return int(rows[0].get("dateCreated") or 0)
        except Exception as exc:
            logger.debug("[bluebubbles] message poll checkpoint unavailable: %s", exc)
        return 0

    async def _poll_messages_api(self, since_date: int) -> tuple[int, List[Dict[str, Any]]]:
        payload = await self._api_post(
            "/api/v1/message/query",
            {"limit": 50, "offset": 0, "with": ["handle", "chats"]},
        )
        rows = payload.get("data") or []
        newest = since_date
        records: List[Dict[str, Any]] = []
        for row in sorted(rows, key=lambda item: int(item.get("dateCreated") or 0)):
            date_created = int(row.get("dateCreated") or 0)
            newest = max(newest, date_created)
            if date_created <= since_date or row.get("isFromMe"):
                continue
            text = row.get("text") or ""
            if not text.strip():
                continue
            handle = row.get("handle") if isinstance(row.get("handle"), dict) else {}
            sender = handle.get("address") or ""
            chats = row.get("chats") or []
            chat = chats[0] if chats and isinstance(chats[0], dict) else {}
            chat_guid = chat.get("guid") or row.get("chatGuid") or ""
            chat_identifier = chat.get("chatIdentifier") or row.get("chatIdentifier") or sender
            if self._poll_allowed_users and not (
                self._user_matches_allowed(sender, self._poll_allowed_users)
                or self._user_matches_allowed(chat_identifier, self._poll_allowed_users)
            ):
                continue
            records.append(
                {
                    "guid": row.get("guid") or f"bb-{row.get('originalROWID')}",
                    "text": text,
                    "sender": sender,
                    "chatGuid": chat_guid or chat_identifier,
                    "chatIdentifier": chat_identifier,
                    "isGroup": bool(";+;" in (chat_guid or "")),
                    "source": "bluebubbles-api-poll",
                }
            )
        return newest, records

    # ------------------------------------------------------------------
    # Chat GUID resolution
    # ------------------------------------------------------------------

    async def _resolve_chat_guid(self, target: str) -> Optional[str]:
        """Resolve an email/phone to a BlueBubbles chat GUID.

        If *target* already contains a semicolon (raw GUID format like
        ``iMessage;-;user@example.com``), it is returned as-is.  Otherwise
        the adapter queries the BlueBubbles chat list and matches on
        ``chatIdentifier`` or participant address.
        """
        target = (target or "").strip()
        if not target:
            return None
        # Already a raw GUID
        if ";" in target:
            return target
        if target in self._guid_cache:
            return self._guid_cache[target]
        try:
            normalized_target = target.lower()
            phone_target = re.sub(r"\D", "", target)
            if phone_target and target.strip().startswith("+"):
                direct_guid = f"any;-;{target.strip()}"
                self._guid_cache[target] = direct_guid
                return direct_guid
            fallback_guid: Optional[str] = None
            for offset in range(0, 2000, 100):
                payload = await self._api_post(
                    "/api/v1/chat/query",
                    {"limit": 100, "offset": offset, "with": ["participants"]},
                )
                chats = payload.get("data", []) or []
                if not chats:
                    break
                for chat in chats:
                    guid = chat.get("guid") or chat.get("chatGuid")
                    identifier = (chat.get("chatIdentifier") or chat.get("identifier") or "").strip()
                    identifier_phone = re.sub(r"\D", "", identifier)
                    if identifier and (
                        identifier.lower() == normalized_target
                        or (phone_target and identifier_phone == phone_target)
                    ):
                        if guid:
                            self._guid_cache[target] = guid
                        return guid
                    for part in chat.get("participants", []) or []:
                        address = (part.get("address") or "").strip()
                        address_phone = re.sub(r"\D", "", address)
                        if address and (
                            address.lower() == normalized_target
                            or (phone_target and address_phone == phone_target)
                        ) and guid:
                            # Prefer a 1:1 chat identifier match. If the only
                            # match is membership in a group, keep looking in
                            # case a direct chat appears later in the page set.
                            fallback_guid = fallback_guid or guid
                if len(chats) < 100:
                    break
            if fallback_guid:
                self._guid_cache[target] = fallback_guid
                return fallback_guid
        except Exception:
            pass
        return None

    async def _create_chat_for_handle(
        self, address: str, message: str
    ) -> SendResult:
        """Create a new chat by sending the first message to *address*."""
        payload = {
            "addresses": [address],
            "message": message,
            "tempGuid": f"temp-{datetime.utcnow().timestamp()}",
        }
        try:
            res = await self._api_post("/api/v1/chat/new", payload)
            data = res.get("data") or {}
            msg_id = data.get("guid") or data.get("messageGuid") or "ok"
            return SendResult(success=True, message_id=str(msg_id), raw_response=res)
        except Exception as exc:
            return SendResult(success=False, error=str(exc))

    # ------------------------------------------------------------------
    # Text sending
    # ------------------------------------------------------------------

    @staticmethod
    def truncate_message(content: str, max_length: int = MAX_TEXT_LENGTH) -> List[str]:
        # Use the base splitter but skip pagination indicators — iMessage
        # bubbles flow naturally without "(1/3)" suffixes.
        chunks = BasePlatformAdapter.truncate_message(content, max_length)
        return [re.sub(r"\s*\(\d+/\d+\)$", "", c) for c in chunks]

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        text = self.format_message(content)
        if not text:
            return SendResult(success=False, error="BlueBubbles send requires text")
        # Split on paragraph breaks first (double newlines) so each thought
        # becomes its own iMessage bubble, then truncate any that are still
        # too long.
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
        chunks: List[str] = []
        for para in (paragraphs or [text]):
            if len(para) <= self.MAX_MESSAGE_LENGTH:
                chunks.append(para)
            else:
                chunks.extend(self.truncate_message(para, max_length=self.MAX_MESSAGE_LENGTH))
        last = SendResult(success=True)
        for chunk in chunks:
            guid = await self._resolve_chat_guid(chat_id)
            if not guid:
                # If the target looks like an address, try creating a new chat
                if self._private_api_enabled and (
                    "@" in chat_id or re.match(r"^\+\d+", chat_id)
                ):
                    return await self._create_chat_for_handle(chat_id, chunk)
                return SendResult(
                    success=False,
                    error=f"BlueBubbles chat not found for target: {chat_id}",
                )
            candidate_guids = [guid]
            # BlueBubbles can report chat GUIDs with a generic "any" service
            # prefix (for example, "any;-;+15551234567"). Its AppleScript
            # send path may then fail with "Can’t make any into type constant".
            # Retry explicit services for outbound sends while keeping the
            # original GUID as the first attempt.
            if guid.startswith("any;"):
                suffix = guid.split(";", 1)[1]
                candidate_guids.extend([f"iMessage;{suffix}", f"SMS;{suffix}"])

            last_error: Optional[str] = None
            for candidate_guid in candidate_guids:
                payload: Dict[str, Any] = {
                    "chatGuid": candidate_guid,
                    "tempGuid": f"temp-{datetime.utcnow().timestamp()}",
                    "message": chunk,
                }
                if reply_to and self._private_api_enabled and self._helper_connected:
                    payload["method"] = "private-api"
                    payload["selectedMessageGuid"] = reply_to
                    payload["partIndex"] = 0
                try:
                    res = await self._api_post("/api/v1/message/text", payload)
                    data = res.get("data") or {}
                    msg_id = data.get("guid") or data.get("messageGuid") or "ok"
                    last = SendResult(
                        success=True, message_id=str(msg_id), raw_response=res
                    )
                    last_error = None
                    break
                except Exception as exc:
                    last_error = str(exc)
                    # Only retry alternate service GUIDs for the generic
                    # BlueBubbles "any" service failure. Other failures should
                    # surface immediately so we do not accidentally duplicate
                    # messages after a timeout or unrelated server error.
                    if not guid.startswith("any;") or "500 Internal Server Error" not in last_error:
                        return SendResult(success=False, error=last_error)
                    continue
            if last_error:
                return SendResult(success=False, error=last_error)
        return last

    # ------------------------------------------------------------------
    # Media sending (outbound)
    # ------------------------------------------------------------------

    async def _send_attachment(
        self,
        chat_id: str,
        file_path: str,
        filename: Optional[str] = None,
        caption: Optional[str] = None,
        is_audio_message: bool = False,
    ) -> SendResult:
        """Send a file attachment via BlueBubbles multipart upload."""
        if not self.client:
            return SendResult(success=False, error="Not connected")
        if not os.path.isfile(file_path):
            return SendResult(success=False, error=f"File not found: {file_path}")

        guid = await self._resolve_chat_guid(chat_id)
        if not guid:
            return SendResult(success=False, error=f"Chat not found: {chat_id}")

        fname = filename or os.path.basename(file_path)
        try:
            with open(file_path, "rb") as f:
                files = {"attachment": (fname, f, "application/octet-stream")}
                data: Dict[str, str] = {
                    "chatGuid": guid,
                    "name": fname,
                    "tempGuid": uuid.uuid4().hex,
                }
                if is_audio_message:
                    data["isAudioMessage"] = "true"
                res = await self.client.post(
                    self._api_url("/api/v1/message/attachment"),
                    files=files,
                    data=data,
                    timeout=120,
                )
                res.raise_for_status()
                result = res.json()

            if caption:
                await self.send(chat_id, caption)

            if result.get("status") == 200:
                rdata = result.get("data") or {}
                msg_id = rdata.get("guid") if isinstance(rdata, dict) else None
                return SendResult(
                    success=True, message_id=msg_id, raw_response=result
                )
            return SendResult(
                success=False,
                error=result.get("message", "Attachment upload failed"),
            )
        except Exception as e:
            return SendResult(success=False, error=str(e))

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        try:
            from gateway.platforms.base import cache_image_from_url

            local_path = await cache_image_from_url(image_url)
            return await self._send_attachment(chat_id, local_path, caption=caption)
        except Exception:
            return await super().send_image(chat_id, image_url, caption, reply_to)

    async def send_image_file(
        self,
        chat_id: str,
        image_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        return await self._send_attachment(chat_id, image_path, caption=caption)

    async def send_voice(
        self,
        chat_id: str,
        audio_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        return await self._send_attachment(
            chat_id, audio_path, caption=caption, is_audio_message=True
        )

    async def send_video(
        self,
        chat_id: str,
        video_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        return await self._send_attachment(chat_id, video_path, caption=caption)

    async def send_document(
        self,
        chat_id: str,
        file_path: str,
        caption: Optional[str] = None,
        file_name: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        return await self._send_attachment(
            chat_id, file_path, filename=file_name, caption=caption
        )

    async def send_animation(
        self,
        chat_id: str,
        animation_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        return await self.send_image(
            chat_id, animation_url, caption, reply_to, metadata
        )

    # ------------------------------------------------------------------
    # Typing indicators
    # ------------------------------------------------------------------

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        if not self._private_api_enabled or not self._helper_connected or not self.client:
            return
        try:
            guid = await self._resolve_chat_guid(chat_id)
            if guid:
                encoded = quote(guid, safe="")
                await self.client.post(
                    self._api_url(f"/api/v1/chat/{encoded}/typing"), timeout=5
                )
        except Exception:
            pass

    async def stop_typing(self, chat_id: str) -> None:
        if not self._private_api_enabled or not self._helper_connected or not self.client:
            return
        try:
            guid = await self._resolve_chat_guid(chat_id)
            if guid:
                encoded = quote(guid, safe="")
                await self.client.delete(
                    self._api_url(f"/api/v1/chat/{encoded}/typing"), timeout=5
                )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Read receipts
    # ------------------------------------------------------------------

    async def mark_read(self, chat_id: str) -> bool:
        if not self._private_api_enabled or not self._helper_connected or not self.client:
            return False
        try:
            guid = await self._resolve_chat_guid(chat_id)
            if guid:
                encoded = quote(guid, safe="")
                await self.client.post(
                    self._api_url(f"/api/v1/chat/{encoded}/read"), timeout=5
                )
                return True
        except Exception:
            pass
        return False

    # ------------------------------------------------------------------
    # Tapback reactions
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Chat info
    # ------------------------------------------------------------------

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        is_group = ";+;" in (chat_id or "")
        info: Dict[str, Any] = {
            "name": chat_id,
            "type": "group" if is_group else "dm",
        }
        try:
            guid = await self._resolve_chat_guid(chat_id)
            if guid:
                encoded = quote(guid, safe="")
                res = await self._api_get(
                    f"/api/v1/chat/{encoded}?with=participants"
                )
                data = (res or {}).get("data", {})
                display_name = (
                    data.get("displayName")
                    or data.get("chatIdentifier")
                    or chat_id
                )
                participants = []
                for p in data.get("participants", []) or []:
                    addr = (p.get("address") or "").strip()
                    if addr:
                        participants.append(addr)
                info["name"] = display_name
                if participants:
                    info["participants"] = participants
        except Exception:
            pass
        return info

    def format_message(self, content: str) -> str:
        return strip_markdown(content)

    # ------------------------------------------------------------------
    # Inbound attachment downloading (from #4588)
    # ------------------------------------------------------------------

    async def _download_attachment(
        self, att_guid: str, att_meta: Dict[str, Any]
    ) -> Optional[str]:
        """Download an attachment from BlueBubbles and cache it locally.

        Returns the local file path on success, None on failure.
        """
        if not self.client:
            return None
        try:
            encoded = quote(att_guid, safe="")
            resp = await self.client.get(
                self._api_url(f"/api/v1/attachment/{encoded}/download"),
                timeout=60,
                follow_redirects=True,
            )
            resp.raise_for_status()
            data = resp.content

            mime = (att_meta.get("mimeType") or "").lower()
            transfer_name = att_meta.get("transferName", "")

            if mime.startswith("image/"):
                ext_map = {
                    "image/jpeg": ".jpg",
                    "image/png": ".png",
                    "image/gif": ".gif",
                    "image/webp": ".webp",
                    "image/heic": ".jpg",
                    "image/heif": ".jpg",
                    "image/tiff": ".jpg",
                }
                ext = ext_map.get(mime, ".jpg")
                return cache_image_from_bytes(data, ext)

            if mime.startswith("audio/"):
                ext_map = {
                    "audio/mp3": ".mp3",
                    "audio/mpeg": ".mp3",
                    "audio/ogg": ".ogg",
                    "audio/wav": ".wav",
                    "audio/x-caf": ".mp3",
                    "audio/mp4": ".m4a",
                    "audio/aac": ".m4a",
                }
                ext = ext_map.get(mime, ".mp3")
                return cache_audio_from_bytes(data, ext)

            # Videos, documents, and everything else
            filename = transfer_name or f"file_{uuid.uuid4().hex[:8]}"
            return cache_document_from_bytes(data, filename)

        except Exception as exc:
            logger.warning(
                "[bluebubbles] failed to download attachment %s: %s",
                _redact(att_guid),
                exc,
            )
            return None

    # ------------------------------------------------------------------
    # Webhook handling
    # ------------------------------------------------------------------

    def _extract_payload_record(
        self, payload: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        data = payload.get("data")
        if isinstance(data, dict):
            return data
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    return item
        if isinstance(payload.get("message"), dict):
            return payload.get("message")
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _value(*candidates: Any) -> Optional[str]:
        for candidate in candidates:
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        return None

    async def _handle_inbound_record(self, record: Dict[str, Any], payload: Dict[str, Any]):
        from aiohttp import web

        # Skip tapback reactions delivered as messages
        assoc_type = record.get("associatedMessageType")
        if isinstance(assoc_type, int) and assoc_type in {
            **_TAPBACK_ADDED,
            **_TAPBACK_REMOVED,
        }:
            return web.Response(text="ok")

        text = (
            self._value(
                record.get("text"), record.get("message"), record.get("body")
            )
            or ""
        )

        # --- Inbound attachment handling ---
        attachments = record.get("attachments") or []
        media_urls: List[str] = []
        media_types: List[str] = []
        msg_type = MessageType.TEXT

        for att in attachments:
            att_guid = att.get("guid", "")
            if not att_guid:
                continue
            cached = await self._download_attachment(att_guid, att)
            if cached:
                mime = (att.get("mimeType") or "").lower()
                media_urls.append(cached)
                media_types.append(mime)
                if mime.startswith("image/"):
                    msg_type = MessageType.PHOTO
                elif mime.startswith("audio/") or (att.get("uti") or "").endswith(
                    "caf"
                ):
                    msg_type = MessageType.VOICE
                elif mime.startswith("video/"):
                    msg_type = MessageType.VIDEO
                else:
                    msg_type = MessageType.DOCUMENT

        # With multiple attachments, prefer PHOTO if any images present
        if len(media_urls) > 1:
            mime_prefixes = {(m or "").split("/")[0] for m in media_types}
            if "image" in mime_prefixes:
                msg_type = MessageType.PHOTO

        if not text and media_urls:
            text = "(attachment)"
        # --- End attachment handling ---

        chat_guid = self._value(
            record.get("chatGuid"),
            payload.get("chatGuid"),
            record.get("chat_guid"),
            payload.get("chat_guid"),
            payload.get("guid"),
        )
        # Fallback: BlueBubbles v1.9+ webhook payloads omit top-level chatGuid;
        # the chat GUID is nested under data.chats[0].guid instead.
        if not chat_guid:
            _chats = record.get("chats") or []
            if _chats and isinstance(_chats[0], dict):
                chat_guid = _chats[0].get("guid") or _chats[0].get("chatGuid")
        chat_identifier = self._value(
            record.get("chatIdentifier"),
            record.get("identifier"),
            payload.get("chatIdentifier"),
            payload.get("identifier"),
        )
        sender = (
            self._value(
                record.get("handle", {}).get("address")
                if isinstance(record.get("handle"), dict)
                else None,
                record.get("sender"),
                record.get("from"),
                record.get("address"),
            )
            or chat_identifier
            or chat_guid
        )
        if not (chat_guid or chat_identifier) and sender:
            chat_identifier = sender
        if not sender or not (chat_guid or chat_identifier) or not text:
            return web.json_response({"error": "missing message fields"}, status=400)

        message_id = self._value(
            record.get("guid"),
            record.get("messageGuid"),
            record.get("id"),
        )
        if message_id:
            if message_id in self._seen_inbound_message_ids:
                return web.Response(text="ok")
            self._seen_inbound_message_ids.add(message_id)

        session_chat_id = chat_guid or chat_identifier
        is_group = bool(record.get("isGroup")) or (";+;" in (chat_guid or ""))
        source = self.build_source(
            chat_id=session_chat_id,
            chat_name=chat_identifier or sender,
            chat_type="group" if is_group else "dm",
            user_id=sender,
            user_name=sender,
            chat_id_alt=chat_identifier,
        )
        event = MessageEvent(
            text=text,
            message_type=msg_type,
            source=source,
            raw_message=payload,
            message_id=message_id,
            reply_to_message_id=self._value(
                record.get("threadOriginatorGuid"),
                record.get("associatedMessageGuid"),
            ),
            media_urls=media_urls,
            media_types=media_types,
        )
        task = asyncio.create_task(self.handle_message(event))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

        # Fire-and-forget read receipt
        if self.send_read_receipts and session_chat_id:
            asyncio.create_task(self.mark_read(session_chat_id))

        return web.Response(text="ok")

    async def _handle_webhook(self, request):
        from aiohttp import web

        token = (
            request.query.get("password")
            or request.query.get("guid")
            or request.headers.get("x-password")
            or request.headers.get("x-guid")
            or request.headers.get("x-bluebubbles-guid")
        )
        if token != self.password:
            return web.json_response({"error": "unauthorized"}, status=401)
        try:
            raw = await request.read()
            body = raw.decode("utf-8", errors="replace")
            try:
                payload = json.loads(body)
            except Exception:
                from urllib.parse import parse_qs

                form = parse_qs(body)
                payload_str = (
                    form.get("payload")
                    or form.get("data")
                    or form.get("message")
                    or [""]
                )[0]
                payload = json.loads(payload_str) if payload_str else {}
        except Exception as exc:
            logger.error("[bluebubbles] webhook parse error: %s", exc)
            return web.json_response({"error": "invalid payload"}, status=400)

        event_type = self._value(payload.get("type"), payload.get("event")) or ""
        # Only process message events; silently acknowledge everything else
        if event_type and event_type not in _MESSAGE_EVENTS:
            return web.Response(text="ok")

        record = self._extract_payload_record(payload) or {}
        is_from_me = bool(
            record.get("isFromMe")
            or record.get("fromMe")
            or record.get("is_from_me")
        )
        if is_from_me:
            return web.Response(text="ok")

        return await self._handle_inbound_record(record, payload)

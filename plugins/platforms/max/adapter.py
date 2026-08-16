"""MAX platform adapter (Hermes plugin).

MAX (max.ru) — Russian messenger. This adapter connects via Long Polling
(GET /updates with marker cursor) and sends replies via POST /messages.

Why Long Polling instead of webhooks:
- MAX requires HTTPS + Russian Trusted CA certs for webhooks (since 2025-05-25)
- Users behind NAT (typical RU ISP, e.g. Rostelecom) have no public endpoint
- Long Polling works from anywhere: the adapter polls platform-api2.max.ru

TLS: MAX uses Russian Trusted Root CA. Set MAX_CA_CERT_PATH to the PEM file,
or the adapter falls back to the default trust store. The cert is
auto-downloaded from the official source (gu-st.ru) on first use, with a
bounded timeout and PEM structure validation.

Security note: trusting the Russian Trusted Root CA is inherent to using the
MAX platform (its API is served with certificates chained to that CA). The
cert is fetched over HTTPS from the official Ministry source (gu-st.ru) and
validated as a PEM certificate before being used.
"""

import asyncio
import json
import logging
import os
import random
import ssl
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

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
)

from agent.secret_scope import UnscopedSecretError as _UnscopedSecretError
from agent.secret_scope import get_secret as _scoped_get_secret

logger = logging.getLogger(__name__)

API_HOST = "platform-api2.max.ru"
API_SCHEME = "https"
DEFAULT_POLL_TIMEOUT = 90
DEFAULT_POLL_LIMIT = 100
POLL_INTERVAL_SECONDS = 1.0  # pause between long-poll requests
MAX_MESSAGE_LENGTH = 4000  # MAX text limit
RECONNECT_BACKOFF = [1, 2, 5, 10, 30]
DEDUP_WINDOW_SECONDS = 300
DEDUP_MAX_SIZE = 2000
_ECHO_MARKER = "hermes-agent-max"  # appended to outgoing text for echo-loop prevention
CA_DOWNLOAD_TIMEOUT = 10  # seconds
MAX_SEND_RATE_PER_CHAT = 2.0  # MAX: max 2 messages/sec per chat
_TRUNCATION_NOTICE = "\n\n✂️ (сообщение обрезано — лимит MAX 4000 симв.)"


def _get_scoped_secret(name, default=None):
    """Scope-aware credential read (same pattern as ntfy/slack adapters)."""
    try:
        val = _scoped_get_secret(name, default)
    except _UnscopedSecretError:
        val = os.getenv(name)
    return val if val is not None else default


def _is_valid_pem_cert(path: str) -> bool:
    """Validate that a file is a PEM certificate (not HTML/truncated junk)."""
    try:
        with open(path, "rb") as f:
            data = f.read()
        if b"-----BEGIN CERTIFICATE-----" not in data:
            return False
        # Try loading it as an X.509 certificate
        ssl.PEM_cert_to_DER_cert(data.decode("utf-8", errors="replace"))
        return True
    except Exception:
        return False


def _default_ca_path() -> Optional[str]:
    """Return the Russian Trusted Root CA path if present (honors HERMES_HOME).

    Auto-downloads the official Russian Trusted Root CA from gu-st.ru on
    first use if no local copy exists, so a fresh install works out of the
    box without manual certificate setup. Download uses a bounded timeout
    and the file is validated as a PEM certificate.
    """
    hermes_home = os.getenv("HERMES_HOME", "") or os.path.expanduser("~/.hermes")
    candidates = [
        os.getenv("MAX_CA_CERT_PATH", "").strip(),
        os.path.join(hermes_home, "max", "certs", "russian_trusted_root_ca_pem.crt"),
        os.path.join(os.path.dirname(__file__), "certs", "russian_trusted_root_ca_pem.crt"),
        os.path.expanduser("~/.hermes/max/certs/russian_trusted_root_ca_pem.crt"),
    ]
    for c in candidates:
        if c and os.path.isfile(c) and _is_valid_pem_cert(c):
            return c

    # Auto-download the official cert (gu-st.ru is the Ministry's official source)
    try:
        import urllib.request

        dest_dir = os.path.join(hermes_home, "max", "certs")
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, "russian_trusted_root_ca_pem.crt")
        url = "https://gu-st.ru/content/lending/russian_trusted_root_ca_pem.crt"
        with urllib.request.urlopen(url, timeout=CA_DOWNLOAD_TIMEOUT) as resp:
            data = resp.read()
        with open(dest, "wb") as f:
            f.write(data)
        if _is_valid_pem_cert(dest):
            logger.info("[max] Auto-downloaded Russian Trusted Root CA to %s", dest)
            return dest
        logger.warning("[max] Downloaded file is not a valid PEM certificate; removing")
        try:
            os.remove(dest)
        except OSError:
            pass
    except Exception as e:
        logger.warning("[max] Auto-download of Russian Trusted Root CA failed: %s", e)
    return None


def check_requirements() -> bool:
    """Check whether the MAX adapter is installable and minimally configured."""
    if not HTTPX_AVAILABLE:
        return False
    token = os.getenv("MAX_BOT_TOKEN", "").strip()
    return bool(token)


def validate_config(config) -> bool:
    """Validate that the configured MAX platform has a token set."""
    extra = getattr(config, "extra", {}) or {}
    token = extra.get("token") or os.getenv("MAX_BOT_TOKEN", "")
    return bool(token)


def is_connected(config) -> bool:
    """Check whether MAX is configured (env or config.yaml)."""
    extra = getattr(config, "extra", {}) or {}
    token = os.getenv("MAX_BOT_TOKEN") or extra.get("token", "")
    return bool(token)


class MaxAdapter(BasePlatformAdapter):
    """MAX adapter — Long Polling inbound, POST /messages outbound."""

    MAX_MESSAGE_LENGTH = MAX_MESSAGE_LENGTH

    def __init__(self, config: PlatformConfig):
        platform = Platform("max")
        super().__init__(config=config, platform=platform)

        extra = config.extra or {}
        self._token: str = extra.get("token") or _get_scoped_secret("MAX_BOT_TOKEN", "")
        self._marker: Optional[int] = None
        self._poll_task: Optional[asyncio.Task] = None
        self._http_client: Optional["httpx.AsyncClient"] = None
        self._seen_messages: Dict[str, float] = {}
        self._ca_path = _default_ca_path()
        self._last_user_id: str = ""
        # Health/status tracking
        self._last_poll_at: Optional[float] = None
        self._last_poll_error: Optional[str] = None
        self._last_error_at: Optional[float] = None
        # Marker persistence
        self._marker_path = os.path.join(
            os.getenv("HERMES_HOME", "") or os.path.expanduser("~/.hermes"),
            "max", "marker.json",
        )
        self._load_marker()
        # Send rate limiting: chat_id -> [timestamps]
        self._send_history: Dict[str, List[float]] = {}

    # -- Marker persistence ------------------------------------------------

    def _load_marker(self) -> None:
        """Load the last-known marker from disk (if any)."""
        try:
            if os.path.isfile(self._marker_path):
                with open(self._marker_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._marker = int(data.get("marker") or 0) or None
        except Exception as e:
            logger.debug("[%s] Could not load marker: %s", self.name, e)

    def _save_marker(self) -> None:
        """Persist the marker to disk so restarts don't replay old updates."""
        if not self._marker:
            return
        try:
            os.makedirs(os.path.dirname(self._marker_path), exist_ok=True)
            with open(self._marker_path, "w", encoding="utf-8") as f:
                json.dump({"marker": self._marker, "saved_at": time.time()}, f)
        except Exception as e:
            logger.debug("[%s] Could not save marker: %s", self.name, e)

    # -- Connection lifecycle -----------------------------------------------

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        """Start the Long Polling loop."""
        if not HTTPX_AVAILABLE:
            logger.warning("[%s] httpx not installed", self.name)
            return False
        if not self._token:
            logger.warning("[%s] MAX_BOT_TOKEN not configured", self.name)
            return False

        try:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(connect=15.0, read=DEFAULT_POLL_TIMEOUT + 15, write=15.0, pool=15.0),
                verify=self._ca_path or True,
            )
            # Validate token and fetch bot info (GET /me)
            await self._fetch_bot_info()
            # Register bot command menu (PATCH /me/commands) — best effort
            await self._register_commands()
            self._poll_task = asyncio.create_task(self._run_poll_loop())
            self._mark_connected()
            logger.info("[%s] Connected — Long Polling %s://%s/updates (marker=%s)",
                        self.name, API_SCHEME, API_HOST, self._marker)
            return True
        except Exception as e:
            logger.error("[%s] Failed to connect: %s", self.name, e)
            return False

    async def _fetch_bot_info(self) -> None:
        """GET /me — validate the token and log bot identity."""
        if self._http_client is None:
            return
        try:
            resp = await self._http_client.get(
                f"{API_SCHEME}://{API_HOST}/me",
                headers={"Authorization": self._token},
                timeout=15.0,
            )
            if resp.status_code == 401:
                logger.error("[%s] Auth failed (401) — MAX_BOT_TOKEN invalid", self.name)
                self._set_fatal_error(
                    "max_unauthorized",
                    "MAX API rejected auth (401). Check MAX_BOT_TOKEN.",
                    retryable=False,
                )
                return
            if resp.status_code < 300:
                data = resp.json()
                bot_name = data.get("first_name") or data.get("username") or "?"
                bot_id = data.get("user_id")
                logger.info("[%s] Authenticated as %s (id=%s)", self.name, bot_name, bot_id)
        except Exception as e:
            logger.warning("[%s] /me check failed: %s", self.name, e)

    async def _register_commands(self) -> None:
        """PATCH /me/commands — set the bot command menu (best effort)."""
        if self._http_client is None:
            return
        commands = [
            {"name": "help", "description": "Помощь и команды"},
            {"name": "new", "description": "Новая сессия"},
            {"name": "sethome", "description": "Установить этот чат домашним"},
            {"name": "reset", "description": "Сбросить сессию"},
        ]
        try:
            body = json.dumps({"commands": commands}).encode("utf-8")
            resp = await self._http_client.patch(
                f"{API_SCHEME}://{API_HOST}/me/commands",
                content=body,
                headers={"Authorization": self._token, "Content-Type": "application/json"},
                timeout=15.0,
            )
            if resp.status_code < 300:
                logger.info("[%s] Bot command menu registered (%d commands)", self.name, len(commands))
            else:
                logger.debug("[%s] /me/commands HTTP %d: %s", self.name, resp.status_code, resp.text[:150])
        except Exception as e:
            logger.debug("[%s] /me/commands failed: %s", self.name, e)

    async def _run_poll_loop(self) -> None:
        """Long Poll GET /updates with marker cursor and reconnect backoff."""
        backoff_idx = 0

        while self._running:
            try:
                await self._poll_once()
                backoff_idx = 0
            except asyncio.CancelledError:
                return
            except Exception as e:
                if not self._running:
                    return
                self._last_poll_error = str(e)
                self._last_error_at = time.time()
                logger.warning("[%s] Poll error: %s", self.name, e)
                delay = RECONNECT_BACKOFF[min(backoff_idx, len(RECONNECT_BACKOFF) - 1)]
                # Jitter avoids thundering-herd when many clients reconnect at once
                delay += random.uniform(0, 1.0)
                await asyncio.sleep(delay)
                backoff_idx += 1
                continue

            # Small pause between polls; long-poll already holds the connection
            await asyncio.sleep(POLL_INTERVAL_SECONDS)

    async def _poll_once(self) -> None:
        """One GET /updates request."""
        if self._http_client is None:
            return
        params: Dict[str, Any] = {
            "timeout": DEFAULT_POLL_TIMEOUT,
            "limit": DEFAULT_POLL_LIMIT,
        }
        if self._marker:
            params["marker"] = self._marker

        try:
            resp = await self._http_client.get(
                f"{API_SCHEME}://{API_HOST}/updates",
                params=params,
                headers={"Authorization": self._token},
            )
        except Exception as e:
            self._last_poll_error = str(e)
            self._last_error_at = time.time()
            raise
        self._last_poll_at = time.time()
        self._last_poll_error = None

        if resp.status_code == 401:
            logger.error("[%s] Auth failed (401) — token invalid. Stopping.", self.name)
            self._set_fatal_error(
                "max_unauthorized",
                "MAX API rejected auth (401). Check MAX_BOT_TOKEN.",
                retryable=False,
            )
            self._running = False
            return
        if resp.status_code >= 400:
            logger.warning("[%s] Poll HTTP %d: %s", self.name, resp.status_code, resp.text[:200])
            return

        try:
            data = resp.json()
        except Exception:
            logger.warning("[%s] Bad JSON from /updates", self.name)
            return

        updates = data.get("updates") or []
        if data.get("marker") is not None:
            self._marker = int(data["marker"])
            self._save_marker()
        for upd in updates:
            await self._handle_update(upd)

    # -- Inbound message processing -----------------------------------------

    async def _handle_update(self, upd: Dict[str, Any]) -> None:
        """Process a single Update object from MAX."""
        update_type = upd.get("update_type") or upd.get("event") or "unknown"
        if update_type != "message_created":
            logger.debug("[%s] Ignoring update type %s", self.name, update_type)
            return

        message = upd.get("message") or upd
        sender = message.get("sender") or upd.get("user") or {}
        if sender.get("is_bot") or sender.get("isBot"):
            logger.debug("[%s] Skipping own/bot message", self.name)
            return

        body_obj = message.get("body") or {}
        text = None
        if isinstance(body_obj, dict):
            text = body_obj.get("text") or body_obj.get("body")
        if not text:
            text = upd.get("body")
        text = (text or "").strip()
        if not text:
            return

        # Echo-loop prevention
        if _ECHO_MARKER in text:
            return

        recipient = message.get("recipient") or {}
        chat_id = str(
            upd.get("chat_id")
            or recipient.get("chat_id")
            or sender.get("user_id")
            or ""
        )
        user_id = str(sender.get("user_id") or "")
        user_name = sender.get("name") or user_id or "?"
        chat_type = recipient.get("chat_type") or "dialog"
        if chat_type == "dialog":
            chat_type = "dm"

        # Real message ID from MAX body.mid, fallback to timestamp
        mid = ""
        if isinstance(body_obj, dict):
            mid = str(body_obj.get("mid") or "")
        msg_id = mid or str(upd.get("timestamp") or uuid.uuid4().hex)
        if self._is_duplicate(msg_id):
            return

        timestamp = datetime.now(tz=timezone.utc)
        try:
            ts = upd.get("timestamp") or message.get("timestamp")
            if ts:
                timestamp = datetime.fromtimestamp(int(ts) / 1000, tz=timezone.utc)
        except (ValueError, OSError, TypeError):
            pass

        source = self.build_source(
            chat_id=chat_id,
            chat_name=user_name,
            chat_type=chat_type,
            user_id=user_id,
            user_name=user_name,
        )
        # Store user_id on metadata so send() can reply to the right recipient
        self._last_user_id = user_id

        message_event = MessageEvent(
            text=text,
            message_type=MessageType.TEXT,
            source=source,
            message_id=msg_id,
            raw_message=upd,
            timestamp=timestamp,
        )

        logger.info("[%s] Message from %s (chat %s): %s", self.name, user_name, chat_id, text[:80])
        await self.handle_message(message_event)

    def _is_duplicate(self, msg_id: str) -> bool:
        now = time.time()
        # Вычищаем записи старше окна дедупликации при каждом вызове —
        # иначе старые ID навсегда останутся в памяти и будут ложно
        # считаться дубликатами (особенно после перезапуска или долгой паузы).
        if self._seen_messages:
            cutoff = now - DEDUP_WINDOW_SECONDS
            self._seen_messages = {k: v for k, v in self._seen_messages.items() if v > cutoff}
        if msg_id in self._seen_messages:
            return True
        self._seen_messages[msg_id] = now
        return False

    # -- Outbound messaging -------------------------------------------------

    async def _rate_limit_send(self, chat_id: str) -> None:
        """Enforce MAX rate limit: max 2 messages/sec per chat."""
        now = time.time()
        history = self._send_history.setdefault(chat_id, [])
        # Keep only the last second
        history[:] = [t for t in history if now - t < 1.0]
        if len(history) >= 2:
            sleep_for = 1.0 - (now - history[0])
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
        self._send_history[chat_id].append(time.time())

    @staticmethod
    def _split_text(
        text: str, limit: int = MAX_MESSAGE_LENGTH
    ) -> List[str]:
        """Split long text into ≤limit chunks, preferring line/word breaks.

        MAX hard-caps a single message at MAX_MESSAGE_LENGTH chars; instead of
        silently truncating (old behaviour), split into several messages. The
        caller spaces the sends to respect MAX's ~2 msg/sec dialog limit.
        """
        if len(text) <= limit:
            return [text]
        chunks: List[str] = []
        remaining = text
        while len(remaining) > limit:
            cut = remaining.rfind("\n", 0, limit)
            if cut <= 0:
                cut = remaining.rfind(" ", 0, limit)
            if cut <= 0:
                cut = limit
            chunks.append(remaining[:cut])
            remaining = remaining[cut:].lstrip("\n")
        if remaining:
            chunks.append(remaining)
        return chunks

    def _smart_truncate(self, content: str) -> str:
        """Truncate to MAX limit, cutting at a markdown-friendly boundary.

        Legacy fallback: used only when a single chunk still exceeds the
        limit (e.g. an unbreakable 5000-char word). Normal long messages are
        split by ``_split_text`` instead.
        """
        if len(content) <= MAX_MESSAGE_LENGTH:
            return content
        limit = MAX_MESSAGE_LENGTH - len(_TRUNCATION_NOTICE)
        cut = content[:limit]
        # Cut at last newline or space if possible
        last_nl = cut.rfind("\n")
        last_sp = cut.rfind(" ")
        boundary = max(last_nl, last_sp)
        if boundary > limit * 0.5:  # only if it's a reasonable cut point
            cut = cut[:boundary]
        return cut.rstrip() + _TRUNCATION_NOTICE

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a message to a MAX user (user_id) or chat (chat_id).

        Long content (>4000 chars) is split into several sequential messages,
        respecting MAX's ~2 msg/sec limit via ``_rate_limit_send``.
        """
        if self._http_client is None:
            return SendResult(success=False, error="HTTP client not initialized")

        metadata = metadata or {}
        # Reply target: for dialogs (dm) use user_id; for chats/channels use chat_id
        user_id = metadata.get("user_id") or self._last_user_id
        chat_type = metadata.get("chat_type") or "dm"
        params: Dict[str, Any] = {}
        if chat_type == "dm" and user_id:
            params["user_id"] = user_id
        else:
            params["chat_id"] = chat_id

        chunks = self._split_text(content)
        last: SendResult = SendResult(success=False, error="no chunks")
        for chunk in chunks:
            # MAX supports markdown formatting for bot messages
            payload = {"text": chunk, "attachments": [], "format": "markdown"}
            body = json.dumps(payload).encode("utf-8")
            try:
                await self._rate_limit_send(str(chat_id))
                resp = await self._http_client.post(
                    f"{API_SCHEME}://{API_HOST}/messages",
                    params=params,
                    content=body,
                    headers={
                        "Authorization": self._token,
                        "Content-Type": "application/json",
                    },
                    timeout=15.0,
                )
                if resp.status_code < 300:
                    last = SendResult(success=True, message_id=uuid.uuid4().hex[:12])
                else:
                    logger.warning("[%s] Send failed HTTP %d: %s", self.name, resp.status_code, resp.text[:200])
                    last = SendResult(success=False, error=f"HTTP {resp.status_code}: {resp.text[:200]}")
                    break
            except Exception as e:
                logger.error("[%s] Send error: %s", self.name, e)
                last = SendResult(success=False, error=str(e))
                break
        return last

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """Send a typing indicator via POST /chats/{chatId}/actions.

        MAX supports ``typing_on`` (and other sending_* actions). The
        indicator lives ~5-6s, and the gateway's ``_keep_typing`` loop calls
        this every ~2s, so the bubble stays visible while the agent works.

        Note: MAX expects the numeric chat id. For dialogs we forward the
        ``chat_id`` we received from updates (recipient.chat_id — a numeric
        dialog id); if metadata carries a ``user_id`` we still use chat_id
        because the actions endpoint is keyed by chat, not user.
        """
        if self._http_client is None or not chat_id:
            return
        try:
            resp = await self._http_client.post(
                f"{API_SCHEME}://{API_HOST}/chats/{chat_id}/actions",
                content=json.dumps({"action": "typing_on"}).encode("utf-8"),
                headers={
                    "Authorization": self._token,
                    "Content-Type": "application/json",
                },
                timeout=5.0,
            )
            if resp.status_code >= 400:
                logger.debug(
                    "[%s] send_typing HTTP %d: %s",
                    self.name,
                    resp.status_code,
                    resp.text[:150],
                )
        except Exception as e:
            logger.debug("[%s] send_typing error: %s", self.name, e)

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Return basic info about a MAX chat."""
        return {"name": chat_id, "type": "dm"}

    async def disconnect(self) -> None:
        """Stop polling (gracefully) and close the HTTP client."""
        self._running = False
        self._mark_disconnected()
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
            self._poll_task = None
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
        self._seen_messages.clear()
        self._save_marker()
        logger.info("[%s] Disconnected", self.name)


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------


def _env_enablement() -> dict | None:
    """Seed PlatformConfig.extra from env vars during gateway config load."""
    token = os.getenv("MAX_BOT_TOKEN", "").strip()
    if not token:
        return None
    seed: dict = {"token": token}
    home = os.getenv("MAX_HOME_CHANNEL", "").strip()
    if home:
        seed["home_channel"] = {"chat_id": home, "name": os.getenv("MAX_HOME_CHANNEL_NAME", home)}
    return seed


async def _standalone_send(
    pconfig,
    chat_id: str,
    message: str,
    *,
    thread_id: Optional[str] = None,
    media_files: Optional[List[str]] = None,
    force_document: bool = False,
) -> Dict[str, Any]:
    """Out-of-process send for cron / send_message_tool fallbacks."""
    if not HTTPX_AVAILABLE:
        return {"error": "max standalone send: httpx not installed"}
    extra = getattr(pconfig, "extra", {}) or {}
    token = extra.get("token") or _get_scoped_secret("MAX_BOT_TOKEN", "")
    if not token:
        return {"error": "max standalone send: MAX_BOT_TOKEN not configured"}
    if media_files:
        logger.warning("[max] standalone send: media_files are not supported yet, sending text only")
    ca_path = _default_ca_path()
    text = (message or "")[:MAX_MESSAGE_LENGTH]
    body = json.dumps({"text": text, "attachments": []}).encode("utf-8")
    params: Dict[str, Any] = {}
    extra2 = getattr(pconfig, "extra", {}) or {}
    user_id = extra2.get("user_id") or os.getenv("MAX_HOME_USER_ID", "").strip()
    if user_id:
        params["user_id"] = user_id
    elif chat_id:
        params["chat_id"] = chat_id
    try:
        async with httpx.AsyncClient(verify=ca_path or True, timeout=15.0) as client:
            resp = await client.post(
                f"{API_SCHEME}://{API_HOST}/messages",
                params=params,
                content=body,
                headers={"Authorization": token, "Content-Type": "application/json"},
            )
        if resp.status_code >= 300:
            return {"error": f"max HTTP {resp.status_code}: {resp.text[:200]}"}
        return {"success": True, "platform": "max", "chat_id": chat_id, "message_id": uuid.uuid4().hex[:12]}
    except Exception as e:
        return {"error": f"max standalone send failed: {e}"}


def register(ctx) -> None:
    """Plugin entry point — called by the Hermes plugin system at startup."""
    ctx.register_platform(
        name="max",
        label="MAX",
        adapter_factory=lambda cfg: MaxAdapter(cfg),
        check_fn=check_requirements,
        validate_config=validate_config,
        is_connected=is_connected,
        required_env=["MAX_BOT_TOKEN"],
        install_hint="Run `hermes setup` to configure MAX. Token from business.max.ru → Чат-боты → Расширенные настройки.",
        env_enablement_fn=_env_enablement,
        cron_deliver_env_var="MAX_HOME_CHANNEL",
        standalone_sender_fn=_standalone_send,
        allowed_users_env="MAX_ALLOWED_USERS",
        allow_all_env="MAX_ALLOW_ALL_USERS",
        max_message_length=MAX_MESSAGE_LENGTH,
        emoji="🟠",
        pii_safe=True,
        allow_update_command=True,
        platform_hint=(
            "You are communicating via MAX messenger (Russia). "
            "Use plain text by default. Keep responses concise; "
            "MAX has a 4000-character per-message limit."
        ),
    )
"""MAX messenger platform adapter: GET /updates long polling in, POST /messages out.

MAX (max.ru, RU messenger) Bot API. Minimal functionality: 1:1 text dialogs only
(no groups/channels/media/buttons/threads — those can be layered on later).

config.yaml ``platforms.max.extra``: ``token`` (required), ``server`` (default
https://platform-api2.max.ru). Env (read at construct time; ``extra`` wins over
env): MAX_BOT_TOKEN, MAX_SERVER_URL, MAX_ALLOWED_USERS, MAX_ALLOW_ALL_USERS,
MAX_HOME_CHANNEL, MAX_HOME_CHANNEL_NAME.

Transport: MAX's own docs flag long polling (``GET /updates``) as a dev/test-grade
transport — rate- and retention-limited, not for production. It's used here
anyway because it needs no public HTTPS endpoint, keeping this adapter
self-contained; a webhook-based adapter is the natural next step if production
delivery volume becomes an issue.

Identity: ``sender.user_id`` is MAX's own numeric user id — a real authenticated
identity (unlike ntfy's topic-as-identity hack). Standard allowlist semantics apply.
"""

import asyncio
import logging
from datetime import datetime, UTC
from typing import Any, Optional

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    httpx = None  # type: ignore[assignment]

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.platforms._shared import get_scoped_secret as _get_scoped_secret, send_error
from gateway.platforms.helpers import MessageDeduplicator
from gateway.platforms._shared import (
    extra_or_secret as _extra_or_secret, seed_extra_from_env as _seed_extra_from_env
)

logger = logging.getLogger(__name__)


class _FatalPollError(Exception):
    """Unrecoverable polling error (401)."""


DEFAULT_SERVER = "https://platform-api2.max.ru"
MAX_MESSAGE_LENGTH = 4000  # MAX text limit per /messages docs
DEDUP_WINDOW_SECONDS = 300
DEDUP_MAX_SIZE = 1000
RECONNECT_BACKOFF = [2, 5, 10, 30, 60]
POLL_TIMEOUT_SECONDS = 30  # GET /updates long-poll timeout (MAX accepts 0-90)
POLL_HTTP_TIMEOUT_SECONDS = POLL_TIMEOUT_SECONDS + 15


def _server_url(extra: dict[str, Any]) -> str:
    return _extra_or_secret(extra, "server", "MAX_SERVER_URL", DEFAULT_SERVER).rstrip("/")


def _ssl_verify() -> Any:
    """MAX's platform-api2.max.ru chains to the Russian Ministry of Digital Development's
    (Минцифры) own root CA, which standard trust stores (incl. certifi) don't carry. A merged
    bundle (certifi + Russian Trusted Root/Sub CA) at ``~/.hermes/certs/ru_trusted_ca_bundle.pem``
    is used when present; falls back to httpx's default verification otherwise (connect will then
    fail with CERTIFICATE_VERIFY_FAILED until that bundle is installed — see plugin.yaml note)."""
    import os
    bundle = os.path.expanduser("~/.hermes/certs/ru_trusted_ca_bundle.pem")
    return bundle if os.path.isfile(bundle) else True


def check_requirements() -> bool:
    """Installable and minimally configured (reads MAX_BOT_TOKEN directly — no full config load)."""
    return HTTPX_AVAILABLE and bool(_get_scoped_secret("MAX_BOT_TOKEN", "").strip())


def validate_config(config) -> bool:
    """True when a bot token is configured (config.yaml ``extra`` or env)."""
    return bool(_extra_or_secret(getattr(config, "extra", {}) or {}, "token", "MAX_BOT_TOKEN"))


def is_connected(config) -> bool:
    """Check whether MAX is configured (env or config.yaml)."""
    return bool(_get_scoped_secret("MAX_BOT_TOKEN") or (getattr(config, "extra", {}) or {}).get("token", ""))


class MaxAdapter(BasePlatformAdapter):
    """MAX messenger adapter: GET /updates long polling in, POST /messages out."""

    MAX_MESSAGE_LENGTH = MAX_MESSAGE_LENGTH

    def __init__(self, config: PlatformConfig):
        super().__init__(config=config, platform=Platform("max"))
        extra = config.extra or {}
        self._server: str = _server_url(extra)
        self._token: str = _extra_or_secret(extra, "token", "MAX_BOT_TOKEN")
        self._poll_task: Optional[asyncio.Task] = None
        self._http_client: Optional[httpx.AsyncClient] = None
        self._marker: Optional[int] = None
        self._dedup = MessageDeduplicator(max_size=DEDUP_MAX_SIZE, ttl_seconds=DEDUP_WINDOW_SECONDS)

    # -- Connection lifecycle -----------------------------------------------

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        """Connect to MAX by starting the GET /updates long-polling task."""
        if not HTTPX_AVAILABLE:
            logger.warning("[%s] httpx not installed. Run: pip install httpx", self.name)
            return False
        if not self._token:
            logger.warning("[%s] MAX_BOT_TOKEN not configured", self.name)
            return False
        try:
            self._http_client = httpx.AsyncClient(
                base_url=self._server, headers={"Authorization": self._token}, timeout=None,
                verify=_ssl_verify())
            self._poll_task = asyncio.create_task(self._run_poll())
            self._mark_connected()
            logger.info("[%s] Connected — long-polling %s/updates", self.name, self._server)
            self._wire_plugin_handlers(None)
            asyncio.create_task(self._sync_command_menu())
            return True
        except Exception as e:
            logger.error("[%s] Failed to connect: %s", self.name, e)
            return False

    # MAX caps PATCH /me's ``commands`` at 32 entries, 1-64 char names, 1-128 char descriptions.
    _MAX_COMMAND_MENU_LIMIT = 32
    _MAX_COMMAND_DESC_LIMIT = 128

    async def _sync_command_menu(self) -> None:
        """Push the same slash-command menu Telegram shows (``/help``, ``/new``, ...) to MAX via
        ``PATCH /me``, reusing the shared, already-sanitized command list so both platforms stay
        in sync without a second source of truth."""
        if not self._http_client:
            return
        try:
            from hermes_cli.commands_platforms import telegram_menu_commands
            pairs, _hidden = telegram_menu_commands(max_commands=self._MAX_COMMAND_MENU_LIMIT)
            commands = [
                {"name": name, "description": (desc or name)[: self._MAX_COMMAND_DESC_LIMIT]}
                for name, desc in pairs]
            resp = await self._http_client.patch("/me/commands", json={"commands": commands}, timeout=15.0)
            if resp.status_code < 300:
                logger.info("[%s] Command menu synced (%d commands)", self.name, len(commands))
            else:
                logger.warning(
                    "[%s] Command menu sync failed HTTP %d: %s",
                    self.name, resp.status_code, resp.text[:200])
        except Exception:
            logger.warning("[%s] Command menu sync failed", self.name, exc_info=True)

    async def _run_poll(self) -> None:
        """Long-poll GET /updates with automatic reconnection."""
        backoff_idx = 0
        while self._running:
            try:
                await self._poll_once()
                backoff_idx = 0  # a clean round resets backoff immediately (unlike streaming)
                continue
            except asyncio.CancelledError:
                return
            except _FatalPollError:
                self._running = False
                return
            except Exception as e:
                if not self._running:
                    return
                logger.warning("[%s] Poll error: %s", self.name, e)
            if not self._running:
                return
            delay = RECONNECT_BACKOFF[min(backoff_idx, len(RECONNECT_BACKOFF) - 1)]
            logger.info("[%s] Retrying in %ds...", self.name, delay)
            await asyncio.sleep(delay)
            backoff_idx += 1

    def _fatal_status(self, status_code: int) -> None:
        """401 is unrecoverable: log, set the fatal state and raise ``_FatalPollError``."""
        if status_code != 401:
            return
        logger.error("[%s] Authentication failed (401) — stopping poll loop. Check MAX_BOT_TOKEN.", self.name)
        self._set_fatal_error(
            "max_unauthorized", "MAX server rejected auth (401). Check MAX_BOT_TOKEN.", retryable=False)
        raise _FatalPollError("401 Unauthorized")

    async def _poll_once(self) -> None:
        """One GET /updates round: dispatch events, advance the marker."""
        params: dict[str, Any] = {
            "limit": 100, "timeout": POLL_TIMEOUT_SECONDS, "types": "message_created"}
        if self._marker is not None:
            params["marker"] = self._marker
        resp = await self._http_client.get(
            "/updates", params=params, timeout=httpx.Timeout(POLL_HTTP_TIMEOUT_SECONDS))
        self._fatal_status(resp.status_code)
        resp.raise_for_status()
        data = resp.json()
        for update in data.get("updates") or []:
            if not self._running:
                return
            if update.get("update_type") == "message_created":
                await self._on_message(update)
        marker = data.get("marker")
        if marker is not None:
            self._marker = marker

    async def disconnect(self) -> None:
        """Disconnect from MAX."""
        self._running = False
        self._mark_disconnected()
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
        self._dedup.clear()
        logger.info("[%s] Disconnected", self.name)

    # -- Inbound message processing -----------------------------------------

    async def _on_message(self, update: dict[str, Any]) -> None:
        """Process an incoming ``message_created`` update."""
        message = update.get("message") or {}
        body = message.get("body") or {}
        msg_id = body.get("mid") or str(body.get("seq") or "")
        if not msg_id:
            logger.debug("[%s] Message with no mid/seq, skipping", self.name)
            return
        if self._dedup.is_duplicate(msg_id):
            logger.debug("[%s] Duplicate message %s, skipping", self.name, msg_id)
            return
        text = (body.get("text") or "").strip()
        if not text:
            logger.debug("[%s] Empty/unsupported (non-text) message body, skipping", self.name)
            return
        sender = message.get("sender") or {}
        recipient = message.get("recipient") or {}
        user_id = str(sender.get("user_id") or "")
        if not user_id:
            logger.debug("[%s] Message with no sender.user_id, skipping", self.name)
            return
        user_name = sender.get("name") or sender.get("first_name") or user_id
        chat_type_raw = recipient.get("chat_type") or "dialog"
        chat_type = "dm" if chat_type_raw == "dialog" else "group"
        # Dialogs: POST /messages addresses the human by sender.user_id — recipient.chat_id in a
        # dialog Update is MAX's internal dialog id, NOT a valid ?user_id= value (sending to it
        # 404s with "dialog.not.found"). Only groups/channels route by recipient.chat_id.
        chat_id = str(recipient.get("chat_id")) if chat_type == "group" and recipient.get("chat_id") else user_id
        source = self.build_source(
            chat_id=chat_id, chat_name=user_name, chat_type=chat_type,
            user_id=user_id, user_name=user_name, message_id=msg_id)
        ts_ms = message.get("timestamp")
        try:
            timestamp = datetime.fromtimestamp(int(ts_ms) / 1000, tz=UTC) if ts_ms else datetime.now(tz=UTC)
        except (ValueError, OSError, TypeError):
            timestamp = datetime.now(tz=UTC)
        message_event = MessageEvent(
            text=text, message_type=MessageType.TEXT, source=source, message_id=msg_id,
            raw_message=update, timestamp=timestamp)
        logger.debug("[%s] Message from %s: %s", self.name, user_id, text[:80])
        await self.handle_message(message_event)

    # -- Outbound messaging -------------------------------------------------

    async def send(
        self, chat_id: str, content: str, reply_to: Optional[str] = None, metadata: Optional[dict[str, Any]] = None,
    ) -> SendResult:
        """Send a message via POST /messages. ``chat_id`` is a dialog user_id or a group/channel chat_id."""
        if not self._http_client:
            return SendResult(success=False, error="HTTP client not initialized")
        if len(content) > self.MAX_MESSAGE_LENGTH:
            logger.warning(
                "[%s] Message truncated from %d to %d chars (MAX limit)",
                self.name, len(content), self.MAX_MESSAGE_LENGTH)
        text = content[:self.MAX_MESSAGE_LENGTH]
        chat_type = (metadata or {}).get("chat_type")
        query_key = "chat_id" if chat_type == "group" else "user_id"
        try:
            resp = await self._http_client.post(
                "/messages", params={query_key: chat_id}, json={"text": text, "notify": True}, timeout=15.0)
            if resp.status_code < 300:
                body = (resp.json() or {}).get("message", {}).get("body", {})
                return SendResult(success=True, message_id=body.get("mid") or "")
            body_text = resp.text
            logger.warning("[%s] Send failed HTTP %d: %s", self.name, resp.status_code, body_text[:200])
            return SendResult(success=False, error=f"HTTP {resp.status_code}: {body_text[:200]}")
        except httpx.TimeoutException:
            return SendResult(success=False, error="Timeout sending to MAX")
        except Exception as e:
            logger.error("[%s] Send error: %s", self.name, e)
            return SendResult(success=False, error=str(e))

    async def get_chat_info(self, chat_id: str) -> dict[str, Any]:
        return {"name": chat_id, "type": "dm"}


# -- Plugin registration -----------------------------------------------------


def _env_enablement() -> dict | None:
    """``env_enablement_fn``: seed ``PlatformConfig.extra`` from the profile's env before adapter
    construction; ``None`` when ``MAX_BOT_TOKEN`` is unset."""
    token = _get_scoped_secret("MAX_BOT_TOKEN", "").strip()
    if not token:
        return None
    seed = _seed_extra_from_env((
        ("MAX_SERVER_URL", "server", lambda v: v.rstrip("/")),
    ), home_env="MAX_HOME_CHANNEL", home_default="")
    return {"token": token, "server": seed.pop("server", DEFAULT_SERVER), **seed}


async def _standalone_send(
    pconfig, chat_id: str, message: str, *,
    thread_id: Optional[str] = None, media_files: Optional[list[str]] = None, force_document: bool = False,
) -> dict[str, Any]:
    """Out-of-process send for cron / send_message_tool when no gateway adapter is live.

    ``thread_id``/``media_files``/``force_document`` are signature parity only — MAX dialogs
    have no thread primitive here, and this minimal adapter sends text only.
    """
    if not HTTPX_AVAILABLE:
        return send_error("MAX standalone send: httpx not installed")
    extra = getattr(pconfig, "extra", {}) or {}
    server = _server_url(extra)
    token = _extra_or_secret(extra, "token", "MAX_BOT_TOKEN")
    target = chat_id or extra.get("home_channel") or _get_scoped_secret("MAX_HOME_CHANNEL", "").strip()
    if not token:
        return send_error("MAX standalone send: MAX_BOT_TOKEN not configured")
    if not target:
        return send_error("MAX standalone send: no destination chat_id/user_id configured")
    text = message[:MAX_MESSAGE_LENGTH]
    try:
        async with httpx.AsyncClient(
            base_url=server, headers={"Authorization": token}, timeout=15.0, verify=_ssl_verify(),
        ) as client:
            resp = await client.post("/messages", params={"user_id": target}, json={"text": text, "notify": True})
        if resp.status_code >= 300:
            return send_error(f"MAX HTTP {resp.status_code}: {resp.text[:200]}")
        body = (resp.json() or {}).get("message", {}).get("body", {})
        return {"success": True, "platform": "max", "chat_id": target, "message_id": body.get("mid") or ""}
    except Exception as e:
        return send_error(f"MAX standalone send failed: {e}")


def register(ctx) -> None:
    """Plugin entry point — called by the Hermes plugin system at startup."""
    from .dashboard_card import DASHBOARD_CARD
    ctx.register_platform(
        name="max", label="MAX", adapter_factory=lambda cfg: MaxAdapter(cfg),
        check_fn=check_requirements, validate_config=validate_config, is_connected=is_connected,
        **DASHBOARD_CARD,
        required_env=["MAX_BOT_TOKEN"], install_hint="pip install httpx   # already a Hermes dependency",
        env_enablement_fn=_env_enablement,  # env-only setups show in `gateway status`
        cron_deliver_env_var="MAX_HOME_CHANNEL",
        standalone_sender_fn=_standalone_send,  # out-of-process cron delivery
        allowed_users_env="MAX_ALLOWED_USERS", allow_all_env="MAX_ALLOW_ALL_USERS",
        max_message_length=MAX_MESSAGE_LENGTH, emoji="💬",
        pii_safe=False,  # MAX user_id/name may carry real-identity PII, unlike ntfy's topic handle
        allow_update_command=True,
        platform_hint=(
            "You are communicating via the MAX messenger (max.ru, RU). "
            "Use plain text only — this minimal adapter has no markdown/HTML formatting, "
            "media, buttons, or thread support yet. Keep responses under 4000 characters."
        ))

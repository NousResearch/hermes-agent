"""Relay platform adapter (Hermes plugin).

Connects Hermes to Relay — the messenger where AI agents appear as
contacts. Inbound messages arrive by long-polling Relay's durable event
log (``GET /v1/events``); replies are sent through ``POST /v1/messages``.
Like Telegram's ``getUpdates``, polling needs no webhook, no public URL,
and no signing secret — right for machines behind NAT. No external SDK —
only httpx, which is already a Hermes dependency.

This adapter ships as a Hermes platform plugin under
``plugins/platforms/relay/``. The Hermes plugin loader scans the
directory at startup, calls :func:`register`, and the platform becomes
available to ``gateway/run.py`` and ``tools/send_message_tool`` through
the registry — no edits to core files required.

Configuration in config.yaml::

    platforms:
      relay:
        enabled: true
        extra:
          token: "relay_agt_live_..."       # Agent Token, shown once at creation
          api_url: "https://api.relayapp.im"
          cursor_path: "~/.hermes/relay_cursor"

Environment variables (env wins over config.yaml ``extra``):

    RELAY_AGENT_TOKEN          Agent Token (required)
    RELAY_API_URL              API origin (default: https://api.relayapp.im)
    RELAY_ALLOWED_USERS        Comma-separated Relay user ids (usr_...)
    RELAY_ALLOW_ALL_USERS      Allow any user (default behavior for DMs)
    RELAY_HOME_CHANNEL         Conversation id (cnv_...) for cron delivery
    RELAY_HOME_CHANNEL_NAME    Human label for the home channel
    RELAY_CURSOR_PATH          Cursor persistence file (default: ~/.hermes/relay_cursor)

Identity model: Relay authenticates senders server-side. ``sender.id``
(``usr_...``) is a real authenticated identity, safe for authorization.
Delivery is at-least-once — the adapter deduplicates by ``event_id`` and
persists the opaque poll cursor so restarts do not replay the log.
"""

import asyncio
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
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

logger = logging.getLogger(__name__)


class _FatalPollError(Exception):
    """Raised when polling hits an unrecoverable error (e.g. 401)."""


DEFAULT_API_URL = "https://api.relayapp.im"
DEFAULT_CURSOR_PATH = "~/.hermes/relay_cursor"
MAX_MESSAGE_LENGTH = 8000  # Relay caps text parts at 8 KB
POLL_TIMEOUT_SECONDS = 25  # server clamps to 0-30
DEDUP_WINDOW_SECONDS = 600
DEDUP_MAX_SIZE = 2000
RECONNECT_BACKOFF = [2, 5, 10, 30, 60]


def _resolve(extra: Dict[str, Any], key: str, env: str, default: str = "") -> str:
    """Env wins over config.yaml ``extra``; both stripped."""
    return (os.getenv(env, "").strip() or str(extra.get(key, "") or "").strip() or default)


def check_requirements() -> bool:
    """Installable and minimally configured (token present, httpx importable)."""
    if not HTTPX_AVAILABLE:
        return False
    return bool(os.getenv("RELAY_AGENT_TOKEN", "").strip())


def validate_config(config) -> bool:
    extra = getattr(config, "extra", {}) or {}
    return bool(_resolve(extra, "token", "RELAY_AGENT_TOKEN"))


def is_connected(config) -> bool:
    extra = getattr(config, "extra", {}) or {}
    return bool(_resolve(extra, "token", "RELAY_AGENT_TOKEN"))


class RelayAdapter(BasePlatformAdapter):
    """Relay adapter: long-poll ``/v1/events`` in; ``POST /v1/messages`` out.

    Streaming uses Relay's native draft lifecycle (draft → append →
    finalize) instead of progressive edits: Relay's v0 developer API has
    no message-edit endpoint, so ``SUPPORTS_MESSAGE_EDITING`` is False and
    the stream consumer's draft transport is the only streaming path.
    """

    MAX_MESSAGE_LENGTH = MAX_MESSAGE_LENGTH
    # No edit API in Relay v0 — tells the stream consumer to skip the
    # progressive-edit cursor path entirely.
    SUPPORTS_MESSAGE_EDITING = False

    def __init__(self, config: PlatformConfig):
        platform = Platform("relay")
        super().__init__(config=config, platform=platform)

        extra = config.extra or {}
        self._api_url: str = _resolve(extra, "api_url", "RELAY_API_URL", DEFAULT_API_URL).rstrip("/")
        self._token: str = _resolve(extra, "token", "RELAY_AGENT_TOKEN")
        self._cursor_path = Path(
            _resolve(extra, "cursor_path", "RELAY_CURSOR_PATH", DEFAULT_CURSOR_PATH)
        ).expanduser()

        self._agent_id: str = ""
        self._agent_handle: str = ""
        self._cursor: str = self._load_cursor()
        self._poll_task: Optional[asyncio.Task] = None
        self._http_client: Optional["httpx.AsyncClient"] = None

        # event_id -> timestamp, at-least-once dedup window
        self._seen_events: Dict[str, float] = {}

        # Streaming drafts: chat_id -> (hermes draft_id, relay message id,
        # text already appended). One open draft per conversation.
        self._open_drafts: Dict[str, tuple] = {}

    # -- Connection lifecycle -----------------------------------------------

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        """Verify the Agent Token, then start the long-poll task."""
        if not HTTPX_AVAILABLE:
            logger.warning("[%s] httpx not installed. Run: pip install httpx", self.name)
            return False
        if not self._token:
            logger.warning("[%s] RELAY_AGENT_TOKEN not configured", self.name)
            return False

        try:
            self._http_client = httpx.AsyncClient(
                headers={"Authorization": f"Bearer {self._token}"},
                timeout=httpx.Timeout(connect=15.0, read=POLL_TIMEOUT_SECONDS + 15, write=15.0, pool=15.0),
            )
            resp = await self._http_client.get(f"{self._api_url}/v1/agents/me")
            if resp.status_code == 401:
                self._set_fatal_error(
                    "relay_unauthorized",
                    "Relay rejected the Agent Token (401). Check RELAY_AGENT_TOKEN.",
                    retryable=False,
                )
                logger.error("[%s] Agent Token rejected (401)", self.name)
                await self._http_client.aclose()
                self._http_client = None
                return False
            resp.raise_for_status()
            agent = resp.json().get("agent", {})
            self._agent_id = agent.get("id", "")
            self._agent_handle = agent.get("handle", "")

            self._poll_task = asyncio.create_task(self._run_poll_loop())
            self._mark_connected()
            logger.info(
                "[%s] Connected as @%s (%s) via %s",
                self.name, self._agent_handle, self._agent_id, self._api_url,
            )
            return True
        except Exception as e:
            logger.error("[%s] Failed to connect: %s", self.name, e)
            if self._http_client:
                await self._http_client.aclose()
                self._http_client = None
            return False

    async def _run_poll_loop(self) -> None:
        """Long-poll /v1/events with automatic backoff on errors."""
        backoff_idx = 0
        while self._running:
            try:
                await self._poll_once()
                backoff_idx = 0
            except asyncio.CancelledError:
                return
            except _FatalPollError:
                self._running = False
                return
            except Exception as e:
                if not self._running:
                    return
                delay = RECONNECT_BACKOFF[min(backoff_idx, len(RECONNECT_BACKOFF) - 1)]
                logger.warning("[%s] Poll error: %s — retrying in %ds", self.name, e, delay)
                await asyncio.sleep(delay)
                backoff_idx += 1

    async def _poll_once(self) -> None:
        params: Dict[str, Any] = {"timeout": POLL_TIMEOUT_SECONDS}
        if self._cursor:
            params["cursor"] = self._cursor

        resp = await self._http_client.get(f"{self._api_url}/v1/events", params=params)
        if resp.status_code == 401:
            self._set_fatal_error(
                "relay_unauthorized",
                "Relay rejected the Agent Token (401) while polling. Check RELAY_AGENT_TOKEN.",
                retryable=False,
            )
            raise _FatalPollError("401 Unauthorized")
        resp.raise_for_status()
        data = resp.json()

        for event in data.get("events", []):
            if not self._running:
                return
            await self._on_event(event)

        next_cursor = data.get("next_cursor")
        if next_cursor and next_cursor != self._cursor:
            self._cursor = next_cursor
            self._save_cursor(next_cursor)

    async def disconnect(self) -> None:
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
            # Never strand a streaming draft as an eternal "typing" bubble.
            for chat_id in list(self._open_drafts):
                await self._finalize_draft(chat_id, parts_text=None)
            await self._http_client.aclose()
            self._http_client = None

        self._seen_events.clear()
        logger.info("[%s] Disconnected", self.name)

    # -- Inbound event processing ---------------------------------------------

    async def _on_event(self, event: Dict[str, Any]) -> None:
        event_id = event.get("event_id") or uuid.uuid4().hex
        if self._is_duplicate(event_id):
            logger.debug("[%s] Duplicate event %s, skipping", self.name, event_id)
            return
        if event.get("event_type") != "message.received":
            logger.debug("[%s] Ignoring event type %s", self.name, event.get("event_type"))
            return

        message = (event.get("data") or {}).get("message") or {}
        sender = message.get("sender") or {}
        # Echo-loop prevention: never react to this agent's own messages.
        if sender.get("id") == self._agent_id:
            return

        text = self._render_text(message)
        if not text:
            logger.debug("[%s] Message %s has no textual content, skipping", self.name, message.get("id"))
            return

        conversation_id = message.get("conversation_id") or ""
        user_id = sender.get("id") or "unknown"

        source = self.build_source(
            chat_id=conversation_id,
            chat_name=f"Relay conversation {conversation_id}",
            chat_type="dm",
            user_id=user_id,
            user_name=user_id,
        )

        timestamp = self._parse_timestamp(message.get("created_at"))
        message_event = MessageEvent(
            text=text,
            message_type=MessageType.TEXT,
            source=source,
            message_id=message.get("id") or event_id,
            raw_message=event,
            timestamp=timestamp,
        )

        logger.debug("[%s] Message in %s: %s", self.name, conversation_id, text[:80])
        await self.handle_message(message_event)

    @staticmethod
    def _render_text(message: Dict[str, Any]) -> str:
        """Join text/link parts in order; fall back to fallback_text."""
        chunks: List[str] = []
        for part in message.get("parts", []):
            kind = part.get("type")
            if kind in ("text", "link") and part.get("text"):
                chunks.append(part["text"])
            elif kind == "link" and part.get("url"):
                chunks.append(part["url"])
            elif kind == "data" and part.get("data") is not None:
                chunks.append(json.dumps(part["data"], ensure_ascii=False))
        rendered = "\n".join(chunks).strip()
        return rendered or (message.get("fallback_text") or "").strip()

    @staticmethod
    def _parse_timestamp(created_at: Optional[str]) -> datetime:
        if created_at:
            try:
                return datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            except ValueError:
                pass
        return datetime.now(tz=timezone.utc)

    # -- Deduplication --------------------------------------------------------

    def _is_duplicate(self, event_id: str) -> bool:
        now = time.time()
        if len(self._seen_events) > DEDUP_MAX_SIZE:
            cutoff = now - DEDUP_WINDOW_SECONDS
            self._seen_events = {k: v for k, v in self._seen_events.items() if v > cutoff}
        if event_id in self._seen_events:
            return True
        self._seen_events[event_id] = now
        return False

    # -- Cursor persistence -----------------------------------------------------

    def _load_cursor(self) -> str:
        try:
            return self._cursor_path.read_text().strip()
        except OSError:
            return ""

    def _save_cursor(self, cursor: str) -> None:
        try:
            self._cursor_path.parent.mkdir(parents=True, exist_ok=True)
            self._cursor_path.write_text(cursor)
        except OSError as e:
            logger.warning("[%s] Could not persist cursor: %s", self.name, e)

    # -- Streaming drafts (draft → append → finalize) ---------------------------

    def supports_draft_streaming(
        self,
        chat_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Relay drafts work in any conversation the agent participates in."""
        return True

    async def send_draft(
        self,
        chat_id: str,
        draft_id: int,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Stream one frame into a native Relay draft.

        The consumer passes the full accumulated text each frame; Relay's
        append endpoint takes deltas, so this tracks what has been appended
        and sends only the suffix. Relay drafts are real messages that must
        be finalized — :meth:`send` finalizes the open draft with the
        authoritative final text when the response completes.
        """
        if not self._http_client:
            return SendResult(success=False, error="HTTP client not initialized")

        open_draft = self._open_drafts.get(chat_id)
        if open_draft and open_draft[0] != draft_id:
            # A new response started while an older draft never finalized
            # (e.g. a crashed stream). Close it out before opening another.
            await self._finalize_draft(chat_id, parts_text=None)
            open_draft = None

        try:
            if open_draft is None:
                resp = await self._http_client.post(
                    f"{self._api_url}/v1/messages",
                    json={"conversation_id": chat_id, "draft": True},
                    headers={"Idempotency-Key": f"hermes-draft-{uuid.uuid4().hex}"},
                    timeout=20.0,
                )
                if resp.status_code >= 300:
                    return SendResult(success=False, error=f"draft open HTTP {resp.status_code}: {resp.text[:200]}")
                message_id = resp.json().get("message_id")
                if not message_id:
                    return SendResult(success=False, error="draft open returned no message_id")
                self._open_drafts[chat_id] = (draft_id, message_id, "")
                open_draft = self._open_drafts[chat_id]

            _, message_id, sent = open_draft
            if not content.startswith(sent):
                # The consumer rewrote earlier text (e.g. reasoning-tag
                # stripping). Appends can't express rewrites — decline this
                # frame; the consumer falls back and send() finalizes with
                # the authoritative final text.
                return SendResult(success=False, error="draft content rewritten; append-only transport declined")

            delta = content[len(sent):]
            if delta:
                resp = await self._http_client.post(
                    f"{self._api_url}/v1/messages/{message_id}/append",
                    json={"text": delta[: self.MAX_MESSAGE_LENGTH]},
                    timeout=20.0,
                )
                if resp.status_code >= 300:
                    return SendResult(success=False, error=f"append HTTP {resp.status_code}: {resp.text[:200]}")
                self._open_drafts[chat_id] = (draft_id, message_id, sent + delta)
            return SendResult(success=True, message_id=message_id)
        except httpx.TimeoutException:
            return SendResult(success=False, error="Timeout streaming draft to Relay")
        except Exception as e:
            logger.error("[%s] Draft frame error: %s", self.name, e)
            return SendResult(success=False, error=str(e))

    async def _finalize_draft(self, chat_id: str, parts_text: Optional[str]) -> Optional[SendResult]:
        """Finalize the open draft for a chat.

        ``parts_text`` replaces the accumulated content when given (the
        authoritative final answer); ``None`` keeps whatever accumulated.
        Returns None when there is no open draft.
        """
        open_draft = self._open_drafts.pop(chat_id, None)
        if not open_draft or not self._http_client:
            return None
        _, message_id, _ = open_draft
        body: Dict[str, Any] = {}
        if parts_text is not None:
            body["parts"] = [{"type": "text", "text": parts_text[: self.MAX_MESSAGE_LENGTH]}]
        try:
            resp = await self._http_client.post(
                f"{self._api_url}/v1/messages/{message_id}/finalize",
                json=body,
                timeout=20.0,
            )
            if resp.status_code < 300:
                return SendResult(success=True, message_id=message_id)
            return SendResult(success=False, error=f"finalize HTTP {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            logger.warning("[%s] Draft finalize error: %s", self.name, e)
            return SendResult(success=False, error=str(e))

    # -- Outbound messaging -----------------------------------------------------

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a text message into a Relay conversation.

        When a streaming draft is open for this conversation, the "send"
        is the final answer of that stream — finalize the draft with the
        authoritative text instead of creating a second message.
        """
        if not self._http_client:
            return SendResult(success=False, error="HTTP client not initialized")
        if not chat_id:
            return SendResult(success=False, error="No conversation id")

        if chat_id in self._open_drafts:
            finalized = await self._finalize_draft(chat_id, parts_text=content)
            if finalized is not None and finalized.success:
                return finalized
            # Finalize failed — fall through and deliver as a fresh message
            # so the reply is never lost.

        if len(content) > self.MAX_MESSAGE_LENGTH:
            logger.warning(
                "[%s] Message truncated from %d to %d chars (Relay text-part limit)",
                self.name, len(content), self.MAX_MESSAGE_LENGTH,
            )
            content = content[: self.MAX_MESSAGE_LENGTH]

        body: Dict[str, Any] = {
            "conversation_id": chat_id,
            "parts": [{"type": "text", "text": content}],
        }
        if reply_to:
            body["reply_to"] = {"message_id": reply_to}

        try:
            resp = await self._http_client.post(
                f"{self._api_url}/v1/messages",
                json=body,
                headers={"Idempotency-Key": f"hermes-{uuid.uuid4().hex}"},
                timeout=20.0,
            )
            if resp.status_code < 300:
                data = resp.json()
                return SendResult(success=True, message_id=data.get("message_id"))
            detail = resp.text[:200]
            logger.warning("[%s] Send failed HTTP %d: %s", self.name, resp.status_code, detail)
            return SendResult(success=False, error=f"HTTP {resp.status_code}: {detail}")
        except httpx.TimeoutException:
            return SendResult(success=False, error="Timeout sending to Relay")
        except Exception as e:
            logger.error("[%s] Send error: %s", self.name, e)
            return SendResult(success=False, error=str(e))

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """Typing indicators are not part of the Relay v0 developer API."""
        pass

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {"name": chat_id, "type": "dm"}


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------


def _env_enablement() -> dict | None:
    """Seed ``PlatformConfig.extra`` from env vars during gateway config load."""
    token = os.getenv("RELAY_AGENT_TOKEN", "").strip()
    if not token:
        return None
    seed: dict = {
        "token": token,
        "api_url": os.getenv("RELAY_API_URL", DEFAULT_API_URL).rstrip("/"),
    }
    cursor_path = os.getenv("RELAY_CURSOR_PATH", "").strip()
    if cursor_path:
        seed["cursor_path"] = cursor_path
    home = os.getenv("RELAY_HOME_CHANNEL", "").strip()
    if home:
        seed["home_channel"] = {
            "chat_id": home,
            "name": os.getenv("RELAY_HOME_CHANNEL_NAME", home),
        }
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
    """Out-of-process send for cron / send_message_tool fallbacks.

    ``thread_id`` and ``media_files`` are accepted for signature parity;
    Relay's v0 developer API sends text parts only.
    """
    if not HTTPX_AVAILABLE:
        return {"error": "relay standalone send: httpx not installed"}

    extra = getattr(pconfig, "extra", {}) or {}
    token = _resolve(extra, "token", "RELAY_AGENT_TOKEN")
    api_url = _resolve(extra, "api_url", "RELAY_API_URL", DEFAULT_API_URL).rstrip("/")
    if not token:
        return {"error": "relay standalone send: RELAY_AGENT_TOKEN not configured"}
    if not chat_id:
        return {"error": "relay standalone send: no conversation id (set RELAY_HOME_CHANNEL)"}

    body = {
        "conversation_id": chat_id,
        "parts": [{"type": "text", "text": message[:MAX_MESSAGE_LENGTH]}],
    }
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(
                f"{api_url}/v1/messages",
                json=body,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Idempotency-Key": f"hermes-cron-{uuid.uuid4().hex}",
                },
            )
        if resp.status_code >= 300:
            return {"error": f"relay HTTP {resp.status_code}: {resp.text[:200]}"}
        data = resp.json()
        return {
            "success": True,
            "platform": "relay",
            "chat_id": chat_id,
            "message_id": data.get("message_id"),
        }
    except Exception as e:
        return {"error": f"relay standalone send failed: {e}"}


def register(ctx) -> None:
    """Plugin entry point — called by the Hermes plugin system at startup."""
    ctx.register_platform(
        name="relay",
        label="Relay",
        adapter_factory=lambda cfg: RelayAdapter(cfg),
        check_fn=check_requirements,
        validate_config=validate_config,
        is_connected=is_connected,
        required_env=["RELAY_AGENT_TOKEN"],
        install_hint="pip install httpx   # already a Hermes dependency",
        env_enablement_fn=_env_enablement,
        cron_deliver_env_var="RELAY_HOME_CHANNEL",
        standalone_sender_fn=_standalone_send,
        allowed_users_env="RELAY_ALLOWED_USERS",
        allow_all_env="RELAY_ALLOW_ALL_USERS",
        max_message_length=MAX_MESSAGE_LENGTH,
        emoji="✳️",
        # Relay sender ids are server-authenticated opaque ids (usr_...),
        # no phone numbers or emails cross this adapter.
        pii_safe=True,
        allow_update_command=True,
        platform_hint=(
            "You are texting inside Relay, a messenger where you appear as a "
            "contact. Write like a person texting: short, direct messages, "
            "plain text, no markdown headings. Long content should be split "
            "into a few short messages rather than one wall of text."
        ),
    )

"""
DingTalk platform adapter using Stream Mode.

Uses dingtalk-stream SDK for real-time message reception without webhooks.
Responses are sent via DingTalk's session webhook (markdown format).

Requires:
    pip install dingtalk-stream httpx
    DINGTALK_CLIENT_ID and DINGTALK_CLIENT_SECRET env vars

Configuration in config.yaml:
    platforms:
      dingtalk:
        enabled: true
        extra:
          client_id: "your-app-key"      # or DINGTALK_CLIENT_ID env var
          client_secret: "your-secret"   # or DINGTALK_CLIENT_SECRET env var
"""

import asyncio
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from urllib.parse import urlparse

try:
    import dingtalk_stream
    from dingtalk_stream import ChatbotHandler, ChatbotMessage
    DINGTALK_STREAM_AVAILABLE = True
except ImportError:
    DINGTALK_STREAM_AVAILABLE = False
    dingtalk_stream = None  # type: ignore[assignment]

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
    ProcessingOutcome,
    SendResult,
)

logger = logging.getLogger(__name__)

MAX_MESSAGE_LENGTH = 20000
DEDUP_WINDOW_SECONDS = 300
DEDUP_MAX_SIZE = 1000
RECONNECT_BACKOFF = [2, 5, 10, 30, 60]
_SESSION_WEBHOOKS_MAX = 500
# DingTalk may return session reply URLs on either host; both must be allowed or
# replies fail with "No session_webhook" after SSRF-style validation drops the URL.
_ALLOWED_SESSION_WEBHOOK_HOSTS = frozenset({"api.dingtalk.com", "oapi.dingtalk.com"})

# Emotion reaction constants (matches openclaw ack-reaction-service)
_EMOTION_NAME = "\U0001f914\u601d\u8003\u4e2d"  # 🤔思考中
_EMOTION_ID = "2659900"
_EMOTION_BG_ID = "im_bg_1"
_EMOTION_ATTACH_DELAYS = [0, 0.4, 1.2]
_EMOTION_RECALL_DELAYS = [0, 1.5, 5.0]


def _session_webhook_is_allowed(url: str) -> bool:
    """True if URL is HTTPS and host is an official DingTalk session-reply domain."""
    if not url or not isinstance(url, str):
        return False
    try:
        parsed = urlparse(url.strip())
    except Exception:
        return False
    if parsed.scheme != "https":
        return False
    host = (parsed.hostname or "").lower()
    return host in _ALLOWED_SESSION_WEBHOOK_HOSTS


def check_dingtalk_requirements() -> bool:
    """Check if DingTalk dependencies are available and configured."""
    if not DINGTALK_STREAM_AVAILABLE or not HTTPX_AVAILABLE:
        return False
    if not os.getenv("DINGTALK_CLIENT_ID") or not os.getenv("DINGTALK_CLIENT_SECRET"):
        return False
    return True


class DingTalkAdapter(BasePlatformAdapter):
    """DingTalk chatbot adapter using Stream Mode.

    The dingtalk-stream SDK maintains a long-lived WebSocket connection.
    Incoming messages arrive via a ChatbotHandler callback. Replies are
    sent via the incoming message's session_webhook URL using httpx.
    """

    MAX_MESSAGE_LENGTH = MAX_MESSAGE_LENGTH

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.DINGTALK)

        extra = config.extra or {}
        self._client_id: str = extra.get("client_id") or os.getenv("DINGTALK_CLIENT_ID", "")
        self._client_secret: str = extra.get("client_secret") or os.getenv("DINGTALK_CLIENT_SECRET", "")

        # Conversation allowlist: if set, only messages from these conversation_ids are processed
        _raw_allowed = extra.get("allowed_conversations") or os.getenv("DINGTALK_ALLOWED_CONVERSATIONS", "")
        if isinstance(_raw_allowed, str):
            self._allowed_conversations: set = {c.strip() for c in _raw_allowed.split(",") if c.strip()}
        elif isinstance(_raw_allowed, list):
            self._allowed_conversations: set = {str(c).strip() for c in _raw_allowed if str(c).strip()}
        else:
            self._allowed_conversations: set = set()

        self._stream_client: Any = None
        self._stream_task: Optional[asyncio.Task] = None
        self._http_client: Optional["httpx.AsyncClient"] = None

        # Message deduplication: msg_id -> timestamp
        self._seen_messages: Dict[str, float] = {}
        # Map chat_id -> session_webhook for reply routing
        self._session_webhooks: Dict[str, str] = {}

        # Access token for DingTalk OpenAPI (emotion reactions, etc.)
        self._access_token: Optional[str] = None
        self._access_token_expires: float = 0.0
        self._access_token_lock: asyncio.Lock = asyncio.Lock()

    # -- Connection lifecycle -----------------------------------------------

    async def connect(self) -> bool:
        """Connect to DingTalk via Stream Mode."""
        if not DINGTALK_STREAM_AVAILABLE:
            logger.warning("[%s] dingtalk-stream not installed. Run: pip install dingtalk-stream", self.name)
            return False
        if not HTTPX_AVAILABLE:
            logger.warning("[%s] httpx not installed. Run: pip install httpx", self.name)
            return False
        if not self._client_id or not self._client_secret:
            logger.warning("[%s] DINGTALK_CLIENT_ID and DINGTALK_CLIENT_SECRET required", self.name)
            return False

        try:
            self._http_client = httpx.AsyncClient(timeout=30.0)

            credential = dingtalk_stream.Credential(self._client_id, self._client_secret)
            self._stream_client = dingtalk_stream.DingTalkStreamClient(credential)

            handler = _IncomingHandler(self)
            self._stream_client.register_callback_handler(
                dingtalk_stream.ChatbotMessage.TOPIC, handler
            )

            self._stream_task = asyncio.create_task(self._run_stream())
            self._mark_connected()
            logger.info("[%s] Connected via Stream Mode", self.name)
            return True
        except Exception as e:
            logger.error("[%s] Failed to connect: %s", self.name, e)
            return False

    async def _run_stream(self) -> None:
        """Run the blocking stream client with auto-reconnection."""
        backoff_idx = 0
        while self._running:
            try:
                logger.debug("[%s] Starting stream client...", self.name)
                await self._stream_client.start()
            except asyncio.CancelledError:
                return
            except Exception as e:
                if not self._running:
                    return
                logger.warning("[%s] Stream client error: %s", self.name, e)

            if not self._running:
                return

            delay = RECONNECT_BACKOFF[min(backoff_idx, len(RECONNECT_BACKOFF) - 1)]
            logger.info("[%s] Reconnecting in %ds...", self.name, delay)
            await asyncio.sleep(delay)
            backoff_idx += 1

    async def disconnect(self) -> None:
        """Disconnect from DingTalk."""
        self._running = False
        self._mark_disconnected()

        if self._stream_task:
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass
            self._stream_task = None

        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

        self._stream_client = None
        self._session_webhooks.clear()
        self._seen_messages.clear()
        logger.info("[%s] Disconnected", self.name)

    # -- Access token management -----------------------------------------------

    async def _get_access_token(self) -> Optional[str]:
        """Get a cached DingTalk OpenAPI access token, refreshing if expired."""
        now = time.time()
        if self._access_token and now < self._access_token_expires:
            return self._access_token

        async with self._access_token_lock:
            now = time.time()
            if self._access_token and now < self._access_token_expires:
                return self._access_token
            if not self._http_client:
                return None
            try:
                resp = await self._http_client.post(
                    "https://api.dingtalk.com/v1.0/oauth2/accessToken",
                    json={"appKey": self._client_id, "appSecret": self._client_secret},
                    timeout=10.0,
                )
                if resp.status_code >= 300:
                    logger.warning("[%s] Access token request failed: HTTP %d", self.name, resp.status_code)
                    return None
                data = resp.json()
                self._access_token = data.get("accessToken")
                expire_in = int(data.get("expireIn", 7200))
                self._access_token_expires = now + expire_in - 300
                return self._access_token
            except Exception as e:
                logger.warning("[%s] Access token request error: %s", self.name, e)
                return None

    # -- Emotion reaction API --------------------------------------------------

    async def _emotion_api(self, endpoint: str, msg_id: str, conv_id: str,
                           retry_delays: list) -> bool:
        """Call DingTalk emotion reply/recall API with retry."""
        token = await self._get_access_token()
        if not token or not self._http_client:
            return False

        headers = {"x-acs-dingtalk-access-token": token, "Content-Type": "application/json"}
        payload = {
            "robotCode": self._client_id,
            "openMsgId": msg_id,
            "openConversationId": conv_id,
            "emotionType": 2,
            "emotionName": _EMOTION_NAME,
            "textEmotion": {
                "emotionId": _EMOTION_ID,
                "emotionName": _EMOTION_NAME,
                "text": _EMOTION_NAME,
                "backgroundId": _EMOTION_BG_ID,
            },
        }

        for i, delay in enumerate(retry_delays):
            if delay > 0:
                await asyncio.sleep(delay)
            try:
                resp = await self._http_client.post(
                    f"https://api.dingtalk.com/v1.0/robot/emotion/{endpoint}",
                    json=payload, headers=headers, timeout=5.0,
                )
                if resp.status_code < 300:
                    return True
                body = resp.json() if "json" in (resp.headers.get("content-type") or "") else {}
                err_code = str(body.get("code", ""))
                if resp.status_code >= 500 or err_code == "system.err":
                    logger.debug("[%s] Emotion %s attempt %d/%d failed (HTTP %d), retrying",
                                 self.name, endpoint, i + 1, len(retry_delays), resp.status_code)
                    continue
                logger.debug("[%s] Emotion %s failed: HTTP %d code=%s",
                             self.name, endpoint, resp.status_code, err_code)
                return False
            except Exception as e:
                logger.debug("[%s] Emotion %s attempt %d error: %s", self.name, endpoint, i + 1, e)
        return False

    # -- Processing lifecycle hooks --------------------------------------------

    def _reactions_enabled(self) -> bool:
        """Check if emoji reactions are enabled."""
        extra = (self.config.extra or {}) if hasattr(self.config, "extra") else {}
        cfg_val = extra.get("reactions")
        if cfg_val is not None:
            return str(cfg_val).lower() not in ("false", "0", "no")
        return os.getenv("DINGTALK_REACTIONS", "true").lower() not in ("false", "0", "no")

    async def on_processing_start(self, event: MessageEvent) -> None:
        """Attach a thinking emoji when processing begins."""
        if not self._reactions_enabled():
            return
        raw = event.raw_message
        msg_id = getattr(raw, "message_id", None)
        conv_id = getattr(raw, "conversation_id", None)
        if msg_id and conv_id:
            await self._emotion_api("reply", msg_id, conv_id, _EMOTION_ATTACH_DELAYS)

    async def on_processing_complete(self, event: MessageEvent, outcome: ProcessingOutcome) -> None:
        """Remove the thinking emoji when processing finishes."""
        if not self._reactions_enabled():
            return
        raw = event.raw_message
        msg_id = getattr(raw, "message_id", None)
        conv_id = getattr(raw, "conversation_id", None)
        if msg_id and conv_id:
            await self._emotion_api("recall", msg_id, conv_id, _EMOTION_RECALL_DELAYS)

    # -- Inbound message processing -----------------------------------------

    async def _on_message(self, message: Any) -> None:
        """Process an incoming DingTalk chatbot message."""
        msg_id = getattr(message, "message_id", None) or uuid.uuid4().hex
        if self._is_duplicate(msg_id):
            logger.debug("[%s] Duplicate message %s, skipping", self.name, msg_id)
            return

        text = self._extract_text(message)
        if not text:
            logger.warning(
                "[%s] Skipping message with no extractable text (msgtype=%s msg_id=%s). "
                "Rich text / TextContent shapes differ from plain dict payloads.",
                self.name,
                getattr(message, "message_type", None),
                msg_id,
            )
            return

        # Chat context
        conversation_id = getattr(message, "conversation_id", "") or ""
        conversation_type = getattr(message, "conversation_type", "1")
        is_group = str(conversation_type) == "2"
        sender_id = getattr(message, "sender_id", "") or ""
        sender_nick = getattr(message, "sender_nick", "") or sender_id
        sender_staff_id = getattr(message, "sender_staff_id", "") or ""

        chat_id = conversation_id or sender_id
        chat_type = "group" if is_group else "dm"

        # Conversation allowlist filter: only filter group chats; DMs always pass
        if self._allowed_conversations and is_group and chat_id not in self._allowed_conversations:
            logger.debug("[%s] Message from unallowed group %s, skipping", self.name, chat_id[:20])
            return

        # Store session webhook for reply routing (validate origin to prevent SSRF)
        session_webhook = getattr(message, "session_webhook", None) or ""
        if session_webhook and chat_id and _session_webhook_is_allowed(session_webhook):
            if len(self._session_webhooks) >= _SESSION_WEBHOOKS_MAX:
                # Evict oldest entry to cap memory growth
                try:
                    self._session_webhooks.pop(next(iter(self._session_webhooks)))
                except StopIteration:
                    pass
            self._session_webhooks[chat_id] = session_webhook

        source = self.build_source(
            chat_id=chat_id,
            chat_name=getattr(message, "conversation_title", None),
            chat_type=chat_type,
            user_id=sender_id,
            user_name=sender_nick,
            user_id_alt=sender_staff_id if sender_staff_id else None,
        )

        # Parse timestamp
        create_at = getattr(message, "create_at", None)
        try:
            timestamp = datetime.fromtimestamp(int(create_at) / 1000, tz=timezone.utc) if create_at else datetime.now(tz=timezone.utc)
        except (ValueError, OSError, TypeError):
            timestamp = datetime.now(tz=timezone.utc)

        event = MessageEvent(
            text=text,
            message_type=MessageType.TEXT,
            source=source,
            message_id=msg_id,
            raw_message=message,
            timestamp=timestamp,
        )

        logger.info("[%s] inbound message: platform=dingtalk sender=%s chat=%s text=%s",
                      self.name, sender_nick, chat_id[:20] if chat_id else "?", text[:80])
        await self.handle_message(event)

    @staticmethod
    def _extract_text(message: "ChatbotMessage") -> str:
        """Extract plain text from a DingTalk chatbot message."""
        text = getattr(message, "text", None)
        content = ""

        if text is not None and text != "":
            if isinstance(text, dict):
                content = (text.get("content") or "").strip()
            elif hasattr(text, "content") and getattr(text, "content", None) is not None:
                # dingtalk-stream uses TextContent objects, not raw dicts
                content = str(text.content).strip()
            elif isinstance(text, str):
                content = text.strip()
            else:
                # Last resort (avoid str(TextContent) repr leaking into the agent)
                inner = getattr(text, "content", None)
                content = str(inner).strip() if inner is not None else ""

        if not content:
            # SDK stores rich messages on rich_text_content.rich_text_list (not ``rich_text``).
            rtc = getattr(message, "rich_text_content", None)
            rtl = getattr(rtc, "rich_text_list", None) if rtc is not None else None
            if rtl and isinstance(rtl, list):
                parts = [
                    item["text"]
                    for item in rtl
                    if isinstance(item, dict) and item.get("text")
                ]
                content = " ".join(parts).strip()

        if not content:
            # Legacy / tests: flat list on ``rich_text``
            rich_text = getattr(message, "rich_text", None)
            if rich_text and isinstance(rich_text, list):
                parts = [
                    item["text"]
                    for item in rich_text
                    if isinstance(item, dict) and item.get("text")
                ]
                content = " ".join(parts).strip()

        if not content and getattr(message, "message_type", None) == "picture":
            img = getattr(message, "image_content", None)
            code = getattr(img, "download_code", None) if img is not None else None
            if code:
                content = "[图片]"

        return content

    # -- Deduplication ------------------------------------------------------

    def _is_duplicate(self, msg_id: str) -> bool:
        """Check and record a message ID. Returns True if already seen."""
        now = time.time()
        if len(self._seen_messages) > DEDUP_MAX_SIZE:
            cutoff = now - DEDUP_WINDOW_SECONDS
            self._seen_messages = {k: v for k, v in self._seen_messages.items() if v > cutoff}

        if msg_id in self._seen_messages:
            return True
        self._seen_messages[msg_id] = now
        return False

    # -- Outbound messaging -------------------------------------------------

    _CHUNK_LIMIT = 3800  # DingTalk markdown card height is fixed; split long messages

    @staticmethod
    def _split_markdown_chunks(text: str, limit: int = 3800) -> list:
        """Split long markdown text into chunks that fit DingTalk's card height.

        Splits on line boundaries, preserving code fences across chunks.
        Mirrors openclaw's splitMarkdownChunks logic.
        """
        if not text or len(text) <= limit:
            return [text]

        chunks = []
        buf = ""
        in_code = False

        for line in text.split("\n"):
            fence_count = line.count("```")
            if len(buf) + len(line) + 1 > limit and buf:
                if in_code:
                    buf += "\n```"
                chunks.append(buf)
                buf = "```\n" if in_code else ""
            buf += ("\n" + line) if buf else line
            if fence_count % 2 == 1:
                in_code = not in_code

        if buf:
            chunks.append(buf)
        return chunks

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a markdown reply via DingTalk session webhook."""
        metadata = metadata or {}

        session_webhook = metadata.get("session_webhook") or self._session_webhooks.get(chat_id)
        if not session_webhook:
            return SendResult(success=False,
                              error="No session_webhook available. Reply must follow an incoming message.")

        if not self._http_client:
            return SendResult(success=False, error="HTTP client not initialized")

        chunks = self._split_markdown_chunks(content[:self.MAX_MESSAGE_LENGTH], self._CHUNK_LIMIT)
        last_result = None

        for i, chunk in enumerate(chunks):
            title = "Hermes" if len(chunks) == 1 else f"Hermes ({i + 1}/{len(chunks)})"
            payload = {
                "msgtype": "markdown",
                "markdown": {"title": title, "text": chunk},
            }

            try:
                resp = await self._http_client.post(session_webhook, json=payload, timeout=15.0)
                if resp.status_code < 300:
                    last_result = SendResult(success=True, message_id=uuid.uuid4().hex[:12])
                else:
                    body = resp.text
                    logger.warning("[%s] Send failed HTTP %d: %s", self.name, resp.status_code, body[:200])
                    return SendResult(success=False, error=f"HTTP {resp.status_code}: {body[:200]}")
            except httpx.TimeoutException:
                return SendResult(success=False, error="Timeout sending message to DingTalk")
            except Exception as e:
                logger.error("[%s] Send error: %s", self.name, e)
                return SendResult(success=False, error=str(e))

        return last_result or SendResult(success=True)

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """DingTalk does not support typing indicators."""
        pass

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Return basic info about a DingTalk conversation."""
        return {"name": chat_id, "type": "group" if "group" in chat_id.lower() else "dm"}


# ---------------------------------------------------------------------------
# Internal stream handler
# ---------------------------------------------------------------------------

class _IncomingHandler(ChatbotHandler if DINGTALK_STREAM_AVAILABLE else object):
    """dingtalk-stream ChatbotHandler that forwards messages to the adapter.

    ``CallbackHandler.raw_process`` passes the raw ``CallbackMessage`` to
    ``process``.  We convert it to ``ChatbotMessage`` here so the rest of
    the adapter works with a well-typed object.
    """

    def __init__(self, adapter: DingTalkAdapter):
        if DINGTALK_STREAM_AVAILABLE:
            super().__init__()
        self._adapter = adapter

    async def process(self, message):
        """Called by dingtalk-stream when a message arrives.

        ``message`` is a ``CallbackMessage`` (the SDK does NOT convert it
        to ``ChatbotMessage`` automatically).  The real payload lives in
        ``message.data`` — a dict with camelCase keys.
        """
        try:
            logger.info(
                "[DingTalk] Incoming callback: type=%s, has_data=%s",
                type(message).__name__,
                bool(getattr(message, "data", None)),
            )
            # Convert CallbackMessage → ChatbotMessage
            data = getattr(message, "data", None)
            if isinstance(data, dict) and data:
                logger.debug("[DingTalk] Raw data keys: %s", list(data.keys()))
                try:
                    chatbot_msg = ChatbotMessage.from_dict(data)
                except Exception:
                    logger.exception("[DingTalk] ChatbotMessage.from_dict failed")
                    chatbot_msg = message
            elif isinstance(message, ChatbotMessage):
                chatbot_msg = message
            else:
                logger.warning(
                    "[DingTalk] Unexpected message shape: %s (no usable data dict)",
                    type(message).__name__,
                )
                chatbot_msg = message

            await self._adapter._on_message(chatbot_msg)
        except Exception:
            logger.exception("[DingTalk] Error processing incoming message")
        return dingtalk_stream.AckMessage.STATUS_OK, "OK"

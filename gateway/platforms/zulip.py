"""Zulip gateway adapter.

Connects to a self-hosted or cloud Zulip instance via its REST API (v1)
and event queue (long-polling) for real-time events. No external Zulip
library required — uses aiohttp which is already a Hermes dependency.

Environment variables:
    ZULIP_SITE                  Server URL (e.g. https://zulip.example.com)
    ZULIP_EMAIL                 Bot email address
    ZULIP_API_KEY               Bot API key
    ZULIP_ALLOWED_USERS         Comma-separated user IDs
    ZULIP_HOME_STREAM           Stream ID for cron/notification delivery
    ZULIP_HOME_TOPIC            Topic within home stream (default: notifications)
    ZULIP_REQUIRE_MENTION       Require @mention in streams (default: true)
    ZULIP_FREE_RESPONSE_STREAMS Comma-separated stream IDs where bot responds
                                without @mention
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from gateway.config import Platform, PlatformConfig
from gateway.platforms.helpers import MessageDeduplicator
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)

logger = logging.getLogger(__name__)

# Zulip's documented per-message content limit is 10,000 characters.
MAX_MESSAGE_LENGTH = 10000

# Long-poll budget — a touch above Zulip's 90s server-side event timeout so
# a clean "no events" response arrives before our HTTP timeout fires.
_EVENT_POLL_TIMEOUT = 95.0

# Reconnect / retry parameters (exponential backoff with jitter).
_RECONNECT_BASE_DELAY = 2.0
_RECONNECT_MAX_DELAY = 60.0
_RECONNECT_JITTER = 0.2


class BadEventQueueError(Exception):
    """Raised when Zulip reports the event queue expired (BAD_EVENT_QUEUE_ID)."""


def check_zulip_requirements() -> bool:
    """Return True if the Zulip adapter can be used."""
    site = os.getenv("ZULIP_SITE", "")
    email = os.getenv("ZULIP_EMAIL", "")
    api_key = os.getenv("ZULIP_API_KEY", "")
    if not site:
        logger.debug("Zulip: ZULIP_SITE not set")
        return False
    if not email:
        logger.warning("Zulip: ZULIP_EMAIL not set")
        return False
    if not api_key:
        logger.warning("Zulip: ZULIP_API_KEY not set")
        return False
    try:
        import aiohttp  # noqa: F401
        return True
    except ImportError:
        logger.warning("Zulip: aiohttp not installed")
        return False


class ZulipAdapter(BasePlatformAdapter):
    """Gateway adapter for Zulip (self-hosted or cloud)."""

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.ZULIP)

        self._site: str = (
            config.extra.get("site", "")
            or os.getenv("ZULIP_SITE", "")
        ).rstrip("/")
        self._email: str = (
            config.extra.get("email", "")
            or os.getenv("ZULIP_EMAIL", "")
        )
        self._api_key: str = config.token or os.getenv("ZULIP_API_KEY", "")

        self._bot_user_id: int = 0
        self._bot_full_name: str = ""
        self._bot_email: str = ""
        # Compiled lazily once we know the bot full name.
        self._mention_re: Optional[re.Pattern[str]] = None

        # aiohttp session + long-poll state
        self._session: Any = None  # aiohttp.ClientSession
        self._queue_id: Optional[str] = None
        self._last_event_id: int = -1
        self._poll_task: Optional[asyncio.Task] = None
        self._closing = False

        # Dedup cache (prevent reprocessing)
        self._dedup = MessageDeduplicator()

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    async def _api_get(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        *,
        timeout: float = 30.0,
    ) -> Dict[str, Any]:
        """GET /api/v1/{path}."""
        import aiohttp
        url = f"{self._site}/api/v1/{path.lstrip('/')}"
        try:
            async with self._session.get(
                url,
                params=params,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                if resp.status == 429:
                    retry_after = _parse_retry_after(resp.headers.get("Retry-After"))
                    logger.warning(
                        "Zulip: rate limited on GET %s — sleeping %.1fs",
                        path, retry_after,
                    )
                    await asyncio.sleep(retry_after)
                    return {}
                if resp.status >= 400:
                    body = await resp.text()
                    logger.error(
                        "Zulip API GET %s → %s: %s", path, resp.status, body[:200]
                    )
                    return {}
                return await resp.json()
        except aiohttp.ClientError as exc:
            logger.error("Zulip API GET %s network error: %s", path, exc)
            return {}

    async def _api_post(
        self,
        path: str,
        data: Dict[str, Any],
        *,
        timeout: float = 30.0,
    ) -> Dict[str, Any]:
        """POST /api/v1/{path} with x-www-form-urlencoded body."""
        import aiohttp
        url = f"{self._site}/api/v1/{path.lstrip('/')}"
        try:
            async with self._session.post(
                url,
                data=data,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                if resp.status == 429:
                    retry_after = _parse_retry_after(resp.headers.get("Retry-After"))
                    logger.warning(
                        "Zulip: rate limited on POST %s — sleeping %.1fs",
                        path, retry_after,
                    )
                    await asyncio.sleep(retry_after)
                    return {}
                if resp.status >= 400:
                    body = await resp.text()
                    logger.error(
                        "Zulip API POST %s → %s: %s", path, resp.status, body[:200]
                    )
                    return {}
                return await resp.json()
        except aiohttp.ClientError as exc:
            logger.error("Zulip API POST %s network error: %s", path, exc)
            return {}

    # ------------------------------------------------------------------
    # Required overrides
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        """Authenticate, register an event queue, start the long-poll loop."""
        import aiohttp

        if not self._site or not self._email or not self._api_key:
            logger.error(
                "Zulip: ZULIP_SITE, ZULIP_EMAIL or ZULIP_API_KEY not configured"
            )
            return False

        self._session = aiohttp.ClientSession(
            auth=aiohttp.BasicAuth(self._email, self._api_key),
            timeout=aiohttp.ClientTimeout(total=30),
        )
        self._closing = False

        me = await self._api_get("users/me")
        if not me or me.get("result") != "success" or "user_id" not in me:
            logger.error(
                "Zulip: failed to authenticate — check ZULIP_SITE, ZULIP_EMAIL, "
                "and ZULIP_API_KEY",
            )
            await self._session.close()
            self._session = None
            return False

        self._bot_user_id = int(me["user_id"])
        self._bot_full_name = me.get("full_name", "") or ""
        self._bot_email = me.get("email", self._email) or self._email
        self._mention_re = _compile_mention_re(self._bot_full_name)

        logger.info(
            "Zulip: authenticated as %s (%s, id=%s) on %s",
            self._bot_full_name,
            self._bot_email,
            self._bot_user_id,
            self._site,
        )

        if not await self._register_queue():
            logger.error("Zulip: failed to register event queue")
            await self._session.close()
            self._session = None
            return False

        self._poll_task = asyncio.create_task(self._poll_loop())
        self._mark_connected()
        return True

    async def disconnect(self) -> None:
        """Stop the long-poll loop, delete the queue, close the session."""
        self._closing = True

        if self._poll_task and not self._poll_task.done():
            self._poll_task.cancel()
            try:
                await self._poll_task
            except (asyncio.CancelledError, Exception):
                pass

        # Best-effort queue cleanup so Zulip's server doesn't keep the queue
        # alive for its full idle timeout after a clean shutdown.
        if self._queue_id and self._session and not self._session.closed:
            try:
                import aiohttp
                url = f"{self._site}/api/v1/events"
                async with self._session.delete(
                    url,
                    params={"queue_id": self._queue_id},
                    timeout=aiohttp.ClientTimeout(total=10),
                ):
                    pass
            except Exception as exc:
                logger.debug("Zulip: queue delete failed: %s", exc)

        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None
        self._queue_id = None

        logger.info("Zulip: disconnected")

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a message (or multiple chunks) to a stream topic or DM."""
        if not content:
            return SendResult(success=True)

        formatted = self.format_message(content)
        chunks = self.truncate_message(formatted, MAX_MESSAGE_LENGTH)

        last_id: Optional[str] = None
        for chunk in chunks:
            payload = self._build_send_payload(chat_id, chunk)
            if payload is None:
                return SendResult(
                    success=False, error=f"Invalid Zulip chat_id: {chat_id}"
                )

            data = await self._api_post("messages", payload)
            if not data or data.get("result") != "success":
                err = (data or {}).get("msg", "Failed to send message")
                return SendResult(success=False, error=err)
            msg_id = data.get("id")
            if msg_id is not None:
                last_id = str(msg_id)

        return SendResult(success=True, message_id=last_id)

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Return a human-readable name and type for a Zulip chat_id."""
        if ":" in chat_id:
            stream_id_str, _, topic = chat_id.partition(":")
            try:
                stream_id = int(stream_id_str)
            except ValueError:
                return {"name": chat_id, "type": "channel"}
            data = await self._api_get(f"streams/{stream_id}")
            stream = (data or {}).get("stream", {}) if data else {}
            stream_name = stream.get("name", "")
            if stream_name:
                name = f"#{stream_name} > {topic}" if topic else f"#{stream_name}"
            else:
                name = chat_id
            return {"name": name, "type": "channel", "topic": topic}

        return {"name": f"DM {chat_id}", "type": "dm"}

    # ------------------------------------------------------------------
    # Optional overrides
    # ------------------------------------------------------------------

    def format_message(self, content: str) -> str:
        """Zulip Markdown is close to standard — pass through unchanged.

        Zulip renders image URLs as inline previews automatically, so we
        don't need to strip the image markdown the way Mattermost does.
        """
        return content

    # ------------------------------------------------------------------
    # Event queue + long-poll loop
    # ------------------------------------------------------------------

    async def _register_queue(self) -> bool:
        """Register an event queue and remember its id + last_event_id."""
        data = await self._api_post(
            "register",
            {
                "event_types": json.dumps(["message"]),
                "apply_markdown": "false",
                # Only deliver events for streams the bot is subscribed to
                # plus DMs addressed to it — matches the default behavior but
                # is made explicit so future Zulip default changes don't
                # silently broaden the firehose.
                "all_public_streams": "false",
            },
        )
        if not data or data.get("result") != "success":
            return False
        self._queue_id = data.get("queue_id")
        self._last_event_id = int(data.get("last_event_id", -1))
        logger.info(
            "Zulip: registered event queue %s (last_event_id=%s)",
            self._queue_id, self._last_event_id,
        )
        return bool(self._queue_id)

    async def _get_events(self) -> List[Dict[str, Any]]:
        """Long-poll /api/v1/events and return the next batch of events."""
        import aiohttp

        if not self._queue_id:
            raise BadEventQueueError("No queue registered")

        url = f"{self._site}/api/v1/events"
        params = {
            "queue_id": self._queue_id,
            "last_event_id": self._last_event_id,
        }
        try:
            async with self._session.get(
                url,
                params=params,
                timeout=aiohttp.ClientTimeout(total=_EVENT_POLL_TIMEOUT),
            ) as resp:
                if resp.status == 429:
                    retry_after = _parse_retry_after(resp.headers.get("Retry-After"))
                    logger.warning(
                        "Zulip: rate limited on /events — sleeping %.1fs",
                        retry_after,
                    )
                    await asyncio.sleep(retry_after)
                    return []
                # Guard against empty/non-JSON responses (network hiccups,
                # unexpected status codes). Zulip's long-poll can return
                # empty bodies on transient errors.
                raw = await resp.read()
                if not raw or not raw.strip():
                    if resp.status != 200:
                        logger.warning(
                            "Zulip /events returned %d with empty body", resp.status
                        )
                    return []
                try:
                    body = json.loads(raw)
                except (json.JSONDecodeError, ValueError):
                    logger.warning(
                        "Zulip /events returned non-JSON (status %d, %d bytes)",
                        resp.status,
                        len(raw),
                    )
                    return []
                if body.get("result") == "error":
                    if body.get("code") == "BAD_EVENT_QUEUE_ID":
                        raise BadEventQueueError(body.get("msg", ""))
                    logger.error(
                        "Zulip /events error: %s", body.get("msg", body)
                    )
                    return []
                return body.get("events", []) or []
        except asyncio.CancelledError:
            # Re-raise so the poll-loop's CancelledError handler runs.
            raise
        except asyncio.TimeoutError:
            # Normal idle long-poll timeout — just poll again.
            return []

    async def _poll_loop(self) -> None:
        """Long-poll the event queue, dispatch messages, backoff on errors."""
        delay = _RECONNECT_BASE_DELAY
        while not self._closing:
            try:
                events = await self._get_events()
            except asyncio.CancelledError:
                return
            except BadEventQueueError:
                logger.warning("Zulip: event queue expired — re-registering")
                if await self._register_queue():
                    delay = _RECONNECT_BASE_DELAY
                    continue
                logger.error("Zulip: queue re-register failed, retrying later")
            except Exception as exc:
                if self._closing:
                    return
                import aiohttp
                status = getattr(exc, "status", None)
                err_str = str(exc).lower()
                if status in (401, 403) or "401" in err_str or "403" in err_str:
                    logger.error(
                        "Zulip: authentication failed (%s) — stopping poll loop",
                        exc,
                    )
                    return
                if isinstance(exc, aiohttp.ClientError):
                    logger.warning(
                        "Zulip: network error %s — retrying in %.0fs",
                        exc, delay,
                    )
                else:
                    logger.exception("Zulip poll loop error")
            else:
                # Successful poll — dispatch events, reset backoff.
                delay = _RECONNECT_BASE_DELAY
                for event in events:
                    eid = event.get("id")
                    if isinstance(eid, int) and eid > self._last_event_id:
                        self._last_event_id = eid
                    if event.get("type") == "message":
                        try:
                            await self._handle_message(event.get("message", {}))
                        except Exception:
                            logger.exception(
                                "Zulip: error dispatching message event"
                            )
                continue

            if self._closing:
                return

            import random
            jitter = delay * _RECONNECT_JITTER * random.random()
            await asyncio.sleep(delay + jitter)
            delay = min(delay * 2, _RECONNECT_MAX_DELAY)

    # ------------------------------------------------------------------
    # Message handling
    # ------------------------------------------------------------------

    async def _handle_message(self, message: Dict[str, Any]) -> None:
        """Process a single Zulip message event."""
        if not message:
            return

        msg_id = str(message.get("id", ""))
        if not msg_id:
            return

        sender_id = message.get("sender_id")

        # Ignore own messages (prevents echo loops).
        if sender_id is not None and int(sender_id) == self._bot_user_id:
            return

        # Ignore other bots when we can identify them.
        if _looks_like_bot(message):
            logger.debug("Zulip: skipping message from another bot")
            return

        if self._dedup.is_duplicate(msg_id):
            return

        msg_type = message.get("type")
        message_text = message.get("content", "") or ""

        chat_id: str
        chat_type: str
        chat_name: Optional[str]
        chat_topic: Optional[str] = None

        if msg_type == "stream":
            stream_id = message.get("stream_id")
            if stream_id is None:
                return
            topic = message.get("subject", "") or ""
            stream_name = message.get("display_recipient") or ""
            if not isinstance(stream_name, str):
                stream_name = ""
            chat_id = f"{stream_id}:{topic}"
            chat_type = "channel"
            chat_name = stream_name or None
            chat_topic = topic or None

            require_mention = os.getenv(
                "ZULIP_REQUIRE_MENTION", "true"
            ).lower() not in ("false", "0", "no")

            free_streams_raw = os.getenv("ZULIP_FREE_RESPONSE_STREAMS", "")
            free_streams = {
                s.strip() for s in free_streams_raw.split(",") if s.strip()
            }
            is_free_stream = str(stream_id) in free_streams

            has_mention = self._has_mention(message_text)

            if require_mention and not is_free_stream and not has_mention:
                logger.debug(
                    "Zulip: skipping stream message without @mention "
                    "(stream_id=%s, topic=%r)",
                    stream_id, topic,
                )
                return

            if has_mention:
                message_text = self._strip_mention(message_text)

        elif msg_type in ("private", "direct"):
            recipients = message.get("display_recipient") or []
            if not isinstance(recipients, list):
                return
            user_ids = sorted(
                str(u["id"])
                for u in recipients
                if isinstance(u, dict) and "id" in u
            )
            if not user_ids:
                return
            chat_id = ",".join(user_ids)
            # 1:1 DM has 2 participants (sender + bot); anything larger is a
            # group DM. Zulip doesn't distinguish them at the API level, but
            # downstream session isolation benefits from the label.
            chat_type = "dm" if len(user_ids) <= 2 else "group"
            chat_name = None
        else:
            return

        sender_name = (
            message.get("sender_full_name")
            or message.get("sender_email")
            or (str(sender_id) if sender_id is not None else "")
        )

        source = self.build_source(
            chat_id=chat_id,
            chat_name=chat_name,
            chat_type=chat_type,
            user_id=str(sender_id) if sender_id is not None else None,
            user_name=sender_name or None,
            thread_id=None,
            chat_topic=chat_topic,
        )

        from gateway.platforms.base import resolve_channel_prompt
        _channel_prompt = resolve_channel_prompt(
            self.config.extra, chat_id, None,
        )

        msg_event = MessageEvent(
            text=message_text,
            message_type=MessageType.TEXT,
            source=source,
            raw_message=message,
            message_id=msg_id,
            channel_prompt=_channel_prompt,
        )

        await self.handle_message(msg_event)

    def _has_mention(self, text: str) -> bool:
        if not self._mention_re:
            return False
        return self._mention_re.search(text) is not None

    def _strip_mention(self, text: str) -> str:
        if not self._mention_re:
            return text
        return self._mention_re.sub("", text).strip()

    # ------------------------------------------------------------------
    # Send helpers
    # ------------------------------------------------------------------

    def _build_send_payload(
        self, chat_id: str, content: str
    ) -> Optional[Dict[str, Any]]:
        """Turn a Hermes chat_id + content into a Zulip /messages payload."""
        if ":" in chat_id:
            stream_part, _, topic = chat_id.partition(":")
            try:
                stream_id = int(stream_part)
            except ValueError:
                return None
            return {
                "type": "stream",
                "to": str(stream_id),
                "topic": topic,
                "content": content,
            }

        if not chat_id.strip():
            return None
        try:
            user_ids = [int(u.strip()) for u in chat_id.split(",") if u.strip()]
        except ValueError:
            return None
        if not user_ids:
            return None
        return {
            "type": "direct",
            "to": json.dumps(user_ids),
            "content": content,
        }


# ---------------------------------------------------------------------------
# Module-private helpers
# ---------------------------------------------------------------------------


def _compile_mention_re(full_name: str) -> Optional[re.Pattern[str]]:
    """Regex that matches Zulip mentions for *full_name*.

    Covers: @**Name**, @_**Name**_ (silent), @**Name|123** (disambiguated).
    """
    if not full_name:
        return None
    return re.compile(
        r"@_?\*\*" + re.escape(full_name) + r"(?:\|\d+)?\*\*_?"
    )


def _parse_retry_after(raw: Optional[str], default: float = 30.0) -> float:
    if not raw:
        return default
    try:
        return max(1.0, float(raw))
    except (TypeError, ValueError):
        return default


def _looks_like_bot(message: Dict[str, Any]) -> bool:
    """Best-effort check for "is this message from another bot?".

    Zulip message events don't consistently include an is_bot flag, so we
    check a few hints without being overly aggressive.
    """
    sender = message.get("sender")
    if isinstance(sender, dict) and sender.get("is_bot"):
        return True
    if message.get("sender_is_bot"):
        return True
    email = message.get("sender_email", "") or ""
    # Zulip's bot owner convention: bot emails contain "-bot@".
    if "-bot@" in email.lower():
        return True
    return False

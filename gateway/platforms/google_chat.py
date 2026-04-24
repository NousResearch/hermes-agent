"""Google Chat gateway adapter.

Connects to Google Chat via Cloud Pub/Sub for inbound messages and the
Chat REST API (v1) for outbound replies.  No additional HTTP server or
relay is needed — the adapter subscribes directly to a Pub/Sub
subscription where Google Chat publishes native events.

Architecture::

    Google Chat  ─→  Pub/Sub topic  ─→  THIS ADAPTER  ─→  Hermes
    Google Chat  ←─  Chat REST API  ←─  THIS ADAPTER  ←─  Hermes

Inbound:
    Google Chat is configured in *Cloud Pub/Sub connection mode*,
    publishing native Chat API events to a topic.  A background
    streaming-pull subscriber reads from the matching subscription
    and dispatches messages to ``self.handle_message(event)``.

    Three event formats are detected automatically:
    1. **Workspace Add-ons** — ``{chat: {messagePayload: {...}}}``
    2. **Native Chat API Pub/Sub** — ``{type: "MESSAGE", message: {...}}``
    3. **Relay / custom format** — flat ``{sender_email, text, ...}``

Outbound:
    When Hermes produces a response, the gateway calls ``send()``.
    This adapter posts the reply to the Google Chat REST API using
    the Chat app's service-account credentials.

Environment variables:
    GOOGLE_CHAT_GCP_PROJECT             GCP project ID for Pub/Sub
    GOOGLE_CHAT_PUBSUB_SUBSCRIPTION     Pub/Sub subscription name
    GOOGLE_CHAT_CREDENTIALS             Path to service-account JSON key
    GOOGLE_CHAT_HOME_CHANNEL            Space name for cron/notification delivery
    GOOGLE_CHAT_ALLOWED_USERS           Comma-separated allowed email addresses
    GOOGLE_CHAT_ALLOW_ALL_USERS         Set to "true" to allow all users

Requirements:
    pip install google-cloud-pubsub google-auth google-api-python-client
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import time
from typing import Any, Dict, List, Optional

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)

logger = logging.getLogger(__name__)

# Google Chat message-length limit (text body).
MAX_MESSAGE_LENGTH = 4096

# Pub/Sub flow control
_MAX_OUTSTANDING_MESSAGES = 10

# Retry parameters for outbound Chat API calls
_RETRY_MAX_ATTEMPTS = 3
_RETRY_BASE_DELAY = 1.0
_RETRY_MAX_DELAY = 8.0
_RETRY_JITTER = 0.3


def check_google_chat_requirements() -> bool:
    """Check if Google Chat platform dependencies are available."""
    try:
        from google.cloud import pubsub_v1  # noqa: F401
    except ImportError:
        logger.warning(
            "Google Chat: google-cloud-pubsub not installed. "
            "Run: pip install google-cloud-pubsub"
        )
        return False
    try:
        from google.oauth2 import service_account  # noqa: F401
        from googleapiclient.discovery import build  # noqa: F401
    except ImportError:
        logger.warning(
            "Google Chat: google-auth or google-api-python-client not installed. "
            "Run: pip install google-auth google-api-python-client"
        )
        return False

    # Require at least the GCP project to be set
    gcp_project = os.getenv("GOOGLE_CHAT_GCP_PROJECT", "")
    if not gcp_project:
        logger.debug("Google Chat: GOOGLE_CHAT_GCP_PROJECT not set")
        return False

    return True


class GoogleChatAdapter(BasePlatformAdapter):
    """Gateway adapter for Google Chat via Cloud Pub/Sub.

    Config keys (read from ``PlatformConfig.extra``):
        gcp_project         GCP project ID (required for Pub/Sub)
        pubsub_subscription Pub/Sub subscription name for inbound messages
        chat_credentials    Path to Chat app service-account JSON key
    """

    def __init__(self, config: PlatformConfig) -> None:
        super().__init__(config, Platform.GOOGLE_CHAT)

        extra = config.extra or {}

        # GCP project
        self._gcp_project: str = extra.get(
            "gcp_project",
            os.getenv("GOOGLE_CHAT_GCP_PROJECT", ""),
        )
        self._subscription: str = extra.get(
            "pubsub_subscription",
            os.getenv(
                "GOOGLE_CHAT_PUBSUB_SUBSCRIPTION",
                "hermes-chat-inbound-sub",
            ),
        )

        # Chat API credentials
        self._credentials_path: str | None = extra.get(
            "chat_credentials",
            os.getenv("GOOGLE_CHAT_CREDENTIALS"),
        )

        # Runtime state
        self._subscriber_client: Any = None
        self._streaming_pull_future: Any = None
        self._chat_service: Any = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._connected: bool = False

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        """Start pulling messages from Pub/Sub."""
        from google.cloud import pubsub_v1

        try:
            self._loop = asyncio.get_running_loop()

            # Build SA credentials once — shared by Chat API and Pub/Sub
            creds = self._build_credentials()

            # Build the Chat API client for outbound messages
            self._chat_service = self._build_chat_service(creds)

            # Start the Pub/Sub subscriber using the same SA credentials
            self._subscriber_client = pubsub_v1.SubscriberClient(
                credentials=creds,
            )
            subscription_path = (
                f"projects/{self._gcp_project}"
                f"/subscriptions/{self._subscription}"
            )

            logger.info(
                "Google Chat: subscribing to %s", subscription_path
            )

            # The streaming pull callback runs in a thread-pool managed by
            # the google-cloud-pubsub library.  We bridge back to asyncio
            # via ``loop.call_soon_threadsafe``.
            self._streaming_pull_future = self._subscriber_client.subscribe(
                subscription_path,
                callback=self._on_pubsub_message,
                flow_control=pubsub_v1.types.FlowControl(
                    max_messages=_MAX_OUTSTANDING_MESSAGES,
                ),
            )

            self._connected = True
            self._mark_connected()
            logger.info("Google Chat: connected and listening")
            return True

        except Exception:
            logger.error(
                "Google Chat: failed to connect", exc_info=True
            )
            return False

    async def disconnect(self) -> None:
        """Stop the Pub/Sub subscriber."""
        self._connected = False

        if self._streaming_pull_future is not None:
            self._streaming_pull_future.cancel()
            self._streaming_pull_future = None

        if self._subscriber_client is not None:
            self._subscriber_client.close()
            self._subscriber_client = None

        if self._chat_service is not None:
            self._chat_service = None

        logger.info("Google Chat: disconnected")

    # ------------------------------------------------------------------
    # Inbound: Pub/Sub → Hermes
    # ------------------------------------------------------------------

    def _on_pubsub_message(self, message: Any) -> None:
        """Pub/Sub callback — runs in a background thread.

        Handles three event formats:
        1. **Workspace Add-ons** (current Chat App Pub/Sub format):
           ``{commonEventObject: {...}, chat: {user: {...}, messagePayload: {...}}}``
        2. **Native Chat API Pub/Sub** (alternate format):
           ``{type: "MESSAGE", message: {sender: {...}, text: "..."}, space: {...}}``
        3. **Relay format** (from an optional Cloud Run relay):
           ``{event_type: "MESSAGE", sender_email: "...", text: "..."}``

        Deserializes the message, converts to ``MessageEvent``, and
        schedules ``handle_message`` on the asyncio event loop.
        """
        try:
            data = json.loads(message.data.decode("utf-8"))
            logger.debug(
                "Google Chat: received Pub/Sub message keys: %s",
                list(data.keys()),
            )

            parsed = self._parse_event(data)
            if parsed is None:
                message.ack()
                return

            text, sender_email, sender_name, space_name, space_type, thread_name, message_name = parsed

            if not text or not text.strip():
                message.ack()
                return

            # Determine chat_type from space type
            chat_type = (
                "dm"
                if space_type in ("DM", "DIRECT_MESSAGE")
                else "group"
            )

            logger.info(
                "Google Chat: message from %s in %s (%s): %s",
                sender_name,
                space_name,
                chat_type,
                text[:100],
            )

            source = self.build_source(
                chat_id=space_name,
                user_id=sender_email,
                user_name=sender_name,
                chat_type=chat_type,
                thread_id=thread_name if thread_name else None,
            )

            event = MessageEvent(
                text=text,
                message_type=MessageType.TEXT,
                source=source,
                message_id=message_name,
                raw_message=data,
            )

            # Bridge to the asyncio event loop
            if self._loop and self._loop.is_running():
                self._loop.call_soon_threadsafe(
                    asyncio.ensure_future,
                    self.handle_message(event),
                )

            message.ack()

        except Exception:
            logger.error(
                "Google Chat: error processing Pub/Sub message",
                exc_info=True,
            )
            # Nack so the message is retried
            message.nack()

    @staticmethod
    def _parse_event(
        data: Dict[str, Any],
    ) -> Optional[tuple[str, str, str, str, str, str, str]]:
        """Parse a Chat API event into a normalized tuple.

        Returns ``(text, sender_email, sender_name, space_name,
        space_type, thread_name, message_name)`` or ``None`` if the
        event should be silently dropped (e.g., non-MESSAGE events,
        unrecognized formats).
        """
        # --- Format 1: Workspace Add-ons ---
        if "chat" in data and "messagePayload" in data.get("chat", {}):
            chat = data["chat"]
            user = chat.get("user", {})
            payload = chat["messagePayload"]
            space = payload.get("space", {})
            chat_msg = payload.get("message", {})

            sender_email = user.get("email", "unknown")
            sender_name = user.get("displayName", sender_email)
            space_name = space.get("name", "")
            space_type = space.get(
                "spaceType", space.get("type", "SPACE")
            )
            thread_name = (
                chat_msg.get("thread", {}).get("name", "")
            )
            text = chat_msg.get(
                "argumentText", chat_msg.get("text", "")
            )
            message_name = chat_msg.get("name", "")
            return (
                text,
                sender_email,
                sender_name,
                space_name,
                space_type,
                thread_name,
                message_name,
            )

        # --- Format 2: Native Chat API Pub/Sub ---
        if "message" in data and isinstance(data["message"], dict):
            event_type = data.get("type", "UNKNOWN")
            if event_type != "MESSAGE":
                logger.info(
                    "Google Chat: ignoring event type: %s", event_type
                )
                return None

            chat_msg = data["message"]
            sender = chat_msg.get("sender", {})
            space = data.get("space", {})

            sender_email = sender.get("email", "unknown")
            sender_name = sender.get("displayName", sender_email)
            space_name = space.get("name", "")
            space_type = space.get(
                "spaceType", space.get("type", "SPACE")
            )
            thread_name = (
                chat_msg.get("thread", {}).get("name", "")
            )
            text = chat_msg.get(
                "argumentText", chat_msg.get("text", "")
            )
            message_name = chat_msg.get("name", "")
            return (
                text,
                sender_email,
                sender_name,
                space_name,
                space_type,
                thread_name,
                message_name,
            )

        # --- Format 3: Relay / flat format ---
        if "event_type" in data or "sender_email" in data:
            event_type = data.get("event_type", "MESSAGE")
            if event_type != "MESSAGE":
                logger.info(
                    "Google Chat: ignoring relay event type: %s",
                    event_type,
                )
                return None

            sender_email = data.get("sender_email", "unknown")
            sender_name = data.get(
                "sender_display_name", sender_email
            )
            space_name = data.get("space_name", "")
            space_type = "SPACE"
            thread_name = data.get("thread_name", "")
            text = data.get("text", "")
            message_name = data.get("message_name", "")
            return (
                text,
                sender_email,
                sender_name,
                space_name,
                space_type,
                thread_name,
                message_name,
            )

        logger.warning(
            "Google Chat: unrecognized event format, keys: %s",
            list(data.keys()),
        )
        return None

    # ------------------------------------------------------------------
    # Outbound: Hermes → Google Chat
    # ------------------------------------------------------------------

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SendResult:
        """Send a text message to a Google Chat space.

        Long messages are automatically chunked to respect the
        4,096-character limit.  Outbound API calls use exponential
        backoff on transient failures.
        """
        if not content:
            return SendResult(success=True)

        if not self._chat_service:
            return SendResult(
                success=False, error="Chat API client not initialized"
            )

        formatted = self.format_message(content)
        chunks = self.truncate_message(formatted, MAX_MESSAGE_LENGTH)

        # Thread replies: use thread_id from metadata
        thread_name = None
        if metadata and metadata.get("thread_id"):
            thread_name = metadata["thread_id"]

        last_msg_name = None
        for chunk in chunks:
            result = await self._send_single_message(
                chat_id, chunk, thread_name
            )
            if not result.success:
                return result
            last_msg_name = result.message_id

        return SendResult(success=True, message_id=last_msg_name)

    async def _send_single_message(
        self,
        chat_id: str,
        text: str,
        thread_name: str | None = None,
    ) -> SendResult:
        """Send a single message chunk with retry/backoff."""
        delay = _RETRY_BASE_DELAY
        last_error: str = ""

        for attempt in range(_RETRY_MAX_ATTEMPTS):
            try:
                body: dict[str, Any] = {"text": text}

                create_kwargs: dict[str, Any] = {
                    "parent": chat_id,
                    "body": body,
                }

                if thread_name:
                    body["thread"] = {"name": thread_name}
                    create_kwargs["messageReplyOption"] = (
                        "REPLY_MESSAGE_FALLBACK_TO_NEW_THREAD"
                    )

                result = await asyncio.to_thread(
                    self._chat_service.spaces()
                    .messages()
                    .create(**create_kwargs)
                    .execute
                )

                msg_name = result.get("name", "")
                logger.debug("Google Chat: sent message %s", msg_name)
                return SendResult(success=True, message_id=msg_name)

            except Exception as exc:
                last_error = str(exc)
                # Check for retryable status codes
                retryable = _is_retryable_error(exc)
                if not retryable or attempt >= _RETRY_MAX_ATTEMPTS - 1:
                    logger.error(
                        "Google Chat: failed to send message to %s: %s",
                        chat_id,
                        exc,
                        exc_info=True,
                    )
                    return SendResult(
                        success=False,
                        error=last_error,
                        retryable=retryable,
                    )

                jitter = delay * _RETRY_JITTER * random.random()
                logger.warning(
                    "Google Chat: send attempt %d/%d failed (%s), "
                    "retrying in %.1fs",
                    attempt + 1,
                    _RETRY_MAX_ATTEMPTS,
                    exc,
                    delay + jitter,
                )
                await asyncio.sleep(delay + jitter)
                delay = min(delay * 2, _RETRY_MAX_DELAY)

        return SendResult(
            success=False, error=last_error, retryable=True
        )

    async def send_typing(
        self,
        chat_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Send typing indicator.

        Google Chat API does not natively support typing indicators
        for Chat apps, so this is a no-op.
        """
        pass

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: str = "",
        reply_to: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SendResult:
        """Send an image to a Google Chat space via an image card."""
        if not self._chat_service:
            return SendResult(
                success=False, error="Chat API client not initialized"
            )

        try:
            body: dict[str, Any] = {
                "cards": [
                    {
                        "sections": [
                            {
                                "widgets": [
                                    {
                                        "image": {
                                            "imageUrl": image_url,
                                        }
                                    }
                                ]
                            }
                        ]
                    }
                ]
            }
            if caption:
                body["text"] = caption

            result = await asyncio.to_thread(
                self._chat_service.spaces()
                .messages()
                .create(parent=chat_id, body=body)
                .execute
            )

            return SendResult(
                success=True,
                message_id=result.get("name", ""),
            )

        except Exception as exc:
            logger.error(
                "Google Chat: failed to send image to %s: %s",
                chat_id,
                exc,
            )
            return SendResult(
                success=False, error=str(exc), retryable=True
            )

    async def get_chat_info(self, chat_id: str) -> dict:
        """Return basic info about a Chat space."""
        return {
            "name": chat_id,
            "type": "space",
            "chat_id": chat_id,
        }

    # ------------------------------------------------------------------
    # Chat API client
    # ------------------------------------------------------------------

    def _build_credentials(self) -> Any:
        """Build Google credentials from SA key or ADC fallback.

        Prefers the explicit service-account JSON key at
        ``GOOGLE_CHAT_CREDENTIALS``.  Falls back to Application Default
        Credentials only when no key path is configured.
        """
        scopes = [
            "https://www.googleapis.com/auth/chat.bot",
            "https://www.googleapis.com/auth/pubsub",
        ]

        if self._credentials_path and os.path.isfile(
            self._credentials_path
        ):
            from google.oauth2 import service_account

            logger.info(
                "Google Chat: using service-account key %s",
                self._credentials_path,
            )
            return service_account.Credentials.from_service_account_file(
                self._credentials_path, scopes=scopes
            )

        # Fall back to ADC (works on GCE, Cloud Run, or with
        # ``gcloud auth application-default login``)
        import google.auth

        logger.warning(
            "Google Chat: no GOOGLE_CHAT_CREDENTIALS set — "
            "falling back to ADC (may require periodic reauth)"
        )
        credentials, _ = google.auth.default(scopes=scopes)
        return credentials

    def _build_chat_service(self, credentials: Any = None) -> Any:
        """Build the Google Chat API client.

        Args:
            credentials: Pre-built credentials.  If ``None``, calls
                         ``_build_credentials()`` internally.
        """
        from googleapiclient.discovery import build

        if credentials is None:
            credentials = self._build_credentials()

        return build(
            "chat",
            "v1",
            credentials=credentials,
            cache_discovery=False,
        )


def _is_retryable_error(exc: Exception) -> bool:
    """Determine if an API error is transient and should be retried."""
    exc_str = str(exc).lower()
    # googleapiclient.errors.HttpError exposes resp.status
    if hasattr(exc, "resp") and hasattr(exc.resp, "status"):
        status = int(exc.resp.status)
        return status in (429, 500, 502, 503, 504)
    # Fallback heuristics
    if "429" in exc_str or "rate limit" in exc_str:
        return True
    if any(
        code in exc_str for code in ("500", "502", "503", "504")
    ):
        return True
    if "timeout" in exc_str or "connection" in exc_str:
        return True
    return False

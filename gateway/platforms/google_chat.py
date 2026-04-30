"""
Google Chat platform adapter.

Uses Google Cloud Pub/Sub (pull subscription) for inbound events and the
Google Chat REST API for outbound messages. Pattern parallels Slack Socket
Mode and Telegram long-polling: no public endpoint required.

Concurrency model
-----------------
The Pub/Sub SubscriberClient invokes its message callback in a background
thread (managed by the client's internal executor). The adapter's
``handle_message`` coroutine must run on the asyncio event loop, so the
callback uses ``asyncio.run_coroutine_threadsafe`` with
``add_done_callback`` (never ``.result()`` — that would block the callback
thread and saturate the Pub/Sub executor under load).

All outbound Chat REST calls go through ``asyncio.to_thread`` because the
googleapiclient is synchronous. This keeps the event loop responsive.

Pub/Sub delivery diagram::

    Pub/Sub stream   ->  callback thread        ->  asyncio loop
    (streaming_pull)     (_on_pubsub_message)       (handle_message)
         |                       |                        |
         |   at-least-once       |  parse + dedup         |  agent work
         |   delivery            |  _submit_on_loop       |  send() response
         |                       |  message.ack()         |
         v                       v                        v

Event type routing
------------------
Inbound envelope carries ``type`` in [MESSAGE, ADDED_TO_SPACE, REMOVED_FROM_SPACE,
CARD_CLICKED]. Only MESSAGE dispatches to the agent. ADDED_TO_SPACE caches the
bot's resource name (belt-and-suspenders on top of eager resolution in connect()).
CARD_CLICKED is ACK'd only in v1 (follow-up PR implements interactivity).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
from pathlib import Path as _Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import httplib2
    from google.cloud import pubsub_v1
    from google.api_core import exceptions as gax_exceptions
    from google.oauth2 import service_account
    from google_auth_httplib2 import AuthorizedHttp
    from googleapiclient.discovery import build as build_service
    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaFileUpload

    GOOGLE_CHAT_AVAILABLE = True
except ImportError:
    GOOGLE_CHAT_AVAILABLE = False
    httplib2 = None  # type: ignore
    pubsub_v1 = None  # type: ignore
    gax_exceptions = None  # type: ignore
    service_account = None  # type: ignore
    AuthorizedHttp = None  # type: ignore
    build_service = None  # type: ignore
    HttpError = Exception  # type: ignore
    MediaFileUpload = None  # type: ignore

import sys
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from gateway.config import Platform, PlatformConfig
from gateway.platforms.helpers import MessageDeduplicator
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    ProcessingOutcome,
    SendResult,
    cache_audio_from_bytes,
    cache_document_from_bytes,
    cache_image_from_bytes,
    cache_video_from_bytes,
)


logger = logging.getLogger(__name__)


# Regex validating Pub/Sub subscription path format.
_SUBSCRIPTION_PATH_RE = re.compile(
    r"^projects/(?P<project>[^/]+)/subscriptions/(?P<sub>[^/]+)$"
)

# SA scopes — chat.bot is sufficient for the bot's own messaging operations
# (messages.create / patch / delete, spaces metadata, memberships,
# media.download for inbound user attachments). The bot CANNOT call
# media.upload — Google requires user OAuth for that endpoint, no scope
# adjustment changes it.
#
# Native attachment delivery (bot → user) is handled via a separate user-
# OAuth flow in ``google_chat_user_oauth.py``: the user grants the bot
# the chat.messages.create scope ONCE via an in-chat consent flow; the
# bot then calls media.upload on the user's behalf when sending files.
# See https://developers.google.com/chat/api/guides/auth/users
_CHAT_SCOPES = [
    "https://www.googleapis.com/auth/chat.bot",
    "https://www.googleapis.com/auth/pubsub",
]

# Google Chat text-message size limit is 4096; leave margin.
_MAX_TEXT_LENGTH = 4000

# Per-space rate-limit hit counter threshold; warn if exceeded.
_RATE_LIMIT_WARN_THRESHOLD = 5

# Sentinel kept in ``_typing_messages`` after ``send()`` patches the typing
# marker into the agent's real response. Two purposes:
#   * ``send_typing`` checks for any value before posting — sentinel keeps
#     ``_keep_typing`` (running on the base-class timer) from creating a
#     fresh "Hermes is thinking…" card during the small window between
#     ``send()`` finishing and the base-class cancelling its typing_task.
#   * ``stop_typing`` checks for the sentinel and skips the API delete —
#     otherwise the safety-net cleanup at base.py:_process_message_background
#     would delete the response we just patched and leave a tombstone.
_TYPING_CONSUMED_SENTINEL = "<consumed>"


def check_google_chat_requirements() -> bool:
    """Check if Google Chat optional dependencies are installed."""
    return GOOGLE_CHAT_AVAILABLE


# Hostnames we trust to host Google Chat attachment download URIs. Anything
# else gets rejected by _is_google_owned_host to block SSRF scenarios where
# a crafted event points downloadUri at a non-Google endpoint (e.g. the
# GCE/GKE metadata service at 169.254.169.254) and the bot's Service Account
# bearer token would be attached to the outbound request.
_TRUSTED_ATTACHMENT_HOSTS = (
    "googleapis.com",
    "chat.google.com",
    "drive.google.com",
    "docs.google.com",
    "lh3.googleusercontent.com",
    "lh4.googleusercontent.com",
    "lh5.googleusercontent.com",
    "lh6.googleusercontent.com",
)


def _is_google_owned_host(url: str) -> bool:
    """Return True iff *url* is https and targets a Google-owned domain."""
    try:
        from urllib.parse import urlparse

        parsed = urlparse(url)
    except Exception:
        return False
    if parsed.scheme != "https":
        return False
    host = (parsed.hostname or "").lower()
    if not host:
        return False
    return any(host == h or host.endswith("." + h) for h in _TRUSTED_ATTACHMENT_HOSTS)


def _redact_sensitive(text: str) -> str:
    """Sanitize subscription paths and email-like tokens from an error string.

    Covers project IDs leaking via Pub/Sub exception messages, plus SA-ish
    email addresses. agent/redact.py handles log-level redaction elsewhere;
    this helper is for user-facing error messages.
    """
    if not text:
        return text
    text = re.sub(
        r"projects/[^/\s]+/subscriptions/[^/\s]+",
        "projects/<redacted>/subscriptions/<redacted>",
        text,
    )
    text = re.sub(
        r"projects/[^/\s]+/topics/[^/\s]+",
        "projects/<redacted>/topics/<redacted>",
        text,
    )
    text = re.sub(
        r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.iam\.gserviceaccount\.com",
        "<sa>@<project>.iam.gserviceaccount.com",
        text,
    )
    return text


def _mime_for_message_type(mime: str) -> MessageType:
    """Map a MIME string to a hermes MessageType.

    Anything not image/audio/video falls through to DOCUMENT so the agent
    still receives the file.
    """
    if not mime:
        return MessageType.DOCUMENT
    if mime.startswith("image/"):
        return MessageType.PHOTO
    if mime.startswith("audio/"):
        return MessageType.AUDIO
    if mime.startswith("video/"):
        return MessageType.VIDEO
    return MessageType.DOCUMENT


class GoogleChatAdapter(BasePlatformAdapter):
    """
    Google Chat bot adapter using Pub/Sub pull + Chat REST API.

    Required environment (see gateway/config.py Google Chat block):
      GOOGLE_CHAT_PROJECT_ID           (or GOOGLE_CLOUD_PROJECT fallback)
      GOOGLE_CHAT_SUBSCRIPTION_NAME    (or GOOGLE_CHAT_SUBSCRIPTION fallback)
      GOOGLE_CHAT_SERVICE_ACCOUNT_JSON (or GOOGLE_APPLICATION_CREDENTIALS)

    Optional:
      GOOGLE_CHAT_ALLOWED_USERS, GOOGLE_CHAT_ALLOW_ALL_USERS
      GOOGLE_CHAT_HOME_CHANNEL
      GOOGLE_CHAT_MAX_MESSAGES (FlowControl, default 1)
      GOOGLE_CHAT_MAX_BYTES    (FlowControl, default 16_777_216 = 16 MiB)
    """

    MAX_MESSAGE_LENGTH = _MAX_TEXT_LENGTH
    # Pub/Sub supervisor configuration.
    _MAX_RECONNECT_ATTEMPTS = 10
    _RECONNECT_BASE_DELAY = 2.0
    _RECONNECT_MAX_DELAY = 120.0

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.GOOGLE_CHAT)
        self._subscriber: Optional[Any] = None
        self._chat_api: Optional[Any] = None
        # User-authed Chat API client built lazily from the OAuth refresh
        # token persisted by ``google_chat_user_oauth.py``. Required for
        # native ``media.upload`` (bot identity is rejected by that
        # endpoint). ``None`` until the user runs ``/setup-files`` once.
        self._user_chat_api: Optional[Any] = None
        self._user_credentials: Optional[Any] = None
        self._credentials: Optional[Any] = None
        self._project_id: Optional[str] = None
        self._subscription_path: Optional[str] = None
        self._streaming_pull_future: Optional[Any] = None
        self._supervisor_task: Optional[asyncio.Task] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._bot_user_id: Optional[str] = None  # users/{id}
        self._dedup = MessageDeduplicator()
        self._typing_messages: Dict[str, str] = {}
        self._shutting_down = False
        self._rate_limit_hits: Dict[str, int] = {}
        # Last-seen inbound thread name per chat_id (space). Google Chat
        # DMs create a NEW thread per top-level user message but the user
        # views them as one logical conversation. We:
        #   (a) drop thread_id from the source for DMs (so session_key
        #       stays stable across top-level messages — see
        #       gateway/session.py:build_session_key).
        #   (b) cache the most recent inbound thread name here so outbound
        #       replies still land in the right visual thread without
        #       re-coupling sessions to threads.
        self._last_inbound_thread: Dict[str, str] = {}
        # FlowControl knobs (env-configurable).
        self._max_messages = int(os.getenv("GOOGLE_CHAT_MAX_MESSAGES", "1"))
        self._max_bytes = int(os.getenv("GOOGLE_CHAT_MAX_BYTES", str(16 * 1024 * 1024)))

    # ------------------------------------------------------------------
    # Configuration loading and validation
    # ------------------------------------------------------------------
    def _load_sa_credentials(self) -> Any:
        """Load Service Account credentials from env or config.extra.

        Priority: explicit path in ``extra['service_account_json']`` ->
        ``GOOGLE_APPLICATION_CREDENTIALS`` env var. google-auth will also
        pick up GOOGLE_APPLICATION_CREDENTIALS automatically if we call
        ``google.auth.default()`` but the explicit path helps with
        deterministic error messages.
        """
        sa_path = (
            self.config.extra.get("service_account_json")
            or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        )
        if not sa_path:
            raise ValueError(
                "No Service Account credentials configured. Set "
                "GOOGLE_CHAT_SERVICE_ACCOUNT_JSON or GOOGLE_APPLICATION_CREDENTIALS."
            )
        # Inline JSON (rare, but supported).
        if sa_path.lstrip().startswith("{"):
            try:
                info = json.loads(sa_path)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Inline SA JSON is not valid JSON: {exc}") from exc
            return service_account.Credentials.from_service_account_info(
                info, scopes=_CHAT_SCOPES
            )
        if not os.path.exists(sa_path):
            raise FileNotFoundError(
                f"Service Account JSON file not found at configured path."
            )
        # Validate file parses before handing to google-auth for nicer error.
        try:
            with open(sa_path, "r", encoding="utf-8") as fh:
                info = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Service Account JSON file is not valid JSON: {exc}"
            ) from exc
        return service_account.Credentials.from_service_account_info(
            info, scopes=_CHAT_SCOPES
        )

    def _validate_config(self) -> Tuple[str, str]:
        """Return (project_id, subscription_path) after validation.

        Raises ValueError with a sanitized message on any config problem.
        """
        project_id = self.config.extra.get("project_id")
        subscription = self.config.extra.get("subscription_name")
        if not project_id:
            raise ValueError(
                "GOOGLE_CHAT_PROJECT_ID (or GOOGLE_CLOUD_PROJECT) is not set."
            )
        if not subscription:
            raise ValueError(
                "GOOGLE_CHAT_SUBSCRIPTION_NAME (or GOOGLE_CHAT_SUBSCRIPTION) is not set."
            )
        match = _SUBSCRIPTION_PATH_RE.match(subscription)
        if not match:
            raise ValueError(
                "GOOGLE_CHAT_SUBSCRIPTION_NAME must match "
                "'projects/<project>/subscriptions/<sub>'."
            )
        if match.group("project") != project_id:
            raise ValueError(
                "project_id in GOOGLE_CHAT_PROJECT_ID does not match the "
                "project embedded in GOOGLE_CHAT_SUBSCRIPTION_NAME."
            )
        return project_id, subscription

    # ------------------------------------------------------------------
    # Loop bridge helpers (thread -> asyncio loop)
    # ------------------------------------------------------------------
    @staticmethod
    def _log_background_failure(future: Any) -> None:
        try:
            future.result()
        except Exception:
            logger.exception("[GoogleChat] Background inbound processing failed")

    @staticmethod
    def _loop_accepts_callbacks(loop: Optional[asyncio.AbstractEventLoop]) -> bool:
        return loop is not None and not bool(getattr(loop, "is_closed", lambda: False)())

    def _submit_on_loop(self, coro: Any) -> None:
        """Schedule a coroutine on the adapter loop from a Pub/Sub callback thread."""
        loop = self._loop
        if not self._loop_accepts_callbacks(loop):
            # Loop already closed (shutdown race). Safe to drop; Pub/Sub will
            # redeliver on next reconnect.
            logger.warning("[GoogleChat] Loop not accepting callbacks; dropping event")
            return
        try:
            future = asyncio.run_coroutine_threadsafe(coro, loop)
        except RuntimeError:
            logger.warning("[GoogleChat] Loop closed between check and submit")
            return
        future.add_done_callback(self._log_background_failure)

    # ------------------------------------------------------------------
    # Bot identity resolution
    # ------------------------------------------------------------------
    def _bot_id_cache_path(self) -> _Path:
        """Location where the resolved bot user_id is cached across restarts."""
        base = os.getenv("HERMES_HOME", str(_Path.home() / ".hermes"))
        return _Path(base) / "google_chat_bot_id.json"

    def _load_cached_bot_id(self) -> Optional[str]:
        path = self._bot_id_cache_path()
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data.get("bot_user_id") or None
        except (OSError, json.JSONDecodeError):
            return None

    def _save_cached_bot_id(self, bot_user_id: str) -> None:
        try:
            path = self._bot_id_cache_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps({"bot_user_id": bot_user_id}),
                encoding="utf-8",
            )
        except OSError:
            logger.debug("[GoogleChat] Could not persist bot_user_id cache", exc_info=True)

    async def _resolve_bot_user_id(self) -> Optional[str]:
        """Resolve ``users/{id}`` via Chat API members.list on a known space.

        Tries the home channel first, then any space from the allowlist.
        If no space is known, returns None and self-filter falls back to
        filtering ``sender.type == 'BOT'`` (which is still safe but less
        precise — own messages and other bots look alike).
        """
        candidate_spaces: List[str] = []
        if self.config.home_channel and self.config.home_channel.chat_id:
            candidate_spaces.append(self.config.home_channel.chat_id)
        # Env-configured allowed spaces (comma-separated). Optional.
        extra_spaces = os.getenv("GOOGLE_CHAT_BOOTSTRAP_SPACES", "").strip()
        if extra_spaces:
            candidate_spaces.extend(
                s.strip() for s in extra_spaces.split(",") if s.strip()
            )
        for space in candidate_spaces:
            try:
                members = await asyncio.to_thread(
                    lambda s=space: self._chat_api.spaces()
                    .members()
                    .list(parent=s, pageSize=50)
                    .execute(http=self._new_authed_http())
                )
            except HttpError as exc:
                logger.debug(
                    "[GoogleChat] members.list failed on %s: %s",
                    space,
                    _redact_sensitive(str(exc)),
                )
                continue
            for member in members.get("memberships", []):
                if member.get("member", {}).get("type") == "BOT":
                    name = member.get("member", {}).get("name")
                    if name:
                        return name
        return None

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------
    async def connect(self) -> bool:
        """Validate config, authenticate, start Pub/Sub pull, resolve bot id."""
        if not GOOGLE_CHAT_AVAILABLE:
            self._set_fatal_error(
                code="missing_deps",
                message="google-cloud-pubsub / google-api-python-client not installed",
                retryable=False,
            )
            return False

        self._loop = asyncio.get_running_loop()
        try:
            project_id, subscription_path = self._validate_config()
            credentials = self._load_sa_credentials()
        except (ValueError, FileNotFoundError) as exc:
            msg = _redact_sensitive(str(exc))
            logger.error("[GoogleChat] Config validation failed: %s", msg)
            self._set_fatal_error(code="config_invalid", message=msg, retryable=False)
            return False

        self._project_id = project_id
        self._subscription_path = subscription_path
        self._credentials = credentials

        # Build Chat REST client (sync; wrap calls in asyncio.to_thread).
        try:
            self._chat_api = await asyncio.to_thread(
                lambda: build_service(
                    "chat",
                    "v1",
                    credentials=credentials,
                    cache_discovery=False,
                )
            )
        except Exception as exc:
            msg = _redact_sensitive(str(exc))
            logger.error("[GoogleChat] Failed to build Chat API client: %s", msg)
            self._set_fatal_error(code="chat_api_init", message=msg, retryable=False)
            return False

        # Attempt to load user OAuth credentials for native attachment
        # delivery. The Chat ``media.upload`` endpoint refuses SA auth, so
        # uploads need a user-issued token. Failure here is NON-fatal:
        # text messaging continues to work; only attachments degrade to
        # a setup-instructions text notice. The user runs ``/setup-files``
        # in chat once to grant chat.messages.create scope.
        try:
            from gateway.platforms.google_chat_user_oauth import (
                load_user_credentials as _load_user_creds,
                build_user_chat_service as _build_user_chat,
            )
            user_creds = await asyncio.to_thread(_load_user_creds)
            if user_creds is not None:
                self._user_credentials = user_creds
                self._user_chat_api = await asyncio.to_thread(
                    lambda: _build_user_chat(user_creds)
                )
                logger.info(
                    "[GoogleChat] User OAuth loaded — native attachment "
                    "delivery enabled"
                )
            else:
                logger.info(
                    "[GoogleChat] No user OAuth token at setup — file "
                    "attachments will degrade to text-only fallback. Run "
                    "/setup-files in chat to enable native attachments."
                )
        except Exception as exc:
            logger.warning(
                "[GoogleChat] User OAuth load failed (attachments will "
                "degrade to text-only fallback): %s",
                _redact_sensitive(str(exc)),
            )
            self._user_credentials = None
            self._user_chat_api = None

        # Sanity check: subscription exists / SA has access.
        self._subscriber = pubsub_v1.SubscriberClient(credentials=credentials)
        try:
            await asyncio.to_thread(
                lambda: self._subscriber.get_subscription(
                    request={"subscription": subscription_path}
                )
            )
        except gax_exceptions.NotFound:
            self._set_fatal_error(
                code="subscription_not_found",
                message="Pub/Sub subscription not found at configured path",
                retryable=False,
            )
            return False
        except gax_exceptions.PermissionDenied:
            self._set_fatal_error(
                code="subscription_permission",
                message=(
                    "Service Account lacks roles/pubsub.subscriber on the "
                    "subscription"
                ),
                retryable=False,
            )
            return False
        except Exception as exc:
            msg = _redact_sensitive(str(exc))
            logger.error("[GoogleChat] subscription.get failed: %s", msg)
            self._set_fatal_error(code="subscription_check", message=msg, retryable=True)
            return False

        # Resolve bot user_id (eager): cache first, then members.list.
        self._bot_user_id = self._load_cached_bot_id()
        if not self._bot_user_id:
            self._bot_user_id = await self._resolve_bot_user_id()
            if self._bot_user_id:
                self._save_cached_bot_id(self._bot_user_id)
            else:
                logger.info(
                    "[GoogleChat] bot_user_id not yet resolved; "
                    "will resolve on first addedToSpace or member lookup"
                )

        # Start the supervisor task that runs the Pub/Sub pull with exponential
        # backoff + jitter on transient errors, bails out after N retries.
        self._supervisor_task = asyncio.create_task(self._run_supervisor())
        self._mark_connected()
        logger.info(
            "[GoogleChat] Connected; project=%s, subscription=<redacted>, "
            "bot_user_id=%s, flow_control(msgs=%s, bytes=%s)",
            project_id,
            self._bot_user_id or "<unresolved>",
            self._max_messages,
            self._max_bytes,
        )
        return True

    async def disconnect(self) -> None:
        """Clean shutdown: stop accepting new messages, wait in-flight, close clients."""
        self._shutting_down = True
        if self._supervisor_task and not self._supervisor_task.done():
            self._supervisor_task.cancel()
            try:
                await asyncio.wait_for(self._supervisor_task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        if self._streaming_pull_future is not None:
            try:
                self._streaming_pull_future.cancel()
                await asyncio.to_thread(self._streaming_pull_future.result, 10.0)
            except Exception:
                pass
            self._streaming_pull_future = None
        if self._subscriber is not None:
            try:
                await asyncio.to_thread(self._subscriber.close)
            except Exception:
                pass
            self._subscriber = None
        self._mark_disconnected()
        logger.info("[GoogleChat] Disconnected")

    # ------------------------------------------------------------------
    # Pub/Sub supervisor (reconnect loop)
    # ------------------------------------------------------------------
    async def _run_supervisor(self) -> None:
        """Run the streaming_pull with exponential backoff; fatal after 10 attempts.

        ``subscribe()`` returns a concurrent.futures.Future that resolves when
        the stream dies. We await ``future.result()`` in a worker thread and
        react to exceptions.
        """
        attempt = 0
        while not self._shutting_down:
            flow = pubsub_v1.types.FlowControl(
                max_messages=self._max_messages,
                max_bytes=self._max_bytes,
            )
            try:
                future = self._subscriber.subscribe(
                    self._subscription_path,
                    callback=self._on_pubsub_message,
                    flow_control=flow,
                )
                self._streaming_pull_future = future
                if attempt > 0:
                    logger.info("[GoogleChat] Pub/Sub stream reconnected after %d attempts", attempt)
                attempt = 0
                # Blocks until stream dies or cancel().
                await asyncio.to_thread(future.result)
                # Normal completion = disconnect requested.
                if self._shutting_down:
                    return
            except asyncio.CancelledError:
                return
            except gax_exceptions.Unauthenticated:
                self._set_fatal_error(
                    code="pubsub_auth",
                    message="Pub/Sub authentication failed (SA key invalid/revoked)",
                    retryable=False,
                )
                return
            except gax_exceptions.PermissionDenied:
                self._set_fatal_error(
                    code="pubsub_permission",
                    message="SA lacks pubsub.subscriber on the subscription",
                    retryable=False,
                )
                return
            except Exception as exc:
                attempt += 1
                msg = _redact_sensitive(str(exc))
                logger.warning(
                    "[GoogleChat] Pub/Sub stream died (attempt %d/%d): %s",
                    attempt,
                    self._MAX_RECONNECT_ATTEMPTS,
                    msg,
                )
                if attempt >= self._MAX_RECONNECT_ATTEMPTS:
                    self._set_fatal_error(
                        code="pubsub_reconnect_exhausted",
                        message=f"Pub/Sub reconnect failed {attempt} times; giving up",
                        retryable=False,
                    )
                    return
                delay = min(
                    self._RECONNECT_MAX_DELAY,
                    self._RECONNECT_BASE_DELAY * (2 ** (attempt - 1)),
                )
                # Full jitter: pick uniformly in [0, delay].
                sleep_for = random.uniform(0, delay)
                try:
                    await asyncio.sleep(sleep_for)
                except asyncio.CancelledError:
                    return

    # ------------------------------------------------------------------
    # Inbound event handling (Pub/Sub callback runs in a thread)
    # ------------------------------------------------------------------
    def _on_pubsub_message(self, message: Any) -> None:
        """Pub/Sub callback — parse envelope and dispatch to asyncio loop.

        Runs in a Pub/Sub SubscriberClient worker thread, NOT the event loop.
        Never block this function; never raise out of it (that triggers
        Pub/Sub nack + infinite redelivery).

        Google Chat Events API uses CloudEvents-style Pub/Sub messages. The
        event type is carried in Pub/Sub message attributes (``ce-type``),
        not in the JSON body. The body is wrapped in a ``chat`` object whose
        keys depend on the event type:

          - google.workspace.chat.message.v1.created
              -> envelope["chat"]["messagePayload"] = {space, message}
          - google.workspace.chat.membership.v1.created
              -> envelope["chat"]["membershipPayload"] = {space, membership}
          - google.workspace.chat.membership.v1.deleted
              -> envelope["chat"]["membershipPayload"] = {space, membership}
        """
        if self._shutting_down:
            message.nack()
            return
        try:
            envelope = json.loads(message.data.decode("utf-8"))
        except Exception:
            logger.exception("[GoogleChat] Could not parse Pub/Sub envelope")
            message.ack()
            return

        attrs = dict(getattr(message, "attributes", {}) or {})
        ce_type = attrs.get("ce-type") or ""
        logger.debug(
            "[GoogleChat] Envelope keys=%s, ce-type=%s",
            list(envelope.keys()),
            ce_type,
        )
        if os.getenv("GOOGLE_CHAT_DEBUG_RAW"):
            # Dangerous flag: contains message text and sender email. Route
            # through the global redaction filter and gate at DEBUG level so
            # default log configurations never surface it. Operators must
            # enable DEBUG logging AND set this env var to see the dump.
            try:
                from agent.redact import redact_sensitive_text

                dump = redact_sensitive_text(json.dumps(envelope))
            except Exception:
                dump = "<redact filter unavailable>"
            logger.debug("[GoogleChat] RAW envelope (redacted): %s", dump[:2000])

        try:
            chat_block = envelope.get("chat") or {}

            # --- Membership events ---
            if "membership" in ce_type or "MEMBERSHIP" in ce_type:
                mpl = chat_block.get("membershipPayload") or {}
                space = mpl.get("space") or {}
                membership = mpl.get("membership") or {}
                if "created" in ce_type:
                    # ADDED_TO_SPACE for this bot — resolve self user_id.
                    member = membership.get("member") or {}
                    if member.get("type") == "BOT" and not self._bot_user_id:
                        name = member.get("name")
                        if name:
                            self._bot_user_id = name
                            self._save_cached_bot_id(name)
                    logger.info(
                        "[GoogleChat] ADDED_TO_SPACE %s", space.get("name", "?")
                    )
                else:
                    logger.info(
                        "[GoogleChat] REMOVED_FROM_SPACE %s", space.get("name", "?")
                    )
                message.ack()
                return

            # --- Card-click events (v2 follow-up) ---
            if "widget" in ce_type or "card" in ce_type.lower():
                logger.info(
                    "[GoogleChat] Card/widget event ack'd (v2 feature, deferred)"
                )
                message.ack()
                return

            # --- Message events ---
            msg_payload_wrapper = chat_block.get("messagePayload") or {}
            if not msg_payload_wrapper:
                logger.debug(
                    "[GoogleChat] Envelope missing messagePayload; ce-type=%s", ce_type
                )
                message.ack()
                return

            msg = msg_payload_wrapper.get("message") or {}
            space = msg_payload_wrapper.get("space") or msg.get("space") or {}
            sender = msg.get("sender") or {}
            sender_type = sender.get("type") or ""

            # Self-filter: drop bot-sourced messages (own replies and other bots).
            if sender_type == "BOT":
                message.ack()
                return

            # Dedup guard — Pub/Sub is at-least-once.
            msg_name = msg.get("name") or ""
            if msg_name and self._dedup.is_duplicate(msg_name):
                logger.debug("[GoogleChat] Dedup drop for %s", msg_name)
                message.ack()
                return

            # Wrap msg with parent-level space so _build_message_event can find it.
            msg_with_space = dict(msg)
            if "space" not in msg_with_space and space:
                msg_with_space["space"] = space

            # Enrich envelope with a synthetic top-level "space" field so the
            # dispatch side has a consistent shape regardless of format.
            enriched_env = dict(envelope)
            if "space" not in enriched_env and space:
                enriched_env["space"] = space

            self._submit_on_loop(self._dispatch_message(msg_with_space, enriched_env))
            message.ack()
        except Exception:
            logger.exception("[GoogleChat] Error in _on_pubsub_message")
            try:
                message.ack()
            except Exception:
                pass

    async def _dispatch_message(self, msg: Dict[str, Any], envelope: Dict[str, Any]) -> None:
        """Translate a Chat message payload to a MessageEvent and hand off.

        Intercepts the ``/setup-files`` admin command BEFORE the agent
        sees it — that's a bot-local OAuth setup flow, not a prompt.
        Everything else flows to ``handle_message`` as normal.
        """
        try:
            event = await self._build_message_event(msg, envelope)
            if event is None:
                return

            # Short-circuit /setup-files before the agent dispatch.
            text = (event.text or "").strip()
            if text.startswith("/setup-files") and event.source is not None:
                handled = await self._handle_setup_files_command(
                    chat_id=event.source.chat_id,
                    thread_id=event.source.thread_id,
                    raw_text=text,
                )
                if handled:
                    return

            await self.handle_message(event)
        except Exception:
            logger.exception("[GoogleChat] _dispatch_message failed")

    async def _handle_setup_files_command(
        self,
        chat_id: str,
        thread_id: Optional[str],
        raw_text: str,
    ) -> bool:
        """Run the in-chat OAuth setup flow for native attachment delivery.

        Returns ``True`` if the message was consumed (no agent dispatch),
        ``False`` if it should fall through.

        Subcommands:
          /setup-files                  → show status + next step
          /setup-files start            → print OAuth URL
          /setup-files revoke           → revoke and delete stored token
          /setup-files <CODE_OR_URL>    → exchange auth code for token

        Pre-requisite: client_secret.json must already be on the host
        (one-time terminal step). The status reply tells the user how to
        do that if it's missing.
        """
        from gateway.platforms import google_chat_user_oauth as oauth_helper

        parts = raw_text.split(maxsplit=1)
        # parts[0] is "/setup-files"; parts[1..] is the optional argument
        arg = parts[1].strip() if len(parts) > 1 else ""

        async def _reply(text: str) -> None:
            body: Dict[str, Any] = {"text": text}
            if thread_id:
                body["thread"] = {"name": thread_id}
            try:
                await self._create_message(chat_id, body)
            except Exception:
                logger.debug(
                    "[GoogleChat] /setup-files reply send failed",
                    exc_info=True,
                )

        # Status / no-arg: show what's set up and what to do next.
        if not arg:
            client_secret_present = (
                oauth_helper._client_secret_path().exists()
            )
            token_present = oauth_helper._token_path().exists()
            creds = oauth_helper.load_user_credentials() if token_present else None
            if creds is not None:
                await _reply(
                    "✅ Native attachment delivery is **active**.\n"
                    "Token: "
                    f"`{oauth_helper._token_path()}`\n"
                    "Send `/setup-files revoke` to disable."
                )
                return True
            if not client_secret_present:
                await _reply(
                    "🔧 Native attachment delivery is **not configured**.\n"
                    "**Step 1 (one-time, on the host):** create OAuth client "
                    "credentials at "
                    "https://console.cloud.google.com/apis/credentials → "
                    "*Create credentials* → *OAuth client ID* → *Desktop app*. "
                    "Download the JSON. Then on the host run:\n"
                    "```\n"
                    "python -m gateway.platforms.google_chat_user_oauth "
                    "--client-secret /path/to/client_secret.json\n"
                    "```\n"
                    "**Step 2:** come back here and send `/setup-files start`."
                )
                return True
            await _reply(
                "🔧 Client credentials are stored but you haven't "
                "authorized yet. Send `/setup-files start` to begin."
            )
            return True

        if arg == "start":
            if not oauth_helper._client_secret_path().exists():
                await _reply(
                    "⚠️ No client credentials stored on the host. Send "
                    "`/setup-files` (no args) for setup instructions."
                )
                return True
            try:
                # Reuse the helper logic but capture stdout via a sync
                # thread so we don't print to the gateway terminal.
                import io
                import contextlib
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    await asyncio.to_thread(oauth_helper.get_auth_url)
                auth_url = buf.getvalue().strip().splitlines()[-1]
            except SystemExit:
                await _reply(
                    "❌ Couldn't generate the OAuth URL. Check the gateway "
                    "logs and verify the client_secret.json is valid."
                )
                return True
            except Exception as exc:
                logger.warning(
                    "[GoogleChat] /setup-files start failed: %s", exc,
                )
                await _reply(f"❌ Error: {exc}")
                return True
            await _reply(
                "1. Open this URL in your browser and authorize:\n"
                f"{auth_url}\n\n"
                "2. After clicking *Allow*, your browser will fail to load "
                "`http://localhost:1/?...&code=...`. That's expected.\n\n"
                "3. Copy the entire failed URL from the browser's URL bar "
                "and paste it back here as: `/setup-files <PASTE_URL>` "
                "(or just the `code=...` value).\n\n"
                "Tip: the URL contains your access grant — keep it private."
            )
            return True

        if arg == "revoke":
            try:
                import io
                import contextlib
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    await asyncio.to_thread(oauth_helper.revoke)
                output = buf.getvalue().strip() or "Revoked."
            except SystemExit:
                output = "Revoke completed (some steps may have been skipped)."
            except Exception as exc:
                logger.warning(
                    "[GoogleChat] /setup-files revoke failed: %s", exc,
                )
                await _reply(f"❌ Error revoking: {exc}")
                return True
            # Wipe in-memory creds so subsequent uploads fall through to
            # the setup-instructions text notice immediately.
            self._user_credentials = None
            self._user_chat_api = None
            await _reply(f"✅ Done.\n```\n{output}\n```")
            return True

        # Anything else is treated as the auth code or the failed-redirect
        # URL the user pasted.
        try:
            import io
            import contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                await asyncio.to_thread(
                    oauth_helper.exchange_auth_code, arg,
                )
            output = buf.getvalue().strip()
        except SystemExit:
            await _reply(
                "❌ Token exchange failed. The code may have expired or "
                "the URL is malformed. Send `/setup-files start` to get "
                "a fresh OAuth URL."
            )
            return True
        except Exception as exc:
            logger.warning(
                "[GoogleChat] /setup-files exchange failed: %s", exc,
            )
            await _reply(f"❌ Error: {exc}")
            return True

        # Re-load credentials into the adapter so the next file send uses
        # them WITHOUT a gateway restart.
        try:
            new_creds = await asyncio.to_thread(oauth_helper.load_user_credentials)
            if new_creds is not None:
                self._user_credentials = new_creds
                self._user_chat_api = await asyncio.to_thread(
                    lambda: oauth_helper.build_user_chat_service(new_creds)
                )
                await _reply(
                    "✅ Authorized! Native attachment delivery is now "
                    "active. Try asking me to send you a PDF."
                )
                return True
        except Exception as exc:
            logger.warning(
                "[GoogleChat] post-exchange creds load failed: %s", exc,
            )

        await _reply(
            "⚠️ Token exchanged but the gateway couldn't load the new "
            "credentials in-memory. Restart the gateway and the token "
            f"at `{oauth_helper._token_path()}` will be picked up.\n"
            f"Helper output:\n```\n{output}\n```"
        )
        return True

    async def _build_message_event(
        self, msg: Dict[str, Any], envelope: Dict[str, Any]
    ) -> Optional[MessageEvent]:
        """Parse a Chat API message into a hermes MessageEvent."""
        space = envelope.get("space") or msg.get("space") or {}
        space_name = space.get("name") or ""  # "spaces/XXX"
        space_type = (space.get("type") or space.get("spaceType") or "").upper()
        thread = msg.get("thread") or {}
        thread_name = thread.get("name") or None
        sender = msg.get("sender") or {}
        sender_name = sender.get("name") or ""
        sender_display = sender.get("displayName") or sender.get("email") or sender_name
        sender_email = sender.get("email") or ""

        chat_type = "dm" if space_type in ("DIRECT_MESSAGE", "DM") else "group"
        text = msg.get("argumentText") or msg.get("text") or ""
        text = text.strip()

        # Slash command: emit MessageType.COMMAND with normalized text.
        slash = msg.get("slashCommand") or {}
        is_slash = bool(slash)
        if is_slash:
            command_id = str(slash.get("commandId") or "")
            if command_id and not text.startswith("/"):
                text = f"/cmd_{command_id} {text}".strip()

        # Attachments: download and cache.
        media_urls: List[str] = []
        media_types: List[str] = []
        message_type = MessageType.TEXT
        attachments = msg.get("attachment") or []
        for att in attachments:
            try:
                local_path, mime = await self._download_attachment(att)
            except Exception:
                logger.exception("[GoogleChat] attachment download failed")
                continue
            if not local_path:
                continue
            media_urls.append(local_path)
            media_types.append(mime or "application/octet-stream")
            # Prefer the first-seen type for MessageType if no text present.
            if message_type == MessageType.TEXT and not text:
                message_type = _mime_for_message_type(mime or "")

        if is_slash:
            message_type = MessageType.COMMAND

        # Cache the inbound thread for outbound reply placement (see
        # _last_inbound_thread docstring). This runs BEFORE the source
        # builds so DMs can drop thread_id without losing the reply
        # destination.
        if thread_name and space_name:
            self._last_inbound_thread[space_name] = thread_name

        # In DMs, do NOT propagate thread_id to the session source.
        # Google Chat DMs spawn a fresh thread per top-level user
        # message, but that's a UI artifact — the conversation is
        # logically one stream. Including thread_id in the session key
        # would make every new top-level message a fresh session with no
        # memory of prior turns. For groups we keep thread_id (different
        # threads ARE different conversations there).
        session_thread_id = None if chat_type == "dm" else thread_name

        source = self.build_source(
            chat_id=space_name,
            chat_name=space.get("displayName") or space.get("name") or "",
            chat_type=chat_type,
            user_id=sender_name,
            user_name=sender_display,
            thread_id=session_thread_id,
            user_id_alt=sender_email or None,
        )
        return MessageEvent(
            text=text,
            message_type=message_type,
            source=source,
            raw_message=msg,
            message_id=msg.get("name") or None,
            media_urls=media_urls,
            media_types=media_types,
        )

    async def _download_attachment(
        self, attachment: Dict[str, Any]
    ) -> Tuple[Optional[str], Optional[str]]:
        """Download an inbound attachment to the local cache; return (path, mime).

        Priority for bot Service Accounts:

          1. ``attachmentDataRef.resourceName`` via ``chat.media.download`` —
             the supported bot path. The Service Account bearer token has
             ``chat.bot`` scope which the Chat API authorises against the
             space membership.
          2. Drive-hosted files (``source == 'DRIVE_FILE'``) require user
             OAuth and Drive scope; skip with a log.
          3. Direct HTTP fetch of ``downloadUri`` only as a last resort —
             that URL is meant for user OAuth tokens (chat.google.com
             returns 401 for SA bearer tokens) and is unlikely to work,
             but we keep the path for forward-compat with Google changes.
        """
        mime = attachment.get("contentType") or ""
        source = attachment.get("source") or ""
        name = attachment.get("name") or ""
        attachment_data_ref = attachment.get("attachmentDataRef") or {}
        resource_name = attachment_data_ref.get("resourceName") or ""
        download_uri = attachment.get("downloadUri") or ""

        # NOTE on ``source == "DRIVE_FILE"``: Google Chat tags BOTH
        # drag-and-drop chat uploads AND Drive-picker shares with this
        # source string, but the two have different access models.
        # Drag-and-drop uploads come with an ``attachmentDataRef.resourceName``
        # that bot SA tokens CAN download via ``media.download_media``.
        # Pure Drive-picker shares often lack that field and require
        # user OAuth + Drive scope (which we deliberately don't request).
        # So we only short-circuit when there's nothing the bot path
        # can use — otherwise try the bot path first.
        if source == "DRIVE_FILE" and not resource_name:
            logger.info(
                "[GoogleChat] Skipping Drive-picker attachment (no "
                "resourceName, would need user-OAuth Drive scope)"
            )
            return None, mime

        data: Optional[bytes] = None

        # Path 1: media.download with attachmentDataRef.resourceName (bot-path).
        if resource_name:
            def _fetch_media() -> bytes:
                req = self._chat_api.media().download_media(
                    resourceName=resource_name,
                )
                from googleapiclient.http import MediaIoBaseDownload
                import io

                buf = io.BytesIO()
                downloader = MediaIoBaseDownload(buf, req)
                done = False
                while not done:
                    _status, done = downloader.next_chunk()
                return buf.getvalue()

            try:
                data = await asyncio.to_thread(_fetch_media)
            except HttpError as exc:
                logger.warning(
                    "[GoogleChat] media.download_media failed: %s",
                    _redact_sensitive(str(exc)),
                )
                data = None

        # Path 2: downloadUri fallback (rarely works with SA tokens, but try).
        if data is None and download_uri:
            if not _is_google_owned_host(download_uri):
                logger.warning(
                    "[GoogleChat] Rejecting attachment fetch: non-Google host"
                )
                return None, mime

            def _fetch_uri() -> bytes:
                import google.auth.transport.requests as gar

                authed_session = gar.AuthorizedSession(self._credentials)
                resp = authed_session.get(download_uri, timeout=30)
                resp.raise_for_status()
                return resp.content

            try:
                data = await asyncio.to_thread(_fetch_uri)
            except Exception as exc:
                logger.warning(
                    "[GoogleChat] downloadUri fetch failed (SA tokens often "
                    "lack access here; this is expected for user-uploaded "
                    "content): %s",
                    _redact_sensitive(str(exc)),
                )
                return None, mime

        if data is None:
            return None, mime

        # Cache based on MIME. Upstream's cache_* helpers expect `ext` for
        # media (image/audio/video) and a positional `filename` for docs.
        filename = name.split("/")[-1] if name else "attachment"
        if "." in filename:
            ext = "." + filename.rsplit(".", 1)[-1].lower()
        else:
            ext = ""
        if mime.startswith("image/"):
            local = cache_image_from_bytes(data, ext=ext or ".jpg")
        elif mime.startswith("audio/"):
            local = cache_audio_from_bytes(data, ext=ext or ".ogg")
        elif mime.startswith("video/"):
            local = cache_video_from_bytes(data, ext=ext or ".mp4")
        else:
            local = cache_document_from_bytes(data, filename)
        return local, mime

    # ------------------------------------------------------------------
    # Outbound send paths
    # ------------------------------------------------------------------
    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a text message.

        Signature matches ``BasePlatformAdapter.send``: ``content`` is the
        message body, ``reply_to`` is an optional message_id (the inbound
        message to thread under), and ``metadata`` may carry ``thread_id``
        (the resolved Google Chat ``spaces/X/threads/Y`` resource name).

        If a typing card is tracked for this chat, transform it in-place via
        ``messages.patch`` — NO delete+create. Google Chat shows a tombstone
        ("Message deleted by its author") on delete, which is visual noise.
        Patch rewrites the text of the existing message seamlessly.

        Also pauses the base class's ``_keep_typing`` loop for this chat so
        it can't post a racing typing card between the patch and the reply.

        If ``content`` exceeds MAX_MESSAGE_LENGTH, the first chunk patches
        the typing card (if any), subsequent chunks are new messages.
        """
        thread_id = self._resolve_thread_id(reply_to, metadata, chat_id=chat_id)
        self.pause_typing_for_chat(chat_id)
        try:
            chunks = self._chunk_text(content)
            if not chunks:
                return SendResult(success=False, error="empty message")

            last_result: Optional[SendResult] = None
            typing_msg_name = self._typing_messages.pop(chat_id, None)
            # Treat any earlier sentinel as "no real card to patch" — defensive.
            if typing_msg_name == _TYPING_CONSUMED_SENTINEL:
                typing_msg_name = None
            patched_typing = False

            for idx, chunk in enumerate(chunks):
                body: Dict[str, Any] = {"text": chunk}
                # Only set thread on new-message create path. Patch inherits.
                if thread_id and (idx > 0 or not typing_msg_name):
                    body["thread"] = {"name": thread_id}
                try:
                    if idx == 0 and typing_msg_name:
                        result = await self._patch_message(typing_msg_name, body)
                        patched_typing = True
                    else:
                        result = await self._create_message(chat_id, body)
                    last_result = result
                except HttpError as exc:
                    status = getattr(getattr(exc, "resp", None), "status", None)
                    if status == 403:
                        self._set_fatal_error(
                            code="chat_forbidden",
                            message="Bot lacks access (removed from space or perms revoked)",
                            retryable=False,
                        )
                        return SendResult(success=False, error=str(exc))
                    if status == 404:
                        # Typing card was deleted out from under us, or space
                        # is gone. Fall through to creating a new message on
                        # the first-chunk patch failure.
                        if idx == 0 and typing_msg_name:
                            logger.info(
                                "[GoogleChat] Typing card disappeared; creating new message"
                            )
                            typing_msg_name = None
                            result = await self._create_message(chat_id, body)
                            last_result = result
                            continue
                        logger.info("[GoogleChat] send target 404; skipping")
                        return SendResult(success=False, error="target not found")
                    if status == 429:
                        self._rate_limit_hits[chat_id] = (
                            self._rate_limit_hits.get(chat_id, 0) + 1
                        )
                        if self._rate_limit_hits[chat_id] >= _RATE_LIMIT_WARN_THRESHOLD:
                            logger.warning(
                                "[GoogleChat] Rate limit hit %d times on chat; throttling",
                                self._rate_limit_hits[chat_id],
                            )
                        raise
                    raise
            if last_result is None:
                return SendResult(success=False, error="empty message")
            # Mark the chat's typing slot as "consumed" so the base class's
            # _keep_typing loop (which may iterate one more time before
            # typing_task.cancel() lands) does not post a fresh marker that
            # the safety-net stop_typing would then delete and tombstone.
            # Cleared in on_processing_complete.
            if patched_typing:
                self._typing_messages[chat_id] = _TYPING_CONSUMED_SENTINEL
            return last_result
        finally:
            self.resume_typing_for_chat(chat_id)

    async def edit_message(
        self,
        chat_id: str,
        message_id: str,
        content: str,
        *,
        finalize: bool = False,
    ) -> SendResult:
        """Edit a previously sent message via ``messages.patch``.

        Required for the gateway tool-progress + token-streaming pipeline:
        ``GatewayStreamConsumer`` and ``send_progress_messages`` both gate
        on this method being overridden (see gateway/run.py:10199 and
        gateway/stream_consumer.py). Without it, Google Chat shows no
        tool activity (no "🔍 web_search…", no progressive token edits).

        ``message_id`` is the Google Chat resource name
        ``spaces/X/messages/Y``. ``finalize`` is unused here — Google
        Chat's patch API has no streaming lifecycle state, so the same
        patch closes the stream and any prior edit.

        404 (message gone) and 403 (perms revoked) are reported as
        non-success; the gateway falls back to ``send()`` for the next
        edit cycle.
        """
        if not message_id:
            return SendResult(success=False, error="missing message_id")
        # Google Chat caps message text at 4096; we use 4000 elsewhere.
        if len(content) > _MAX_TEXT_LENGTH:
            content = content[: _MAX_TEXT_LENGTH - 1] + "…"
        try:
            return await self._patch_message(message_id, {"text": content})
        except HttpError as exc:
            status = getattr(getattr(exc, "resp", None), "status", None)
            if status == 429:
                self._rate_limit_hits[chat_id] = (
                    self._rate_limit_hits.get(chat_id, 0) + 1
                )
            return SendResult(
                success=False, error=_redact_sensitive(str(exc))
            )
        except Exception as exc:
            logger.debug("[GoogleChat] edit_message failed", exc_info=True)
            return SendResult(success=False, error=str(exc))

    async def delete_message(self, chat_id: str, message_id: str) -> bool:
        """Delete a message — used sparingly (deletion creates a tombstone).

        The base contract returns False on unsupported. We do support it,
        but most internal code should prefer ``edit_message`` to avoid the
        "Message deleted by its author" tombstone. Provided so the
        gateway's stream-consumer fallback paths (e.g. removing an aborted
        partial preview) work correctly when explicit deletion is the
        right call.
        """
        if not message_id:
            return False

        def _do_delete() -> None:
            (
                self._chat_api.spaces()
                .messages()
                .delete(name=message_id)
                .execute(http=self._new_authed_http())
            )

        try:
            await asyncio.to_thread(_do_delete)
            return True
        except HttpError as exc:
            status = getattr(getattr(exc, "resp", None), "status", None)
            if status in (403, 404):
                return False
            logger.debug(
                "[GoogleChat] delete_message failed: %s",
                _redact_sensitive(str(exc)),
            )
            return False
        except Exception:
            logger.debug("[GoogleChat] delete_message failed", exc_info=True)
            return False

    async def _patch_message(
        self, message_name: str, body: Dict[str, Any]
    ) -> SendResult:
        """Update a message's text (and optionally cards) in-place."""
        update_mask_fields = []
        if "text" in body:
            update_mask_fields.append("text")
        if "cardsV2" in body:
            update_mask_fields.append("cardsV2")
        update_mask = ",".join(update_mask_fields) or "text"

        # Patch body cannot carry thread (immutable).
        patch_body = {k: v for k, v in body.items() if k not in ("thread",)}

        def _do_patch() -> Dict[str, Any]:
            return (
                self._chat_api.spaces()
                .messages()
                .patch(name=message_name, updateMask=update_mask, body=patch_body)
                .execute(http=self._new_authed_http())
            )

        resp = await asyncio.to_thread(_do_patch)
        return SendResult(success=True, message_id=resp.get("name", message_name))

    def _chunk_text(self, text: str) -> List[str]:
        if not text:
            return []
        if len(text) <= _MAX_TEXT_LENGTH:
            return [text]
        chunks: List[str] = []
        remaining = text
        while remaining:
            if len(remaining) <= _MAX_TEXT_LENGTH:
                chunks.append(remaining)
                break
            # Try to split on a newline near the cutoff.
            cut = remaining.rfind("\n", 0, _MAX_TEXT_LENGTH)
            if cut < _MAX_TEXT_LENGTH // 2:
                cut = _MAX_TEXT_LENGTH
            chunks.append(remaining[:cut])
            remaining = remaining[cut:].lstrip()
        return chunks

    def _resolve_thread_id(
        self,
        reply_to: Optional[str],
        metadata: Optional[Dict[str, Any]],
        chat_id: Optional[str] = None,
    ) -> Optional[str]:
        """Return the Google Chat thread resource name to reply under, or None.

        Priority:
          1. ``metadata['thread_id']`` — populated by the gateway's session
             plumbing from ``SessionSource.thread_id`` (the inbound
             ``thread.name``). Canonical path for groups.
          2. ``metadata['thread_name']`` / ``metadata['thread_ts']`` — Slack
             precedent aliases that the broader codebase sometimes passes.
          3. ``reply_to`` if it already looks like a thread resource name
             (``spaces/X/threads/Y``). Message names ``spaces/X/messages/Y``
             cannot be converted to threads without an extra API call.
          4. ``self._last_inbound_thread[chat_id]`` — Google Chat DMs spawn
             a new thread per top-level user message, and the adapter
             intentionally drops thread_id from the source so the session
             key stays stable. Without this fallback, DM replies would
             land at top-level (a fresh thread separate from the user's),
             visually disconnected from the user's question.
        """
        if metadata:
            for key in ("thread_id", "thread_name", "thread_ts"):
                value = metadata.get(key)
                if value:
                    return str(value)
        if reply_to and "/threads/" in reply_to and "/messages/" not in reply_to:
            return reply_to
        if chat_id:
            cached = self._last_inbound_thread.get(chat_id)
            if cached:
                return cached
        return None

    def _new_authed_http(self) -> Any:
        """Return a fresh AuthorizedHttp.

        googleapiclient's discovery client is NOT thread-safe because httplib2
        shares SSL state between calls. Passing a fresh http= to each
        ``execute()`` avoids record-layer failures when calls run in
        ``asyncio.to_thread`` workers. Cheap (~no network).
        """
        return AuthorizedHttp(self._credentials, http=httplib2.Http(timeout=30))

    async def _create_message(
        self, chat_id: str, body: Dict[str, Any]
    ) -> SendResult:
        """POST spaces/{space}/messages via REST, returning SendResult."""

        def _do_create() -> Dict[str, Any]:
            return (
                self._chat_api.spaces()
                .messages()
                .create(parent=chat_id, body=body)
                .execute(http=self._new_authed_http())
            )

        resp = await asyncio.to_thread(_do_create)
        return SendResult(success=True, message_id=resp.get("name"))

    async def send_typing(self, chat_id: str, metadata: Any = None) -> None:
        """Post a visible 'Hermes is thinking…' marker message.

        NOT ephemeral (Google Chat has no ephemeral text messages outside
        slash command responses). ``send()`` PATCHes this marker in-place
        with the real response (no deletion tombstone). The typing card is
        either patched by ``send()`` (success) or by
        ``on_processing_complete`` (failure / cancellation).

        IMPORTANT — must place the typing card in the user's thread:
        ``messages.patch`` cannot change a message's ``thread`` (it's
        immutable on update). If we create the typing card at top-level
        and the user is replying inside thread T, send() will patch the
        top-level card in place — leaving the bot's whole response
        stranded outside the user's thread. We resolve the thread the
        same way send() does (metadata override + last-inbound-thread
        cache) so the typing card and the patched reply share a thread.
        """
        # If already showing a typing marker, do nothing.
        if chat_id in self._typing_messages:
            return

        thread_id = self._resolve_thread_id(
            reply_to=None, metadata=metadata, chat_id=chat_id,
        )
        body: Dict[str, Any] = {"text": "Hermes is thinking…"}
        if thread_id:
            body["thread"] = {"name": thread_id}
        try:
            result = await self._create_message(chat_id, body)
        except Exception:
            logger.debug("[GoogleChat] send_typing failed; skipping")
            return
        if result.success and result.message_id:
            self._typing_messages[chat_id] = result.message_id

    async def stop_typing(self, chat_id: str) -> None:
        """Stop the typing indicator — NO-OP when a live card is tracked.

        Google Chat has no separate typing API: the "Hermes is thinking…"
        marker is a real message that ``send()`` patches in-place with the
        agent's reply. Deleting the marker creates a "Message deleted by
        its author" tombstone, which is visual noise.

        Upstream code (gateway/run.py and gateway/platforms/base.py) calls
        ``stop_typing`` at three moments per turn — typically BEFORE
        ``send()`` runs (so deleting the slot would leave ``send()``
        nothing to patch, forcing it to create a fresh message and leaving
        the original card as a tombstone). To fix this without modifying
        upstream contracts, ``stop_typing`` here is intentionally a NO-OP
        when the slot holds a real ``message_name``: the card is left in
        place so ``send()`` can patch it.

        Three cases:
          * Slot empty → nothing to do.
          * Slot holds SENTINEL → ``send()`` already patched the card;
            pop the sentinel so the next turn starts clean.
          * Slot holds a real ``message_name`` → leave it for ``send()``
            to consume. NO-OP.

        Stranded cards on error / cancellation paths (where ``send()``
        never runs) are reaped by ``on_processing_complete`` — see that
        hook for the patch-to-final-state cleanup.
        """
        current = self._typing_messages.get(chat_id)
        if not current:
            return
        if current == _TYPING_CONSUMED_SENTINEL:
            self._typing_messages.pop(chat_id, None)
            return
        # Real message_name — leave it for send() to patch. Deliberate no-op.
        return

    async def on_processing_complete(
        self, event: MessageEvent, outcome: ProcessingOutcome
    ) -> None:
        """Reap typing card after the message-handling cycle ends.

        SUCCESS: ``send()`` set the SENTINEL after patching. Pop it.

        FAILURE / CANCELLED: ``send()`` may not have run, leaving a real
        ``message_name`` in the slot. Patching the card to a final state
        (``"(interrupted)"``) avoids the tombstone that ``messages.delete``
        would create. If ``send()`` did run (e.g. base.py error-send branch
        patched it), the slot holds the SENTINEL — pop and exit.
        """
        if event.source is None:
            return
        chat_id = event.source.chat_id
        try:
            current = self._typing_messages.pop(chat_id, None)
            if not current or current == _TYPING_CONSUMED_SENTINEL:
                return
            # Real message_name still in slot — send() never ran.  Patch the
            # card with a benign final state instead of deleting (no tombstone).
            label = "(interrupted)" if outcome == ProcessingOutcome.CANCELLED else "(no reply)"
            try:
                await self._patch_message(current, {"text": label})
            except Exception:
                logger.debug(
                    "[GoogleChat] on_processing_complete patch fallback failed",
                    exc_info=True,
                )
        except Exception:
            logger.debug(
                "[GoogleChat] cleanup in on_processing_complete failed", exc_info=True
            )

    # ------------------------------------------------------------------
    # Attachment send paths
    # ------------------------------------------------------------------
    async def _consume_typing_card_with_text(
        self, chat_id: str, text: str
    ) -> Optional[SendResult]:
        """Patch the tracked typing card with ``text`` (no tombstone).

        Returns ``None`` if there's no real typing card to patch (caller
        should create a new message). Returns the patch result if the
        card was successfully patched. Raises on transient HttpErrors so
        the caller can decide whether to fall back to ``_create_message``.

        Leaves the SENTINEL in place when present: a previous ``send()``
        already consumed the typing card, and the SENTINEL must stay in
        the slot to keep the base class's ``_keep_typing`` loop from
        creating a fresh "Hermes is thinking…" card during any subsequent
        attachment send (which would later be reaped as "(no reply)").
        """
        current = self._typing_messages.get(chat_id)
        if not current or current == _TYPING_CONSUMED_SENTINEL:
            return None
        # Real msg_id — pop and patch.
        self._typing_messages.pop(chat_id, None)
        try:
            result = await self._patch_message(current, {"text": text})
            self._typing_messages[chat_id] = _TYPING_CONSUMED_SENTINEL
            return result
        except HttpError as exc:
            status = getattr(getattr(exc, "resp", None), "status", None)
            if status == 404:
                # Card disappeared — caller should create a new message.
                return None
            raise

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send an inline image via attachment URL (no upload).

        If a typing card is tracked for this chat, patch it in-place with
        the image (caption + URL) — same anti-tombstone pattern used by
        ``send()``. Otherwise create a new message.
        """
        thread_id = self._resolve_thread_id(reply_to, metadata, chat_id=chat_id)
        text_parts: List[str] = []
        if caption:
            text_parts.append(caption)
        text_parts.append(image_url)
        text = "\n".join(text_parts)

        try:
            patched = await self._consume_typing_card_with_text(chat_id, text)
            if patched is not None:
                return patched
            body: Dict[str, Any] = {"text": text}
            if thread_id:
                body["thread"] = {"name": thread_id}
            return await self._create_message(chat_id, body)
        except HttpError as exc:
            return SendResult(success=False, error=_redact_sensitive(str(exc)))

    async def send_image_file(
        self,
        chat_id: str,
        image_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs: Any,
    ) -> SendResult:
        return await self._send_file(
            chat_id, image_path, caption,
            mime_hint="image/*",
            thread_id=self._resolve_thread_id(reply_to, kwargs.get("metadata"), chat_id=chat_id),
        )

    async def send_document(
        self,
        chat_id: str,
        file_path: str,
        caption: Optional[str] = None,
        file_name: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs: Any,
    ) -> SendResult:
        return await self._send_file(
            chat_id, file_path, caption,
            mime_hint=None,
            thread_id=self._resolve_thread_id(reply_to, kwargs.get("metadata"), chat_id=chat_id),
            override_filename=file_name,
        )

    async def send_voice(
        self,
        chat_id: str,
        audio_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs: Any,
    ) -> SendResult:
        return await self._send_file(
            chat_id, audio_path, caption,
            mime_hint="audio/ogg",
            thread_id=self._resolve_thread_id(reply_to, kwargs.get("metadata"), chat_id=chat_id),
        )

    async def send_video(
        self,
        chat_id: str,
        video_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs: Any,
    ) -> SendResult:
        return await self._send_file(
            chat_id, video_path, caption,
            mime_hint="video/mp4",
            thread_id=self._resolve_thread_id(reply_to, kwargs.get("metadata"), chat_id=chat_id),
        )

    async def send_animation(
        self,
        chat_id: str,
        animation_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Google Chat has no native animation type; fall back to send_image."""
        return await self.send_image(
            chat_id, animation_url, caption=caption,
            reply_to=reply_to, metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Native attachment delivery via user OAuth
    #
    # Google Chat's media.upload endpoint hard-rejects SA authentication
    # ("This method doesn't support app authentication with a service
    # account"). The bot itself cannot upload files. Instead the user
    # grants the bot the chat.messages.create scope ONCE via an in-chat
    # OAuth consent flow (``/setup-files``); the resulting refresh token
    # lets the bot call media.upload AS the user, producing native Chat
    # attachments (file widget, inline preview, click-to-download).
    #
    # See https://developers.google.com/chat/api/guides/auth/users for
    # the upstream limitation that makes user OAuth necessary, and
    # ``gateway/platforms/google_chat_user_oauth.py`` for the helper
    # script + library functions backing this path.
    # ------------------------------------------------------------------
    @staticmethod
    def _is_app_auth_attachment_error(exc: HttpError) -> bool:
        """Detect Google Chat's media.upload bot-auth rejection.

        Returns True for the canonical ``"doesn't support app
        authentication"`` wording (and the legacy
        ``ACCESS_TOKEN_SCOPE_INSUFFICIENT`` variant some older clients
        still see). Used to flag a misuse — calling ``media.upload``
        through the SA-authed Chat API client instead of the user-authed
        one. With correct routing this error should never fire in the
        adapter; it remains as a defensive check.
        """
        text = str(exc) or ""
        return (
            "doesn't support app authentication" in text
            or "ACCESS_TOKEN_SCOPE_INSUFFICIENT" in text
        )

    async def _send_file(
        self,
        chat_id: str,
        path: str,
        caption: Optional[str],
        mime_hint: Optional[str],
        thread_id: Optional[str] = None,
        override_filename: Optional[str] = None,
    ) -> SendResult:
        """Native Chat attachment via user-OAuth media.upload.

        Two-step on the wire: ``media.upload`` then
        ``spaces.messages.create`` with the returned ``attachmentDataRef``.
        BOTH calls go through the user-authed Chat API client
        (``self._user_chat_api``) — the SA-authed client is rejected by
        ``media.upload`` regardless of scopes.

        If user OAuth is not configured (``self._user_chat_api is None``)
        or the upload fails with auth errors, the method posts a
        text-only fallback message explaining how to run ``/setup-files``
        and returns ``success=False`` so callers know delivery failed.

        Google Chat ``messages.patch`` cannot add an attachment to an
        existing message, so we cannot transform the typing card directly
        into the file message. Instead we patch the typing card with the
        caption (or a single space when none) so it retires without a
        tombstone, then create the attachment message.
        """
        if not os.path.exists(path):
            return SendResult(success=False, error=f"file not found: {path}")

        filename = override_filename or os.path.basename(path) or "upload.bin"
        mime = mime_hint or "application/octet-stream"

        # No user OAuth → can't upload natively. Surface clear setup
        # instructions in chat instead of silently failing.
        if self._user_chat_api is None:
            return await self._post_attachment_fallback(
                chat_id=chat_id,
                path=path,
                filename=filename,
                caption=caption,
                thread_id=thread_id,
            )

        # Pre-patch the typing card with the caption (or single space) so
        # it retires without a tombstone before the attachment message is
        # posted.
        try:
            await self._consume_typing_card_with_text(chat_id, caption or " ")
        except Exception:
            logger.debug(
                "[GoogleChat] _send_file pre-patch typing-card failed",
                exc_info=True,
            )

        # Refresh user creds if expired (token may have aged out between
        # adapter startup and now).
        try:
            from gateway.platforms.google_chat_user_oauth import (
                refresh_or_none as _refresh_creds,
            )
            refreshed = await asyncio.to_thread(
                _refresh_creds, self._user_credentials,
            )
            if refreshed is None:
                logger.warning(
                    "[GoogleChat] User OAuth refresh returned None — "
                    "treating as unconfigured"
                )
                self._user_credentials = None
                self._user_chat_api = None
                return await self._post_attachment_fallback(
                    chat_id=chat_id,
                    path=path,
                    filename=filename,
                    caption=caption,
                    thread_id=thread_id,
                )
            self._user_credentials = refreshed
        except Exception:
            logger.debug(
                "[GoogleChat] user-OAuth refresh failed (continuing with "
                "existing creds)", exc_info=True,
            )

        def _upload() -> Dict[str, Any]:
            media = MediaFileUpload(path, mimetype=mime, resumable=False)
            return (
                self._user_chat_api.media()
                .upload(
                    parent=chat_id,
                    body={"filename": filename},
                    media_body=media,
                )
                .execute()
            )

        try:
            upload_resp = await asyncio.to_thread(_upload)
        except HttpError as exc:
            status = getattr(getattr(exc, "resp", None), "status", None)
            if status in (401, 403):
                logger.warning(
                    "[GoogleChat] media.upload auth failure (token "
                    "revoked or scope missing) — falling back to text "
                    "notice. Status=%s", status,
                )
                self._user_credentials = None
                self._user_chat_api = None
                return await self._post_attachment_fallback(
                    chat_id=chat_id,
                    path=path,
                    filename=filename,
                    caption=caption,
                    thread_id=thread_id,
                )
            return SendResult(
                success=False, error=_redact_sensitive(str(exc))
            )

        attachment_ref = upload_resp.get("attachmentDataRef")
        if not attachment_ref:
            return SendResult(
                success=False,
                error="upload returned no attachmentDataRef",
            )

        body: Dict[str, Any] = {
            "attachment": [{"attachmentDataRef": attachment_ref}],
        }
        if caption:
            body["text"] = caption
        if thread_id:
            body["thread"] = {"name": thread_id}

        # The accompanying messages.create that references the attachment
        # also needs user auth (the attachmentDataRef is bound to the
        # uploading principal).
        def _create_with_attachment() -> Dict[str, Any]:
            return (
                self._user_chat_api.spaces()
                .messages()
                .create(parent=chat_id, body=body)
                .execute()
            )

        try:
            resp = await asyncio.to_thread(_create_with_attachment)
            return SendResult(
                success=True, message_id=resp.get("name"),
            )
        except HttpError as exc:
            return SendResult(
                success=False, error=_redact_sensitive(str(exc))
            )

    async def _post_attachment_fallback(
        self,
        chat_id: str,
        path: str,
        filename: str,
        caption: Optional[str],
        thread_id: Optional[str],
    ) -> SendResult:
        """Post a text notice when native attachment delivery is unavailable.

        Tells the user that file delivery requires a one-time consent
        flow (``/setup-files``) and reports the local-host path so the
        file isn't lost. Returns ``success=False`` so callers know the
        attachment did not land.
        """
        lines = []
        if caption:
            lines.append(caption)
        lines.extend([
            f"⚠️ No he podido adjuntar **{filename}**.",
            "Google Chat sólo permite adjuntar archivos cuando el bot tiene "
            "permiso explícito tuyo (OAuth de usuario). Es un consentimiento "
            "único que se hace desde este chat.",
            "**Para activarlo:** envía `/setup-files` y sigue las instrucciones.",
            f"Mientras tanto el archivo está en el host: `{path}`",
        ])
        body: Dict[str, Any] = {"text": "\n".join(lines)}
        if thread_id:
            body["thread"] = {"name": thread_id}
        try:
            await self._create_message(chat_id, body)
        except Exception:
            logger.debug(
                "[GoogleChat] attachment fallback notice send failed",
                exc_info=True,
            )
        return SendResult(
            success=False,
            error="google_chat: native attachment requires user OAuth — "
            "run /setup-files in chat",
        )

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Return {name, type, chat_id} for a space."""
        try:
            info = await asyncio.to_thread(
                lambda: self._chat_api.spaces()
                .get(name=chat_id)
                .execute(http=self._new_authed_http())
            )
        except HttpError as exc:
            logger.debug(
                "[GoogleChat] get_chat_info failed: %s", _redact_sensitive(str(exc))
            )
            return {"name": chat_id, "type": "group", "chat_id": chat_id}
        space_type = (info.get("spaceType") or info.get("type") or "").upper()
        display = info.get("displayName") or chat_id
        return {
            "name": display,
            "type": "dm" if space_type in ("DIRECT_MESSAGE", "DM") else "group",
            "chat_id": chat_id,
        }

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
from typing import Any, Callable, Dict, List, Optional, Tuple

# Heavy google-cloud + googleapiclient imports are deferred to first
# adapter use. Importing them eagerly here added ~110ms wall and ~33MB
# RSS to *every* CLI invocation (the plugin loader imports this module at
# ``model_tools`` import time, so ``hermes status``, ``hermes chat``, etc.
# all paid the cost even though they never instantiate the adapter).
#
# All names below are module globals that ``_load_google_modules()``
# rebinds on first call. The ``HttpError = Exception`` placeholder is
# important: ``except HttpError as exc:`` clauses elsewhere in this
# module bind the *current* module-global at try/except evaluation time,
# so as long as ``_load_google_modules()`` runs before any such
# ``try`` block executes (which it does — ``__init__`` calls it), the
# rebound real ``googleapiclient.errors.HttpError`` is what actually
# matches at runtime.
GOOGLE_CHAT_AVAILABLE: bool = False
httplib2: Any = None  # type: ignore
pubsub_v1: Any = None  # type: ignore
gax_exceptions: Any = None  # type: ignore
service_account: Any = None  # type: ignore
AuthorizedHttp: Any = None  # type: ignore
build_service: Any = None  # type: ignore
HttpError: Any = Exception  # type: ignore
MediaFileUpload: Any = None  # type: ignore

_google_modules_loaded: bool = False


def _load_google_modules() -> bool:
    """Lazily import the heavy google-cloud + googleapiclient stack.

    Idempotent. Returns True if the optional deps are installed and
    were successfully imported, False otherwise. On success, mutates
    the module globals so existing code using ``pubsub_v1``,
    ``service_account``, ``HttpError``, etc. transparently uses the
    real classes.

    Why deferred: the import chain pulls in google.cloud.pubsub_v1,
    googleapiclient, grpc, and friends — about 33MB RSS and 110ms wall
    on a fresh interpreter. Plugin discovery imports this module on
    every CLI invocation, even ones that never touch a gateway.
    """
    global GOOGLE_CHAT_AVAILABLE, _google_modules_loaded
    global httplib2, pubsub_v1, gax_exceptions, service_account
    global AuthorizedHttp, build_service, HttpError, MediaFileUpload
    if _google_modules_loaded:
        return GOOGLE_CHAT_AVAILABLE
    _google_modules_loaded = True
    try:
        import httplib2 as _httplib2
        from google.cloud import pubsub_v1 as _pubsub_v1
        from google.api_core import exceptions as _gax_exceptions
        from google.oauth2 import service_account as _service_account
        from google_auth_httplib2 import AuthorizedHttp as _AuthorizedHttp
        from googleapiclient.discovery import build as _build_service
        from googleapiclient.errors import HttpError as _HttpError
        from googleapiclient.http import MediaFileUpload as _MediaFileUpload
    except ImportError:
        GOOGLE_CHAT_AVAILABLE = False
        return False
    httplib2 = _httplib2
    pubsub_v1 = _pubsub_v1
    gax_exceptions = _gax_exceptions
    service_account = _service_account
    AuthorizedHttp = _AuthorizedHttp
    build_service = _build_service
    HttpError = _HttpError
    MediaFileUpload = _MediaFileUpload
    GOOGLE_CHAT_AVAILABLE = True
    return True

from gateway.config import Platform, PlatformConfig

# Trigger registration of the dynamic ``google_chat`` enum member at module
# import time.  ``_missing_()`` caches the pseudo-member in
# ``_value2member_map_`` *and* ``_member_map_``, so after this call
# ``Platform.GOOGLE_CHAT`` resolves via attribute access too.  Without this
# line, any code (including tests) that references ``Platform.GOOGLE_CHAT``
# before an adapter instance is constructed would hit ``AttributeError``.
# Built-ins avoid this because they have explicit enum members; plugin
# platforms earn the attribute by asking for it once.
Platform("google_chat")
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


# Pin the logger name to the legacy module path so operator log filters,
# grep aliases, and the gateway's bundled log views keep matching after
# the in-tree → plugin migration. ``__name__`` resolves to
# ``hermes_plugins.platforms__google_chat.adapter`` once the plugin
# loader namespaces this module, which would silently break every
# downstream log-monitor that greps for ``gateway.platforms.google_chat``.
logger = logging.getLogger("gateway.platforms.google_chat")


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
# OAuth flow in ``oauth.py`` (this plugin's helper module): the user grants the bot
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

# Outbound retry parameters. Google's Chat REST API returns transient 5xx
# and 429 occasionally — without a retry wrapper, single hiccups drop
# user-visible messages. Backoff stays bounded so a true outage is still
# surfaced quickly. Pattern lifted from PR #14965.
_RETRY_MAX_ATTEMPTS = 3
_RETRY_BASE_DELAY = 1.0
_RETRY_MAX_DELAY = 8.0
_RETRY_JITTER = 0.3
_RETRYABLE_HTTP_STATUSES = frozenset({429, 500, 502, 503, 504})


def _is_retryable_error(exc: BaseException) -> bool:
    """Classify outbound API errors as transient (retryable) vs permanent.

    Retries are applied to:
      - HTTP 429 (rate-limited)
      - HTTP 5xx (server errors)
      - Network/transport failures (timeout, connection reset, DNS)

    Authentication errors (401/403), client errors (4xx other than 429),
    and well-formed non-retryable failures are NOT retried — those
    indicate a misconfiguration or revoked token, not a hiccup.
    """
    # googleapiclient.errors.HttpError carries resp.status
    resp = getattr(exc, "resp", None)
    status = getattr(resp, "status", None)
    if isinstance(status, int):
        return status in _RETRYABLE_HTTP_STATUSES
    # Fallback heuristics for SSL/socket errors that don't carry an
    # HTTP status: text matches against common transport-layer wording.
    text = str(exc).lower()
    if "timeout" in text or "timed out" in text:
        return True
    if "connection" in text and ("reset" in text or "refused" in text or "aborted" in text):
        return True
    if "broken pipe" in text or "remote disconnected" in text:
        return True
    return False

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
    """Check if Google Chat optional dependencies are installed.

    Triggers the lazy import of the google-cloud + googleapiclient stack
    on first call. Subsequent calls hit the cached result. This is the
    canonical "are the deps available" probe used by the plugin registry
    and the adapter's own startup gate.
    """
    return _load_google_modules()


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


class _ThreadCountStore:
    """Per-(chat_id, thread_name) inbound message counter, persisted to disk.

    Drives the DM main-flow vs side-thread heuristic:

    - prev_count == 0 (first time we see this thread) → "main flow":
      Google Chat just auto-created a fresh thread for the user's
      top-level message. Treat it as part of the shared DM session;
      bot replies at top-level (no thread.name on outbound).
    - prev_count >= 1 (we've already seen this thread) → "side thread":
      user explicitly engaged a thread that's been around. Isolate
      session by thread, route bot reply into the same thread.

    Persistence is essential: without it, every gateway restart wipes
    counts and active side-threads silently demote to "main flow",
    which leaks main-flow context into the user's isolated thread
    (the bug Ramón reported across 4 iterations of the in-memory
    version).

    File format (JSON):
        {"<chat_id>": {"<thread_name>": <int_count>, ...}, ...}

    Failure modes are non-fatal: a missing or corrupt file resets to
    empty (logged as warning) so the adapter never crashes on disk
    issues. The next ``incr`` will write a fresh file.

    Save strategy: write-through after every ``incr``. The file is
    tiny (a few KB even for very active bots), so the simplicity of
    write-through outweighs the cost of debouncing for now.
    """

    def __init__(self, path: _Path):
        self._path = path
        self._counts: Dict[str, Dict[str, int]] = {}
        self._loaded = False

    def load(self) -> None:
        """Load counts from disk. Safe to call multiple times.

        Missing file → empty store. Corrupt JSON → empty store + warn.
        """
        self._loaded = True
        if not self._path.exists():
            self._counts = {}
            return
        try:
            raw = self._path.read_text()
            data = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError as exc:
            logger.warning(
                "[GoogleChat] thread-count store at %s is corrupt; "
                "starting fresh: %s",
                self._path, exc,
            )
            self._counts = {}
            return
        except OSError as exc:
            logger.warning(
                "[GoogleChat] could not read thread-count store at %s: %s",
                self._path, exc,
            )
            self._counts = {}
            return
        # Validate shape — anything off-schema gets dropped silently.
        clean: Dict[str, Dict[str, int]] = {}
        if isinstance(data, dict):
            for chat_id, threads in data.items():
                if not isinstance(chat_id, str) or not isinstance(threads, dict):
                    continue
                clean_threads: Dict[str, int] = {}
                for thread_name, count in threads.items():
                    if isinstance(thread_name, str) and isinstance(count, int):
                        clean_threads[thread_name] = count
                if clean_threads:
                    clean[chat_id] = clean_threads
        self._counts = clean

    def get(self, chat_id: str, thread_name: str) -> int:
        """Return the current count for (chat_id, thread_name), or 0."""
        return self._counts.get(chat_id, {}).get(thread_name, 0)

    def incr(self, chat_id: str, thread_name: str) -> int:
        """Increment count and write through to disk. Returns the
        PRE-increment value (the heuristic input — "have we seen this
        thread before this message?")."""
        chat_counts = self._counts.setdefault(chat_id, {})
        prev = chat_counts.get(thread_name, 0)
        chat_counts[thread_name] = prev + 1
        self._save()
        return prev

    def _save(self) -> None:
        """Atomic write of the counts dict to disk.

        Failure is non-fatal — log warning and continue. The in-memory
        counts stay consistent within the running process; only restart
        recovery is affected.
        """
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(self._path.suffix + ".tmp")
            tmp.write_text(json.dumps(self._counts, separators=(",", ":")))
            os.replace(tmp, self._path)
        except OSError as exc:
            logger.warning(
                "[GoogleChat] could not persist thread-count store to %s: %s",
                self._path, exc,
            )


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
        # ``Platform("google_chat")`` resolves via ``_missing_()`` → pseudo-member
        # cached in ``_value2member_map_``.  We deliberately do NOT add an enum
        # attribute to ``gateway.config.Platform`` — bundled platform plugins
        # are looked up by value, not attribute (matches Teams, IRC).
        super().__init__(config, Platform("google_chat"))
        # Trigger the deferred google-cloud + googleapiclient import here so
        # that any code path which constructs the adapter and then calls
        # methods directly (notably the test suite, which builds an adapter
        # and invokes ``_send_file`` / ``_create_message`` / etc. without
        # going through ``connect()``) sees real classes for ``MediaFileUpload``,
        # ``service_account``, ``HttpError``, and friends. The module-level
        # globals were previously eager-imported; making this lazy saved
        # ~110ms / ~33MB on every CLI invocation. Idempotent — pays the cost
        # exactly once per process.
        _load_google_modules()
        self._subscriber: Optional[Any] = None
        self._chat_api: Optional[Any] = None
        # User-authed Chat API client built lazily from the OAuth refresh
        # token persisted by the plugin's ``oauth.py`` helper. Required for
        # native ``media.upload`` (bot identity is rejected by that
        # endpoint).
        #
        # Multi-user mode: each user runs ``/setup-files`` ONCE in their
        # own DM and the resulting refresh token is stored under their
        # email. ``_send_file`` looks up the requesting user's email via
        # ``_last_sender_by_chat`` and uses THAT user's token, so when
        # User B asks for a file in B's DM the bot uploads as B (not as
        # whoever first set up files long ago).
        #
        # ``_user_credentials`` / ``_user_chat_api`` keep their old names
        # but now hold the LEGACY single-user token (if any) — used as a
        # last-ditch fallback when the requesting user has no per-user
        # token yet. Pre-multi-user installs continue to work unchanged.
        self._user_chat_api: Optional[Any] = None
        self._user_credentials: Optional[Any] = None
        # Per-email caches. Populated lazily by ``_get_user_chat_for_chat``.
        self._user_creds_by_email: Dict[str, Any] = {}
        self._user_chat_api_by_email: Dict[str, Any] = {}
        # chat_id → most-recent inbound sender's email. Populated in
        # ``_build_message_event`` whenever the inbound event carries a
        # non-empty ``sender.email``. Drives the per-user token lookup
        # in ``_send_file`` so the bot uploads as the user who triggered
        # the request, not as some other authorized user.
        self._last_sender_by_chat: Dict[str, str] = {}
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
        # Inbound message count per (chat_id, thread_name). Drives the
        # DM main-flow vs side-thread heuristic in _build_message_event
        # and the outbound thread routing in _resolve_thread_id.
        # Persisted to ${HERMES_HOME}/google_chat_thread_counts.json so
        # active side-threads survive gateway restarts (the bug that
        # made the in-memory version of this heuristic flaky for
        # multi-restart sessions).
        try:
            from hermes_constants import get_hermes_home as _get_hermes_home
            _hermes_home = _get_hermes_home()
        except (ModuleNotFoundError, ImportError):
            _hermes_home = _Path.home() / ".hermes"
        self._thread_count_store = _ThreadCountStore(
            _hermes_home / "google_chat_thread_counts.json"
        )
        # In-flight typing-card creates per chat_id. send_typing() reserves
        # an Event here BEFORE starting the API call so concurrent calls
        # from base.py's _keep_typing wait instead of duplicating cards.
        # Cleared in the create_and_record finally.
        self._typing_card_inflight: Dict[str, asyncio.Event] = {}
        # Orphaned typing cards (created by background tasks that lost a
        # race with send() / another concurrent create). Cleaned up at
        # end-of-turn by on_processing_complete via patch-to-empty so
        # they don't sit in the chat forever as "Hermes is thinking…".
        self._orphan_typing_messages: Dict[str, List[str]] = {}
        # FlowControl knobs (env-configurable).
        self._max_messages = int(os.getenv("GOOGLE_CHAT_MAX_MESSAGES", "1"))
        self._max_bytes = int(os.getenv("GOOGLE_CHAT_MAX_BYTES", str(16 * 1024 * 1024)))

    # ------------------------------------------------------------------
    # Configuration loading and validation
    # ------------------------------------------------------------------
    def _load_sa_credentials(self) -> Any:
        """Load Service Account credentials from env or config.extra,
        falling back to Application Default Credentials.

        Priority:
          1. Explicit ``extra['service_account_json']`` (path or inline JSON)
          2. ``GOOGLE_APPLICATION_CREDENTIALS`` env var (path)
          3. Application Default Credentials via ``google.auth.default()``
             — works on Cloud Run / GCE / GKE with a workload identity
             attached, or locally via ``gcloud auth application-default
             login``. Lets operators run the gateway in GCP without
             managing SA key files. Pattern lifted from PR #14965.
        """
        sa_path = (
            self.config.extra.get("service_account_json")
            or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        )
        if sa_path:
            # Inline JSON (rare, but supported).
            if sa_path.lstrip().startswith("{"):
                try:
                    info = json.loads(sa_path)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Inline SA JSON is not valid JSON: {exc}"
                    ) from exc
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

        # No explicit SA configured — try ADC. This is the Cloud Run / GCE
        # path; google-auth picks up the workload identity automatically.
        try:
            import google.auth as google_auth
        except ImportError:
            google_auth = None  # type: ignore[assignment]
        if google_auth is None:
            raise ValueError(
                "No Service Account credentials configured. Set "
                "GOOGLE_CHAT_SERVICE_ACCOUNT_JSON or GOOGLE_APPLICATION_CREDENTIALS, "
                "or install google-auth to use Application Default Credentials."
            )
        try:
            credentials, _project = google_auth.default(scopes=_CHAT_SCOPES)
        except Exception as exc:
            raise ValueError(
                "No Service Account credentials configured and Application "
                "Default Credentials are unavailable. Set "
                "GOOGLE_CHAT_SERVICE_ACCOUNT_JSON or run "
                "``gcloud auth application-default login``. "
                f"ADC error: {exc}"
            ) from exc
        logger.info(
            "[GoogleChat] No SA JSON configured; using Application "
            "Default Credentials"
        )
        return credentials

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
        # First call into the heavy google-cloud stack — trigger the lazy
        # import. ``_load_google_modules()`` is idempotent and rebinds the
        # module globals (``pubsub_v1``, ``service_account``, ``HttpError``,
        # …) used throughout this file. Anything that runs *before* this
        # call would see the placeholders, so connect() is the natural
        # gate.
        if not _load_google_modules():
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

        # Attempt to load LEGACY single-user OAuth credentials at startup.
        # In multi-user mode each user's token is loaded lazily by
        # ``_load_per_user_chat_api`` on first send. The legacy slot is
        # kept as a last-ditch fallback for pre-multi-user installs and
        # for groups where the asker has no per-user token yet. Failure
        # here is NON-fatal: text messaging continues to work; only
        # attachments degrade to a setup-instructions text notice.
        try:
            from .oauth import (
                load_user_credentials as _load_user_creds,
                build_user_chat_service as _build_user_chat,
                list_authorized_emails as _list_emails,
            )
            user_creds = await asyncio.to_thread(_load_user_creds)
            if user_creds is not None:
                self._user_credentials = user_creds
                self._user_chat_api = await asyncio.to_thread(
                    lambda: _build_user_chat(user_creds)
                )
                logger.info(
                    "[GoogleChat] Legacy user OAuth loaded — fallback "
                    "attachment delivery enabled"
                )
            authorized = await asyncio.to_thread(_list_emails)
            if authorized:
                logger.info(
                    "[GoogleChat] %d per-user OAuth tokens on disk: %s",
                    len(authorized), ", ".join(authorized),
                )
            elif user_creds is None:
                logger.info(
                    "[GoogleChat] No user OAuth tokens at setup — file "
                    "attachments will degrade to text-only fallback. "
                    "Each user runs /setup-files once in their own DM "
                    "to enable native attachments."
                )
        except Exception as exc:
            logger.warning(
                "[GoogleChat] User OAuth load failed (attachments will "
                "degrade to text-only fallback): %s",
                _redact_sensitive(str(exc)),
            )
            self._user_credentials = None
            self._user_chat_api = None

        # Load the persistent thread-count store so the side-thread
        # heuristic in _build_message_event survives gateway restarts.
        try:
            await asyncio.to_thread(self._thread_count_store.load)
        except Exception:
            logger.warning(
                "[GoogleChat] thread-count store load failed (treating "
                "all threads as fresh)", exc_info=True,
            )

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
    @staticmethod
    def _extract_message_payload(
        envelope: Dict[str, Any], ce_type: str = ""
    ) -> Optional[Tuple[Dict[str, Any], Dict[str, Any], str]]:
        """Detect Pub/Sub envelope format and return ``(message, space, format_name)``.

        Three known formats are accepted. Returns ``None`` when the envelope
        is unrecognized, is a non-MESSAGE event, or otherwise should be
        silently dropped.

        Format 1 — Workspace Add-ons (canonical, ce-type-driven)::

            {"chat": {"messagePayload": {"message": {...}, "space": {...}}}}

        Format 2 — Native Chat API Pub/Sub (alternative configuration where
        the Chat app publishes events directly without the Workspace
        Add-ons wrapper)::

            {"type": "MESSAGE", "message": {...}, "space": {...}}

        Format 3 — Relay / flat (a custom Cloud Run relay that flattens the
        Chat event into top-level fields)::

            {"event_type": "MESSAGE", "sender_email": "...", "text": "...",
             "space_name": "spaces/X", "thread_name": "spaces/X/threads/Y",
             "message_name": "spaces/X/messages/M.M"}

        For format 3 the helper synthesizes a Chat-API-shaped ``message``
        dict so downstream code (``_dispatch_message`` →
        ``_build_message_event``) can consume it without branching.
        """
        # Format 1: Workspace Add-ons. The chat block carries one of
        # messagePayload / membershipPayload / cardClickedPayload depending
        # on the ce-type. ``_on_pubsub_message`` handles the membership and
        # card branches before reaching this helper, so here we only accept
        # message payloads.
        chat_block = envelope.get("chat") or {}
        msg_payload_wrapper = chat_block.get("messagePayload") if chat_block else None
        if msg_payload_wrapper:
            msg = msg_payload_wrapper.get("message") or {}
            space = msg_payload_wrapper.get("space") or msg.get("space") or {}
            return msg, space, "workspace_addons"

        # Format 2: Native Chat API Pub/Sub. Detected by a top-level
        # ``message`` object plus a ``type`` field; only MESSAGE events
        # flow through here.
        if isinstance(envelope.get("message"), dict):
            if envelope.get("type", "") != "MESSAGE":
                return None
            msg = envelope["message"]
            space = envelope.get("space") or msg.get("space") or {}
            return msg, space, "native_chat_api"

        # Format 3: Relay / flat. A custom Cloud Run relay typically
        # forwards Chat events with this shape so the bot can run without
        # direct GCP credentials.
        if "event_type" in envelope or "sender_email" in envelope:
            if envelope.get("event_type", "MESSAGE") != "MESSAGE":
                return None
            sender_email = (envelope.get("sender_email") or "").strip()
            sender_display = (
                envelope.get("sender_display_name")
                or sender_email
                or "Unknown"
            )
            # The Chat resource name is unknown for relay events; synthesize
            # a stable surrogate from the sender email so dedup keys and
            # session IDs stay deterministic across redelivery.
            sender_name_surrogate = (
                "users/relay-"
                + (sender_email or "unknown").replace("@", "_at_").replace(".", "_")
            )
            text = envelope.get("text", "") or ""
            # Honor the relay's declared sender_type when present so the
            # downstream BOT self-filter (sender_type == "BOT") fires for
            # bot-originated messages forwarded by the relay. Hardcoding
            # "HUMAN" here meant the bot would re-process its own replies
            # if the relay forwarded them, and allowed a relay envelope to
            # impersonate any allowlisted user without ever being marked
            # as a bot. Default to "HUMAN" for backward compatibility when
            # the relay does not provide the field.
            #
            # Operator contract: the relay MUST forward sender.type from
            # the upstream Chat event as ``sender_type``. Relays that
            # forward bot replies as HUMAN (or omit the field) cannot be
            # distinguished from genuine humans here.
            sender_type_raw = (envelope.get("sender_type") or "HUMAN")
            sender_type = str(sender_type_raw).strip().upper() or "HUMAN"
            if sender_type not in {"HUMAN", "BOT"}:
                sender_type = "HUMAN"
            msg: Dict[str, Any] = {
                "name": envelope.get("message_name", "") or "",
                "sender": {
                    "name": sender_name_surrogate,
                    "email": sender_email,
                    "displayName": sender_display,
                    "type": sender_type,
                },
                "text": text,
                "argumentText": text,
            }
            thread_name = envelope.get("thread_name") or ""
            if thread_name:
                msg["thread"] = {"name": thread_name}
            space = {
                "name": envelope.get("space_name", "") or "",
                "spaceType": envelope.get("space_type", "SPACE"),
            }
            return msg, space, "relay_flat"

        return None

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
            extracted = self._extract_message_payload(envelope, ce_type)
            if extracted is None:
                logger.debug(
                    "[GoogleChat] Envelope did not match a known message format; "
                    "ce-type=%s, keys=%s", ce_type, list(envelope.keys())
                )
                message.ack()
                return

            msg, space, _fmt = extracted
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
                # The sender's email (user_id_alt) is the per-user OAuth
                # key — the bot stores this user's token at
                # ${HERMES_HOME}/google_chat_user_tokens/<sanitized>.json
                # so when User B asks for a file later in B's DM, B's
                # token gets used (not the first person who set up files).
                sender_email = (
                    event.source.user_id_alt
                    if event.source and event.source.user_id_alt
                    else None
                )
                handled = await self._handle_setup_files_command(
                    chat_id=event.source.chat_id,
                    thread_id=event.source.thread_id,
                    raw_text=text,
                    sender_email=sender_email,
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
        sender_email: Optional[str] = None,
    ) -> bool:
        """Run the in-chat OAuth setup flow for native attachment delivery.

        Returns ``True`` if the message was consumed (no agent dispatch),
        ``False`` if it should fall through.

        Multi-user mode: ``sender_email`` is the asker's identity, which
        is also the per-user OAuth key. ``status`` / ``start`` / ``revoke``
        / code-exchange all operate on THIS user's token slot. When
        ``sender_email`` is ``None`` (e.g. tests, or older inbound events
        without a populated email field) the handler falls back to the
        legacy single-user path so pre-multi-user installs keep working.

        Subcommands:
          /setup-files                  → show status + next step
          /setup-files start            → print OAuth URL
          /setup-files revoke           → revoke and delete stored token
          /setup-files <CODE_OR_URL>    → exchange auth code for token

        Pre-requisite: client_secret.json must already be on the host
        (one-time terminal step). The status reply tells the user how to
        do that if it's missing.
        """
        from . import oauth as oauth_helper

        # Normalize the email: lowercase + strip. The on-disk token path
        # is sanitized further inside the helper, but having the same
        # normalization at both ends keeps cache lookups consistent.
        sender_key = sender_email.strip().lower() if sender_email else None

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
            token_path = oauth_helper._token_path(sender_key)
            token_present = token_path.exists()
            creds = (
                oauth_helper.load_user_credentials(sender_key)
                if token_present else None
            )
            if creds is not None:
                who = sender_key or "shared (legacy)"
                await _reply(
                    "✅ Native attachment delivery is **active** for "
                    f"`{who}`.\n"
                    f"Token: `{token_path}`\n"
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
                    "python -m plugins.platforms.google_chat.oauth "
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
                    await asyncio.to_thread(
                        oauth_helper.get_auth_url, sender_key,
                    )
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
                    await asyncio.to_thread(oauth_helper.revoke, sender_key)
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
            # the setup-instructions text notice immediately. Scope the
            # eviction to the sender's slot — Bob revoking shouldn't
            # break Alice's per-user token nor wipe the shared legacy
            # fallback that other users may still depend on.
            if sender_key:
                self._user_creds_by_email.pop(sender_key, None)
                self._user_chat_api_by_email.pop(sender_key, None)
            else:
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
                    oauth_helper.exchange_auth_code, arg, sender_key,
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
            new_creds = await asyncio.to_thread(
                oauth_helper.load_user_credentials, sender_key,
            )
            if new_creds is not None:
                new_api = await asyncio.to_thread(
                    lambda: oauth_helper.build_user_chat_service(new_creds)
                )
                if sender_key:
                    self._user_creds_by_email[sender_key] = new_creds
                    self._user_chat_api_by_email[sender_key] = new_api
                else:
                    self._user_credentials = new_creds
                    self._user_chat_api = new_api
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
            f"at `{oauth_helper._token_path(sender_key)}` will be picked "
            f"up.\nHelper output:\n```\n{output}\n```"
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

        # Cache the asker's email per chat_id so _send_file can pick the
        # right per-user OAuth token when the agent later wants to send
        # an attachment in this conversation. Lower-cased so cache hits
        # match the sanitized token-file lookup.
        if sender_email and space_name:
            self._last_sender_by_chat[space_name] = sender_email.strip().lower()

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

        # Increment the persistent inbound count for this thread.
        # The PRE-increment value (==0 for the very first time we see
        # this thread, persisted across gateway restarts) drives the
        # main-flow-vs-side-thread heuristic below.
        prev_thread_count = 0
        if thread_name and space_name:
            prev_thread_count = self._thread_count_store.incr(
                space_name, thread_name
            )

        # Session-thread + outbound-thread routing for DMs:
        # - prev_count == 0  → first message in this thread. Google Chat
        #   creates a fresh thread per top-level message in the DM input
        #   box; treat as "main flow" so all top-level messages share
        #   one DM session and the user keeps continuity. The bot's
        #   reply ALSO must NOT thread with the user message — if we
        #   pass thread.name on outbound, Chat displays the pair as an
        #   expandable thread under the user's message instead of two
        #   adjacent top-level cards.
        # - prev_count >= 1  → user explicitly engaged a thread that
        #   already had messages (clicked "Reply in thread" on a prior
        #   message). Isolate session by chat_id+thread_id, AND keep
        #   the bot's reply inside that thread.
        #
        # For groups, threads ARE meaningful conversational containers
        # (Telegram forum / Discord thread parity); always isolate AND
        # always reply in-thread.
        if chat_type == "dm":
            is_side_thread = prev_thread_count > 0
            session_thread_id = thread_name if is_side_thread else None
            # Outbound thread cache: populated only when side-thread, so
            # _resolve_thread_id falls through to "no thread" on main
            # flow and the bot reply lands as a top-level sibling.
            if thread_name and space_name and is_side_thread:
                self._last_inbound_thread[space_name] = thread_name
            elif space_name:
                self._last_inbound_thread.pop(space_name, None)
        else:
            session_thread_id = thread_name
            # Groups always reply in-thread.
            if thread_name and space_name:
                self._last_inbound_thread[space_name] = thread_name

        source = self.build_source(
            chat_id=space_name,
            chat_name=space.get("displayName") or space.get("name") or "",
            chat_type=chat_type,
            # ``user_id`` is the canonical identity used by allowlists,
            # session keys, and audit. Operators configure
            # ``GOOGLE_CHAT_ALLOWED_USERS`` with email addresses (the
            # value Google Chat surfaces in its UI), so the email is
            # the natural canonical id. The Chat resource name
            # ``users/{id}`` moves to ``user_id_alt`` for traceability
            # and Chat-API operations that need it. Falls back to the
            # resource name when sender has no email (rare — bot-to-bot
            # or system events). Pattern lifted from PR #14965.
            user_id=(sender_email or sender_name),
            user_name=sender_display,
            thread_id=session_thread_id,
            user_id_alt=(sender_name or None),
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
            # Convert standard Markdown emitted by the LLM to Chat's dialect
            # and strip invisible Unicode that renders as tofu (□). Runs
            # BEFORE chunking so the size limit applies to the rendered
            # form, not the source markdown.
            chunks = self._chunk_text(self.format_message(content))
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

    # ------------------------------------------------------------------
    # Outbound formatting
    # ------------------------------------------------------------------
    # Invisible Unicode codepoints that render as tofu (□) in Google
    # Chat's restricted font stack. ZWJ/ZWNJ/ZWS are the glue inside
    # composite emoji and bidirectional text; Variation Selectors
    # control text-vs-emoji presentation but Chat ignores them and
    # often shows a blank box. Pattern lifted from PR #14965.
    _INVISIBLE_RE = re.compile(
        "["
        "​"          # Zero-Width Space
        "‌"          # Zero-Width Non-Joiner
        "‍"          # Zero-Width Joiner (ZWJ)
        "‎‏"    # LTR / RTL marks
        "⁠"          # Word Joiner
        "﻿"          # BOM / Zero-Width No-Break Space
        "︀-️"   # Variation Selectors 1-16 (VS1–VS16)
        "\U000e0100-\U000e01ef"  # Variation Selectors 17-256
        "]"
    )

    @classmethod
    def format_message(cls, content: str) -> str:
        """Convert standard Markdown to Google Chat's formatting dialect.

        Google Chat renders a small subset: ``*bold*``, ``_italic_``,
        ``~strikethrough~``, fenced/inline code. Standard Markdown
        constructs (``**bold**``, ``# headers``, ``[text](url)``) do
        not render and need conversion before they reach Chat.

        Code blocks (fenced AND inline) are protected from transformation
        via placeholder substitution so backticks-wrapped content with
        literal asterisks or brackets stays intact. Invisible Unicode
        codepoints that render as tofu in Chat's restricted font stack
        are stripped at the end. Empty/None input passes through.

        Pattern lifted from PR #14965.
        """
        if not content:
            return content

        text = content
        placeholders: Dict[str, str] = {}
        counter = [0]

        def _ph(value: str) -> str:
            key = f"\x00GC{counter[0]}\x00"
            counter[0] += 1
            placeholders[key] = value
            return key

        # Protect fenced and inline code blocks from transformation.
        # Fenced blocks first (``` ... ```), then inline code (`...`).
        text = re.sub(
            r"(```(?:[^\n]*\n)?[\s\S]*?```)",
            lambda m: _ph(m.group(0)),
            text,
        )
        text = re.sub(r"(`[^`]+`)", lambda m: _ph(m.group(0)), text)

        # Headers (## Title) → *Title* (Chat has no header support).
        text = re.sub(
            r"^#{1,6}\s+(.+)$",
            lambda m: _ph(f"*{m.group(1).strip()}*"),
            text,
            flags=re.MULTILINE,
        )

        # Bold+italic: ***text*** → *_text_*
        text = re.sub(
            r"\*\*\*(.+?)\*\*\*",
            lambda m: _ph(f"*_{m.group(1)}_*"),
            text,
        )

        # Bold: **text** → *text* (Chat uses single asterisks).
        text = re.sub(
            r"\*\*(.+?)\*\*",
            lambda m: _ph(f"*{m.group(1)}*"),
            text,
        )

        # Markdown links [text](url) → <url|text> (Slack-style angle-bracket).
        text = re.sub(
            r"\[([^\]]+)\]\(([^)]+)\)",
            lambda m: _ph(f"<{m.group(2)}|{m.group(1)}>"),
            text,
        )

        # Strip invisible Unicode that renders as tofu.
        text = cls._INVISIBLE_RE.sub("", text)

        # Collapse double spaces left over from stripped chars.
        text = re.sub(r"  +", " ", text)

        # Restore protected regions.
        for key, value in placeholders.items():
            text = text.replace(key, value)

        return text

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

    async def _call_with_retry(
        self,
        sync_fn: Callable[[], Any],
        *,
        op_name: str = "chat-api-call",
    ) -> Any:
        """Run ``sync_fn`` in a thread with bounded retry + jittered backoff.

        Wraps a sync Chat API call (typically a ``.execute()``) so transient
        429/5xx/timeout failures don't drop user-visible messages. Permanent
        failures (auth, client errors, validation) bubble up on the first
        attempt — see :func:`_is_retryable_error`. Cancellation propagates
        immediately, no extra retries after a CancelledError.

        Pattern lifted from PR #14965.
        """
        delay = _RETRY_BASE_DELAY
        last_exc: Optional[BaseException] = None
        for attempt in range(1, _RETRY_MAX_ATTEMPTS + 1):
            try:
                return await asyncio.to_thread(sync_fn)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                last_exc = exc
                retryable = _is_retryable_error(exc)
                if not retryable or attempt >= _RETRY_MAX_ATTEMPTS:
                    raise
                jitter = delay * _RETRY_JITTER * random.random()
                wait = min(delay + jitter, _RETRY_MAX_DELAY + _RETRY_JITTER)
                logger.warning(
                    "[GoogleChat] %s attempt %d/%d failed (%s); "
                    "retrying in %.2fs",
                    op_name, attempt, _RETRY_MAX_ATTEMPTS,
                    _redact_sensitive(str(exc)), wait,
                )
                try:
                    await asyncio.sleep(wait)
                except asyncio.CancelledError:
                    raise
                delay = min(delay * 2, _RETRY_MAX_DELAY)
        # Defensive — the loop above always either returns or re-raises.
        if last_exc is not None:
            raise last_exc
        raise RuntimeError(f"{op_name}: retry loop exited without result")

    async def _create_message(
        self, chat_id: str, body: Dict[str, Any]
    ) -> SendResult:
        """POST spaces/{space}/messages via REST, returning SendResult.

        When ``body`` carries ``thread.name``, we MUST pass
        ``messageReplyOption=REPLY_MESSAGE_FALLBACK_TO_NEW_THREAD`` —
        otherwise Google Chat silently ignores ``thread.name`` and
        creates a new thread anyway. From the official docs:

            "Default. Starts a new thread. Using this option ignores
             any thread ID or threadKey that's included."

        See https://developers.google.com/workspace/chat/api/reference/rest/v1/spaces.messages/create
        """
        kwargs: Dict[str, Any] = {"parent": chat_id, "body": body}
        thread_meta = body.get("thread") or {}
        if thread_meta.get("name"):
            # FALLBACK_TO_NEW_THREAD: try the requested thread; if Chat
            # can't route there (e.g. thread no longer exists), create a
            # new one rather than erroring. Safer than REPLY_MESSAGE_OR_FAIL
            # for a chat-bot context where stale thread names are rare
            # but possible.
            kwargs["messageReplyOption"] = "REPLY_MESSAGE_FALLBACK_TO_NEW_THREAD"

        def _do_create() -> Dict[str, Any]:
            return (
                self._chat_api.spaces()
                .messages()
                .create(**kwargs)
                .execute(http=self._new_authed_http())
            )

        resp = await self._call_with_retry(_do_create, op_name="messages.create")
        # Track outbound destination thread in the persistent count store
        # so a future user "Reply in thread" on the bot's message resolves
        # to a known thread (prev_count >= 1 → side thread). Without
        # this, threads created by the bot's own outbound look fresh
        # the first time the user engages them, and the heuristic
        # incorrectly classifies the engagement as main-flow → bot
        # replies at top-level instead of in the thread.
        resp_thread = (resp.get("thread") or {}).get("name") or ""
        if chat_id and resp_thread:
            try:
                self._thread_count_store.incr(chat_id, resp_thread)
            except Exception:
                logger.debug(
                    "[GoogleChat] outbound thread-count incr failed",
                    exc_info=True,
                )
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
        same way send() does.

        IMPORTANT — cancellation safety:
        ``base.py``'s ``_keep_typing`` calls this through
        ``asyncio.wait_for(send_typing, timeout=1.5)``. When the
        create-API call takes longer than 1.5s, ``wait_for`` cancels
        ``send_typing`` mid-flight — but the underlying ``asyncio.to_thread``
        keeps running and creates a card in Chat that we have NO way to
        track (the storage line never runs). Next ``_keep_typing`` tick
        sees an empty slot and creates a SECOND card. Result: one orphan
        "Hermes is thinking…" stuck in chat forever, plus one card that
        gets patched into the reply.

        Fix: reserve the slot with an in-flight ``Event``, run the
        create in a background task, and ``await asyncio.shield`` it.
        Cancellation of THIS coroutine no longer cancels the create —
        the task runs to completion and the msg_id lands in the slot
        regardless.
        """
        # Already have a card (real msg_id, sentinel, or in-flight) — bail.
        if chat_id in self._typing_messages:
            return
        if chat_id in self._typing_card_inflight:
            # Another create is already running for this chat. Wait for
            # it to finish so we honor the contract "if called, the card
            # is up by the time we return". Bounded wait — if the
            # background task is stuck, _keep_typing will retry.
            try:
                await asyncio.wait_for(
                    self._typing_card_inflight[chat_id].wait(),
                    timeout=5.0,
                )
            except (asyncio.TimeoutError, KeyError):
                pass
            return

        thread_id = self._resolve_thread_id(
            reply_to=None, metadata=metadata, chat_id=chat_id,
        )
        body: Dict[str, Any] = {"text": "Hermes is thinking…"}
        if thread_id:
            body["thread"] = {"name": thread_id}

        completed = asyncio.Event()
        self._typing_card_inflight[chat_id] = completed

        async def _create_and_record() -> None:
            try:
                result = await self._create_message(chat_id, body)
                if result.success and result.message_id:
                    # Only overwrite the slot if nothing else has claimed it
                    # in the meantime (e.g. send() racing ahead of us).
                    if chat_id not in self._typing_messages:
                        self._typing_messages[chat_id] = result.message_id
                    else:
                        # Slot already populated — likely send() patched
                        # something or another create completed first.
                        # Our card is ORPHANED here, but at least it's a
                        # known orphan we can clean up at end of turn.
                        # Track for cleanup by on_processing_complete.
                        self._orphan_typing_messages.setdefault(
                            chat_id, []
                        ).append(result.message_id)
            except Exception:
                logger.debug(
                    "[GoogleChat] send_typing background create failed",
                    exc_info=True,
                )
            finally:
                self._typing_card_inflight.pop(chat_id, None)
                completed.set()

        task = asyncio.create_task(_create_and_record())
        # Shield the task from cancellation of our awaiter. If
        # _keep_typing's wait_for times out, our coroutine is cancelled
        # but the task continues in the background — so the msg_id
        # eventually lands in the slot even when the API call is slow.
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            # The shielded task keeps running. Re-raise so the caller's
            # cancellation semantics are preserved.
            raise

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
        """Reap typing card(s) after the message-handling cycle ends.

        SUCCESS: ``send()`` set the SENTINEL after patching. Pop it.

        FAILURE / CANCELLED: ``send()`` may not have run, leaving a real
        ``message_name`` in the slot. Patching the card to a final state
        (``"(interrupted)"``) avoids the tombstone that ``messages.delete``
        would create. If ``send()`` did run (e.g. base.py error-send branch
        patched it), the slot holds the SENTINEL — pop and exit.

        Orphan cards: when a background ``send_typing`` task creates a
        card AFTER ``send()`` already populated the slot (race window
        when the API call takes longer than _keep_typing's wait_for
        timeout), the orphan id is stashed in ``self._orphan_typing_messages``.
        Patch each orphan with an empty-ish marker so the user doesn't
        see "Hermes is thinking…" stuck forever.
        """
        if event.source is None:
            return
        chat_id = event.source.chat_id
        try:
            current = self._typing_messages.pop(chat_id, None)
            if current and current != _TYPING_CONSUMED_SENTINEL:
                # Real message_name still in slot — send() never ran. Patch
                # with a benign final state instead of deleting (no tombstone).
                label = (
                    "(interrupted)" if outcome == ProcessingOutcome.CANCELLED
                    else "(no reply)"
                )
                try:
                    await self._patch_message(current, {"text": label})
                except Exception:
                    logger.debug(
                        "[GoogleChat] on_processing_complete patch fallback failed",
                        exc_info=True,
                    )
            # Reap orphan typing cards (background creates that lost a
            # race with send()). Patch them to a single dot so they
            # gracefully retire — the user already saw the real reply
            # in another card, this one is just visual noise to clear.
            orphans = self._orphan_typing_messages.pop(chat_id, [])
            for orphan_id in orphans:
                try:
                    await self._patch_message(orphan_id, {"text": "·"})
                except Exception:
                    logger.debug(
                        "[GoogleChat] orphan typing-card patch failed: %s",
                        orphan_id, exc_info=True,
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
    # ``plugins/platforms/google_chat/oauth.py`` for the helper
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

    _LEGACY_USER_IDENTITY = "__legacy__"

    async def _load_per_user_chat_api(self, email: str) -> Optional[Any]:
        """Get (or build + cache) a user-authed Chat client for ``email``.

        Hits ``self._user_chat_api_by_email`` first; on miss, loads the
        per-user token from disk, refreshes if needed, builds an API
        client, and caches both. Refresh failures evict the slot so the
        next request goes back through the disk path (and ultimately the
        text-notice fallback if the user has revoked).
        """
        from .oauth import (
            load_user_credentials as _load,
            build_user_chat_service as _build,
            refresh_or_none as _refresh,
        )

        cached_api = self._user_chat_api_by_email.get(email)
        cached_creds = self._user_creds_by_email.get(email)
        if cached_api is not None and cached_creds is not None:
            try:
                refreshed = await asyncio.to_thread(_refresh, cached_creds, email)
            except Exception:
                logger.debug(
                    "[GoogleChat] cached per-user refresh raised", exc_info=True,
                )
                refreshed = None
            if refreshed is None:
                self._user_chat_api_by_email.pop(email, None)
                self._user_creds_by_email.pop(email, None)
                return None
            self._user_creds_by_email[email] = refreshed
            return cached_api

        try:
            creds = await asyncio.to_thread(_load, email)
            if creds is None:
                return None
            api = await asyncio.to_thread(lambda: _build(creds))
        except Exception:
            logger.debug(
                "[GoogleChat] per-user creds load/build failed for %s",
                email, exc_info=True,
            )
            return None

        self._user_creds_by_email[email] = creds
        self._user_chat_api_by_email[email] = api
        return api

    async def _acquire_user_chat_api(
        self, sender_email: Optional[str]
    ) -> Tuple[Optional[Any], Optional[str]]:
        """Resolve the user-authed Chat client for an outbound attachment.

        Lookup order:
          1. Per-user token for ``sender_email`` — the asker's identity.
          2. Legacy single-user fallback (``self._user_chat_api``) for
             pre-multi-user installs.
          3. None — caller posts the setup-instructions text notice.

        Returns ``(client, identity_label)`` where ``identity_label`` is
        the sanitized email or the literal ``"__legacy__"`` sentinel.
        ``_invalidate_user_creds`` uses the label to evict the right slot
        on auth failure.
        """
        if sender_email:
            api = await self._load_per_user_chat_api(sender_email)
            if api is not None:
                return api, sender_email

        if self._user_chat_api is not None:
            try:
                from .oauth import (
                    refresh_or_none as _refresh,
                )
                refreshed = await asyncio.to_thread(
                    _refresh, self._user_credentials, None,
                )
            except Exception:
                logger.debug(
                    "[GoogleChat] legacy creds refresh raised", exc_info=True,
                )
                refreshed = None
            if refreshed is None:
                logger.warning(
                    "[GoogleChat] legacy user-OAuth refresh returned None — "
                    "evicting fallback creds"
                )
                self._user_credentials = None
                self._user_chat_api = None
                return None, None
            self._user_credentials = refreshed
            return self._user_chat_api, self._LEGACY_USER_IDENTITY

        return None, None

    def _invalidate_user_creds(self, identity: Optional[str]) -> None:
        """Drop creds for ``identity`` after an auth failure.

        ``identity`` comes from ``_acquire_user_chat_api`` — either the
        sender email (per-user slot) or ``__legacy__`` for the fallback
        slot. None is a no-op.
        """
        if not identity:
            return
        if identity == self._LEGACY_USER_IDENTITY:
            self._user_credentials = None
            self._user_chat_api = None
            return
        self._user_creds_by_email.pop(identity, None)
        self._user_chat_api_by_email.pop(identity, None)

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
        BOTH calls go through a user-authed Chat API client — the
        SA-authed client is rejected by ``media.upload`` regardless of
        scopes.

        Multi-user routing: the bot looks up the most recent inbound
        sender for this ``chat_id`` and uses THAT user's stored OAuth
        token. Falls back to a legacy single-user token when present
        (for pre-multi-user installs), and to a setup-instructions text
        notice when neither is available.

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

        sender_email = self._last_sender_by_chat.get(chat_id)
        chat_api, identity = await self._acquire_user_chat_api(sender_email)

        # No user OAuth → can't upload natively. Surface clear setup
        # instructions in chat instead of silently failing.
        if chat_api is None:
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

        def _upload() -> Dict[str, Any]:
            media = MediaFileUpload(path, mimetype=mime, resumable=False)
            return (
                chat_api.media()
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
                    "[GoogleChat] media.upload auth failure for identity=%s "
                    "(token revoked or scope missing) — falling back to "
                    "text notice. Status=%s", identity, status,
                )
                self._invalidate_user_creds(identity)
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
        # uploading principal). messageReplyOption is required for the
        # thread.name in body to actually be honored — see
        # _create_message docstring for the API quirk.
        create_kwargs: Dict[str, Any] = {"parent": chat_id, "body": body}
        if thread_id:
            create_kwargs["messageReplyOption"] = (
                "REPLY_MESSAGE_FALLBACK_TO_NEW_THREAD"
            )

        def _create_with_attachment() -> Dict[str, Any]:
            return (
                chat_api.spaces()
                .messages()
                .create(**create_kwargs)
                .execute()
            )

        try:
            resp = await asyncio.to_thread(_create_with_attachment)
            # Track outbound destination thread (see _create_message
            # comment for why — same reasoning applies to the
            # user-OAuth attachment path).
            resp_thread = (resp.get("thread") or {}).get("name") or ""
            if chat_id and resp_thread:
                try:
                    self._thread_count_store.incr(chat_id, resp_thread)
                except Exception:
                    logger.debug(
                        "[GoogleChat] outbound thread-count incr failed",
                        exc_info=True,
                    )
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


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------


def _validate_config(config: PlatformConfig) -> bool:
    """Plugin-side config gate: require both Pub/Sub project and subscription.

    Mirrors the legacy dispatch entry in ``gateway/config.py`` so the
    registry can decide whether the platform is configured without
    importing the legacy table.
    """
    extra = getattr(config, "extra", {}) or {}
    return bool(
        extra.get("project_id") and extra.get("subscription_name")
    )


def _check_for_registry() -> bool:
    """``check_fn`` for the platform registry pass — stricter than the
    deps-only ``check_google_chat_requirements``.

    The registry pass at ``gateway/config.py:_apply_env_overrides`` adds
    the platform to ``cfg.platforms`` whenever ``check_fn`` returns True.
    For backward compat with the pre-plugin behavior, we ALSO require
    the minimum Pub/Sub env vars so an unconfigured user doesn't
    accidentally see ``google_chat`` enabled. This matches the legacy
    ``if gc_project and gc_subscription`` gate.
    """
    if not check_google_chat_requirements():
        return False
    project = (
        os.getenv("GOOGLE_CHAT_PROJECT_ID")
        or os.getenv("GOOGLE_CLOUD_PROJECT")
    )
    subscription = (
        os.getenv("GOOGLE_CHAT_SUBSCRIPTION_NAME")
        or os.getenv("GOOGLE_CHAT_SUBSCRIPTION")
    )
    return bool(project and subscription)


def _is_connected(config: PlatformConfig) -> bool:
    """``GatewayConfig.get_connected_platforms()`` polls this."""
    return bool(getattr(config, "enabled", False)) and _validate_config(config)


def _env_enablement() -> Optional[Dict[str, Any]]:
    """Seed ``PlatformConfig.extra`` from env vars during
    ``_apply_env_overrides``.

    The registry's env-enablement hook is called BEFORE the adapter is
    constructed, so ``gateway status`` and ``get_connected_platforms()``
    reflect env-only configuration without instantiating the Pub/Sub client.
    Returns ``None`` when the required Pub/Sub project/subscription aren't
    set; the caller then skips auto-enabling the platform.

    The special ``home_channel`` key in the returned dict is handled by the
    core hook — it becomes a proper ``HomeChannel`` dataclass on the
    ``PlatformConfig`` rather than being merged into ``extra``.
    """
    project = (
        os.getenv("GOOGLE_CHAT_PROJECT_ID")
        or os.getenv("GOOGLE_CLOUD_PROJECT")
    )
    subscription = (
        os.getenv("GOOGLE_CHAT_SUBSCRIPTION_NAME")
        or os.getenv("GOOGLE_CHAT_SUBSCRIPTION")
    )
    if not (project and subscription):
        return None
    seed: Dict[str, Any] = {
        "project_id": project,
        "subscription_name": subscription,
    }
    sa_json = (
        os.getenv("GOOGLE_CHAT_SERVICE_ACCOUNT_JSON")
        or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    )
    if sa_json:
        seed["service_account_json"] = sa_json
    home = os.getenv("GOOGLE_CHAT_HOME_CHANNEL")
    if home:
        seed["home_channel"] = {
            "chat_id": home,
            "name": os.getenv("GOOGLE_CHAT_HOME_CHANNEL_NAME", "Home"),
        }
    return seed


def interactive_setup() -> None:
    """Walk the user through Google Chat configuration via ``hermes setup``.

    The setup wizard at ``hermes_cli/gateway.py`` calls this for plugin
    platforms instead of using the in-tree ``_PLATFORMS`` data block. The
    flow mirrors the in-tree built-ins: print the GCP setup instructions,
    prompt for env vars, persist them to ``~/.hermes/.env`` so the next
    gateway restart picks them up.
    """
    from hermes_cli.cli_output import (
        print_info,
        print_success,
        print_warning,
        prompt,
        prompt_yes_no,
    )
    from hermes_cli.config import get_env_value, save_env_value

    existing_sub = get_env_value("GOOGLE_CHAT_SUBSCRIPTION_NAME")
    if existing_sub:
        print_info(f"Google Chat: already configured (subscription: {existing_sub})")
        if not prompt_yes_no("Reconfigure Google Chat?", False):
            return

    print_info("Google Chat needs a GCP project, a Pub/Sub topic + subscription,")
    print_info("and a Service Account with Pub/Sub Subscriber on the subscription.")
    print_info("Walkthrough:")
    print_info("  1. Create or select a GCP project; enable Google Chat API + Cloud Pub/Sub API.")
    print_info("  2. Create a Service Account (no project-level IAM role needed).")
    print_info("  3. Create a Pub/Sub topic (e.g. hermes-chat-events) and a Pull subscription.")
    print_info("  4. On the TOPIC: add chat-api-push@system.gserviceaccount.com as Pub/Sub Publisher.")
    print_info("  5. On the SUBSCRIPTION: grant your Service Account Pub/Sub Subscriber.")
    print_info("  6. Download the Service Account JSON key.")
    print_info("  7. Google Chat API console → Configuration: connection = Cloud Pub/Sub,")
    print_info("     point at the topic, enable 1:1 + group, restrict visibility.")
    print_info("  8. Install the bot in a space (fires ADDED_TO_SPACE and resolves its user_id).")
    print_info("")
    print_info("Full guide: website/docs/user-guide/messaging/google_chat.md")
    print_info("")

    project = prompt(
        "GCP project ID (e.g. my-project)",
        default=get_env_value("GOOGLE_CHAT_PROJECT_ID") or "",
    )
    if not project:
        print_warning("Project ID is required — skipping Google Chat setup")
        return
    save_env_value("GOOGLE_CHAT_PROJECT_ID", project.strip())

    subscription = prompt(
        "Pub/Sub subscription (projects/<proj>/subscriptions/<sub>)",
        default=get_env_value("GOOGLE_CHAT_SUBSCRIPTION_NAME") or "",
    )
    if not subscription:
        print_warning("Subscription is required — skipping Google Chat setup")
        return
    save_env_value("GOOGLE_CHAT_SUBSCRIPTION_NAME", subscription.strip())

    sa_path = prompt(
        "Path to Service Account JSON (or inline JSON)",
        default=get_env_value("GOOGLE_CHAT_SERVICE_ACCOUNT_JSON") or "",
        password=True,
    )
    if sa_path:
        save_env_value("GOOGLE_CHAT_SERVICE_ACCOUNT_JSON", sa_path.strip())

    if prompt_yes_no("Restrict access to specific users? (recommended)", True):
        allowed = prompt(
            "Allowed user emails (comma-separated)",
            default=get_env_value("GOOGLE_CHAT_ALLOWED_USERS") or "",
        )
        if allowed:
            save_env_value("GOOGLE_CHAT_ALLOWED_USERS", allowed.replace(" ", ""))
            print_success("Allowlist configured")
        else:
            save_env_value("GOOGLE_CHAT_ALLOWED_USERS", "")
    else:
        save_env_value("GOOGLE_CHAT_ALLOW_ALL_USERS", "true")
        print_warning("⚠️  Open access — anyone who can DM the bot can command it.")

    home = prompt(
        "Home space for cron/notification delivery (e.g. spaces/AAAA, or empty)",
        default=get_env_value("GOOGLE_CHAT_HOME_CHANNEL") or "",
    )
    if home:
        save_env_value("GOOGLE_CHAT_HOME_CHANNEL", home.strip())

    print()
    print_success("Google Chat configuration saved to ~/.hermes/.env")
    print_info("Restart the gateway: hermes gateway restart")


# Strict resource-name pattern.  ``spaces/<id>`` and ``users/<id>`` must
# only contain Google Chat's documented character set; anything else
# means a tampered chat_id trying to break out of the REST URL path
# (path traversal, ``?`` query injection, ``#`` fragment truncation).
_GCHAT_CHAT_ID_RE = re.compile(r"^(?:spaces|users)/[A-Za-z0-9_-]+$")


async def _standalone_send(
    pconfig,
    chat_id: str,
    message: str,
    *,
    thread_id: Optional[str] = None,
    media_files: Optional[List[str]] = None,
    force_document: bool = False,
) -> Dict[str, Any]:
    """POST a single Google Chat message via the REST API without the SDK.

    Used by ``tools/send_message_tool._send_via_adapter`` when the gateway
    runner is not in this process (e.g. ``hermes cron`` running as a
    separate process from ``hermes gateway``).  Without this hook,
    ``deliver=google_chat`` cron jobs fail with ``No live adapter for
    platform``.

    Configuration: requires service-account credentials via
    ``GOOGLE_CHAT_SERVICE_ACCOUNT_JSON``, ``GOOGLE_APPLICATION_CREDENTIALS``,
    or Application Default Credentials, and a space resource name as
    ``chat_id`` (e.g. ``spaces/AAAA-BBBB`` or ``users/<id>``).

    Security: ``chat_id`` is validated against the documented Google Chat
    resource-name character set before substitution into the REST URL so
    a tampered value cannot path-traverse or query-inject.

    ``media_files`` and ``force_document`` are accepted for signature
    parity but are not implemented for the standalone path; messages with
    attachments send as text-only.  The live adapter handles attachments.
    """
    if not chat_id:
        return {"error": "Google Chat standalone send: chat_id (space resource) is required"}
    if not _GCHAT_CHAT_ID_RE.match(chat_id):
        return {"error": (
            f"Google Chat standalone send: chat_id {chat_id!r} must match "
            f"'spaces/<id>' or 'users/<id>' with only [A-Za-z0-9_-] in the id"
        )}
    if thread_id is not None and not re.match(r"^spaces/[A-Za-z0-9_-]+/threads/[A-Za-z0-9_-]+$", thread_id):
        return {"error": (
            f"Google Chat standalone send: thread_id {thread_id!r} must match "
            f"'spaces/<id>/threads/<id>'"
        )}

    extra = getattr(pconfig, "extra", {}) or {}
    sa_value = (
        extra.get("service_account_json")
        or os.getenv("GOOGLE_CHAT_SERVICE_ACCOUNT_JSON")
        or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    )

    if service_account is None:
        return {"error": "Google Chat standalone send: google-auth not installed"}

    try:
        from google.auth.transport.requests import Request as _GoogleAuthRequest
    except Exception as e:
        return {"error": f"Google Chat standalone send: google-auth import failed: {e}"}

    try:
        if sa_value:
            stripped = sa_value.lstrip()
            if stripped.startswith("{"):
                try:
                    info = json.loads(sa_value)
                except json.JSONDecodeError as exc:
                    return {"error": f"Google Chat standalone send: inline SA JSON is invalid: {exc}"}
                creds = service_account.Credentials.from_service_account_info(info, scopes=_CHAT_SCOPES)
            else:
                if not os.path.exists(sa_value):
                    return {"error": f"Google Chat standalone send: SA JSON file not found at {sa_value}"}
                try:
                    with open(sa_value, "r", encoding="utf-8") as fh:
                        info = json.load(fh)
                except json.JSONDecodeError as exc:
                    return {"error": f"Google Chat standalone send: SA JSON file is invalid: {exc}"}
                creds = service_account.Credentials.from_service_account_info(info, scopes=_CHAT_SCOPES)
        else:
            try:
                import google.auth as _google_auth
            except ImportError:
                return {"error": (
                    "Google Chat standalone send: no SA credentials configured "
                    "and google-auth is not installed for ADC fallback"
                )}
            try:
                creds, _project = _google_auth.default(scopes=_CHAT_SCOPES)
            except Exception as exc:
                return {"error": (
                    f"Google Chat standalone send: no SA credentials configured "
                    f"and Application Default Credentials are unavailable: {exc}"
                )}
    except asyncio.CancelledError:
        raise
    except Exception as e:
        return {"error": f"Google Chat standalone send: credential load failed: {e}"}

    # Bound the synchronous urllib3-backed token refresh so a hung Google
    # STS endpoint cannot stall the cron scheduler indefinitely.
    try:
        await asyncio.wait_for(
            asyncio.to_thread(creds.refresh, _GoogleAuthRequest()),
            timeout=10.0,
        )
    except asyncio.TimeoutError:
        return {"error": "Google Chat standalone send: token refresh timed out"}
    except asyncio.CancelledError:
        raise
    except Exception as e:
        return {"error": f"Google Chat standalone send: token refresh failed: {e}"}

    token = getattr(creds, "token", None)
    if not token:
        return {"error": "Google Chat standalone send: refreshed credentials have no token"}

    body: Dict[str, Any] = {"text": message}
    if thread_id:
        body["thread"] = {"name": thread_id}

    url = f"https://chat.googleapis.com/v1/{chat_id}/messages"
    try:
        import aiohttp as _aiohttp
    except ImportError:
        return {"error": "Google Chat standalone send: aiohttp not installed"}

    try:
        async with _aiohttp.ClientSession(timeout=_aiohttp.ClientTimeout(total=30.0)) as session:
            async with session.post(
                url,
                json=body,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
            ) as resp:
                if resp.status >= 400:
                    text = await resp.text()
                    return {"error": (
                        f"Google Chat standalone send: API returned "
                        f"{resp.status}: {text[:300]}"
                    )}
                payload = await resp.json()
        return {
            "success": True,
            "message_id": payload.get("name"),
        }
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.debug("Google Chat standalone send raised", exc_info=True)
        return {"error": f"Google Chat standalone send failed: {e}"}


def register(ctx) -> None:
    """Plugin entry point — called by the Hermes plugin system at startup.

    Registers the Google Chat adapter under the ``google_chat`` name.
    The gateway's ``_create_adapter`` consults the platform registry
    BEFORE its built-in if/elif chain, so this registration is what
    drives adapter creation at runtime.
    """
    ctx.register_platform(
        name="google_chat",
        label="Google Chat",
        adapter_factory=lambda cfg: GoogleChatAdapter(cfg),
        check_fn=_check_for_registry,
        validate_config=_validate_config,
        is_connected=_is_connected,
        required_env=[
            "GOOGLE_CHAT_PROJECT_ID",
            "GOOGLE_CHAT_SUBSCRIPTION_NAME",
            "GOOGLE_CHAT_SERVICE_ACCOUNT_JSON",
        ],
        install_hint="pip install 'hermes-agent[google_chat]'",
        setup_fn=interactive_setup,
        # Env-driven auto-configuration — the core env-populator hook calls
        # this during ``_apply_env_overrides`` and seeds
        # ``PlatformConfig.extra`` + home_channel from env vars.  Without this
        # the adapter would still work on explicit config.yaml entries, but
        # env-only setup (GOOGLE_CHAT_PROJECT_ID/_SUBSCRIPTION_NAME/...) would
        # not flow through to ``gateway status`` or ``get_connected_platforms``.
        env_enablement_fn=_env_enablement,
        # Cron home-channel delivery support.  Lets ``deliver=google_chat``
        # cron jobs route to the configured home space without editing
        # cron/scheduler.py's hardcoded sets.
        cron_deliver_env_var="GOOGLE_CHAT_HOME_CHANNEL",
        # Out-of-process cron delivery via the Chat REST API.  Without this
        # hook, deliver=google_chat cron jobs fail with "No live adapter"
        # when cron runs separately from the gateway.
        standalone_sender_fn=_standalone_send,
        # Auth env vars for _is_user_authorized() integration.
        allowed_users_env="GOOGLE_CHAT_ALLOWED_USERS",
        allow_all_env="GOOGLE_CHAT_ALLOW_ALL_USERS",
        # Chat caps text messages at 4096 chars; we leave margin to fit
        # the "Hermes is thinking..." marker patches and edit overhead.
        max_message_length=4000,
        emoji="💬",
        allow_update_command=True,
        platform_hint=(
            "You are on Google Chat. Limited markdown subset is rendered: "
            "*bold*, _italic_, ~strike~, `code`. No headings or lists. "
            "Message size limit: 4000 characters; longer responses are split "
            "across multiple messages. You are in a space (DM or group). "
            "Images render inline; audio, video, and document attachments "
            "render as download cards (no native voice/video UI). To send "
            "files, include MEDIA:/absolute/path/to/file in your response. "
            "Native file attachments require the user to run /setup-files "
            "once in their own DM — until they do, file requests fall back "
            "to a text notice with the host path. Do NOT generate interactive "
            "Card v2 buttons — Google Chat interactivity is not yet supported "
            "by this gateway; ask for typed confirmations instead. While you "
            "are generating a response, a 'Hermes is thinking…' marker message "
            "appears in the space and is deleted once your response is ready. "
            "You do NOT have access to Google Chat-specific APIs — you cannot "
            "search space history, list space members, or manage spaces. Do "
            "not promise to perform these actions; explain that you can only "
            "read messages sent directly to you and respond in the same "
            "space/thread."
        ),
    )

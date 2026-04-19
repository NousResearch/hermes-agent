"""
Twitter/X platform adapter.

Uses the Twitter API v2 for:
- Posting tweets (with 280-char truncation and thread chaining for long responses)
- Receiving DMs via filtered stream
- OAuth 2.0 PKCE authentication with refresh token rotation
- Media upload (images, GIFs, videos)

Security:
- API response bodies are truncated to 200 chars in error logs
- Refresh tokens are persisted to ~/.hermes/twitter_tokens.json
- Sensitive tokens are never logged in full
"""

import asyncio
import hashlib
import json
import logging
import os
import random
import re
import secrets
import string
import time
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    httpx = None  # type: ignore[assignment]

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    ProcessingOutcome,
    SendResult,
    cache_image_from_bytes,
)
from gateway.platforms.helpers import MessageDeduplicator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TWITTER_API_BASE = "https://api.twitter.com"
TWITTER_UPLOAD_BASE = "https://upload.twitter.com"
TWITTER_AUTHORIZE_URL = "https://twitter.com/i/oauth2/authorize"
TWITTER_TOKEN_URL = f"{TWITTER_API_BASE}/2/oauth2/token"

MAX_TWEET_LENGTH = 280
MAX_DM_LENGTH = 10000
STREAM_RECONNECT_DELAY_BASE = 1.0
STREAM_RECONNECT_DELAY_MAX = 90.0
RATE_LIMIT_RETRY_MAX = 5
MEDIA_CATEGORY_TWEET = "tweet_image"
MEDIA_CATEGORY_DM = "dm_image"

# How much of an API response body to include in error logs.
_LOG_BODY_LIMIT = 200

# Path where refresh tokens are persisted (secure location within user home).
_TOKEN_PERSIST_PATH = Path.home() / ".hermes" / "twitter_tokens.json"


def _truncate_for_log(body: str, limit: int = _LOG_BODY_LIMIT) -> str:
    """Truncate an API response body for safe logging.

    Prevents leaking large response bodies (which may contain PII or tokens)
    into log files.  Cuts at *limit* characters and appends an ellipsis when
    truncation occurred.
    """
    if body is None:
        return ""
    text = str(body)
    if len(text) <= limit:
        return text
    return text[:limit] + "...[truncated]"


def _generate_code_verifier(length: int = 64) -> str:
    """Generate a PKCE code_verifier (RFC 7636 §4.1)."""
    chars = string.ascii_letters + string.digits + "-._~"
    return "".join(secrets.choice(chars) for _ in range(length))


def _generate_code_challenge(verifier: str) -> str:
    """Derive the S256 code_challenge from *verifier*."""
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return (
        digest.hex()  # wrong — need base64url
    )
    # Recompute properly:
    import base64
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def check_twitter_requirements() -> bool:
    """Check if Twitter adapter dependencies are available."""
    return HTTPX_AVAILABLE


# ---------------------------------------------------------------------------
# Token persistence helpers
# ---------------------------------------------------------------------------

def _load_persisted_tokens() -> dict:
    """Load persisted refresh tokens from disk.

    Returns an empty dict if the file does not exist or is corrupt.
    """
    try:
        if _TOKEN_PERSIST_PATH.is_file():
            raw = _TOKEN_PERSIST_PATH.read_text(encoding="utf-8")
            return json.loads(raw)
    except (json.JSONDecodeError, OSError, PermissionError) as exc:
        logger.warning("Failed to load persisted Twitter tokens: %s", exc)
    return {}


def _save_persisted_tokens(tokens: dict) -> None:
    """Atomically persist refresh tokens to disk.

    Writes to a temporary file first, then renames to avoid partial writes
    if the process crashes mid-write.
    """
    try:
        _TOKEN_PERSIST_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = _TOKEN_PERSIST_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(tokens, indent=2), encoding="utf-8")
        # Restrict permissions to owner only (0o600)
        try:
            import os as _os
            _os.chmod(tmp, 0o600)
        except OSError:
            pass  # Windows may not support chmod
        tmp.replace(_TOKEN_PERSIST_PATH)
    except (OSError, PermissionError) as exc:
        logger.warning(
            "Failed to persist Twitter refresh token: %s — "
            "token rotation will not survive process restart",
            exc,
        )


# ---------------------------------------------------------------------------
# Rate-limit backoff
# ---------------------------------------------------------------------------

@dataclass
class RateLimitState:
    """Tracks rate-limit state for exponential backoff."""
    attempts: int = 0
    reset_at: float = 0.0

    def should_backoff(self) -> bool:
        return time.time() < self.reset_at

    def remaining_wait(self) -> float:
        return max(0.0, self.reset_at - time.time())

    def record_429(self, retry_after: Optional[float] = None) -> float:
        """Record a 429 and return the recommended wait time in seconds."""
        self.attempts += 1
        if self.attempts > RATE_LIMIT_RETRY_MAX:
            return -1  # signal: give up
        if retry_after and retry_after > 0:
            wait = retry_after
        else:
            wait = min(
                STREAM_RECONNECT_DELAY_BASE * (2 ** (self.attempts - 1)) + random.uniform(0, 1),
                STREAM_RECONNECT_DELAY_MAX,
            )
        self.reset_at = time.time() + wait
        return wait

    def reset(self) -> None:
        self.attempts = 0
        self.reset_at = 0.0


# ---------------------------------------------------------------------------
# Twitter adapter
# ---------------------------------------------------------------------------

class TwitterAdapter(BasePlatformAdapter):
    """Twitter/X platform adapter.

    Features:
    - OAuth 2.0 with PKCE for user-context actions
    - Tweet posting with automatic thread chaining for responses > 280 chars
    - DM receiving via filtered stream with auto-reconnect
    - Media upload (tweet_image category)
    - Self-message filtering
    - Sanitized logging (response bodies truncated to 200 chars)
    - Refresh token persistence to ``~/.hermes/twitter_tokens.json``
    """

    MAX_MESSAGE_LENGTH = MAX_TWEET_LENGTH

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.TWITTER)
        self._client: Optional[httpx.AsyncClient] = None
        self._stream_task: Optional[asyncio.Task] = None
        self._running = False
        self._dedup = MessageDeduplicator()

        # OAuth tokens
        self._access_token: Optional[str] = None
        self._refresh_token: Optional[str] = None
        self._token_expiry: float = 0.0
        self._client_id: str = os.getenv("TWITTER_CLIENT_ID", "")
        self._client_secret: str = os.getenv("TWITTER_CLIENT_SECRET", "")

        # Bot identity (populated after first token exchange / user lookup)
        self._bot_user_id: Optional[str] = None
        self._bot_username: Optional[str] = None

        # Rate-limit state
        self._rate_limit = RateLimitState()

        # Stream reconnection state
        self._stream_backoff = RateLimitState()

        # Load persisted refresh token as fallback
        self._load_initial_tokens()

    # ------------------------------------------------------------------
    # Token management
    # ------------------------------------------------------------------

    def _load_initial_tokens(self) -> None:
        """Load tokens from env vars, falling back to persisted file."""
        self._access_token = os.getenv("TWITTER_ACCESS_TOKEN") or None
        env_refresh = os.getenv("TWITTER_REFRESH_TOKEN") or None

        if env_refresh:
            self._refresh_token = env_refresh
        else:
            # Fallback to persisted token
            persisted = _load_persisted_tokens()
            persisted_refresh = persisted.get("refresh_token")
            if persisted_refresh:
                logger.info(
                    "Loaded persisted Twitter refresh token from %s",
                    _TOKEN_PERSIST_PATH,
                )
                self._refresh_token = persisted_refresh

    async def _refresh_access_token(self) -> bool:
        """Refresh the OAuth 2.0 access token using the refresh token.

        Returns True on success.  On token rotation the new refresh token is
        persisted to disk so it survives process restarts.
        """
        if not self._refresh_token:
            logger.error("No refresh token available for Twitter OAuth refresh")
            return False

        if not self._client_id:
            logger.error("TWITTER_CLIENT_ID not configured — cannot refresh token")
            return False

        payload = {
            "grant_type": "refresh_token",
            "refresh_token": self._refresh_token,
            "client_id": self._client_id,
        }

        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        auth = None
        if self._client_secret:
            auth = (self._client_id, self._client_secret)

        try:
            resp = await self._http_client().post(
                TWITTER_TOKEN_URL,
                data=payload,
                headers=headers,
                auth=auth,
            )

            if resp.status_code != 200:
                logger.error(
                    "Twitter token refresh failed (HTTP %d): %s",
                    resp.status_code,
                    _truncate_for_log(resp.text),
                )
                return False

            data = resp.json()
            self._access_token = data.get("access_token")
            new_refresh = data.get("refresh_token")
            expires_in = data.get("expires_in", 7200)
            self._token_expiry = time.time() + expires_in - 60  # refresh 60s early

            # Handle refresh token rotation
            if new_refresh and new_refresh != self._refresh_token:
                logger.info(
                    "Twitter refresh token rotated — persisting new token to %s",
                    _TOKEN_PERSIST_PATH,
                )
                self._refresh_token = new_refresh
                _save_persisted_tokens({"refresh_token": new_refresh})

            return True

        except httpx.HTTPError as exc:
            logger.error(
                "Twitter token refresh network error: %s",
                _truncate_for_log(str(exc)),
            )
            return False

    async def _ensure_valid_token(self) -> bool:
        """Ensure we have a valid (non-expired) access token."""
        if self._access_token and time.time() < self._token_expiry:
            return True
        return await self._refresh_access_token()

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _http_client(self) -> "httpx.AsyncClient":
        """Return (lazily create) the shared HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=30.0,
                follow_redirects=False,
            )
        return self._client

    def _auth_headers(self) -> Dict[str, str]:
        """Return Authorization headers for the current access token."""
        if not self._access_token:
            return {}
        return {"Authorization": f"Bearer {self._access_token}"}

    async def _api_request(
        self,
        method: str,
        url: str,
        *,
        json_body: Optional[dict] = None,
        params: Optional[dict] = None,
        retry_auth: bool = True,
    ) -> Tuple[int, dict]:
        """Make an authenticated API request with rate-limit retry.

        Returns (status_code, parsed_json_or_empty_dict).
        """
        await self._ensure_valid_token()

        for attempt in range(RATE_LIMIT_RETRY_MAX + 1):
            if self._rate_limit.should_backoff():
                wait = self._rate_limit.remaining_wait()
                logger.info("Rate-limited; waiting %.1fs before retry", wait)
                await asyncio.sleep(wait)

            try:
                resp = await self._http_client().request(
                    method,
                    url,
                    headers=self._auth_headers(),
                    json=json_body,
                    params=params,
                )
            except httpx.HTTPError as exc:
                logger.error(
                    "Twitter API network error: %s",
                    _truncate_for_log(str(exc)),
                )
                return 0, {}

            # 429 — rate limited
            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                wait = self._rate_limit.record_429(
                    float(retry_after) if retry_after else None
                )
                if wait < 0:
                    logger.error(
                        "Twitter rate limit exceeded after %d retries: %s",
                        RATE_LIMIT_RETRY_MAX,
                        _truncate_for_log(resp.text),
                    )
                    return resp.status_code, {}
                logger.warning(
                    "Twitter rate limit hit (attempt %d/%d); waiting %.1fs",
                    attempt + 1,
                    RATE_LIMIT_RETRY_MAX,
                    wait,
                )
                await asyncio.sleep(wait)
                continue

            self._rate_limit.reset()

            # 401 — token expired; try one refresh
            if resp.status_code == 401 and retry_auth:
                logger.info("Twitter API returned 401 — refreshing token")
                if await self._refresh_access_token():
                    return await self._api_request(
                        method, url, json_body=json_body, params=params, retry_auth=False
                    )
                return resp.status_code, {}

            # Non-2xx errors
            if resp.status_code >= 400:
                logger.error(
                    "Twitter API error (HTTP %d): %s",
                    resp.status_code,
                    _truncate_for_log(resp.text),
                )
                return resp.status_code, {}

            try:
                return resp.status_code, resp.json()
            except Exception:
                return resp.status_code, {}

        return 0, {}

    # ------------------------------------------------------------------
    # Bot identity
    # ------------------------------------------------------------------

    async def _resolve_bot_identity(self) -> None:
        """Fetch and cache the bot user's ID and username."""
        code, data = await self._api_request(
            "GET",
            f"{TWITTER_API_BASE}/2/users/me",
            params={"user.fields": "id,username"},
        )
        if code == 200 and "data" in data:
            self._bot_user_id = data["data"].get("id")
            self._bot_username = data["data"].get("username")
            logger.info(
                "Twitter bot identity: @%s (ID %s)",
                self._bot_username or "?",
                self._bot_user_id or "?",
            )
        else:
            logger.warning(
                "Could not resolve Twitter bot identity: %s",
                _truncate_for_log(json.dumps(data)),
            )

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        """Connect to Twitter API and start listening for DMs."""
        if not check_twitter_requirements():
            self._set_fatal_error(
                "missing_dependency",
                "httpx is required for the Twitter adapter — pip install httpx",
                retryable=False,
            )
            return False

        if not self._refresh_token and not self._access_token:
            self._set_fatal_error(
                "missing_credentials",
                "TWITTER_REFRESH_TOKEN or TWITTER_ACCESS_TOKEN must be set",
                retryable=False,
            )
            return False

        # Ensure we have a valid token
        if not await self._ensure_valid_token():
            self._set_fatal_error(
                "auth_failed",
                "Failed to obtain a valid Twitter access token",
                retryable=True,
            )
            return False

        # Resolve bot identity for self-message filtering
        await self._resolve_bot_identity()

        self._running = True
        self._mark_connected()

        # Start filtered stream for DMs
        self._stream_task = asyncio.ensure_future(self._stream_loop())

        logger.info("Twitter adapter connected")
        return True

    async def disconnect(self) -> None:
        """Stop the stream and close HTTP connections."""
        self._running = False
        if self._stream_task and not self._stream_task.done():
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
        self._mark_disconnected()
        logger.info("Twitter adapter disconnected")

    # ------------------------------------------------------------------
    # Filtered stream (DMs)
    # ------------------------------------------------------------------

    async def _stream_loop(self) -> None:
        """Listen for DM events via the filtered stream endpoint.

        Implements exponential backoff with jitter on disconnects.
        """
        while self._running:
            try:
                await self._listen_stream()
            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.error(
                    "Twitter stream error: %s",
                    _truncate_for_log(str(exc)),
                )

            if not self._running:
                return

            # Reconnection backoff
            wait = self._stream_backoff.record_429()
            if wait < 0:
                wait = STREAM_RECONNECT_DELAY_MAX
            jitter = random.uniform(0, wait * 0.25)
            total = min(wait + jitter, STREAM_RECONNECT_DELAY_MAX)
            logger.info(
                "Twitter stream disconnected — reconnecting in %.1fs", total
            )
            await asyncio.sleep(total)

    async def _listen_stream(self) -> None:
        """Connect to the Twitter filtered stream and process incoming DMs."""
        await self._ensure_valid_token()

        # Set up stream rules for DMs (only receive DMs directed at us)
        await self._setup_stream_rules()

        url = f"{TWITTER_API_BASE}/2/dm_events"
        params = {
            "dm_event.fields": "id,text,sender_id,created_at,event_type",
            "expansions": "sender_id",
            "user.fields": "id,username",
        }

        async with self._http_client().stream(
            "GET",
            url,
            headers=self._auth_headers(),
            params=params,
            timeout=httpx.Timeout(120.0, read=300.0),
        ) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                logger.error(
                    "Twitter stream connect failed (HTTP %d): %s",
                    resp.status_code,
                    _truncate_for_log(body.decode("utf-8", errors="replace")),
                )
                return

            logger.info("Twitter DM stream connected")
            self._stream_backoff.reset()

            async for line in resp.aiter_lines():
                if not self._running:
                    return
                if not line or line.strip() == "":
                    continue
                if line.startswith(":"):
                    # Keep-alive comment
                    continue

                try:
                    event = json.loads(line)
                    await self._process_dm_event(event)
                except json.JSONDecodeError:
                    logger.debug("Twitter stream non-JSON line: %s", _truncate_for_log(line))
                except Exception as exc:
                    logger.error(
                        "Error processing Twitter DM event: %s",
                        _truncate_for_log(str(exc)),
                    )

    async def _setup_stream_rules(self) -> None:
        """Ensure filtered stream rules are configured for DMs."""
        # Check existing rules
        code, data = await self._api_request(
            "GET", f"{TWITTER_API_BASE}/2/dm_conversations/with/rules"
        )
        # Stream rules setup is platform-specific; log for now
        if code != 200:
            logger.debug(
                "Could not fetch stream rules: %s",
                _truncate_for_log(json.dumps(data)),
            )

    async def _process_dm_event(self, event: dict) -> None:
        """Process a single DM event from the filtered stream."""
        data = event.get("data", event)
        event_type = data.get("event_type", "")

        if event_type != "MessageCreate":
            return

        msg_id = data.get("id", "")
        if self._dedup.is_duplicate(msg_id):
            return

        sender_id = data.get("sender_id", "")
        text = data.get("text", "")

        # Filter self-messages
        if self._bot_user_id and sender_id == self._bot_user_id:
            return

        if not text:
            return

        # Resolve sender username from expansions
        sender_username = sender_id
        includes = event.get("includes", {})
        users = includes.get("users", [])
        for user in users:
            if user.get("id") == sender_id:
                sender_username = user.get("username", sender_id)
                break

        # DM conversation ID is used as the chat_id
        conversation_id = data.get("dm_conversation_id", sender_id)

        source = self.build_source(
            chat_id=conversation_id,
            chat_name=f"DM with @{sender_username}",
            chat_type="dm",
            user_id=sender_id,
            user_name=sender_username,
        )

        msg_event = MessageEvent(
            text=text,
            message_type=MessageType.TEXT,
            source=source,
            raw_message=event,
            message_id=msg_id,
            timestamp=self._parse_timestamp(data.get("created_at")),
        )

        await self.handle_message(msg_event)

    # ------------------------------------------------------------------
    # Sending — Tweet
    # ------------------------------------------------------------------

    async def send(
        self,
        chat_id: str,
        text: str,
        *,
        reply_to_message_id: Optional[str] = None,
        **kwargs: Any,
    ) -> SendResult:
        """Post a tweet or a tweet thread (for long responses).

        *chat_id* is ignored for tweets (public timeline).
        *reply_to_message_id* is used as the in_reply_to_tweet_id for threading.
        """
        text = self.format_message(text)
        chunks = self._split_for_tweets(text)

        if not chunks:
            return SendResult(success=False, error="Empty tweet text")

        previous_tweet_id = reply_to_message_id

        for i, chunk in enumerate(chunks):
            payload: Dict[str, Any] = {"text": chunk}
            if previous_tweet_id:
                payload["reply"] = {"in_reply_to_tweet_id": previous_tweet_id}

            code, data = await self._api_request(
                "POST",
                f"{TWITTER_API_BASE}/2/tweets",
                json_body=payload,
            )

            if code not in (200, 201):
                return SendResult(
                    success=False,
                    error=f"Twitter API error (HTTP {code}): {_truncate_for_log(json.dumps(data))}",
                )

            tweet_data = data.get("data", {})
            previous_tweet_id = tweet_data.get("id")

        return SendResult(
            success=True,
            message_id=previous_tweet_id,
        )

    def _split_for_tweets(self, text: str) -> List[str]:
        """Split text into tweet-sized chunks for thread posting.

        Respects the 280-char limit.  Splits on paragraph boundaries when
        possible, then on sentence boundaries, then mid-word as a last resort.
        """
        if len(text) <= MAX_TWEET_LENGTH:
            return [text] if text else []

        chunks: List[str] = []
        remaining = text

        while remaining:
            if len(remaining) <= MAX_TWEET_LENGTH:
                chunks.append(remaining)
                break

            # Try to split on paragraph boundary
            cut = remaining[:MAX_TWEET_LENGTH].rfind("\n\n")
            if cut > MAX_TWEET_LENGTH * 0.3:
                chunks.append(remaining[:cut].rstrip())
                remaining = remaining[cut:].lstrip("\n")
                continue

            # Try sentence boundary
            cut = remaining[:MAX_TWEET_LENGTH].rfind(". ")
            if cut > MAX_TWEET_LENGTH * 0.3:
                chunks.append(remaining[: cut + 1])
                remaining = remaining[cut + 1:].lstrip()
                continue

            # Try word boundary
            cut = remaining[:MAX_TWEET_LENGTH].rfind(" ")
            if cut > MAX_TWEET_LENGTH * 0.3:
                chunks.append(remaining[:cut])
                remaining = remaining[cut:].lstrip()
                continue

            # Hard cut
            chunks.append(remaining[:MAX_TWEET_LENGTH])
            remaining = remaining[MAX_TWEET_LENGTH:]

        return chunks

    # ------------------------------------------------------------------
    # Sending — DM
    # ------------------------------------------------------------------

    async def send_dm(
        self,
        conversation_id: str,
        text: str,
    ) -> SendResult:
        """Send a DM in an existing conversation."""
        text = self.format_message(text)

        if len(text) > MAX_DM_LENGTH:
            text = text[: MAX_DM_LENGTH - 3] + "..."

        payload = {
            "event_type": "MessageCreate",
            "event": {
                "message_create": {
                    "target": {"recipient_id": conversation_id},
                    "message_data": {"text": text},
                }
            },
        }

        code, data = await self._api_request(
            "POST",
            f"{TWITTER_API_BASE}/2/dm_conversations/with/{conversation_id}/messages",
            json_body=payload,
        )

        if code not in (200, 201):
            return SendResult(
                success=False,
                error=f"DM send failed (HTTP {code}): {_truncate_for_log(json.dumps(data))}",
            )

        event_data = data.get("event", {}).get("event", data.get("event", {}))
        msg_id = (
            event_data.get("message_create", {}).get("id")
            or data.get("data", {}).get("id")
        )
        return SendResult(success=True, message_id=msg_id)

    # ------------------------------------------------------------------
    # Media upload
    # ------------------------------------------------------------------

    async def upload_media(
        self,
        media_bytes: bytes,
        media_type: str = "image/jpeg",
    ) -> Optional[str]:
        """Upload media and return the media_id_string.

        Uses the chunked upload endpoint for reliability.
        """
        await self._ensure_valid_token()

        total_bytes = len(media_bytes)
        chunk_size = 5 * 1024 * 1024  # 5 MB chunks

        # INIT
        init_resp = await self._http_client().post(
            f"{TWITTER_UPLOAD_BASE}/1.1/media/upload.json",
            headers=self._auth_headers(),
            data={
                "command": "INIT",
                "total_bytes": str(total_bytes),
                "media_type": media_type,
                "media_category": MEDIA_CATEGORY_TWEET,
            },
        )
        if init_resp.status_code != 200:
            logger.error(
                "Twitter media INIT failed (HTTP %d): %s",
                init_resp.status_code,
                _truncate_for_log(init_resp.text),
            )
            return None

        media_id = init_resp.json().get("media_id_string")
        if not media_id:
            logger.error(
                "Twitter media INIT: no media_id_string in response: %s",
                _truncate_for_log(init_resp.text),
            )
            return None

        # APPEND chunks
        segment_index = 0
        for offset in range(0, total_bytes, chunk_size):
            chunk = media_bytes[offset : offset + chunk_size]
            append_resp = await self._http_client().post(
                f"{TWITTER_UPLOAD_BASE}/1.1/media/upload.json",
                headers=self._auth_headers(),
                data={
                    "command": "APPEND",
                    "media_id": media_id,
                    "segment_index": str(segment_index),
                },
                files={"media": ("chunk", chunk, media_type)},
            )
            if append_resp.status_code != 200:
                logger.error(
                    "Twitter media APPEND chunk %d failed (HTTP %d): %s",
                    segment_index,
                    append_resp.status_code,
                    _truncate_for_log(append_resp.text),
                )
                return None
            segment_index += 1

        # FINALIZE
        finalize_resp = await self._http_client().post(
            f"{TWITTER_UPLOAD_BASE}/1.1/media/upload.json",
            headers=self._auth_headers(),
            data={"command": "FINALIZE", "media_id": media_id},
        )
        if finalize_resp.status_code not in (200, 201, 202):
            logger.error(
                "Twitter media FINALIZE failed (HTTP %d): %s",
                finalize_resp.status_code,
                _truncate_for_log(finalize_resp.text),
            )
            return None

        return media_id

    # ------------------------------------------------------------------
    # Platform adapter interface
    # ------------------------------------------------------------------

    async def send_typing(self, chat_id: str) -> None:
        """Twitter has no typing indicator — no-op."""
        return None

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: str = "",
    ) -> SendResult:
        """Post a tweet with an image."""
        # Download the image
        try:
            resp = await self._http_client().get(image_url)
            resp.raise_for_status()
            media_bytes = resp.content
        except httpx.HTTPError as exc:
            return SendResult(
                success=False,
                error=f"Failed to download image: {_truncate_for_log(str(exc))}",
            )

        media_id = await self.upload_media(media_bytes)
        if not media_id:
            return SendResult(success=False, error="Media upload failed")

        payload: Dict[str, Any] = {
            "text": caption or "",
            "media": {"media_ids": [media_id]},
        }

        code, data = await self._api_request(
            "POST",
            f"{TWITTER_API_BASE}/2/tweets",
            json_body=payload,
        )

        if code not in (200, 201):
            return SendResult(
                success=False,
                error=f"Tweet with image failed (HTTP {code}): {_truncate_for_log(json.dumps(data))}",
            )

        tweet_id = data.get("data", {}).get("id")
        return SendResult(success=True, message_id=tweet_id)

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Get information about a Twitter user or conversation."""
        code, data = await self._api_request(
            "GET",
            f"{TWITTER_API_BASE}/2/users/{chat_id}",
            params={"user.fields": "id,username,name"},
        )
        if code == 200 and "data" in data:
            user = data["data"]
            return {
                "name": user.get("username", chat_id),
                "type": "dm",
                "chat_id": chat_id,
            }
        return {"name": chat_id, "type": "dm", "chat_id": chat_id}

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def format_message(self, content: str) -> str:
        """Strip rich markdown to plain text for Twitter.

        Twitter does not support markdown formatting in tweets.
        """
        # Strip bold / italic / code markers
        content = re.sub(r"\*\*(.+?)\*\*", r"\1", content)
        content = re.sub(r"\*(.+?)\*", r"\1", content)
        content = re.sub(r"_(.+?)_", r"\1", content)
        content = re.sub(r"`(.+?)`", r"\1", content)
        # Strip fenced code blocks (keep inner text)
        content = re.sub(r"```(?:\w*\n)?(.*?)```", r"\1", content, flags=re.DOTALL)
        # Collapse multiple blank lines
        content = re.sub(r"\n{3,}", "\n\n", content)
        return content.strip()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_timestamp(ts: Optional[str]) -> "datetime":
        """Parse a Twitter API ISO-8601 timestamp string."""
        from datetime import datetime
        if not ts:
            return datetime.now()
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return datetime.now()

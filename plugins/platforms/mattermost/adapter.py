"""Mattermost gateway adapter — REST API v4 + WebSocket via aiohttp (no Mattermost SDK).

Environment variables:
    MATTERMOST_URL              Server URL (e.g. https://mm.example.com)
    MATTERMOST_TOKEN            Bot token or personal-access token
    MATTERMOST_ALLOWED_USERS    Comma-separated user IDs
    MATTERMOST_HOME_CHANNEL     Channel ID for cron/notification delivery
"""

from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import os
import re
from pathlib import Path
from urllib.parse import unquote as _unquote
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from gateway.config import Platform, PlatformConfig
from gateway.platforms.helpers import MessageDeduplicator
from gateway.platforms.helpers import cancel_task
from gateway.platforms.base import gateway_trust_env, BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.platforms._shared import (
    apply_yaml_bridge as _apply_yaml_bridge, env_is_connected as _env_is_connected,
    extra_or_secret as _extra_or_secret, get_scoped_secret as _get_scoped_secret,
    send_error
)
from plugins.platforms.matrix.adapter import _extra_csv_set  # matrix pattern: read MATTERMOST_ALLOWED_USERS

logger = logging.getLogger(__name__)

_Metadata = Optional[Dict[str, Any]]

# Server default is 16383, but 4000 is the practical limit for readable messages.
MAX_POST_LENGTH = 4000

# Channel type codes returned by the Mattermost API ("P" private → treat as group).
_CHANNEL_TYPE_MAP = {"D": "dm", "G": "group", "P": "group", "O": "channel"}

_MATTERMOST_DISABLE_MENTIONS_PROPS = {"disable_mentions": True}

_RECONNECT_BASE_DELAY, _RECONNECT_MAX_DELAY, _RECONNECT_JITTER = 2.0, 60.0, 0.2  # exponential backoff

_POST_WITH_FILE_ERROR = "Failed to post with file"
_MEDIA_MSG_TYPES = (("image/", MessageType.PHOTO), ("audio/", MessageType.VOICE))  # first match wins
_INBOUND_CACHE_EXT = {"image/": ".png", "audio/": ".ogg"}  # mime prefix → default extension for cached media

# ── Approval UX (sloth 2026-09-26; v2 2026-09-27) ─────────────────────────
# Reaction-driven approval: the bot seeds its own approval card with one
# reaction per offered choice. Any non-bot user tapping one resolves the
# pending approval. Slash-free text matching for ``approve``/``deny``/``yes``
# was tried first and removed 2026-09-26: bare text in chat is too ambiguous
# and the slash form (``/approve`` or ``<space>/approve`` to bypass the
# Mattermost slash-router) is reliable, so we stick with that.
# Reaction → choice map for B-side approval (matrix parity: 🌀=session,
# ♾️=always). Emoji short names are what Mattermost's v4 reaction API accepts
# in POST /api/v4/reactions — same set Matrix uses.
_REACTION_TO_CHOICE = {
    "white_check_mark": "once",
    "cyclone": "session",       # 🌀
    "infinity": "always",       # ♾️
    "no_entry_sign": "deny",
}
_REACTION_LABEL = {
    "once":    "✅ Approved once",
    "session": "🌀 Approved for this session",
    "always":  "♾️ Approved always",
    "deny":    "❌ Denied",
}
_EMOJI_FOR_CHOICE = {
    "once": "white_check_mark", "session": "cyclone",
    "always": "infinity",       "deny": "no_entry_sign",
}
# Bounded so a stuck adapter doesn't grow the registry forever; one entry per
# outstanding approval card. 64 is generous — the runner also bounds it.
_APPROVAL_PROMPTS_MAX = 64

# ── Clarify (choice-picker) UX (v2 2026-09-27) ──────────────────────────────
# Numbered-emoji picker for multiple-choice questions the agent asks the
# user. Mirrors matrix's `_MATRIX_CHOICE_PICKER_REACTIONS` so cross-platform
# fixes (de-dup, expiry) get written once. 12 slots total; more choices
# fall through to the gateway text fallback (numbered list).
_MATTERMOST_PICKER_REACTIONS = (
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "keycap_ten", "a", "b",
)
_CLARIFY_PICKER_TTL_SECONDS = 300  # matrix parity (5 min)
_CLARIFY_PICKER_MAX = 64           # bounded; older entries dropped on overflow


@dataclass
class _MattermostClarifyPicker:
    """One pending reaction-driven clarify picker on Mattermost.

    Mirrors ``_MatrixPickerPrompt`` (matrix/adapter.py). Lifecycle:
    registered on ``send_clarify`` (after the post is on the server),
    mutated to ``resolved=True`` on first valid tap, dropped from the
    registry at the end of the resolve path.
    """
    chat_id: str
    post_id: str
    clarify_id: str
    choices: List[str]     # emoji short names (e.g. ["one","two",...])
    responses: List[str]   # human labels in the same order
    expires_at: float
    # ``numeric_labels`` is set on register (parallel to ``choices``: "1", "2", ..., "10" for
    # ``keycap_ten``). It feeds the typeable-fallback hint line on the card body. Empty default
    # for backwards compat with registered entries created before the v3 display fix.
    numeric_labels: List[str] = field(default_factory=list)
    resolved: bool = False


def _is_user_authorized_mattermost(user_ids: Set[str], user_id: str) -> bool:
    """Mirror matrix's ``_is_authorized_user`` (matrix/adapter.py:2536).

    Empty set = allow all (parity with ``MATTERMOST_ALLOW_ALL_USERS`` and the
    gateway-wide allow-all); non-empty set = explicit allowlist. Returns False
    for an empty user_id (no identity, can't be paired but the gateway text
    fallback will handle it).
    """
    if not user_id:
        return False
    return not user_ids or user_id in user_ids


def _parse_reaction_event(data: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Parse the nested ``data.reaction`` JSON a Mattermost ``reaction_added``
    WS event carries. Returns ``(post_id, user_id, emoji_name)`` — all three
    may be None/empty if the payload is malformed.

    Single source of truth so approval and clarify-picker handlers don't
    each duplicate the JSON-shape handling.
    """
    raw = data.get("reaction")
    if not raw:
        return None, None, None
    try:
        reaction = json.loads(raw) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, TypeError):
        return None, None, None
    return (
        str(reaction.get("post_id") or "") or None,
        str(reaction.get("user_id") or "") or None,
        str(reaction.get("emoji_name") or "") or None,
    )


def _with_mentions_disabled(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Return a post payload that prevents Mattermost from firing mentions."""
    props, disable = payload.get("props"), _MATTERMOST_DISABLE_MENTIONS_PROPS
    payload["props"] = {**props, **disable} if isinstance(props, dict) else dict(disable)
    return payload


def _channel_id_set(raw: Any) -> set:
    """Parse a list or comma-separated string of channel IDs into a stripped set."""
    items = raw if isinstance(raw, list) else str(raw).split(",")
    return {str(c).strip() for c in items if str(c).strip()}


def _post_result(data: Dict[str, Any], error: str) -> SendResult:
    if not data or "id" not in data:
        return SendResult(success=False, error=error)
    return SendResult(success=True, message_id=data["id"])


def _url_filename(url: str, fallback: str) -> str:
    return url.rsplit("/", 1)[-1].split("?")[0] or fallback


def _url_and_token(config) -> Tuple[str, str]:
    """(server URL, token): ``config`` first, MATTERMOST_URL / MATTERMOST_TOKEN env fallback."""
    extra = getattr(config, "extra", {}) or {}
    return (extra.get("url") or _get_scoped_secret("MATTERMOST_URL", ""),
            getattr(config, "token", None) or _get_scoped_secret("MATTERMOST_TOKEN", ""))


def check_mattermost_requirements() -> bool:
    """Return True if the Mattermost adapter runtime dependency is available."""
    try:
        import aiohttp  # noqa: F401
        return True
    except ImportError:
        logger.warning("Mattermost: aiohttp not installed")
        return False


def validate_mattermost_config(config: PlatformConfig) -> bool:
    """Return True when Mattermost has enough config to connect."""
    url, token = _url_and_token(config)
    if not token.strip():
        logger.debug("Mattermost: MATTERMOST_TOKEN not set")
        return False
    if not url.strip():
        logger.warning("Mattermost: MATTERMOST_URL not set")
        return False
    return True


class MattermostAdapter(BasePlatformAdapter):
    """Gateway adapter for Mattermost (self-hosted or cloud)."""

    splits_long_messages = True  # send() chunks via truncate_message(MAX_POST_LENGTH)

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.MATTERMOST)
        self._base_url, self._token = _url_and_token(config)
        self._base_url = self._base_url.rstrip("/")
        self._bot_user_id = self._bot_username = ""
        self._session: Any = None  # aiohttp.ClientSession
        self._ws: Any = None  # aiohttp.ClientWebSocketResponse
        self._ws_task: Optional[asyncio.Task] = None
        self._reconnect_task: Optional[asyncio.Task] = None
        self._closing = False
        # Reply mode: "thread" to nest replies, "off" for flat messages.
        self._reply_mode: str = (
            config.extra.get("reply_mode", "") or _get_scoped_secret("MATTERMOST_REPLY_MODE", "off")).lower()
        self._last_post_status: Optional[int] = None  # POST-only, read by the broken-thread-root fallback
        self._last_post_error: str = ""
        self._dedup = MessageDeduplicator()
        # Approval UX: post_id → {session_key, requester_user_id, choices, request_id, chat_id}.
        # Bounded dict — the runner's wait also caps outstanding approvals, so this is the
        # adapter-side mirror; one entry per outstanding card.
        self._approval_prompts_by_event: Dict[str, Dict[str, Any]] = {}
        # Double-click guard: same post_id is resolved once; second tap is a no-op.
        self._approval_resolved: Dict[str, bool] = {}
        # Clarify-picker UX (v2): post_id → _MattermostClarifyPicker for the lifetime of
        # the picker. Mirrors matrix's `_choice_picker_prompts_by_event`; one entry per
        # outstanding picker card.
        self._clarify_picker_prompts_by_event: Dict[str, "_MattermostClarifyPicker"] = {}
        # v2: adapter-side auth gate, sourced from MATTERMOST_ALLOWED_USERS via the same
        # YAML→env bridge matrix uses for MATRIX_ALLOWED_USERS (matrix/adapter.py:537).
        # Empty set = allow all (matches `MATTERMOST_ALLOW_ALL_USERS` semantics); non-empty
        # = explicit allowlist (only those user_ids can resolve reaction-driven prompts).
        self._allowed_user_ids: Set[str] = _extra_csv_set(config, "allowed_users", "MATTERMOST_ALLOWED_USERS")

    # --- HTTP helpers ---

    def _headers(self) -> Dict[str, str]:
        return {**self._auth_header(), "Content-Type": "application/json"}

    def _auth_header(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self._token}"}

    async def _api(self, method: str, path: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """{method} /api/v4/{path}; POST also records _last_post_status/_last_post_error."""
        import aiohttp
        if ".." in path:
            logger.error("MM API path traversal blocked: %s", path)
            return {}
        url = f"{self._base_url}/api/v4/{path.lstrip('/')}"
        is_post = method == "POST"
        if is_post:
            self._last_post_status, self._last_post_error = None, ""
        kwargs: Dict[str, Any] = {"headers": self._headers()}
        if payload is not None:
            kwargs["json"] = payload
        if method != "PUT":  # PUT relies on the session default timeout
            kwargs["timeout"] = aiohttp.ClientTimeout(total=30)
        try:
            async with getattr(self._session, method.lower())(url, **kwargs) as resp:
                if is_post:
                    self._last_post_status = resp.status
                if resp.status >= 400:
                    body = await resp.text()
                    if is_post:
                        self._last_post_error = body or ""
                    logger.error("MM API %s %s → %s: %s", method, path, resp.status, body[:200])
                    return {}
                return await resp.json()
        except aiohttp.ClientError as exc:
            if is_post:
                self._last_post_error = str(exc)
            logger.error("MM API %s %s network error: %s", method, path, exc)
            return {}

    async def _api_get(self, path: str) -> Dict[str, Any]:
        return await self._api("GET", path)

    async def _api_post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return await self._api("POST", path, payload)

    def _last_post_failure_is_broken_thread_root(self) -> bool:
        """Return True only for clear invalid/missing Mattermost thread roots."""
        body = (self._last_post_error or "").lower()
        if self._last_post_status not in {400, 404} or not body:
            return False
        return (any(marker in body for marker in ("root_id", "rootid", "root id", "thread", "post"))
                and any(marker in body for marker in ("invalid", "not found", "does not exist", "missing")))

    async def _post_preserving_thread(
        self, chat_id: str, payload: Dict[str, Any], metadata: _Metadata) -> Dict[str, Any]:
        """Post once, optionally falling back flat for final notify content."""
        data = await self._api_post("posts", payload)
        if (data or "root_id" not in payload or not (isinstance(metadata, dict) and metadata.get("notify"))
                or not self._last_post_failure_is_broken_thread_root()):
            return data
        flat_payload = {k: v for k, v in payload.items() if k != "root_id"}
        body = str(flat_payload.get("message") or "")
        flat_payload["message"] = self.warning_text(
            ("⚠️ Mattermost thread delivery failed; posting final reply in channel.\n\n" + body).strip(), body)
        logger.warning("Mattermost: falling back to flat channel delivery for notify-worthy post in %s", chat_id)
        return await self._api_post("posts", flat_payload)

    async def _post_message(self, chat_id: str, message: str, reply_to: Optional[str], metadata: _Metadata,
                            file_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Build a mentions-disabled post payload (+ optional root_id) and post it."""
        base: Dict[str, Any] = {"channel_id": chat_id, "message": message}
        if file_ids is not None:
            base["file_ids"] = file_ids
        payload = _with_mentions_disabled(base)
        if self._reply_mode == "thread":
            # root_id from reply_to, else metadata["thread_id"]/["root_id"], resolved to the true thread root.
            candidate = reply_to or (
                isinstance(metadata, dict) and (metadata.get("thread_id") or metadata.get("root_id")))
            if candidate:
                payload["root_id"] = await self._resolve_root_id(str(candidate))
        return await self._post_preserving_thread(chat_id, payload, metadata)

    async def _post_with_file(self, chat_id: str, file_id: str, caption: Optional[str], reply_to: Optional[str],
                              metadata: _Metadata) -> SendResult:
        return _post_result(await self._post_message(chat_id, caption or "", reply_to, metadata, [file_id]),
                            _POST_WITH_FILE_ERROR)

    async def _upload_file(self, channel_id: str, file_data: bytes, filename: str,
                           content_type: str = "application/octet-stream") -> Optional[str]:
        """Upload a file and return its file ID, or None on failure."""
        import aiohttp
        form = aiohttp.FormData()
        form.add_field("channel_id", channel_id)
        form.add_field("files", file_data, filename=filename, content_type=content_type)
        async with self._session.post(f"{self._base_url}/api/v4/files", headers=self._auth_header(), data=form,
                                      timeout=aiohttp.ClientTimeout(total=60)) as resp:
            if resp.status >= 400:
                body = await resp.text()
                logger.error("MM file upload → %s: %s", resp.status, body[:200])
                return None
            infos = (await resp.json()).get("file_infos", [])
            return infos[0]["id"] if infos else None

    # --- Required overrides ---

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        """Connect to Mattermost and start the WebSocket listener."""
        import aiohttp
        if not self._base_url or not self._token:
            logger.error("Mattermost: URL or token not configured")
            return False
        self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30), trust_env=gateway_trust_env())
        self._closing = False
        me = await self._api_get("users/me")
        if not me or "id" not in me:
            logger.error("Mattermost: failed to authenticate — check MATTERMOST_TOKEN and MATTERMOST_URL")
            await self._session.close()
            return False
        self._bot_user_id, self._bot_username = me["id"], me.get("username", "")
        logger.info(
            "Mattermost: authenticated as @%s (%s) on %s", self._bot_username, self._bot_user_id, self._base_url)
        self._ws_task = asyncio.create_task(self._ws_loop())
        self._mark_connected()
        self._wire_plugin_handlers(None)  # plugin-registered native handlers
        return True

    async def disconnect(self) -> None:
        self._closing = True
        await cancel_task(self._ws_task)
        if self._reconnect_task and not self._reconnect_task.done():
            self._reconnect_task.cancel()
        if self._ws:
            await self._ws.close()
            self._ws = None
        if self._session and not self._session.closed:
            await self._session.close()
        logger.info("Mattermost: disconnected")

    async def _resolve_root_id(self, post_id: str) -> str:
        """Resolve a post_id to its thread root_id (a reply's own ID causes "Invalid RootId parameter")."""
        if not post_id:
            return post_id
        data = await self._api_get(f"posts/{post_id}")
        return data["root_id"] if data and data.get("root_id") else post_id

    async def send(
        self, chat_id: str, content: str, reply_to: Optional[str] = None, metadata: _Metadata = None) -> SendResult:
        """Send a message (or multiple chunks) to a channel; reply_to / metadata["thread_id"] is the root post."""
        if not content:
            return SendResult(success=True)
        result = SendResult(success=True)
        for chunk in self.truncate_message(self.format_message(content), MAX_POST_LENGTH):
            result = _post_result(await self._post_message(chat_id, chunk, reply_to, metadata), "Failed to create post")
            if not result.success:
                break
        return result

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        data = await self._api_get(f"channels/{chat_id}")
        if not data:
            return {"name": chat_id, "type": "channel"}
        return {"name": data.get("display_name") or data.get("name") or chat_id,
                "type": _CHANNEL_TYPE_MAP.get(data.get("type", "O"), "channel")}

    # --- Optional overrides ---

    async def send_typing(self, chat_id: str, metadata: _Metadata = None) -> None:
        await self._api_post(f"users/{self._bot_user_id}/typing", {"channel_id": chat_id})

    async def edit_message(self, chat_id: str, message_id: str, content: str, *, finalize: bool = False) -> SendResult:
        payload = _with_mentions_disabled({"message": self.format_message(content)})
        return _post_result(await self._api("PUT", f"posts/{message_id}/patch", payload), "Failed to edit post")

    async def send_image(self, chat_id: str, image_url: str, caption: Optional[str] = None,
                         reply_to: Optional[str] = None, metadata: _Metadata = None) -> SendResult:
        return await self._send_url_as_file(chat_id, image_url, caption, reply_to, "image", metadata)

    async def send_image_file(self, chat_id: str, image_path: str, caption: Optional[str] = None,
                              reply_to: Optional[str] = None, metadata: _Metadata = None) -> SendResult:
        return await self._send_local_file(chat_id, image_path, caption, reply_to, metadata=metadata)

    async def send_document(
        self, chat_id: str, file_path: str, caption: Optional[str] = None, file_name: Optional[str] = None,
        reply_to: Optional[str] = None, metadata: _Metadata = None) -> SendResult:
        return await self._send_local_file(chat_id, file_path, caption, reply_to, file_name, metadata)

    async def send_voice(self, chat_id: str, audio_path: str, caption: Optional[str] = None,
                         reply_to: Optional[str] = None, metadata: _Metadata = None, **kwargs: Any) -> SendResult:
        return await self._send_local_file(chat_id, audio_path, caption, reply_to, metadata=metadata)

    async def send_video(self, chat_id: str, video_path: str, caption: Optional[str] = None,
                         reply_to: Optional[str] = None, metadata: _Metadata = None) -> SendResult:
        return await self._send_local_file(chat_id, video_path, caption, reply_to, metadata=metadata)

    def format_message(self, content: str) -> str:
        """Mattermost renders standard Markdown; reduce ![alt](url) to the bare URL (inline preview)."""
        return re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", r"\2", content)

    # --- File helpers ---

    async def _send_url_as_file(self, chat_id: str, url: str, caption: Optional[str], reply_to: Optional[str],
                                kind: str = "file", metadata: _Metadata = None) -> SendResult:
        """Download a URL and upload it as a file attachment (text fallback with the URL on failure)."""
        from tools.url_safety import is_safe_url

        async def fallback() -> SendResult:
            return await self.send(chat_id, f"{caption or ''}\n{url}".strip(), reply_to, metadata=metadata)

        if not is_safe_url(url):
            logger.warning("Mattermost: blocked unsafe URL (SSRF protection)")
            return await fallback()
        import aiohttp
        for attempt in range(3):  # retry 5xx/429 and network errors twice with linear backoff
            try:
                async with self._session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if (resp.status >= 500 or resp.status == 429) and attempt < 2:
                        logger.debug("Mattermost download retry %d/2 for %s (status %d)",
                                     attempt + 1, url[:80], resp.status)
                    elif resp.status >= 400:
                        return await fallback()
                    else:
                        file_data, ct = await resp.read(), resp.content_type or "application/octet-stream"
                        break
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                if attempt == 2:
                    logger.warning("Mattermost: failed to download %s after %d attempts: %s", url, attempt + 1, exc)
                    return await fallback()
            await asyncio.sleep(1.5 * (attempt + 1))
        file_id = await self._upload_file(chat_id, file_data, _url_filename(url, f"{kind}.png"), ct)
        return await self._post_with_file(chat_id, file_id, caption, reply_to, metadata) if file_id else await fallback()

    async def _send_local_file(
        self, chat_id: str, file_path: str, caption: Optional[str], reply_to: Optional[str],
        file_name: Optional[str] = None, metadata: _Metadata = None) -> SendResult:
        """Upload a local file and attach it to a post."""
        p = Path(file_path)
        if not p.exists():
            logger.warning("Mattermost: local file not found, skipping: %s", file_path)
            return SendResult(success=True, message_id=None)
        fname = file_name or p.name
        file_id = await self._upload_file(chat_id, p.read_bytes(), fname,
                                          mimetypes.guess_type(fname)[0] or "application/octet-stream")
        if not file_id:
            return SendResult(success=False, error="File upload failed")
        return await self._post_with_file(chat_id, file_id, caption, reply_to, metadata)

    async def _load_batch_image(self, image_url: str, index: int) -> Optional[Tuple[bytes, str, str]]:
        """Read a file:// or remote image for a batch post → (data, filename, content_type), or None to skip."""
        import aiohttp
        if image_url.startswith("file://"):
            local_path = _unquote(image_url[7:])
            p = Path(local_path)
            if not p.exists():
                logger.warning("Mattermost: skipping missing image %s", local_path)
                return None
            return p.read_bytes(), p.name, mimetypes.guess_type(p.name)[0] or "image/png"
        from tools.url_safety import is_safe_url
        if not is_safe_url(image_url):
            logger.warning("Mattermost: blocked unsafe image URL in batch")
            return None
        try:
            async with self._session.get(image_url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status >= 400:
                    logger.warning("Mattermost: failed to download image (HTTP %d): %s", resp.status, image_url[:80])
                    return None
                file_data, ct = await resp.read(), resp.content_type or "image/png"
        except Exception as dl_err:
            logger.warning("Mattermost: download failed for %s: %s", image_url[:80], dl_err)
            return None
        return file_data, _url_filename(image_url, f"image_{index}.png"), ct

    async def send_multiple_images(self, chat_id: str, images: List[Tuple[str, str]],
                                   metadata: _Metadata = None, human_delay: float = 0.0) -> SendResult:
        """Send a batch of images as one post; chunked at Mattermost's 5-``file_ids`` cap, each chunk
        falling back to the base per-image loop on failure."""
        if not images:
            return SendResult(success=False, error="no images to send")
        chunks = [images[i:i + 5] for i in range(0, len(images), 5)]  # Mattermost post file_ids cap
        delivered = False
        for chunk_idx, chunk in enumerate(chunks):
            if human_delay > 0 and chunk_idx > 0:
                await asyncio.sleep(human_delay)
            file_ids, caption_parts = [], []
            try:
                for image_url, alt_text in chunk:
                    if alt_text:
                        caption_parts.append(alt_text)
                    loaded = await self._load_batch_image(image_url, len(file_ids))
                    if loaded is not None and (fid := await self._upload_file(chat_id, *loaded)):
                        file_ids.append(fid)
                if not file_ids:
                    continue
                logger.info("Mattermost: sending %d image(s) as single post (chunk %d/%d)",
                            len(file_ids), chunk_idx + 1, len(chunks))
                data = await self._post_message(chat_id, "\n".join(caption_parts), None, metadata, file_ids)
                if data and "id" in data:
                    delivered = True
                else:
                    logger.warning("Mattermost: multi-image post failed, falling back")
                    fallback = await super().send_multiple_images(chat_id, chunk, metadata, human_delay=human_delay)
                    delivered = delivered or fallback.success
            except Exception as e:
                logger.warning("Mattermost: multi-image send failed (chunk %d/%d), falling back: %s",
                               chunk_idx + 1, len(chunks), e, exc_info=True)
                fallback = await super().send_multiple_images(chat_id, chunk, metadata, human_delay=human_delay)
                delivered = delivered or fallback.success
        return SendResult(success=delivered, error=None if delivered else "all images failed to send")

    # --- WebSocket ---

    async def _ws_loop(self) -> None:
        """Connect to the WebSocket and listen for events, reconnecting on failure."""
        import aiohttp
        import random
        delay = _RECONNECT_BASE_DELAY
        while not self._closing:
            try:
                await self._ws_connect_and_listen()
                delay = _RECONNECT_BASE_DELAY  # clean disconnect — reset backoff
            except asyncio.CancelledError:
                return
            except Exception as exc:
                if self._closing:
                    return
                # Permanent auth failure: escalate via the fatal-error hook (a bare return leaves is_connected()
                # healthy with a dead listener). Type-based: substring "401" matching misclassified transient errors.
                if isinstance(exc, aiohttp.WSServerHandshakeError) and exc.status in {401, 403}:
                    logger.error("Mattermost WS auth failed (HTTP %d) — stopping reconnect", exc.status)
                    # Escalate through the fatal-error hook instead of a bare return: the old silent exit
                    # left _running True, so is_connected() kept reporting healthy while the listener was
                    # dead and the gateway was never told (OOF-156 class). Type-based only — the substring
                    # fallback that used to sit below this branch misclassified transient errors whose
                    # message merely contained "401" (#80489).
                    self._set_fatal_error(
                        "mattermost_auth_error",
                        f"Mattermost WebSocket authentication rejected (HTTP {exc.status}). The bot token is "
                        "invalid, revoked, or lacks permission — check MATTERMOST_TOKEN and the bot account in "
                        "the System Console.", retryable=False)
                    await self._notify_fatal_error()
                    return
                logger.warning("Mattermost WS error: %s — reconnecting in %.0fs", exc, delay)
            if self._closing:
                return
            await asyncio.sleep(delay + delay * _RECONNECT_JITTER * random.random())
            delay = min(delay * 2, _RECONNECT_MAX_DELAY)

    async def _ws_connect_and_listen(self) -> None:
        """Single WebSocket session: connect, authenticate, process events."""
        ws_url = re.sub(r"^http", "ws", self._base_url) + "/api/v4/websocket"  # https→wss, http→ws
        logger.info("Mattermost: connecting to %s", ws_url)
        self._ws = await self._session.ws_connect(ws_url, heartbeat=30.0)
        await self._ws.send_json({"seq": 1, "action": "authentication_challenge", "data": {"token": self._token}})
        logger.info("Mattermost: WebSocket connected and authenticated")

        async for raw_msg in self._ws:
            if self._closing:
                return
            kind = raw_msg.type
            if kind in {kind.TEXT, kind.BINARY}:
                try:
                    event = json.loads(raw_msg.data)
                except (json.JSONDecodeError, TypeError):
                    continue
                await self._handle_ws_event(event)
            elif kind in {kind.ERROR, kind.CLOSE, kind.CLOSING, kind.CLOSED}:
                logger.info("Mattermost: WebSocket closed (%s)", kind)
                break

    def _apply_channel_gating(self, channel_id: str, message_text: str) -> Optional[str]:
        """Mention-gate a non-DM post; return the cleaned text, or None to ignore it. allowed_channels is a
        whitelist checked first (@mentions elsewhere are ignored); require_mention (default true) is
        bypassed in free_response_channels."""
        allowed_channels = _channel_id_set(_extra_or_secret(self.config.extra, "allowed_channels", "MATTERMOST_ALLOWED_CHANNELS", blank_is_unset=False))
        if allowed_channels and channel_id not in allowed_channels:
            logger.debug("Mattermost: ignoring message in non-allowed channel: %s", channel_id)
            return None
        require_mention = str(_extra_or_secret(self.config.extra, "require_mention", "MATTERMOST_REQUIRE_MENTION", "true", blank_is_unset=False)
                              ).lower() not in {"false", "0", "no"}
        free_channels = _channel_id_set(
            _extra_or_secret(self.config.extra, "free_response_channels", "MATTERMOST_FREE_RESPONSE_CHANNELS", blank_is_unset=False))
        mention_patterns = [f"@{self._bot_username}", f"@{self._bot_user_id}"]
        has_mention = any(pattern.lower() in message_text.lower() for pattern in mention_patterns)
        if require_mention and channel_id not in free_channels and not has_mention:
            logger.debug("Mattermost: skipping non-DM message without @mention (channel=%s)", channel_id)
            return None
        if has_mention:  # strip the @mention so the agent sees clean input
            for pattern in mention_patterns:
                message_text = re.sub(re.escape(pattern), "", message_text, flags=re.IGNORECASE).strip()
        return message_text

    async def _download_attachments(self, file_ids: List[str]) -> Tuple[List[str], List[str]]:
        """Download attachments now (URLs need auth headers downstream tools lack) → (paths, mime types)."""
        import aiohttp
        from gateway.platforms.base import (
            cache_audio_from_bytes_async,
            cache_document_from_bytes_async,
            cache_image_from_bytes_async,
        )
        media_urls, media_types = [], []
        cache_fns = {"image/": cache_image_from_bytes_async, "audio/": cache_audio_from_bytes_async}
        for fid in file_ids:
            try:
                file_info = await self._api_get(f"files/{fid}/info")
                fname = file_info.get("name", f"file_{fid}")
                mime = file_info.get("mime_type", "application/octet-stream")
                async with self._session.get(
                    f"{self._base_url}/api/v4/files/{fid}", headers=self._auth_header(),
                    timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status >= 400:
                        logger.warning("Mattermost: failed to download file %s: HTTP %s", fid, resp.status)
                        continue
                    file_data = await resp.read()
                    prefix = next((p for p in cache_fns if mime.startswith(p)), None)
                    if prefix:
                        media_urls.append(
                            await cache_fns[prefix](file_data, Path(fname).suffix or _INBOUND_CACHE_EXT[prefix]))
                    else:
                        media_urls.append(await cache_document_from_bytes_async(file_data, fname))
                    media_types.append(mime)
            except Exception as exc:
                logger.warning("Mattermost: error downloading file %s: %s", fid, exc)
        return media_urls, media_types

    async def _handle_ws_event(self, event: Dict[str, Any]) -> None:
        event_type = event.get("event")
        # Reactions arrive as a separate WS event; handle them BEFORE the
        # ``posted`` filter so we don't need to fake a post payload.
        if event_type == "reaction_added":
            reaction_data = event.get("data", {})
            # v2: clarify pickers are registered BEFORE approval entries on the
            # same post (they share the bot's reaction card UX), so check the
            # picker registry first — a tap on a picker card must not be
            # interpreted as an approval tap. _parse_reaction_event is the
            # shared parser; both handlers reuse it.
            post_id, _user_id, _emoji = _parse_reaction_event(reaction_data)
            if post_id and post_id in self._clarify_picker_prompts_by_event:
                await self._handle_picker_reaction(reaction_data)
            else:
                await self._handle_approval_reaction(reaction_data)
            return
        if event_type != "posted":
            return
        data = event.get("data", {})
        try:
            post = json.loads(data.get("post") or "")
        except (json.JSONDecodeError, TypeError):
            return
        # Ignore own messages, system posts and redeliveries.
        sender_id, post_id = post.get("user_id", ""), post.get("id", "")
        if sender_id == self._bot_user_id or post.get("type") or self._dedup.is_duplicate(post_id):
            return
        channel_id, is_dm = post.get("channel_id", ""), data.get("channel_type", "O") == "D"
        message_text = post.get("message", "")
        if not is_dm:  # DMs need no gating; channels are mention-gated.
            message_text = self._apply_channel_gating(channel_id, message_text)
            if message_text is None:
                return
        # Thread support: replies use root_id; in thread mode a top-level channel post is itself a valid root.
        thread_id = post.get("root_id") or None
        if not thread_id and self._reply_mode == "thread" and not is_dm and post_id:
            thread_id = post_id
        if message_text[:1].isspace() and message_text.lstrip().startswith("/"):
            message_text = message_text.lstrip()
        media_urls, media_types = await self._download_attachments(post.get("file_ids") or [])
        # Classification restored 2026-09-27 (regression in v1; see PR #124249 review).
        # Precedence matches main: slash → COMMAND; media → PHOTO/VOICE/DOCUMENT;
        # else TEXT. The MessageType.TEXT hardcode that v1 introduced dropped this
        # block and broke test_mattermost.py::test_leading_space_slash_command_is_command.
        if message_text.startswith("/"):
            msg_type = MessageType.COMMAND
        elif media_types:
            msg_type = next(
                (mt for prefix, mt in _MEDIA_MSG_TYPES if any(m.startswith(prefix) for m in media_types)),
                MessageType.DOCUMENT)
        else:
            msg_type = MessageType.TEXT
        source = self.build_source(
            chat_id=channel_id, chat_type=_CHANNEL_TYPE_MAP.get(data.get("channel_type", "O"), "channel"),
            user_id=sender_id, user_name=data.get("sender_name", "").lstrip("@") or sender_id,
            thread_id=thread_id, message_id=post_id)
        from gateway.platforms.base import resolve_channel_prompt
        await self.handle_message(MessageEvent(
            text=message_text, message_type=msg_type, source=source, raw_message=post, message_id=post_id,
            media_urls=media_urls or None, media_types=media_types or None,
            channel_prompt=resolve_channel_prompt(self.config.extra, channel_id, None)))

    # ── Reaction-driven approval (sloth 2026-09-26) ──

    async def _handle_approval_reaction(self, data: Dict[str, Any]) -> None:
        """``reaction_added`` → resolve the matching pending approval, if any.

        Multiplex-safe: the lookup key is the bot's own approval ``post_id``
        (carried in the WS event), and the value is the ``session_key`` the
        gateway runner is waiting on. No public HTTP callback URL needed —
        the reaction rides the bot's existing WebSocket.

        WS event payload shape (verified against Mattermost master
        ``app/reaction.go::sendReactionEvent``):
            data.reaction = JSON-encoded ``model.Reaction`` string
            = {"user_id","post_id","emoji_name","create_at"}
        """
        post_id, user_id, emoji_name = _parse_reaction_event(data)
        if not post_id or not user_id or not emoji_name:
            return
        # v2: adapter-side allowlist gate, sourced from MATTERMOST_ALLOWED_USERS.
        # Mirrors matrix's _is_authorized_user (matrix/adapter.py:2536). An
        # unauthorized user tapping a bot-seeded reaction must NOT resolve
        # the pending approval — only allowlisted users can drive approvals.
        if not _is_user_authorized_mattermost(self._allowed_user_ids, user_id):
            logger.info(
                "Mattermost: ignoring approval reaction from unauthorized user %s on %s",
                user_id, post_id)
            return
        entry = self._approval_prompts_by_event.get(post_id)
        if not entry:
            return  # not an approval card, or already resolved
        bot_user_id = str(getattr(self, "_bot_user_id", "") or "")
        if bot_user_id and user_id == bot_user_id:
            logger.debug(
                "Mattermost: ignoring self-reaction %s on %s by bot %s",
                emoji_name, post_id, bot_user_id)
            return
        choice = _REACTION_TO_CHOICE.get(emoji_name)
        if not choice:
            return  # white_check_mark / cyclone / infinity / no_entry_sign only
        # Double-click guard: resolve only once per post.
        if self._approval_resolved.get(post_id):
            return
        self._approval_resolved[post_id] = True
        session_key = entry["session_key"]
        # v2: pass request_id so a tap on card B resolves only card B's command,
        # not the session's FIFO head (tools/approval.py:141).
        request_id = entry.get("request_id") or None
        # Resolve FIRST (unblocks the agent thread) so a tap landing after a
        # timeout (count == 0) doesn't lie about the resolution. Same pattern as
        # Telegram ``_handle_exec_approval_callback``.
        count = 0
        try:
            from tools.approval import resolve_gateway_approval
            count = resolve_gateway_approval(session_key, choice, request_id=request_id)
            logger.info("Mattermost reaction resolved %d approval(s) for session %s "
                        "(choice=%s, user=%s, post=%s, request_id=%s)",
                        count, session_key, choice, user_id, post_id, request_id or "")
        except Exception as exc:
            logger.error("Failed to resolve gateway approval from Mattermost reaction: %s", exc)
        # Tidy the registry regardless of outcome.
        self._approval_prompts_by_event.pop(post_id, None)
        # Remove ALL the seeded emoji (✅ 🌀 ♾️ ❌), not just the one the user
        # tapped. v3 fix for the persistent "card stays cluttered with 3
        # emoji after I tapped one" bug — same pattern as the picker handler
        # (`for emoji in picker.choices: delete`). Failures here are silent;
        # they're cosmetic and the post is already resolved.
        for seeded_emoji in entry.get("seeded_emojis") or (emoji_name,):
            asyncio.create_task(self._delete_reaction(post_id, seeded_emoji))
        if count:
            label = _REACTION_LABEL.get(choice, "Resolved")
            try:
                await self.edit_message(
                    entry["chat_id"], post_id,
                    f"{label}",
                    finalize=True)
            except Exception as exc:
                logger.debug("Mattermost: failed to edit approval card after resolve: %s", exc)

    async def _handle_picker_reaction(self, data: Dict[str, Any]) -> None:
        """``reaction_added`` → resolve a pending clarify-picker card.

        Mirrors matrix's ``_handle_choice_picker_reaction``
        (matrix/adapter.py:2504). The picker card is the bot's own message with
        one reaction per choice (1️⃣, 2️⃣, …); a tap from a non-bot user maps
        the reaction back to the choice label and resolves the underlying
        clarify via ``tools.clarify_gateway.resolve_gateway_clarify``.
        """
        post_id, user_id, emoji_name = _parse_reaction_event(data)
        if not post_id or not user_id or not emoji_name:
            return
        picker = self._clarify_picker_prompts_by_event.get(post_id)
        if not picker:
            return  # not one of ours (or already tidied)
        if picker.resolved:
            return
        # v2: same allowlist gate as approval.
        if not _is_user_authorized_mattermost(self._allowed_user_ids, user_id):
            logger.info(
                "Mattermost: ignoring picker reaction from unauthorized user %s on %s",
                user_id, post_id)
            return
        bot_user_id = str(getattr(self, "_bot_user_id", "") or "")
        if bot_user_id and user_id == bot_user_id:
            return  # bot's own seeded reaction
        # TTL guard: if the picker expired, drop the entry and ignore.
        # A picker with ``expires_at=0`` (Unix epoch) is treated as already
        # expired — covers both the legacy uninitialized shape and explicit
        # "always-expired" sentinels from tests.
        import time as _time
        if _time.time() > float(picker.expires_at):
            self._clarify_picker_prompts_by_event.pop(post_id, None)
            return
        if emoji_name not in picker.choices:
            return  # not a choice emoji for this picker
        picker.resolved = True
        idx = picker.choices.index(emoji_name)
        response = picker.responses[idx]
        from tools.clarify_gateway import resolve_gateway_clarify
        resolved = False
        try:
            resolved = resolve_gateway_clarify(picker.clarify_id, response)
        except Exception as exc:
            logger.error("Failed to resolve gateway clarify from Mattermost picker: %s", exc)
        if not resolved:
            # Already-resolved or unknown clarify id (likely a duplicate tap).
            return
        logger.info("Mattermost picker resolved clarify %s → %r (user=%s, post=%s)",
                    picker.clarify_id, response, user_id, post_id)
        # Tidy: drop the entry and retract the bot's seeded reactions so the
        # user sees the card is done.
        self._clarify_picker_prompts_by_event.pop(post_id, None)
        for emoji in picker.choices:
            asyncio.create_task(self._delete_reaction(post_id, emoji))
        try:
            await self.edit_message(
                picker.chat_id, post_id,
                f"✅ {response}",
                finalize=True,
            )
        except Exception as exc:
            logger.debug("Mattermost: failed to edit picker card after resolve: %s", exc)

    async def _send_reaction(self, post_id: str, emoji_name: str) -> bool:
        """Add a reaction to a post (``POST /api/v4/reactions`` with JSON body).

        The reaction create endpoint takes ``{"user_id", "post_id", "emoji_name"}``
        in the request body, NOT URL path params — verified against
        ``api/v4/source/reactions.yaml`` in upstream Mattermost master.

        Earlier confusion: the URL ``/api/v4/posts/{id}/reactions/{emoji_name}``
        is the DELETE endpoint (path-encoded emoji), not POST. POST lives at the
        top-level ``/api/v4/reactions`` collection and uses a JSON body. The bot's
        user id is required because ``saveReaction`` enforces
        ``reaction.UserId == session.UserId`` (forbid impersonation).
        """
        if not self._session or not self._bot_user_id:
            return False
        import aiohttp
        url = f"{self._base_url}/api/v4/reactions"
        body = {"user_id": self._bot_user_id, "post_id": post_id, "emoji_name": emoji_name}
        try:
            async with self._session.post(
                url,
                json=body,
                headers=self._auth_header(),
                timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status in {200, 201}:
                    return True
                # 409 Conflict = we already seeded it; that's fine.
                if resp.status == 409:
                    return True
                err_body = await resp.text()
                logger.warning("Mattermost reaction seed FAIL url=%s body=%s status=%s resp=%s",
                               url, body, resp.status, err_body[:400])
                return False
        except Exception as exc:  # noqa: BLE001
            logger.warning("Mattermost: failed to seed reaction %s on %s: %s",
                           emoji_name, post_id, exc)
            return False

    async def _delete_reaction(self, post_id: str, emoji_name: str) -> bool:
        """Remove the bot's own reaction from a post (used after a tap resolves the card)."""
        if not self._session or not self._bot_user_id:
            return False
        import aiohttp
        try:
            async with self._session.delete(
                f"{self._base_url}/api/v4/users/{self._bot_user_id}/posts/{post_id}/reactions/{emoji_name}",
                headers=self._auth_header(),
                timeout=aiohttp.ClientTimeout(total=15)) as resp:
                # 204 No Content = gone; 404 = wasn't there, also fine.
                return resp.status in {200, 204, 404}
        except Exception as exc:  # noqa: BLE001
            logger.debug("Mattermost: failed to delete reaction %s on %s: %s",
                         emoji_name, post_id, exc)
            return False

    async def _send_exec_approval_prompt(self, prompt) -> Any:  # noqa: ANN401 — ExecApprovalPrompt type
        """Render the shared approval card as plain text, register the prompt,
        then seed one reaction per offered choice on the bot's own post. The
        user can tap an emoji or type ``/approve`` / ``/deny`` — both routes
        resolve via ``tools.approval.resolve_gateway_approval``.

        Mirrors the Matrix reaction-driven flow at ``plugins/platforms/matrix/adapter.py``
        :func:`_send_exec_approval_prompt` (the same ``prompt.text`` and
        ``prompt.choices`` contract from the gateway base template). v2: emoji
        set is matrix-parity (✅ once / 🌀 session / ♾️ always / 🚫 deny), and
        only the choices the prompt actually offers are seeded — when
        ``allow_permanent=False`` the bot skips ♾️, etc.
        """
        # Bound the registry — drop oldest entries if the cap is exceeded.
        if len(self._approval_prompts_by_event) >= _APPROVAL_PROMPTS_MAX:
            oldest = next(iter(self._approval_prompts_by_event))
            self._approval_prompts_by_event.pop(oldest, None)
            self._approval_resolved.pop(oldest, None)
        result = await self.send(prompt.chat_id, prompt.text, reply_to=None,
                                 metadata=prompt.metadata)
        if not result.success or not result.message_id:
            return result
        post_id = str(result.message_id)
        requester = ""
        if isinstance(prompt.metadata, dict):
            requester = str(prompt.metadata.get("requester_user_id") or "")
        choices = list(getattr(prompt, "choices", []) or [])
        # Only seed emojis that the user could mean (matrix legend is the
        # canonical mapping; emoji short names are what Mattermost's v4
        # reaction API accepts in POST /api/v4/reactions).
        emojis = [_EMOJI_FOR_CHOICE[c] for c in choices if c in _EMOJI_FOR_CHOICE]
        self._approval_prompts_by_event[post_id] = {
            "session_key": prompt.session_key,
            "requester_user_id": requester,
            "chat_id": prompt.chat_id,
            "choices": list(choices),
            # v2: store request_id so a tap on card B resolves only B's command
            # (not the session's FIFO head). Default '' = legacy adapter path.
            "request_id": getattr(prompt, "request_id", "") or "",
            # v3: remember the emoji short names we actually seeded so a tap
            # can retract the *whole* approval card (✅ 🌀 ♾️ ❌), not just
            # the one emoji the user touched. Same pattern as
            # ``_MattermostClarifyPicker.choices``.
            "seeded_emojis": list(emojis),
        }
        # Seed reactions AFTER the post is on the server — Mattermost returns the
        # post id synchronously from POST /posts so this is safe to fire next.
        # Inline await (no ``create_task``): keeps the post+reactions bundle
        # visible to the test/observer in the right order; the HTTP round-trips
        # are sub-millisecond on a healthy local LAN.
        for emoji in emojis:
            ok = await self._send_reaction(post_id, emoji)
            # DEBUG, not INFO: lets an operator trace which short_name failed
            # to seed if a card ever lands short of reactions, without
            # spamming production logs.
            logger.debug(
                "Mattermost approval seed: post=%s emoji=%s ok=%s choices=%s",
                post_id, emoji, ok, list(emojis))
        return result

    # ── Clarify (choice-picker) — v2 2026-09-27 ─────────────────────────────
    async def send_clarify(
        self, chat_id: str, question: str, choices: Optional[list], clarify_id: str,
        session_key: str, metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        """Override base send_clarify with a reaction-driven picker (matrix parity).

        Behavior:
        - No choices (open-ended): fall through to ``super().send_clarify`` so the
          gateway text-intercept captures the next message (numbered-list UX).
        - 1–12 choices: post the question with one reaction per choice (1️⃣ 2️⃣ …)
          and register a ``_MattermostClarifyPicker``. The reaction handler resolves
          via ``tools.clarify_gateway.resolve_gateway_clarify``.
        - >12 choices: cap at 12; remaining choices are dropped (mirror matrix).
          Open-ended tail is preserved as a fallback line.
        """
        if not choices:
            return await super().send_clarify(
                chat_id, question, choices, clarify_id, session_key, metadata)
        flat_choices = [str(c) for c in choices][:len(_MATTERMOST_PICKER_REACTIONS)]
        if not flat_choices:
            return await super().send_clarify(
                chat_id, question, choices, clarify_id, session_key, metadata)
        reactions = list(_MATTERMOST_PICKER_REACTIONS[:len(flat_choices)])
        # Bound the registry; drop the oldest entry on overflow so a stuck
        # gateway can't grow this dict unbounded.
        if len(self._clarify_picker_prompts_by_event) >= _CLARIFY_PICKER_MAX:
            oldest = next(iter(self._clarify_picker_prompts_by_event))
            self._clarify_picker_prompts_by_event.pop(oldest, None)
        # Build the card body — single-row options with tap-or-type hint
        # replaces the legacy plain inline list. Pure helper, easy to unit-test.
        from plugins.platforms.mattermost.adapter_clarify import (
            format_clarify_picker_body, numeric_labels_for,
        )
        body = format_clarify_picker_body(
            question=question, choices=reactions, responses=flat_choices,
        )
        # Send the card. If send fails, fall back to the base text path so the
        # user can still answer by typing.
        result = await self.send(chat_id, body, metadata=metadata)
        if not result.success or not result.message_id:
            return await super().send_clarify(
                chat_id, question, choices, clarify_id, session_key, metadata)
        post_id = str(result.message_id)
        import time as _time
        self._clarify_picker_prompts_by_event[post_id] = _MattermostClarifyPicker(
            chat_id=chat_id, post_id=post_id, clarify_id=clarify_id,
            choices=reactions, responses=flat_choices,
            numeric_labels=numeric_labels_for(reactions),
            expires_at=_time.time() + _CLARIFY_PICKER_TTL_SECONDS,
        )
        # Seed reactions AFTER the post is on the server — same ordering as
        # approval; Mattermost returns the post id synchronously from POST /posts
        # so this is safe to fire inline. Inline await keeps the card+reactions
        # bundle visible to the test/observer in the right order.
        for emoji in reactions:
            await self._send_reaction(post_id, emoji)
        return result


# --- Plugin standalone-send (out-of-process cron delivery via Mattermost REST) ---

async def _standalone_send(pconfig, chat_id: str, message: str, *, thread_id: Optional[str] = None,
                           media_files: Optional[list] = None, force_document: bool = False) -> Dict[str, Any]:
    """Send via the Mattermost v4 REST API without a live gateway adapter (out-of-process cron).

    Token/URL: ``pconfig`` with env fallback. ``media_files`` upload via ``POST /files`` and attach by
    file_id; ``thread_id`` becomes ``root_id``. ``force_document`` is signature parity only (unused).
    """
    try:
        import aiohttp
    except ImportError:
        return send_error("aiohttp not installed. Run: pip install aiohttp")

    base_url, token = _url_and_token(pconfig)
    base_url, token = base_url.rstrip("/"), token.strip()
    if not base_url or not token:
        return send_error("Mattermost standalone send: MATTERMOST_URL and MATTERMOST_TOKEN must both be set")
    upload_headers = {"Authorization": f"Bearer {token}"}
    headers = {**upload_headers, "Content-Type": "application/json"}
    try:
        # One ClientSession (with proxy) covers the optional uploads + final post.
        from gateway.platforms.base import resolve_proxy_url, proxy_kwargs_for_aiohttp
        _sess_kw, _req_kw = proxy_kwargs_for_aiohttp(resolve_proxy_url(platform_env_var="MATTERMOST_PROXY"))
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60), **_sess_kw) as session:
            file_ids: List[str] = []
            for media in media_files or []:
                file_path = media.get("path") if isinstance(media, dict) else media
                if not file_path or not os.path.exists(file_path):
                    continue
                form = aiohttp.FormData()
                form.add_field("channel_id", chat_id)  # required so the server can attribute the upload
                with open(file_path, "rb") as fh:
                    form.add_field("files", fh.read(), filename=os.path.basename(file_path))
                async with session.post(f"{base_url}/api/v4/files", data=form, headers=upload_headers,
                                        **_req_kw) as upload_resp:
                    if upload_resp.status not in {200, 201}:
                        body = await upload_resp.text()
                        return send_error(f"Mattermost file upload failed ({upload_resp.status}): {body[:400]}")
                    upload_data = await upload_resp.json()
                    file_ids.extend(info["id"] for info in upload_data.get("file_infos", []) if info.get("id"))
            payload: Dict[str, Any] = {"channel_id": chat_id, "message": message}
            if thread_id:
                payload["root_id"] = thread_id
            if file_ids:
                payload["file_ids"] = file_ids
            async with session.post(f"{base_url}/api/v4/posts", headers=headers, json=payload, **_req_kw) as resp:
                if resp.status not in {200, 201}:
                    body = await resp.text()
                    return send_error(f"Mattermost API error ({resp.status}): {body[:400]}")
                data = await resp.json()
            return {"success": True, "platform": "mattermost", "chat_id": chat_id, "message_id": data.get("id")}
    except aiohttp.ClientError as exc:
        return send_error(f"Mattermost send failed (network): {exc}")
    except Exception as exc:  # noqa: BLE001
        return send_error(f"Mattermost send failed: {exc}")


# --- Interactive setup wizard ---

def interactive_setup() -> None:
    """Guide the user through Mattermost bot setup (URL + token, allowlist, home channel)."""
    from hermes_cli.config import remove_env_value, save_env_value
    from hermes_cli.cli_output import prompt, print_header, print_info, print_success
    from hermes_cli.setup_platforms import declines_reconfigure

    def info(*lines: str) -> None:
        for line in lines:
            print_info(line)

    print_header("Mattermost")
    if declines_reconfigure("Mattermost", "Reconfigure Mattermost?", "MATTERMOST_TOKEN"):
        return
    info("Works with any self-hosted Mattermost instance.",
         "   1. In Mattermost: Integrations → Bot Accounts → Add Bot Account", "   2. Copy the bot token")
    print()
    mm_url = prompt("Mattermost server URL (e.g. https://mm.example.com)")
    if mm_url:
        save_env_value("MATTERMOST_URL", mm_url.rstrip("/"))
    token = prompt("Bot token", password=True)
    if not token:
        return
    save_env_value("MATTERMOST_TOKEN", token)
    print_success("Mattermost token saved")
    print()
    info("🔒 Security: Restrict who can use your bot", "   To find your user ID: click your avatar → Profile",
         "   or use the API: GET /api/v4/users/me")
    print()
    allowed_users = prompt("Allowed user IDs (comma-separated, leave empty for open access)")
    if allowed_users:
        save_env_value("MATTERMOST_ALLOWED_USERS", allowed_users.replace(" ", ""))
        print_success("Mattermost allowlist configured")
    else:
        print_info("⚠️  No allowlist set - anyone who can message the bot can use it!")
    print()
    info("📬 Home Channel: where Hermes delivers cron job results and notifications.",
         "   To get a channel ID: click channel name → View Info → copy the ID",
         "   You can also set this later by typing /set-home in a Mattermost channel.")
    home_channel = prompt("Home channel ID (leave empty to set later with /set-home)").strip()
    if home_channel:
        save_env_value("MATTERMOST_HOME_CHANNEL", home_channel)
    elif remove_env_value("MATTERMOST_HOME_CHANNEL"):
        print_info("Home channel cleared.")
    print_info("   Open config in your editor:  hermes config edit")


# --- YAML → env config bridge (apply_yaml_config_fn) ---

_YAML_BRIDGE = (  # (yaml key, env var, kind) for apply_yaml_bridge; allowed_channels is a whitelist
    ("require_mention", "MATTERMOST_REQUIRE_MENTION", "lower"),
    ("free_response_channels", "MATTERMOST_FREE_RESPONSE_CHANNELS", "csv"),
    ("allowed_channels", "MATTERMOST_ALLOWED_CHANNELS", "csv"))


def _apply_yaml_config(yaml_cfg: dict, mattermost_cfg: dict) -> dict | None:
    """``apply_yaml_config_fn`` (#24836 / #25443): ``config.yaml`` ``mattermost:`` keys → env vars (env wins;
    skipped under a multiplexed secondary profile) + ``PlatformConfig.extra`` (extra-first readers)."""
    return _apply_yaml_bridge(mattermost_cfg, _YAML_BRIDGE)



_is_connected = _env_is_connected("MATTERMOST_TOKEN", "MATTERMOST_URL")



def register(ctx) -> None:
    """Plugin entry point — called by the Hermes plugin system."""
    ctx.register_platform(
        name="mattermost", label="Mattermost", adapter_factory=MattermostAdapter,
        check_fn=check_mattermost_requirements, validate_config=validate_mattermost_config,
        is_connected=_is_connected, required_env=["MATTERMOST_URL", "MATTERMOST_TOKEN"],
        install_hint="pip install aiohttp", setup_fn=interactive_setup,
        apply_yaml_config_fn=_apply_yaml_config,  # YAML→env bridge (see _YAML_BRIDGE)
        allowed_users_env="MATTERMOST_ALLOWED_USERS", allow_all_env="MATTERMOST_ALLOW_ALL_USERS",
        cron_deliver_env_var="MATTERMOST_HOME_CHANNEL",
        standalone_sender_fn=_standalone_send,  # out-of-process cron; without it `deliver=mattermost` fails
        max_message_length=MAX_POST_LENGTH, emoji="💬", allow_update_command=True)

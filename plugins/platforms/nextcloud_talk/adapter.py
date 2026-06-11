"""Nextcloud Talk platform adapter (Hermes bundled plugin).

Connects Hermes to Nextcloud Talk via the User API (Basic Auth +
App Password) using OCS long-polling. No bot framework, no HMAC
webhook setup — Hermes appears as a regular Nextcloud user in
whatever conversations the operator joins it to.

This adapter ships as a bundled platform plugin under
``plugins/platforms/nextcloud_talk/``. The Hermes plugin loader
discovers it at startup, calls :func:`register`, and the platform
becomes available to ``gateway/run.py`` and ``tools/send_message_tool``
through the platform registry — no core file edits required.

Configuration in config.yaml::

    platforms:
      nextcloud_talk:
        enabled: true
        extra:
          conversations:                  # required — list of conversations to join
            - token: "abc123def"
              alias: "general"            # optional friendly name for routing
            - token: "ghi456jkl"
              alias: "alerts"
          channel_aliases:                # optional — extra alias → token map
            "ops": "abc123def"
          edit_ack_into_response: true    # default true; edit "⏳ Thinking..." into reply
          show_status_updates: true       # default true; send "⏳ Thinking..." ack
          poll_timeout: 30                # default 30 (max 60)
          stt:                            # optional speech-to-text for voice messages
            provider: llamacpp
            base_url: "http://stt-host:8094/v1"
            api_key: "..."
            language: "de"

Environment variables (all read at adapter construct time; env wins
over config.yaml ``extra``):

    NEXTCLOUD_TALK_URL              Base URL of your Nextcloud instance (required)
    NEXTCLOUD_TALK_USERNAME         Nextcloud user the bot logs in as (required)
    NEXTCLOUD_TALK_APP_PASSWORD     Nextcloud App Password (required, sensitive)
    NEXTCLOUD_TALK_CONVERSATIONS    Comma-separated conversation tokens (alt to config.yaml)
    NEXTCLOUD_TALK_ALLOWED_USERS    Allowlist of NC usernames
    NEXTCLOUD_TALK_ALLOW_ALL_USERS  Allow any participant — dev only
    NEXTCLOUD_TALK_HOME_CHANNEL     Default conversation for cron / notification delivery
    NEXTCLOUD_TALK_HOME_CHANNEL_NAME  Human label for the home channel
    NEXTCLOUD_TALK_POLL_TIMEOUT     Long-poll timeout (default 30, max 60)
    NEXTCLOUD_TALK_SHOW_STATUS_UPDATES  Send "⏳ Thinking..." acks (default true)

Identity model: Nextcloud Talk has a native user identity (``actorId``).
Allowlists are matched against ``actorId`` (the Nextcloud username, e.g.
``alice`` — not a display name and not an email). System messages and
the bot's own messages are filtered out before allowlist evaluation.
"""

import asyncio
import json
import logging
import os
import tempfile
import uuid
from collections import deque
from typing import Any, Deque, Dict, List, Optional

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

from gateway.config import PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)

logger = logging.getLogger(__name__)

# Portable across platforms — resolves to /tmp on Linux/macOS and the
# user's temp directory on native Windows.
MEDIA_TEMP_DIR = os.path.join(tempfile.gettempdir(), "hermes-media")

# Nextcloud Talk hard limit is 32 KB per message. Hermes splits at 4 KB
# so messages stay readable in mobile clients.
MAX_MESSAGE_LENGTH = 4096


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _parse_user_message(msg: dict, *, own_user_id: str) -> Optional[dict]:
    """Parse a Talk User-API message dict into a normalized dict.

    Returns None for own messages and system messages.
    """
    if msg.get("actorId") == own_user_id:
        return None
    if msg.get("systemMessage"):
        return None

    text = msg.get("message", "")
    attachment = None

    params = msg.get("messageParameters") or {}
    file_info = params.get("file")
    if isinstance(file_info, dict) and file_info.get("type") == "file":
        attachment = {
            "type": "file",
            "id": str(file_info.get("id", "")),
            "name": str(file_info.get("name", "")),
            "path": str(file_info.get("path", "")),
            "link": str(file_info.get("link", "")),
            "mimetype": str(file_info.get("mimetype", "")),
            "size": str(file_info.get("size", "0")),
        }
        text = text.replace("{file}", "").strip()

    return {
        "text": text,
        "message_id": msg.get("id"),
        "chat_id": msg.get("token", ""),
        "user_id": msg.get("actorId", ""),
        "user_name": msg.get("actorDisplayName", ""),
        "attachment": attachment,
    }


def _classify_attachment(mimetype: str) -> str:
    """Classify a mimetype into a handler category."""
    if mimetype.startswith("image/"):
        return "image"
    if mimetype.startswith("audio/") or mimetype.startswith("video/"):
        return "audio"
    if mimetype.startswith("text/") or mimetype in (
        "application/pdf", "application/json", "application/xml",
    ):
        return "document"
    return "other"


# ─────────────────────────────────────────────────────────────────────
# Plugin entry-point hooks
# ─────────────────────────────────────────────────────────────────────

def check_requirements() -> bool:
    """Check that this adapter is installable.

    A pure dependency check — does NOT inspect env vars (those are checked
    by ``validate_config()`` once the gateway config has been loaded, since
    secrets may come from config.yaml ``extra`` instead of env directly).
    """
    return HTTPX_AVAILABLE


def validate_config(config) -> bool:
    """Validate that the configured platform has the bare minimum to start."""
    extra = getattr(config, "extra", {}) or {}
    has_url = bool(extra.get("nextcloud_url") or os.getenv("NEXTCLOUD_TALK_URL", "").strip())
    has_user = bool(extra.get("username") or os.getenv("NEXTCLOUD_TALK_USERNAME", "").strip())
    # Same env-var indirection as the adapter __init__ — a platform configured
    # with a custom app_password_env must not fail validation just because the
    # DEFAULT variable is unset.
    pw_env = str(extra.get("app_password_env", "NEXTCLOUD_TALK_APP_PASSWORD"))
    has_pw = bool(os.getenv(pw_env, "").strip())
    has_conversations = bool(extra.get("conversations") or os.getenv("NEXTCLOUD_TALK_CONVERSATIONS", "").strip())
    return has_url and has_user and has_pw and has_conversations


def is_connected(config) -> bool:
    """Check whether Nextcloud Talk is configured (env or config.yaml)."""
    return validate_config(config)


def _env_enablement() -> Optional[dict]:
    """Seed ``PlatformConfig.extra`` from env vars during gateway config load.

    Returns ``None`` when the platform isn't minimally configured.
    The special ``home_channel`` key is handled by the core hook —
    it becomes a proper ``HomeChannel`` dataclass on the ``PlatformConfig``
    rather than being merged into ``extra``.
    """
    url = os.getenv("NEXTCLOUD_TALK_URL", "").strip()
    user = os.getenv("NEXTCLOUD_TALK_USERNAME", "").strip()
    pw = os.getenv("NEXTCLOUD_TALK_APP_PASSWORD", "").strip()
    if not (url and user and pw):
        return None

    seed: dict = {
        "nextcloud_url": url.rstrip("/"),
        "username": user,
    }

    conv_csv = os.getenv("NEXTCLOUD_TALK_CONVERSATIONS", "").strip()
    if conv_csv:
        seed["conversations"] = [
            {"token": t.strip()}
            for t in conv_csv.split(",")
            if t.strip()
        ]

    poll_timeout = os.getenv("NEXTCLOUD_TALK_POLL_TIMEOUT", "").strip()
    if poll_timeout.isdigit():
        seed["poll_timeout"] = min(int(poll_timeout), 60)

    show_status = os.getenv("NEXTCLOUD_TALK_SHOW_STATUS_UPDATES", "").strip().lower()
    if show_status:
        seed["show_status_updates"] = show_status in ("1", "true", "yes")

    home = os.getenv("NEXTCLOUD_TALK_HOME_CHANNEL", "").strip()
    if home:
        seed["home_channel"] = {
            "chat_id": home,
            "name": os.getenv("NEXTCLOUD_TALK_HOME_CHANNEL_NAME", home),
        }
    return seed


# ─────────────────────────────────────────────────────────────────────
# Talk User API client
# ─────────────────────────────────────────────────────────────────────

class TalkUserClient:
    """HTTP client for Nextcloud Talk User API.

    Uses Basic Auth (username + App Password) and the standard OCS REST
    endpoints. ``poll_timeout`` controls the server-side long-poll
    duration of ``get_messages`` (default 30 s, capped at 60 by Talk).
    """

    _OCS_HEADERS = {"OCS-APIRequest": "true", "Accept": "application/json"}

    def __init__(
        self,
        *,
        base_url: str,
        username: str,
        password: str,
        poll_timeout: int = 30,
        http_client=None,
    ):
        self._base_url = base_url.rstrip("/")
        self._username = username
        self._auth = (username, password)
        self._poll_timeout = poll_timeout
        if http_client is not None:
            self._http = http_client
        else:
            self._http = httpx.AsyncClient(
                auth=self._auth,
                headers=self._OCS_HEADERS,
                timeout=poll_timeout + 15,
                # Disable connection keep-alive to avoid CLOSE_WAIT pile-up
                # on long-running polling loops (CLOSE_WAIT accumulates when
                # the server idles out before the client closes).
                limits=httpx.Limits(max_keepalive_connections=0),
            )

    async def close(self) -> None:
        try:
            await self._http.aclose()
        except Exception:
            pass

    async def send_message(
        self,
        token: str,
        message: str,
        reply_to: Optional[int] = None,
    ) -> "tuple[bool, Optional[int], Optional[str]]":
        url = f"{self._base_url}/ocs/v2.php/apps/spreed/api/v1/chat/{token}"
        data: Dict[str, Any] = {"message": message}
        if reply_to is not None:
            data["replyTo"] = reply_to
        try:
            resp = await self._http.post(url, data=data)
        except Exception as exc:
            return False, None, f"connection error: {exc}"
        if resp.status_code in (200, 201):
            try:
                return True, resp.json()["ocs"]["data"]["id"], None
            except Exception:
                return True, None, None
        return False, None, f"HTTP {resp.status_code}: {resp.text[:200]}"

    async def edit_message(
        self,
        token: str,
        message_id: int,
        new_text: str,
    ) -> "tuple[bool, Optional[str]]":
        url = f"{self._base_url}/ocs/v2.php/apps/spreed/api/v1/chat/{token}/{message_id}"
        try:
            resp = await self._http.put(url, data={"message": new_text})
        except Exception as exc:
            return False, f"connection error: {exc}"
        if resp.status_code == 200:
            return True, None
        return False, f"HTTP {resp.status_code}: {resp.text[:200]}"

    async def get_messages(
        self,
        token: str,
        last_known_id: int,
        timeout: Optional[int] = None,
    ) -> "tuple[int, list]":
        url = f"{self._base_url}/ocs/v2.php/apps/spreed/api/v1/chat/{token}"
        params = {
            "lookIntoFuture": 1,
            "lastKnownMessageId": last_known_id,
            "timeout": timeout or self._poll_timeout,
            "setReadMarker": 0,
            "includeLastKnown": 0,
        }
        resp = await self._http.get(url, params=params)
        if resp.status_code == 304:
            return 304, []
        if resp.status_code == 200:
            data = resp.json().get("ocs", {}).get("data", [])
            return 200, data if isinstance(data, list) else []
        return resp.status_code, []

    async def join_conversation(self, token: str) -> "tuple[bool, Optional[str]]":
        url = f"{self._base_url}/ocs/v2.php/apps/spreed/api/v4/room/{token}/participants/active"
        try:
            resp = await self._http.post(url)
        except Exception as exc:
            return False, f"connection error: {exc}"
        if resp.status_code in (200, 201):
            return True, None
        return False, f"HTTP {resp.status_code}: {resp.text[:200]}"

    async def list_conversations(self) -> "tuple[bool, list, Optional[str]]":
        url = f"{self._base_url}/ocs/v2.php/apps/spreed/api/v4/room"
        try:
            resp = await self._http.get(url)
        except Exception as exc:
            return False, [], f"connection error: {exc}"
        if resp.status_code == 200:
            data = resp.json().get("ocs", {}).get("data", [])
            return True, (data if isinstance(data, list) else []), None
        return False, [], f"HTTP {resp.status_code}: {resp.text[:200]}"

    async def get_latest_message_id(self, token: str) -> int:
        """Return the newest message id in a conversation.

        Returns 0 only for a genuinely empty conversation. Transport errors
        and non-200 responses RAISE instead — silently returning 0 here made
        a poll started during a Nextcloud outage replay the entire
        conversation history once the server came back (lastKnownMessageId=0
        means "give me everything").
        """
        url = f"{self._base_url}/ocs/v2.php/apps/spreed/api/v1/chat/{token}"
        params = {
            "lookIntoFuture": 0,
            "limit": 1,
            "setReadMarker": 0,
            "includeLastKnown": 1,
        }
        resp = await self._http.get(url, params=params)
        if resp.status_code == 200:
            data = resp.json().get("ocs", {}).get("data", [])
            if data and isinstance(data, list):
                return max(m.get("id", 0) for m in data)
            return 0
        raise RuntimeError(
            f"get_latest_message_id failed for {token}: HTTP {resp.status_code}"
        )

    async def upload_file(
        self,
        remote_path: str,
        local_path: str,
    ) -> "tuple[bool, Optional[str]]":
        url = f"{self._base_url}/remote.php/dav/files/{self._username}/{remote_path.lstrip('/')}"
        try:
            with open(local_path, "rb") as f:
                data = f.read()
            resp = await self._http.put(url, content=data)
            if resp.status_code in (200, 201, 204):
                return True, None
            # Auto-create parent folder on 404/409 then retry.
            if resp.status_code in (404, 409):
                parent = remote_path.rsplit("/", 1)[0] if "/" in remote_path else ""
                if parent:
                    mkcol_url = f"{self._base_url}/remote.php/dav/files/{self._username}/{parent}/"
                    await self._http.request("MKCOL", mkcol_url)
                    resp2 = await self._http.put(url, content=data)
                    if resp2.status_code in (200, 201, 204):
                        return True, None
                    return False, f"HTTP {resp2.status_code} (after MKCOL)"
            return False, f"HTTP {resp.status_code}"
        except Exception as exc:
            return False, f"upload error: {exc}"

    async def share_file_to_chat(
        self,
        token: str,
        file_path: str,
        talk_meta: Optional[dict] = None,
    ) -> "tuple[bool, Optional[int], Optional[str]]":
        url = f"{self._base_url}/ocs/v2.php/apps/files_sharing/api/v1/shares"
        data: Dict[str, Any] = {"shareType": "10", "shareWith": token, "path": file_path}
        if talk_meta:
            data["talkMetaData"] = json.dumps(talk_meta)
        try:
            resp = await self._http.post(url, data=data)
            if resp.status_code == 200:
                return True, resp.json()["ocs"]["data"]["id"], None
            return False, None, f"HTTP {resp.status_code}: {resp.text[:200]}"
        except Exception as exc:
            return False, None, f"share error: {exc}"

    async def download_file(
        self,
        remote_path: str,
        local_path: str,
    ) -> "tuple[bool, Optional[str], Optional[str]]":
        url = f"{self._base_url}/remote.php/dav/files/{self._username}/{remote_path.lstrip('/')}"
        try:
            resp = await self._http.get(url)
            if resp.status_code == 200:
                os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
                with open(local_path, "wb") as f:
                    f.write(resp.content)
                return True, local_path, None
            return False, None, f"HTTP {resp.status_code}"
        except Exception as exc:
            return False, None, f"download error: {exc}"


# ─────────────────────────────────────────────────────────────────────
# Speech-to-text providers (pluggable)
# ─────────────────────────────────────────────────────────────────────

class PlaceholderSTT:
    """No-op STT provider."""

    async def transcribe(self, audio_path: str) -> Optional[str]:
        return None


class LlamaCppSTT:
    """STT via a llama.cpp-compatible /v1/audio/transcriptions endpoint."""

    def __init__(self, base_url: str, api_key: str = "", language: str = ""):
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._language = language

    async def transcribe(self, audio_path: str) -> Optional[str]:
        url = f"{self._base_url}/audio/transcriptions"
        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        try:
            with open(audio_path, "rb") as f:
                files = {"file": (os.path.basename(audio_path), f, "audio/ogg")}
                data: Dict[str, Any] = {"model": "whisper-1"}
                if self._language:
                    data["language"] = self._language
                async with httpx.AsyncClient(timeout=60.0) as client:
                    resp = await client.post(url, headers=headers, data=data, files=files)
            if resp.status_code == 200:
                return (resp.json().get("text") or "").strip() or None
        except Exception as exc:
            logger.warning("talk: STT failed: %s", exc)
        return None


# ─────────────────────────────────────────────────────────────────────
# Adapter
# ─────────────────────────────────────────────────────────────────────

class NextcloudTalkAdapter(BasePlatformAdapter):
    """Nextcloud Talk User-API adapter (OCS long-poll)."""

    MAX_MESSAGE_LENGTH = MAX_MESSAGE_LENGTH
    supports_code_blocks = True
    typed_command_prefix = "!"  # Talk intercepts "/" commands client-side
    # send() chunks at MAX_MESSAGE_LENGTH itself; without this flag
    # gateway/delivery.py would truncate long content (e.g. cron output)
    # before the adapter ever sees it.
    splits_long_messages = True

    # Session commands (/new, /reset) are NOT handled here — they forward to
    # the gateway as command events, which resets the correct per-user session
    # key and runs the full runner-reset cleanup. Only genuinely
    # platform-local commands belong in this set.
    _LOCAL_COMMANDS = {"/help"}

    def __init__(self, config: PlatformConfig):
        from gateway.config import Platform
        # ``Platform("nextcloud_talk")`` resolves via ``Platform._missing_()``
        # — Hermes' plugin pattern; no edit to the Platform enum needed.
        super().__init__(config=config, platform=Platform("nextcloud_talk"))

        extra = getattr(config, "extra", {}) or {}

        # ── Required: NC URL ────────────────────────────────────────
        self._nextcloud_url = (
            str(extra.get("nextcloud_url") or os.getenv("NEXTCLOUD_TALK_URL", "")).rstrip("/")
        )
        if not self._nextcloud_url:
            raise ValueError("nextcloud_talk: NEXTCLOUD_TALK_URL (or config.extra.nextcloud_url) is required")

        # ── Required: username ──────────────────────────────────────
        self._username = (
            str(extra.get("username") or os.getenv("NEXTCLOUD_TALK_USERNAME", ""))
        ).strip()
        if not self._username:
            raise ValueError("nextcloud_talk: NEXTCLOUD_TALK_USERNAME (or config.extra.username) is required")

        # ── Required: app password ──────────────────────────────────
        # The app password is ALWAYS read from env (NEXTCLOUD_TALK_APP_PASSWORD)
        # to keep secrets out of config.yaml. Operators may override the env-var
        # name via config.extra.app_password_env for backwards compat.
        pw_env = str(extra.get("app_password_env", "NEXTCLOUD_TALK_APP_PASSWORD"))
        self._password = (os.environ.get(pw_env) or "").strip()
        if not self._password:
            raise ValueError(
                f"nextcloud_talk: env var {pw_env} is missing or empty. "
                "Generate an App Password in Nextcloud (Settings → Security) and set it."
            )

        # ── Required: conversations ─────────────────────────────────
        raw_conversations = extra.get("conversations") or []
        if not raw_conversations:
            csv = os.getenv("NEXTCLOUD_TALK_CONVERSATIONS", "").strip()
            if csv:
                raw_conversations = [{"token": t.strip()} for t in csv.split(",") if t.strip()]
        if not raw_conversations:
            raise ValueError(
                "nextcloud_talk: 'conversations' is required (config.extra.conversations "
                "or NEXTCLOUD_TALK_CONVERSATIONS env)."
            )
        self._conversations: List[Dict[str, Any]] = [dict(c) for c in raw_conversations]

        # ── Options ─────────────────────────────────────────────────
        self._edit_ack_into_response = bool(extra.get("edit_ack_into_response", True))
        self._show_status_updates = bool(extra.get("show_status_updates", True))
        self._poll_timeout = int(extra.get("poll_timeout", 30))
        if self._poll_timeout > 60:
            self._poll_timeout = 60

        # ── Channel alias map ───────────────────────────────────────
        self._channel_aliases: Dict[str, str] = {}
        for conv in self._conversations:
            alias = str(conv.get("alias", "")).strip()
            token = str(conv.get("token", "")).strip()
            if alias and token:
                self._channel_aliases[alias] = token
        raw_aliases = extra.get("channel_aliases") or {}
        if isinstance(raw_aliases, dict):
            for k, v in raw_aliases.items():
                k, v = str(k).strip(), str(v).strip()
                if k and v:
                    self._channel_aliases[k] = v

        # ── STT provider ────────────────────────────────────────────
        stt_config = extra.get("stt") or {}
        if stt_config.get("provider") == "llamacpp" and stt_config.get("base_url"):
            self._stt = LlamaCppSTT(
                base_url=stt_config["base_url"],
                api_key=stt_config.get("api_key", ""),
                language=stt_config.get("language", ""),
            )
        else:
            self._stt = PlaceholderSTT()

        # ── Runtime state ───────────────────────────────────────────
        self._chat_name_cache: Dict[str, str] = {}
        self._client: Optional[TalkUserClient] = None
        self._poll_tasks: List[asyncio.Task] = []
        self._shutdown: bool = False
        # "Thinking..." acks pending an edit-into-response, scoped PER
        # conversation (a single global slot raced across conversations).
        # Each chat keeps a FIFO of ack message ids: the gateway serializes
        # turns per session, so replies arrive in turn order and each reply
        # edits the oldest outstanding ack of its own conversation.
        self._pending_acks: Dict[str, Deque[int]] = {}

    # ── Classification ──────────────────────────────────────────────

    def _classify_chat(self, chat_id: str) -> str:
        tokens = {c["token"] for c in self._conversations}
        return "group" if chat_id in tokens else "dm"

    def _is_chat_allowed(self, chat_id: str) -> bool:
        tokens = {c["token"] for c in self._conversations}
        return chat_id in tokens

    def _record_chat_name(self, chat_id: str, name: str) -> None:
        if name:
            self._chat_name_cache[chat_id] = name

    def _resolve_chat_identifier(self, name: Optional[str]) -> Optional[str]:
        """Resolve a user-supplied conversation name to its token.

        Order:
          1. Config alias (case-insensitive)
          2. Learned display name (case-insensitive)
          3. Raw token fallback (alphanumeric + ``-``/``_``)
        """
        if not name:
            return None
        s = name.strip()
        if not s:
            return None
        lo = s.lower()
        for alias, token in self._channel_aliases.items():
            if alias.lower() == lo:
                return token
        for chat_id, display_name in self._chat_name_cache.items():
            if display_name and display_name.strip().lower() == lo:
                return chat_id
        if all(ch.isalnum() or ch in ("-", "_") for ch in s):
            return s
        return None

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {
            "chat_id": chat_id,
            "name": self._chat_name_cache.get(chat_id, chat_id),
            "type": self._classify_chat(chat_id),
        }

    # ── Attachment handling ────────────────────────────────────────

    async def _download_attachment(self, attachment: dict) -> Optional[str]:
        name = attachment.get("name", "attachment")
        unique = uuid.uuid4().hex[:8]
        local_path = os.path.join(MEDIA_TEMP_DIR, f"{unique}-{name}")
        os.makedirs(MEDIA_TEMP_DIR, exist_ok=True)
        remote_path = attachment.get("path", name)
        ok, path, err = await self._client.download_file(remote_path, local_path)
        if ok:
            return path
        # Fallback: share-link download
        link = attachment.get("link", "")
        if link:
            share_token = link.rstrip("/").split("/")[-1]
            share_url = f"index.php/s/{share_token}/download"
            ok2, path2, _ = await self._client.download_file(share_url, local_path)
            if ok2:
                return path2
        logger.warning("nextcloud_talk: could not download %s (%s)", name, err)
        return None

    # ── Local commands ─────────────────────────────────────────────

    async def _handle_command(self, text: str, chat_id: str) -> Optional[str]:
        """Adapter-local commands. Returns reply or None to forward to runtime.

        Session commands (``/new``, ``/reset``, ``/stop``, …) intentionally
        return ``None`` here: they flow to the gateway as command events, which
        resolves the per-user session key from the full event source
        (``user_id`` included) and performs the runner-reset cleanup a local
        ``reset_session()`` call would skip.
        """
        cmd = text.strip().split(maxsplit=1)[0].lower()
        if cmd not in self._LOCAL_COMMANDS:
            return None

        if cmd == "/help":
            return (
                "**Hermes on Nextcloud Talk**\n\n"
                "Talk intercepts `/`-prefixed commands client-side. "
                "Use `!` instead — e.g. `!new`, `!reset`, `!help`, `!approve`.\n\n"
                "Built-in command summary:\n"
                "- `!new` / `!reset` — reset the session\n"
                "- `!help` — this help\n\n"
                "Anything else is forwarded to Hermes."
            )
        return None

    # ── Connection lifecycle ───────────────────────────────────────

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        if self._poll_tasks:
            return True
        if self._client is None:
            self._client = TalkUserClient(
                base_url=self._nextcloud_url,
                username=self._username,
                password=self._password,
                poll_timeout=self._poll_timeout,
            )
        self._shutdown = False

        ok, rooms, err = await self._client.list_conversations()
        if ok:
            room_tokens = {r.get("token") for r in rooms}
            for conv in self._conversations:
                token = conv["token"]
                if token not in room_tokens:
                    logger.warning("nextcloud_talk: not a member of %s", token)
                await self._client.join_conversation(token)
        else:
            logger.warning("nextcloud_talk: could not list conversations: %s", err)

        loop = asyncio.get_event_loop()
        for conv in self._conversations:
            task = loop.create_task(self._poll_loop(conv["token"]))
            self._poll_tasks.append(task)

        logger.info(
            "nextcloud_talk: polling %d conversation(s) as %s",
            len(self._conversations), self._username,
        )
        return True

    async def disconnect(self) -> None:
        self._shutdown = True
        for task in self._poll_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
        self._poll_tasks.clear()
        if self._client:
            await self._client.close()
            self._client = None

    # ── Poll loop ──────────────────────────────────────────────────

    async def _poll_loop(self, token: str) -> None:
        # Cursor init must succeed before polling starts. Falling back to 0
        # would replay the whole conversation history after an outage — the
        # agent would then "answer" every old message (incident 2026-07-12).
        last_known_id: Optional[int] = None
        while not self._shutdown:
            try:
                last_known_id = await self._client.get_latest_message_id(token)
                break
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "nextcloud_talk: cursor init failed for %s (%s), retrying in 15s",
                    token, exc,
                )
                await asyncio.sleep(15)
        if last_known_id is None:
            return
        logger.info("nextcloud_talk: poll started for %s (from ID %d)", token, last_known_id)
        while not self._shutdown:
            try:
                status, messages = await self._client.get_messages(
                    token, last_known_id, timeout=self._poll_timeout,
                )
                if status == 304:
                    continue
                if status == 401:
                    logger.error("nextcloud_talk: auth failed, stopping poll for %s", token)
                    break
                for msg in messages:
                    msg_id = msg.get("id", 0)
                    if msg_id > last_known_id:
                        last_known_id = msg_id
                    await self._on_poll_message(msg, token)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("nextcloud_talk: poll error, retrying in 5s")
                await asyncio.sleep(5)

    async def _on_poll_message(self, msg: Dict[str, Any], token: str) -> None:
        parsed = _parse_user_message(msg, own_user_id=self._username)
        if parsed is None:
            return

        chat_id = parsed["chat_id"] or token
        text = parsed["text"]

        if not self._is_chat_allowed(chat_id):
            return

        self._record_chat_name(chat_id, parsed.get("user_name", ""))

        # Normalize "!" prefix to "/" — Talk intercepts "/" commands client-side,
        # so users type "!approve" / "!new" / "!help" and the adapter rewrites.
        if text.startswith("!"):
            text = "/" + text[1:]

        # Adapter-local commands shortcut the runtime.
        if text.startswith("/"):
            reply = await self._handle_command(text, chat_id)
            if reply is not None:
                await self.send(chat_id, reply)
                return

        # Send "Thinking..." ack
        ack_msg_id = None
        if self._show_status_updates:
            try:
                ok, mid, _ = await self._client.send_message(chat_id, "⏳ Thinking...")
                if ok:
                    ack_msg_id = mid
            except Exception:
                pass
        if ack_msg_id is not None:
            self._pending_acks.setdefault(chat_id, deque()).append(ack_msg_id)

        # Attachment handling
        media_urls: List[str] = []
        event_type = MessageType.TEXT
        attachment = parsed.get("attachment")
        if attachment:
            local_path = await self._download_attachment(attachment)
            if local_path:
                media_urls.append(local_path)
                category = _classify_attachment(attachment.get("mimetype", ""))
                if category == "image":
                    event_type = MessageType.PHOTO
                elif category == "audio":
                    transcribed = await self._stt.transcribe(local_path)
                    if transcribed:
                        text = transcribed
                        event_type = MessageType.TEXT
                    else:
                        if not text:
                            text = (
                                f"Voice memo received ({attachment['name']}, "
                                f"{attachment['size']} bytes) but speech-to-text "
                                "is not available."
                            )
                        event_type = MessageType.VOICE
                elif category == "document":
                    event_type = MessageType.DOCUMENT
                else:
                    if not text:
                        text = f"User shared {attachment['name']} ({attachment['mimetype']})"
                    event_type = MessageType.DOCUMENT
            else:
                if not text:
                    text = f"[Attachment {attachment['name']} could not be downloaded]"

        source = self.build_source(
            chat_id=chat_id,
            chat_type=self._classify_chat(chat_id),
            user_id=parsed["user_id"],
            user_name=parsed.get("user_name", ""),
        )
        event = MessageEvent(
            text=text or "",
            message_type=event_type,
            source=source,
            raw_message=msg,
            message_id=str(parsed["message_id"]) if parsed["message_id"] else None,
            media_urls=media_urls,
        )

        try:
            await self.handle_message(event)
        except Exception:
            logger.exception("nextcloud_talk: handle_message raised")

    # ── Send paths ────────────────────────────────────────────────

    async def _upload_and_share(
        self,
        chat_id: str,
        local_path: str,
        caption: Optional[str] = None,
        talk_meta: Optional[dict] = None,
    ) -> SendResult:
        """Upload file to WebDAV and share into a conversation."""
        # Strip markdown/quote artifacts.
        local_path = local_path.strip()
        while local_path and local_path[0] in "*_`'\"":
            local_path = local_path[1:]
        while local_path and local_path[-1] in "*_`'\"":
            local_path = local_path[:-1]

        if self._client is None:
            return SendResult(success=False, error="Not connected")

        filename = os.path.basename(local_path)
        remote_path = f"Talk-Uploads/{uuid.uuid4().hex[:8]}-{filename}"
        ok, err = await self._client.upload_file(remote_path, local_path)
        if not ok:
            return SendResult(success=False, error=f"Upload: {err}")
        ok, share_id, err = await self._client.share_file_to_chat(
            chat_id, f"/{remote_path}", talk_meta=talk_meta,
        )
        if not ok:
            return SendResult(success=False, error=f"Share: {err}")
        if caption:
            await self._client.send_message(chat_id, caption)
        return SendResult(success=True, message_id=str(share_id) if share_id else None)

    async def send_voice(self, chat_id, audio_path, caption=None, reply_to=None, **kwargs):
        # Talk's voice-message type ONLY accepts audio/mpeg or audio/wav.
        # OGG/Opus is explicitly rejected. The TTS tool auto-converts MP3→OGG
        # for Telegram compat, but the original MP3 is still on disk — prefer it.
        if audio_path.lower().endswith(".ogg"):
            mp3_path = audio_path.rsplit(".", 1)[0] + ".mp3"
            if os.path.exists(mp3_path):
                audio_path = mp3_path
        return await self._upload_and_share(
            chat_id, audio_path, caption,
            talk_meta={"messageType": "voice-message"},
        )

    async def send_image(self, chat_id, image_url, caption=None, reply_to=None, metadata=None):
        return await self._upload_and_share(chat_id, image_url, caption)

    async def send_document(self, chat_id, file_path, caption=None, file_name=None, reply_to=None, **kwargs):
        return await self._upload_and_share(chat_id, file_path, caption)

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        if self._client is None:
            return SendResult(
                success=False,
                error="nextcloud_talk: not connected",
                retryable=False,
            )

        # Ack-edit: if a "Thinking..." ack is pending for THIS conversation,
        # edit the oldest one into the response instead of sending a new
        # message. Scoped per chat so concurrent conversations (and queued
        # same-chat turns) can never consume each other's ack.
        ack_queue = self._pending_acks.get(chat_id)
        if ack_queue and self._edit_ack_into_response:
            ack_msg_id = ack_queue.popleft()
            if not ack_queue:
                self._pending_acks.pop(chat_id, None)
            ok, err = await self._client.edit_message(chat_id, ack_msg_id, content)
            if ok:
                return SendResult(success=True, message_id=str(ack_msg_id))
            logger.warning("nextcloud_talk: ack edit failed (%s), sending new message", err)

        chunks = self.truncate_message(content, self.MAX_MESSAGE_LENGTH)
        last_id: Optional[int] = None
        for i, chunk in enumerate(chunks):
            ok, msg_id, err = await self._client.send_message(
                chat_id, chunk,
                reply_to=int(reply_to) if i == 0 and reply_to and str(reply_to).isdigit() else None,
            )
            if not ok:
                retryable = bool(err and ("HTTP 5" in err or "connection" in err.lower()))
                return SendResult(success=False, error=err, retryable=retryable)
            last_id = msg_id
        return SendResult(success=True, message_id=str(last_id) if last_id is not None else None)

    async def edit_message(
        self,
        chat_id: str,
        message_id: str,
        content: str,
    ) -> SendResult:
        if self._client is None:
            return SendResult(success=False, error="nextcloud_talk: not connected", retryable=False)
        try:
            mid_int = int(message_id)
        except (ValueError, TypeError):
            return SendResult(success=False, error=f"invalid message_id: {message_id}", retryable=False)
        ok, err = await self._client.edit_message(chat_id, mid_int, content)
        if ok:
            return SendResult(success=True, message_id=message_id)
        retryable = bool(err and ("HTTP 5" in err or "connection" in err.lower()))
        return SendResult(success=False, error=err, retryable=retryable)

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        # No-op — the "⏳ Thinking..." ack in _on_poll_message is the equivalent feedback.
        return

    async def stop_typing(self, chat_id: str) -> None:
        return


# ─────────────────────────────────────────────────────────────────────
# Standalone send (out-of-process cron delivery)
# ─────────────────────────────────────────────────────────────────────

async def _standalone_send(
    pconfig,
    chat_id: str,
    message: str,
    *,
    thread_id: Optional[str] = None,
    media_files: Optional[List[str]] = None,
    force_document: bool = False,
) -> Dict[str, Any]:
    """Out-of-process send for cron / send_message_tool fallback paths.

    Used when the gateway runner is not in this process (e.g. ``hermes cron``
    running standalone). Without this hook, ``deliver=nextcloud_talk`` cron
    jobs fail with ``No live adapter for platform``.

    ``thread_id`` is accepted for signature parity but unused — Talk has no
    thread primitive at this layer. Media uploads via the standalone path are
    not supported (would need a full client instance); callers should prefer
    the live adapter path when sending attachments.
    """
    if not HTTPX_AVAILABLE:
        return {"error": "nextcloud_talk standalone send: httpx not installed"}

    extra = getattr(pconfig, "extra", {}) or {}
    base_url = (
        extra.get("nextcloud_url") or os.getenv("NEXTCLOUD_TALK_URL", "")
    ).rstrip("/")
    username = extra.get("username") or os.getenv("NEXTCLOUD_TALK_USERNAME", "")
    pw_env = extra.get("app_password_env", "NEXTCLOUD_TALK_APP_PASSWORD")
    password = os.environ.get(pw_env, "")
    if not (base_url and username and password):
        return {"error": "nextcloud_talk standalone send: missing URL/username/app password"}
    if not chat_id:
        return {"error": "nextcloud_talk standalone send: chat_id (conversation token) required"}

    url = f"{base_url}/ocs/v2.php/apps/spreed/api/v1/chat/{chat_id}"
    headers = {"OCS-APIRequest": "true", "Accept": "application/json"}
    try:
        async with httpx.AsyncClient(timeout=20.0, auth=(username, password), headers=headers) as client:
            # Split long messages — Talk hard-cap is 32K, we chunk at MAX_MESSAGE_LENGTH.
            chunks: List[str] = []
            remaining = message
            while len(remaining) > MAX_MESSAGE_LENGTH:
                chunks.append(remaining[:MAX_MESSAGE_LENGTH])
                remaining = remaining[MAX_MESSAGE_LENGTH:]
            chunks.append(remaining)

            last_id = None
            for chunk in chunks:
                resp = await client.post(url, data={"message": chunk})
                if resp.status_code not in (200, 201):
                    return {"error": f"nextcloud_talk HTTP {resp.status_code}: {resp.text[:200]}"}
                try:
                    last_id = resp.json()["ocs"]["data"]["id"]
                except Exception:
                    last_id = None
            return {
                "success": True,
                "platform": "nextcloud_talk",
                "chat_id": chat_id,
                "message_id": str(last_id) if last_id is not None else uuid.uuid4().hex[:12],
            }
    except Exception as e:
        return {"error": f"nextcloud_talk standalone send failed: {e}"}


# ─────────────────────────────────────────────────────────────────────
# Plugin registration
# ─────────────────────────────────────────────────────────────────────

def register(ctx) -> None:
    """Plugin entry point — called by the Hermes plugin system at startup."""
    # Display metadata for the dashboard messaging card. Feature-detected so
    # the plugin also loads on cores whose PlatformEntry predates the
    # description/docs_url fields (unknown kwargs raise TypeError there).
    from dataclasses import fields as _dataclass_fields

    from gateway.platform_registry import PlatformEntry as _PlatformEntry

    _entry_fields = {f.name for f in _dataclass_fields(_PlatformEntry)}
    _display_kwargs = {}
    if "description" in _entry_fields:
        _display_kwargs["description"] = (
            "Connect Hermes to Nextcloud Talk conversations as a regular "
            "Nextcloud user (app password) — DMs, groups, attachments, and "
            "voice messages."
        )
    if "docs_url" in _entry_fields:
        _display_kwargs["docs_url"] = (
            "https://docs.nextcloud.com/server/latest/user_manual/en/"
            "session_management.html#managing-devices"
        )

    ctx.register_platform(
        name="nextcloud_talk",
        label="Nextcloud Talk",
        adapter_factory=lambda cfg: NextcloudTalkAdapter(cfg),
        check_fn=check_requirements,
        validate_config=validate_config,
        is_connected=is_connected,
        required_env=[
            "NEXTCLOUD_TALK_URL",
            "NEXTCLOUD_TALK_USERNAME",
            "NEXTCLOUD_TALK_APP_PASSWORD",
        ],
        install_hint="pip install httpx   # already a Hermes dependency",
        env_enablement_fn=_env_enablement,
        cron_deliver_env_var="NEXTCLOUD_TALK_HOME_CHANNEL",
        standalone_sender_fn=_standalone_send,
        allowed_users_env="NEXTCLOUD_TALK_ALLOWED_USERS",
        allow_all_env="NEXTCLOUD_TALK_ALLOW_ALL_USERS",
        max_message_length=MAX_MESSAGE_LENGTH,
        emoji="☁️",
        # actorId is the Nextcloud username — never a phone number / email.
        pii_safe=True,
        allow_update_command=True,
        platform_hint=(
            "You are talking to a user via Nextcloud Talk. The user types "
            "messages from the Talk web/mobile/desktop client. Markdown is "
            "rendered; you can use **bold**, *italic*, `code`, and fenced "
            "code blocks. Talk intercepts '/'-prefixed commands client-side, "
            "so user-facing instructions tell users to type '!approve' / "
            "'!new' / '!reset' instead of the slash form."
        ),
        **_display_kwargs,
    )

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
              mode: mention               # optional: mention | watch | free (default free)
            - token: "ghi456jkl"
              alias: "alerts"
          channel_aliases:                # optional — extra alias → token map
            "ops": "abc123def"
          default_room_mode: free         # mode for conversations without an explicit one
          display_name: "Hermes"          # optional; plain-text @"Hermes" mention detection
          watch_flush_interval: 900       # seconds between watch-mode digests (default 900)
          federation:                     # optional — Talk federation invitations
            allowed_inviters:             # full cloud ids; invites from anyone else stay pending
              - "alice@cloud.example.com"
            poll_interval: 60             # seconds between invitation checks (default 60)
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
    NEXTCLOUD_TALK_DISPLAY_NAME     Bot display name for plain-text mention detection
    NEXTCLOUD_TALK_ROOM_MODES       "token1=mention,token2=watch" per-conversation modes
    NEXTCLOUD_TALK_DEFAULT_ROOM_MODE  mention | watch | free (default free)
    NEXTCLOUD_TALK_WATCH_FLUSH_INTERVAL  Seconds between watch-mode digests (default 900)
    NEXTCLOUD_TALK_ALLOWED_INVITERS  Comma-separated full cloud ids allowed to invite
                                    Hermes into federated conversations
    NEXTCLOUD_TALK_FEDERATION_POLL_INTERVAL  Seconds between invitation checks (default 60)

Identity model: Nextcloud Talk has a native user identity (``actorId``).
Allowlists are matched against ``actorId`` (the Nextcloud username, e.g.
``alice`` — not a display name and not an email). System messages and
the bot's own messages are filtered out before allowlist evaluation.

Room modes (group conversations only — DMs are always ``free``):

* ``free`` (default) — every authorized message starts an agent turn.
* ``mention`` — only messages that @mention the bot (rich ``messageParameters``
  of type ``user`` with the bot's id, ``@all``/``@here``-style ``call``
  mentions, or a plain-text ``@username`` / ``@"Display Name"``) start a turn.
* ``watch`` — direct mentions are answered immediately; everything else is
  buffered and handed to the agent every ``watch_flush_interval`` seconds
  as ONE internal digest that asks "is there anything here worth
  answering?". The agent ends the digest turn with ``NO_REPLY`` when
  nothing warrants a response, and the gateway then sends nothing.

``!``-prefixed commands (``!new``, ``!help`` …) are always addressed to the
bot and bypass mode gating. Mode gating runs AFTER the pre-effect
authorization gate: a watch buffer only ever holds messages from
authorized actors, which matters because the digest is an ``internal``
event that bypasses the gateway's per-sender allowlist.

Federation: with ``federation.allowed_inviters`` configured, the adapter
polls Talk's federation-invitation endpoint and auto-accepts PENDING
invitations whose ``inviterCloudId`` exactly matches an allowlisted full
cloud id (``user@host``, case-insensitive). Every other invitation is
left pending for a human — never declined. An accepted conversation is
adopted into the live conversation list for this process; add its token
to ``conversations`` to keep polling it across restarts.
"""

import asyncio
import json
import logging
import os
import re
import tempfile
import uuid
from collections import deque
from typing import Any, Optional

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

# Per-conversation room modes (see module docstring).
ROOM_MODE_FREE = "free"
ROOM_MODE_MENTION = "mention"
ROOM_MODE_WATCH = "watch"
VALID_ROOM_MODES = frozenset({ROOM_MODE_FREE, ROOM_MODE_MENTION, ROOM_MODE_WATCH})
DEFAULT_ROOM_MODE = ROOM_MODE_FREE
DEFAULT_WATCH_FLUSH_INTERVAL = 900  # seconds between watch-mode digests
# A watch buffer is bounded so a runaway room can never grow process memory
# without limit between flushes; the oldest entries are dropped first.
WATCH_BUFFER_MAX_MESSAGES = 200

# Talk federation invitations. Only ever act on a PENDING one: the list
# endpoint has been observed to keep returning already-accepted invites, and
# re-POSTing accept on those always 400s.
DEFAULT_FEDERATION_POLL_INTERVAL = 60
INVITE_STATE_PENDING = 0


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _read_file_bytes(path: str) -> bytes:
    """Blocking file read — always called via ``asyncio.to_thread``."""
    with open(path, "rb") as f:
        return f.read()


def _write_file_bytes(path: str, data: bytes) -> None:
    """Blocking file write (creates parents) — always via ``asyncio.to_thread``."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


def _normalize_room_mode(value: Any, default: str = DEFAULT_ROOM_MODE) -> str:
    """Coerce a configured mode to one of :data:`VALID_ROOM_MODES`.

    Unknown values fall back to *default* (logged by the caller) — a typo in
    one room's mode must never break config loading for the others.
    """
    mode = str(value or "").strip().lower()
    return mode if mode in VALID_ROOM_MODES else default


def _parse_room_modes_csv(raw: Optional[str]) -> dict[str, str]:
    """Parse ``NEXTCLOUD_TALK_ROOM_MODES`` (``"tok1=mention,tok2=watch"``).

    Malformed pairs and unknown modes are skipped, never raised.
    """
    modes: dict[str, str] = {}
    if not raw:
        return modes
    for pair in raw.split(","):
        token, sep, mode = pair.partition("=")
        token, mode = token.strip(), mode.strip().lower()
        if sep and token and mode in VALID_ROOM_MODES:
            modes[token] = mode
    return modes


def _is_mention(
    text: str,
    message_parameters: Optional[dict[str, Any]],
    own_user_id: str,
    own_display_name: str,
    room_token: str = "",
) -> bool:
    """True if a Talk message addresses the bot.

    Independent signals, any one is sufficient:

    1. Rich ``messageParameters`` of type ``user`` whose ``id`` is the bot's
       Nextcloud user id (Talk always mentions THIS server's local account
       by its local id, also inside federated rooms).
    2. Rich ``messageParameters`` of type ``call`` — Talk's ``@all`` /
       ``@here`` style room-wide mention; its ``id`` is the room token.
    3. Plain-text fallback for clients that send no rich parameters:
       ``@username`` or Talk's quoted ``@"Display Name"`` syntax,
       case-insensitive.
    """
    params = message_parameters if isinstance(message_parameters, dict) else {}
    for param in params.values():
        if not isinstance(param, dict):
            continue
        ptype = param.get("type")
        if ptype == "user" and param.get("id") == own_user_id:
            return True
        if ptype == "call" and (not room_token or param.get("id") in (room_token, "all")):
            return True

    text = text or ""
    own_user_id = (own_user_id or "").strip()
    own_display_name = (own_display_name or "").strip()
    if own_user_id and re.search(rf"(?<!\w)@{re.escape(own_user_id)}(?!\w)", text, re.IGNORECASE):
        return True
    if own_display_name:
        if re.search(rf'@"{re.escape(own_display_name)}"', text, re.IGNORECASE):
            return True
        if re.search(rf"(?<!\w)@{re.escape(own_display_name)}(?!\w)", text, re.IGNORECASE):
            return True
    return False


def _normalize_cloud_id(value: Any) -> str:
    """Canonical form of a federation cloud id (``user@host``).

    Strips the ``federated_users/`` actor prefix some Talk surfaces add, a
    scheme, and a trailing slash; lower-cases the whole id (Nextcloud user
    ids are matched case-insensitively by Talk itself).
    """
    cloud_id = str(value or "").strip().removeprefix("federated_users/")
    user, sep, host = cloud_id.partition("@")
    if sep:
        host = re.sub(r"^https?://", "", host.strip()).rstrip("/")
        cloud_id = f"{user.strip()}@{host}"
    return cloud_id.lower()


def _is_allowed_inviter(inviter_cloud_id: Any, allowed_inviters: list[str]) -> bool:
    """True only if the inviter's FULL cloud id is on the explicit allowlist.

    Deliberately stricter than matching the local part alone: ``alice`` on
    the allowlist must not admit ``alice@attacker.example``. Entries are
    full ``user@host`` cloud ids; an entry without a host never matches. An
    empty allowlist admits nobody (fail closed).
    """
    inviter = _normalize_cloud_id(inviter_cloud_id)
    if not inviter or "@" not in inviter:
        return False
    for entry in allowed_inviters or []:
        candidate = _normalize_cloud_id(entry)
        if candidate and "@" in candidate and candidate == inviter:
            return True
    return False


def _is_pending_invite(invite: dict[str, Any]) -> bool:
    """True if a federation invitation is still pending.

    A missing ``state`` (older Talk versions never returned the field) is
    treated as pending so nothing regresses there.
    """
    state = invite.get("state")
    return state is None or state == INVITE_STATE_PENDING


def _build_watch_digest(room_label: str, entries: list[dict[str, Any]]) -> str:
    """Render buffered watch-mode messages as ONE review prompt for the agent."""
    lines = [
        f"[Nextcloud Talk watch-mode digest: {len(entries)} message(s) in "
        f"\"{room_label}\" since the last review]",
        "",
    ]
    for entry in entries:
        who = entry.get("user_name") or entry.get("user_id") or "?"
        uid = entry.get("user_id") or "?"
        text = (entry.get("text") or "").strip()
        attachment = entry.get("attachment")
        if attachment:
            text = (text + " " if text else "") + f"[attachment: {attachment}]"
        lines.append(f"- {who} ({uid}) #{entry.get('message_id', '?')}: {text}")
    lines += [
        "",
        "Review this conversation. Reply ONLY if something here genuinely "
        "warrants a response from you — a question addressed to you, a task "
        "you can help with, or something you must act on. Otherwise reply "
        "with exactly NO_REPLY.",
    ]
    return "\n".join(lines)


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

    display_name = os.getenv("NEXTCLOUD_TALK_DISPLAY_NAME", "").strip()
    if display_name:
        seed["display_name"] = display_name

    room_modes = _parse_room_modes_csv(os.getenv("NEXTCLOUD_TALK_ROOM_MODES", ""))
    if room_modes:
        seed["room_modes"] = room_modes
    default_mode = os.getenv("NEXTCLOUD_TALK_DEFAULT_ROOM_MODE", "").strip().lower()
    if default_mode in VALID_ROOM_MODES:
        seed["default_room_mode"] = default_mode
    watch_interval = os.getenv("NEXTCLOUD_TALK_WATCH_FLUSH_INTERVAL", "").strip()
    if watch_interval.isdigit():
        seed["watch_flush_interval"] = int(watch_interval)

    inviters_csv = os.getenv("NEXTCLOUD_TALK_ALLOWED_INVITERS", "").strip()
    federation_interval = os.getenv("NEXTCLOUD_TALK_FEDERATION_POLL_INTERVAL", "").strip()
    if inviters_csv or federation_interval.isdigit():
        federation: dict = {}
        if inviters_csv:
            federation["allowed_inviters"] = [
                s.strip() for s in inviters_csv.split(",") if s.strip()
            ]
        if federation_interval.isdigit():
            federation["poll_interval"] = int(federation_interval)
        seed["federation"] = federation

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
        data: dict[str, Any] = {"message": message}
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

    async def list_federation_invitations(self) -> "tuple[bool, list, Optional[str], int]":
        """List this user's federation invitations.

        Returns ``(ok, invitations, error, status_code)``. A 404 means the
        federation endpoint is not available on this server (Talk too old
        or federation disabled) — callers treat that as "nothing to do",
        not as an error.
        """
        url = f"{self._base_url}/ocs/v2.php/apps/spreed/api/v1/federation/invitation"
        try:
            resp = await self._http.get(url)
        except Exception as exc:
            return False, [], f"connection error: {exc}", 0
        if resp.status_code == 200:
            data = resp.json().get("ocs", {}).get("data", [])
            return True, (data if isinstance(data, list) else []), None, 200
        return False, [], f"HTTP {resp.status_code}: {resp.text[:200]}", resp.status_code

    async def accept_federation_invitation(
        self, invite_id: int,
    ) -> "tuple[bool, Optional[dict], Optional[str]]":
        """Accept a federation invitation. Returns ``(ok, room, error)``;
        ``room`` is Talk's conversation payload (its ``token`` is the LOCAL
        proxy token this user polls the federated conversation through)."""
        url = f"{self._base_url}/ocs/v2.php/apps/spreed/api/v1/federation/invitation/{invite_id}"
        try:
            resp = await self._http.post(url)
        except Exception as exc:
            return False, None, f"connection error: {exc}"
        if resp.status_code in (200, 201):
            try:
                data = resp.json().get("ocs", {}).get("data", {})
            except Exception:
                data = {}
            return True, (data if isinstance(data, dict) else {}), None
        return False, None, f"HTTP {resp.status_code}: {resp.text[:200]}"

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
            data = await asyncio.to_thread(_read_file_bytes, local_path)
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
        data: dict[str, Any] = {"shareType": "10", "shareWith": token, "path": file_path}
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
                await asyncio.to_thread(_write_file_bytes, local_path, resp.content)
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
            audio = await asyncio.to_thread(_read_file_bytes, audio_path)
            files = {"file": (os.path.basename(audio_path), audio, "audio/ogg")}
            data: dict[str, Any] = {"model": "whisper-1"}
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
        self._conversations: list[dict[str, Any]] = [dict(c) for c in raw_conversations]

        # ── Options ─────────────────────────────────────────────────
        self._edit_ack_into_response = bool(extra.get("edit_ack_into_response", True))
        self._show_status_updates = bool(extra.get("show_status_updates", True))
        self._poll_timeout = int(extra.get("poll_timeout", 30))
        self._poll_timeout = min(self._poll_timeout, 60)

        # ── Channel alias map ───────────────────────────────────────
        self._channel_aliases: dict[str, str] = {}
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

        # ── Room modes (mention / watch / free) ─────────────────────
        self._display_name: str = str(
            extra.get("display_name") or os.getenv("NEXTCLOUD_TALK_DISPLAY_NAME", "")
        ).strip()
        raw_default_mode = extra.get("default_room_mode") or os.getenv(
            "NEXTCLOUD_TALK_DEFAULT_ROOM_MODE", DEFAULT_ROOM_MODE
        )
        self._default_room_mode = _normalize_room_mode(raw_default_mode)
        if self._default_room_mode != str(raw_default_mode).strip().lower():
            logger.warning(
                "nextcloud_talk: unknown default_room_mode %r, using %r",
                raw_default_mode, self._default_room_mode,
            )
        self._room_modes: dict[str, str] = {}
        raw_modes = extra.get("room_modes")
        if isinstance(raw_modes, str):
            raw_modes = _parse_room_modes_csv(raw_modes)
        elif not isinstance(raw_modes, dict):
            raw_modes = _parse_room_modes_csv(os.getenv("NEXTCLOUD_TALK_ROOM_MODES", ""))
        for token, mode in raw_modes.items():
            self._set_room_mode(str(token).strip(), mode)
        for conv in self._conversations:
            if conv.get("mode") is not None:
                self._set_room_mode(str(conv.get("token", "")).strip(), conv["mode"])
        try:
            self._watch_flush_interval = max(1, int(
                extra.get("watch_flush_interval")
                or os.getenv("NEXTCLOUD_TALK_WATCH_FLUSH_INTERVAL", "")
                or DEFAULT_WATCH_FLUSH_INTERVAL
            ))
        except (TypeError, ValueError):
            self._watch_flush_interval = DEFAULT_WATCH_FLUSH_INTERVAL

        # ── Federation invitations ──────────────────────────────────
        federation = extra.get("federation") or {}
        if not isinstance(federation, dict):
            federation = {}
        raw_inviters = federation.get("allowed_inviters")
        if raw_inviters is None:
            raw_inviters = os.getenv("NEXTCLOUD_TALK_ALLOWED_INVITERS", "")
        if isinstance(raw_inviters, str):
            raw_inviters = [s for s in raw_inviters.split(",")]
        self._allowed_inviters: list[str] = []
        for entry in raw_inviters or []:
            cloud_id = _normalize_cloud_id(entry)
            if not cloud_id:
                continue
            if "@" not in cloud_id:
                logger.warning(
                    "nextcloud_talk: ignoring federation inviter %r — entries must be "
                    "full cloud ids (user@host)", entry,
                )
                continue
            self._allowed_inviters.append(cloud_id)
        try:
            self._federation_poll_interval = max(5, int(
                federation.get("poll_interval")
                or os.getenv("NEXTCLOUD_TALK_FEDERATION_POLL_INTERVAL", "")
                or DEFAULT_FEDERATION_POLL_INTERVAL
            ))
        except (TypeError, ValueError):
            self._federation_poll_interval = DEFAULT_FEDERATION_POLL_INTERVAL

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
        self._chat_name_cache: dict[str, str] = {}
        self._client: Optional[TalkUserClient] = None
        self._poll_tasks: list[asyncio.Task] = []
        self._shutdown: bool = False
        # "Thinking..." acks pending an edit-into-response, scoped PER
        # conversation (a single global slot raced across conversations).
        # Each chat keeps a FIFO of ack message ids: the gateway serializes
        # turns per session, so replies arrive in turn order and each reply
        # edits the oldest outstanding ack of its own conversation.
        self._pending_acks: dict[str, deque[int]] = {}
        # Watch-mode buffers, per conversation token. Only ever filled AFTER
        # the pre-effect authorization gate (the digest is an internal event
        # that bypasses the gateway's per-sender allowlist).
        self._watch_buffers: dict[str, list[dict[str, Any]]] = {}
        self._watch_flush_task: Optional[asyncio.Task] = None
        # Federation: invitation ids already decided in this process (accepted,
        # or logged once as "left pending"), so each poll stays quiet.
        self._invitation_task: Optional[asyncio.Task] = None
        self._invites_seen: dict[Any, str] = {}
        self._federation_unavailable_logged = False

    # ── Classification ──────────────────────────────────────────────

    def _classify_chat(self, chat_id: str) -> str:
        tokens = {c["token"] for c in self._conversations}
        return "group" if chat_id in tokens else "dm"

    def _set_room_mode(self, token: str, mode: Any) -> None:
        if not token:
            return
        normalized = _normalize_room_mode(mode, self._default_room_mode)
        if normalized != str(mode or "").strip().lower():
            logger.warning(
                "nextcloud_talk: unknown room mode %r for %s, using %r",
                mode, token, normalized,
            )
        self._room_modes[token] = normalized

    def _room_mode(self, chat_id: str) -> str:
        """Effective mode for a conversation. DMs are always ``free``."""
        if self._classify_chat(chat_id) == "dm":
            return ROOM_MODE_FREE
        return self._room_modes.get(chat_id, self._default_room_mode)

    def _has_watch_rooms(self) -> bool:
        tokens = {c["token"] for c in self._conversations}
        return any(self._room_mode(t) == ROOM_MODE_WATCH for t in tokens)

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

    async def get_chat_info(self, chat_id: str) -> dict[str, Any]:
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

        # Background loops are only started when something needs them, so a
        # default configuration makes no extra requests at all.
        if self._has_watch_rooms() and self._watch_flush_task is None:
            self._watch_flush_task = loop.create_task(self._watch_flush_loop())
        if self._allowed_inviters and self._invitation_task is None:
            self._invitation_task = loop.create_task(self._invitation_loop())

        logger.info(
            "nextcloud_talk: polling %d conversation(s) as %s",
            len(self._conversations), self._username,
        )
        return True

    async def disconnect(self) -> None:
        self._shutdown = True
        background = [self._watch_flush_task, self._invitation_task]
        self._watch_flush_task = None
        self._invitation_task = None
        for task in [*self._poll_tasks, *background]:
            if task is None:
                continue
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

    # ── Watch-mode digest loop ─────────────────────────────────────

    async def _watch_flush_loop(self) -> None:
        while not self._shutdown:
            try:
                await asyncio.sleep(self._watch_flush_interval)
            except asyncio.CancelledError:
                raise
            if self._shutdown:
                return
            try:
                await self._flush_watch_buffers()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("nextcloud_talk: watch-mode flush failed")

    def _buffer_watch_message(self, chat_id: str, parsed: dict[str, Any]) -> None:
        attachment = parsed.get("attachment") or {}
        buffer = self._watch_buffers.setdefault(chat_id, [])
        buffer.append({
            "user_id": parsed.get("user_id", ""),
            "user_name": parsed.get("user_name", ""),
            "text": parsed.get("text", ""),
            "message_id": parsed.get("message_id"),
            "attachment": attachment.get("name") if attachment else None,
        })
        if len(buffer) > WATCH_BUFFER_MAX_MESSAGES:
            del buffer[: len(buffer) - WATCH_BUFFER_MAX_MESSAGES]

    async def _flush_watch_buffers(self) -> None:
        """Hand every non-empty watch buffer to the agent as ONE digest turn.

        The digest is an ``internal`` event: it never interrupts a running
        turn, carries no reply-to, and a bare ``NO_REPLY`` answer makes the
        gateway send nothing. Its source is the most recent buffered sender
        — every entry passed the pre-effect authorization gate before it was
        buffered, so the adapter is the authority here.
        """
        for chat_id in list(self._watch_buffers):
            entries = self._watch_buffers.pop(chat_id, [])
            if not entries:
                continue
            last = entries[-1]
            room_label = next(
                (str(c.get("alias")) for c in self._conversations
                 if c.get("token") == chat_id and c.get("alias")),
                None,
            ) or self._chat_name_cache.get(chat_id) or chat_id
            source = self.build_source(
                chat_id=chat_id,
                chat_name=room_label,
                chat_type="group",
                user_id=last.get("user_id") or None,
                user_name=last.get("user_name") or None,
            )
            event = MessageEvent(
                text=_build_watch_digest(room_label, entries),
                message_type=MessageType.TEXT,
                source=source,
                message_id=f"watch-digest-{chat_id}-{last.get('message_id') or uuid.uuid4().hex[:8]}",
                internal=True,
            )
            try:
                await self.handle_message(event)
            except Exception:
                logger.exception("nextcloud_talk: watch digest for %s raised", chat_id)

    # ── Federation invitations ─────────────────────────────────────

    async def _invitation_loop(self) -> None:
        while not self._shutdown:
            try:
                await self._process_federation_invitations()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("nextcloud_talk: federation invitation check failed")
            try:
                await asyncio.sleep(self._federation_poll_interval)
            except asyncio.CancelledError:
                raise

    async def _process_federation_invitations(self) -> None:
        """Auto-accept PENDING invitations from allowlisted inviters only.

        Everything else is left pending for a human to decide — never
        declined. Each invitation is decided (and logged) once per process.
        """
        if not self._allowed_inviters or self._client is None:
            return
        ok, invitations, err, status = await self._client.list_federation_invitations()
        if not ok:
            if status == 404:
                if not self._federation_unavailable_logged:
                    logger.info(
                        "nextcloud_talk: federation invitation endpoint not available "
                        "(HTTP 404) — Talk federation disabled or too old"
                    )
                    self._federation_unavailable_logged = True
                return
            logger.warning("nextcloud_talk: could not list federation invitations: %s", err)
            return

        for invite in invitations:
            if not isinstance(invite, dict):
                continue
            invite_id = invite.get("id")
            if invite_id is None or invite_id in self._invites_seen:
                continue
            inviter = invite.get("inviterCloudId") or ""
            room_name = invite.get("roomName") or invite.get("remoteToken") or "?"
            if not _is_pending_invite(invite):
                self._invites_seen[invite_id] = "not-pending"
                continue
            if not _is_allowed_inviter(inviter, self._allowed_inviters):
                self._invites_seen[invite_id] = "left-pending"
                logger.info(
                    "nextcloud_talk: federation invitation %s to %r from %r left pending "
                    "(inviter not on allowed_inviters)",
                    invite_id, room_name, inviter,
                )
                continue
            ok, room, err = await self._client.accept_federation_invitation(invite_id)
            if not ok:
                # Not recorded as seen: a transient failure is retried next pass.
                logger.warning(
                    "nextcloud_talk: could not accept federation invitation %s from %r: %s",
                    invite_id, inviter, err,
                )
                continue
            self._invites_seen[invite_id] = "accepted"
            logger.info(
                "nextcloud_talk: accepted federation invitation %s to %r from %r",
                invite_id, room_name, inviter,
            )
            token = str((room or {}).get("token") or invite.get("localToken") or "").strip()
            if token:
                self._adopt_conversation(token, (room or {}).get("displayName") or room_name)

    def _adopt_conversation(self, token: str, name: Optional[str] = None) -> None:
        """Add a conversation to the live lane set and start polling it.

        Used for accepted federation invitations: allowlisting the inviter
        is the operator's explicit authorization for that conversation to
        become a lane. Caller authorization still runs per message. The
        adoption is process-local — persist it by adding the token to
        ``conversations`` in config.yaml.
        """
        if any(c.get("token") == token for c in self._conversations):
            return
        self._conversations.append({"token": token, "federated": True})
        if name:
            self._record_chat_name(token, str(name))
        logger.info(
            "nextcloud_talk: adopted conversation %s (%s) with room mode %r — add it to "
            "platforms.nextcloud_talk.extra.conversations to keep it across restarts",
            token, name or token, self._room_mode(token),
        )
        if self._client is not None and not self._shutdown and self._poll_tasks:
            self._poll_tasks.append(asyncio.get_event_loop().create_task(self._poll_loop(token)))
            if self._room_mode(token) == ROOM_MODE_WATCH and self._watch_flush_task is None:
                self._watch_flush_task = asyncio.get_event_loop().create_task(self._watch_flush_loop())

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
                    last_known_id = max(last_known_id, msg_id)
                    await self._on_poll_message(msg, token)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("nextcloud_talk: poll error, retrying in 5s")
                await asyncio.sleep(5)

    async def _on_poll_message(self, msg: dict[str, Any], token: str) -> None:
        parsed = _parse_user_message(msg, own_user_id=self._username)
        if parsed is None:
            return

        chat_id = parsed["chat_id"] or token
        text = parsed["text"]

        if not self._is_chat_allowed(chat_id):
            return

        # Caller authorization BEFORE any consequential effect.
        #
        # Conversation membership (``_is_chat_allowed``) only selects the lane;
        # it is a routing coordinate, not caller authority. The canonical,
        # profile-/multiplex-aware authorization owner is the gateway callback
        # the runner wired via ``set_authorization_check``. We consume that
        # proof here — right after projecting the native principal
        # ``(actorId, chat_type, chat_id)`` — so that an unallowlisted
        # participant in an otherwise-configured Talk conversation cannot
        # trigger ANY credentialed/observable side effect (local ``!help``
        # reply, "Thinking…" status send, ``_pending_acks`` mutation,
        # WebDAV attachment download, or STT egress) before the eventual
        # agent turn is refused downstream.
        #
        # Fail closed: only an explicit ``True`` permits effects. ``False``
        # (denied) and ``None`` (no check wired / unknown) both short-circuit.
        # The central gateway admission gate in ``handle_message`` stays as
        # defense in depth.
        actor_id = parsed["user_id"]
        chat_type = self._classify_chat(chat_id)
        if self._is_sender_authorized(actor_id, chat_type, chat_id) is not True:
            logger.info(
                "nextcloud_talk: dropping pre-effect message from unauthorized "
                "actor %r in %s conversation %s",
                actor_id, chat_type, chat_id,
            )
            return

        self._record_chat_name(chat_id, parsed.get("user_name", ""))

        # Normalize "!" prefix to "/" — Talk intercepts "/" commands client-side,
        # so users type "!approve" / "!new" / "!help" and the adapter rewrites.
        if text.startswith("!"):
            text = "/" + text[1:]

        # Room-mode gating (group conversations only; commands are always
        # addressed to the bot and bypass it). Runs AFTER the authorization
        # gate so a watch buffer never holds an unauthorized actor's message,
        # and BEFORE the "Thinking..." ack / attachment download so a gated
        # message costs nothing.
        mode = self._room_mode(chat_id)
        if mode != ROOM_MODE_FREE and not text.startswith("/"):
            mentioned = _is_mention(
                parsed["text"], msg.get("messageParameters"),
                self._username, self._display_name, chat_id,
            )
            if not mentioned:
                if mode == ROOM_MODE_WATCH:
                    self._buffer_watch_message(chat_id, parsed)
                return

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
        media_urls: list[str] = []
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
            chat_type=chat_type,
            user_id=actor_id,
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
        metadata: Optional[dict[str, Any]] = None,
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
    media_files: Optional[list[str]] = None,
    force_document: bool = False,
) -> dict[str, Any]:
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
            chunks: list[str] = []
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

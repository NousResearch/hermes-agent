"""WeChat (personal) messaging platform adapter via iLink long-polling API.

Based on the openclaw-weixin plugin protocol (https://github.com/Tencent/openclaw-weixin):
  - iLink gateway HTTP long-polling for message send/receive
  - AES-128-ECB media encryption
  - ClawBot QR code binding for authentication
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import secrets
import stat
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Pattern, Set
from urllib.parse import urlparse

try:
    import aiohttp

    AIOHTTP_AVAILABLE = True
except ImportError:
    aiohttp = None  # type: ignore[assignment]
    AIOHTTP_AVAILABLE = False

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RETRY_BACKOFF_SECONDS = (1, 2, 4, 8, 16, 30)
ALLOWED_ILINK_HOSTS = {"localhost", "127.0.0.1", "ilinkai.weixin.qq.com"}
_ILINK_APP_ID = "openclaw-weixin"
_ILINK_AUTH_TYPE = "ilink_bot_token"
_CHANNEL_VERSION = "hermes-gateway"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class WeChatAdapterConfig:
    ilink_url: str
    ilink_token: str
    require_mention: bool = False
    free_response_chats: Set[str] = field(default_factory=set)
    mention_patterns: List[Pattern[str]] = field(default_factory=list)
    poll_timeout_seconds: int = 35
    profile: str = "default"


def _as_bool(value, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("true", "1", "yes", "on"):
            return True
        if normalized in ("false", "0", "no", "off"):
            return False
    return default


def _as_str_set(value) -> Set[str]:
    if value is None:
        return set()
    if isinstance(value, list):
        return {str(v).strip() for v in value if str(v).strip()}
    return {part.strip() for part in str(value).split(",") if part.strip()}


def _compile_mention_patterns(raw_value) -> List[Pattern[str]]:
    if raw_value is None:
        return []

    value = raw_value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            value = json.loads(text)
        except Exception:
            value = [part.strip() for part in text.split(",") if part.strip()]

    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []

    patterns: List[Pattern[str]] = []
    for entry in value:
        if not isinstance(entry, str) or not entry.strip():
            continue
        try:
            patterns.append(re.compile(entry, re.IGNORECASE))
        except re.error:
            continue
    return patterns


def _from_platform_config(config: PlatformConfig) -> WeChatAdapterConfig:
    extra: Dict[str, object] = config.extra or {}
    ilink_url = str(
        extra.get("ilink_url")
        or os.getenv("WECHAT_ILINK_URL", "")
    ).strip()
    ilink_token = str(
        config.token
        or extra.get("ilink_token")
        or os.getenv("WECHAT_ILINK_TOKEN", "")
    ).strip()

    require_mention = extra.get("require_mention")
    if require_mention is None:
        require_mention = os.getenv("WECHAT_REQUIRE_MENTION")

    free_response = extra.get("free_response_chats")
    if free_response is None:
        free_response = extra.get("free_response")
    if free_response is None:
        free_response = os.getenv("WECHAT_FREE_RESPONSE_CHATS")

    mention_patterns = extra.get("mention_patterns")
    if mention_patterns is None:
        mention_patterns = os.getenv("WECHAT_MENTION_PATTERNS")

    poll_timeout_seconds = extra.get("poll_timeout_seconds")
    if poll_timeout_seconds is None:
        poll_timeout_seconds = os.getenv("WECHAT_POLL_TIMEOUT_SECONDS", "35")
    try:
        poll_timeout_seconds = int(poll_timeout_seconds)
    except Exception:
        poll_timeout_seconds = 35

    profile = str(extra.get("profile") or os.getenv("HERMES_ACTIVE_PROFILE", "")).strip() or None

    return WeChatAdapterConfig(
        ilink_url=ilink_url.rstrip("/"),
        ilink_token=ilink_token,
        require_mention=_as_bool(require_mention, default=False),
        free_response_chats=_as_str_set(free_response),
        mention_patterns=_compile_mention_patterns(mention_patterns),
        poll_timeout_seconds=max(5, min(120, poll_timeout_seconds)),
        profile=profile,
    )


# ---------------------------------------------------------------------------
# Session persistence
# ---------------------------------------------------------------------------

@dataclass
class WeChatSessionState:
    """Persistent WeChat auth/session state stored on disk."""

    bearer_token: Optional[str] = None
    context_tokens: Dict[str, Any] = field(default_factory=dict)
    cursor: Optional[str] = None
    base_url: Optional[str] = None


class WeChatSessionStore:
    """Manages ~/.hermes/profiles/<profile>/.wechat_session."""

    def __init__(self, profile: Optional[str] = None):
        self._profile = self._resolve_profile(profile)
        base = self._resolve_base_dir(self._profile)
        self.path = base / ".wechat_session"

    @staticmethod
    def _resolve_profile(profile: Optional[str]) -> str:
        from hermes_cli.config import get_hermes_home
        value = str(profile or "").strip()
        if value:
            return value
        hermes_home = get_hermes_home()
        if hermes_home.parent.name == "profiles" and hermes_home.name:
            return hermes_home.name
        env_profile = str(os.getenv("HERMES_ACTIVE_PROFILE", "")).strip()
        if env_profile:
            return env_profile
        active_profile_file = hermes_home / "active_profile"
        try:
            file_value = active_profile_file.read_text(encoding="utf-8").strip()
            if file_value:
                return file_value
        except OSError:
            pass
        return "default"

    @staticmethod
    def _resolve_base_dir(profile: str) -> Path:
        from hermes_cli.config import get_hermes_home
        hermes_home = get_hermes_home()
        # Already inside a profile directory (e.g. ~/.hermes/profiles/owl)
        if hermes_home.parent.name == "profiles":
            if hermes_home.name == profile:
                return hermes_home
            # Different profile requested — go up to ~/.hermes/ and back down
            return hermes_home.parent / profile
        # Root hermes home (e.g. ~/.hermes/)
        if profile == "default":
            return hermes_home
        return hermes_home / "profiles" / profile

    def load(self) -> WeChatSessionState:
        if not self.path.exists():
            return WeChatSessionState()
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            return WeChatSessionState(
                bearer_token=raw.get("bearer_token"),
                context_tokens=self._drop_sensitive_cdn_keys(raw.get("context_tokens") or {}),
                cursor=raw.get("cursor"),
                base_url=raw.get("base_url") or raw.get("baseurl"),
            )
        except Exception:
            return WeChatSessionState()

    def save(self, state: WeChatSessionState) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(state)
        payload["context_tokens"] = self._drop_sensitive_cdn_keys(payload.get("context_tokens") or {})

        tmp_path = Path(f"{self.path}.tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.chmod(tmp_path, stat.S_IRUSR | stat.S_IWUSR)
        tmp_path.replace(self.path)
        os.chmod(self.path, stat.S_IRUSR | stat.S_IWUSR)

    @classmethod
    def _drop_sensitive_cdn_keys(cls, value: Any) -> Any:
        if isinstance(value, dict):
            sanitized: Dict[str, Any] = {}
            for key, raw in value.items():
                key_text = str(key).strip()
                if cls._is_sensitive_cdn_key(key_text):
                    continue
                sanitized[key_text] = cls._drop_sensitive_cdn_keys(raw)
            return sanitized
        if isinstance(value, list):
            return [cls._drop_sensitive_cdn_keys(item) for item in value]
        return value

    @staticmethod
    def _is_sensitive_cdn_key(key: str) -> bool:
        lowered = key.lower()
        return "filekey" in lowered


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_wechat_uin() -> str:
    """X-WECHAT-UIN: random uint32 -> decimal string -> base64, matching iLink API spec."""
    uint32 = secrets.randbits(32)
    return base64.b64encode(str(uint32).encode()).decode()


def check_wechat_requirements() -> bool:
    return AIOHTTP_AVAILABLE


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class WeChatAdapter(BasePlatformAdapter):
    """WeChat iLink adapter using long-polling APIs."""

    MAX_MESSAGE_LENGTH = 4000

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.WECHAT)
        self._settings: WeChatAdapterConfig = _from_platform_config(config)
        self._wechat_session_store = WeChatSessionStore(profile=self._settings.profile)
        self._session_state: WeChatSessionState = self._wechat_session_store.load()
        self._api_base_url = self._resolve_runtime_base_url()

        self._http: Optional["aiohttp.ClientSession"] = None
        self._poll_task: Optional[asyncio.Task] = None
        self._mention_cleanup_re = re.compile(r"^\s*@?hermes[,:\-\s]*", re.IGNORECASE)
        # context_token cache: chat_id -> context_token for iLink outbound replies
        self._context_tokens: Dict[str, str] = {}

    async def connect(self) -> bool:
        if not AIOHTTP_AVAILABLE:
            message = "WeChat startup failed: aiohttp not installed"
            self._set_fatal_error("wechat_missing_dependency", message, retryable=True)
            logger.warning("[%s] %s", self.name, message)
            return False

        if not self._api_base_url:
            message = "WeChat startup failed: WECHAT_ILINK_URL or session base URL is required"
            self._set_fatal_error("wechat_missing_url", message, retryable=True)
            logger.warning("[%s] %s", self.name, message)
            return False
        parsed_url = urlparse(self._api_base_url)
        host = str(parsed_url.hostname or "").strip().lower()
        if host not in ALLOWED_ILINK_HOSTS:
            message = "WeChat startup failed: iLink base URL hostname not in allowlist"
            self._set_fatal_error("wechat_non_local_url", message, retryable=False)
            logger.warning("[%s] %s (got: %s)", self.name, message, self._api_base_url)
            return False
        if host not in {"localhost", "127.0.0.1"} and parsed_url.scheme != "https":
            message = "WeChat startup failed: remote iLink base URL requires HTTPS"
            self._set_fatal_error("wechat_insecure_remote_url", message, retryable=False)
            logger.warning("[%s] %s (got: %s)", self.name, message, self._api_base_url)
            return False

        if not self._settings.ilink_token and not self._session_state.bearer_token:
            message = "WeChat startup failed: WECHAT_ILINK_TOKEN is required"
            self._set_fatal_error("wechat_missing_token", message, retryable=True)
            logger.warning("[%s] %s", self.name, message)
            return False

        timeout = aiohttp.ClientTimeout(
            total=self._settings.poll_timeout_seconds + 10,
            connect=10,
            sock_connect=10,
            sock_read=self._settings.poll_timeout_seconds + 5,
        )
        self._http = aiohttp.ClientSession(timeout=timeout)
        self._mark_connected()
        self._poll_task = asyncio.create_task(self._poll_loop())
        logger.info("[%s] WeChat long-poll started at %s", self.name, self._api_base_url)
        return True

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

        if self._http:
            await self._http.close()
            self._http = None

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        if not self._http:
            return SendResult(success=False, error="WeChat adapter not connected", retryable=True)

        context_token = self._context_tokens.get(str(chat_id))
        client_id = f"hermes-wechat-{secrets.token_hex(8)}"
        msg: Dict[str, Any] = {
            "from_user_id": "",
            "to_user_id": str(chat_id),
            "client_id": client_id,
            "message_type": 2,   # BOT
            "message_state": 2,  # FINISH
            "item_list": [{"type": 1, "text_item": {"text": self.format_message(content)}}] if content else None,
        }
        if context_token:
            msg["context_token"] = context_token
        payload: Dict[str, Any] = {
            "msg": msg,
            "base_info": {"channel_version": _CHANNEL_VERSION},
        }

        try:
            async with self._http.post(
                self._url("ilink/bot/sendmessage"),
                headers=self._auth_headers(),
                json=payload,
            ) as resp:
                data = await self._safe_json(resp)
                if resp.status >= 400:
                    return SendResult(
                        success=False,
                        error=f"HTTP {resp.status}: {self._errmsg(data)}",
                        raw_response=data,
                        retryable=resp.status >= 500,
                    )
                if self._is_session_expired(data):
                    message = "WeChat session expired (errcode -14). Re-authentication required"
                    self._set_fatal_error("wechat_session_timeout", message, retryable=False)
                    await self._notify_fatal_error()
                    return SendResult(success=False, error=message, raw_response=data)
                if not self._is_ok(data):
                    return SendResult(
                        success=False,
                        error=self._errmsg(data),
                        raw_response=data,
                    )
                message_id = self._extract_message_id(data)
                return SendResult(success=True, message_id=message_id, raw_response=data)
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            return SendResult(success=False, error=str(exc), retryable=True)
        except Exception as exc:
            return SendResult(success=False, error=str(exc), retryable=False)

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        return None

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {"name": f"WeChat {chat_id}", "type": "dm", "chat_id": str(chat_id)}

    # -- Polling loop -------------------------------------------------------

    async def _poll_loop(self) -> None:
        backoff_idx = 0
        while self._running:
            try:
                await self._poll_once()
                backoff_idx = 0
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if not self._running:
                    return
                delay = _RETRY_BACKOFF_SECONDS[min(backoff_idx, len(_RETRY_BACKOFF_SECONDS) - 1)]
                backoff_idx += 1
                logger.warning("[%s] WeChat poll error, retrying in %ss: %s", self.name, delay, exc)
                await asyncio.sleep(delay)

    async def _poll_once(self) -> None:
        if not self._http:
            raise RuntimeError("HTTP session not initialized")

        payload: Dict[str, Any] = {
            "get_updates_buf": self._session_state.cursor or "",
            "base_info": {"channel_version": _CHANNEL_VERSION},
        }

        async with self._http.post(
            self._url("ilink/bot/getupdates"),
            headers=self._auth_headers(),
            json=payload,
        ) as resp:
            if not resp.ok:
                raise RuntimeError(f"getupdates HTTP {resp.status}")
            data = await self._safe_json(resp)

        if self._is_session_expired(data):
            message = "WeChat session expired (errcode -14). Re-authentication required"
            self._set_fatal_error("wechat_session_timeout", message, retryable=False)
            await self._notify_fatal_error()
            self._running = False
            return

        if not self._is_ok(data):
            raise RuntimeError(self._errmsg(data))

        self._merge_session_tokens(data)

        updates = self._extract_updates(data)
        for update in updates:
            event = self._to_message_event(update)
            if event is None:
                continue
            if not self._should_process_message(event):
                continue
            await self.handle_message(event)

    # -- Message parsing ----------------------------------------------------

    def _to_message_event(self, update: Dict[str, Any]) -> Optional[MessageEvent]:
        # iLink native WeixinMessage format: user fields at top level, content in item_list
        item_list = update.get("item_list")
        if isinstance(item_list, list):
            for item in item_list:
                if not isinstance(item, dict):
                    continue
                # Text messages
                if isinstance(item.get("text_item"), dict):
                    return self._from_ilink_text_item(item["text_item"], update)
                # Voice messages with server-side transcription
                voice_item = item.get("voice_item")
                if isinstance(voice_item, dict) and voice_item.get("text"):
                    return self._from_ilink_text_item({"text": voice_item["text"]}, update)
            ilink_item_types = sorted(i.get("type") for i in item_list if isinstance(i, dict) and i.get("type") is not None)
            logger.debug("[%s] wechat: WeixinMessage has no text items, item types=%s", self.name, ilink_item_types)
            return None

        # Legacy flat format: text_item at top level of update dict
        text_item = update.get("text_item")
        if isinstance(text_item, dict):
            return self._from_ilink_text_item(text_item, update)

        # Log and skip non-text iLink item types (key names are safe; values are not logged)
        ilink_item_keys = {k for k in update if k.endswith("_item")}
        if ilink_item_keys:
            logger.debug("[%s] wechat: skipping non-text iLink item type=%s", self.name, sorted(ilink_item_keys))
            return None

        # Legacy / simplified format (test stubs and older protocol variants)
        text = str(
            update.get("text")
            or update.get("content")
            or (update.get("message") or {}).get("text")
            or ""
        ).strip()
        if not text:
            return None

        chat_id = str(
            update.get("chat_id")
            or update.get("conversation_id")
            or (update.get("chat") or {}).get("id")
            or ""
        ).strip()
        if not chat_id:
            return None

        sender = update.get("from") or update.get("sender") or {}
        user_id = str(
            update.get("user_id")
            or sender.get("id")
            or sender.get("user_id")
            or ""
        ).strip()
        user_name = str(
            update.get("user_name")
            or sender.get("name")
            or sender.get("nickname")
            or ""
        ).strip() or None

        is_group = bool(
            update.get("is_group")
            or update.get("chat_type") == "group"
            or str(chat_id).startswith("group:")
        )
        chat_type = "group" if is_group else "dm"

        cleaned_text = self._clean_mention_prefix(text)

        source = self.build_source(
            chat_id=chat_id,
            chat_type=chat_type,
            user_id=user_id or None,
            user_name=user_name,
            chat_name=(update.get("chat") or {}).get("name") if isinstance(update.get("chat"), dict) else None,
        )
        return MessageEvent(
            text=cleaned_text,
            message_type=MessageType.TEXT,
            source=source,
            raw_message=update,
            message_id=str(update.get("message_id") or update.get("msgid") or "") or None,
            timestamp=datetime.now(),
        )

    def _from_ilink_text_item(self, text_item: Dict[str, Any], raw: Dict[str, Any]) -> Optional[MessageEvent]:
        """Parse an iLink item_list text_item into a MessageEvent."""
        text = str(text_item.get("text") or "").strip()
        if not text:
            return None

        # User/session fields live on the parent WeixinMessage (raw), not on TextItem.
        # Fall back to text_item for legacy flat formats where these may be co-located.
        from_user_id = str(raw.get("from_user_id") or text_item.get("from_user_id") or "").strip()
        to_user_id = str(raw.get("to_user_id") or text_item.get("to_user_id") or "").strip()
        session_id = str(raw.get("session_id") or text_item.get("session_id") or "").strip()
        context_token = str(raw.get("context_token") or text_item.get("context_token") or "").strip()

        # Group detection: WeChat group wxids end with @chatroom
        is_group = to_user_id.endswith("@chatroom") or session_id.endswith("@chatroom")

        # Reply target: group wxid for groups, sender wxid for DMs
        reply_to_user = to_user_id if is_group else from_user_id
        # chat_id: for groups use session_id as stable key; for DMs use from_user_id
        # (matching OpenClaw — DM replies must target from_user_id, not session_id)
        chat_id = (session_id or reply_to_user) if is_group else (from_user_id or reply_to_user)
        if not chat_id:
            return None

        # Cache context_token for outbound replies to this chat
        if context_token:
            self._context_tokens[chat_id] = context_token

        chat_type = "group" if is_group else "dm"
        cleaned_text = self._clean_mention_prefix(text)
        source = self.build_source(
            chat_id=chat_id,
            chat_type=chat_type,
            user_id=from_user_id or None,
            thread_id=session_id if is_group and session_id else None,
        )
        return MessageEvent(
            text=cleaned_text,
            message_type=MessageType.TEXT,
            source=source,
            raw_message=raw,
            message_id=str(raw.get("msgid") or raw.get("message_id") or "") or None,
            timestamp=datetime.now(),
        )

    def _should_process_message(self, event: MessageEvent) -> bool:
        if event.source.chat_type != "group":
            return True

        chat_id = str(event.source.chat_id)
        if chat_id in self._settings.free_response_chats:
            return True
        if not self._settings.require_mention:
            return True

        raw = event.raw_message if isinstance(event.raw_message, dict) else {}
        text_item = raw.get("text_item")
        if isinstance(text_item, dict):
            original_text = str(text_item.get("text") or "")
        else:
            original_text = str(raw.get("text") or raw.get("content") or "")
        check_text = original_text or (event.text or "")

        if "@hermes" in check_text.lower():
            return True
        return any(pattern.search(check_text) for pattern in self._settings.mention_patterns)

    # -- Session / auth helpers ---------------------------------------------

    def _merge_session_tokens(self, response: Dict[str, Any]) -> None:
        changed = False

        cursor = response.get("get_updates_buf") or response.get("cursor") or response.get("next_cursor")
        if cursor is not None and str(cursor) != str(self._session_state.cursor):
            self._session_state.cursor = str(cursor)
            changed = True

        bearer = response.get("bearer_token") or response.get("token") or response.get("access_token")
        if bearer and bearer != self._session_state.bearer_token:
            self._session_state.bearer_token = str(bearer)
            changed = True

        context_tokens = response.get("context_tokens") or response.get("context")
        if isinstance(context_tokens, dict) and context_tokens != self._session_state.context_tokens:
            self._session_state.context_tokens = context_tokens
            changed = True

        base_url = self._extract_base_url(response)
        normalized_base_url = self._normalize_base_url(base_url)
        if normalized_base_url and normalized_base_url != str(self._session_state.base_url or ""):
            self._session_state.base_url = normalized_base_url
            self._api_base_url = normalized_base_url
            changed = True

        if changed:
            self._wechat_session_store.save(self._session_state)

    @staticmethod
    def _extract_updates(response: Dict[str, Any]) -> List[Dict[str, Any]]:
        if isinstance(response.get("updates"), list):
            return [item for item in response["updates"] if isinstance(item, dict)]
        if isinstance(response.get("messages"), list):
            return [item for item in response["messages"] if isinstance(item, dict)]
        # iLink native: { msgs: WeixinMessage[] } where each WeixinMessage has item_list
        msgs = response.get("msgs")
        if isinstance(msgs, list):
            return [item for item in msgs if isinstance(item, dict)]
        if isinstance(msgs, dict):
            item_list = msgs.get("item_list")
            if isinstance(item_list, list):
                return [item for item in item_list if isinstance(item, dict)]
        if isinstance(response.get("data"), dict):
            data = response["data"]
            data_msgs = data.get("msgs")
            if isinstance(data_msgs, dict):
                item_list = data_msgs.get("item_list")
                if isinstance(item_list, list):
                    return [item for item in item_list if isinstance(item, dict)]
            for key in ("updates", "messages", "items"):
                if isinstance(data.get(key), list):
                    return [item for item in data[key] if isinstance(item, dict)]
        return []

    def _auth_headers(self) -> Dict[str, str]:
        token = self._session_state.bearer_token or self._settings.ilink_token
        headers = {
            "Content-Type": "application/json",
            "AuthorizationType": _ILINK_AUTH_TYPE,
            "X-WECHAT-UIN": _random_wechat_uin(),
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        # SKRouteTag for clustered iLink deployments (matching openclaw-weixin)
        route_tag = os.getenv("WECHAT_ILINK_ROUTE_TAG", "").strip()
        if route_tag:
            headers["SKRouteTag"] = route_tag
        return headers

    def _url(self, path: str) -> str:
        base = self._api_base_url.rstrip("/") + "/"
        return f"{base}{path.lstrip('/')}"

    def _resolve_runtime_base_url(self) -> str:
        session_base_url = self._normalize_base_url(self._session_state.base_url)
        if session_base_url:
            return session_base_url
        return self._normalize_base_url(self._settings.ilink_url)

    @staticmethod
    def _extract_base_url(response: Dict[str, Any]) -> str:
        for key in ("base_url", "baseurl", "ilink_url", "ilinkurl"):
            value = response.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        data = response.get("data")
        if isinstance(data, dict):
            for key in ("base_url", "baseurl", "ilink_url", "ilinkurl"):
                value = data.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return ""

    @staticmethod
    def _normalize_base_url(value: Optional[str]) -> str:
        text = str(value or "").strip().rstrip("/")
        if not text:
            return ""
        parsed = urlparse(text)
        host = str(parsed.hostname or "").strip().lower()
        if parsed.scheme not in ("http", "https"):
            return ""
        if host not in ALLOWED_ILINK_HOSTS:
            return ""
        if host not in {"localhost", "127.0.0.1"} and parsed.scheme != "https":
            return ""
        return text

    @staticmethod
    async def _safe_json(resp: "aiohttp.ClientResponse") -> Dict[str, Any]:
        body = await resp.text()
        if not body:
            return {}
        try:
            data = json.loads(body)
            if isinstance(data, dict):
                return data
            return {"data": data}
        except json.JSONDecodeError:
            return {"raw": body}

    @staticmethod
    def _is_ok(response: Dict[str, Any]) -> bool:
        # Check both errcode and ret fields (server may use either)
        for key in ("errcode", "ret"):
            val = response.get(key)
            if val is not None and val != 0 and str(val) != "0":
                return False
        return True

    @staticmethod
    def _is_session_expired(response: Dict[str, Any]) -> bool:
        # Session expiry can appear in either errcode or ret
        _EXPIRED = -14
        return response.get("errcode") == _EXPIRED or response.get("ret") == _EXPIRED

    @staticmethod
    def _errmsg(response: Dict[str, Any]) -> str:
        return str(response.get("errmsg") or response.get("message") or response.get("error") or "request failed")

    @staticmethod
    def _extract_message_id(response: Dict[str, Any]) -> Optional[str]:
        for key in ("message_id", "msgid", "id"):
            value = response.get(key)
            if value:
                return str(value)
        if isinstance(response.get("data"), dict):
            data = response["data"]
            for key in ("message_id", "msgid", "id"):
                value = data.get(key)
                if value:
                    return str(value)
        return None

    def _clean_mention_prefix(self, text: str) -> str:
        cleaned = self._mention_cleanup_re.sub("", text or "").strip()
        return cleaned or text

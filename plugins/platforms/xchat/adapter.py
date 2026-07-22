"""X Chat platform adapter (Hermes plugin).

Connects the Hermes gateway to X's end-to-end encrypted direct messages
via the official X Chat API. All plaintext stays local: inbound
``encoded_event`` blobs are decrypted with the Chat XDK (``chatxdk``) and
outbound replies are encrypted + signed before they ever reach X.

Transport model
---------------
Inbound is a polling loop over ``GET /2/chat/conversations/{id}/events``
(the same shape as X's own bot example). Conversations are auto-discovered
via ``GET /2/chat/conversations`` (or pinned with
``XCHAT_CONVERSATION_IDS``); each is polled every ``XCHAT_POLL_INTERVAL``
seconds with exponential backoff on errors. Outbound goes through
``POST /2/chat/conversations/{id}/messages``.

Identity / key state (written by ``hermes xchat setup``):

* ``XCHAT_ACCESS_TOKEN``   OAuth2 user token (dm.read, dm.write, users.read, tweet.read)
* ``XCHAT_USER_ID``        the bot account's numeric user id
* ``XCHAT_SIGNING_KEY_VERSION``  registered public-key version
* private-key blob at ``~/.hermes/xchat/private_keys.b64`` (mode 600), or
  ``XCHAT_PRIVATE_KEYS_B64`` env override

The E2EE session is one ``chat_xdk.Chat`` instance with ``set_identity`` +
``set_cache_keys(True)``: KeyChange events route through the batch decrypt
path to feed the verified-key cache, so encrypt calls need no explicit
conversation key.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)

from .api import HTTPX_AVAILABLE, XChatApi, XChatApiError, XChatRateLimited
from .crypto import XChatCrypto, message_text

logger = logging.getLogger(__name__)

# X DMs cap out around 10k chars; stay under it so chunking kicks in first.
MAX_MESSAGE_LENGTH = 9500

DEFAULT_POLL_INTERVAL = 10.0
DISCOVERY_INTERVAL = 300.0  # re-list conversations every 5 minutes
ERROR_BACKOFF = [5, 15, 30, 60, 120]
DEDUP_MAX_SIZE = 5000

# Group-chat mention wake words — same defaults as the other Hermes channels
# so group gating behaves identically everywhere.
_DEFAULT_MENTION_PATTERNS = [
    r"(?<![\w@])@?hermes\s+agent\b[,:\-]?",
    r"(?<![\w@])@?hermes\b[,:\-]?",
]


def _state_dir() -> Path:
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "xchat"


def _read_key_blob() -> str:
    """Private-key blob: env override first, then the setup-written file."""
    env_blob = os.getenv("XCHAT_PRIVATE_KEYS_B64", "").strip()
    if env_blob:
        return env_blob
    blob_path = _state_dir() / "private_keys.b64"
    try:
        return blob_path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def check_requirements() -> bool:
    """True when the adapter is minimally configured (token + key material).

    Deliberately does NOT import chatxdk — the native SDK lazy-installs at
    connect time; a pre-flight check must stay cheap.
    """
    if not HTTPX_AVAILABLE:
        return False
    if not os.getenv("XCHAT_ACCESS_TOKEN", "").strip():
        return False
    return bool(_read_key_blob())


def validate_config(config) -> bool:
    extra = getattr(config, "extra", {}) or {}
    token = extra.get("access_token") or os.getenv("XCHAT_ACCESS_TOKEN", "")
    return bool(token)


def is_connected(config) -> bool:
    extra = getattr(config, "extra", {}) or {}
    token = os.getenv("XCHAT_ACCESS_TOKEN") or extra.get("access_token", "")
    return bool(token)


def _compile_mention_patterns(raw: Any) -> List[re.Pattern]:
    """Accept list / JSON string / comma- or newline-separated string / None."""
    patterns: List[str]
    if raw is None or raw == "":
        patterns = _DEFAULT_MENTION_PATTERNS
    elif isinstance(raw, list):
        patterns = [str(p) for p in raw if str(p).strip()]
    else:
        text = str(raw).strip()
        if text.startswith("["):
            try:
                patterns = [str(p) for p in json.loads(text)]
            except (ValueError, TypeError):
                patterns = [text]
        else:
            parts = re.split(r"[\n,]+", text)
            patterns = [p.strip() for p in parts if p.strip()]
        if not patterns:
            patterns = _DEFAULT_MENTION_PATTERNS
    compiled = []
    for p in patterns:
        try:
            compiled.append(re.compile(p, re.IGNORECASE))
        except re.error:
            logger.warning("[xchat] invalid mention pattern skipped: %r", p)
    return compiled or [re.compile(p, re.IGNORECASE) for p in _DEFAULT_MENTION_PATTERNS]


class XChatAdapter(BasePlatformAdapter):
    """X Chat (encrypted X DMs) adapter."""

    MAX_MESSAGE_LENGTH = MAX_MESSAGE_LENGTH

    def __init__(self, config: PlatformConfig):
        platform = Platform("xchat")
        super().__init__(config=config, platform=platform)

        extra = config.extra or {}
        self._access_token: str = (
            extra.get("access_token") or os.getenv("XCHAT_ACCESS_TOKEN", "")
        ).strip()
        self._refresh_token: str = (
            extra.get("refresh_token") or os.getenv("XCHAT_REFRESH_TOKEN", "")
        ).strip()
        self._client_id: str = (
            extra.get("client_id") or os.getenv("XCHAT_CLIENT_ID", "")
        ).strip()
        self._client_secret: str = (
            extra.get("client_secret") or os.getenv("XCHAT_CLIENT_SECRET", "")
        ).strip()
        self._bot_user_id: str = str(
            extra.get("user_id") or os.getenv("XCHAT_USER_ID", "")
        ).strip()
        self._signing_key_version: str = str(
            extra.get("signing_key_version")
            or os.getenv("XCHAT_SIGNING_KEY_VERSION", "1")
        ).strip() or "1"

        try:
            self._poll_interval = float(
                extra.get("poll_interval") or os.getenv("XCHAT_POLL_INTERVAL", "") or DEFAULT_POLL_INTERVAL
            )
        except (TypeError, ValueError):
            self._poll_interval = DEFAULT_POLL_INTERVAL
        self._poll_interval = max(2.0, self._poll_interval)

        # Pinned conversations (skip discovery when set).
        conv_raw = extra.get("conversation_ids") or os.getenv("XCHAT_CONVERSATION_IDS", "")
        if isinstance(conv_raw, list):
            self._pinned_conversations = [str(c).strip() for c in conv_raw if str(c).strip()]
        else:
            self._pinned_conversations = [
                c.strip() for c in str(conv_raw).split(",") if c.strip()
            ]

        # Group mention gating.
        env_require = os.getenv("XCHAT_REQUIRE_MENTION")
        if env_require is not None:
            self.require_mention = env_require.strip().lower() in {"1", "true", "yes"}
        else:
            self.require_mention = bool(extra.get("require_mention", False))
        self._mention_patterns = _compile_mention_patterns(
            extra.get("mention_patterns") or os.getenv("XCHAT_MENTION_PATTERNS")
        )

        # Runtime state
        self._api: Optional[XChatApi] = None
        self._crypto: Optional[XChatCrypto] = None
        self._poll_task: Optional[asyncio.Task] = None
        self._running = False
        self._lock_acquired = False

        # Per-conversation cursors + dedup
        self._conversations: Set[str] = set(self._pinned_conversations)
        self._backlog_loaded: Set[str] = set()
        self._seen_event_ids: Dict[str, float] = {}
        self._conversation_keys: Dict[str, Dict[str, bytes]] = {}
        self._last_event_id: Dict[str, str] = {}

        # Signing-key roster (accumulated; the SDK store is replaced wholesale)
        self._signing_keys: List[Dict[str, str]] = []
        self._known_senders: Set[str] = set()

        logger.info(
            "[xchat] adapter initialized: user_id=%s poll=%.0fs pinned=%d refresh=%s",
            self._bot_user_id or "?",
            self._poll_interval,
            len(self._pinned_conversations),
            "yes" if (self._refresh_token and self._client_id) else "no",
        )

    # -- Connection lifecycle -------------------------------------------------

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        if not HTTPX_AVAILABLE:
            logger.warning("[xchat] httpx not installed")
            return False
        if not self._access_token:
            logger.warning("[xchat] XCHAT_ACCESS_TOKEN not configured — run `hermes xchat setup`")
            return False

        key_blob = _read_key_blob()
        if not key_blob:
            logger.warning(
                "[xchat] no private-key blob found (~/.hermes/xchat/private_keys.b64 "
                "or XCHAT_PRIVATE_KEYS_B64) — run `hermes xchat setup`"
            )
            return False

        # One credential = one gateway. Prevents two profiles polling (and
        # double-replying) on the same bot account.
        try:
            from gateway.status import acquire_scoped_lock

            ok, holder = acquire_scoped_lock("xchat", self._access_token[:16])
            if not ok:
                logger.error("[xchat] credential already in use by another gateway: %s", holder)
                return False
            self._lock_acquired = True
        except Exception:
            logger.debug("[xchat] scoped lock unavailable; continuing", exc_info=True)

        self._api = XChatApi(
            self._access_token,
            refresh_token=self._refresh_token,
            client_id=self._client_id,
            client_secret=self._client_secret,
            on_token_refresh=self._persist_rotated_tokens,
        )

        # Derive the bot's own user id when not configured.
        if not self._bot_user_id:
            try:
                me = await self._api.get_my_user()
                self._bot_user_id = str(me.get("id") or "")
            except XChatApiError as e:
                logger.error("[xchat] failed to resolve bot user id: %s", e)
                await self._teardown()
                return False
        if not self._bot_user_id:
            logger.error("[xchat] could not determine bot user id")
            await self._teardown()
            return False

        # Unlock the E2EE session. chatxdk lazy-installs here on first use.
        try:
            crypto = XChatCrypto()
            crypto.load_keys(key_blob, self._signing_key_version)
            crypto.set_identity(self._bot_user_id)
            crypto.set_cache_keys(True)
            self._crypto = crypto
        except Exception as e:
            logger.error("[xchat] failed to initialize Chat XDK session: %s", e)
            await self._teardown()
            return False

        self._running = True
        self._poll_task = asyncio.create_task(self._run_poll_loop())
        self._mark_connected()
        logger.info("[xchat] connected as user %s", self._bot_user_id)
        return True

    async def disconnect(self) -> None:
        self._running = False
        if self._poll_task is not None:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except (asyncio.CancelledError, Exception):
                pass
            self._poll_task = None
        await self._teardown()
        logger.info("[xchat] disconnected")

    async def _teardown(self) -> None:
        if self._api is not None:
            await self._api.aclose()
            self._api = None
        if self._lock_acquired:
            try:
                from gateway.status import release_scoped_lock

                release_scoped_lock("xchat", self._access_token[:16])
            except Exception:
                pass
            self._lock_acquired = False

    async def _persist_rotated_tokens(self, access_token: str, refresh_token: str) -> None:
        """X rotates the refresh token on every renewal — persist both to .env."""
        self._access_token = access_token
        self._refresh_token = refresh_token
        try:
            from hermes_cli.config import save_env_value

            save_env_value("XCHAT_ACCESS_TOKEN", access_token)
            if refresh_token:
                save_env_value("XCHAT_REFRESH_TOKEN", refresh_token)
        except Exception:
            logger.warning("[xchat] failed to persist rotated OAuth tokens", exc_info=True)

    # -- Polling loop -----------------------------------------------------------

    async def _run_poll_loop(self) -> None:
        backoff_idx = 0
        last_discovery = 0.0
        while self._running:
            try:
                now = time.monotonic()
                if not self._pinned_conversations and (
                    now - last_discovery >= DISCOVERY_INTERVAL or not self._conversations
                ):
                    await self._discover_conversations()
                    last_discovery = now

                for conv_id in list(self._conversations):
                    if not self._running:
                        return
                    await self._poll_conversation(conv_id)

                backoff_idx = 0
                await asyncio.sleep(self._poll_interval)
            except asyncio.CancelledError:
                return
            except XChatRateLimited as e:
                wait = max(30.0, (e.reset_epoch - time.time()) if e.reset_epoch else 60.0)
                wait = min(wait, 900.0)
                logger.warning("[xchat] rate limited — sleeping %.0fs", wait)
                await asyncio.sleep(wait)
            except Exception as e:
                delay = ERROR_BACKOFF[min(backoff_idx, len(ERROR_BACKOFF) - 1)]
                backoff_idx += 1
                logger.warning("[xchat] poll error (retry in %ds): %s", delay, e)
                await asyncio.sleep(delay)

    async def _discover_conversations(self) -> None:
        assert self._api is not None
        token: Optional[str] = None
        found: Set[str] = set()
        for _ in range(10):  # hard page cap
            page = await self._api.get_conversations(max_results=100, pagination_token=token)
            for conv in page.get("data") or []:
                cid = str(conv.get("conversation_id") or conv.get("id") or "").strip()
                if cid:
                    found.add(cid)
            token = (page.get("meta") or {}).get("next_token")
            if not token:
                break
        new = found - self._conversations
        if new:
            logger.info("[xchat] discovered %d new conversation(s)", len(new))
        self._conversations |= found

    async def _poll_conversation(self, conv_id: str) -> None:
        assert self._api is not None and self._crypto is not None
        page = await self._api.get_events(conv_id, max_results=50)
        raw = page.get("data") or []
        if not raw:
            return

        # Events arrive newest-first; process oldest-first.
        raw = list(reversed(raw))
        await self._register_signing_keys(raw)

        if conv_id not in self._backlog_loaded:
            # First sight of this conversation: batch-decrypt to seed the
            # SDK's verified-key cache, but do NOT reply to the backlog.
            events_b64 = [e["encoded_event"] for e in raw if e.get("encoded_event")]
            if events_b64:
                try:
                    batch = self._crypto.decrypt_batch(events_b64)
                    keys = (batch.get("conversation_keys") or {}).get("keys") or {}
                    self._conversation_keys.setdefault(conv_id, {}).update(keys)
                except Exception as e:
                    logger.warning("[xchat] backlog decrypt failed conv=%s: %s", conv_id, e)
            for item in raw:
                eid = str(item.get("id") or "")
                if eid:
                    self._seen_event_ids[eid] = time.time()
            self._backlog_loaded.add(conv_id)
            return

        for item in raw:
            event_id = str(item.get("id") or "")
            if not event_id or event_id in self._seen_event_ids:
                continue
            self._seen_event_ids[event_id] = time.time()
            self._prune_dedup()

            event_b64 = item.get("encoded_event")
            if not event_b64:
                continue
            try:
                event = self._crypto.decrypt_one(
                    event_b64, self._conversation_keys.get(conv_id) or None
                )
            except Exception as e:
                logger.warning("[xchat] decrypt failed conv=%s event=%s: %s", conv_id, event_id, e)
                continue

            etype = event.get("type")
            if etype == "KeyChange":
                # Key rotation: route through the batch path — it verifies the
                # change and feeds the SDK's verified-key cache.
                try:
                    rotated = self._crypto.decrypt_batch([event_b64])
                    keys = (rotated.get("conversation_keys") or {}).get("keys") or {}
                    self._conversation_keys.setdefault(conv_id, {}).update(keys)
                except Exception as e:
                    logger.warning("[xchat] key-change processing failed conv=%s: %s", conv_id, e)
                continue
            if etype != "Message":
                continue

            sender_id = str(event.get("sender_id") or item.get("sender_id") or "")
            if sender_id == self._bot_user_id:
                continue  # echo of our own reply

            text = message_text(event)
            if not text:
                continue

            # The signature covers the canonical conversation id embedded in
            # the event — prefer it for replies.
            canonical_conv = str(event.get("conversation_id") or conv_id)
            await self._dispatch_inbound(
                conv_id=canonical_conv,
                sender_id=sender_id,
                text=text,
                message_id=event_id,
                raw=item,
            )

    def _prune_dedup(self) -> None:
        if len(self._seen_event_ids) <= DEDUP_MAX_SIZE:
            return
        # Drop the oldest half.
        items = sorted(self._seen_event_ids.items(), key=lambda kv: kv[1])
        for eid, _ in items[: len(items) // 2]:
            self._seen_event_ids.pop(eid, None)

    async def _register_signing_keys(self, events: List[Dict[str, Any]]) -> None:
        """Fetch new senders' public keys into the SDK's signing-key store."""
        assert self._api is not None and self._crypto is not None
        senders = {
            str(e.get("sender_id"))
            for e in events
            if e.get("sender_id") and str(e.get("sender_id")) != self._bot_user_id
        } - self._known_senders
        for sender_id in senders:
            try:
                for pk in await self._api.get_public_keys(sender_id):
                    self._signing_keys.append(
                        {
                            "user_id": sender_id,
                            "public_key_version": str(pk.get("public_key_version") or ""),
                            "public_key": pk.get("signing_public_key") or "",
                            "identity_public_key": pk.get("public_key") or "",
                            "identity_public_key_signature": pk.get("identity_public_key_signature") or "",
                        }
                    )
                self._known_senders.add(sender_id)
            except Exception:
                logger.warning("[xchat] public-key fetch failed sender=%s", sender_id)
        if senders and self._signing_keys:
            # The SDK store is replaced wholesale — push the full roster.
            self._crypto.set_signing_keys(self._signing_keys)

    # -- Inbound dispatch --------------------------------------------------------

    def _message_matches_mention_patterns(self, text: str) -> bool:
        return any(p.search(text) for p in self._mention_patterns)

    def _clean_mention_text(self, text: str) -> str:
        """Strip ONLY a leading wake-word match — never mid-prompt words."""
        stripped = text.lstrip()
        for p in self._mention_patterns:
            m = p.match(stripped)
            if m:
                return stripped[m.end():].lstrip()
        return text

    async def _dispatch_inbound(
        self,
        *,
        conv_id: str,
        sender_id: str,
        text: str,
        message_id: str,
        raw: Dict[str, Any],
    ) -> None:
        is_group = conv_id.startswith("g")
        chat_type = "group" if is_group else "dm"

        if is_group and self.require_mention:
            if not self._message_matches_mention_patterns(text):
                return
            text = self._clean_mention_text(text)
            if not text:
                return

        source = self.build_source(
            chat_id=conv_id,
            chat_name=None,
            chat_type=chat_type,
            user_id=sender_id,
            user_name=None,
            message_id=message_id,
        )
        event = MessageEvent(
            text=text,
            message_type=MessageType.TEXT,
            source=source,
            raw_message=raw,
            message_id=message_id,
        )
        await self.handle_message(event)

    # -- Outbound ------------------------------------------------------------------

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        if self._api is None or self._crypto is None:
            return SendResult(success=False, error="xchat adapter not connected")
        if len(content) > MAX_MESSAGE_LENGTH:
            content = content[:MAX_MESSAGE_LENGTH]
        try:
            body = self._crypto.encrypt_text(chat_id, content)
        except ValueError:
            # No verified conversation key cached yet. For a 1:1, the key
            # cache seeds from the conversation backlog; a brand-new
            # conversation the bot initiates needs a key-change first —
            # out of scope for reply flows (the poll loop always seeds
            # keys before we ever reply).
            return SendResult(
                success=False,
                error=(
                    "No verified conversation key for this conversation yet. "
                    "The key cache seeds from inbound events — reply flows "
                    "always have it; initiating brand-new conversations is "
                    "not supported yet."
                ),
            )
        except Exception as e:
            return SendResult(success=False, error=f"encrypt failed: {e}")
        try:
            out = await self._api.send_message(chat_id, body)
        except XChatApiError as e:
            logger.warning("[xchat] send failed conv=%s: %s", chat_id, e)
            return SendResult(success=False, error=str(e))
        data = out.get("data") or {}
        msg_id = str(data.get("message_id") or body.get("message_id") or "")
        # Suppress the echo when it comes back around the poll loop.
        for eid_key in ("event_id", "id"):
            eid = data.get(eid_key)
            if eid:
                self._seen_event_ids[str(eid)] = time.time()
        return SendResult(success=True, message_id=msg_id)

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        if self._api is None:
            return
        try:
            await self._api.send_typing(chat_id)
        except Exception:
            pass  # best-effort

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        chat_type = "group" if str(chat_id).startswith("g") else "dm"
        return {"name": str(chat_id), "type": chat_type, "chat_id": str(chat_id)}


# ---------------------------------------------------------------------------
# Plugin registration


def _env_enablement() -> Optional[dict]:
    """Seed ``PlatformConfig.extra`` from env vars during gateway config load."""
    token = os.getenv("XCHAT_ACCESS_TOKEN", "").strip()
    if not token:
        return None
    seed: dict = {"access_token": token}
    for env, key in (
        ("XCHAT_REFRESH_TOKEN", "refresh_token"),
        ("XCHAT_CLIENT_ID", "client_id"),
        ("XCHAT_CLIENT_SECRET", "client_secret"),
        ("XCHAT_USER_ID", "user_id"),
        ("XCHAT_SIGNING_KEY_VERSION", "signing_key_version"),
        ("XCHAT_CONVERSATION_IDS", "conversation_ids"),
        ("XCHAT_POLL_INTERVAL", "poll_interval"),
    ):
        val = os.getenv(env, "").strip()
        if val:
            seed[key] = val
    home = os.getenv("XCHAT_HOME_CHANNEL", "").strip()
    if home:
        seed["home_channel"] = {
            "chat_id": home,
            "name": os.getenv("XCHAT_HOME_CHANNEL_NAME", home),
        }
    return seed


async def _standalone_send(
    pconfig,
    chat_id: str,
    message: str,
    *,
    thread_id: Optional[str] = None,
    media_files: Optional[List[Any]] = None,
    force_document: bool = False,
) -> Dict[str, Any]:
    """Out-of-process encrypted send for cron / send_message_tool.

    Opens an ephemeral API client + Chat XDK session, seeds the
    conversation key from the conversation's event backlog, encrypts,
    sends, and closes. ``thread_id`` / ``media_files`` are accepted for
    signature parity — X Chat has no thread primitive and media requires
    the full streaming-encrypt flow (not wired yet).
    """
    if not HTTPX_AVAILABLE:
        return {"error": "xchat standalone send: httpx not installed"}

    extra = getattr(pconfig, "extra", {}) or {}
    access_token = (extra.get("access_token") or os.getenv("XCHAT_ACCESS_TOKEN", "")).strip()
    if not access_token:
        return {"error": "xchat standalone send: XCHAT_ACCESS_TOKEN not configured"}
    key_blob = _read_key_blob()
    if not key_blob:
        return {"error": "xchat standalone send: private-key blob missing — run `hermes xchat setup`"}
    user_id = str(extra.get("user_id") or os.getenv("XCHAT_USER_ID", "")).strip()
    key_version = str(
        extra.get("signing_key_version") or os.getenv("XCHAT_SIGNING_KEY_VERSION", "1")
    ).strip() or "1"

    api = XChatApi(
        access_token,
        refresh_token=(extra.get("refresh_token") or os.getenv("XCHAT_REFRESH_TOKEN", "")).strip(),
        client_id=(extra.get("client_id") or os.getenv("XCHAT_CLIENT_ID", "")).strip(),
        client_secret=(extra.get("client_secret") or os.getenv("XCHAT_CLIENT_SECRET", "")).strip(),
    )
    try:
        if not user_id:
            me = await api.get_my_user()
            user_id = str(me.get("id") or "")
        if not user_id:
            return {"error": "xchat standalone send: could not resolve bot user id"}

        crypto = XChatCrypto()
        crypto.load_keys(key_blob, key_version)
        crypto.set_identity(user_id)
        crypto.set_cache_keys(True)

        # Seed the conversation key from the backlog (KeyChange events).
        page = await api.get_events(chat_id, max_results=50)
        events_b64 = [e["encoded_event"] for e in (page.get("data") or []) if e.get("encoded_event")]
        canonical = chat_id
        if events_b64:
            try:
                batch = crypto.decrypt_batch(events_b64)
                for m in batch.get("messages") or []:
                    conv = (m.get("event") or {}).get("conversation_id")
                    if conv:
                        canonical = str(conv)
                        break
            except Exception as e:
                logger.debug("[xchat] standalone backlog decrypt: %s", e)

        try:
            body = crypto.encrypt_text(canonical, message)
        except ValueError:
            return {
                "error": (
                    "xchat: no verified conversation key — the target must have "
                    "an existing conversation with the bot"
                )
            }
        out = await api.send_message(canonical, body)
        data = out.get("data") or {}
        return {
            "success": True,
            "platform": "xchat",
            "chat_id": canonical,
            "message_id": str(data.get("message_id") or body.get("message_id") or ""),
        }
    except XChatApiError as e:
        return {"error": f"xchat standalone send failed: {e}"}
    except Exception as e:
        return {"error": f"xchat standalone send failed: {e}"}
    finally:
        await api.aclose()


def register(ctx) -> None:
    """Plugin entry point — called by the Hermes plugin loader at startup."""
    from . import cli as _cli

    ctx.register_platform(
        name="xchat",
        label="X Chat (encrypted DMs)",
        adapter_factory=lambda cfg: XChatAdapter(cfg),
        check_fn=check_requirements,
        validate_config=validate_config,
        is_connected=is_connected,
        required_env=["XCHAT_ACCESS_TOKEN"],
        install_hint=(
            "Run: hermes xchat setup  (stores the OAuth2 user token, registers "
            "the bot's E2EE keys, saves the private-key blob)."
        ),
        setup_fn=_cli.gateway_setup,
        env_enablement_fn=_env_enablement,
        cron_deliver_env_var="XCHAT_HOME_CHANNEL",
        standalone_sender_fn=_standalone_send,
        allowed_users_env="XCHAT_ALLOWED_USERS",
        allow_all_env="XCHAT_ALLOW_ALL_USERS",
        max_message_length=MAX_MESSAGE_LENGTH,
        emoji="𝕏",
        pii_safe=False,
        allow_update_command=True,
        platform_hint=(
            "You are communicating via X Chat — X's end-to-end encrypted "
            "direct messages. Treat replies like regular chat messages: "
            "short and conversational. Markdown is NOT rendered — use plain "
            "text. User identifiers are numeric X user ids; conversation ids "
            "starting with 'g' are group chats."
        ),
    )

    ctx.register_cli_command(
        name="xchat",
        help="Set up and manage the X Chat (encrypted X DMs) integration",
        setup_fn=_cli.register_cli,
        handler_fn=_cli.dispatch,
    )

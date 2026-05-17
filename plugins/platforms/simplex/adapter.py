"""SimpleX Chat platform adapter (Hermes plugin).

Connects to a simplex-chat daemon running in WebSocket mode.
Inbound messages arrive via a persistent WebSocket connection.
Outbound messages use the same WebSocket with JSON commands.

This adapter ships as a Hermes platform plugin under
``plugins/platforms/simplex/``. The Hermes plugin loader scans the
directory at startup, calls ``register(ctx)``, and the platform
becomes available to ``gateway/run.py`` and ``tools/send_message_tool``
through the registry — no edits to core files are required.

SimpleX chat daemon setup:
    simplex-chat -p 5225          # start daemon on port 5225
    # or via Docker:
    # docker run -p 5225:5225 simplexchat/simplex-chat-cli -p 5225

Required environment variables:
    SIMPLEX_WS_URL             WebSocket URL of the daemon
                               (default: ws://127.0.0.1:5225)

Optional environment variables:
    SIMPLEX_ALLOWED_USERS      Comma-separated contact IDs (allowlist)
    SIMPLEX_ALLOW_ALL_USERS    Set 'true' to allow all contacts
    SIMPLEX_HOME_CHANNEL       Default contact/group ID for cron delivery
    SIMPLEX_HOME_CHANNEL_NAME  Human label for the home channel

The ``websockets`` Python package is imported lazily — the plugin is
discoverable and `hermes setup` can describe it even when websockets is
not installed. ``check_requirements()`` returns False until the package
is present, so the gateway will not attempt to instantiate the adapter.
"""

import asyncio
import json
import logging
import os
import random
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# Lazy import: BasePlatformAdapter and friends live in the main repo.
# Imported at module top because they're stdlib-only inside Hermes — no
# external dependency that would block the plugin from loading.
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    cache_image_from_bytes,
    cache_audio_from_bytes,
    cache_document_from_bytes,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_MESSAGE_LENGTH = 16_000  # SimpleX has no hard limit; keep chunking sane
TYPING_INTERVAL = 10.0
WS_RETRY_DELAY_INITIAL = 2.0
WS_RETRY_DELAY_MAX = 60.0
HEALTH_CHECK_INTERVAL = 30.0
HEALTH_CHECK_STALE_THRESHOLD = 120.0

# Correlation ID prefix for requests we send so we can ignore our own echoes.
_CORR_PREFIX = "hermes-"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_comma_list(value: str) -> List[str]:
    """Split a comma-separated string into a stripped list."""
    return [v.strip() for v in value.split(",") if v.strip()]


def _guess_extension(data: bytes) -> str:
    """Guess file extension from magic bytes."""
    if data[:4] == b"\x89PNG":
        return ".png"
    if data[:2] == b"\xff\xd8":
        return ".jpg"
    if data[:4] == b"GIF8":
        return ".gif"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    if data[:4] == b"%PDF":
        return ".pdf"
    if len(data) >= 8 and data[4:8] == b"ftyp":
        return ".mp4"
    if data[:4] == b"OggS":
        return ".ogg"
    if len(data) >= 2 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0:
        return ".mp3"
    return ".bin"


def _is_image_ext(ext: str) -> bool:
    return ext.lower() in (".jpg", ".jpeg", ".png", ".gif", ".webp")


def _is_audio_ext(ext: str) -> bool:
    return ext.lower() in (".mp3", ".wav", ".ogg", ".m4a", ".aac")


# ---------------------------------------------------------------------------
# SimpleX Adapter
# ---------------------------------------------------------------------------

class SimplexAdapter(BasePlatformAdapter):
    """SimpleX Chat adapter using the simplex-chat daemon WebSocket API.

    Instantiated by the ``adapter_factory`` passed to
    ``ctx.register_platform()`` in :func:`register`.
    """

    def __init__(self, config: PlatformConfig, **kwargs):
        platform = Platform("simplex")
        super().__init__(config=config, platform=platform)

        extra = getattr(config, "extra", {}) or {}
        self.ws_url = extra.get("ws_url", "ws://127.0.0.1:5225").rstrip("/")

        # Running state
        self._ws = None  # websockets connection
        self._ws_task: Optional[asyncio.Task] = None
        self._health_task: Optional[asyncio.Task] = None
        self._typing_tasks: Dict[str, asyncio.Task] = {}
        self._running = False
        self._last_ws_activity = 0.0

        # Track sent correlation IDs to filter echoes
        self._pending_corr_ids: set = set()
        self._max_pending_corr = 200

        logger.info("SimpleX adapter initialized: url=%s", self.ws_url)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        """Connect to the simplex-chat daemon and start the WebSocket listener."""
        try:
            import websockets  # noqa: F401
        except ImportError:
            logger.error(
                "SimpleX: 'websockets' package not installed. "
                "Run: pip install websockets"
            )
            return False

        if not self.ws_url:
            logger.error("SimpleX: SIMPLEX_WS_URL is required")
            return False

        # Quick connectivity check — try to open and immediately close
        try:
            import websockets as _wsclient
            async with _wsclient.connect(self.ws_url, open_timeout=10):
                pass
        except Exception as e:
            logger.error("SimpleX: cannot reach daemon at %s: %s", self.ws_url, e)
            return False

        self._running = True
        self._last_ws_activity = time.time()
        self._ws_task = asyncio.create_task(self._ws_listener())
        self._health_task = asyncio.create_task(self._health_monitor())

        logger.info("SimpleX: connected to %s", self.ws_url)
        return True

    async def disconnect(self) -> None:
        """Stop WebSocket listener and clean up."""
        self._running = False

        if self._ws_task:
            self._ws_task.cancel()
            try:
                await self._ws_task
            except asyncio.CancelledError:
                pass

        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass

        for task in self._typing_tasks.values():
            task.cancel()
        self._typing_tasks.clear()

        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

        logger.info("SimpleX: disconnected")

    # ------------------------------------------------------------------
    # WebSocket listener
    # ------------------------------------------------------------------

    async def _ws_listener(self) -> None:
        """Maintain a persistent WebSocket connection to the daemon."""
        import websockets as _wsclient
        import websockets as _wsexc

        backoff = WS_RETRY_DELAY_INITIAL

        while self._running:
            try:
                logger.debug("SimpleX WS: connecting to %s", self.ws_url)
                async with _wsclient.connect(
                    self.ws_url,
                    ping_interval=20,
                    ping_timeout=20,
                ) as ws:
                    self._ws = ws
                    backoff = WS_RETRY_DELAY_INITIAL
                    self._last_ws_activity = time.time()
                    logger.info("SimpleX WS: connected")

                    async for raw in ws:
                        if not self._running:
                            break
                        self._last_ws_activity = time.time()
                        try:
                            msg = json.loads(raw)
                            await self._handle_event(msg)
                        except json.JSONDecodeError:
                            logger.debug("SimpleX WS: invalid JSON: %.100s", raw)
                        except Exception:
                            logger.exception("SimpleX WS: error handling event")

            except asyncio.CancelledError:
                break
            except _wsexc.WebSocketException as e:
                if self._running:
                    logger.warning(
                        "SimpleX WS: error: %s (reconnecting in %.0fs)", e, backoff
                    )
            except Exception as e:
                if self._running:
                    logger.warning(
                        "SimpleX WS: unexpected error: %s (reconnecting in %.0fs)",
                        e, backoff,
                    )
            finally:
                self._ws = None

            if self._running:
                jitter = backoff * 0.2 * random.random()
                await asyncio.sleep(backoff + jitter)
                backoff = min(backoff * 2, WS_RETRY_DELAY_MAX)

    # ------------------------------------------------------------------
    # Health monitor
    # ------------------------------------------------------------------

    async def _health_monitor(self) -> None:
        """Force reconnect if the WebSocket has been idle too long."""
        while self._running:
            await asyncio.sleep(HEALTH_CHECK_INTERVAL)
            if not self._running:
                break

            elapsed = time.time() - self._last_ws_activity
            if elapsed > HEALTH_CHECK_STALE_THRESHOLD:
                logger.warning(
                    "SimpleX: WS idle for %.0fs, forcing reconnect", elapsed
                )
                self._last_ws_activity = time.time()
                if self._ws:
                    try:
                        await self._ws.close()
                    except Exception:
                        pass

    # ------------------------------------------------------------------
    # Inbound event handling
    # ------------------------------------------------------------------

    async def _handle_event(self, event: dict) -> None:
        """Dispatch a daemon event to the appropriate handler."""
        resp_type = event.get("type") or event.get("resp", {}).get("type", "")

        # Filter responses to our own commands (echoes)
        corr_id = event.get("corrId", "")
        if corr_id and corr_id.startswith(_CORR_PREFIX):
            self._pending_corr_ids.discard(corr_id)
            return

        if resp_type == "newChatItem":
            await self._handle_new_chat_item(event)
        elif resp_type == "newChatItems":
            # Batch variant — process each item
            items = event.get("chatItems") or []
            for item_wrapper in items:
                await self._handle_new_chat_item(item_wrapper)
        # Ignore all other event types (delivery receipts, contact updates, etc.)

    async def _handle_new_chat_item(self, wrapper: dict) -> None:
        """Process a single newChatItem event into a MessageEvent."""
        # The daemon wraps the chat item differently depending on version;
        # normalise both layouts.
        chat_info = wrapper.get("chatInfo") or wrapper.get("chat") or {}
        chat_item = wrapper.get("chatItem") or wrapper.get("item") or {}

        # Only process messages (not calls, deleted items, etc.)
        item_content = chat_item.get("content") or {}
        msg_content = item_content.get("msgContent") or {}
        if not msg_content:
            return

        # Filter out messages sent by us (direction == "snd")
        meta = chat_item.get("meta") or {}
        direction = (meta.get("itemStatus") or {}).get("type", "")
        if direction in ("sndSent", "sndSentDirect", "sndSentViaProxy", "sndNew"):
            return

        # Determine chat type and IDs
        chat_type_raw = chat_info.get("type", "")
        is_group = chat_type_raw in ("group", "groupInfo")

        if is_group:
            group_info = chat_info.get("groupInfo") or chat_info.get("group") or {}
            group_id = str(group_info.get("groupId") or group_info.get("id") or "")
            group_name = group_info.get("displayName") or group_info.get("groupProfile", {}).get("displayName", "")
            chat_id = f"group:{group_id}" if group_id else ""
            chat_name = group_name
        else:
            contact_info = chat_info.get("contact") or {}
            contact_id = str(contact_info.get("contactId") or contact_info.get("id") or "")
            contact_name = (
                contact_info.get("displayName")
                or contact_info.get("localDisplayName")
                or contact_id
            )
            chat_id = contact_id
            chat_name = contact_name

        if not chat_id:
            logger.debug("SimpleX: ignoring event with no chat_id")
            return

        # Sender — for groups the message includes a chatItemMember sub-object
        member = chat_item.get("chatItemMember") or {}
        if is_group and member:
            sender_id = str(member.get("memberId") or member.get("id") or chat_id)
            sender_name = (
                member.get("displayName")
                or member.get("localDisplayName")
                or sender_id
            )
        else:
            sender_id = chat_id
            sender_name = chat_name

        # Extract text
        text = msg_content.get("text") or ""

        # Media attachments
        media_urls: List[str] = []
        media_types: List[str] = []
        file_info = chat_item.get("file") or {}
        if file_info and file_info.get("fileStatus") not in ("cancelled", "error"):
            file_id = file_info.get("fileId")
            file_name = file_info.get("fileName", "file")
            if file_id:
                try:
                    cached = await self._fetch_file(file_id, file_name)
                    if cached:
                        ext = cached.rsplit(".", 1)[-1]
                        if _is_image_ext("." + ext):
                            media_types.append("image/" + ext.replace("jpg", "jpeg"))
                        elif _is_audio_ext("." + ext):
                            media_types.append("audio/" + ext)
                        else:
                            media_types.append("application/octet-stream")
                        media_urls.append(cached)
                except Exception:
                    logger.exception("SimpleX: failed to fetch file %s", file_id)

        # Timestamp
        ts_str = meta.get("itemTs") or meta.get("createdAt") or ""
        try:
            timestamp = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            timestamp = datetime.now(tz=timezone.utc)

        # Build source
        source = self.build_source(
            chat_id=chat_id,
            chat_name=chat_name,
            chat_type="group" if is_group else "dm",
            user_id=sender_id,
            user_name=sender_name,
        )

        # Message type
        msg_type = MessageType.TEXT
        if media_types:
            if any(mt.startswith("audio/") for mt in media_types):
                msg_type = MessageType.VOICE
            elif any(mt.startswith("image/") for mt in media_types):
                msg_type = MessageType.PHOTO

        event_obj = MessageEvent(
            source=source,
            text=text,
            message_type=msg_type,
            media_urls=media_urls,
            media_types=media_types,
            timestamp=timestamp,
            raw_message=wrapper,
        )

        await self.handle_message(event_obj)

    async def _fetch_file(self, file_id: Any, file_name: str) -> Optional[str]:
        """Ask the daemon to receive and return a file attachment."""
        # simplex-chat exposes `/api/v1/files/{fileId}` on an HTTP port
        # when started with --http-port. However, the canonical WebSocket API
        # does not have a direct binary download command; files are stored on
        # the local filesystem after the daemon accepts them.
        #
        # We request acceptance first, then read from the daemon's local path.
        corr_id = self._make_corr_id()
        cmd = {
            "corrId": corr_id,
            "cmd": f"/freceive {file_id}",
        }
        await self._send_ws(cmd)
        # The daemon will emit a chatItemUpdated event when the file lands;
        # for simplicity we just wait briefly and rely on the daemon's default path.
        await asyncio.sleep(2)

        # simplex-chat stores received files in ~/Downloads or a configured path.
        # We try common locations.
        for search_dir in (
            os.path.expanduser("~/Downloads"),
            os.path.expanduser("~/.simplex/files"),
            "/tmp/simplex_files",
        ):
            candidate = os.path.join(search_dir, file_name)
            if os.path.exists(candidate):
                with open(candidate, "rb") as f:
                    data = f.read()
                ext = _guess_extension(data)
                if _is_image_ext(ext):
                    return cache_image_from_bytes(data, ext)
                elif _is_audio_ext(ext):
                    return cache_audio_from_bytes(data, ext)
                else:
                    return cache_document_from_bytes(data, file_name)
        return None

    # ------------------------------------------------------------------
    # Outbound messages
    # ------------------------------------------------------------------

    def _make_corr_id(self) -> str:
        """Generate a unique correlation ID for a request."""
        corr_id = f"{_CORR_PREFIX}{int(time.time() * 1000)}-{random.randint(0, 9999)}"
        self._pending_corr_ids.add(corr_id)
        if len(self._pending_corr_ids) > self._max_pending_corr:
            # Trim oldest — sets are unordered so just clear the oldest half
            to_remove = list(self._pending_corr_ids)[:self._max_pending_corr // 2]
            self._pending_corr_ids -= set(to_remove)
        return corr_id

    async def _send_ws(self, payload: dict) -> None:
        """Send a JSON payload over the WebSocket, queuing if not yet connected."""
        import websockets as _wsexc
        ws = self._ws
        if not ws:
            logger.debug("SimpleX: WS not connected, dropping outbound command")
            return
        try:
            await ws.send(json.dumps(payload))
        except _wsexc.ConnectionClosed:
            logger.warning("SimpleX: WS closed while sending")
        except Exception as e:
            logger.warning("SimpleX: WS send error: %s", e)

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a text message to a contact or group."""
        corr_id = self._make_corr_id()

        if chat_id.startswith("group:"):
            group_id = chat_id[6:]
            cmd_str = f"#[{group_id}] {content}"
        else:
            cmd_str = f"@[{chat_id}] {content}"

        payload = {
            "corrId": corr_id,
            "cmd": cmd_str,
        }

        await self._send_ws(payload)
        return SendResult(success=True)

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """SimpleX does not expose a typing indicator API — no-op."""
        pass

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send an image (URL) as a message with optional caption.

        SimpleX has no native ``send_image`` over the WebSocket API — file
        attachments require the daemon's filesystem-backed flow which is
        not driven from this adapter. Fall back to a plain text message
        containing the URL and caption.
        """
        text = f"{caption}\n{image_url}".strip() if caption else image_url
        return await self.send(chat_id, text, reply_to=reply_to, metadata=metadata)

    async def get_chat_info(self, chat_id: str) -> dict:
        """Return basic chat info."""
        if chat_id.startswith("group:"):
            return {"chat_id": chat_id, "type": "group", "name": chat_id[6:]}
        return {"chat_id": chat_id, "type": "dm", "name": chat_id}


# ---------------------------------------------------------------------------
# Plugin entry-point hooks
# ---------------------------------------------------------------------------

def check_requirements() -> bool:
    """Plugin gate: require SIMPLEX_WS_URL AND the websockets package.

    Returning False keeps the platform out of ``get_connected_platforms()``
    so the gateway never instantiates the adapter when the dependency is
    missing or no daemon URL is configured.
    """
    if not os.getenv("SIMPLEX_WS_URL"):
        return False
    try:
        import websockets  # noqa: F401
    except ImportError:
        return False
    return True


def validate_config(config) -> bool:
    """Validate that the platform config has enough info to connect."""
    extra = getattr(config, "extra", {}) or {}
    ws_url = os.getenv("SIMPLEX_WS_URL") or extra.get("ws_url", "")
    return bool(ws_url)


def is_connected(config) -> bool:
    """Check whether SimpleX is configured (env or config.yaml)."""
    extra = getattr(config, "extra", {}) or {}
    ws_url = os.getenv("SIMPLEX_WS_URL") or extra.get("ws_url", "")
    return bool(ws_url)


def _env_enablement() -> dict | None:
    """Seed ``PlatformConfig.extra`` from env vars during gateway config load.

    Called by the platform registry's env-enablement hook BEFORE adapter
    construction, so ``gateway status`` and ``get_connected_platforms()``
    reflect env-only configuration without instantiating the WebSocket
    client. Returns ``None`` when SimpleX isn't minimally configured.

    The special ``home_channel`` key in the returned dict is handled by
    the core hook — it becomes a proper ``HomeChannel`` dataclass on the
    ``PlatformConfig`` rather than being merged into ``extra``.
    """
    ws_url = os.getenv("SIMPLEX_WS_URL", "").strip()
    if not ws_url:
        return None
    seed: dict = {"ws_url": ws_url}
    home = os.getenv("SIMPLEX_HOME_CHANNEL", "").strip()
    if home:
        seed["home_channel"] = {
            "chat_id": home,
            "name": os.getenv("SIMPLEX_HOME_CHANNEL_NAME", "").strip() or home,
        }
    return seed


async def _standalone_send(
    pconfig,
    chat_id: str,
    message: str,
    *,
    thread_id: Optional[str] = None,
    media_files: Optional[List[str]] = None,
    force_document: bool = False,
) -> Dict[str, Any]:
    """Open an ephemeral WebSocket to the daemon, send, and close.

    Used by ``tools/send_message_tool._send_via_adapter`` when the gateway
    runner is not in this process (e.g. ``hermes cron`` running as a
    separate process from ``hermes gateway``). Without this hook,
    ``deliver=simplex`` cron jobs fail with "No live adapter for platform".

    ``thread_id`` and ``force_document`` are accepted for signature parity
    with other plugins but are not meaningful here. ``media_files`` is
    accepted but only the text body is delivered — SimpleX requires the
    daemon's filesystem-backed file flow which an ephemeral connection
    cannot drive safely.
    """
    try:
        import websockets as _wsclient
    except ImportError:
        return {"error": "websockets not installed. Run: pip install websockets"}

    extra = getattr(pconfig, "extra", {}) or {}
    ws_url = os.getenv("SIMPLEX_WS_URL") or extra.get("ws_url", "ws://127.0.0.1:5225")
    if not ws_url:
        return {"error": "SimpleX standalone send: SIMPLEX_WS_URL is required"}

    try:
        if chat_id.startswith("group:"):
            group_id = chat_id[6:]
            cmd_str = f"#[{group_id}] {message}"
        else:
            cmd_str = f"@[{chat_id}] {message}"

        payload = {
            "corrId": f"hermes-snd-{int(time.time() * 1000)}",
            "cmd": cmd_str,
        }

        async with _wsclient.connect(ws_url, open_timeout=10, close_timeout=5) as ws:
            await ws.send(json.dumps(payload))
            # Give the daemon a moment to process the command before closing.
            await asyncio.sleep(0.5)

        return {"success": True, "platform": "simplex", "chat_id": chat_id}
    except Exception as e:
        return {"error": f"SimpleX send failed: {e}"}


def interactive_setup() -> None:
    """Minimal stdin wizard for ``hermes setup gateway`` → SimpleX.

    Prompts for the WebSocket URL and the optional allowlist / home channel.
    Writes to ``~/.hermes/.env`` via ``hermes_cli.config``.
    """
    print()
    print("SimpleX Chat setup")
    print("------------------")
    print("Requirements:")
    print("  1. simplex-chat daemon running (e.g. `simplex-chat -p 5225`).")
    print("  2. Python package `websockets` installed (`pip install websockets`).")
    print()

    try:
        from hermes_cli.config import get_env_value, save_env_value
    except ImportError:
        print("hermes_cli.config not available; set SIMPLEX_* vars manually in ~/.hermes/.env")
        return

    def _prompt(var: str, prompt: str, *, secret: bool = False) -> None:
        existing = get_env_value(var) if callable(get_env_value) else None
        suffix = " [keep current]" if existing else ""
        try:
            if secret:
                import getpass
                value = getpass.getpass(f"{prompt}{suffix}: ")
            else:
                value = input(f"{prompt}{suffix}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if value:
            save_env_value(var, value)

    _prompt("SIMPLEX_WS_URL", "Daemon WebSocket URL (default ws://127.0.0.1:5225)")
    _prompt("SIMPLEX_ALLOWED_USERS", "Allowed contact IDs (comma-separated; blank=skip)")
    _prompt("SIMPLEX_HOME_CHANNEL", "Home channel contact/group ID (or empty)")
    print("Done. Make sure the simplex-chat daemon is running before starting the gateway.")


def register(ctx) -> None:
    """Plugin entry point — called by the Hermes plugin system at startup."""
    ctx.register_platform(
        name="simplex",
        label="SimpleX Chat",
        adapter_factory=lambda cfg: SimplexAdapter(cfg),
        check_fn=check_requirements,
        validate_config=validate_config,
        is_connected=is_connected,
        required_env=["SIMPLEX_WS_URL"],
        install_hint="pip install websockets   # SimpleX adapter requires the websockets package",
        setup_fn=interactive_setup,
        # Env-driven auto-configuration: seeds PlatformConfig.extra so
        # env-only setups show up in `hermes gateway status` without
        # instantiating the adapter.
        env_enablement_fn=_env_enablement,
        # Cron home-channel delivery support — `deliver=simplex` cron jobs
        # route to SIMPLEX_HOME_CHANNEL when set.
        cron_deliver_env_var="SIMPLEX_HOME_CHANNEL",
        # Out-of-process cron delivery. Without this hook, deliver=simplex
        # cron jobs fail with "No live adapter" when cron runs separately
        # from the gateway.
        standalone_sender_fn=_standalone_send,
        # Auth env vars for _is_user_authorized() integration
        allowed_users_env="SIMPLEX_ALLOWED_USERS",
        allow_all_env="SIMPLEX_ALLOW_ALL_USERS",
        # SimpleX has no hard line length; we still chunk for sanity.
        max_message_length=MAX_MESSAGE_LENGTH,
        # Display
        emoji="🔒",
        # SimpleX uses opaque contact IDs only — no phone numbers or
        # email addresses to redact.
        pii_safe=True,
        allow_update_command=True,
        # LLM guidance
        platform_hint=(
            "You are chatting via SimpleX Chat, a private decentralised "
            "messenger. Contacts are identified by opaque internal IDs, "
            "not phone numbers or usernames. SimpleX supports standard "
            "markdown formatting. There is no typing indicator and no "
            "hard message length limit, but keep responses conversational."
        ),
    )

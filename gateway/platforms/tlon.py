"""
Tlon (Urbit) platform adapter for Hermes Gateway.

Connects to a Tlon ship via Eyre HTTP API:
- Authenticates with ship +code
- Subscribes to channel messages (channels /v2) and DMs (chat /v3) via SSE
- Sends messages back via pokes

Requires: aiohttp (pip install aiohttp)

Environment variables:
  TLON_SHIP_URL    - Ship URL (e.g. https://sampel-palnet.tlon.network)
  TLON_SHIP_NAME   - Ship name (e.g. ~sampel-palnet)
  TLON_SHIP_CODE   - Ship +code for authentication
  TLON_CHANNELS    - Comma-separated channel nests to monitor (e.g. chat/~host/channel)
  TLON_DM_ALLOWLIST - Comma-separated ships allowed to DM (empty = all allowed)
  TLON_HOME_CHANNEL - Default channel for cron delivery
  TLON_ALLOWED_USERS - Comma-separated ships allowed to interact
  TLON_ALLOW_ALL_USERS - Set to "true" to allow all users (default: false)
  TLON_AUTO_DISCOVER - Set to "true" to auto-discover all group channels
"""

import asyncio
import json
import logging
import os
import re
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    cache_image_from_bytes,
)

# Maximum message length for Tlon (generous - Tlon handles long messages well)
MAX_MESSAGE_LENGTH = 10000


def check_tlon_requirements() -> bool:
    """Check if aiohttp is available for HTTP/SSE communication."""
    try:
        import aiohttp
        return True
    except ImportError:
        logger.warning("Tlon adapter requires aiohttp. Install with: pip install aiohttp")
        return False


def _normalize_ship(ship: str) -> str:
    """Normalize a ship name to include ~ prefix."""
    ship = ship.strip()
    if ship and not ship.startswith("~"):
        ship = "~" + ship
    return ship


def _parse_channel_nest(nest: str) -> Optional[Dict[str, str]]:
    """Parse a channel nest like 'chat/~host/channel-name'."""
    parts = nest.split("/", 2)
    if len(parts) != 3:
        return None
    return {
        "type": parts[0],       # chat, heap, diary
        "host": parts[1],       # ~host-ship
        "name": parts[2],       # channel-name
    }


def _extract_message_text(content: Any) -> str:
    """
    Extract plain text from Tlon's story/content format.

    Tlon messages use a 'story' format: an array of blocks.
    Each block is either:
      - {"inline": [...]} with text strings, links, mentions, etc.
      - {"block": {"image": {...}}} for images
      - {"block": {"cite": {...}}} for quotes
    """
    if not content:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                # Inline block: {"inline": [...]}
                if "inline" in block:
                    for inline in block["inline"]:
                        if isinstance(inline, str):
                            parts.append(inline)
                        elif isinstance(inline, dict):
                            # Bold, italic, etc.
                            if "bold" in inline:
                                parts.append(_extract_inline_text(inline["bold"]))
                            elif "italics" in inline:
                                parts.append(_extract_inline_text(inline["italics"]))
                            elif "strike" in inline:
                                parts.append(_extract_inline_text(inline["strike"]))
                            elif "blockquote" in inline:
                                parts.append(_extract_inline_text(inline["blockquote"]))
                            elif "inline-code" in inline:
                                parts.append(inline["inline-code"])
                            elif "code" in inline:
                                parts.append(inline["code"])
                            elif "link" in inline:
                                link = inline["link"]
                                href = link.get("href", "")
                                link_content = link.get("content", href)
                                parts.append(link_content if link_content else href)
                            elif "ship" in inline:
                                parts.append(_normalize_ship(inline["ship"]))
                            elif "break" in inline:
                                parts.append("\n")
                            elif "tag" in inline:
                                parts.append(f"#{inline['tag']}")
                # Block types
                elif "block" in block:
                    b = block["block"]
                    if isinstance(b, dict):
                        if "image" in b:
                            img = b["image"]
                            alt = img.get("alt", "")
                            src = img.get("src", "")
                            parts.append(f"[image: {alt or src}]")
                        elif "cite" in b:
                            parts.append("[quoted message]")
                        elif "code" in b:
                            code = b["code"]
                            lang = code.get("lang", "")
                            body = code.get("code", "")
                            parts.append(f"```{lang}\n{body}\n```")
        return " ".join(p for p in parts if p).strip()

    return str(content)


def _extract_inline_text(inlines: Any) -> str:
    """Recursively extract text from inline content."""
    if isinstance(inlines, str):
        return inlines
    if isinstance(inlines, list):
        parts = []
        for item in inlines:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if "ship" in item:
                    parts.append(_normalize_ship(item["ship"]))
                elif "link" in item:
                    parts.append(item["link"].get("content", item["link"].get("href", "")))
                elif "bold" in item:
                    parts.append(_extract_inline_text(item["bold"]))
                elif "italics" in item:
                    parts.append(_extract_inline_text(item["italics"]))
                elif "inline-code" in item:
                    parts.append(item["inline-code"])
                elif "break" in item:
                    parts.append("\n")
        return "".join(parts)
    return ""


def _text_to_story(text: str) -> list:
    """
    Convert plain text/markdown to Tlon's story format.

    Returns a list of story blocks suitable for use as post content.
    """
    # Simple conversion: wrap text as inline blocks, split on double newlines
    # for paragraph breaks
    blocks = []
    paragraphs = text.split("\n\n")
    for i, para in enumerate(paragraphs):
        if not para.strip():
            continue
        # Convert each paragraph to an inline block
        inlines = []
        # Handle @mentions (ship names)
        parts = re.split(r'(~[a-z][\w-]*)', para)
        for part in parts:
            if re.match(r'^~[a-z][\w-]*$', part):
                inlines.append({"ship": part.lstrip("~")})
            elif part:
                inlines.append(part)
        if inlines:
            blocks.append({"inline": inlines})
        # Add line break between paragraphs (except after last)
        if i < len(paragraphs) - 1:
            blocks.append({"inline": [{"break": None}]})
    if not blocks:
        blocks = [{"inline": [""]}]
    return blocks


def _da_from_unix(unix_ms: int) -> str:
    """
    Convert Unix timestamp (ms) to Urbit @da format (dot-separated @ud).

    This produces an @ud-formatted bigint suitable for post IDs.
    The Urbit epoch is ~292 billion years before Unix epoch.
    """
    # Urbit @da epoch offset from Unix epoch (in milliseconds)
    # ~2000.1.1 in Unix = 946684800000 ms
    # But we need the actual @da conversion...
    # For message IDs, Tlon uses the timestamp as a bigint @ud
    # We just need a unique, monotonically increasing number
    return str(unix_ms)


class TlonSSEClient:
    """
    Manages an Eyre SSE channel for subscribing to Tlon events.

    Handles:
    - Authentication via POST /~/login
    - Channel creation via PUT /~/channel/{id}
    - SSE event streaming via GET /~/channel/{id}
    - Event acknowledgement
    - Reconnection with exponential backoff
    """

    def __init__(
        self,
        url: str,
        code: str,
        ship: str,
        *,
        auto_reconnect: bool = True,
        max_reconnect_attempts: int = 10,
        reconnect_delay: float = 1.0,
        max_reconnect_delay: float = 30.0,
    ):
        self.url = url.rstrip("/")
        self.code = code
        self.ship = _normalize_ship(ship)
        self.auto_reconnect = auto_reconnect
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_delay = max_reconnect_delay

        self.cookie: Optional[str] = None
        self.channel_id: Optional[str] = None
        self.channel_url: Optional[str] = None
        self._session: Optional[Any] = None  # aiohttp.ClientSession
        self._sse_task: Optional[asyncio.Task] = None
        self._aborted = False
        self._connected = False
        self._reconnect_attempts = 0
        self._action_counter = 0

        # Subscription tracking
        self._subscriptions: List[Dict[str, Any]] = []
        self._event_handlers: Dict[int, Dict[str, Any]] = {}

        # Event ack tracking
        self._last_heard_event_id = -1
        self._last_acked_event_id = -1
        self._ack_threshold = 20

    async def authenticate(self) -> str:
        """Authenticate with the ship and return the cookie."""
        import aiohttp

        if not self._session:
            self._session = aiohttp.ClientSession()

        async with self._session.post(
            f"{self.url}/~/login",
            data={"password": self.code},
            allow_redirects=False,
            timeout=aiohttp.ClientTimeout(total=15),
        ) as resp:
            if resp.status not in (200, 204, 302, 303, 307):
                raise ConnectionError(f"Auth failed: HTTP {resp.status}")
            cookie = resp.headers.get("set-cookie", "")
            if not cookie:
                # Try from cookies jar
                for c in self._session.cookie_jar:
                    if c.key.startswith("urbauth"):
                        cookie = f"{c.key}={c.value}"
                        break
            if not cookie:
                raise ConnectionError("No auth cookie received")
            self.cookie = cookie
            logger.info("[tlon] Authenticated as %s", self.ship)
            return cookie

    async def _new_channel_id(self) -> str:
        """Generate a new unique channel ID."""
        ts = int(time.time())
        uid = uuid.uuid4().hex[:8]
        return f"{ts}-{uid}"

    def _next_action_id(self) -> int:
        """Get the next action ID for channel operations."""
        self._action_counter += 1
        return self._action_counter

    async def subscribe(
        self,
        app: str,
        path: str,
        on_event: Optional[Any] = None,
        on_error: Optional[Any] = None,
        on_quit: Optional[Any] = None,
    ) -> int:
        """
        Subscribe to a Gall agent path.

        Returns the subscription ID.
        """
        sub_id = self._next_action_id()
        sub = {
            "id": sub_id,
            "action": "subscribe",
            "ship": self.ship.lstrip("~"),
            "app": app,
            "path": path,
        }
        self._subscriptions.append(sub)
        self._event_handlers[sub_id] = {
            "event": on_event,
            "err": on_error,
            "quit": on_quit,
        }

        # If already connected, send subscription immediately
        if self._connected:
            await self._send_actions([sub])

        return sub_id

    async def _send_actions(self, actions: List[Dict[str, Any]]) -> None:
        """Send actions to the Eyre channel."""
        import aiohttp

        headers = {"Content-Type": "application/json"}
        if self.cookie:
            headers["Cookie"] = self.cookie

        async with self._session.put(
            self.channel_url,
            json=actions,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            if resp.status not in (200, 204):
                text = await resp.text()
                raise ConnectionError(
                    f"Channel action failed: HTTP {resp.status} - {text[:200]}"
                )

    async def connect(self) -> None:
        """
        Create the Eyre channel with initial subscriptions and start
        the SSE event stream.
        """
        self.channel_id = await self._new_channel_id()
        self.channel_url = f"{self.url}/~/channel/{self.channel_id}"

        # Create channel with all pending subscriptions
        if self._subscriptions:
            await self._send_actions(self._subscriptions)

        # Start SSE stream
        await self._open_stream()
        self._connected = True
        self._reconnect_attempts = 0
        logger.info("[tlon] SSE connected on channel %s", self.channel_id)

    async def _open_stream(self) -> None:
        """Open the SSE GET stream."""
        import aiohttp

        headers = {
            "Accept": "text/event-stream",
        }
        if self.cookie:
            headers["Cookie"] = self.cookie

        self._sse_task = asyncio.create_task(self._stream_loop(headers))

    async def _stream_loop(self, headers: Dict[str, str]) -> None:
        """Read the SSE stream and dispatch events."""
        import aiohttp

        try:
            async with self._session.get(
                self.channel_url,
                headers=headers,
                timeout=aiohttp.ClientTimeout(
                    total=None,  # No total timeout for SSE
                    sock_read=None,  # No read timeout
                    connect=60,
                ),
            ) as resp:
                if resp.status != 200:
                    raise ConnectionError(f"SSE stream failed: HTTP {resp.status}")

                buffer = ""
                async for chunk in resp.content.iter_any():
                    if self._aborted:
                        break
                    buffer += chunk.decode("utf-8", errors="replace")

                    while "\n\n" in buffer:
                        event_data, buffer = buffer.split("\n\n", 1)
                        await self._process_event(event_data)

        except asyncio.CancelledError:
            return
        except Exception as e:
            if not self._aborted:
                logger.error("[tlon] SSE stream error: %s", e)
                self._connected = False
                if self.auto_reconnect:
                    await self._attempt_reconnect()

    async def _process_event(self, event_data: str) -> None:
        """Parse and dispatch a single SSE event."""
        lines = event_data.split("\n")
        data = None
        event_id = None

        for line in lines:
            if line.startswith("id: "):
                try:
                    event_id = int(line[4:])
                except ValueError:
                    pass
            elif line.startswith("data: "):
                data = line[6:]

        if not data:
            return

        # Track and ack events
        if event_id is not None and event_id > self._last_heard_event_id:
            self._last_heard_event_id = event_id
            if event_id - self._last_acked_event_id > self._ack_threshold:
                asyncio.create_task(self._ack(event_id))

        try:
            parsed = json.loads(data)
        except json.JSONDecodeError:
            logger.debug("[tlon] Non-JSON SSE data: %s", data[:100])
            return

        # Handle quit events (agent kicked us)
        if parsed.get("response") == "quit":
            sub_id = parsed.get("id")
            if sub_id and sub_id in self._event_handlers:
                handler = self._event_handlers[sub_id]
                if handler.get("quit"):
                    handler["quit"]()
                # Auto-resubscribe
                asyncio.create_task(self._resubscribe(sub_id))
            return

        # Dispatch to handlers
        sub_id = parsed.get("id")
        event_json = parsed.get("json")

        if sub_id and sub_id in self._event_handlers:
            handler = self._event_handlers[sub_id]
            if handler.get("event") and event_json is not None:
                try:
                    await handler["event"](event_json)
                except Exception as e:
                    logger.error("[tlon] Event handler error: %s", e)
        elif event_json is not None:
            # Broadcast to all handlers
            for handler in self._event_handlers.values():
                if handler.get("event"):
                    try:
                        await handler["event"](event_json)
                    except Exception as e:
                        logger.error("[tlon] Broadcast handler error: %s", e)

    async def _ack(self, event_id: int) -> None:
        """Acknowledge events up to event_id."""
        self._last_acked_event_id = event_id
        try:
            await self._send_actions([{
                "id": self._next_action_id(),
                "action": "ack",
                "event-id": event_id,
            }])
        except Exception as e:
            logger.debug("[tlon] Ack failed: %s", e)

    async def _resubscribe(self, old_sub_id: int) -> None:
        """Re-subscribe after a quit event."""
        old_sub = None
        for sub in self._subscriptions:
            if sub["id"] == old_sub_id:
                old_sub = sub
                break
        if not old_sub:
            return

        handlers = self._event_handlers.get(old_sub_id)
        if not handlers:
            return

        for attempt in range(5):
            delay = min(2.0 * (2 ** attempt), 30.0)
            logger.info("[tlon] Resubscribing to %s%s in %.0fs...",
                       old_sub["app"], old_sub["path"], delay)
            await asyncio.sleep(delay)

            if self._aborted or not self._connected:
                return

            try:
                new_id = self._next_action_id()
                new_sub = {**old_sub, "id": new_id}
                self._subscriptions.append(new_sub)
                self._event_handlers[new_id] = handlers
                del self._event_handlers[old_sub_id]
                await self._send_actions([new_sub])
                logger.info("[tlon] Resubscribed to %s%s", old_sub["app"], old_sub["path"])
                return
            except Exception as e:
                logger.error("[tlon] Resubscribe failed: %s", e)

    async def _attempt_reconnect(self) -> None:
        """Reconnect with exponential backoff."""
        if self._aborted:
            return

        while self._reconnect_attempts < self.max_reconnect_attempts:
            self._reconnect_attempts += 1
            delay = min(
                self.reconnect_delay * (2 ** (self._reconnect_attempts - 1)),
                self.max_reconnect_delay,
            )
            logger.info("[tlon] Reconnecting in %.1fs (attempt %d/%d)...",
                       delay, self._reconnect_attempts, self.max_reconnect_attempts)
            await asyncio.sleep(delay)

            if self._aborted:
                return

            try:
                # Re-authenticate
                await self.authenticate()
                # New channel
                self.channel_id = await self._new_channel_id()
                self.channel_url = f"{self.url}/~/channel/{self.channel_id}"
                # Reconnect
                await self.connect()
                logger.info("[tlon] Reconnected successfully!")
                return
            except Exception as e:
                logger.error("[tlon] Reconnect failed: %s", e)

        # Reset and keep trying
        logger.warning("[tlon] Max reconnect attempts reached, resetting counter...")
        await asyncio.sleep(10)
        self._reconnect_attempts = 0
        await self._attempt_reconnect()

    async def poke(self, app: str, mark: str, json_data: Any) -> None:
        """Send a poke to a Gall agent."""
        action = {
            "id": self._next_action_id(),
            "action": "poke",
            "ship": self.ship.lstrip("~"),
            "app": app,
            "mark": mark,
            "json": json_data,
        }
        await self._send_actions([action])

    async def scry(self, path: str) -> Any:
        """Scry a path and return the JSON response."""
        import aiohttp

        headers = {}
        if self.cookie:
            headers["Cookie"] = self.cookie

        full_path = path if path.endswith(".json") else f"{path}.json"
        async with self._session.get(
            f"{self.url}{full_path}",
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise Exception(f"Scry failed: HTTP {resp.status} - {text[:200]}")
            return await resp.json()

    async def close(self) -> None:
        """Close the SSE connection and clean up."""
        self._aborted = True
        self._connected = False

        if self._sse_task:
            self._sse_task.cancel()
            try:
                await self._sse_task
            except asyncio.CancelledError:
                pass

        # Try to clean up the Eyre channel
        if self._session and self.channel_url:
            try:
                # Unsubscribe
                unsubs = [
                    {"id": sub["id"], "action": "unsubscribe", "subscription": sub["id"]}
                    for sub in self._subscriptions
                ]
                if unsubs:
                    await self._send_actions(unsubs)
            except Exception:
                pass

            try:
                headers = {}
                if self.cookie:
                    headers["Cookie"] = self.cookie
                await self._session.delete(
                    self.channel_url,
                    headers=headers,
                    timeout=5,
                )
            except Exception:
                pass

        if self._session:
            await self._session.close()
            self._session = None


class TlonAdapter(BasePlatformAdapter):
    """
    Hermes Gateway adapter for Tlon (Urbit).

    Connects to a Tlon ship and monitors channels + DMs for messages,
    dispatching them to the Hermes agent session store.
    """

    MAX_MESSAGE_LENGTH = MAX_MESSAGE_LENGTH

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.TLON)

        # Read config from env vars (following Hermes convention)
        self.ship_url = os.getenv("TLON_SHIP_URL", "").rstrip("/")
        self.ship_name = _normalize_ship(os.getenv("TLON_SHIP_NAME", ""))
        self.ship_code = os.getenv("TLON_SHIP_CODE", "")

        # Channels to monitor
        channels_str = os.getenv("TLON_CHANNELS", "")
        self.monitored_channels: Set[str] = set(
            ch.strip() for ch in channels_str.split(",") if ch.strip()
        )

        # DM allowlist
        dm_str = os.getenv("TLON_DM_ALLOWLIST", "")
        self.dm_allowlist: Set[str] = set(
            _normalize_ship(s) for s in dm_str.split(",") if s.strip()
        )

        # User allowlist (for authorization)
        users_str = os.getenv("TLON_ALLOWED_USERS", "")
        self.allowed_users: Set[str] = set(
            _normalize_ship(s) for s in users_str.split(",") if s.strip()
        )
        self.allow_all = os.getenv("TLON_ALLOW_ALL_USERS", "").lower() in ("true", "1", "yes")

        # Auto-discover channels
        self.auto_discover = os.getenv("TLON_AUTO_DISCOVER", "").lower() in ("true", "1", "yes")

        # SSE client
        self._sse: Optional[TlonSSEClient] = None

        # Dedup tracker
        self._processed_ids: Set[str] = set()
        self._max_processed = 2000

        # Bot nickname cache
        self._bot_nickname: Optional[str] = None

    async def connect(self) -> bool:
        """Connect to the Tlon ship and start listening."""
        if not self.ship_url or not self.ship_name or not self.ship_code:
            logger.error("[tlon] Missing config: TLON_SHIP_URL, TLON_SHIP_NAME, TLON_SHIP_CODE")
            return False

        try:
            self._sse = TlonSSEClient(
                url=self.ship_url,
                code=self.ship_code,
                ship=self.ship_name,
            )

            # Authenticate
            await self._sse.authenticate()

            # Fetch bot profile for nickname
            try:
                profile = await self._sse.scry("/contacts/v1/self.json")
                if profile and isinstance(profile, dict):
                    self._bot_nickname = profile.get("nickname", {}).get("value")
                    if self._bot_nickname:
                        logger.info("[tlon] Bot nickname: %s", self._bot_nickname)
            except Exception as e:
                logger.debug("[tlon] Could not fetch self profile: %s", e)

            # Auto-discover channels from groups
            if self.auto_discover:
                try:
                    discovered = await self._discover_channels()
                    self.monitored_channels.update(discovered)
                    logger.info("[tlon] Auto-discovered %d channel(s)", len(discovered))
                except Exception as e:
                    logger.warning("[tlon] Auto-discovery failed: %s", e)

            if self.monitored_channels:
                logger.info("[tlon] Monitoring %d channel(s): %s",
                           len(self.monitored_channels),
                           ", ".join(sorted(self.monitored_channels)))
            else:
                logger.info("[tlon] No group channels configured (DMs only)")

            # Subscribe to channels firehose (/v2) for group messages
            await self._sse.subscribe(
                app="channels",
                path="/v2",
                on_event=self._handle_channel_event,
                on_error=lambda e: logger.error("[tlon] Channels error: %s", e),
                on_quit=lambda: logger.info("[tlon] Channels quit received"),
            )

            # Subscribe to chat firehose (/v3) for DMs
            await self._sse.subscribe(
                app="chat",
                path="/v3",
                on_event=self._handle_dm_event,
                on_error=lambda e: logger.error("[tlon] Chat error: %s", e),
                on_quit=lambda: logger.info("[tlon] Chat quit received"),
            )

            # Connect and start streaming
            await self._sse.connect()

            self._running = True
            logger.info("[tlon] Connected and listening!")
            return True

        except Exception as e:
            logger.error("[tlon] Connection failed: %s", e)
            return False

    async def disconnect(self) -> None:
        """Disconnect from the Tlon ship."""
        self._running = False
        if self._sse:
            await self._sse.close()
            self._sse = None
        logger.info("[tlon] Disconnected")

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """
        Send a message to a Tlon channel or DM.

        chat_id: channel nest (e.g. "chat/~host/channel") or ship name for DMs
        """
        if not self._sse or not self._sse._connected:
            return SendResult(success=False, error="Not connected")

        try:
            sent_at = int(time.time() * 1000)
            story = _text_to_story(content)

            if chat_id.startswith("~"):
                # DM
                await self._send_dm(chat_id, story, sent_at, reply_to)
            else:
                # Channel post
                await self._send_channel_post(chat_id, story, sent_at, reply_to)

            msg_id = f"{self.ship_name}/{sent_at}"
            return SendResult(success=True, message_id=msg_id)

        except Exception as e:
            logger.error("[tlon] Send failed: %s", e)
            return SendResult(success=False, error=str(e))

    async def _send_dm(
        self,
        to_ship: str,
        story: list,
        sent_at: int,
        reply_to: Optional[str] = None,
    ) -> None:
        """Send a DM via %chat poke."""
        to_ship = _normalize_ship(to_ship)

        if reply_to:
            # Reply to a specific message in a DM thread
            await self._sse.poke(
                app="chat",
                mark="chat-action",
                json_data={
                    "ship": to_ship.lstrip("~"),
                    "diff": {
                        "reply": {
                            "id": reply_to,
                            "delta": {
                                "add": {
                                    "memo": {
                                        "content": story,
                                        "author": self.ship_name.lstrip("~"),
                                        "sent": sent_at,
                                    }
                                }
                            },
                        }
                    },
                },
            )
        else:
            await self._sse.poke(
                app="chat",
                mark="chat-action",
                json_data={
                    "ship": to_ship.lstrip("~"),
                    "diff": {
                        "add": {
                            "essay": {
                                "content": story,
                                "author": self.ship_name.lstrip("~"),
                                "sent": sent_at,
                            }
                        }
                    },
                },
            )

    async def _send_channel_post(
        self,
        nest: str,
        story: list,
        sent_at: int,
        reply_to: Optional[str] = None,
    ) -> None:
        """Send a post to a channel (chat, heap, diary)."""
        if reply_to:
            await self._sse.poke(
                app="channels",
                mark="channel-action",
                json_data={
                    "channel": {
                        "nest": nest,
                        "action": {
                            "post": {
                                "reply": {
                                    "id": reply_to,
                                    "delta": {
                                        "add": {
                                            "memo": {
                                                "content": story,
                                                "author": self.ship_name.lstrip("~"),
                                                "sent": sent_at,
                                            }
                                        }
                                    },
                                }
                            }
                        },
                    }
                },
            )
        else:
            await self._sse.poke(
                app="channels",
                mark="channel-action",
                json_data={
                    "channel": {
                        "nest": nest,
                        "action": {
                            "post": {
                                "add": {
                                    "essay": {
                                        "content": story,
                                        "author": self.ship_name.lstrip("~"),
                                        "sent": sent_at,
                                    }
                                }
                            }
                        },
                    }
                },
            )

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
    ) -> SendResult:
        """Send an image as a Tlon story block with optional caption."""
        story = []
        if caption:
            story.extend(_text_to_story(caption))
        # Add image block
        story.append({
            "block": {
                "image": {
                    "src": image_url,
                    "alt": caption or "",
                    "width": 0,
                    "height": 0,
                }
            }
        })

        sent_at = int(time.time() * 1000)
        try:
            if chat_id.startswith("~"):
                await self._send_dm(chat_id, story, sent_at, reply_to)
            else:
                await self._send_channel_post(chat_id, story, sent_at, reply_to)
            return SendResult(success=True, message_id=f"{self.ship_name}/{sent_at}")
        except Exception as e:
            return SendResult(success=False, error=str(e))

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Get info about a chat/channel."""
        if chat_id.startswith("~"):
            return {
                "name": chat_id,
                "type": "dm",
                "chat_id": chat_id,
            }
        parsed = _parse_channel_nest(chat_id)
        return {
            "name": parsed["name"] if parsed else chat_id,
            "type": "group",
            "chat_id": chat_id,
        }

    def _is_bot_mentioned(self, text: str) -> bool:
        """Check if the bot is mentioned in the text."""
        text_lower = text.lower()
        # Check ship name mention
        if self.ship_name.lower() in text_lower:
            return True
        # Check nickname mention
        if self._bot_nickname and self._bot_nickname.lower() in text_lower:
            return True
        return False

    def _strip_bot_mention(self, text: str) -> str:
        """Remove bot mentions from text."""
        # Remove ship name
        text = re.sub(
            re.escape(self.ship_name),
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()
        # Remove nickname if set
        if self._bot_nickname:
            text = re.sub(
                re.escape(self._bot_nickname),
                "",
                text,
                flags=re.IGNORECASE,
            ).strip()
        return text

    def _mark_processed(self, msg_id: str) -> bool:
        """
        Mark a message ID as processed. Returns True if this is new,
        False if already processed (duplicate).
        """
        if msg_id in self._processed_ids:
            return False
        self._processed_ids.add(msg_id)
        # Trim old entries
        if len(self._processed_ids) > self._max_processed:
            # Remove oldest entries (set doesn't preserve order, but this is fine
            # for dedup purposes - we just prevent unbounded growth)
            excess = len(self._processed_ids) - self._max_processed
            to_remove = list(self._processed_ids)[:excess]
            for item in to_remove:
                self._processed_ids.discard(item)
        return True

    async def _discover_channels(self) -> Set[str]:
        """Discover channels from groups the bot is a member of."""
        channels: Set[str] = set()
        try:
            # Scry all groups
            groups = await self._sse.scry("/groups/v1/groups.json")
            if not isinstance(groups, dict):
                return channels

            for group_flag, group_data in groups.items():
                if not isinstance(group_data, dict):
                    continue
                group_channels = group_data.get("channels", {})
                if not isinstance(group_channels, dict):
                    continue
                for nest in group_channels:
                    # Only monitor chat and heap channels
                    if nest.startswith("chat/") or nest.startswith("heap/"):
                        channels.add(nest)
        except Exception as e:
            logger.debug("[tlon] Channel discovery scry failed: %s", e)
        return channels

    async def _handle_channel_event(self, event: Any) -> None:
        """Handle a channels firehose (/v2) event."""
        try:
            if not isinstance(event, dict):
                return

            nest = event.get("nest")
            if not nest:
                return

            # Auto-watch channels from firehose
            if nest not in self.monitored_channels:
                if self.auto_discover and (nest.startswith("chat/") or nest.startswith("heap/")):
                    self.monitored_channels.add(nest)
                    logger.info("[tlon] Auto-watching channel: %s", nest)
                else:
                    return

            response = event.get("response")
            if not response:
                return

            # Extract post content (new posts and replies)
            essay = (response.get("post", {})
                     .get("r-post", {})
                     .get("set", {})
                     .get("essay"))
            memo = (response.get("post", {})
                    .get("r-post", {})
                    .get("reply", {})
                    .get("r-reply", {})
                    .get("set", {})
                    .get("memo"))

            content = memo or essay
            if not content:
                return

            is_thread_reply = bool(memo)
            msg_id = (
                response.get("post", {}).get("r-post", {}).get("reply", {}).get("id")
                if is_thread_reply
                else response.get("post", {}).get("id")
            )

            if not msg_id or not self._mark_processed(str(msg_id)):
                return

            sender = _normalize_ship(content.get("author", ""))
            if not sender or sender == self.ship_name:
                return

            text = _extract_message_text(content.get("content"))
            if not text.strip():
                return

            # In group channels, only respond to mentions
            if not self._is_bot_mentioned(text):
                return

            # Check user authorization
            if not self._is_user_allowed(sender):
                logger.info("[tlon] Unauthorized user %s in %s", sender, nest)
                return

            # Strip bot mention from text
            clean_text = self._strip_bot_mention(text)

            # Get parent ID for thread context
            seal = (
                response.get("post", {}).get("r-post", {}).get("reply", {})
                .get("r-reply", {}).get("set", {}).get("seal")
                if is_thread_reply
                else response.get("post", {}).get("r-post", {}).get("set", {}).get("seal")
            )
            parent_id = None
            if seal:
                parent_id = seal.get("parent-id") or seal.get("parent")

            # Build message event
            parsed = _parse_channel_nest(nest)
            source = self.build_source(
                chat_id=nest,
                chat_name=parsed["name"] if parsed else nest,
                chat_type="group",
                user_id=sender,
                user_name=sender,
                thread_id=str(parent_id) if parent_id else None,
            )

            event_obj = MessageEvent(
                text=clean_text,
                message_type=MessageType.TEXT,
                source=source,
                message_id=str(msg_id),
                reply_to_message_id=str(parent_id) if parent_id else None,
                timestamp=datetime.fromtimestamp(
                    content.get("sent", time.time() * 1000) / 1000
                ),
            )

            await self.handle_message(event_obj)

        except Exception as e:
            logger.error("[tlon] Channel event error: %s", e, exc_info=True)

    async def _handle_dm_event(self, event: Any) -> None:
        """Handle a chat firehose (/v3) event."""
        try:
            # Handle DM invite arrays
            if isinstance(event, list):
                for invite in event:
                    ship = _normalize_ship(invite.get("ship", ""))
                    if ship and self._is_user_allowed(ship):
                        try:
                            await self._sse.poke(
                                app="chat",
                                mark="chat-dm-rsvp",
                                json_data={"ship": ship.lstrip("~"), "ok": True},
                            )
                            logger.info("[tlon] Auto-accepted DM invite from %s", ship)
                        except Exception as e:
                            logger.error("[tlon] Failed to accept DM from %s: %s", ship, e)
                return

            if not isinstance(event, dict):
                return

            if "whom" not in event or "response" not in event:
                return

            whom = event["whom"]
            msg_id = event.get("id")
            response = event["response"]

            # Extract message content
            essay = response.get("add", {}).get("essay") if isinstance(response.get("add"), dict) else None
            dm_reply_memo = None
            dm_reply = response.get("reply")
            if isinstance(dm_reply, dict):
                dm_reply_memo = (dm_reply.get("delta", {})
                                .get("add", {})
                                .get("memo"))

            content = essay or dm_reply_memo
            if not content:
                return

            is_thread_reply = bool(dm_reply_memo)
            effective_id = msg_id
            if is_thread_reply and dm_reply:
                effective_id = dm_reply.get("id") or dm_reply.get("delta", {}).get("add", {}).get("id") or msg_id

            if not effective_id or not self._mark_processed(str(effective_id)):
                return

            sender = _normalize_ship(content.get("author", ""))
            # Extract DM partner from whom field
            partner = _normalize_ship(whom) if isinstance(whom, str) else ""

            # Use partner for routing, author for identity
            effective_sender = partner or sender
            if not effective_sender or effective_sender == self.ship_name:
                return

            text = _extract_message_text(content.get("content"))
            if not text.strip():
                return

            # Check DM authorization
            if not self._is_user_allowed(effective_sender):
                logger.info("[tlon] Unauthorized DM from %s", effective_sender)
                return

            # Build message event
            source = self.build_source(
                chat_id=effective_sender,
                chat_name=effective_sender,
                chat_type="dm",
                user_id=effective_sender,
                user_name=effective_sender,
                thread_id=str(msg_id) if is_thread_reply else None,
            )

            event_obj = MessageEvent(
                text=text,
                message_type=MessageType.TEXT,
                source=source,
                message_id=str(effective_id),
                reply_to_message_id=str(msg_id) if is_thread_reply else None,
                timestamp=datetime.fromtimestamp(
                    content.get("sent", time.time() * 1000) / 1000
                ),
            )

            await self.handle_message(event_obj)

        except Exception as e:
            logger.error("[tlon] DM event error: %s", e, exc_info=True)

    def _is_user_allowed(self, ship: str) -> bool:
        """Check if a ship is authorized to interact with the bot."""
        # Global allow-all
        global_allow = os.getenv("GATEWAY_ALLOW_ALL_USERS", "").lower() in ("true", "1", "yes")
        if global_allow or self.allow_all:
            return True

        # Check global allowlist
        global_users = os.getenv("GATEWAY_ALLOWED_USERS", "")
        if global_users:
            allowed = set(_normalize_ship(s) for s in global_users.split(",") if s.strip())
            if ship in allowed:
                return True

        # Check Tlon-specific allowlist
        if self.allowed_users and ship in self.allowed_users:
            return True

        # If no allowlists configured at all, deny
        if not self.allowed_users and not global_users:
            return False

        return False

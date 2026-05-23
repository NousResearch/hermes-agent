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
  TLON_BOT_ALIASES - Comma-separated names that count as mentions (default: Hermes)
  TLON_OWNER_LISTEN_ENABLED - Set to "false" to require mentions from owner in groups
  TLON_CHANNEL_REFRESH_INTERVAL - Seconds between group/channel discovery refreshes (default: 120)
"""

import asyncio
import contextlib
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
)
from gateway.platforms.tlon_approval import (
    PendingApproval,
    create_pending_approval,
    emoji_to_approval_action,
    find_pending_approval,
    format_approval_request,
    format_blocked_list,
    format_confirmation,
    format_pending_list,
    has_duplicate_pending,
    normalize_notification_id,
    prune_expired,
)
from gateway.platforms.tlon_discovery import (
    TlonDiscovery,
    parse_groups_ui_init,
    parse_legacy_groups,
    pending_group_invites,
)
from gateway.platforms.tlon_media import (
    combined_message_type,
    download_blob_attachments,
    download_story_images,
    format_blob_annotations,
    parse_blob_data,
)
from gateway.platforms.tlon_settings import (
    SETTINGS_BUCKET,
    SETTINGS_DESK,
    TlonSettings,
    apply_settings_update,
    parse_settings_event,
    parse_settings_response,
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


def _parse_csv(value: str) -> List[str]:
    """Parse a comma-separated env value into non-empty stripped strings."""
    return [item.strip() for item in value.split(",") if item.strip()]


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


def _extract_author_ship(author: Any) -> str:
    """Extract a normalized ship from a Tlon author field."""
    if isinstance(author, dict):
        for key in ("ship", "id", "patp"):
            value = author.get(key)
            if isinstance(value, str) and value:
                return _normalize_ship(value)
        return ""
    if isinstance(author, str):
        return _normalize_ship(author)
    return ""


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
                    text = _extract_inline_text(block["inline"])
                    if text:
                        parts.append(text)
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
                        elif "header" in b:
                            parts.append(_extract_inline_text(b["header"].get("content", [])))
                        elif "rule" in b:
                            parts.append("---")
        return "\n".join(p for p in parts if p).strip()

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
                elif "strike" in item:
                    parts.append(_extract_inline_text(item["strike"]))
                elif "blockquote" in item:
                    parts.append(_extract_inline_text(item["blockquote"]))
                elif "inline-code" in item:
                    parts.append(item["inline-code"])
                elif "code" in item:
                    parts.append(item["code"])
                elif "break" in item:
                    parts.append("\n")
                elif "tag" in item:
                    parts.append(f"#{item['tag']}")
        return "".join(parts)
    return ""


def _merge_adjacent_strings(inlines: List[Any]) -> List[Any]:
    """Merge neighboring string inlines after markdown parsing."""
    merged: List[Any] = []
    for inline in inlines:
        if (
            isinstance(inline, str)
            and merged
            and isinstance(merged[-1], str)
        ):
            merged[-1] += inline
        else:
            merged.append(inline)
    return merged


def _parse_inline_markdown(text: str) -> List[Any]:
    """Parse the subset of markdown that Tlon story inlines support."""
    result: List[Any] = []
    remaining = text

    while remaining:
        image_match = re.match(r'^!\[([^\]]*)\]\(([^)]+)\)', remaining)
        if image_match:
            result.append({
                "__image": {
                    "src": image_match.group(2),
                    "alt": image_match.group(1),
                }
            })
            remaining = remaining[len(image_match.group(0)):]
            continue

        ship_match = re.match(r'^(~[a-z][a-z0-9-]*)', remaining)
        if ship_match:
            result.append({"ship": ship_match.group(1)})
            remaining = remaining[len(ship_match.group(0)):]
            continue

        bold_match = re.match(r'^\*\*(.+?)\*\*|^__(.+?)__', remaining)
        if bold_match:
            content = bold_match.group(1) or bold_match.group(2)
            result.append({"bold": _parse_inline_markdown(content)})
            remaining = remaining[len(bold_match.group(0)):]
            continue

        italics_match = re.match(r'^\*([^*]+?)\*|^_([^_]+?)_(?![a-zA-Z0-9])', remaining)
        if italics_match:
            content = italics_match.group(1) or italics_match.group(2)
            result.append({"italics": _parse_inline_markdown(content)})
            remaining = remaining[len(italics_match.group(0)):]
            continue

        strike_match = re.match(r'^~~(.+?)~~', remaining)
        if strike_match:
            result.append({"strike": _parse_inline_markdown(strike_match.group(1))})
            remaining = remaining[len(strike_match.group(0)):]
            continue

        code_match = re.match(r'^`([^`]+)`', remaining)
        if code_match:
            result.append({"inline-code": code_match.group(1)})
            remaining = remaining[len(code_match.group(0)):]
            continue

        link_match = re.match(r'^\[([^\]]+)\]\(([^)]+)\)', remaining)
        if link_match:
            result.append({
                "link": {
                    "href": link_match.group(2),
                    "content": link_match.group(1),
                }
            })
            remaining = remaining[len(link_match.group(0)):]
            continue

        url_match = re.match(r'^(https?://[^\s<>"\]]+)', remaining)
        if url_match:
            url = url_match.group(1)
            result.append({"link": {"href": url, "content": url}})
            remaining = remaining[len(url):]
            continue

        special_indices = [
            idx for idx in (
                remaining.find("!["),
                remaining.find("**"),
                remaining.find("__"),
                remaining.find("~~"),
                remaining.find("`"),
                remaining.find("["),
                remaining.find("~"),
                remaining.find("\n"),
                remaining.find("*"),
                remaining.find("_"),
            )
            if idx >= 0
        ]
        url_index = re.search(r'https?://', remaining)
        if url_index:
            special_indices.append(url_index.start())

        next_token_index = min(special_indices) if special_indices else -1
        if next_token_index > 0:
            result.append(remaining[:next_token_index])
            remaining = remaining[next_token_index:]
            continue

        result.append(remaining[0])
        remaining = remaining[1:]

    return _merge_adjacent_strings(result)


def _replace_newlines_with_breaks(inlines: List[Any]) -> List[Any]:
    """Turn literal newlines inside string inlines into Tlon break elements."""
    with_breaks: List[Any] = []
    for inline in inlines:
        if isinstance(inline, str) and "\n" in inline:
            pieces = inline.split("\n")
            for index, piece in enumerate(pieces):
                if piece:
                    with_breaks.append(piece)
                if index < len(pieces) - 1:
                    with_breaks.append({"break": None})
        else:
            with_breaks.append(inline)
    return with_breaks


def _split_image_markers(inlines: List[Any]) -> Tuple[List[Any], List[Dict[str, Any]]]:
    """Hoist markdown image markers out of inline content into story blocks."""
    clean: List[Any] = []
    images: List[Dict[str, Any]] = []
    for inline in inlines:
        if isinstance(inline, dict) and "__image" in inline:
            image = inline["__image"]
            images.append({
                "block": {
                    "image": {
                        "src": image.get("src", ""),
                        "alt": image.get("alt", ""),
                        "width": 0,
                        "height": 0,
                    }
                }
            })
        else:
            clean.append(inline)
    return clean, images


def _text_to_story(text: str) -> list:
    """
    Convert plain text/markdown to Tlon's story format.

    Returns a list of story blocks suitable for use as post content.
    """
    story: List[Dict[str, Any]] = []
    lines = text.split("\n")
    index = 0

    while index < len(lines):
        line = lines[index]

        if line.startswith("```"):
            lang = line[3:].strip() or "plaintext"
            code_lines = []
            index += 1
            while index < len(lines) and not lines[index].startswith("```"):
                code_lines.append(lines[index])
                index += 1
            if index < len(lines):
                index += 1
            story.append({
                "block": {
                    "code": {
                        "code": "\n".join(code_lines),
                        "lang": lang,
                    }
                }
            })
            continue

        header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
        if header_match:
            level = len(header_match.group(1))
            story.append({
                "block": {
                    "header": {
                        "tag": f"h{level}",
                        "content": _parse_inline_markdown(header_match.group(2)),
                    }
                }
            })
            index += 1
            continue

        if re.match(r'^(-{3,}|\*{3,})$', line.strip()):
            story.append({"block": {"rule": None}})
            index += 1
            continue

        if line.startswith("> "):
            quote_lines = []
            while index < len(lines) and lines[index].startswith("> "):
                quote_lines.append(lines[index][2:])
                index += 1
            story.append({
                "inline": [{
                    "blockquote": _parse_inline_markdown("\n".join(quote_lines))
                }]
            })
            continue

        if not line.strip():
            index += 1
            continue

        paragraph_lines = []
        while (
            index < len(lines)
            and lines[index].strip()
            and not lines[index].startswith("#")
            and not lines[index].startswith("```")
            and not lines[index].startswith("> ")
            and not re.match(r'^(-{3,}|\*{3,})$', lines[index].strip())
        ):
            paragraph_lines.append(lines[index])
            index += 1

        inlines = _parse_inline_markdown("\n".join(paragraph_lines))
        inlines = _replace_newlines_with_breaks(inlines)
        clean_inlines, image_blocks = _split_image_markers(inlines)

        if clean_inlines:
            story.append({"inline": clean_inlines})
        story.extend(image_blocks)

    return story or [{"inline": [""]}]


def _format_ud(num: int) -> str:
    """
    Format a number as Urbit @ud (dot-separated groups of 3 digits).

    Example: 170141184505128523237 → "170.141.184.505.128.523.237"
    """
    s = str(num)
    # Insert dots every 3 digits from the right
    groups = []
    while len(s) > 3:
        groups.append(s[-3:])
        s = s[:-3]
    groups.append(s)
    return ".".join(reversed(groups))


def _normalize_post_id(message_id: Any) -> str:
    """Normalize post IDs for equality across dotted and raw @ud forms."""
    return str(message_id or "").replace(".", "")


# Urbit @da epoch offset and second size (from @urbit/aura)
_DA_UNIX_EPOCH = 170141184475152167957503069145530368000
_DA_SECOND = 18446744073709551616


def _da_from_unix(unix_ms: int) -> str:
    """
    Convert Unix timestamp (ms) to Urbit @da bigint, returned as @ud string.

    Replicates: formatUd(da.fromUnix(sentAt).toString()) from @urbit/aura.
    """
    time_since_epoch = unix_ms * _DA_SECOND // 1000
    da_value = _DA_UNIX_EPOCH + time_since_epoch
    return _format_ud(da_value)


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

        action_types = [a.get("action", "?") for a in actions]
        logger.debug("[tlon] Sending %d action(s) to %s: %s",
                     len(actions), self.channel_url, action_types)

        # Let the cookie jar handle auth (set by authenticate())
        async with self._session.put(
            self.channel_url,
            json=actions,
            headers={"Content-Type": "application/json"},
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            if resp.status not in (200, 204):
                text = await resp.text()
                logger.error("[tlon] Channel action failed: HTTP %d - %s",
                            resp.status, text[:200])
                raise ConnectionError(
                    f"Channel action failed: HTTP {resp.status} - {text[:200]}"
                )
            logger.debug("[tlon] Action(s) sent OK: HTTP %d", resp.status)

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
        # Let cookie jar handle auth
        headers = {"Accept": "text/event-stream"}
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

        logger.debug("[tlon] SSE event id=%s, data=%s", event_id, data[:120])

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

        if parsed.get("response") == "err":
            sub_id = parsed.get("id")
            if sub_id and sub_id in self._event_handlers:
                handler = self._event_handlers[sub_id]
                if handler.get("err"):
                    handler["err"](parsed.get("json") or parsed)
            return

        # Dispatch to handlers
        sub_id = parsed.get("id")
        event_json = parsed.get("json")
        resp_type = parsed.get("response", "")

        logger.debug("[tlon] Dispatching: sub_id=%s, response=%s, has_json=%s, handlers=%s",
                     sub_id, resp_type, event_json is not None, list(self._event_handlers.keys()))

        if sub_id and sub_id in self._event_handlers:
            handler = self._event_handlers[sub_id]
            if handler.get("event") and event_json is not None:
                try:
                    await handler["event"](event_json)
                except Exception as e:
                    logger.error("[tlon] Event handler error: %s", e)
        elif event_json is not None:
            # Some %channels/%groups events arrive without a usable
            # subscription id. OpenClaw broadcasts these to all handlers and
            # lets each handler filter by event shape; do the same so group
            # mentions do not disappear before _handle_channel_event sees them.
            logger.debug("[tlon] Broadcasting event with unknown sub_id=%s", sub_id)
            for handler in list(self._event_handlers.values()):
                if not handler.get("event"):
                    continue
                try:
                    await handler["event"](event_json)
                except Exception as e:
                    logger.error("[tlon] Event handler error: %s", e)

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
        """
        Send a poke via a one-shot Eyre channel.

        Uses a separate channel from the SSE stream (matching openclaw-tlon's
        http-poke.ts pattern). Sending pokes on the SSE channel can cause
        them to be silently dropped.
        """
        import aiohttp

        poke_channel_id = await self._new_channel_id()
        poke_url = f"{self.url}/~/channel/{poke_channel_id}"

        action = {
            "id": int(time.time() * 1000),
            "action": "poke",
            "ship": self.ship.lstrip("~"),
            "app": app,
            "mark": mark,
            "json": json_data,
        }

        logger.info("[tlon] One-shot poke to %s mark=%s json=%s",
                    poke_url, mark, json.dumps(json_data)[:300])
        async with self._session.put(
            poke_url,
            json=[action],
            headers={"Content-Type": "application/json"},
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            if resp.status not in (200, 204):
                text = await resp.text()
                logger.error("[tlon] Poke failed: HTTP %d - %s", resp.status, text[:200])
                raise ConnectionError(f"Poke failed: HTTP {resp.status}")
            logger.debug("[tlon] Poke PUT OK: HTTP %d", resp.status)

        # Read SSE ack/nack from the one-shot channel when Eyre provides one.
        try:
            ack_url = f"{poke_url}?msg=0"
            async with self._session.get(
                ack_url,
                headers={"Accept": "text/event-stream"},
                timeout=aiohttp.ClientTimeout(total=5),
            ) as ack_resp:
                async for line in ack_resp.content:
                    decoded = line.decode("utf-8", errors="replace").strip()
                    if not decoded:
                        continue
                    if decoded.startswith("data:"):
                        data_str = decoded[5:].strip()
                        try:
                            data = json.loads(data_str)
                            if data.get("ok") is not None:
                                logger.info("[tlon] Poke ACK: %s", data)
                                break
                            elif data.get("err"):
                                logger.error("[tlon] Poke NACK: %s", data)
                                raise ConnectionError(f"Poke rejected: {data.get('err')}")
                            elif "ok" in str(data) or "err" in str(data):
                                logger.info("[tlon] Poke response: %s", data)
                                break
                        except json.JSONDecodeError:
                            if "ok" in data_str or "err" in data_str:
                                logger.info("[tlon] Poke SSE raw: %s", data_str[:200])
                                break
        except asyncio.TimeoutError:
            logger.warning("[tlon] Poke ack read timed out (5s)")
        except ConnectionError:
            raise
        except Exception as e:
            logger.warning("[tlon] Poke ack read error: %s", e)
        finally:
            with contextlib.suppress(Exception):
                await self._session.delete(
                    poke_url,
                    timeout=aiohttp.ClientTimeout(total=5),
                )

    async def scry(self, path: str) -> Any:
        """Scry a path and return the JSON response."""
        import aiohttp

        if path.startswith("/~/scry/"):
            path = path[len("/~/scry"):]
        elif path.startswith("~/scry/"):
            path = "/" + path[len("~/scry/"):]

        full_path = path if path.endswith(".json") else f"{path}.json"
        # Use /~/scry prefix for Eyre scry endpoint
        scry_url = f"{self.url}/~/scry{full_path}"
        async with self._session.get(
            scry_url,
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
                await self._session.delete(
                    self.channel_url,
                    timeout=aiohttp.ClientTimeout(total=5),
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
        self.auto_accept_dm_invites = os.getenv(
            "TLON_AUTO_ACCEPT_DM_INVITES",
            "",
        ).lower() in ("true", "1", "yes")
        self.auto_accept_group_invites = os.getenv(
            "TLON_AUTO_ACCEPT_GROUP_INVITES",
            "",
        ).lower() in ("true", "1", "yes")

        self.owner_ship = _normalize_ship(os.getenv("TLON_OWNER_SHIP", ""))
        default_auth = os.getenv("TLON_DEFAULT_AUTHORIZED_SHIPS", "")
        self.default_authorized_ships: Set[str] = set(
            _normalize_ship(s) for s in default_auth.split(",") if s.strip()
        )
        self.blocked_ships: Set[str] = set(
            _normalize_ship(s)
            for s in os.getenv("TLON_BLOCKED_SHIPS", "").split(",")
            if s.strip()
        )
        alias_values = _parse_csv(os.getenv("TLON_BOT_ALIASES", "Hermes"))
        self.bot_aliases: List[str] = []
        for alias in alias_values:
            if alias and alias.lower() not in {item.lower() for item in self.bot_aliases}:
                self.bot_aliases.append(alias)
        self.owner_listen_enabled = (
            os.getenv("TLON_OWNER_LISTEN_ENABLED", "true").lower()
            not in ("false", "0", "no")
        )
        self.owner_listen_disabled_channels: Set[str] = set(
            _parse_csv(os.getenv("TLON_OWNER_LISTEN_DISABLED_CHANNELS", ""))
        )
        self.channel_rules: Dict[str, Dict[str, Any]] = self._load_channel_rules_from_env()
        self.pending_approvals: List[PendingApproval] = []
        self._exec_approval_prompts: Dict[str, Dict[str, str]] = {}
        self._exec_approval_prompt_by_session: Dict[str, str] = {}
        self._processed_approval_reactions: Set[str] = set()

        # SSE client
        self._sse: Optional[TlonSSEClient] = None
        self._settings = TlonSettings()
        self._settings_loaded = False

        # Dedup tracker
        self._processed_ids: Set[str] = set()
        self._processed_dm_invites: Set[str] = set()
        self._max_processed = 2000
        self._dm_poll_task: Optional[asyncio.Task] = None
        self._channel_refresh_task: Optional[asyncio.Task] = None
        self._dm_poll_initialized: Set[str] = set()
        self.dm_poll_enabled = (
            os.getenv("TLON_DM_POLL_ENABLED", "true").lower()
            not in ("false", "0", "no")
        )
        self.dm_poll_interval = self._env_float("TLON_DM_POLL_INTERVAL", 10.0)
        self.dm_poll_limit = max(1, self._env_int("TLON_DM_POLL_LIMIT", 20))
        self.dm_poll_initial_catchup_seconds = max(
            0.0,
            self._env_float("TLON_DM_POLL_INITIAL_CATCHUP_SECONDS", 1800.0),
        )
        self.channel_refresh_interval = self._env_float(
            "TLON_CHANNEL_REFRESH_INTERVAL",
            120.0,
        )

        # Bot nickname cache
        self._bot_nickname: Optional[str] = None

        # Send dedup: prevent identical messages within a short window
        self._recent_sends: Dict[str, float] = {}  # hash -> timestamp

        # Thread-safe dedup lock for message processing
        self._process_lock = asyncio.Lock()

        self._channel_to_group: Dict[str, str] = {}
        self._group_names: Dict[str, str] = {}
        self._participated_threads: Set[Tuple[str, str]] = set()
        self._blob_retry_delay = self._env_float("TLON_BLOB_RETRY_DELAY", 5.0)

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        raw = os.getenv(name, "")
        if not raw:
            return default
        try:
            return float(raw)
        except ValueError:
            logger.warning("[tlon] Ignoring invalid %s=%r", name, raw)
            return default

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        raw = os.getenv(name, "")
        if not raw:
            return default
        try:
            return int(raw)
        except ValueError:
            logger.warning("[tlon] Ignoring invalid %s=%r", name, raw)
            return default

    def _load_channel_rules_from_env(self) -> Dict[str, Dict[str, Any]]:
        raw = os.getenv("TLON_CHANNEL_RULES", "")
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except ValueError:
            logger.warning("[tlon] Ignoring invalid TLON_CHANNEL_RULES JSON")
            return {}
        if not isinstance(parsed, dict):
            return {}
        rules: Dict[str, Dict[str, Any]] = {}
        for nest, rule in parsed.items():
            if isinstance(nest, str) and isinstance(rule, dict):
                rules[nest] = rule
        return rules

    async def _load_settings(self) -> None:
        if not self._sse:
            return
        try:
            raw = await self._sse.scry("/settings/all.json")
        except Exception as e:
            self._settings_loaded = True
            logger.debug("[tlon] Settings load skipped: %s", e)
            return
        self._settings = parse_settings_response(raw)
        self._settings_loaded = True
        self._apply_settings(self._settings)
        logger.info("[tlon] Settings loaded from %s/%s", SETTINGS_DESK, SETTINGS_BUCKET)

    async def _handle_settings_event(self, event: Any) -> None:
        update = parse_settings_event(event)
        if not update:
            return
        key, value = update
        self._settings = apply_settings_update(self._settings, key, value)
        self._apply_settings(self._settings)
        logger.info("[tlon] Settings updated: %s", key)

    def _apply_settings(self, settings: TlonSettings) -> None:
        if settings.group_channels is not None:
            self.monitored_channels.update(ch for ch in settings.group_channels if ch)
        if settings.dm_allowlist is not None:
            self.dm_allowlist = {_normalize_ship(s) for s in settings.dm_allowlist if s}
        if settings.auto_discover is not None:
            self.auto_discover = settings.auto_discover
        if settings.auto_accept_dm_invites is not None:
            self.auto_accept_dm_invites = settings.auto_accept_dm_invites
        if settings.auto_accept_group_invites is not None:
            self.auto_accept_group_invites = settings.auto_accept_group_invites
        if settings.channel_rules:
            self.channel_rules = settings.channel_rules
        if settings.default_authorized_ships is not None:
            self.default_authorized_ships = {
                _normalize_ship(s) for s in settings.default_authorized_ships if s
            }
        if settings.owner_ship:
            self.owner_ship = _normalize_ship(settings.owner_ship)
        if settings.pending_approvals is not None:
            self.pending_approvals = [
                PendingApproval.from_dict(item)
                for item in settings.pending_approvals
                if isinstance(item, dict)
            ]
        if settings.owner_listen_enabled is not None:
            self.owner_listen_enabled = settings.owner_listen_enabled
        if settings.owner_listen_disabled_channels is not None:
            self.owner_listen_disabled_channels = {
                item for item in settings.owner_listen_disabled_channels if item
            }

    async def _put_settings_entry(self, key: str, value: Any) -> None:
        if not self._sse:
            return
        try:
            await self._sse.poke(
                app="settings",
                mark="settings-event",
                json_data={
                    "put-entry": {
                        "desk": SETTINGS_DESK,
                        "bucket-key": SETTINGS_BUCKET,
                        "entry-key": key,
                        "value": value,
                    }
                },
            )
        except Exception as e:
            logger.debug("[tlon] Failed to write settings entry %s: %s", key, e)

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

            await self._load_settings()

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

            # Subscribe to OpenClaw-compatible settings-store updates. This is
            # intentionally best-effort; the adapter still runs from env config
            # when %settings is unavailable on the ship.
            await self._sse.subscribe(
                app="settings",
                path=f"/desk/{SETTINGS_DESK}",
                on_event=self._handle_settings_event,
                on_error=lambda e: logger.debug("[tlon] Settings error: %s", e),
                on_quit=lambda: logger.info("[tlon] Settings subscription quit received"),
            )

            # Match OpenClaw's group read side: subscribe to groups-ui for
            # live channel/group membership changes, and foreigns for group
            # invite updates. These are best-effort; channels/chat firehoses
            # remain the primary inbound message streams.
            await self._sse.subscribe(
                app="groups",
                path="/groups/ui",
                on_event=self._handle_groups_ui_event,
                on_error=lambda e: logger.error("[tlon] Groups-ui error: %s", e),
                on_quit=lambda: logger.info("[tlon] Groups-ui quit received"),
            )
            logger.info("[tlon] Subscribed to groups-ui for real-time channel detection")

            await self._sse.subscribe(
                app="groups",
                path="/v1/foreigns",
                on_event=self._handle_group_foreigns_event,
                on_error=lambda e: logger.error("[tlon] Foreigns error: %s", e),
                on_quit=lambda: logger.info("[tlon] Foreigns quit received"),
            )
            logger.info("[tlon] Subscribed to foreigns (/v1/foreigns) for group invites")

            # Connect and start streaming
            await self._sse.connect()

            self._running = True
            self._start_dm_history_poller()
            self._start_channel_refresh()
            logger.info("[tlon] Connected and listening!")
            return True

        except Exception as e:
            logger.error("[tlon] Connection failed: %s", e)
            return False

    async def disconnect(self) -> None:
        """Disconnect from the Tlon ship."""
        self._running = False
        if self._dm_poll_task:
            self._dm_poll_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._dm_poll_task
            self._dm_poll_task = None
        if self._channel_refresh_task:
            self._channel_refresh_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._channel_refresh_task
            self._channel_refresh_task = None
        if self._sse:
            await self._sse.close()
            self._sse = None
        logger.info("[tlon] Disconnected")

    def _start_channel_refresh(self) -> None:
        if (
            not self.auto_discover
            or self.channel_refresh_interval <= 0
            or (self._channel_refresh_task and not self._channel_refresh_task.done())
        ):
            return
        self._channel_refresh_task = asyncio.create_task(self._channel_refresh_loop())
        logger.info(
            "[tlon] Channel discovery refresh enabled every %.1fs",
            self.channel_refresh_interval,
        )

    async def _channel_refresh_loop(self) -> None:
        try:
            while self._running:
                await asyncio.sleep(self.channel_refresh_interval)
                await self._refresh_discovered_channels()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("[tlon] Channel discovery refresh stopped: %s", e, exc_info=True)

    async def _refresh_discovered_channels(self) -> None:
        if not self._sse or not self.auto_discover:
            return
        discovered = await self._discover_channels()
        for nest in sorted(discovered):
            if nest not in self.monitored_channels:
                self.monitored_channels.add(nest)
                logger.info("[tlon] Now watching new channel: %s", nest)

    def _dm_poll_targets(self) -> Set[str]:
        """Return explicit DM partners worth polling as an SSE fallback."""
        targets: Set[str] = set()
        for ship in (
            [self.owner_ship]
            + list(self.allowed_users)
            + list(self.dm_allowlist)
            + list(self.default_authorized_ships)
        ):
            normalized = _normalize_ship(ship)
            if normalized and normalized != self.ship_name:
                targets.add(normalized)
        return targets

    def _start_dm_history_poller(self) -> None:
        if (
            not self.dm_poll_enabled
            or self.dm_poll_interval <= 0
            or not self._dm_poll_targets()
            or (self._dm_poll_task and not self._dm_poll_task.done())
        ):
            return
        self._dm_poll_task = asyncio.create_task(self._dm_history_poll_loop())
        logger.info(
            "[tlon] DM history fallback poller enabled for %d target(s) every %.1fs",
            len(self._dm_poll_targets()),
            self.dm_poll_interval,
        )

    async def _dm_history_poll_loop(self) -> None:
        try:
            while self._running:
                await self._poll_dm_histories()
                await asyncio.sleep(self.dm_poll_interval)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("[tlon] DM history fallback poller stopped: %s", e, exc_info=True)

    async def _poll_dm_histories(self) -> None:
        if not self._sse or not self._sse._connected:
            return
        for partner in sorted(self._dm_poll_targets()):
            try:
                await self._poll_dm_history(partner)
            except Exception as e:
                logger.debug("[tlon] DM history poll failed for %s: %s", partner, e)

    async def _poll_dm_history(self, partner: str) -> None:
        posts = await self._fetch_dm_history_posts(partner)
        if partner not in self._dm_poll_initialized:
            await self._initialize_dm_history(partner, posts)
            self._dm_poll_initialized.add(partner)
            return

        for post in posts:
            if self._should_process_dm_history_post(post):
                await self._process_dm_history_post(partner, post)

    async def _fetch_dm_history_posts(self, partner: str) -> List[Dict[str, Any]]:
        if not self._sse:
            return []
        data = await self._sse.scry(
            f"/chat/v4/dm/{partner}/writs/newest/{self.dm_poll_limit}/heavy"
        )
        posts = self._extract_dm_history_posts(data)
        posts.sort(key=lambda post: self._dm_history_sent(post) or 0)
        return posts

    @staticmethod
    def _extract_dm_history_posts(data: Any) -> List[Dict[str, Any]]:
        if not isinstance(data, dict):
            return []
        writs = data.get("writs")
        if isinstance(writs, dict):
            posts: List[Dict[str, Any]] = []
            for key, value in writs.items():
                if isinstance(value, dict):
                    post = dict(value)
                    post.setdefault("_history_key", key)
                    posts.append(post)
            return posts
        if isinstance(writs, list):
            return [post for post in writs if isinstance(post, dict)]
        return []

    async def _initialize_dm_history(
        self,
        partner: str,
        posts: List[Dict[str, Any]],
    ) -> None:
        """Seed old history and catch up at most one recent unanswered DM."""
        now_ms = int(time.time() * 1000)
        catchup_after_ms = now_ms - int(self.dm_poll_initial_catchup_seconds * 1000)
        latest_own_sent = max(
            (
                sent
                for post in posts
                if self._dm_history_author(post) == self.ship_name
                and not self._is_dm_history_status_notice(post)
                for sent in [self._dm_history_sent(post)]
                if sent is not None
            ),
            default=0,
        )

        catchup: List[Dict[str, Any]] = []
        for post in posts:
            msg_id = self._dm_history_id(post)
            if not msg_id:
                continue
            author = self._dm_history_author(post)
            sent = self._dm_history_sent(post)
            if (
                author
                and author != self.ship_name
                and sent is not None
                and sent >= catchup_after_ms
                and sent > latest_own_sent
            ):
                catchup.append(post)
                continue
            self._mark_processed(msg_id)

        if not catchup:
            return

        # Process only the newest recent unanswered DM on startup. If several
        # old user commands accumulated while the gateway was down, avoid
        # replaying side-effectful requests unexpectedly.
        selected = max(catchup, key=lambda post: self._dm_history_sent(post) or 0)
        selected_id = self._dm_history_id(selected)
        for post in catchup:
            msg_id = self._dm_history_id(post)
            if msg_id and msg_id != selected_id:
                self._mark_processed(msg_id)

        logger.info(
            "[tlon] DM history fallback catching up latest unanswered DM from %s",
            partner,
        )
        await self._process_dm_history_post(partner, selected)

    def _should_process_dm_history_post(self, post: Dict[str, Any]) -> bool:
        msg_id = self._dm_history_id(post)
        if not msg_id or msg_id in self._processed_ids:
            return False
        author = self._dm_history_author(post)
        return bool(author and author != self.ship_name)

    async def _process_dm_history_post(
        self,
        partner: str,
        post: Dict[str, Any],
    ) -> None:
        event = self._dm_history_event(partner, post)
        if event:
            await self._handle_dm_event(event)

    def _dm_history_event(
        self,
        partner: str,
        post: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        essay = self._dm_history_essay(post)
        msg_id = self._dm_history_id(post)
        if not essay or not msg_id:
            return None
        return {
            "whom": partner,
            "id": msg_id,
            "response": {"add": {"essay": essay}},
        }

    @staticmethod
    def _dm_history_essay(post: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        essay = post.get("essay") or post.get("memo")
        return essay if isinstance(essay, dict) else None

    def _dm_history_author(self, post: Dict[str, Any]) -> str:
        essay = self._dm_history_essay(post)
        if essay:
            return _extract_author_ship(essay.get("author"))
        return _extract_author_ship(post.get("author"))

    def _dm_history_text(self, post: Dict[str, Any]) -> str:
        essay = self._dm_history_essay(post)
        if not essay:
            return ""
        return _extract_message_text(essay.get("content"))

    def _is_dm_history_status_notice(self, post: Dict[str, Any]) -> bool:
        text = self._dm_history_text(post)
        return (
            "Gateway shutting down" in text
            or "Gateway online" in text
            or "Gateway restarted" in text
        )

    def _dm_history_sent(self, post: Dict[str, Any]) -> Optional[int]:
        essay = self._dm_history_essay(post)
        raw = (essay or {}).get("sent") if essay else None
        if raw is None:
            raw = post.get("sent") or post.get("sentAt")
        try:
            return int(raw)
        except (TypeError, ValueError):
            return None

    def _dm_history_id(self, post: Dict[str, Any]) -> str:
        seal = post.get("seal") if isinstance(post.get("seal"), dict) else {}
        msg_id = (
            post.get("id")
            or seal.get("id")
            or post.get("writId")
            or post.get("_history_key")
        )
        if msg_id:
            msg_id = str(msg_id)
            if "/" in msg_id:
                return msg_id
            author = self._dm_history_author(post)
            return f"{author}/{msg_id}" if author else msg_id

        essay = self._dm_history_essay(post)
        sent = self._dm_history_sent(post)
        author = _extract_author_ship((essay or {}).get("author")) if essay else ""
        if author and sent is not None:
            return f"{author}/{_da_from_unix(sent)}"
        return ""

    async def handle_message(self, event) -> None:
        """Override base adapter's handle_message to bypass the pending-message
        replay system which causes echo loops on Tlon.

        The base adapter queues messages that arrive while an agent is running
        and replays them after the response is sent — but those replayed
        messages re-trigger the agent, creating duplicate responses.

        Instead we process each message directly in its own background task
        with no replay/interrupt machinery."""
        if not self._message_handler:
            return

        async def _run():
            try:
                response = await self._message_handler(event)
                if response:
                    _, response = self.extract_media(response)
                    images, text_content = self.extract_images(response)
                    if text_content:
                        # For thread replies, pass reply_to so the response
                        # goes into the thread (not as a new top-level post).
                        # reply_to_message_id is set by inbound handlers when
                        # the message came from a thread.
                        reply_to = getattr(event, "reply_to_message_id", None)
                        logger.info("[tlon] Sending response (%d chars) to %s reply_to=%s",
                                    len(text_content), event.source.chat_id, reply_to)
                        await self.send(
                            chat_id=event.source.chat_id,
                            content=text_content,
                            reply_to=reply_to,
                        )
                    for img_url, alt in images:
                        await self.send_image(
                            chat_id=event.source.chat_id,
                            image_url=img_url,
                            caption=alt or None,
                        )
            except Exception as e:
                logger.error("[tlon] handle_message error: %s", e, exc_info=True)

        asyncio.create_task(_run())

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
            logger.error("[tlon] Send called but not connected!")
            return SendResult(success=False, error="Not connected")

        try:
            sent_at = int(time.time() * 1000)

            # Dedup: skip identical messages to the same chat within 30s
            import hashlib
            send_hash = hashlib.md5(f"{chat_id}:{content}".encode()).hexdigest()
            now = time.time()
            # Clean old entries
            self._recent_sends = {k: v for k, v in self._recent_sends.items() if now - v < 30}
            if send_hash in self._recent_sends:
                logger.info("[tlon] Dedup: skipping duplicate send to %s (%d chars)", chat_id, len(content))
                return SendResult(success=True, message_id=f"dedup/{sent_at}")
            self._recent_sends[send_hash] = now

            story = _text_to_story(content)
            logger.info("[tlon] Sending to %s (%d chars, story=%d blocks)",
                       chat_id, len(content), len(story))

            if chat_id.startswith("~"):
                # DM — pass reply_to for thread replies
                # reply_to should be the parent writ-id (e.g. "~ship/170.141...")
                msg_id = await self._send_dm(chat_id, story, sent_at, reply_to=reply_to)
            else:
                # Channel post — pass reply_to for thread replies
                # reply_to should be the parent post ID (bare or @ud formatted)
                formatted_reply = None
                if reply_to:
                    # Format as @ud if it's a bare digit string
                    bare = str(reply_to).replace(".", "")
                    if bare.isdigit():
                        formatted_reply = _format_ud(int(bare))
                    else:
                        formatted_reply = str(reply_to)
                msg_id = await self._send_channel_post(chat_id, story, sent_at, reply_to=formatted_reply)
                if formatted_reply:
                    self._participated_threads.add((chat_id, _normalize_post_id(formatted_reply)))

            logger.info("[tlon] ✓ Message sent: %s", msg_id)
            return SendResult(success=True, message_id=msg_id)

        except Exception as e:
            logger.error("[tlon] Send failed: %s", e, exc_info=True)
            return SendResult(success=False, error=str(e))

    async def _send_dm(
        self,
        to_ship: str,
        story: list,
        sent_at: int,
        reply_to: Optional[str] = None,
    ) -> str:
        """Send a DM via %chat poke."""
        to_ship = _normalize_ship(to_ship)
        # Author uses ~ prefix (matching @tloncorp/api)
        author = self.ship_name  # e.g. "~timryd-macnus"

        # Build the writ ID: author/formatUd(da.fromUnix(sentAt))
        ud_time = _da_from_unix(sent_at)
        writ_id = f"{author}/{ud_time}"

        if reply_to:
            # DM reply uses "reply" delta with "memo" (not "essay")
            delta = {
                "reply": {
                    "id": writ_id,
                    "meta": None,
                    "delta": {
                        "add": {
                            "memo": {
                                "content": story,
                                "author": author,
                                "sent": sent_at,
                            },
                            "time": None,
                        },
                    },
                }
            }
            dm_json = {
                "ship": to_ship,
                "diff": {
                    "id": reply_to,
                    "delta": delta,
                },
            }
        else:
            # Top-level DM uses "add" delta with "essay"
            delta = {
                "add": {
                    "essay": {
                        "content": story,
                        "author": author,
                        "sent": sent_at,
                        "kind": "/chat",
                        "meta": None,
                        "blob": None,
                    },
                    "time": None,
                }
            }
            dm_json = {
                "ship": to_ship,
                "diff": {
                    "id": writ_id,
                    "delta": delta,
                },
            }

        logger.debug("[tlon] DM poke JSON: %s", json.dumps(dm_json)[:500])
        await self._sse.poke(
            app="chat",
            mark=os.getenv("TLON_DM_ACTION_MARK", "chat-dm-action-1"),
            json_data=dm_json,
        )
        return writ_id

    async def send_exec_approval(
        self,
        chat_id: str,
        command: str,
        session_key: str,
        description: str = "dangerous command",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a Tlon reaction-based dangerous-command approval prompt.

        Tlon has no gateway-native buttons, so mirror OpenClaw's approval UX:
        the owner reacts to the prompt with 👍, 👎, or 🛑. The prompt is sent
        to the configured owner DM when available so command status does not
        leak into group channels.
        """
        cmd_preview = command[:2000] + "..." if len(command) > 2000 else command
        text = (
            "⚠️ Dangerous command requires approval:\n"
            f"```\n{cmd_preview}\n```\n"
            f"Reason: {description}\n\n"
            "React to this message: 👍 approve once · 👎 deny · 🛑 deny/stop\n\n"
            "Or reply `/approve`, `/approve session`, `/approve always`, or `/deny`."
        )

        target_chat_id = self.owner_ship or chat_id
        result = await self.send(target_chat_id, text, metadata=metadata)
        if not result.success or not result.message_id:
            return result

        normalized_id = normalize_notification_id(result.message_id)
        old_id = self._exec_approval_prompt_by_session.get(session_key)
        if old_id:
            self._exec_approval_prompts.pop(old_id, None)
        self._exec_approval_prompts[normalized_id] = {
            "session_key": session_key,
            "chat_id": target_chat_id,
        }
        self._exec_approval_prompt_by_session[session_key] = normalized_id
        logger.info(
            "[tlon] Registered reaction approval prompt %s for session %s",
            normalized_id,
            session_key,
        )
        return result

    async def _send_channel_post(
        self,
        nest: str,
        story: list,
        sent_at: int,
        reply_to: Optional[str] = None,
    ) -> str:
        """Send a post to a channel (chat, heap, diary)."""
        # Author field WITH ~ prefix (matching @tloncorp/api convention)
        author = self.ship_name if self.ship_name.startswith("~") else f"~{self.ship_name}"

        # Determine kind from nest type
        kind = "/chat"
        if nest.startswith("diary/"):
            kind = "/diary"
        elif nest.startswith("heap/"):
            kind = "/heap"

        if reply_to:
            # Channel reply: post.reply.action.add has flat fields
            action_json = {
                "channel": {
                    "nest": nest,
                    "action": {
                        "post": {
                            "reply": {
                                "id": reply_to,
                                "action": {
                                    "add": {
                                        "content": story,
                                        "author": author,
                                        "sent": sent_at,
                                    }
                                },
                            }
                        }
                    },
                }
            }
        else:
            # Top-level post: post.add has essay fields directly (no wrapper)
            action_json = {
                "channel": {
                    "nest": nest,
                    "action": {
                        "post": {
                            "add": {
                                "content": story,
                                "author": author,
                                "sent": sent_at,
                                "kind": kind,
                                "meta": None,
                                "blob": None,
                            }
                        }
                    },
                }
            }

        logger.debug("[tlon] Channel poke JSON: %s", json.dumps(action_json)[:500])
        await self._sse.poke(
            app="channels",
            mark=os.getenv("TLON_CHANNEL_ACTION_MARK", "channel-action-1"),
            json_data=action_json,
        )
        return _da_from_unix(sent_at)

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
                msg_id = await self._send_dm(chat_id, story, sent_at, reply_to)
            else:
                msg_id = await self._send_channel_post(chat_id, story, sent_at, reply_to)
            return SendResult(success=True, message_id=msg_id)
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
        for name in self._bot_mention_names():
            if self._mention_name_matches(text, name):
                return True
        return False

    def _strip_bot_mention(self, text: str) -> str:
        """Remove bot mentions from text."""
        for name in self._bot_mention_names():
            text = self._remove_mention_name(text, name)
        return text.strip()

    def _bot_mention_names(self) -> List[str]:
        names = [self.ship_name, self._bot_nickname, *self.bot_aliases]
        result: List[str] = []
        seen: Set[str] = set()
        for name in names:
            if not name:
                continue
            key = name.lower()
            if key in seen:
                continue
            seen.add(key)
            result.append(name)
        return result

    def _mention_name_matches(self, text: str, name: str) -> bool:
        pattern = self._mention_pattern(name)
        return bool(re.search(pattern, text, flags=re.IGNORECASE))

    def _remove_mention_name(self, text: str, name: str) -> str:
        pattern = self._mention_pattern(name, allow_trailing_punctuation=True)
        return re.sub(pattern, "", text, flags=re.IGNORECASE).strip()

    @staticmethod
    def _mention_pattern(name: str, *, allow_trailing_punctuation: bool = False) -> str:
        suffix = r"(?:[:,]\s*)?" if allow_trailing_punctuation else ""
        if name.startswith("~"):
            return rf"(?<![\w~]){re.escape(name)}(?![\w-]){suffix}"
        return rf"(?<![\w~]){re.escape(name)}(?![\w-]){suffix}"

    def _mark_processed(self, msg_id: str) -> bool:
        """
        Mark a message ID as processed. Returns True if this is new,
        False if already processed (duplicate).
        """
        if msg_id in self._processed_ids:
            logger.info("[tlon] Dedup: already processed %s", msg_id[:40])
            return False
        self._processed_ids.add(msg_id)
        logger.info("[tlon] Dedup: marking new %s (total=%d)", msg_id[:40], len(self._processed_ids))
        # Trim old entries
        if len(self._processed_ids) > self._max_processed:
            # Remove oldest entries (set doesn't preserve order, but this is fine
            # for dedup purposes - we just prevent unbounded growth)
            excess = len(self._processed_ids) - self._max_processed
            to_remove = list(self._processed_ids)[:excess]
            for item in to_remove:
                self._processed_ids.discard(item)
        return True

    async def _prepare_media_context(
        self,
        *,
        story_content: Any,
        blob: Optional[str],
        text: str,
    ) -> Tuple[str, List[str], List[str], MessageType]:
        """Download Tlon attachments and prepend blob annotations to text."""
        media_paths: List[str] = []
        media_types: List[str] = []
        notices: List[str] = []

        try:
            story_attachments = await download_story_images(story_content)
            for item in story_attachments:
                media_paths.append(item.path)
                media_types.append(item.content_type)
        except Exception as e:
            logger.debug("[tlon] Story image download failed: %s", e)

        blob_entries = parse_blob_data(blob)
        if blob_entries:
            annotations = format_blob_annotations(blob_entries)
            if annotations:
                text = f"{annotations}\n{text}".strip()
            try:
                blob_attachments, blob_notices = await download_blob_attachments(blob_entries)
                notices.extend(blob_notices)
                for item in blob_attachments:
                    media_paths.append(item.path)
                    media_types.append(item.content_type)
            except Exception as e:
                logger.debug("[tlon] Blob download failed: %s", e)

        if notices:
            text = f"{chr(10).join(notices)}\n{text}".strip()

        return text, media_paths, media_types, combined_message_type(media_types)

    def _is_owner(self, ship: str) -> bool:
        return bool(self.owner_ship and _normalize_ship(ship) == self.owner_ship)

    def _should_owner_listen(self, ship: str, nest: str) -> bool:
        return (
            self.owner_listen_enabled
            and self._is_owner(ship)
            and nest not in self.owner_listen_disabled_channels
        )

    def _is_blocked(self, ship: str) -> bool:
        return _normalize_ship(ship) in self.blocked_ships

    def _is_channel_allowed(self, ship: str, nest: str) -> bool:
        ship = _normalize_ship(ship)
        if self._is_blocked(ship):
            return False
        if self._is_owner(ship):
            return True
        if self.allow_all or os.getenv("GATEWAY_ALLOW_ALL_USERS", "").lower() in ("true", "1", "yes"):
            return True
        global_users = os.getenv("GATEWAY_ALLOWED_USERS", "")
        if global_users:
            allowed = {_normalize_ship(s) for s in global_users.split(",") if s.strip()}
            if ship in allowed:
                return True
        if self.allowed_users and ship in self.allowed_users:
            return True

        rule = self.channel_rules.get(nest) or {}
        mode = rule.get("mode") or ("restricted" if self.owner_ship else "open")
        if mode == "open":
            return True

        allowed = set(self.default_authorized_ships)
        allowed.update(_normalize_ship(s) for s in rule.get("allowedShips", []) if isinstance(s, str))
        return ship in allowed

    async def _queue_approval(self, approval: PendingApproval) -> None:
        if not self.owner_ship:
            return
        self.pending_approvals = prune_expired(self.pending_approvals)
        if has_duplicate_pending(
            self.pending_approvals,
            approval_type=approval.type,
            requesting_ship=approval.requesting_ship,
            channel_nest=approval.channel_nest,
            group_flag=approval.group_flag,
        ):
            logger.info("[tlon] Approval already pending for %s", approval.requesting_ship)
            return

        try:
            result = await self.send(self.owner_ship, format_approval_request(approval))
            if result.success and result.message_id:
                approval.notification_message_id = normalize_notification_id(result.message_id)
            logger.info("[tlon] Queued approval %s for %s", approval.id, approval.requesting_ship)
        except Exception as e:
            logger.debug("[tlon] Failed to notify owner about approval %s: %s", approval.id, e)
        self.pending_approvals.append(approval)
        await self._put_settings_entry(
            "pendingApprovals",
            json.dumps([item.to_dict() for item in self.pending_approvals]),
        )

    async def _handle_owner_command(self, sender: str, text: str) -> Optional[str]:
        if not self._is_owner(sender) or not text.startswith("/"):
            return None

        parts = text.strip().split()
        command = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else None

        if command == "/pending":
            return format_pending_list(self.pending_approvals)
        if command in {"/blocked", "/banned"}:
            return format_blocked_list(list(self.blocked_ships))
        if command in {"/unban", "/unblock"}:
            if not arg:
                return "Usage: /unban ~ship"
            ship = _normalize_ship(arg)
            self.blocked_ships.discard(ship)
            self.dm_allowlist.discard(ship)
            return f"Unblocked {ship}."

        action_by_command = {
            "/allow": "approve",
            "/approve": "approve",
            "/reject": "deny",
            "/deny": "deny",
            "/ban": "block",
            "/block": "block",
        }
        action = action_by_command.get(command)
        if not action:
            return None

        approval = find_pending_approval(self.pending_approvals, arg)
        if not approval:
            if command in {"/approve", "/deny"}:
                return None
            return "No matching pending Tlon approval."

        return await self._execute_approval_action(approval, action)

    async def _handle_dm_reaction_event(self, event: Dict[str, Any], response: Dict[str, Any]) -> bool:
        """Handle OpenClaw-style approval reactions on owner DM prompts.

        Returns True when the event was a reaction event and should not fall
        through to normal message handling.
        """
        add_react = response.get("add-react")
        del_react = response.get("del-react")
        if not add_react and not del_react:
            return False
        if not isinstance(add_react, dict):
            return True

        react_author = _extract_author_ship(add_react.get("author") or add_react.get("ship"))
        react_emoji = str(add_react.get("react") or add_react.get("emoji") or "")
        if not react_author or react_author == self.ship_name:
            return True

        message_id = normalize_notification_id(event.get("id"))
        reaction_key = f"{message_id}:{react_author}:{react_emoji}"
        if reaction_key in self._processed_approval_reactions:
            return True
        self._processed_approval_reactions.add(reaction_key)
        if len(self._processed_approval_reactions) > self._max_processed:
            self._processed_approval_reactions = set(
                list(self._processed_approval_reactions)[-self._max_processed:]
            )

        action = emoji_to_approval_action(react_emoji)
        if not action:
            return True

        if not self._is_owner(react_author):
            logger.info(
                "[tlon] Ignoring approval reaction %s from non-owner %s",
                react_emoji,
                react_author,
            )
            return True

        pending = next(
            (
                approval
                for approval in prune_expired(self.pending_approvals)
                if approval.notification_message_id == message_id
            ),
            None,
        )
        if pending:
            logger.info(
                "[tlon] Reaction-based Tlon approval: %s -> %s for %s",
                react_emoji,
                action,
                pending.id,
            )
            confirmation = await self._execute_approval_action(pending, action)
            await self.send(self.owner_ship or react_author, confirmation)
            return True

        prompt = self._exec_approval_prompts.get(message_id)
        if not prompt:
            return True

        choice = "once" if action == "approve" else "deny"
        session_key = prompt.get("session_key", "")
        try:
            from tools.approval import resolve_gateway_approval

            count = resolve_gateway_approval(session_key, choice)
        except Exception as exc:
            logger.error("[tlon] Failed to resolve exec approval reaction: %s", exc)
            return True

        if count:
            self._exec_approval_prompts.pop(message_id, None)
            self._exec_approval_prompt_by_session.pop(session_key, None)
            logger.info(
                "[tlon] Reaction resolved %d exec approval(s) for session %s choice=%s",
                count,
                session_key,
                choice,
            )
            if choice == "once":
                await self.send(self.owner_ship or react_author, "Approved command once. Continuing.")
            else:
                await self.send(self.owner_ship or react_author, "Denied command. Continuing.")
        return True

    async def _execute_approval_action(self, approval: PendingApproval, action: str) -> str:
        if action == "approve":
            if approval.type == "dm":
                self.dm_allowlist.add(approval.requesting_ship)
                await self._put_settings_entry("dmAllowlist", sorted(self.dm_allowlist))
            elif approval.type == "channel" and approval.channel_nest:
                rule = dict(self.channel_rules.get(approval.channel_nest) or {})
                allowed = {
                    _normalize_ship(s)
                    for s in rule.get("allowedShips", [])
                    if isinstance(s, str)
                }
                allowed.add(approval.requesting_ship)
                rule["mode"] = "restricted"
                rule["allowedShips"] = sorted(allowed)
                self.channel_rules[approval.channel_nest] = rule
                await self._put_settings_entry("channelRules", json.dumps(self.channel_rules))
            elif approval.type == "group" and approval.group_flag and self._sse:
                await self._sse.poke(
                    app="groups",
                    mark="group-join",
                    json_data={"flag": approval.group_flag, "join-all": True},
                )
            await self._dispatch_pending_message(approval)
        elif action == "block":
            self.blocked_ships.add(approval.requesting_ship)
            self.dm_allowlist.discard(approval.requesting_ship)
            await self._put_settings_entry("dmAllowlist", sorted(self.dm_allowlist))

        self.pending_approvals = [
            item for item in self.pending_approvals if item.id != approval.id
        ]
        await self._put_settings_entry(
            "pendingApprovals",
            json.dumps([item.to_dict() for item in self.pending_approvals]),
        )
        return format_confirmation(approval, action)

    async def _dispatch_pending_message(self, approval: PendingApproval) -> None:
        raw = approval.original_message or {}
        if not raw:
            return
        source = self.build_source(
            chat_id=str(raw.get("chat_id") or approval.requesting_ship),
            chat_name=str(raw.get("chat_name") or raw.get("chat_id") or approval.requesting_ship),
            chat_type=str(raw.get("chat_type") or "dm"),
            user_id=approval.requesting_ship,
            user_name=approval.requesting_ship,
            thread_id=str(raw.get("thread_id")) if raw.get("thread_id") else None,
            parent_chat_id=(
                str(raw.get("parent_chat_id"))
                if raw.get("parent_chat_id")
                else None
            ),
        )
        event_obj = MessageEvent(
            text=str(raw.get("text") or ""),
            message_type=MessageType(str(raw.get("message_type") or MessageType.TEXT.value)),
            source=source,
            message_id=str(raw.get("message_id") or approval.id),
            reply_to_message_id=(
                str(raw.get("reply_to_message_id"))
                if raw.get("reply_to_message_id")
                else None
            ),
            timestamp=datetime.fromtimestamp(float(raw.get("timestamp") or time.time())),
            media_urls=list(raw.get("media_urls") or []),
            media_types=list(raw.get("media_types") or []),
        )
        await self.handle_message(event_obj)

    async def _handle_group_invites(self, foreigns: Dict[str, Any]) -> None:
        if not self._sse:
            return
        allowlist = {
            _normalize_ship(s)
            for s in (self._settings.group_invite_allowlist or [])
            if s
        }
        for invite in pending_group_invites(foreigns):
            inviter = _normalize_ship(
                str(invite.get("ship") or invite.get("inviter") or invite.get("invitedBy") or "")
            )
            group_flag = invite.get("groupFlag")
            if not isinstance(group_flag, str):
                continue
            if self._is_owner(inviter):
                await self._accept_group_invite(group_flag)
                continue
            if not self.auto_accept_group_invites:
                await self._queue_group_invite_approval(inviter, invite)
                continue
            if not allowlist:
                logger.info(
                    "[tlon] Rejected group invite from %s to %s (empty group invite allowlist)",
                    inviter or "(unknown)",
                    group_flag,
                )
                await self._queue_group_invite_approval(inviter, invite)
                continue
            if inviter not in allowlist:
                logger.info(
                    "[tlon] Rejected group invite from %s to %s (not in group invite allowlist)",
                    inviter or "(unknown)",
                    group_flag,
                )
                await self._queue_group_invite_approval(inviter, invite)
                continue
            await self._accept_group_invite(group_flag)

    async def _accept_group_invite(self, group_flag: str) -> None:
        if not self._sse:
            return
        try:
            await self._sse.poke(
                app="groups",
                mark="group-join",
                json_data={"flag": group_flag, "join-all": True},
            )
            logger.info("[tlon] Auto-accepted group invite to %s", group_flag)
        except Exception as e:
            logger.debug("[tlon] Failed to accept group invite %s: %s", group_flag, e)

    async def _queue_group_invite_approval(self, inviter: str, invite: Dict[str, Any]) -> None:
        if not self.owner_ship or not inviter:
            return
        group_flag = invite.get("groupFlag")
        if not isinstance(group_flag, str):
            return
        approval = create_pending_approval(
            approval_type="group",
            requesting_ship=inviter,
            group_flag=group_flag,
            group_title=(
                invite.get("groupTitle")
                if isinstance(invite.get("groupTitle"), str)
                else None
            ),
            existing_ids=[item.id for item in self.pending_approvals],
        )
        await self._queue_approval(approval)

    async def _handle_group_foreigns_event(self, event: Any) -> None:
        """Handle OpenClaw-compatible groups /v1/foreigns updates."""
        if not isinstance(event, dict):
            return
        await self._handle_group_invites(event)

    async def _handle_groups_ui_event(self, event: Any) -> None:
        """
        Handle OpenClaw-compatible groups /groups/ui updates.

        OpenClaw uses this stream to notice new channels after group joins or
        invite acceptance. It watches chat/heap channels from either
        ``event.channels`` or ``event.join.channels``.
        """
        if not isinstance(event, dict):
            return

        group_flag = self._groups_ui_group_flag(event)

        channels = event.get("channels")
        if isinstance(channels, dict):
            for channel_nest in channels.keys():
                await self._watch_group_channel(
                    channel_nest,
                    group_flag=group_flag,
                    reason="new channel (invite accepted)",
                    persist=self.auto_accept_group_invites,
                )

        join = event.get("join")
        if isinstance(join, dict):
            join_group = join.get("group")
            if isinstance(join_group, str):
                group_flag = join_group
            join_channels = join.get("channels")
            if isinstance(join_channels, list):
                for channel_nest in join_channels:
                    await self._watch_group_channel(
                        channel_nest,
                        group_flag=group_flag,
                        reason="joined channel",
                        persist=self.auto_accept_group_invites,
                    )

    @staticmethod
    def _groups_ui_group_flag(event: Dict[str, Any]) -> Optional[str]:
        flag = event.get("flag")
        if isinstance(flag, str):
            return flag
        group = event.get("group")
        if isinstance(group, dict):
            group_flag = group.get("flag") or group.get("id")
            if isinstance(group_flag, str):
                return group_flag
        return None

    async def _watch_group_channel(
        self,
        channel_nest: Any,
        *,
        group_flag: Optional[str],
        reason: str,
        persist: bool = False,
    ) -> bool:
        if not isinstance(channel_nest, str):
            return False
        if not channel_nest.startswith(("chat/", "heap/")):
            return False
        if group_flag:
            self._channel_to_group[channel_nest] = group_flag
        if channel_nest in self.monitored_channels:
            return False

        self.monitored_channels.add(channel_nest)
        logger.info("[tlon] Auto-detected %s: %s", reason, channel_nest)
        if persist:
            await self._persist_group_channel(channel_nest)
        return True

    async def _persist_group_channel(self, channel_nest: str) -> None:
        current = list(self._settings.group_channels or [])
        if channel_nest in current:
            return
        current.append(channel_nest)
        self._settings.group_channels = current
        await self._put_settings_entry("groupChannels", current)
        logger.info("[tlon] Persisted %s to settings store", channel_nest)

    async def _discover_channels(self) -> Set[str]:
        """Discover channels from groups the bot is a member of."""
        discovered = TlonDiscovery()
        for path in ("/groups-ui/v6/init.json", "/groups-ui/v7/init.json"):
            try:
                init_data = await self._sse.scry(path)
                discovered = parse_groups_ui_init(init_data)
                break
            except Exception as e:
                logger.debug("[tlon] groups-ui discovery failed at %s: %s", path, e)
        else:
            try:
                groups = await self._sse.scry("/groups/v1/groups.json")
                discovered = parse_legacy_groups(groups)
            except Exception as legacy_e:
                logger.debug("[tlon] legacy channel discovery failed: %s", legacy_e)

        self._channel_to_group.update(discovered.channel_to_group)
        self._group_names.update(discovered.group_names)

        if self.auto_accept_group_invites and discovered.foreigns:
            await self._handle_group_invites(discovered.foreigns)

        return discovered.channels

    async def _handle_channel_event(self, event: Any) -> None:
        """
        Handle a channels firehose (/v2) event.

        Event structure for new posts:
        {
          "nest": "chat/~host/channel",
          "response": {
            "post": {
              "id": "170141...",
              "r-post": {
                "set": {
                  "revision": "0",
                  "seal": { "id": "...", ... },
                  "essay": {
                    "author": "~ship",
                    "sent": 1773...,
                    "kind": "/chat",
                    "content": [{"inline": ["text"]}],
                    ...
                  },
                  "type": "post"
                }
              }
            }
          }
        }
        """
        try:
            if not isinstance(event, dict):
                return

            nest = event.get("nest")
            if not nest:
                return

            # Match OpenClaw: if the channels firehose delivers a chat/heap
            # event, the bot is in that channel, so watch it immediately.
            if nest not in self.monitored_channels:
                if nest.startswith(("chat/", "heap/")):
                    self.monitored_channels.add(nest)
                    logger.info("[tlon] Auto-watching channel from firehose: %s", nest)
                else:
                    return

            response = event.get("response")
            if not response:
                return

            # Extract post data
            post = response.get("post")
            if not post or not isinstance(post, dict):
                return

            msg_id = post.get("id")
            r_post = post.get("r-post", {})
            if not r_post:
                return

            # Two event shapes:
            # 1) Top-level post: r-post.set.essay  (type="post")
            # 2) Thread reply:   r-post.reply["r-reply"].set["reply-essay"]
            #    OpenClaw and current %channels use reply-essay; keep memo/essay
            #    as compatibility fallbacks for older or simplified fixtures.
            post_data = r_post.get("set") or {}
            essay = post_data.get("essay") if isinstance(post_data, dict) else None

            reply_data = r_post.get("reply")
            reply_memo = None
            reply_id = None
            is_thread_reply = False
            if reply_data and isinstance(reply_data, dict):
                reply_id = reply_data.get("id")
                r_reply = reply_data.get("r-reply", {})
                if r_reply:
                    reply_set = r_reply.get("set")
                    if reply_set and isinstance(reply_set, dict):
                        reply_memo = (
                            reply_set.get("reply-essay")
                            or reply_set.get("memo")
                            or reply_set.get("essay")
                        )
                        is_thread_reply = True

            content = reply_memo or essay
            if not content:
                return

            effective_id = reply_id if is_thread_reply else msg_id

            event_type = "reply" if is_thread_reply else "post"
            logger.info("[tlon] Channel event: nest=%s msg_id=%s type=%s is_reply=%s",
                        nest, msg_id, event_type, is_thread_reply)

            # Use lock to prevent race condition with concurrent event processing
            async with self._process_lock:
                if not effective_id or not self._mark_processed(str(effective_id)):
                    logger.info("[tlon] Channel dedup: skipping %s", effective_id)
                    return
                # Lock released after this block — but we've claimed the msg_id

            sender = _extract_author_ship(content.get("author"))
            if not sender or sender == self.ship_name:
                return

            # Get seal for thread context
            if is_thread_reply:
                reply_set = reply_data.get("r-reply", {}).get("set", {})
                seal = reply_set.get("seal", {}) if isinstance(reply_set, dict) else {}
            else:
                seal = post_data.get("seal", {}) if isinstance(post_data, dict) else {}
            parent_id = seal.get("parent-id") or seal.get("parent")

            raw_text = _extract_message_text(content.get("content"))
            effective_blob = content.get("blob")

            # Thread replies often arrive as memo events without blob metadata.
            # Use the v5 reply-essay scry OpenClaw uses before deciding the
            # event is empty.
            if is_thread_reply and not effective_blob and not raw_text.strip() and msg_id and effective_id:
                effective_blob = await self._fetch_reply_blob(nest, msg_id, effective_id)

            # Top-level file/image uploads can race the SSE event. If the
            # message has no text/blob yet, wait briefly and retry through v4,
            # which preserves essay.blob.
            if (
                not is_thread_reply
                and not effective_blob
                and not raw_text.strip()
                and effective_id
                and self._blob_retry_delay > 0
            ):
                await asyncio.sleep(self._blob_retry_delay)
                effective_blob = await self._fetch_post_blob(nest, effective_id)

            text, media_urls, media_types, message_type = await self._prepare_media_context(
                story_content=content.get("content"),
                blob=effective_blob,
                text=raw_text,
            )
            if not text.strip() and not media_urls:
                return

            logger.info("[tlon] Channel msg from %s in %s: %s",
                       sender, nest, text[:80])

            mentioned = self._is_bot_mentioned(text)
            owner_listen = self._should_owner_listen(sender, nest)

            # If a text message already triggers the bot, do one delayed blob
            # retry so captioned file/PDF uploads are visible to the agent too.
            if (
                not is_thread_reply
                and not effective_blob
                and raw_text.strip()
                and (mentioned or owner_listen)
                and self._blob_retry_delay > 0
            ):
                await asyncio.sleep(self._blob_retry_delay)
                retry_blob = await self._fetch_post_blob(nest, effective_id)
                if retry_blob:
                    effective_blob = retry_blob
                    text, media_urls, media_types, message_type = await self._prepare_media_context(
                        story_content=content.get("content"),
                        blob=effective_blob,
                        text=raw_text,
                    )
                    mentioned = self._is_bot_mentioned(text)

            thread_key = (nest, _normalize_post_id(parent_id)) if parent_id else None
            in_participated_thread = bool(thread_key and thread_key in self._participated_threads)
            owner_blob_only = bool(
                self._is_owner(sender)
                and (media_urls or effective_blob)
                and not raw_text.strip()
            )

            # In group channels, respond to mentions, participated threads, or
            # owner messages when owner-listen is enabled.
            if not (mentioned or in_participated_thread or owner_blob_only or owner_listen):
                logger.debug("[tlon] Not mentioned, ignoring")
                return

            # Check user authorization
            group_id = self._channel_to_group.get(nest)
            if not group_id and self._sse:
                try:
                    await self._discover_channels()
                    group_id = self._channel_to_group.get(nest)
                except Exception as exc:
                    logger.debug("[tlon] Group lookup refresh failed for %s: %s", nest, exc)
            group_name = self._group_names.get(group_id or "")

            if not self._is_channel_allowed(sender, nest):
                logger.info("[tlon] Unauthorized user %s in %s", sender, nest)
                if self.owner_ship:
                    approval = create_pending_approval(
                        approval_type="channel",
                        requesting_ship=sender,
                        channel_nest=nest,
                        existing_ids=[item.id for item in self.pending_approvals],
                        message_preview=text[:200],
                        original_message={
                            "chat_id": nest,
                            "chat_name": (_parse_channel_nest(nest) or {}).get("name", nest),
                            "chat_type": "group",
                            "parent_chat_id": group_id,
                            "text": self._strip_bot_mention(text) if mentioned else text,
                            "message_id": str(effective_id),
                            "reply_to_message_id": str(parent_id) if parent_id else None,
                            "thread_id": str(parent_id) if parent_id else None,
                            "timestamp": content.get("sent", time.time() * 1000) / 1000,
                            "media_urls": media_urls,
                            "media_types": media_types,
                            "message_type": message_type.value,
                        },
                    )
                    await self._queue_approval(approval)
                return

            # Strip bot mention from text
            clean_text = self._strip_bot_mention(text) if mentioned else text
            logger.info("[tlon] Processing message from %s: %s", sender, clean_text[:80])

            # Build message event
            parsed = _parse_channel_nest(nest)
            channel_name = parsed["name"] if parsed else nest
            chat_name = f"{group_name} / {channel_name}" if group_name else channel_name
            source = self.build_source(
                chat_id=nest,
                chat_name=chat_name,
                chat_type="group",
                user_id=sender,
                user_name=sender,
                thread_id=str(parent_id) if parent_id else None,
                parent_chat_id=group_id,
            )

            event_obj = MessageEvent(
                text=clean_text,
                message_type=message_type,
                source=source,
                message_id=str(effective_id),
                reply_to_message_id=str(parent_id) if parent_id else None,
                timestamp=datetime.fromtimestamp(
                    content.get("sent", time.time() * 1000) / 1000
                ),
                media_urls=media_urls,
                media_types=media_types,
            )

            if parent_id:
                self._participated_threads.add((nest, _normalize_post_id(parent_id)))
            await self.handle_message(event_obj)

        except Exception as e:
            logger.error("[tlon] Channel event error: %s", e, exc_info=True)

    async def _fetch_post_blob(self, nest: str, post_id: Any) -> Optional[str]:
        """Fetch a top-level post blob through channels v4.

        Tlon's firehose can arrive before upload metadata has propagated, and
        older/lightweight scries can strip blobs. OpenClaw uses channels v4
        around/post scries here because they preserve essay.blob.
        """
        if not self._sse or not nest or not post_id:
            return None
        formatted_id = self._format_scry_post_id(post_id)
        if not formatted_id:
            return None
        try:
            data = await self._sse.scry(
                f"/channels/v4/{nest}/posts/around/{formatted_id}/1/post"
            )
        except Exception as e:
            logger.debug("[tlon] Blob post scry failed for %s/%s: %s", nest, post_id, e)
            return None

        posts = self._posts_from_scry_response(data)
        wanted = _normalize_post_id(post_id)
        for post in posts:
            seal = post.get("seal") if isinstance(post, dict) else {}
            essay = post.get("essay") if isinstance(post, dict) else None
            if not isinstance(essay, dict):
                r_post = post.get("r-post", {}) if isinstance(post, dict) else {}
                set_data = r_post.get("set", {}) if isinstance(r_post, dict) else {}
                essay = set_data.get("essay") if isinstance(set_data, dict) else None
                seal = set_data.get("seal", seal) if isinstance(set_data, dict) else seal
            seal_id = _normalize_post_id((seal or {}).get("id") if isinstance(seal, dict) else "")
            if wanted and seal_id and seal_id != wanted and len(posts) != 1:
                continue
            blob = essay.get("blob") if isinstance(essay, dict) else None
            if blob is not None:
                return blob if isinstance(blob, str) else json.dumps(blob)
        return None

    async def _fetch_reply_blob(self, nest: str, parent_id: Any, reply_id: Any) -> Optional[str]:
        """Fetch a thread reply blob through channels v5 reply essays."""
        if not self._sse or not nest or not parent_id or not reply_id:
            return None
        formatted_parent = self._format_scry_post_id(parent_id)
        if not formatted_parent:
            return None
        try:
            data = await self._sse.scry(
                f"/channels/v5/{nest}/posts/post/id/{formatted_parent}/replies/newest/5"
            )
        except Exception as e:
            logger.debug(
                "[tlon] Blob reply scry failed for %s/%s/%s: %s",
                nest,
                parent_id,
                reply_id,
                e,
            )
            return None

        replies = self._posts_from_scry_response(data)
        wanted = _normalize_post_id(reply_id)
        for reply in replies:
            if not isinstance(reply, dict):
                continue
            seal = reply.get("seal") or {}
            essay = reply.get("reply-essay") or reply.get("memo")
            r_reply = reply.get("r-reply")
            if isinstance(r_reply, dict):
                set_data = r_reply.get("set", {})
                if isinstance(set_data, dict):
                    essay = essay or set_data.get("reply-essay") or set_data.get("memo")
                    seal = seal or set_data.get("seal", {})
            seal_id = _normalize_post_id(seal.get("id") if isinstance(seal, dict) else "")
            if wanted and seal_id and seal_id != wanted and len(replies) != 1:
                continue
            blob = essay.get("blob") if isinstance(essay, dict) else None
            if blob is not None:
                return blob if isinstance(blob, str) else json.dumps(blob)
        return None

    @staticmethod
    def _posts_from_scry_response(data: Any) -> List[Dict[str, Any]]:
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        if not isinstance(data, dict):
            return []
        for key in ("posts", "replies", "writs"):
            value = data.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
            if isinstance(value, dict):
                return [item for item in value.values() if isinstance(item, dict)]
        return [item for item in data.values() if isinstance(item, dict)]

    @staticmethod
    def _format_scry_post_id(post_id: Any) -> str:
        bare = str(post_id or "").split("/")[-1].replace(".", "")
        if bare.isdigit():
            return _format_ud(int(bare))
        return str(post_id or "")

    async def _handle_dm_event(self, event: Any) -> None:
        """Handle a chat firehose (/v3) event."""
        try:
            logger.info("[tlon] _handle_dm_event called, keys=%s",
                        list(event.keys()) if isinstance(event, dict) else type(event).__name__)
            # Handle DM invite arrays
            if isinstance(event, list):
                for invite in event:
                    if isinstance(invite, dict):
                        ship = _normalize_ship(str(invite.get("ship") or ""))
                    elif isinstance(invite, str):
                        ship = _normalize_ship(invite)
                    else:
                        continue
                    if not ship or ship in self._processed_dm_invites:
                        continue
                    if ship and (
                        self.auto_accept_dm_invites
                        or self._is_owner(ship)
                        or self._is_user_allowed(ship, is_dm=True)
                    ):
                        try:
                            await self._sse.poke(
                                app="chat",
                                mark="chat-dm-rsvp",
                                json_data={"ship": ship.lstrip("~"), "ok": True},
                            )
                            logger.info("[tlon] Auto-accepted DM invite from %s", ship)
                        except Exception as e:
                            logger.error("[tlon] Failed to accept DM from %s: %s", ship, e)
                        self._processed_dm_invites.add(ship)
                    elif self.owner_ship:
                        approval = create_pending_approval(
                            approval_type="dm",
                            requesting_ship=ship,
                            existing_ids=[item.id for item in self.pending_approvals],
                            message_preview="(DM invite - no message yet)",
                        )
                        await self._queue_approval(approval)
                        self._processed_dm_invites.add(ship)
                return

            if not isinstance(event, dict):
                return

            if "whom" not in event or "response" not in event:
                return

            whom = event["whom"]
            msg_id = event.get("id")
            response = event["response"]
            if isinstance(response, dict) and await self._handle_dm_reaction_event(event, response):
                return

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
                effective_id = (
                    dm_reply.get("id")
                    or dm_reply.get("delta", {}).get("add", {}).get("id")
                    or (
                        f"{_extract_author_ship(content.get('author'))}/{_da_from_unix(content.get('sent'))}"
                        if content.get("author") and content.get("sent")
                        else msg_id
                    )
                )

            async with self._process_lock:
                if not effective_id or not self._mark_processed(str(effective_id)):
                    return

            sender = _extract_author_ship(content.get("author"))
            # Extract DM partner from whom field
            partner = _normalize_ship(whom) if isinstance(whom, str) else ""

            logger.info("[tlon] DM event: whom=%s, sender=%s, self=%s",
                        whom, sender, self.ship_name)

            # Skip our own messages (author == us)
            if sender == self.ship_name:
                logger.info("[tlon] Skipping own DM message")
                return

            # Use partner for routing, author for identity
            effective_sender = partner or sender
            if not effective_sender:
                return

            raw_text = _extract_message_text(content.get("content"))
            text, media_urls, media_types, message_type = await self._prepare_media_context(
                story_content=content.get("content"),
                blob=content.get("blob"),
                text=raw_text,
            )
            if not text.strip() and not media_urls:
                return

            owner_command_response = await self._handle_owner_command(sender, text)
            if owner_command_response is not None:
                await self.send(effective_sender, owner_command_response, reply_to=msg_id)
                return

            # Check DM authorization (includes dm_allowlist)
            if not self._is_user_allowed(effective_sender, is_dm=True):
                logger.info("[tlon] Unauthorized DM from %s", effective_sender)
                if self.owner_ship:
                    approval = create_pending_approval(
                        approval_type="dm",
                        requesting_ship=effective_sender,
                        existing_ids=[item.id for item in self.pending_approvals],
                        message_preview=text[:200],
                        original_message={
                            "chat_id": effective_sender,
                            "chat_name": effective_sender,
                            "chat_type": "dm",
                            "text": text,
                            "message_id": str(effective_id),
                            "reply_to_message_id": str(msg_id) if is_thread_reply else None,
                            "thread_id": str(msg_id) if is_thread_reply else None,
                            "timestamp": content.get("sent", time.time() * 1000) / 1000,
                            "media_urls": media_urls,
                            "media_types": media_types,
                            "message_type": message_type.value,
                        },
                    )
                    await self._queue_approval(approval)
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
                message_type=message_type,
                source=source,
                message_id=str(effective_id),
                reply_to_message_id=str(msg_id) if is_thread_reply else None,
                timestamp=datetime.fromtimestamp(
                    content.get("sent", time.time() * 1000) / 1000
                ),
                media_urls=media_urls,
                media_types=media_types,
            )

            logger.info("[tlon] >>> Calling handle_message for DM from %s, msg_id=%s", effective_sender, effective_id)
            await self.handle_message(event_obj)

        except Exception as e:
            logger.error("[tlon] DM event error: %s", e, exc_info=True)

    def _is_user_allowed(self, ship: str, is_dm: bool = False) -> bool:
        """Check if a ship is authorized to interact with the bot."""
        ship = _normalize_ship(ship)
        if self._is_blocked(ship):
            return False
        if self._is_owner(ship):
            return True

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

        if self.default_authorized_ships and ship in self.default_authorized_ships:
            return True

        # Check DM-specific allowlist
        if is_dm and self.dm_allowlist and ship in self.dm_allowlist:
            return True

        if is_dm and self.owner_ship:
            return False

        # If no allowlists configured at all, allow (open by default)
        if (
            not self.allowed_users
            and not global_users
            and not self.dm_allowlist
            and not self.default_authorized_ships
        ):
            return True

        return False

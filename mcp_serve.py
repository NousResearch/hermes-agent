"""
Hermes MCP Server — expose messaging conversations as MCP tools.

Starts a stdio MCP server that lets any MCP client (Claude Code, Cursor, Codex,
etc.) list conversations, read message history, send messages, poll for live
events, and manage approval requests across all connected platforms.

Matches OpenClaw's 9-tool MCP channel bridge surface:
  conversations_list, conversation_get, messages_read, attachments_fetch,
  events_poll, events_wait, messages_send, permissions_list_open,
  permissions_respond

Plus: channels_list (Hermes-specific extra)

Usage:
    hermes mcp serve
    hermes mcp serve --verbose

MCP client config (e.g. claude_desktop_config.json):
    {
        "mcpServers": {
            "hermes": {
                "command": "hermes",
                "args": ["mcp", "serve"]
            }
        }
    }
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

logger = logging.getLogger("hermes.mcp_serve")

# ---------------------------------------------------------------------------
# Lazy MCP SDK import
# ---------------------------------------------------------------------------

_MCP_SERVER_AVAILABLE = False
try:
    from mcp.server.fastmcp import FastMCP

    _MCP_SERVER_AVAILABLE = True
except ImportError:
    FastMCP = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_sessions_dir() -> Path:
    """Return the sessions directory using HERMES_HOME."""
    try:
        from hermes_constants import get_hermes_home
        return get_hermes_home() / "sessions"
    except ImportError:
        return Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes")) / "sessions"


def _get_session_db():
    """Get a SessionDB instance for reading message transcripts."""
    try:
        from hermes_state import SessionDB
        return SessionDB()
    except Exception as e:
        logger.debug("SessionDB unavailable: %s", e)
        return None


def _load_sessions_index() -> dict:
    """Load the gateway session routing index.

    Returns a dict of session_key -> entry_dict with platform routing info.

    state.db is the primary source (#9006): gateway sessions persist their
    routing metadata (session_key, chat/thread ids, display_name, origin) on
    the durable session row, so a single database read replaces the old
    dual-file sessions.json dependency.  Falls back to sessions.json for
    pre-migration databases where no gateway rows carry a session_key yet.
    """
    entries = _load_sessions_index_from_db()
    if entries:
        return entries
    return _load_sessions_index_from_json()


def _row_to_index_entry(row: dict) -> dict:
    """Convert a state.db gateway session row to the sessions.json entry shape."""
    origin = {}
    origin_json = row.get("origin_json")
    if origin_json:
        try:
            parsed = json.loads(origin_json)
            if isinstance(parsed, dict):
                origin = parsed
        except (TypeError, ValueError):
            pass
    if not origin:
        # Pre-origin_json rows: synthesize the minimal origin from columns.
        origin = {
            "platform": row.get("source", ""),
            "chat_id": row.get("chat_id"),
            "chat_type": row.get("chat_type"),
            "thread_id": row.get("thread_id"),
            "user_id": row.get("user_id"),
        }

    def _iso(ts) -> str:
        try:
            return datetime.fromtimestamp(float(ts)).isoformat() if ts else ""
        except (TypeError, ValueError, OSError):
            return ""

    input_tokens = int(row.get("input_tokens") or 0)
    output_tokens = int(row.get("output_tokens") or 0)
    return {
        "session_id": str(row.get("id", "")),
        "session_key": row.get("session_key", ""),
        "platform": row.get("source", ""),
        "chat_type": row.get("chat_type") or origin.get("chat_type", ""),
        "display_name": row.get("display_name") or origin.get("chat_name") or "",
        "origin": origin,
        "created_at": _iso(row.get("started_at")),
        "updated_at": _iso(row.get("last_active") or row.get("started_at")),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


def _load_sessions_index_from_db() -> dict:
    """Build the routing index from state.db gateway session rows."""
    db = _get_session_db()
    if db is None:
        return {}
    try:
        lister = getattr(db, "list_gateway_sessions", None)
        if not callable(lister):
            return {}
        rows = lister(active_only=True)
        entries = {}
        for row in rows:
            key = row.get("session_key")
            if not key:
                continue
            entries[key] = _row_to_index_entry(row)
        return entries
    except Exception as e:
        logger.debug("Failed to load gateway sessions from state.db: %s", e)
        return {}
    finally:
        try:
            db.close()
        except Exception:
            pass


def _load_sessions_index_from_json() -> dict:
    """Legacy fallback: load the gateway sessions.json index directly.

    Used only for pre-migration databases whose gateway rows don't carry a
    session_key yet.  This avoids importing the full SessionStore which
    needs GatewayConfig.
    """
    sessions_file = _get_sessions_dir() / "sessions.json"
    if not sessions_file.exists():
        return {}
    try:
        with open(sessions_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Drop documentation/metadata sentinels (keys starting with "_", e.g.
        # the "_README" note the gateway writes into the index). They are not
        # session entries and would break consumers that treat every value as
        # an entry dict.
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if not str(k).startswith("_")}
        return {}
    except Exception as e:
        logger.debug("Failed to load sessions.json: %s", e)
        return {}


def _load_channel_directory() -> dict:
    """Load the cached channel directory for available targets."""
    try:
        from hermes_constants import get_hermes_home
        directory_file = get_hermes_home() / "channel_directory.json"
    except ImportError:
        directory_file = Path(
            os.environ.get("HERMES_HOME", Path.home() / ".hermes")
        ) / "channel_directory.json"

    if not directory_file.exists():
        return {}
    try:
        with open(directory_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.debug("Failed to load channel_directory.json: %s", e)
        return {}


def _coerce_int(
    value,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    """Coerce value to int with fallback and clamping.

    Used at MCP tool boundaries to handle invalid types from external clients.
    Returns default if value cannot be converted to int.
    """
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        coerced = default
    return max(minimum, min(coerced, maximum))


def _extract_message_content(msg: dict) -> str:
    """Extract text content from a message, handling multi-part content."""
    content = msg.get("content", "")
    if isinstance(content, list):
        text_parts = [
            p.get("text", "") for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        ]
        return "\n".join(text_parts)
    return str(content) if content else ""


def _extract_attachments(msg: dict) -> List[dict]:
    """Extract non-text attachments from a message.

    Finds: multi-part image/file content blocks, MEDIA: tags in text,
    image URLs, and file references.
    """
    attachments = []
    content = msg.get("content", "")

    # Multi-part content blocks (image_url, file, etc.)
    if isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type", "")
            if ptype == "image_url":
                url = part.get("image_url", {}).get("url", "") if isinstance(part.get("image_url"), dict) else ""
                if url:
                    attachments.append({"type": "image", "url": url})
            elif ptype == "image":
                url = part.get("url", part.get("source", {}).get("url", ""))
                if url:
                    attachments.append({"type": "image", "url": url})
            elif ptype not in {"text",}:
                # Unknown non-text content type
                attachments.append({"type": ptype, "data": part})

    # MEDIA: tags in text content
    text = _extract_message_content(msg)
    if text:
        media_pattern = re.compile(r'MEDIA:\s*(\S+)')
        for match in media_pattern.finditer(text):
            path = match.group(1)
            attachments.append({"type": "media", "path": path})

    return attachments


# ---------------------------------------------------------------------------
# Event Bridge — polls SessionDB for new messages, maintains event queue
# ---------------------------------------------------------------------------

QUEUE_LIMIT = 1000
POLL_INTERVAL = 0.2  # seconds between DB polls (200ms)


@dataclass
class QueueEvent:
    """An event in the bridge's in-memory queue."""
    cursor: int
    type: str  # "message", "approval_requested", "approval_resolved"
    session_key: str = ""
    data: dict = field(default_factory=dict)


class EventBridge:
    """Background poller that watches SessionDB for new messages and
    maintains an in-memory event queue with waiter support.

    This is the Hermes equivalent of OpenClaw's WebSocket gateway bridge.
    Instead of WebSocket events, we poll the SQLite database for changes.
    """

    def __init__(self):
        self._queue: List[QueueEvent] = []
        self._cursor = 0
        self._lock = threading.Lock()
        self._new_event = threading.Event()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_poll_timestamps: Dict[str, float] = {}  # session_key -> unix timestamp
        # In-memory approval tracking (populated from events)
        self._pending_approvals: Dict[str, dict] = {}
        # mtime cache — skip expensive work when state.db hasn't changed
        self._state_db_mtime: float = 0.0
        self._cached_sessions_index: dict = {}

    def start(self):
        """Start the background polling thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        logger.debug("EventBridge started")

    def stop(self):
        """Stop the background polling thread."""
        self._running = False
        self._new_event.set()  # Wake any waiters
        if self._thread:
            self._thread.join(timeout=5)
        logger.debug("EventBridge stopped")

    def poll_events(
        self,
        after_cursor: int = 0,
        session_key: Optional[str] = None,
        limit: int = 20,
    ) -> dict:
        """Return events since after_cursor, optionally filtered by session_key."""
        with self._lock:
            events = [
                e for e in self._queue
                if e.cursor > after_cursor
                and (not session_key or e.session_key == session_key)
            ][:limit]

        next_cursor = events[-1].cursor if events else after_cursor
        return {
            "events": [
                {"cursor": e.cursor, "type": e.type,
                 "session_key": e.session_key, **e.data}
                for e in events
            ],
            "next_cursor": next_cursor,
        }

    def wait_for_event(
        self,
        after_cursor: int = 0,
        session_key: Optional[str] = None,
        timeout_ms: int = 30000,
    ) -> Optional[dict]:
        """Block until a matching event arrives or timeout expires."""
        deadline = time.monotonic() + (timeout_ms / 1000.0)

        while time.monotonic() < deadline:
            with self._lock:
                for e in self._queue:
                    if e.cursor > after_cursor and (
                        not session_key or e.session_key == session_key
                    ):
                        return {
                            "cursor": e.cursor, "type": e.type,
                            "session_key": e.session_key, **e.data,
                        }

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            self._new_event.clear()
            self._new_event.wait(timeout=min(remaining, POLL_INTERVAL))

        return None

    def list_pending_approvals(self) -> List[dict]:
        """List approval requests observed during this bridge session."""
        with self._lock:
            return sorted(
                self._pending_approvals.values(),
                key=lambda a: a.get("created_at", ""),
            )

    def respond_to_approval(self, approval_id: str, decision: str) -> dict:
        """Resolve a pending approval (best-effort without gateway IPC)."""
        with self._lock:
            approval = self._pending_approvals.pop(approval_id, None)

        if not approval:
            return {"error": f"Approval not found: {approval_id}"}

        self._enqueue(QueueEvent(
            cursor=0,  # Will be set by _enqueue
            type="approval_resolved",
            session_key=approval.get("session_key", ""),
            data={"approval_id": approval_id, "decision": decision},
        ))

        return {"resolved": True, "approval_id": approval_id, "decision": decision}

    def _enqueue(self, event: QueueEvent) -> None:
        """Add an event to the queue and wake any waiters."""
        with self._lock:
            self._cursor += 1
            event.cursor = self._cursor
            self._queue.append(event)
            # Trim queue to limit
            while len(self._queue) > QUEUE_LIMIT:
                self._queue.pop(0)
        self._new_event.set()

    def _poll_loop(self):
        """Background loop: poll SessionDB for new messages."""
        db = _get_session_db()
        if not db:
            logger.warning("EventBridge: SessionDB unavailable, event polling disabled")
            return

        while self._running:
            try:
                self._poll_once(db)
            except Exception as e:
                logger.debug("EventBridge poll error: %s", e)
            time.sleep(POLL_INTERVAL)

    def _poll_once(self, db):
        """Check for new messages across all sessions.

        Uses a single mtime check on state.db to skip work when nothing
        has changed — makes 200ms polling essentially free.  Since #9006
        the routing index itself lives in state.db (session rows carry
        session_key/origin metadata), so a new conversation and its first
        message land in the SAME file and one mtime check covers both —
        eliminating the old dual-file (sessions.json + state.db) race that
        could drop brand-new conversations (#8925).
        """
        try:
            from hermes_constants import get_hermes_home
            db_file = get_hermes_home() / "state.db"
        except ImportError:
            db_file = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes")) / "state.db"

        try:
            db_mtime = db_file.stat().st_mtime if db_file.exists() else 0.0
        except OSError:
            db_mtime = 0.0

        if db_mtime == self._state_db_mtime:
            return  # Nothing changed since last poll — skip entirely

        self._state_db_mtime = db_mtime
        # Refresh the routing index from state.db on every change tick —
        # it's a single indexed query and it can never lag the messages
        # table (both live in the same database file).
        self._cached_sessions_index = _load_sessions_index()
        entries = self._cached_sessions_index

        for session_key, entry in entries.items():
            session_id = entry.get("session_id", "")
            if not session_id:
                continue

            last_seen = self._last_poll_timestamps.get(session_key, 0.0)

            try:
                messages = db.get_messages(session_id)
            except Exception:
                continue

            if not messages:
                continue

            # Normalize timestamps to float for comparison
            def _ts_float(ts) -> float:
                if isinstance(ts, (int, float)):
                    return float(ts)
                if isinstance(ts, str) and ts:
                    try:
                        return float(ts)
                    except ValueError:
                        # ISO string — parse to epoch
                        try:
                            from datetime import datetime
                            return datetime.fromisoformat(ts).timestamp()
                        except Exception:
                            return 0.0
                return 0.0

            # Find messages newer than our last seen timestamp
            new_messages = []
            for msg in messages:
                ts = _ts_float(msg.get("timestamp", 0))
                role = msg.get("role", "")
                if role not in {"user", "assistant"}:
                    continue
                if ts > last_seen:
                    new_messages.append(msg)

            for msg in new_messages:
                content = _extract_message_content(msg)
                if not content:
                    continue
                self._enqueue(QueueEvent(
                    cursor=0,
                    type="message",
                    session_key=session_key,
                    data={
                        "role": msg.get("role", ""),
                        "content": content[:500],
                        "timestamp": str(msg.get("timestamp", "")),
                        "message_id": str(msg.get("id", "")),
                    },
                ))

            # Update last seen to the most recent message timestamp
            all_ts = [_ts_float(m.get("timestamp", 0)) for m in messages]
            if all_ts:
                latest = max(all_ts)
                if latest > last_seen:
                    self._last_poll_timestamps[session_key] = latest


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

def create_mcp_server(event_bridge: Optional[EventBridge] = None) -> "FastMCP":
    """Create and return the Hermes MCP server with all tools registered."""
    if not _MCP_SERVER_AVAILABLE:
        raise ImportError(
            "MCP server requires the 'mcp' package. "
            f"Install with: {sys.executable} -m pip install 'mcp'"
        )

    mcp = FastMCP(
        "hermes",
        instructions=(
            "Hermes Agent messaging bridge. Use these tools to interact with "
            "conversations across Telegram, Discord, Slack, WhatsApp, Signal, "
            "Matrix, and other connected platforms."
        ),
    )

    bridge = event_bridge or EventBridge()

    # -- conversations_list ------------------------------------------------

    @mcp.tool()
    def conversations_list(
        platform: Optional[str] = None,
        limit: int = 50,
        search: Optional[str] = None,
    ) -> str:
        """List active messaging conversations across connected platforms.

        Returns conversations with their session keys (needed for messages_read),
        platform, chat type, display name, and last activity time.

        Args:
            platform: Filter by platform name (telegram, discord, slack, etc.)
            limit: Maximum number of conversations to return (default 50)
            search: Optional text to filter conversations by name
        """
        limit = _coerce_int(limit, default=50, minimum=1, maximum=200)
        entries = _load_sessions_index()
        conversations = []

        for key, entry in entries.items():
            origin = entry.get("origin", {})
            entry_platform = entry.get("platform") or origin.get("platform", "")

            if platform and entry_platform.lower() != platform.lower():
                continue

            display_name = entry.get("display_name", "")
            chat_name = origin.get("chat_name", "")
            if search:
                search_lower = search.lower()
                if (search_lower not in display_name.lower()
                        and search_lower not in chat_name.lower()
                        and search_lower not in key.lower()):
                    continue

            conversations.append({
                "session_key": key,
                "session_id": entry.get("session_id", ""),
                "platform": entry_platform,
                "chat_type": entry.get("chat_type", origin.get("chat_type", "")),
                "display_name": display_name,
                "chat_name": chat_name,
                "user_name": origin.get("user_name", ""),
                "updated_at": entry.get("updated_at", ""),
            })

        conversations.sort(key=lambda c: c.get("updated_at", ""), reverse=True)
        conversations = conversations[:limit]

        return json.dumps({
            "count": len(conversations),
            "conversations": conversations,
        }, indent=2)

    # -- conversation_get --------------------------------------------------

    @mcp.tool()
    def conversation_get(session_key: str) -> str:
        """Get detailed info about one conversation by its session key.

        Args:
            session_key: The session key from conversations_list
        """
        entries = _load_sessions_index()
        entry = entries.get(session_key)

        if not entry:
            return json.dumps({"error": f"Conversation not found: {session_key}"})

        origin = entry.get("origin", {})
        return json.dumps({
            "session_key": session_key,
            "session_id": entry.get("session_id", ""),
            "platform": entry.get("platform") or origin.get("platform", ""),
            "chat_type": entry.get("chat_type", origin.get("chat_type", "")),
            "display_name": entry.get("display_name", ""),
            "user_name": origin.get("user_name", ""),
            "chat_name": origin.get("chat_name", ""),
            "chat_id": origin.get("chat_id", ""),
            "thread_id": origin.get("thread_id"),
            "updated_at": entry.get("updated_at", ""),
            "created_at": entry.get("created_at", ""),
            "input_tokens": entry.get("input_tokens", 0),
            "output_tokens": entry.get("output_tokens", 0),
            "total_tokens": entry.get("total_tokens", 0),
        }, indent=2)

    # -- messages_read -----------------------------------------------------

    @mcp.tool()
    def messages_read(
        session_key: str,
        limit: int = 50,
    ) -> str:
        """Read recent messages from a conversation.

        Returns the message history in chronological order with role, content,
        and timestamp for each message.

        Args:
            session_key: The session key from conversations_list
            limit: Maximum number of messages to return (default 50, most recent)
        """
        limit = _coerce_int(limit, default=50, minimum=1, maximum=200)
        entries = _load_sessions_index()
        entry = entries.get(session_key)
        if not entry:
            return json.dumps({"error": f"Conversation not found: {session_key}"})

        session_id = entry.get("session_id", "")
        if not session_id:
            return json.dumps({"error": "No session ID for this conversation"})

        db = _get_session_db()
        if not db:
            return json.dumps({"error": "Session database unavailable"})

        try:
            all_messages = db.get_messages(session_id)
        except Exception as e:
            return json.dumps({"error": f"Failed to read messages: {e}"})

        filtered = []
        for msg in all_messages:
            role = msg.get("role", "")
            if role in {"user", "assistant"}:
                content = _extract_message_content(msg)
                if content:
                    filtered.append({
                        "id": str(msg.get("id", "")),
                        "role": role,
                        "content": content[:2000],
                        "timestamp": msg.get("timestamp", ""),
                    })

        messages = filtered[-limit:]

        return json.dumps({
            "session_key": session_key,
            "count": len(messages),
            "total_in_session": len(filtered),
            "messages": messages,
        }, indent=2)

    # -- attachments_fetch -------------------------------------------------

    @mcp.tool()
    def attachments_fetch(
        session_key: str,
        message_id: str,
    ) -> str:
        """List non-text attachments for a message in a conversation.

        Extracts images, media files, and other non-text content blocks
        from the specified message.

        Args:
            session_key: The session key from conversations_list
            message_id: The message ID from messages_read
        """
        entries = _load_sessions_index()
        entry = entries.get(session_key)
        if not entry:
            return json.dumps({"error": f"Conversation not found: {session_key}"})

        session_id = entry.get("session_id", "")
        if not session_id:
            return json.dumps({"error": "No session ID for this conversation"})

        db = _get_session_db()
        if not db:
            return json.dumps({"error": "Session database unavailable"})

        try:
            all_messages = db.get_messages(session_id)
        except Exception as e:
            return json.dumps({"error": f"Failed to read messages: {e}"})

        # Find the target message
        target_msg = None
        for msg in all_messages:
            if str(msg.get("id", "")) == message_id:
                target_msg = msg
                break

        if not target_msg:
            return json.dumps({"error": f"Message not found: {message_id}"})

        attachments = _extract_attachments(target_msg)

        return json.dumps({
            "message_id": message_id,
            "count": len(attachments),
            "attachments": attachments,
        }, indent=2)

    # -- events_poll -------------------------------------------------------

    @mcp.tool()
    def events_poll(
        after_cursor: int = 0,
        session_key: Optional[str] = None,
        limit: int = 20,
    ) -> str:
        """Poll for new conversation events since a cursor position.

        Returns events that have occurred since the given cursor. Use the
        returned next_cursor value for subsequent polls.

        Event types: message, approval_requested, approval_resolved

        Args:
            after_cursor: Return events after this cursor (0 for all)
            session_key: Optional filter to one conversation
            limit: Maximum events to return (default 20)
        """
        after_cursor = _coerce_int(after_cursor, default=0, minimum=0, maximum=10**18)
        limit = _coerce_int(limit, default=20, minimum=1, maximum=200)
        result = bridge.poll_events(
            after_cursor=after_cursor,
            session_key=session_key,
            limit=limit,
        )
        return json.dumps(result, indent=2)

    # -- events_wait -------------------------------------------------------

    @mcp.tool()
    def events_wait(
        after_cursor: int = 0,
        session_key: Optional[str] = None,
        timeout_ms: int = 30000,
    ) -> str:
        """Wait for the next conversation event (long-poll).

        Blocks until a matching event arrives or the timeout expires.
        Use this for near-real-time event delivery without polling.

        Args:
            after_cursor: Wait for events after this cursor
            session_key: Optional filter to one conversation
            timeout_ms: Maximum wait time in milliseconds (default 30000)
        """
        after_cursor = _coerce_int(after_cursor, default=0, minimum=0, maximum=10**18)
        timeout_ms = _coerce_int(
            timeout_ms,
            default=30000,
            minimum=0,
            maximum=300000,
        )  # Cap at 5 minutes
        event = bridge.wait_for_event(
            after_cursor=after_cursor,
            session_key=session_key,
            timeout_ms=timeout_ms,
        )
        if event:
            return json.dumps({"event": event}, indent=2)
        return json.dumps({"event": None, "reason": "timeout"}, indent=2)

    # -- messages_send -----------------------------------------------------

    @mcp.tool()
    def messages_send(
        target: str,
        message: str,
    ) -> str:
        """Send a message to a platform conversation.

        The target format is "platform:chat_id" — same format used by the
        channels_list tool. You can also use human-friendly channel names
        that will be resolved automatically.

        Examples:
            target="telegram:6308981865"
            target="discord:#general"
            target="slack:#engineering"

        Args:
            target: Platform target in "platform:identifier" format
            message: The message text to send
        """
        if not target or not message:
            return json.dumps({"error": "Both target and message are required"})

        try:
            from tools.send_message_tool import send_message_tool
            result_str = send_message_tool(
                {"action": "send", "target": target, "message": message}
            )
            return result_str
        except ImportError:
            return json.dumps({"error": "Send message tool not available"})
        except Exception as e:
            return json.dumps({"error": f"Send failed: {e}"})

    # -- channels_list -----------------------------------------------------

    @mcp.tool()
    def channels_list(platform: Optional[str] = None) -> str:
        """List available messaging channels and targets across platforms.

        Returns channels that you can send messages to. The target strings
        returned here can be used directly with the messages_send tool.

        Args:
            platform: Filter by platform name (telegram, discord, slack, etc.)
        """
        directory = _load_channel_directory()
        if not directory:
            entries = _load_sessions_index()
            targets = []
            seen = set()
            for key, entry in entries.items():
                origin = entry.get("origin", {})
                p = entry.get("platform") or origin.get("platform", "")
                chat_id = origin.get("chat_id", "")
                if not p or not chat_id:
                    continue
                if platform and p.lower() != platform.lower():
                    continue
                target_str = f"{p}:{chat_id}"
                if target_str in seen:
                    continue
                seen.add(target_str)
                targets.append({
                    "target": target_str,
                    "platform": p,
                    "name": entry.get("display_name") or origin.get("chat_name", ""),
                    "chat_type": entry.get("chat_type", origin.get("chat_type", "")),
                })
            return json.dumps({"count": len(targets), "channels": targets}, indent=2)

        channels = []
        for plat, entries_list in directory.get("platforms", {}).items():
            if platform and plat.lower() != platform.lower():
                continue
            if isinstance(entries_list, list):
                for ch in entries_list:
                    if isinstance(ch, dict):
                        chat_id = ch.get("id", ch.get("chat_id", ""))
                        channels.append({
                            "target": f"{plat}:{chat_id}" if chat_id else plat,
                            "platform": plat,
                            "name": ch.get("name", ch.get("display_name", "")),
                            "chat_type": ch.get("type", ""),
                        })

        return json.dumps({"count": len(channels), "channels": channels}, indent=2)

    # -- permissions_list_open ---------------------------------------------

    @mcp.tool()
    def permissions_list_open() -> str:
        """List pending approval requests observed during this bridge session.

        Returns exec and plugin approval requests that the bridge has seen
        since it started. Approvals are live-session only — older approvals
        from before the bridge connected are not included.
        """
        approvals = bridge.list_pending_approvals()
        return json.dumps({
            "count": len(approvals),
            "approvals": approvals,
        }, indent=2)

    # -- permissions_respond -----------------------------------------------

    @mcp.tool()
    def permissions_respond(
        id: str,
        decision: str,
    ) -> str:
        """Respond to a pending approval request.

        Args:
            id: The approval ID from permissions_list_open
            decision: One of "allow-once", "allow-always", or "deny"
        """
        if decision not in {"allow-once", "allow-always", "deny"}:
            return json.dumps({
                "error": f"Invalid decision: {decision}. "
                         f"Must be allow-once, allow-always, or deny"
            })

        result = bridge.respond_to_approval(id, decision)
        return json.dumps(result, indent=2)

    return mcp


_PROFILE_ROUTER_PUBLIC_HTTP_TOOL_NAMES = frozenset({
    "profiles_list",
    "profile_get",
    "profile_health",
    "profile_context_get",
    "skills_list",
    "skill_view",
    "session_search",
    "viking_search",
    "viking_read",
    "workspace_open",
    "workspace_instructions_get",
    "workspace_context_status",
    "workspace_get",
    "workspace_close",
    "workspace_file_list",
    "workspace_file_read",
    "workspace_file_stat",
    "workspace_file_search",
    "workspace_diff",
})


def _profile_router_http_base_url(host: str, port: int, public_url: str | None = None) -> str:
    """Return the URL origin used in MCP auth metadata.

    Local development can derive the origin from ``host``/``port``. Remote
    deployments behind TLS reverse proxies should pass ``public_url`` so
    ChatGPT and other remote MCP clients see the externally reachable HTTPS
    origin instead of the localhost backend URL.
    """

    configured_public_url = public_url
    if configured_public_url:
        text = configured_public_url.strip().rstrip("/")
        parsed = urlparse(text)
        if (
            parsed.scheme != "https"
            or not parsed.netloc
            or parsed.params
            or parsed.query
            or parsed.fragment
            or parsed.path not in {"", "/"}
        ):
            raise ValueError(
                "profile-router public URL must be an HTTPS origin such as "
                "https://mcp.example.com; do not include /mcp, query, or fragment"
            )
        return text

    public_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    if ":" in public_host and not public_host.startswith("["):
        public_host = f"[{public_host}]"
    return f"http://{public_host}:{int(port)}"


@dataclass(frozen=True)
class _FallbackAuthSettings:
    """Small AuthSettings stand-in for tests when the optional MCP SDK is absent."""

    issuer_url: str
    resource_server_url: str
    required_scopes: list[str]


def _profile_router_auth_settings(*, issuer_url: str, resource_server_url: str):
    try:
        from mcp.server.auth.settings import AuthSettings
    except ImportError:
        return _FallbackAuthSettings(
            issuer_url=issuer_url,
            resource_server_url=resource_server_url,
            required_scopes=[],
        )
    return AuthSettings(
        **{
            "issuer_url": issuer_url,
            "resource_server_url": resource_server_url,
            "required_scopes": [],
        }
    )


def create_profile_router_mcp_server(
    *,
    public_http: bool = False,
    http_auth: bool = False,
    host: str = "127.0.0.1",
    port: int = 8765,
    streamable_http_path: str = "/mcp",
    public_url: str | None = None,
    token_store_path: str | None = None,
    audit_log_path: str | None = None,
):
    """Create the no-model Hermes profile-router MCP server.

    The stdio developer path remains local and unauthenticated. ``public_http``
    selects the restricted HTTP registration surface independently from auth,
    so an auth misconfiguration cannot expose private tools. The normal HTTP
    entrypoint also enables hash-stored personal bearer-token authentication.
    """
    if not _MCP_SERVER_AVAILABLE or FastMCP is None:
        raise ImportError(
            "MCP server requires the 'mcp' package. "
            f"Install with: {sys.executable} -m pip install 'mcp'"
        )
    if os.name == "nt":
        raise RuntimeError(
            "The profile router requires a POSIX host; use WSL on Windows."
        )
    if http_auth and not public_http:
        raise ValueError("profile-router HTTP auth requires public_http=True")

    from mcp_profile_router import (
        assert_default_tools_are_no_model,
        assert_no_model_loop_tools_absent,
        profile_context_get as _profile_context_get,
        profile_get as _profile_get,
        profile_health as _profile_health,
        profiles_list as _profiles_list,
        cron_create_script_only as _cron_create_script_only,
        cron_list as _cron_list,
        cron_pause as _cron_pause,
        cron_resume as _cron_resume,
        cron_run as _cron_run,
        message_send as _message_send,
        telegram_send as _telegram_send,
        workspace_production_action_list as _workspace_production_action_list,
        workspace_production_action_status as _workspace_production_action_status,
        workspace_production_action_run as _workspace_production_action_run,
        server_alias_list as _server_alias_list,
        server_status_check as _server_status_check,
        server_service_logs as _server_service_logs,
        server_docker_ps as _server_docker_ps,
        server_docker_logs as _server_docker_logs,
        server_port_check as _server_port_check,
        server_command_run as _server_command_run,
        server_shell_run as _server_shell_run,
        workspace_web_fetch as _workspace_web_fetch,
        workspace_python_run as _workspace_python_run,
        profile_skill_create as _profile_skill_create,
        profile_skill_delete as _profile_skill_delete,
        profile_skill_edit as _profile_skill_edit,
        profile_skill_patch as _profile_skill_patch,
        profile_skill_remove_file as _profile_skill_remove_file,
        profile_skill_write_file as _profile_skill_write_file,
        profile_memory_add as _profile_memory_add,
        profile_memory_list as _profile_memory_list,
        profile_memory_remove as _profile_memory_remove,
        profile_memory_replace as _profile_memory_replace,
        directory_create as _directory_create,
        file_delete as _file_delete,
        file_move as _file_move,
        patch_apply as _patch_apply,
        file_patch as _file_patch,
        file_write as _file_write,
        git_branch as _git_branch,
        git_add as _git_add,
        git_commit as _git_commit,
        git_push as _git_push,
        git_checkout as _git_checkout,
        git_restore as _git_restore,
        git_rebase as _git_rebase,
        git_merge as _git_merge,
        github_pr_status as _github_pr_status,
        github_pr_create as _github_pr_create,
        github_pr_update as _github_pr_update,
        github_pr_ready as _github_pr_ready,
        github_pr_merge as _github_pr_merge,
        github_issue_view as _github_issue_view,
        github_issue_comment as _github_issue_comment,
        git_diff as _git_diff,
        git_log as _git_log,
        git_status as _git_status,
        terminal_run as _terminal_run,
        process_start as _process_start,
        process_kill as _process_kill,
        process_list as _process_list,
        process_log as _process_log,
        process_poll as _process_poll,
        session_search as _session_search,
        skill_view as _skill_view,
        skills_list as _skills_list,
        viking_read as _viking_read,
        viking_search as _viking_search,
        workspace_close as _workspace_close,
        workspace_context_status as _workspace_context_status,
        workspace_diff as _workspace_diff,
        workspace_file_list as _workspace_file_list,
        workspace_file_read as _workspace_file_read,
        workspace_file_stat as _workspace_file_stat,
        workspace_file_search as _workspace_file_search,
        workspace_status_probe as _workspace_status_probe,
        workspace_scratch_smoke as _workspace_scratch_smoke,
        workspace_get as _workspace_get,
        workspace_instructions_get as _workspace_instructions_get,
        workspace_open as _workspace_open,
    )
    from mcp_profile_router_auth import (
        ProfileRouterAuditLogger,
        ProfileRouterAuthError,
        ProfileRouterBearerTokenVerifier,
        ProfileRouterTokenStore,
        PROFILE_ROUTER_CRON_SCOPE,
        PROFILE_ROUTER_MESSAGING_SCOPE,
        PROFILE_ROUTER_OWNER_SCOPE,
        PROFILE_ROUTER_TERMINAL_SCOPE,
        PROFILE_ROUTER_WRITE_SCOPE,
        VIKING_PROFILE_ROUTER_SCOPE,
        current_access_token_identity,
        extract_result_audit_fields,
        require_current_access_token_scope,
    )

    assert_default_tools_are_no_model()
    assert_no_model_loop_tools_absent()

    auth_kwargs = {}
    audit_logger = ProfileRouterAuditLogger(audit_log_path) if http_auth else None
    if http_auth:
        token_store = ProfileRouterTokenStore(token_store_path)
        base_url = _profile_router_http_base_url(host, port, public_url=public_url)
        auth_kwargs = {
            "token_verifier": ProfileRouterBearerTokenVerifier(token_store),
            "auth": _profile_router_auth_settings(
                issuer_url=base_url,
                resource_server_url=f"{base_url}{streamable_http_path}",
            ),
        }

    mcp = FastMCP(
        "hermes-profile-router",
        instructions=(
            "Hermes Agent no-model profile router. This v1 surface is self-hosted "
            "and read-only by default. HTTP mode is bound to localhost unless the "
            "operator chooses otherwise and requires Hermes-generated bearer tokens. "
            "Public tools expose profile/context hydration, bounded profile skills, "
            "bounded profile session search, and bounded workspace read/diff only: "
            "profiles_list, profile_get, profile_health, profile_context_get, "
            "skills_list, skill_view, session_search, workspace_open, "
            "workspace_instructions_get, workspace_context_status, workspace_get, "
            "workspace_close, workspace_file_list, workspace_file_read, "
            "workspace_file_stat, workspace_file_search, workspace_diff, and policy-gated local/private OpenViking context "
            "tools viking_search and viking_read. Local stdio developer mode also registers "
            "file_patch, patch_apply, file_write, workspace_status_probe, workspace_scratch_smoke, file_move, file_delete, directory_create, terminal_run, "
            "tracked-process scaffolds process_start, process_list, process_poll, process_log, process_kill, "
            "read-only Git scaffolds git_status, git_diff, git_log, git_branch, script-only no_agent cron "
            "scaffolds cron_list, cron_pause, cron_resume, cron_run, cron_create_script_only, no-delivery "
            "messaging dry-run scaffolds message_send and telegram_send, deterministic workspace_python_run, "
            "and profile-scoped profile_skill_* / profile_memory_* wrappers for the public gateway's "
            "explicit production wrappers; those direct tools require fresh "
            "workspace context plus filesystem.write, terminal.execution, git.enabled, cron.enabled/allowed_scripts, "
            "or messaging.enabled/allowed_recipients policy. It does "
            "not expose broad conversation messaging, external delivery, deploy, "
            "Git push/merge, model-backed cron jobs, or agent-loop execution tools."
        ),
        host=host,
        port=port,
        streamable_http_path=streamable_http_path,
        **auth_kwargs,
    )

    def _auth_error_result(tool_name: str, exc: ProfileRouterAuthError) -> str:
        return json.dumps(
            {
                "ok": False,
                "error": {"code": exc.code, "message": exc.message},
                "cost_class": "no_model",
                "llm_calls": 0,
            },
            indent=2,
            sort_keys=True,
        )

    def _audit_tool_result(
        *,
        tool_name: str,
        scope: str,
        result: str,
        audit_profile_ref: str | None = None,
        audit_workspace_id: str | None = None,
    ) -> None:
        if audit_logger is None:
            return
        identity = current_access_token_identity()
        fields = extract_result_audit_fields(result)
        audit_logger.append(
            {
                "token_id": identity.get("token_id"),
                "token_hash_prefix": identity.get("token_hash_prefix"),
                "profile": audit_profile_ref,
                "workspace_id": audit_workspace_id,
                "tool": tool_name,
                "scope": scope,
                **fields,
            }
        )

    def _call_tool(
        tool_name: str,
        required_scope,
        fn,
        *,
        audit_profile_ref: str | None = None,
        audit_workspace_id: str | None = None,
        **kwargs,
    ) -> str:
        required_scopes = (
            (required_scope,)
            if isinstance(required_scope, str)
            else tuple(required_scope or ())
        )
        try:
            if http_auth:
                for scope in required_scopes:
                    require_current_access_token_scope(scope)
            result = fn(**kwargs)
        except ProfileRouterAuthError as exc:
            result = _auth_error_result(tool_name, exc)
        try:
            _audit_tool_result(
                tool_name=tool_name,
                scope=",".join(required_scopes),
                result=result,
                audit_profile_ref=audit_profile_ref,
                audit_workspace_id=audit_workspace_id,
            )
        except ProfileRouterAuthError as exc:
            return _auth_error_result(tool_name, exc)
        return result

    @mcp.tool()
    def profiles_list(active_only: bool = True) -> str:
        """List enabled Hermes profiles as fully qualified refs with no LLM calls."""
        return _call_tool(
            "profiles_list",
            "context:read",
            _profiles_list,
            active_only=active_only,
        )

    @mcp.tool()
    def profile_get(profile_ref: str) -> str:
        """Get non-secret metadata for one fully qualified Hermes profile ref."""
        return _call_tool(
            "profile_get",
            "context:read",
            _profile_get,
            audit_profile_ref=profile_ref,
            profile_ref=profile_ref,
        )

    @mcp.tool()
    def profile_health(profile_ref: str) -> str:
        """Report read-only profile health without invoking any model or tool."""
        return _call_tool(
            "profile_health",
            "context:read",
            _profile_health,
            audit_profile_ref=profile_ref,
            profile_ref=profile_ref,
        )

    @mcp.tool()
    def profile_context_get(profile_ref: str) -> str:
        """Load bounded profile SOUL/policy context without invoking any model."""
        return _call_tool(
            "profile_context_get",
            "context:read",
            _profile_context_get,
            audit_profile_ref=profile_ref,
            profile_ref=profile_ref,
        )

    @mcp.tool()
    def skills_list(profile_ref: str, limit: int | None = 100) -> str:
        """List bounded sanitized profile skill metadata without invoking any model."""
        return _call_tool(
            "skills_list",
            "context:read",
            _skills_list,
            audit_profile_ref=profile_ref,
            profile_ref=profile_ref,
            limit=limit,
        )

    @mcp.tool()
    def skill_view(profile_ref: str, name: str, file_path: str | None = None) -> str:
        """Read a bounded sanitized profile skill file without invoking any model."""
        return _call_tool(
            "skill_view",
            "context:read",
            _skill_view,
            audit_profile_ref=profile_ref,
            profile_ref=profile_ref,
            name=name,
            file_path=file_path,
        )

    @mcp.tool()
    def session_search(
        profile_ref: str,
        query: str | None = None,
        limit: int | None = 3,
        sort: str | None = None,
    ) -> str:
        """Search bounded sanitized profile session snippets without invoking any model."""
        return _call_tool(
            "session_search",
            "context:read",
            _session_search,
            audit_profile_ref=profile_ref,
            profile_ref=profile_ref,
            query=query,
            limit=limit,
            sort=sort,
        )

    @mcp.tool()
    def viking_search(
        query: str,
        mode: str = "auto",
        scope: str | None = None,
        limit: int | None = 10,
    ) -> str:
        """Search bounded local/private OpenViking context without invoking a model."""
        return _call_tool(
            "viking_search",
            VIKING_PROFILE_ROUTER_SCOPE,
            _viking_search,
            query=query,
            mode=mode,
            scope=scope,
            limit=limit,
        )

    @mcp.tool()
    def viking_read(uri: str, level: str = "overview") -> str:
        """Read bounded local/private OpenViking context without invoking a model."""
        return _call_tool(
            "viking_read",
            VIKING_PROFILE_ROUTER_SCOPE,
            _viking_read,
            uri=uri,
            level=level,
        )

    @mcp.tool()
    def workspace_open(profile_ref: str, root: str, mode: str = "checkout") -> str:
        """Open a read-only, policy-gated workspace and return an opaque ID."""
        return _call_tool(
            "workspace_open",
            "workspace:read",
            _workspace_open,
            audit_profile_ref=profile_ref,
            profile_ref=profile_ref,
            root=root,
            mode=mode,
        )

    @mcp.tool()
    def workspace_instructions_get(workspace_id: str) -> str:
        """Hydrate workspace AGENTS/project instructions and return a context token."""
        return _call_tool(
            "workspace_instructions_get",
            "workspace:read",
            _workspace_instructions_get,
            audit_workspace_id=workspace_id,
            workspace_id=workspace_id,
        )

    @mcp.tool()
    def workspace_context_status(workspace_id: str) -> str:
        """Report whether hydrated workspace context is loaded or stale."""
        return _call_tool(
            "workspace_context_status",
            "workspace:read",
            _workspace_context_status,
            audit_workspace_id=workspace_id,
            workspace_id=workspace_id,
        )

    @mcp.tool()
    def workspace_get(workspace_id: str) -> str:
        """Inspect an opened workspace by opaque ID without exposing its root."""
        return _call_tool(
            "workspace_get",
            "workspace:read",
            _workspace_get,
            audit_workspace_id=workspace_id,
            workspace_id=workspace_id,
        )

    @mcp.tool()
    def workspace_close(workspace_id: str) -> str:
        """Close an opened workspace and remove its server-side registry entry."""
        return _call_tool(
            "workspace_close",
            "workspace:read",
            _workspace_close,
            audit_workspace_id=workspace_id,
            workspace_id=workspace_id,
        )

    @mcp.tool()
    def workspace_file_list(
        workspace_id: str,
        path: str | None = None,
        file_glob: str | None = None,
        limit: int | None = 200,
        context_token: str | None = None,
    ) -> str:
        """List bounded sanitized workspace files after context hydration."""
        return _call_tool(
            "workspace_file_list",
            "workspace:read",
            _workspace_file_list,
            audit_workspace_id=workspace_id,
            workspace_id=workspace_id,
            path=path,
            file_glob=file_glob,
            limit=limit,
            context_token=context_token,
        )

    @mcp.tool()
    def workspace_file_read(
        workspace_id: str,
        path: str,
        offset: int | None = 1,
        limit: int | None = 200,
        context_token: str | None = None,
    ) -> str:
        """Read a bounded sanitized text slice after context hydration."""
        return _call_tool(
            "workspace_file_read",
            "workspace:read",
            _workspace_file_read,
            audit_workspace_id=workspace_id,
            workspace_id=workspace_id,
            path=path,
            offset=offset,
            limit=limit,
            context_token=context_token,
        )

    @mcp.tool()
    def workspace_file_stat(
        workspace_id: str,
        path: str,
        context_token: str | None = None,
    ) -> str:
        """Return bounded sanitized file/directory metadata after context hydration."""
        return _call_tool(
            "workspace_file_stat",
            "workspace:read",
            _workspace_file_stat,
            audit_workspace_id=workspace_id,
            workspace_id=workspace_id,
            path=path,
            context_token=context_token,
        )

    @mcp.tool()
    def workspace_file_search(
        workspace_id: str,
        pattern: str,
        path: str | None = None,
        file_glob: str | None = None,
        output_mode: str = "content",
        limit: int | None = 50,
        context_token: str | None = None,
    ) -> str:
        """Search bounded sanitized workspace files after context hydration."""
        return _call_tool(
            "workspace_file_search",
            "workspace:read",
            _workspace_file_search,
            audit_workspace_id=workspace_id,
            workspace_id=workspace_id,
            pattern=pattern,
            path=path,
            file_glob=file_glob,
            output_mode=output_mode,
            limit=limit,
            context_token=context_token,
        )

    @mcp.tool()
    def workspace_diff(
        workspace_id: str,
        context_token: str | None = None,
        max_files: int | None = 100,
    ) -> str:
        """Return a bounded sanitized read-only Git diff after context hydration."""
        return _call_tool(
            "workspace_diff",
            "diff:read",
            _workspace_diff,
            audit_workspace_id=workspace_id,
            workspace_id=workspace_id,
            context_token=context_token,
            max_files=max_files,
        )

    # HTTP is the public v1 surface and must never register private mutation,
    # terminal, Git, cron, messaging, production, or server-shell wrappers.
    # Keep the local stdio developer surface available without duplicating the
    # wrapper implementations: decorators below become no-ops in HTTP mode.
    profile_router_server = mcp
    if public_http:
        registered_public_tools = frozenset(
            profile_router_server._tool_manager._tools
        )
        if registered_public_tools != _PROFILE_ROUTER_PUBLIC_HTTP_TOOL_NAMES:
            unexpected = sorted(
                registered_public_tools - _PROFILE_ROUTER_PUBLIC_HTTP_TOOL_NAMES
            )
            missing = sorted(
                _PROFILE_ROUTER_PUBLIC_HTTP_TOOL_NAMES - registered_public_tools
            )
            raise RuntimeError(
                "Unsafe profile-router HTTP tool registration "
                f"(unexpected={unexpected}, missing={missing})"
            )

        class _PrivateToolRegistrarDisabled:
            @staticmethod
            def tool(*_args, **_kwargs):
                return lambda tool_fn: tool_fn

        mcp = _PrivateToolRegistrarDisabled()

    @mcp.tool()
    def git_status(
        workspace_id: str,
        context_token: str | None = None,
        limit: int | None = 100,
    ) -> str:
        """Private direct read-only Git status for explicit future gateway wrappers."""
        return _call_tool(
            "git_status",
            "diff:read",
            _git_status,
            audit_workspace_id=workspace_id,
            workspace_id=workspace_id,
            context_token=context_token,
            limit=limit,
        )

    @mcp.tool()
    def git_diff(
        workspace_id: str,
        context_token: str | None = None,
        max_files: int | None = 100,
    ) -> str:
        """Private direct read-only Git diff for explicit future gateway wrappers."""
        return _call_tool(
            "git_diff",
            "diff:read",
            _git_diff,
            audit_workspace_id=workspace_id,
            workspace_id=workspace_id,
            context_token=context_token,
            max_files=max_files,
        )

    @mcp.tool()
    def git_log(
        workspace_id: str,
        context_token: str | None = None,
        limit: int | None = 50,
    ) -> str:
        """Private direct read-only Git log for explicit future gateway wrappers."""
        return _call_tool(
            "git_log",
            "diff:read",
            _git_log,
            audit_workspace_id=workspace_id,
            workspace_id=workspace_id,
            context_token=context_token,
            limit=limit,
        )

    @mcp.tool()
    def git_branch(
        workspace_id: str,
        context_token: str | None = None,
        limit: int | None = 100,
    ) -> str:
        """Private direct read-only Git branch metadata for explicit future gateway wrappers."""
        return _call_tool(
            "git_branch",
            "diff:read",
            _git_branch,
            audit_workspace_id=workspace_id,
            workspace_id=workspace_id,
            context_token=context_token,
            limit=limit,
        )

    @mcp.tool()
    def git_add(workspace_id: str, paths: list[str], context_token: str | None = None, update: bool = False) -> str:
        """Private direct owner-mode Git add wrapper."""
        return _call_tool("git_add", "workspace:write", _git_add, audit_workspace_id=workspace_id, workspace_id=workspace_id, paths=paths, context_token=context_token, update=update)

    @mcp.tool()
    def git_commit(workspace_id: str, message: str, context_token: str | None = None) -> str:
        """Private direct owner-mode Git commit wrapper."""
        return _call_tool("git_commit", "workspace:write", _git_commit, audit_workspace_id=workspace_id, workspace_id=workspace_id, message=message, context_token=context_token)

    @mcp.tool()
    def git_push(workspace_id: str, context_token: str | None = None, remote: str = "origin", branch: str | None = None, force_with_lease: bool = False) -> str:
        """Private direct owner-mode Git push wrapper."""
        return _call_tool("git_push", "workspace:write", _git_push, audit_workspace_id=workspace_id, workspace_id=workspace_id, context_token=context_token, remote=remote, branch=branch, force_with_lease=force_with_lease)

    @mcp.tool()
    def git_checkout(workspace_id: str, branch: str, context_token: str | None = None, create: bool = False, start_point: str | None = None) -> str:
        """Private direct owner-mode Git checkout wrapper."""
        return _call_tool("git_checkout", "workspace:write", _git_checkout, audit_workspace_id=workspace_id, workspace_id=workspace_id, branch=branch, context_token=context_token, create=create, start_point=start_point)

    @mcp.tool()
    def git_restore(workspace_id: str, paths: list[str], context_token: str | None = None, staged: bool = False, worktree: bool = True) -> str:
        """Private direct owner-mode Git restore wrapper."""
        return _call_tool("git_restore", "workspace:write", _git_restore, audit_workspace_id=workspace_id, workspace_id=workspace_id, paths=paths, context_token=context_token, staged=staged, worktree=worktree)

    @mcp.tool()
    def git_rebase(workspace_id: str, upstream: str, context_token: str | None = None) -> str:
        """Private direct owner-mode Git rebase wrapper."""
        return _call_tool("git_rebase", "workspace:write", _git_rebase, audit_workspace_id=workspace_id, workspace_id=workspace_id, upstream=upstream, context_token=context_token)

    @mcp.tool()
    def git_merge(workspace_id: str, ref: str, context_token: str | None = None, no_ff: bool = False) -> str:
        """Private direct owner-mode Git merge wrapper."""
        return _call_tool("git_merge", "workspace:write", _git_merge, audit_workspace_id=workspace_id, workspace_id=workspace_id, ref=ref, context_token=context_token, no_ff=no_ff)

    @mcp.tool()
    def github_pr_status(workspace_id: str, context_token: str | None = None, selector: str | int | None = None) -> str:
        """Private direct GitHub PR status wrapper."""
        return _call_tool("github_pr_status", "workspace:write", _github_pr_status, audit_workspace_id=workspace_id, workspace_id=workspace_id, context_token=context_token, selector=selector)

    @mcp.tool()
    def github_pr_create(workspace_id: str, title: str, context_token: str | None = None, body: str | None = None, base: str | None = None, head: str | None = None, draft: bool = False, labels: list[str] | None = None) -> str:
        """Private direct GitHub PR create wrapper."""
        return _call_tool("github_pr_create", "workspace:write", _github_pr_create, audit_workspace_id=workspace_id, workspace_id=workspace_id, title=title, context_token=context_token, body=body, base=base, head=head, draft=draft, labels=labels)

    @mcp.tool()
    def github_pr_update(workspace_id: str, selector: str | int, context_token: str | None = None, title: str | None = None, body: str | None = None, labels: list[str] | None = None) -> str:
        """Private direct GitHub PR update wrapper."""
        return _call_tool("github_pr_update", "workspace:write", _github_pr_update, audit_workspace_id=workspace_id, workspace_id=workspace_id, selector=selector, context_token=context_token, title=title, body=body, labels=labels)

    @mcp.tool()
    def github_pr_ready(workspace_id: str, selector: str | int, context_token: str | None = None) -> str:
        """Private direct GitHub PR ready wrapper."""
        return _call_tool("github_pr_ready", "workspace:write", _github_pr_ready, audit_workspace_id=workspace_id, workspace_id=workspace_id, selector=selector, context_token=context_token)

    @mcp.tool()
    def github_pr_merge(workspace_id: str, selector: str | int, context_token: str | None = None, method: str = "merge", delete_branch: bool = False) -> str:
        """Private direct GitHub PR merge wrapper."""
        return _call_tool("github_pr_merge", "workspace:write", _github_pr_merge, audit_workspace_id=workspace_id, workspace_id=workspace_id, selector=selector, context_token=context_token, method=method, delete_branch=delete_branch)

    @mcp.tool()
    def github_issue_view(workspace_id: str, issue: str | int, context_token: str | None = None) -> str:
        """Private direct GitHub issue view wrapper."""
        return _call_tool("github_issue_view", "workspace:write", _github_issue_view, audit_workspace_id=workspace_id, workspace_id=workspace_id, issue=issue, context_token=context_token)

    @mcp.tool()
    def github_issue_comment(workspace_id: str, issue: str | int, body: str, context_token: str | None = None) -> str:
        """Private direct GitHub issue comment wrapper."""
        return _call_tool("github_issue_comment", "workspace:write", _github_issue_comment, audit_workspace_id=workspace_id, workspace_id=workspace_id, issue=issue, body=body, context_token=context_token)

    @mcp.tool()
    def cron_list(
        workspace_id: str,
        context_token: str | None = None,
        include_disabled: bool = False,
        limit: int | None = 50,
    ) -> str:
        """Private direct no-model cron listing for explicit future gateway wrappers."""
        return _call_tool(
            "cron_list",
            PROFILE_ROUTER_CRON_SCOPE,
            _cron_list,
            audit_workspace_id=workspace_id,
            workspace_id=workspace_id,
            context_token=context_token,
            include_disabled=include_disabled,
            limit=limit,
        )

    @mcp.tool()
    def cron_pause(
        workspace_id: str,
        job_ref: str,
        context_token: str | None = None,
        reason: str | None = None,
    ) -> str:
        """Private direct pause for script-only no_agent cron jobs."""
        return _call_tool(
            "cron_pause",
            PROFILE_ROUTER_CRON_SCOPE,
            _cron_pause,
            audit_workspace_id=workspace_id,
            workspace_id=workspace_id,
            job_ref=job_ref,
            context_token=context_token,
            reason=reason,
        )

    @mcp.tool()
    def cron_resume(
        workspace_id: str,
        job_ref: str,
        context_token: str | None = None,
    ) -> str:
        """Private direct resume for allowlisted script-only no_agent cron jobs."""
        return _call_tool(
            "cron_resume",
            PROFILE_ROUTER_CRON_SCOPE,
            _cron_resume,
            audit_workspace_id=workspace_id,
            workspace_id=workspace_id,
            job_ref=job_ref,
            context_token=context_token,
        )

    @mcp.tool()
    def cron_run(
        workspace_id: str,
        job_ref: str,
        context_token: str | None = None,
    ) -> str:
        """Private direct trigger for allowlisted script-only no_agent cron jobs."""
        return _call_tool(
            "cron_run",
            PROFILE_ROUTER_CRON_SCOPE,
            _cron_run,
            audit_workspace_id=workspace_id,
            workspace_id=workspace_id,
            job_ref=job_ref,
            context_token=context_token,
        )

    @mcp.tool()
    def cron_create_script_only(
        workspace_id: str,
        schedule: str,
        script: str,
        context_token: str | None = None,
        name: str | None = None,
        repeat: int | None = None,
    ) -> str:
        """Private direct creation for allowlisted script-only no_agent cron jobs."""
        return _call_tool(
            "cron_create_script_only",
            PROFILE_ROUTER_CRON_SCOPE,
            _cron_create_script_only,
            audit_workspace_id=workspace_id,
            workspace_id=workspace_id,
            schedule=schedule,
            script=script,
            context_token=context_token,
            name=name,
            repeat=repeat,
        )

    @mcp.tool()
    def message_send(
        workspace_id: str,
        destination: str,
        message: str,
        context_token: str | None = None,
        dry_run: bool = True,
    ) -> str:
        """Private direct messaging dry-run for explicit future gateway wrappers."""
        return _call_tool(
            "message_send",
            PROFILE_ROUTER_MESSAGING_SCOPE,
            _message_send,
            audit_workspace_id=workspace_id,
            workspace_id=workspace_id,
            destination=destination,
            message=message,
            context_token=context_token,
            dry_run=dry_run,
        )

    @mcp.tool()
    def telegram_send(
        workspace_id: str,
        recipient: str,
        message: str,
        context_token: str | None = None,
        dry_run: bool = True,
    ) -> str:
        """Private direct Telegram dry-run for explicit future gateway wrappers."""
        return _call_tool(
            "telegram_send",
            PROFILE_ROUTER_MESSAGING_SCOPE,
            _telegram_send,
            audit_workspace_id=workspace_id,
            workspace_id=workspace_id,
            recipient=recipient,
            message=message,
            context_token=context_token,
            dry_run=dry_run,
        )

    @mcp.tool()
    def workspace_production_action_list(workspace_id: str, context_token: str | None = None) -> str:
        """Private direct list of explicit production action groups."""
        return _call_tool("workspace_production_action_list", PROFILE_ROUTER_TERMINAL_SCOPE, _workspace_production_action_list, audit_workspace_id=workspace_id, workspace_id=workspace_id, context_token=context_token)

    @mcp.tool()
    def workspace_production_action_status(workspace_id: str, action_name: str, context_token: str | None = None) -> str:
        """Private direct status for one explicit production action group."""
        return _call_tool("workspace_production_action_status", PROFILE_ROUTER_TERMINAL_SCOPE, _workspace_production_action_status, audit_workspace_id=workspace_id, workspace_id=workspace_id, action_name=action_name, context_token=context_token)

    @mcp.tool()
    def workspace_production_action_run(workspace_id: str, action_name: str, context_token: str | None = None, args: dict | None = None) -> str:
        """Private direct execution for one explicit production action group."""
        return _call_tool("workspace_production_action_run", (PROFILE_ROUTER_TERMINAL_SCOPE, PROFILE_ROUTER_OWNER_SCOPE), _workspace_production_action_run, audit_workspace_id=workspace_id, workspace_id=workspace_id, action_name=action_name, context_token=context_token, args=args)

    @mcp.tool()
    def server_alias_list(profile_ref: str) -> str:
        """Private direct server alias list without exposing SSH targets."""
        return _call_tool("server_alias_list", PROFILE_ROUTER_TERMINAL_SCOPE, _server_alias_list, audit_profile_ref=profile_ref, profile_ref=profile_ref)

    @mcp.tool()
    def server_status_check(profile_ref: str, alias: str) -> str:
        """Private direct server status check through an explicit alias."""
        return _call_tool("server_status_check", PROFILE_ROUTER_TERMINAL_SCOPE, _server_status_check, audit_profile_ref=profile_ref, profile_ref=profile_ref, alias=alias)

    @mcp.tool()
    def server_service_logs(profile_ref: str, alias: str, service: str, lines: int | None = 100) -> str:
        """Private direct bounded service logs through an explicit alias."""
        return _call_tool("server_service_logs", PROFILE_ROUTER_TERMINAL_SCOPE, _server_service_logs, audit_profile_ref=profile_ref, profile_ref=profile_ref, alias=alias, service=service, lines=lines)

    @mcp.tool()
    def server_docker_ps(profile_ref: str, alias: str) -> str:
        """Private direct docker ps through an explicit alias."""
        return _call_tool("server_docker_ps", PROFILE_ROUTER_TERMINAL_SCOPE, _server_docker_ps, audit_profile_ref=profile_ref, profile_ref=profile_ref, alias=alias)

    @mcp.tool()
    def server_docker_logs(profile_ref: str, alias: str, container: str, lines: int | None = 100) -> str:
        """Private direct bounded docker logs through an explicit alias."""
        return _call_tool("server_docker_logs", PROFILE_ROUTER_TERMINAL_SCOPE, _server_docker_logs, audit_profile_ref=profile_ref, profile_ref=profile_ref, alias=alias, container=container, lines=lines)

    @mcp.tool()
    def server_port_check(profile_ref: str, alias: str, port: int) -> str:
        """Private direct allowlisted port check through an explicit alias."""
        return _call_tool("server_port_check", PROFILE_ROUTER_TERMINAL_SCOPE, _server_port_check, audit_profile_ref=profile_ref, profile_ref=profile_ref, alias=alias, port=port)

    @mcp.tool()
    def server_command_run(profile_ref: str, alias: str, command_name: str) -> str:
        """Private direct named server command group through an explicit alias."""
        return _call_tool("server_command_run", (PROFILE_ROUTER_TERMINAL_SCOPE, PROFILE_ROUTER_OWNER_SCOPE), _server_command_run, audit_profile_ref=profile_ref, profile_ref=profile_ref, alias=alias, command_name=command_name)

    @mcp.tool()
    def server_shell_run(profile_ref: str, alias: str, command: str, timeout_seconds: int | None = 60, max_output_chars: int | None = 40000) -> str:
        """Private direct owner-mode raw server shell command through an explicit alias."""
        return _call_tool("server_shell_run", (PROFILE_ROUTER_TERMINAL_SCOPE, PROFILE_ROUTER_OWNER_SCOPE), _server_shell_run, audit_profile_ref=profile_ref, profile_ref=profile_ref, alias=alias, command=command, timeout_seconds=timeout_seconds, max_output_chars=max_output_chars)

    @mcp.tool()
    def workspace_web_fetch(workspace_id: str, url: str, context_token: str | None = None, method: str = "GET") -> str:
        """Private deterministic allowlisted web/API fetch with SSRF guards."""
        return _call_tool("workspace_web_fetch", PROFILE_ROUTER_TERMINAL_SCOPE, _workspace_web_fetch, audit_workspace_id=workspace_id, workspace_id=workspace_id, url=url, context_token=context_token, method=method)

    @mcp.tool()
    def file_patch(
        workspace_id: str,
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
        context_token: str | None = None,
    ) -> str:
        """Private direct text patch for explicit production gateway wrappers."""
        return _call_tool(
            "file_patch",
            PROFILE_ROUTER_WRITE_SCOPE,
            _file_patch,
            audit_workspace_id=workspace_id,
            workspace_id=workspace_id,
            path=path,
            old_string=old_string,
            new_string=new_string,
            replace_all=replace_all,
            context_token=context_token,
        )

    @mcp.tool()
    def patch_apply(
        workspace_id: str,
        patches: list[dict],
        context_token: str | None = None,
    ) -> str:
        """Private direct bounded multi-file patch for explicit future gateway wrappers."""
        return _call_tool(
            "patch_apply",
            PROFILE_ROUTER_WRITE_SCOPE,
            _patch_apply,
            audit_workspace_id=workspace_id,
            workspace_id=workspace_id,
            patches=patches,
            context_token=context_token,
        )

    @mcp.tool()
    def file_write(
        workspace_id: str,
        path: str,
        content: str,
        context_token: str | None = None,
    ) -> str:
        """Private direct UTF-8 write for explicit production gateway wrappers."""
        return _call_tool(
            "file_write",
            PROFILE_ROUTER_WRITE_SCOPE,
            _file_write,
            audit_workspace_id=workspace_id,
            workspace_id=workspace_id,
            path=path,
            content=content,
            context_token=context_token,
        )

    @mcp.tool()
    def workspace_status_probe(
        workspace_id: str,
        context_token: str | None = None,
    ) -> str:
        """Private fixed status probe for ChatGPT-safe public action smoke."""
        return _call_tool(
            "workspace_status_probe",
            PROFILE_ROUTER_TERMINAL_SCOPE,
            _workspace_status_probe,
            audit_workspace_id=workspace_id,
            workspace_id=workspace_id,
            context_token=context_token,
        )

    @mcp.tool()
    def workspace_scratch_smoke(
        workspace_id: str,
        context_token: str | None = None,
    ) -> str:
        """Private fixed scratch write/read/patch/delete smoke for ChatGPT-safe public action smoke."""
        return _call_tool(
            "workspace_scratch_smoke",
            PROFILE_ROUTER_WRITE_SCOPE,
            _workspace_scratch_smoke,
            audit_workspace_id=workspace_id,
            workspace_id=workspace_id,
            context_token=context_token,
        )

    @mcp.tool()
    def file_move(
        workspace_id: str,
        source_path: str,
        destination_path: str,
        context_token: str | None = None,
    ) -> str:
        """Private direct file move/rename for explicit future gateway wrappers."""
        return _call_tool(
            "file_move",
            PROFILE_ROUTER_WRITE_SCOPE,
            _file_move,
            audit_workspace_id=workspace_id,
            workspace_id=workspace_id,
            source_path=source_path,
            destination_path=destination_path,
            context_token=context_token,
        )

    @mcp.tool()
    def file_delete(
        workspace_id: str,
        path: str,
        context_token: str | None = None,
    ) -> str:
        """Private direct file delete for explicit future gateway wrappers."""
        return _call_tool(
            "file_delete",
            PROFILE_ROUTER_WRITE_SCOPE,
            _file_delete,
            audit_workspace_id=workspace_id,
            workspace_id=workspace_id,
            path=path,
            context_token=context_token,
        )

    @mcp.tool()
    def directory_create(
        workspace_id: str,
        path: str,
        parents: bool = False,
        exist_ok: bool = False,
        context_token: str | None = None,
    ) -> str:
        """Private direct directory creation for explicit future gateway wrappers."""
        return _call_tool(
            "directory_create",
            PROFILE_ROUTER_WRITE_SCOPE,
            _directory_create,
            audit_workspace_id=workspace_id,
            workspace_id=workspace_id,
            path=path,
            parents=parents,
            exist_ok=exist_ok,
            context_token=context_token,
        )

    @mcp.tool()
    def workspace_python_run(
        workspace_id: str,
        code: str,
        timeout: int = 30,
        working_directory: str = ".",
        context_token: str | None = None,
        max_output_chars: int | None = 40000,
    ) -> str:
        """Private direct deterministic Python runner for explicit gateway wrappers."""
        return _call_tool(
            "workspace_python_run",
            PROFILE_ROUTER_TERMINAL_SCOPE,
            _workspace_python_run,
            audit_workspace_id=workspace_id,
            workspace_id=workspace_id,
            code=code,
            timeout=timeout,
            working_directory=working_directory,
            context_token=context_token,
            max_output_chars=max_output_chars,
        )

    @mcp.tool()
    def profile_skill_create(profile_ref: str, name: str, content: str, category: str | None = None, overwrite: bool = False) -> str:
        """Create a profile-scoped skill after skills.write policy."""
        return _call_tool("profile_skill_create", PROFILE_ROUTER_WRITE_SCOPE, _profile_skill_create, audit_profile_ref=profile_ref, profile_ref=profile_ref, name=name, content=content, category=category, overwrite=overwrite)

    @mcp.tool()
    def profile_skill_patch(profile_ref: str, name: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
        """Patch a profile-scoped skill after skills.write policy."""
        return _call_tool("profile_skill_patch", PROFILE_ROUTER_WRITE_SCOPE, _profile_skill_patch, audit_profile_ref=profile_ref, profile_ref=profile_ref, name=name, old_string=old_string, new_string=new_string, replace_all=replace_all)

    @mcp.tool()
    def profile_skill_edit(profile_ref: str, name: str, content: str) -> str:
        """Replace a profile-scoped skill after skills.write policy."""
        return _call_tool("profile_skill_edit", PROFILE_ROUTER_WRITE_SCOPE, _profile_skill_edit, audit_profile_ref=profile_ref, profile_ref=profile_ref, name=name, content=content)

    @mcp.tool()
    def profile_skill_write_file(profile_ref: str, name: str, file_path: str, content: str) -> str:
        """Write a safe profile skill support file after skills.write policy."""
        return _call_tool("profile_skill_write_file", PROFILE_ROUTER_WRITE_SCOPE, _profile_skill_write_file, audit_profile_ref=profile_ref, profile_ref=profile_ref, name=name, file_path=file_path, content=content)

    @mcp.tool()
    def profile_skill_remove_file(profile_ref: str, name: str, file_path: str) -> str:
        """Remove a safe profile skill support file after skills.write policy."""
        return _call_tool("profile_skill_remove_file", PROFILE_ROUTER_WRITE_SCOPE, _profile_skill_remove_file, audit_profile_ref=profile_ref, profile_ref=profile_ref, name=name, file_path=file_path)

    @mcp.tool()
    def profile_skill_delete(profile_ref: str, name: str, absorbed_into: str | None = None, confirm_delete: bool = False) -> str:
        """Delete a profile-scoped skill only with delete policy and explicit intent."""
        return _call_tool("profile_skill_delete", PROFILE_ROUTER_WRITE_SCOPE, _profile_skill_delete, audit_profile_ref=profile_ref, profile_ref=profile_ref, name=name, absorbed_into=absorbed_into, confirm_delete=confirm_delete)

    @mcp.tool()
    def profile_memory_add(profile_ref: str, content: str, target: str | None = "memory") -> str:
        """Add a bounded profile memory after memory.write policy."""
        return _call_tool("profile_memory_add", PROFILE_ROUTER_WRITE_SCOPE, _profile_memory_add, audit_profile_ref=profile_ref, profile_ref=profile_ref, content=content, target=target)

    @mcp.tool()
    def profile_memory_replace(profile_ref: str, old_text: str, new_content: str, target: str | None = "memory") -> str:
        """Replace an exact profile memory entry after memory.write policy."""
        return _call_tool("profile_memory_replace", PROFILE_ROUTER_WRITE_SCOPE, _profile_memory_replace, audit_profile_ref=profile_ref, profile_ref=profile_ref, old_text=old_text, new_content=new_content, target=target)

    @mcp.tool()
    def profile_memory_remove(profile_ref: str, old_text: str, target: str | None = "memory") -> str:
        """Remove an exact profile memory entry after memory.write policy."""
        return _call_tool("profile_memory_remove", PROFILE_ROUTER_WRITE_SCOPE, _profile_memory_remove, audit_profile_ref=profile_ref, profile_ref=profile_ref, old_text=old_text, target=target)

    @mcp.tool()
    def profile_memory_list(profile_ref: str, target: str | None = "memory", limit: int | None = 50) -> str:
        """List bounded redacted profile memory entries after memory.write policy."""
        return _call_tool("profile_memory_list", PROFILE_ROUTER_WRITE_SCOPE, _profile_memory_list, audit_profile_ref=profile_ref, profile_ref=profile_ref, target=target, limit=limit)

    @mcp.tool()
    def process_start(
        workspace_id: str,
        command: str,
        timeout: int = 30,
        working_directory: str = ".",
        context_token: str | None = None,
        max_output_chars: int | None = 20000,
    ) -> str:
        """Private direct runtime-owned process launch for explicit future gateway wrappers."""
        return _call_tool(
            "process_start",
            PROFILE_ROUTER_TERMINAL_SCOPE,
            _process_start,
            audit_workspace_id=workspace_id,
            workspace_id=workspace_id,
            command=command,
            timeout=timeout,
            working_directory=working_directory,
            context_token=context_token,
            max_output_chars=max_output_chars,
        )

    @mcp.tool()
    def process_list(
        workspace_id: str,
        context_token: str | None = None,
        limit: int | None = 50,
    ) -> str:
        """Private direct tracked-process listing for explicit future gateway wrappers."""
        return _call_tool(
            "process_list",
            PROFILE_ROUTER_TERMINAL_SCOPE,
            _process_list,
            audit_workspace_id=workspace_id,
            workspace_id=workspace_id,
            context_token=context_token,
            limit=limit,
        )

    @mcp.tool()
    def process_poll(
        workspace_id: str,
        process_id: str,
        context_token: str | None = None,
    ) -> str:
        """Private direct tracked-process polling for explicit future gateway wrappers."""
        return _call_tool(
            "process_poll",
            PROFILE_ROUTER_TERMINAL_SCOPE,
            _process_poll,
            audit_workspace_id=workspace_id,
            workspace_id=workspace_id,
            process_id=process_id,
            context_token=context_token,
        )

    @mcp.tool()
    def process_log(
        workspace_id: str,
        process_id: str,
        context_token: str | None = None,
        max_chars: int | None = 20000,
    ) -> str:
        """Private direct tracked-process bounded log read for explicit future gateway wrappers."""
        return _call_tool(
            "process_log",
            PROFILE_ROUTER_TERMINAL_SCOPE,
            _process_log,
            audit_workspace_id=workspace_id,
            workspace_id=workspace_id,
            process_id=process_id,
            context_token=context_token,
            max_chars=max_chars,
        )

    @mcp.tool()
    def process_kill(
        workspace_id: str,
        process_id: str,
        context_token: str | None = None,
    ) -> str:
        """Private direct tracked-process kill for explicit future gateway wrappers."""
        return _call_tool(
            "process_kill",
            PROFILE_ROUTER_TERMINAL_SCOPE,
            _process_kill,
            audit_workspace_id=workspace_id,
            workspace_id=workspace_id,
            process_id=process_id,
            context_token=context_token,
        )

    @mcp.tool()
    def terminal_run(
        workspace_id: str,
        command: str,
        timeout: int = 30,
        working_directory: str = ".",
        context_token: str | None = None,
        max_output_chars: int | None = 60000,
    ) -> str:
        """Private direct read/test terminal allowlist for explicit production gateway wrappers."""
        return _call_tool(
            "terminal_run",
            PROFILE_ROUTER_TERMINAL_SCOPE,
            _terminal_run,
            audit_workspace_id=workspace_id,
            workspace_id=workspace_id,
            command=command,
            timeout=timeout,
            working_directory=working_directory,
            context_token=context_token,
            max_output_chars=max_output_chars,
        )

    return profile_router_server


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_mcp_server(verbose: bool = False) -> None:
    """Start the Hermes MCP server on stdio."""
    if not _MCP_SERVER_AVAILABLE:
        print(
            "Error: MCP server requires the 'mcp' package.\n"
            f"Install with: {sys.executable} -m pip install 'mcp'",
            file=sys.stderr,
        )
        sys.exit(1)

    if verbose:
        logging.basicConfig(level=logging.DEBUG, stream=sys.stderr)
    else:
        logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

    bridge = EventBridge()
    bridge.start()

    server = create_mcp_server(event_bridge=bridge)

    import asyncio

    async def _run():
        try:
            await server.run_stdio_async()
        finally:
            bridge.stop()

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        bridge.stop()


def run_profile_router_mcp_server(
    verbose: bool = False,
    *,
    transport: str = "stdio",
    host: str = "127.0.0.1",
    port: int = 8765,
    streamable_http_path: str = "/mcp",
    public_url: str | None = None,
    token_store_path: str | None = None,
    audit_log_path: str | None = None,
) -> None:
    """Start the no-model Hermes profile-router MCP server."""
    if not _MCP_SERVER_AVAILABLE:
        print(
            "Error: MCP server requires the 'mcp' package.\n"
            f"Install with: {sys.executable} -m pip install 'mcp'",
            file=sys.stderr,
        )
        sys.exit(1)

    if transport not in {"stdio", "streamable-http"}:
        print("Error: profile-router transport must be 'stdio' or 'streamable-http'", file=sys.stderr)
        sys.exit(2)

    if verbose:
        logging.basicConfig(level=logging.DEBUG, stream=sys.stderr)
    else:
        logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

    public_http = transport == "streamable-http"
    server = create_profile_router_mcp_server(
        public_http=public_http,
        http_auth=public_http,
        host=host,
        port=port,
        streamable_http_path=streamable_http_path,
        public_url=public_url,
        token_store_path=token_store_path,
        audit_log_path=audit_log_path,
    )

    import asyncio

    async def _run():
        if transport == "streamable-http":
            await server.run_streamable_http_async()
        else:
            await server.run_stdio_async()

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        return

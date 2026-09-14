"""
Hermes MCP Server — expose messaging conversations as MCP tools (`hermes mcp serve`).

A stdio MCP server letting any MCP client (Claude Code, Cursor, Codex, ...) list
conversations, read history, send messages, poll live events, and manage approvals.
Matches OpenClaw's 9-tool channel bridge surface plus the Hermes-specific
channels_list. Client config: {"mcpServers": {"hermes": {"command": "hermes", "args": ["mcp", "serve"]}}}
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

logger = logging.getLogger("hermes.mcp_serve")

# mcp 2.0 removed `mcp.server.fastmcp`; `mcp.server.MCPServer` keeps the same
# `@server.tool()` / `run_stdio_async()` surface (docstring -> description,
# signature -> input schema).
_MCP_SERVER_AVAILABLE = False
try:
    from mcp.server import MCPServer

    _MCP_SERVER_AVAILABLE = True
except ImportError:
    MCPServer = None  # type: ignore[assignment,misc]


# --- Helpers -----------------------------------------------------------------

def _hermes_home() -> Path:
    try:
        from hermes_constants import get_hermes_home
        return get_hermes_home()
    except ImportError:
        return Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))


def _get_sessions_dir() -> Path:
    return _hermes_home() / "sessions"


def _read_state_db_mtime() -> float:
    try:
        return (_hermes_home() / "state.db").stat().st_mtime
    except OSError:  # missing file included
        return 0.0


def _read_json(path: Path):
    """Parsed JSON file, or {} when missing/unreadable."""
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.debug("Failed to load %s: %s", path.name, e)
        return {}


def _close_quietly(db, what: str) -> None:
    try:
        db.close()
    except Exception:
        logger.debug("Failed to close MCP %s SessionDB", what, exc_info=True)


def _get_session_db():
    """SessionDB instance for reading message transcripts, or None."""
    try:
        from hermes_state_registry import acquire
        return acquire()
    except Exception as e:
        logger.debug("SessionDB unavailable: %s", e)
        return None


def _load_session_messages(session_id: str):
    """(messages, error) for one session; closes the temporary database handle."""
    db = _get_session_db()
    if db is None:
        return None, "Session database unavailable"
    try:
        return db.get_messages(session_id), None
    except Exception as e:
        return None, f"Failed to read messages: {e}"
    finally:
        try:
            from hermes_state_registry import release_or_close
            release_or_close(db)
        except Exception:
            logger.debug("Failed to close MCP SessionDB", exc_info=True)


def _load_sessions_index() -> dict:
    """Gateway routing index: session_key -> entry dict.

    state.db is primary (gateway session rows carry session_key/origin metadata);
    sessions.json is the fallback for pre-migration databases without session_keys.

    state.db is the primary source (#9006): gateway sessions persist their routing metadata (session_key,
    chat/thread ids, display_name, origin) on the durable session row, so a single database read replaces
    the old dual-file sessions.json dependency.
    """
    return _load_sessions_index_from_db() or _load_sessions_index_from_json()


def _iso(ts) -> str:
    try:
        return datetime.fromtimestamp(float(ts)).isoformat() if ts else ""
    except (TypeError, ValueError, OSError):
        return ""


def _row_to_index_entry(row: dict) -> dict:
    """Convert a state.db gateway session row to the sessions.json entry shape."""
    origin = {}
    if row.get("origin_json"):
        try:
            parsed = json.loads(row["origin_json"])
            if isinstance(parsed, dict):
                origin = parsed
        except (TypeError, ValueError):
            pass
    if not origin:  # pre-origin_json rows: synthesize the minimal origin from columns
        origin = {"platform": row.get("source", ""), **{k: row.get(k) for k in ("chat_id", "chat_type", "thread_id", "user_id")}}

    input_tokens = int(row.get("input_tokens") or 0)
    output_tokens = int(row.get("output_tokens") or 0)
    return {
        "session_id": str(row.get("id", "")), "session_key": row.get("session_key", ""),
        "platform": row.get("source", ""),
        "chat_type": row.get("chat_type") or origin.get("chat_type", ""),
        "display_name": row.get("display_name") or origin.get("chat_name") or "",
        "origin": origin,
        "created_at": _iso(row.get("started_at")),
        "updated_at": _iso(row.get("last_active") or row.get("started_at")),
        "input_tokens": input_tokens, "output_tokens": output_tokens,
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
        return {row["session_key"]: _row_to_index_entry(row) for row in lister(active_only=True) if row.get("session_key")}
    except Exception as e:
        logger.debug("Failed to load gateway sessions from state.db: %s", e)
        return {}
    finally:
        try:
            db.close()
        except Exception:
            pass


def _load_sessions_index_from_json() -> dict:
    """Legacy fallback: read sessions.json directly (avoids importing SessionStore,
    which needs GatewayConfig). Keys starting with "_" are metadata sentinels
    (e.g. "_README"), not session entries."""
    data = _read_json(_get_sessions_dir() / "sessions.json")
    return {k: v for k, v in data.items() if not str(k).startswith("_")} if isinstance(data, dict) else {}


def _load_channel_directory() -> dict:
    """Load the cached channel directory for available targets."""
    return _read_json(_hermes_home() / "channel_directory.json")


def _coerce_int(value, *, default: int, minimum: int, maximum: int) -> int:
    """Clamped int for MCP tool boundaries; *default* when the client sent an unconvertible value."""
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        coerced = default
    return max(minimum, min(coerced, maximum))


def _extract_message_content(msg: dict) -> str:
    """Extract text content from a message, handling multi-part content."""
    content = msg.get("content", "")
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return str(content) if content else ""


def _extract_attachments(msg: dict) -> List[dict]:
    """Non-text attachments: image/file content blocks plus MEDIA: tags in the text."""
    attachments = []
    content = msg.get("content", "")
    for part in content if isinstance(content, list) else ():
        if not isinstance(part, dict):
            continue
        ptype = part.get("type", "")
        if ptype == "image_url":
            url = part.get("image_url", {}).get("url", "") if isinstance(part.get("image_url"), dict) else ""
        elif ptype == "image":
            url = part.get("url", part.get("source", {}).get("url", ""))
        else:
            if ptype != "text":
                attachments.append({"type": ptype, "data": part})
            continue
        if url:
            attachments.append({"type": "image", "url": url})
    for match in re.finditer(r'MEDIA:\s*(\S+)', _extract_message_content(msg)):
        attachments.append({"type": "media", "path": match.group(1)})
    return attachments


# --- Event Bridge — polls SessionDB for new messages, maintains event queue ---

QUEUE_LIMIT = 1000
POLL_INTERVAL = 0.2  # seconds between DB polls (200ms)


@dataclass
class QueueEvent:
    """An event in the bridge's in-memory queue."""
    cursor: int
    type: str  # "message", "approval_requested", "approval_resolved"
    session_key: str = ""
    data: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"cursor": self.cursor, "type": self.type, "session_key": self.session_key, **self.data}


def _ts_float(ts) -> float:
    """Normalize a message timestamp (epoch int/float or ISO string) to float."""
    if isinstance(ts, (int, float)):
        return float(ts)
    if not (isinstance(ts, str) and ts):
        return 0.0
    try:
        return float(ts)
    except ValueError:
        try:
            return datetime.fromisoformat(ts).timestamp()
        except Exception:
            return 0.0


def _latest_ts(messages) -> float:
    """Newest normalized timestamp among *messages* (0.0 when none)."""
    return max((_ts_float(m.get("timestamp", 0)) for m in (messages or ())), default=0.0)


class EventBridge:
    """Background poller watching SessionDB for new messages, feeding an in-memory
    event queue with waiter support (the Hermes analogue of OpenClaw's WebSocket
    gateway bridge, polling SQLite instead)."""

    def __init__(self):
        self._queue: List[QueueEvent] = []
        self._cursor = 0
        self._lock = threading.Lock()
        self._new_event = threading.Event()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_poll_timestamps: Dict[str, float] = {}  # session_key -> unix timestamp
        self._pending_approvals: Dict[str, dict] = {}  # populated from events
        self._state_db_mtime: float = 0.0  # skip polling work when state.db is unchanged
        self._cached_sessions_index: dict = {}

    def start(self):
        """Start the background polling thread."""
        if self._running:
            return
        # Baseline existing history BEFORE polling so startup never replays old
        # messages as events; sessions appearing later default to last_seen=0.0
        # in _poll_once, so new-conversation delivery is preserved.
        # Unit tests that drive _poll_once directly bypass start() and still observe first-poll delivery.
        # See #13414.
        self._establish_baseline()
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        logger.debug("EventBridge started")

    def stop(self):
        """Stop the background polling thread and wake any waiters."""
        self._running = False
        self._new_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        logger.debug("EventBridge stopped")

    def _matching(self, after_cursor: int, session_key: Optional[str], limit: int) -> List[dict]:
        with self._lock:
            return [e.as_dict() for e in self._queue
                    if e.cursor > after_cursor and (not session_key or e.session_key == session_key)][:limit]

    def poll_events(self, after_cursor: int = 0, session_key: Optional[str] = None, limit: int = 20) -> dict:
        """Return events since after_cursor, optionally filtered by session_key."""
        events = self._matching(after_cursor, session_key, limit)
        return {"events": events, "next_cursor": events[-1]["cursor"] if events else after_cursor}

    def wait_for_event(self, after_cursor: int = 0, session_key: Optional[str] = None, timeout_ms: int = 30000) -> Optional[dict]:
        """Block until a matching event arrives or timeout expires."""
        deadline = time.monotonic() + (timeout_ms / 1000.0)
        while time.monotonic() < deadline:
            found = self._matching(after_cursor, session_key, 1)
            if found:
                return found[0]
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            self._new_event.clear()
            self._new_event.wait(timeout=min(remaining, POLL_INTERVAL))
        return None

    def list_pending_approvals(self) -> List[dict]:
        """List approval requests observed during this bridge session."""
        with self._lock:
            return sorted(self._pending_approvals.values(), key=lambda a: a.get("created_at", ""))

    def respond_to_approval(self, approval_id: str, decision: str) -> dict:
        """Resolve a pending approval (best-effort without gateway IPC)."""
        with self._lock:
            approval = self._pending_approvals.pop(approval_id, None)
        if not approval:
            return {"error": f"Approval not found: {approval_id}"}
        self._enqueue(QueueEvent(0, "approval_resolved", approval.get("session_key", ""),  # cursor set by _enqueue
                                 {"approval_id": approval_id, "decision": decision}))
        return {"resolved": True, "approval_id": approval_id, "decision": decision}

    def _enqueue(self, event: QueueEvent) -> None:
        """Add an event to the queue (trimmed to QUEUE_LIMIT) and wake any waiters."""
        with self._lock:
            self._cursor += 1
            event.cursor = self._cursor
            self._queue.append(event)
            while len(self._queue) > QUEUE_LIMIT:
                self._queue.pop(0)
        self._new_event.set()

    def _establish_baseline(self) -> None:
        """Record per-session latest timestamps and the state.db mtime WITHOUT
        emitting events. Only sessions existing now are baselined; later ones
        default to last_seen=0.0 in _poll_once, so their first message is delivered."""
        db = _get_session_db()
        if not db:
            return
        try:
            self._state_db_mtime = _read_state_db_mtime()
            try:
                self._cached_sessions_index = _load_sessions_index()
            except Exception:
                self._cached_sessions_index = {}
            for session_key, entry in self._cached_sessions_index.items():
                session_id = entry.get("session_id", "")
                if not session_id:
                    continue
                try:
                    latest = _latest_ts(db.get_messages(session_id))
                except Exception:
                    continue
                if latest > 0.0:
                    self._last_poll_timestamps[session_key] = latest
        finally:
            _close_quietly(db, "baseline")

    def _poll_loop(self):
        """Background loop: poll SessionDB for new messages."""
        db = _get_session_db()
        if not db:
            logger.warning("EventBridge: SessionDB unavailable, event polling disabled")
            return
        try:
            while self._running:
                try:
                    self._poll_once(db)
                except Exception as e:
                    logger.debug("EventBridge poll error: %s", e)
                time.sleep(POLL_INTERVAL)
        finally:
            _close_quietly(db, "polling")

    def _poll_once(self, db):
        """Check for new messages across all sessions.

        One state.db mtime check gates all work, making 200ms polling nearly free.
        The routing index lives in the same file as the messages, so a new
        conversation and its first message land under a single mtime change (no
        dual-file race that could drop brand-new conversations).

        See #8925, #9006.
        """
        db_mtime = _read_state_db_mtime()
        if db_mtime == self._state_db_mtime:
            return
        self._state_db_mtime = db_mtime
        # Refresh the index on every change tick: one indexed query, never lags messages.
        self._cached_sessions_index = _load_sessions_index()

        for session_key, entry in self._cached_sessions_index.items():
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
            for msg in messages:
                if msg.get("role", "") not in {"user", "assistant"} or _ts_float(msg.get("timestamp", 0)) <= last_seen:
                    continue
                content = _extract_message_content(msg)
                if not content:
                    continue
                self._enqueue(QueueEvent(0, "message", session_key, {
                    "role": msg.get("role", ""), "content": content[:500],
                    "timestamp": str(msg.get("timestamp", "")), "message_id": str(msg.get("id", "")),
                }))
            latest = _latest_ts(messages)
            if latest > last_seen:
                self._last_poll_timestamps[session_key] = latest


# --- MCP Server ---------------------------------------------------------------

def _conversation_messages(session_key: str):
    """(messages, error_json) for a conversation; exactly one is None."""
    entry = _load_sessions_index().get(session_key)
    if not entry:
        return None, json.dumps({"error": f"Conversation not found: {session_key}"})
    session_id = entry.get("session_id", "")
    if not session_id:
        return None, json.dumps({"error": "No session ID for this conversation"})
    messages, error = _load_session_messages(session_id)
    if error:
        return None, json.dumps({"error": error})
    return messages, None


def _platform_matches(wanted: Optional[str], actual: str) -> bool:
    return not wanted or actual.lower() == wanted.lower()


class _ToolHandlers:
    """The MCP tool handlers; each method named in _TOOL_NAMES is registered as one tool.

    Method docstrings are the wire-format tool descriptions and signatures the
    input schemas — do not reword or reflow them.
    """

    def __init__(self, bridge: EventBridge):
        self.bridge = bridge

    def conversations_list(self, platform: Optional[str] = None, limit: int = 50, search: Optional[str] = None) -> str:
        """List active messaging conversations across connected platforms.

        Returns conversations with their session keys (needed for messages_read),
        platform, chat type, display name, and last activity time.

        Args:
            platform: Filter by platform name (telegram, discord, slack, etc.)
            limit: Maximum number of conversations to return (default 50)
            search: Optional text to filter conversations by name
        """
        limit = _coerce_int(limit, default=50, minimum=1, maximum=200)
        conversations = []
        for key, entry in _load_sessions_index().items():
            origin = entry.get("origin", {})
            entry_platform = entry.get("platform") or origin.get("platform", "")
            if not _platform_matches(platform, entry_platform):
                continue
            display_name = entry.get("display_name", "")
            chat_name = origin.get("chat_name", "")
            if search and not any(search.lower() in s.lower() for s in (display_name, chat_name, key)):
                continue
            conversations.append({
                "session_key": key, "session_id": entry.get("session_id", ""), "platform": entry_platform,
                "chat_type": entry.get("chat_type", origin.get("chat_type", "")),
                "display_name": display_name, "chat_name": chat_name,
                "user_name": origin.get("user_name", ""), "updated_at": entry.get("updated_at", ""),
            })

        conversations = sorted(conversations, key=lambda c: c.get("updated_at", ""), reverse=True)[:limit]
        return json.dumps({"count": len(conversations), "conversations": conversations}, indent=2)

    def conversation_get(self, session_key: str) -> str:
        """Get detailed info about one conversation by its session key.

        Args:
            session_key: The session key from conversations_list
        """
        entry = _load_sessions_index().get(session_key)
        if not entry:
            return json.dumps({"error": f"Conversation not found: {session_key}"})
        origin = entry.get("origin", {})
        return json.dumps({
            "session_key": session_key, "session_id": entry.get("session_id", ""),
            "platform": entry.get("platform") or origin.get("platform", ""),
            "chat_type": entry.get("chat_type", origin.get("chat_type", "")),
            "display_name": entry.get("display_name", ""),
            "user_name": origin.get("user_name", ""), "chat_name": origin.get("chat_name", ""),
            "chat_id": origin.get("chat_id", ""), "thread_id": origin.get("thread_id"),
            "updated_at": entry.get("updated_at", ""), "created_at": entry.get("created_at", ""),
            "input_tokens": entry.get("input_tokens", 0), "output_tokens": entry.get("output_tokens", 0),
            "total_tokens": entry.get("total_tokens", 0),
        }, indent=2)

    def messages_read(self, session_key: str, limit: int = 50) -> str:
        """Read recent messages from a conversation.

        Returns the message history in chronological order with role, content,
        and timestamp for each message.

        Args:
            session_key: The session key from conversations_list
            limit: Maximum number of messages to return (default 50, most recent)
        """
        limit = _coerce_int(limit, default=50, minimum=1, maximum=200)
        all_messages, error = _conversation_messages(session_key)
        if error:
            return error
        filtered = []
        for msg in all_messages:
            role = msg.get("role", "")
            content = _extract_message_content(msg) if role in {"user", "assistant"} else ""
            if content:
                filtered.append({"id": str(msg.get("id", "")), "role": role,
                                 "content": content[:2000], "timestamp": msg.get("timestamp", "")})
        messages = filtered[-limit:]
        return json.dumps({"session_key": session_key, "count": len(messages),
                           "total_in_session": len(filtered), "messages": messages}, indent=2)

    def attachments_fetch(self, session_key: str, message_id: str) -> str:
        """List non-text attachments for a message in a conversation.

        Extracts images, media files, and other non-text content blocks
        from the specified message.

        Args:
            session_key: The session key from conversations_list
            message_id: The message ID from messages_read
        """
        all_messages, error = _conversation_messages(session_key)
        if error:
            return error
        target_msg = next((m for m in all_messages if str(m.get("id", "")) == message_id), None)
        if not target_msg:
            return json.dumps({"error": f"Message not found: {message_id}"})
        attachments = _extract_attachments(target_msg)
        return json.dumps({"message_id": message_id, "count": len(attachments), "attachments": attachments}, indent=2)

    def events_poll(self, after_cursor: int = 0, session_key: Optional[str] = None, limit: int = 20) -> str:
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
        result = self.bridge.poll_events(after_cursor=after_cursor, session_key=session_key, limit=limit)
        return json.dumps(result, indent=2)

    def events_wait(self, after_cursor: int = 0, session_key: Optional[str] = None, timeout_ms: int = 30000) -> str:
        """Wait for the next conversation event (long-poll).

        Blocks until a matching event arrives or the timeout expires.
        Use this for near-real-time event delivery without polling.

        Args:
            after_cursor: Wait for events after this cursor
            session_key: Optional filter to one conversation
            timeout_ms: Maximum wait time in milliseconds (default 30000)
        """
        after_cursor = _coerce_int(after_cursor, default=0, minimum=0, maximum=10**18)
        timeout_ms = _coerce_int(timeout_ms, default=30000, minimum=0, maximum=300000)  # cap 5 min
        event = self.bridge.wait_for_event(after_cursor=after_cursor, session_key=session_key, timeout_ms=timeout_ms)
        return json.dumps({"event": event} if event else {"event": None, "reason": "timeout"}, indent=2)

    def messages_send(self, target: str, message: str) -> str:
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
            return send_message_tool({"action": "send", "target": target, "message": message})
        except ImportError:
            return json.dumps({"error": "Send message tool not available"})
        except Exception as e:
            return json.dumps({"error": f"Send failed: {e}"})

    def channels_list(self, platform: Optional[str] = None) -> str:
        """List available messaging channels and targets across platforms.

        Returns channels that you can send messages to. The target strings
        returned here can be used directly with the messages_send tool.

        Args:
            platform: Filter by platform name (telegram, discord, slack, etc.)
        """
        directory = _load_channel_directory()
        if not directory:
            # No cached directory: derive send targets from the routing index.
            targets, seen = [], set()
            for key, entry in _load_sessions_index().items():
                origin = entry.get("origin", {})
                p = entry.get("platform") or origin.get("platform", "")
                chat_id = origin.get("chat_id", "")
                target_str = f"{p}:{chat_id}"
                if not p or not chat_id or not _platform_matches(platform, p) or target_str in seen:
                    continue
                seen.add(target_str)
                targets.append({"target": target_str, "platform": p,
                                "name": entry.get("display_name") or origin.get("chat_name", ""),
                                "chat_type": entry.get("chat_type", origin.get("chat_type", ""))})
            return json.dumps({"count": len(targets), "channels": targets}, indent=2)
        channels = []
        for plat, entries_list in directory.get("platforms", {}).items():
            if not _platform_matches(platform, plat) or not isinstance(entries_list, list):
                continue
            for ch in entries_list:
                if isinstance(ch, dict):
                    chat_id = ch.get("id", ch.get("chat_id", ""))
                    channels.append({"target": f"{plat}:{chat_id}" if chat_id else plat, "platform": plat,
                                     "name": ch.get("name", ch.get("display_name", "")), "chat_type": ch.get("type", "")})
        return json.dumps({"count": len(channels), "channels": channels}, indent=2)

    def permissions_list_open(self) -> str:
        """List pending approval requests observed during this bridge session.

        Returns exec and plugin approval requests that the bridge has seen
        since it started. Approvals are live-session only — older approvals
        from before the bridge connected are not included.
        """
        approvals = self.bridge.list_pending_approvals()
        return json.dumps({"count": len(approvals), "approvals": approvals}, indent=2)

    def permissions_respond(self, id: str, decision: str) -> str:
        """Respond to a pending approval request.

        Args:
            id: The approval ID from permissions_list_open
            decision: One of "allow-once", "allow-always", or "deny"
        """
        if decision not in {"allow-once", "allow-always", "deny"}:
            return json.dumps({"error": f"Invalid decision: {decision}. Must be allow-once, allow-always, or deny"})
        return json.dumps(self.bridge.respond_to_approval(id, decision), indent=2)


# Registration order == list_tools order (wire format).
_TOOL_NAMES = (
    "conversations_list", "conversation_get", "messages_read", "attachments_fetch",
    "events_poll", "events_wait", "messages_send", "channels_list",
    "permissions_list_open", "permissions_respond",
)


def create_mcp_server(event_bridge: Optional[EventBridge] = None) -> "MCPServer":
    """Create and return the Hermes MCP server with all tools registered."""
    if not _MCP_SERVER_AVAILABLE:
        raise ImportError(f"MCP server requires the 'mcp' package. Install with: {sys.executable} -m pip install 'mcp'")
    mcp = MCPServer("hermes", instructions=(
        "Hermes Agent messaging bridge. Use these tools to interact with "
        "conversations across Telegram, Discord, Slack, WhatsApp, Signal, "
        "Matrix, and other connected platforms."
    ))
    handlers = _ToolHandlers(event_bridge or EventBridge())
    for name in _TOOL_NAMES:
        mcp.tool()(getattr(handlers, name))
    return mcp


def run_mcp_server(verbose: bool = False) -> None:
    """Start the Hermes MCP server on stdio."""
    if not _MCP_SERVER_AVAILABLE:
        print("Error: MCP server requires the 'mcp' package.\n"
              f"Install with: {sys.executable} -m pip install 'mcp'", file=sys.stderr)
        sys.exit(1)
    logging.basicConfig(level=logging.DEBUG if verbose else logging.WARNING, stream=sys.stderr)
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

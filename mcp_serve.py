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

import inspect
import json
import logging
import os
import re
import secrets
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set
from urllib.parse import parse_qs, urlencode

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

def _schema_property_to_annotation(prop: dict) -> Any:
    prop_type = prop.get("type") if isinstance(prop, dict) else None
    if isinstance(prop_type, list):
        prop_type = next((item for item in prop_type if item != "null"), None)
    if prop_type == "string":
        return str
    if prop_type == "integer":
        return int
    if prop_type == "number":
        return float
    if prop_type == "boolean":
        return bool
    if prop_type == "array":
        return list
    if prop_type == "object":
        return dict
    return Any


def _safe_identifier(name: str) -> str:
    cleaned = re.sub(r"\W", "_", name or "arg")
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"arg_{cleaned}"
    if cleaned in {"class", "def", "from", "global", "lambda", "pass", "return"}:
        cleaned = f"{cleaned}_"
    return cleaned


def _make_registry_tool_wrapper(tool_name: str):
    def _tool(**kwargs):
        from tools.registry import registry
        return registry.dispatch(tool_name, dict(kwargs))

    _tool.__name__ = f"_mcp_registry_tool_{_safe_identifier(tool_name)}"
    return _tool


def _apply_schema_signature(wrapper, schema: dict) -> Any:
    parameters_schema = schema.get("parameters") or {}
    properties = parameters_schema.get("properties") or {}
    required = set(parameters_schema.get("required") or [])
    params = []
    name_map: dict[str, str] = {}

    for raw_name, prop in properties.items():
        param_name = _safe_identifier(str(raw_name))
        while param_name in name_map:
            param_name = f"{param_name}_"
        name_map[param_name] = str(raw_name)
        default = inspect.Parameter.empty if raw_name in required else None
        params.append(
            inspect.Parameter(
                param_name,
                inspect.Parameter.KEYWORD_ONLY,
                default=default,
                annotation=_schema_property_to_annotation(prop if isinstance(prop, dict) else {}),
            )
        )

    if name_map:
        original = wrapper

        def _mapped_tool(**kwargs):
            return original(**{name_map.get(key, key): value for key, value in kwargs.items()})

        _mapped_tool.__name__ = wrapper.__name__
        wrapper = _mapped_tool

    wrapper.__signature__ = inspect.Signature(params)  # type: ignore[attr-defined]
    return wrapper


def _register_registry_tools_as_mcp(
    mcp,
    *,
    expose_toolsets: Optional[Sequence[str]] = None,
    expose_tools: Optional[Sequence[str]] = None,
    expose_plugin_tools: bool = False,
) -> List[str]:
    include_toolsets: Set[str] = set(_comma_values(expose_toolsets))
    include_tools: Set[str] = set(_comma_values(expose_tools))
    if not include_toolsets and not include_tools and not expose_plugin_tools:
        return []

    try:
        from tools.registry import registry
        from hermes_cli.plugins import discover_plugins, get_plugin_manager
        discover_plugins()
    except Exception as exc:
        logger.warning("Could not discover plugin tools for MCP serving: %s", exc)
        return []

    plugin_tool_names = set(getattr(get_plugin_manager(), "_plugin_tool_names", set()))
    registered: List[str] = []
    for name in registry.get_all_tool_names():
        entry = registry.get_entry(name)
        if not entry:
            continue
        should_expose = (
            name in include_tools
            or entry.toolset in include_toolsets
            or (expose_plugin_tools and name in plugin_tool_names)
        )
        if not should_expose:
            continue
        if entry.check_fn:
            try:
                if not entry.check_fn():
                    continue
            except Exception:
                logger.debug("Skipping MCP-exposed tool %s because check_fn failed", name, exc_info=True)
                continue

        schema = {**(entry.schema or {}), "name": entry.name}
        wrapper = _make_registry_tool_wrapper(entry.name)
        wrapper = _apply_schema_signature(wrapper, schema)
        mcp.add_tool(
            wrapper,
            name=entry.name,
            description=entry.description or schema.get("description") or "Hermes tool",
        )
        tool = mcp._tool_manager.get_tool(entry.name)
        if tool is not None:
            tool.parameters = schema.get("parameters") or tool.parameters
        registered.append(entry.name)

    if registered:
        logger.info("Exposed %d Hermes registry tools over MCP", len(registered))
    return registered


def create_mcp_server(
    event_bridge: Optional[EventBridge] = None,
    *,
    expose_toolsets: Optional[Sequence[str]] = None,
    expose_tools: Optional[Sequence[str]] = None,
    expose_plugin_tools: bool = False,
) -> "FastMCP":
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

    _register_registry_tools_as_mcp(
        mcp,
        expose_toolsets=expose_toolsets,
        expose_tools=expose_tools,
        expose_plugin_tools=expose_plugin_tools,
    )

    return mcp


# ---------------------------------------------------------------------------
# Streamable HTTP authentication helpers
# ---------------------------------------------------------------------------

@dataclass
class McpHttpAuthConfig:
    """Authentication and OAuth-compatibility settings for HTTP MCP serving."""

    psk: Optional[str] = None
    psk_header: str = "X-Hermes-MCP-PSK"
    allow_query_token: bool = False
    oauth_compatible: bool = False
    oauth_client_id: Optional[str] = None
    oauth_client_secret: Optional[str] = None
    public_base_url: str = "http://127.0.0.1:8666"
    path: str = "/mcp"
    token_ttl_seconds: int = 2_592_000
    code_ttl_seconds: int = 300

    def __post_init__(self) -> None:
        self.path = _normalize_path(self.path)
        self.public_base_url = self.public_base_url.rstrip("/")
        if self.oauth_compatible and not self.oauth_client_id:
            self.oauth_client_id = self.psk


@dataclass
class _IssuedToken:
    token: str
    client_id: str
    expires_at: float


@dataclass
class _AuthorizationCode:
    code: str
    client_id: str
    redirect_uri: str
    expires_at: float


class _OAuthTokenStore:
    """Small in-memory token/code store for single-process HTTP MCP servers."""

    def __init__(self) -> None:
        self._tokens: Dict[str, _IssuedToken] = {}
        self._codes: Dict[str, _AuthorizationCode] = {}
        self._lock = threading.Lock()

    def issue_token(self, client_id: str, ttl_seconds: int) -> str:
        token = secrets.token_urlsafe(32)
        with self._lock:
            self._tokens[token] = _IssuedToken(
                token=token,
                client_id=client_id,
                expires_at=time.time() + ttl_seconds,
            )
        return token

    def validate_token(self, token: str) -> bool:
        with self._lock:
            issued = self._tokens.get(token)
            if not issued:
                return False
            if issued.expires_at < time.time():
                self._tokens.pop(token, None)
                return False
            return True

    def issue_code(self, client_id: str, redirect_uri: str, ttl_seconds: int) -> str:
        code = secrets.token_urlsafe(24)
        with self._lock:
            self._codes[code] = _AuthorizationCode(
                code=code,
                client_id=client_id,
                redirect_uri=redirect_uri,
                expires_at=time.time() + ttl_seconds,
            )
        return code

    def consume_code(self, code: str, client_id: str) -> bool:
        with self._lock:
            issued = self._codes.pop(code, None)
            if not issued:
                return False
            if issued.client_id != client_id:
                return False
            return issued.expires_at >= time.time()


def _normalize_path(path: str) -> str:
    path = (path or "/mcp").strip()
    if not path.startswith("/"):
        path = f"/{path}"
    if len(path) > 1:
        path = path.rstrip("/")
    return path


def _read_env_secret(env_name: Optional[str]) -> Optional[str]:
    if not env_name:
        return None
    value = os.environ.get(env_name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _comma_values(values: Optional[Sequence[str]]) -> List[str]:
    result: List[str] = []
    for item in values or []:
        for part in str(item).split(","):
            part = part.strip()
            if part:
                result.append(part)
    return result


def _extract_bearer_token(headers) -> Optional[str]:
    auth = headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        token = auth[7:].strip()
        return token or None
    return None


def _constant_time_equals(left: Optional[str], right: Optional[str]) -> bool:
    if not left or not right:
        return False
    return secrets.compare_digest(str(left), str(right))


def _client_secret_valid(config: McpHttpAuthConfig, supplied: Optional[str]) -> bool:
    if not config.oauth_client_secret:
        return True
    return _constant_time_equals(config.oauth_client_secret, supplied)


def _client_id_valid(config: McpHttpAuthConfig, supplied: Optional[str]) -> bool:
    return _constant_time_equals(config.oauth_client_id, supplied)


def _metadata_url(config: McpHttpAuthConfig) -> str:
    return f"{config.public_base_url}/.well-known/oauth-protected-resource{config.path}"


def _auth_challenge(config: McpHttpAuthConfig) -> str:
    if config.oauth_compatible:
        return f'Bearer resource_metadata="{_metadata_url(config)}"'
    return "Bearer"


def _is_authenticated(request, config: McpHttpAuthConfig, store: _OAuthTokenStore) -> bool:
    bearer = _extract_bearer_token(request.headers)
    if bearer:
        if _constant_time_equals(config.psk, bearer):
            return True
        if config.oauth_compatible and store.validate_token(bearer):
            return True

    header_token = request.headers.get(config.psk_header)
    if _constant_time_equals(config.psk, header_token):
        return True

    if config.allow_query_token:
        query_token = request.query_params.get("access_token") or request.query_params.get("psk")
        if _constant_time_equals(config.psk, query_token):
            return True
        if config.oauth_compatible and query_token and store.validate_token(query_token):
            return True

    return False


def _configure_transport_security(server, host: str, allowed_hosts: Sequence[str], allowed_origins: Sequence[str]) -> None:
    hosts = list(dict.fromkeys([host, "127.0.0.1", "localhost", *allowed_hosts]))
    origins = list(dict.fromkeys([*allowed_origins]))
    if origins and "null" not in origins:
        origins.append("null")
    try:
        from mcp.server.transport_security import TransportSecuritySettings
        server.settings.transport_security = TransportSecuritySettings(
            allowed_hosts=hosts,
            allowed_origins=origins,
        )
    except Exception as exc:  # pragma: no cover - depends on MCP SDK version
        logger.debug("Could not configure MCP transport security: %s", exc)


def create_streamable_http_app(
    server,
    *,
    auth_config: Optional[McpHttpAuthConfig] = None,
    health_path: str = "/health",
):
    """Create the Starlette app for Streamable HTTP, including optional auth.

    The returned app serves the MCP endpoint at ``server.settings.streamable_http_path``.
    If ``auth_config`` is provided, the MCP path requires either a PSK bearer/header
    token or an OAuth-compatible bearer token issued by the local token endpoint.
    """
    if not hasattr(server, "streamable_http_app"):
        raise RuntimeError("Installed MCP SDK does not support Streamable HTTP")

    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.responses import JSONResponse, PlainTextResponse, RedirectResponse

    store = _OAuthTokenStore()
    app = server.streamable_http_app()
    health_path = _normalize_path(health_path)
    config = auth_config

    async def health(_request):
        return PlainTextResponse("ok")

    app.add_route(health_path, health, methods=["GET", "HEAD"])

    if config and config.oauth_compatible:
        async def authorization_server_metadata(_request):
            return JSONResponse({
                "issuer": config.public_base_url,
                "authorization_endpoint": f"{config.public_base_url}{config.path}/authorize",
                "token_endpoint": f"{config.public_base_url}{config.path}/token",
                "response_types_supported": ["code"],
                "grant_types_supported": ["authorization_code", "client_credentials"],
                "token_endpoint_auth_methods_supported": ["none", "client_secret_post"],
                "code_challenge_methods_supported": ["plain", "S256"],
                "scopes_supported": ["mcp"],
            })

        async def protected_resource_metadata(_request):
            return JSONResponse({
                "resource": f"{config.public_base_url}{config.path}",
                "authorization_servers": [config.public_base_url],
                "bearer_methods_supported": ["header"],
                "scopes_supported": ["mcp"],
            })

        async def authorize(request):
            params = request.query_params
            client_id = params.get("client_id")
            redirect_uri = params.get("redirect_uri")
            response_type = params.get("response_type")
            state = params.get("state")

            if response_type != "code":
                return JSONResponse({"error": "unsupported_response_type"}, status_code=400)
            if not _client_id_valid(config, client_id):
                return JSONResponse({"error": "invalid_client"}, status_code=401)
            if not redirect_uri:
                return JSONResponse({"error": "invalid_request", "error_description": "redirect_uri is required"}, status_code=400)

            code = store.issue_code(client_id or "", redirect_uri, config.code_ttl_seconds)
            query = {"code": code}
            if state is not None:
                query["state"] = state
            separator = "&" if "?" in redirect_uri else "?"
            return RedirectResponse(f"{redirect_uri}{separator}{urlencode(query)}", status_code=302)

        async def token(request):
            body = (await request.body()).decode("utf-8")
            parsed = parse_qs(body, keep_blank_values=True)
            form = {key: values[-1] if values else "" for key, values in parsed.items()}
            grant_type = form.get("grant_type")
            client_id = form.get("client_id")
            client_secret = form.get("client_secret")

            if not _client_id_valid(config, client_id) or not _client_secret_valid(config, client_secret):
                return JSONResponse({"error": "invalid_client"}, status_code=401)

            if grant_type == "client_credentials":
                access_token = store.issue_token(client_id or "", config.token_ttl_seconds)
            elif grant_type == "authorization_code":
                code = form.get("code")
                if not store.consume_code(str(code or ""), str(client_id or "")):
                    return JSONResponse({"error": "invalid_grant"}, status_code=400)
                access_token = store.issue_token(client_id or "", config.token_ttl_seconds)
            else:
                return JSONResponse({"error": "unsupported_grant_type"}, status_code=400)

            return JSONResponse({
                "access_token": access_token,
                "token_type": "Bearer",
                "expires_in": config.token_ttl_seconds,
                "scope": "mcp",
            })

        app.add_route("/.well-known/oauth-authorization-server", authorization_server_metadata, methods=["GET"])
        app.add_route(f"/.well-known/oauth-protected-resource{config.path}", protected_resource_metadata, methods=["GET"])
        app.add_route(f"{config.path}/authorize", authorize, methods=["GET"])
        app.add_route(f"{config.path}/token", token, methods=["POST"])

    if config:
        class McpAuthMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):
                path = request.url.path.rstrip("/") or "/"
                mcp_path = config.path.rstrip("/") or "/"
                if path == mcp_path and not _is_authenticated(request, config, store):
                    return JSONResponse(
                        {
                            "error": "unauthorized",
                            "hint": "Authenticate with OAuth bearer token, bearer/header PSK, or an enabled query token.",
                        },
                        status_code=401,
                        headers={"WWW-Authenticate": _auth_challenge(config)},
                    )
                return await call_next(request)

        app.add_middleware(McpAuthMiddleware)

    return app


def run_mcp_http_server(
    *,
    verbose: bool = False,
    host: str = "127.0.0.1",
    port: int = 8666,
    path: str = "/mcp",
    public_base_url: Optional[str] = None,
    auth_token_env: Optional[str] = None,
    auth_header: str = "X-Hermes-MCP-PSK",
    allow_query_token: bool = False,
    oauth_compatible: bool = False,
    oauth_client_id_env: Optional[str] = None,
    oauth_client_secret_env: Optional[str] = None,
    token_ttl_seconds: int = 2_592_000,
    code_ttl_seconds: int = 300,
    allowed_hosts: Optional[Sequence[str]] = None,
    allowed_origins: Optional[Sequence[str]] = None,
    expose_toolsets: Optional[Sequence[str]] = None,
    expose_tools: Optional[Sequence[str]] = None,
    expose_plugin_tools: bool = False,
    health_path: str = "/health",
) -> None:
    """Start the Hermes MCP server over Streamable HTTP."""
    if not _MCP_SERVER_AVAILABLE:
        print(
            "Error: MCP HTTP server requires the 'mcp' package.\n"
            f"Install with: {sys.executable} -m pip install 'mcp'",
            file=sys.stderr,
        )
        sys.exit(1)

    logging.basicConfig(level=logging.DEBUG if verbose else logging.WARNING, stream=sys.stderr)

    path = _normalize_path(path)
    base_url = (public_base_url or f"http://{host}:{port}").rstrip("/")
    psk = _read_env_secret(auth_token_env)
    oauth_client_id = _read_env_secret(oauth_client_id_env) if oauth_client_id_env else None
    oauth_client_secret = _read_env_secret(oauth_client_secret_env) if oauth_client_secret_env else None

    if (auth_token_env or oauth_compatible) and not psk and not oauth_client_id:
        print(
            "Error: HTTP auth requested but no auth token/client id was found in the configured environment variables.",
            file=sys.stderr,
        )
        sys.exit(1)

    bridge = EventBridge()
    bridge.start()
    server = create_mcp_server(
        event_bridge=bridge,
        expose_toolsets=expose_toolsets,
        expose_tools=expose_tools,
        expose_plugin_tools=expose_plugin_tools,
    )
    server.settings.host = host
    server.settings.port = int(port)
    server.settings.streamable_http_path = path
    server.settings.json_response = True
    _configure_transport_security(
        server,
        host,
        _comma_values(allowed_hosts),
        _comma_values(allowed_origins),
    )

    auth_config = None
    if psk or oauth_compatible:
        auth_config = McpHttpAuthConfig(
            psk=psk,
            psk_header=auth_header,
            allow_query_token=allow_query_token,
            oauth_compatible=oauth_compatible,
            oauth_client_id=oauth_client_id,
            oauth_client_secret=oauth_client_secret,
            public_base_url=base_url,
            path=path,
            token_ttl_seconds=token_ttl_seconds,
            code_ttl_seconds=code_ttl_seconds,
        )

    app = create_streamable_http_app(server, auth_config=auth_config, health_path=health_path)

    import uvicorn

    try:
        uvicorn.run(
            app,
            host=host,
            port=int(port),
            log_level="debug" if verbose else "warning",
            access_log=not allow_query_token,
        )
    finally:
        bridge.stop()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_mcp_server(
    verbose: bool = False,
    *,
    expose_toolsets: Optional[Sequence[str]] = None,
    expose_tools: Optional[Sequence[str]] = None,
    expose_plugin_tools: bool = False,
) -> None:
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

    server = create_mcp_server(
        event_bridge=bridge,
        expose_toolsets=expose_toolsets,
        expose_tools=expose_tools,
        expose_plugin_tools=expose_plugin_tools,
    )

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

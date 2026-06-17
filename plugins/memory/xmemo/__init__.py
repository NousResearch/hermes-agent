"""XMemo memory provider plugin for Hermes Agent.

Provides user-owned cloud memory via XMemo's REST API: orchestrated recall,
semantic search, durable fact storage, working state, reminders, and session
snapshots.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

from .client import XMemoClient
from .config import load_config, save_config

logger = logging.getLogger(__name__)

# Circuit breaker: pause API calls after consecutive failures.
_BREAKER_THRESHOLD = 5
_BREAKER_COOLDOWN_SECONDS = 120

# Max time prefetch() may wait for an in-flight background recall. Keep this
# short because prefetch() runs on the API-call critical path.
_PREFETCH_JOIN_TIMEOUT_SECONDS = 0.25


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

SEARCH_SCHEMA = {
    "name": "xmemo_search",
    "description": (
        "Search XMemo memories by natural-language query. "
        "Returns relevant facts ranked by semantic similarity. "
        "Use this when the user asks about saved information or when prior "
        "context could change the answer."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to search for.",
            },
            "limit": {
                "type": "integer",
                "description": "Max results (default 5, max 20).",
            },
            "memory_type": {
                "type": "string",
                "description": "Optional memory type filter (e.g. semantic, episodic, working).",
            },
        },
        "required": ["query"],
    },
}

REMEMBER_SCHEMA = {
    "name": "xmemo_remember",
    "description": (
        "Save a durable fact to XMemo. Use for explicit preferences, decisions, "
        "conventions, architecture notes, action items, or bug-fix context that "
        "should survive across sessions. Skip transient chat."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The fact to remember. One clear concept per call.",
            },
            "path": {
                "type": "string",
                "description": "Logical path or category, e.g. 'notes/decisions' or 'hermes/preferences'.",
            },
            "memory_type": {
                "type": "string",
                "description": "Memory type: semantic, episodic, procedural, working, identity (default semantic).",
            },
            "importance": {
                "type": "number",
                "description": "Importance from 0.0 to 1.0 (default 0.7).",
            },
        },
        "required": ["content", "path"],
    },
}

UPDATE_STATE_SCHEMA = {
    "name": "xmemo_update_state",
    "description": (
        "Save the current working state to XMemo with TTL. Use for active task, "
        "next action, or blocker during long-running work so future sessions can resume."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "current_task": {
                "type": "string",
                "description": "Short description of the active task.",
            },
            "next_action": {
                "type": "string",
                "description": "The very next step the agent should take.",
            },
            "blocked_reason": {
                "type": "string",
                "description": "Why work is blocked, if applicable.",
            },
            "ttl_seconds": {
                "type": "integer",
                "description": "Time-to-live in seconds (default 86400 = 1 day).",
            },
        },
        "required": [],
    },
}

RECALL_CONTEXT_SCHEMA = {
    "name": "xmemo_recall_context",
    "description": (
        "Build a bounded, ranked context pack from XMemo memories. "
        "Use when you need a focused memory summary rather than raw search results."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to recall context for.",
            },
            "max_items": {
                "type": "integer",
                "description": "Max memory items to include (default 5, max 20).",
            },
            "memory_type": {
                "type": "string",
                "description": "Optional memory type filter (semantic, episodic, working, identity, procedural).",
            },
        },
        "required": ["query"],
    },
}

RECORD_EVENT_SCHEMA = {
    "name": "xmemo_record_event",
    "description": (
        "Record a significant session event, milestone, decision, or handoff note "
        "to the XMemo timeline. Use for durable audit-style notes, not transient chat."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The event note to save.",
            },
            "event_type": {
                "type": "string",
                "description": "Event type: event, milestone, decision, handoff (default event).",
            },
        },
        "required": ["content"],
    },
}

CREATE_REMINDER_SCHEMA = {
    "name": "xmemo_create_reminder",
    "description": (
        "Create a TODO or action item in XMemo to revisit later. "
        "Use when the user asks you to follow up, save a task, or remind them."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "What to remember to do.",
            },
            "due_at": {
                "type": "string",
                "description": "Optional due time as ISO 8601 string.",
            },
        },
        "required": ["content"],
    },
}

LIST_REMINDERS_SCHEMA = {
    "name": "xmemo_list_reminders",
    "description": (
        "List XMemo TODO/action items. Use when the user asks what tasks, follow-ups, "
        "or reminders are pending or done."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "item_status": {
                "type": "string",
                "description": "Filter by status: open or completed (default open).",
            },
            "limit": {
                "type": "integer",
                "description": "Max results (default 20).",
            },
        },
        "required": [],
    },
}

COMPLETE_REMINDER_SCHEMA = {
    "name": "xmemo_complete_reminder",
    "description": (
        "Mark a XMemo TODO/action item as completed. "
        "Use when the user says a saved task is done, resolved, or no longer needed."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "todo_id": {
                "type": "string",
                "description": "The exact TODO item ID from xmemo_list_reminders.",
            },
            "note": {
                "type": "string",
                "description": "Optional completion note.",
            },
        },
        "required": ["todo_id"],
    },
}

MARK_USED_SCHEMA = {
    "name": "xmemo_mark_used",
    "description": (
        "Tell XMemo that a recalled memory influenced the current answer. "
        "Call this after using a specific memory returned by xmemo_search or "
        "xmemo_recall_context."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {
                "type": "string",
                "description": "Exact memory ID returned by xmemo_search.",
            },
            "context": {
                "type": "string",
                "description": "Optional short note on how the memory was used.",
            },
        },
        "required": ["memory_id"],
    },
}

FORGET_SCHEMA = {
    "name": "xmemo_forget",
    "description": (
        "Delete a memory from XMemo. Use only when the user explicitly asks to "
        "forget or remove a specific saved fact. Requires an exact memory ID."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {
                "type": "string",
                "description": "Exact memory ID from xmemo_search.",
            },
            "reason": {
                "type": "string",
                "description": "Optional reason for deletion.",
            },
        },
        "required": ["memory_id"],
    },
}

# Schemas exposed by default. Workflow/destructive tools are opt-in via config.
_CORE_TOOL_SCHEMAS = [
    RECALL_CONTEXT_SCHEMA,
    SEARCH_SCHEMA,
    REMEMBER_SCHEMA,
    UPDATE_STATE_SCHEMA,
]

_WORKFLOW_TOOL_SCHEMAS = [
    RECORD_EVENT_SCHEMA,
    CREATE_REMINDER_SCHEMA,
    LIST_REMINDERS_SCHEMA,
    COMPLETE_REMINDER_SCHEMA,
]

_FEEDBACK_TOOL_SCHEMAS = [
    MARK_USED_SCHEMA,
]

_DESTRUCTIVE_TOOL_SCHEMAS = [
    FORGET_SCHEMA,
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_trivial_prompt(text: str) -> bool:
    """Skip recall for acknowledgements, slash commands, and empty input."""
    if not text or not text.strip():
        return True
    cleaned = text.strip().lower()
    if cleaned.startswith("/"):
        return True
    return bool(
        re.match(
            r"^(yes|no|ok|okay|sure|thanks|thank you|y|n|yep|nope|yeah|nah|"
            r"continue|go ahead|do it|proceed|got it|cool|nice|great|done|next|lgtm|k)$",
            cleaned,
            re.IGNORECASE,
        )
    )


def _format_search_results(results: List[Dict[str, Any]]) -> str:
    """Format search results into a concise text block."""
    if not results:
        return ""
    lines = []
    for i, item in enumerate(results, 1):
        content = item.get("content", "")
        if not content:
            continue
        memory_type = item.get("memory_type", "semantic")
        path = item.get("path", "")
        score = item.get("similarity") or item.get("score")
        header = f"{i}. [{memory_type}]"
        if path:
            header += f" {path}"
        if score is not None:
            header += f" (sim {score:.3f})"
        lines.append(header)
        lines.append(f"   {content.strip()}")
    return "\n".join(lines)


def _format_recall_context(context: Dict[str, Any]) -> str:
    """Extract context_text from a recall_context response."""
    if not context:
        return ""
    text = context.get("context_text", "")
    if text and text.strip():
        return text.strip()
    items = context.get("items", [])
    if not items:
        return ""
    lines = []
    for i, item in enumerate(items, 1):
        content = item.get("content", "")
        if not content:
            continue
        lines.append(f"{i}. {content.strip()}")
    return "\n".join(lines)


def _session_key(session_id: str) -> str:
    """Normalize session id for cache keys."""
    return session_id or "__default__"


def _as_bool(value: Any) -> bool:
    """Parse bool-like values from JSON/config strings."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _is_high_signal_turn(user_content: str, assistant_content: str) -> bool:
    """Detect turns that likely contain durable facts without LLM extraction."""
    text = f"{user_content} {assistant_content}".lower()
    high_signal_phrases = [
        "remember",
        "save this",
        "write this down",
        "keep in mind",
        "going forward",
        "from now on",
        "we decided",
        "decision:",
        "architecture decision",
        "root cause",
        "fix was",
        "lesson learned",
        "runbook",
        "handoff",
        "blocked by",
        "blocker:",
    ]
    return any(phrase in text for phrase in high_signal_phrases)


def _redact_for_log(text: str, max_len: int = 200) -> str:
    """Truncate and redact sensitive-looking content before storing/logging."""
    if not text:
        return ""
    if len(text) > max_len:
        text = text[:max_len] + "..."
    # Mask likely tokens/keys in logs (best-effort).
    return re.sub(r"\b[a-zA-Z0-9_-]{24,}\b", "[REDACTED]", text)


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

class XMemoMemoryProvider(MemoryProvider):
    """XMemo cloud memory provider for Hermes Agent."""

    def __init__(self):
        self._config: Dict[str, Any] = {}
        self._client: Optional[XMemoClient] = None
        self._client_lock = threading.Lock()

        # Per-session prefetch cache
        self._prefetch_results: Dict[str, str] = {}
        self._prefetch_threads: Dict[str, threading.Thread] = {}
        self._prefetch_lock = threading.Lock()

        # Background worker references for clean shutdown
        self._snapshot_thread: Optional[threading.Thread] = None

        # Circuit breaker state
        self._consecutive_failures = 0
        self._breaker_open_until = 0.0

        # Session / runtime metadata
        self._session_id = ""
        self._turn_count = 0
        self._agent_context = "primary"
        self._auto_write_enabled = True

    @property
    def name(self) -> str:
        return "xmemo"

    def is_available(self) -> bool:
        """Check if XMemo is configured. No network calls and no file writes."""
        try:
            cfg = load_config(create_instance=False)
            return bool(cfg.get("api_key"))
        except Exception:
            return False

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "api_key",
                "description": "XMemo service token",
                "secret": True,
                "required": True,
                "env_var": "XMEMO_KEY",
                "url": "https://xmemo.dev",
            },
            {
                "key": "base_url",
                "description": "XMemo service URL",
                "default": "https://xmemo.dev",
            },
            {
                "key": "agent_id",
                "description": "Agent family identifier",
                "default": "hermes",
            },
            {
                "key": "bucket",
                "description": "Default storage namespace",
                "default": "work",
                "choices": ["work", "private", "public"],
            },
            {
                "key": "scope",
                "description": "Default project/session scope",
                "default": "hermes/default",
            },
            {
                "key": "timeout_seconds",
                "description": "REST request timeout",
                "default": "5.0",
            },
            {
                "key": "enable_workflow_tools",
                "description": "Expose reminder/event workflow tools",
                "default": "false",
                "choices": ["true", "false"],
            },
            {
                "key": "enable_destructive_tools",
                "description": "Expose the xmemo_forget destructive tool",
                "default": "false",
                "choices": ["true", "false"],
            },
            {
                "key": "capture_timeline",
                "description": "Record high-signal turns to the XMemo timeline",
                "default": "false",
                "choices": ["true", "false"],
            },
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        """Write non-secret config to $HERMES_HOME/xmemo.json."""
        save_config(values, hermes_home=hermes_home)

    def post_setup(self, hermes_home: str, config: Dict[str, Any]) -> None:
        """Run the full XMemo setup wizard after provider selection."""
        from .cli import cmd_setup
        cmd_setup(provider=self, hermes_home=hermes_home, config=config)

    def _is_breaker_open(self) -> bool:
        if self._consecutive_failures < _BREAKER_THRESHOLD:
            return False
        if time.monotonic() >= self._breaker_open_until:
            self._consecutive_failures = 0
            return False
        return True

    def _record_success(self) -> None:
        self._consecutive_failures = 0

    def _record_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= _BREAKER_THRESHOLD:
            self._breaker_open_until = time.monotonic() + _BREAKER_COOLDOWN_SECONDS
            logger.warning(
                "XMemo circuit breaker tripped after %d consecutive failures. "
                "Pausing API calls for %ds.",
                self._consecutive_failures,
                _BREAKER_COOLDOWN_SECONDS,
            )

    def _get_client(self) -> XMemoClient:
        """Thread-safe client accessor with lazy initialization."""
        with self._client_lock:
            if self._client is not None:
                return self._client
            self._client = XMemoClient(
                base_url=self._config.get("base_url", "https://xmemo.dev"),
                api_key=self._config.get("api_key", ""),
                agent_id=self._config.get("agent_id", "hermes"),
                agent_instance_id=self._config.get("agent_instance_id", ""),
                timeout=float(self._config.get("timeout_seconds", 5.0)),
            )
            return self._client

    def initialize(self, session_id: str, **kwargs) -> None:
        """Initialize XMemo provider for a session."""
        self._config = load_config(create_instance=True)
        self._session_id = session_id or ""
        self._turn_count = 0

        self._agent_context = kwargs.get("agent_context", "primary") or "primary"
        self._auto_write_enabled = self._agent_context == "primary"

        # Scope per-profile if the active Hermes profile differs from default
        profile = kwargs.get("agent_identity") or "default"
        configured_scope = self._config.get("scope", "hermes/default")
        if configured_scope == "hermes/default" and profile != "default":
            self._config["scope"] = f"hermes/{profile}"
            try:
                save_config(self._config)
            except Exception:
                pass

        if not self._config.get("api_key"):
            logger.debug("XMemo not configured — plugin inactive")
            return

        # Optional lightweight health check; failure does not block startup.
        try:
            client = self._get_client()
            client.health()
            self._record_success()
        except Exception as exc:
            logger.debug("XMemo health check failed (non-blocking): %s", exc)

    def system_prompt_block(self) -> str:
        """Return static provider instructions for the system prompt."""
        if not self._config.get("api_key"):
            return ""
        scope = self._config.get("scope", "hermes/default")
        return (
            "# XMemo Memory\n"
            "Active. User-owned cloud memory is available.\n"
            f"Scope: {scope}.\n"
            "Use xmemo_search to recall saved facts before answering. "
            "Use xmemo_remember to store durable facts (preferences, decisions, conventions, action items). "
            "Use xmemo_update_state to save the current task state with TTL."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Return prefetched XMemo context for the upcoming turn."""
        if self._is_breaker_open():
            return ""

        if _is_trivial_prompt(query):
            return ""

        key = _session_key(session_id or self._session_id)
        thread = self._prefetch_threads.get(key)
        if thread and thread.is_alive():
            thread.join(timeout=_PREFETCH_JOIN_TIMEOUT_SECONDS)

        with self._prefetch_lock:
            result = self._prefetch_results.pop(key, "")

        if not result:
            return ""
        return f"## XMemo Memory\n{result}"

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Fire a background recall for the next turn."""
        if self._is_breaker_open():
            return
        if not self._config.get("api_key"):
            return
        if _is_trivial_prompt(query):
            return

        key = _session_key(session_id or self._session_id)

        # Guard against a hung prior thread for this session
        prior = self._prefetch_threads.get(key)
        if prior and prior.is_alive():
            logger.debug("XMemo prefetch skipped: prior thread still running for session %s", key)
            return

        def _run() -> None:
            try:
                client = self._get_client()
                context = client.recall_context(
                    query=query,
                    bucket=self._config.get("bucket", "work"),
                    scope=self._config.get("scope", "hermes/default"),
                    max_items=int(self._config.get("prefetch_max_items", 5)),
                    max_tokens=int(self._config.get("prefetch_max_tokens", 900)),
                    prefer_working=True,
                )
                text = _format_recall_context(context)
                if text:
                    with self._prefetch_lock:
                        self._prefetch_results[key] = text
                self._record_success()
            except Exception as exc:
                self._record_failure()
                logger.debug("XMemo prefetch failed: %s", exc)

        t = threading.Thread(target=_run, daemon=True, name=f"xmemo-prefetch-{key}")
        with self._prefetch_lock:
            self._prefetch_threads[key] = t
        t.start()

    def sync_turn(
        self, user_content: str, assistant_content: str, *, session_id: str = ""
    ) -> None:
        """Persist a completed turn to XMemo if it is high-signal."""
        if self._is_breaker_open():
            return
        if not self._config.get("api_key"):
            return
        if not self._auto_write_enabled:
            return

        self._turn_count += 1

        # Automatic timeline writes are opt-in only. When disabled, do not record
        # any turn — even high-signal ones — to avoid surprising privacy behavior.
        if not _as_bool(self._config.get("capture_timeline", False)):
            return

        # When enabled, still only persist high-signal turns to avoid noise.
        if not _is_high_signal_turn(user_content, assistant_content):
            return

        # Defensive truncation to avoid storing long raw outputs or secrets.
        safe_user = _redact_for_log(user_content, max_len=240)
        summary = f"Turn {self._turn_count}: {safe_user[:120]}..."

        try:
            client = self._get_client()
            client.record_event(
                content=summary,
                event_type="session_event",
                bucket=self._config.get("bucket", "work"),
                scope=self._config.get("scope", "hermes/default"),
                session_id=session_id or self._session_id,
            )
            self._record_success()
        except Exception as exc:
            self._record_failure()
            logger.debug("XMemo sync_turn failed: %s", exc)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        # Must be callable BEFORE initialize() because MemoryManager.add_provider()
        # indexes tool names for routing immediately after loading. Read config
        # from disk if we have not been initialized yet.
        cfg = self._config if self._config else load_config(create_instance=False)
        schemas = list(_CORE_TOOL_SCHEMAS)
        if _as_bool(cfg.get("enable_workflow_tools", False)):
            schemas.extend(_WORKFLOW_TOOL_SCHEMAS)
        if _as_bool(cfg.get("enable_destructive_tools", False)):
            schemas.extend(_DESTRUCTIVE_TOOL_SCHEMAS)
        # Feedback tools remain internal by default; can be exposed via config later.
        return schemas

    def handle_tool_call(
        self, tool_name: str, args: Dict[str, Any], **kwargs
    ) -> str:
        """Route a tool call to the correct XMemo API."""
        if self._is_breaker_open():
            return json.dumps({
                "error": "XMemo API temporarily unavailable after multiple failures. Will retry automatically."
            })

        try:
            client = self._get_client()
        except Exception as exc:
            return tool_error(str(exc))

        if tool_name == "xmemo_search":
            return self._handle_search(client, args)
        if tool_name == "xmemo_remember":
            return self._handle_remember(client, args)
        if tool_name == "xmemo_update_state":
            return self._handle_update_state(client, args)
        if tool_name == "xmemo_recall_context":
            return self._handle_recall_context(client, args)
        if tool_name == "xmemo_record_event":
            return self._handle_record_event(client, args)
        if tool_name == "xmemo_create_reminder":
            return self._handle_create_reminder(client, args)
        if tool_name == "xmemo_list_reminders":
            return self._handle_list_reminders(client, args)
        if tool_name == "xmemo_complete_reminder":
            return self._handle_complete_reminder(client, args)
        if tool_name == "xmemo_mark_used":
            return self._handle_mark_used(client, args)
        if tool_name == "xmemo_forget":
            return self._handle_forget(client, args)

        return tool_error(f"Unknown XMemo tool: {tool_name}")

    def _handle_search(self, client: XMemoClient, args: Dict[str, Any]) -> str:
        query = args.get("query", "").strip()
        if not query:
            return tool_error("Missing required parameter: query")

        try:
            limit = min(int(args.get("limit", 5)), 20)
        except (ValueError, TypeError):
            limit = 5
        memory_type = args.get("memory_type", "%")

        try:
            results = client.search(
                query=query,
                bucket=self._config.get("bucket", "work"),
                scope=self._config.get("scope", "hermes/default"),
                memory_type=memory_type,
                limit=limit,
            )
            self._record_success()
            if not results:
                return json.dumps({"result": "No relevant XMemo memories found."})
            return json.dumps({
                "results": results,
                "formatted": _format_search_results(results),
                "count": len(results),
            })
        except Exception as exc:
            self._record_failure()
            return tool_error(f"XMemo search failed: {exc}")

    def _handle_remember(self, client: XMemoClient, args: Dict[str, Any]) -> str:
        content = args.get("content", "").strip()
        path = args.get("path", "").strip()
        if not content:
            return tool_error("Missing required parameter: content")
        if not path:
            return tool_error("Missing required parameter: path")

        memory_type = args.get("memory_type", "semantic")
        importance = args.get("importance")
        if importance is not None:
            try:
                importance = float(importance)
            except (ValueError, TypeError):
                importance = None

        try:
            result = client.remember(
                content=content,
                path=path,
                bucket=self._config.get("bucket", "work"),
                scope=self._config.get("scope", "hermes/default"),
                memory_type=memory_type,
                importance=importance,
            )
            self._record_success()
            memory_id = result.get("id") if isinstance(result, dict) else None
            return json.dumps({
                "result": "Saved to XMemo.",
                "memory_id": memory_id,
            })
        except Exception as exc:
            self._record_failure()
            return tool_error(f"XMemo remember failed: {exc}")

    def _handle_update_state(self, client: XMemoClient, args: Dict[str, Any]) -> str:
        current_task = args.get("current_task", "").strip()
        next_action = args.get("next_action", "").strip()
        blocked_reason = args.get("blocked_reason", "").strip()
        try:
            ttl_seconds = int(args.get("ttl_seconds", 86400))
        except (ValueError, TypeError):
            ttl_seconds = 86400

        if not any([current_task, next_action, blocked_reason]):
            return tool_error("At least one of current_task, next_action, or blocked_reason is required")

        try:
            result = client.update_state(
                current_task=current_task,
                next_action=next_action,
                blocked_reason=blocked_reason,
                bucket=self._config.get("bucket", "work"),
                scope=self._config.get("scope", "hermes/default"),
                ttl_seconds=ttl_seconds,
            )
            self._record_success()
            return json.dumps({
                "result": "Working state saved to XMemo.",
                "state_key": result.get("state_key") if isinstance(result, dict) else None,
                "id": result.get("id") if isinstance(result, dict) else None,
            })
        except Exception as exc:
            self._record_failure()
            return tool_error(f"XMemo update_state failed: {exc}")

    def _handle_recall_context(self, client: XMemoClient, args: Dict[str, Any]) -> str:
        query = args.get("query", "").strip()
        if not query:
            return tool_error("Missing required parameter: query")

        try:
            max_items = min(int(args.get("max_items", 5)), 20)
        except (ValueError, TypeError):
            max_items = 5
        memory_type = args.get("memory_type", "auto")

        try:
            context = client.recall_context(
                query=query,
                bucket=self._config.get("bucket", "work"),
                scope=self._config.get("scope", "hermes/default"),
                max_items=max_items,
                max_tokens=int(self._config.get("prefetch_max_tokens", 900)),
                memory_type=memory_type,
                prefer_working=True,
            )
            self._record_success()
            text = _format_recall_context(context)
            if not text:
                return json.dumps({"result": "No relevant XMemo context found."})
            return json.dumps({
                "context": text,
                "items": context.get("items", []) if isinstance(context, dict) else [],
            })
        except Exception as exc:
            self._record_failure()
            return tool_error(f"XMemo recall_context failed: {exc}")

    def _handle_record_event(self, client: XMemoClient, args: Dict[str, Any]) -> str:
        content = args.get("content", "").strip()
        if not content:
            return tool_error("Missing required parameter: content")

        event_type = args.get("event_type", "event").strip() or "event"

        try:
            result = client.record_event(
                content=content,
                event_type=event_type,
                bucket=self._config.get("bucket", "work"),
                scope=self._config.get("scope", "hermes/default"),
                session_id=self._session_id,
            )
            self._record_success()
            return json.dumps({
                "result": "Event recorded in XMemo timeline.",
                "event_id": result.get("id") if isinstance(result, dict) else None,
            })
        except Exception as exc:
            self._record_failure()
            return tool_error(f"XMemo record_event failed: {exc}")

    def _handle_create_reminder(self, client: XMemoClient, args: Dict[str, Any]) -> str:
        content = args.get("content", "").strip()
        if not content:
            return tool_error("Missing required parameter: content")

        due_at = args.get("due_at", "").strip()

        try:
            result = client.create_reminder(
                content=content,
                due_at=due_at,
                bucket=self._config.get("bucket", "work"),
                scope=self._config.get("scope", "hermes/default"),
                session_id=self._session_id,
            )
            self._record_success()
            return json.dumps({
                "result": "Reminder saved to XMemo.",
                "todo_id": result.get("id") if isinstance(result, dict) else None,
            })
        except Exception as exc:
            self._record_failure()
            return tool_error(f"XMemo create_reminder failed: {exc}")

    def _handle_list_reminders(self, client: XMemoClient, args: Dict[str, Any]) -> str:
        item_status = args.get("item_status", "open") or "open"
        try:
            limit = min(int(args.get("limit", 20)), 100)
        except (ValueError, TypeError):
            limit = 20

        try:
            items = client.list_reminders(
                bucket=self._config.get("bucket", "work"),
                scope=self._config.get("scope", "hermes/default"),
                item_status=item_status,
                limit=limit,
            )
            self._record_success()
            if not items:
                return json.dumps({"result": f"No {item_status} XMemo reminders found."})
            return json.dumps({
                "items": items,
                "count": len(items),
            })
        except Exception as exc:
            self._record_failure()
            return tool_error(f"XMemo list_reminders failed: {exc}")

    def _handle_complete_reminder(self, client: XMemoClient, args: Dict[str, Any]) -> str:
        todo_id = args.get("todo_id", "").strip()
        if not todo_id:
            return tool_error("Missing required parameter: todo_id")

        note = args.get("note", "").strip()

        try:
            result = client.complete_reminder(
                todo_id=todo_id,
                note=note,
                bucket=self._config.get("bucket", "work"),
                scope=self._config.get("scope", "hermes/default"),
            )
            self._record_success()
            return json.dumps({
                "result": "Reminder marked completed.",
                "todo_id": result.get("id") if isinstance(result, dict) else todo_id,
            })
        except Exception as exc:
            self._record_failure()
            return tool_error(f"XMemo complete_reminder failed: {exc}")

    def _handle_mark_used(self, client: XMemoClient, args: Dict[str, Any]) -> str:
        memory_id = args.get("memory_id", "").strip()
        if not memory_id:
            return tool_error("Missing required parameter: memory_id")

        context = args.get("context", "").strip()

        try:
            result = client.mark_used(
                memory_id=memory_id,
                context=context,
            )
            self._record_success()
            return json.dumps({
                "result": "Memory usage recorded in XMemo.",
                "memory_id": result.get("id") if isinstance(result, dict) else memory_id,
            })
        except Exception as exc:
            self._record_failure()
            return tool_error(f"XMemo mark_used failed: {exc}")

    def _handle_forget(self, client: XMemoClient, args: Dict[str, Any]) -> str:
        memory_id = args.get("memory_id", "").strip()
        if not memory_id:
            return tool_error("Missing required parameter: memory_id")

        reason = args.get("reason", "").strip()

        try:
            result = client.forget(
                memory_id=memory_id,
                reason=reason,
                bucket=self._config.get("bucket", "work"),
                scope=self._config.get("scope", "hermes/default"),
            )
            self._record_success()
            return json.dumps({
                "result": "Memory deleted from XMemo.",
                "memory_id": result.get("id") if isinstance(result, dict) else memory_id,
            })
        except Exception as exc:
            self._record_failure()
            return tool_error(f"XMemo forget failed: {exc}")

    def shutdown(self) -> None:
        """Clean shutdown: flush threads and close client."""
        for t in list(self._prefetch_threads.values()):
            if t and t.is_alive():
                t.join(timeout=1.0)
        if self._snapshot_thread and self._snapshot_thread.is_alive():
            self._snapshot_thread.join(timeout=5.0)
        with self._client_lock:
            if self._client is not None:
                try:
                    self._client.close()
                except Exception as exc:
                    logger.debug("XMemo client close failed: %s", exc)
                self._client = None

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        rewound: bool = False,
        **kwargs,
    ) -> None:
        """Update session tracking and clean stale prefetch cache."""
        old_key = _session_key(self._session_id)
        self._session_id = new_session_id or ""

        if reset or rewound:
            with self._prefetch_lock:
                self._prefetch_results.pop(old_key, None)
                old_thread = self._prefetch_threads.pop(old_key, None)
            if old_thread and old_thread.is_alive():
                old_thread.join(timeout=1.0)

        if reset:
            self._turn_count = 0

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Mirror Hermes built-in memory writes to XMemo."""
        if self._is_breaker_open():
            return
        if not self._config.get("api_key"):
            return
        if not self._auto_write_enabled:
            return
        if action not in {"add", "replace"}:
            # Remove is not mirrored until we have stable remote id mapping.
            return
        if not content:
            return

        path = f"hermes/builtin-memory/{target}"
        try:
            client = self._get_client()
            client.remember(
                content=content,
                path=path,
                bucket=self._config.get("bucket", "work"),
                scope=self._config.get("scope", "hermes/default"),
                memory_type="semantic",
            )
            self._record_success()
        except Exception as exc:
            self._record_failure()
            logger.debug("XMemo on_memory_write mirror failed: %s", exc)

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """Capture a restart snapshot at session end."""
        if self._is_breaker_open() or not self._config.get("api_key"):
            return
        if not self._auto_write_enabled:
            return

        def _snapshot() -> None:
            try:
                client = self._get_client()
                client.create_restart_snapshot(
                    session_id=self._session_id,
                    bucket=self._config.get("bucket", "work"),
                    scope=self._config.get("scope", "hermes/default"),
                )
                self._record_success()
            except Exception as exc:
                self._record_failure()
                logger.debug("XMemo session-end snapshot failed: %s", exc)

        self._snapshot_thread = threading.Thread(
            target=_snapshot, daemon=True, name="xmemo-snapshot"
        )
        self._snapshot_thread.start()


def register(ctx) -> None:
    """Register XMemo as a memory provider plugin."""
    ctx.register_memory_provider(XMemoMemoryProvider())

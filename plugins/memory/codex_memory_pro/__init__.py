"""codex-memory-pro MemoryProvider plugin for Hermes Agent.

Provides automatic memory lifecycle hooks + a curated set of manual tools.
Replaces the raw MCP server connection with first-class MemoryProvider integration.

Automatic hooks (no model involvement needed):
  system_prompt_block()  → memory_context_cache  (governance rules, learnings)
  prefetch()             → memory_prepare        (task-relevant recall)
  sync_turn()            → memory_capture        (persist each turn)
  on_memory_write()      → memory_persist        (mirror built-in memory)
  on_session_end()       → final memory_capture  (end-of-session extraction)
  on_pre_compress()      → extract key facts     (preserve insights across compression)
  on_delegation()        → persist subagent results

Exposed tools (10 — model decides when to call):
  cmp_recall, cmp_search, cmp_remember, cmp_profile, cmp_project_context,
  cmp_task_status, cmp_task_create, cmp_task_update, cmp_task_complete, cmp_task_quick

Communication: spawns and manages the codex-memory-pro MCP server as a private
stdio subprocess, connected via the ``mcp`` Python package.  This means the
plugin is self-contained — no ``mcp_servers`` entry in config.yaml is needed.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default paths (overridable via config)
# ---------------------------------------------------------------------------
_DEFAULT_SERVER_SCRIPT = "/home/wxst/codex-memory-pro/src/integrations/cline/mcp-server.mjs"
_DEFAULT_BASE_DIR = "/home/wxst/codex-memory-pro"

# ---------------------------------------------------------------------------
# Tool schemas (only the 10 tools the model needs to decide on)
# ---------------------------------------------------------------------------

_RECALL_SCHEMA = {
    "name": "cmp_recall",
    "description": (
        "Semantic recall over codex-memory-pro stored memories. "
        "Returns the most relevant memory excerpts ranked by relevance. "
        "Use for looking up specific facts, preferences, past decisions, or historical context."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query — keywords, phrases, or natural language."},
            "project": {"type": "string", "description": "Project ID for scoping (e.g. codex-memory-pro)."},
            "scope": {"type": "string", "description": "Scope filter: global or project:xxx (e.g. project:codex-memory-pro). If both scope and project are provided, scope takes priority."},
            "limit": {"type": "integer", "description": "Max results (default 8)."},
        },
        "required": ["query"],
    },
}

_SEARCH_SCHEMA = {
    "name": "cmp_search",
    "description": (
        "Unified search across memories AND Obsidian documents. "
        "Modes: memories (pure memory), documents (Obsidian vault), hybrid (both). "
        "Use when you need to find information that might be in documents, not just memories."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query."},
            "project": {"type": "string", "description": "Project ID for scoping."},
            "scope": {"type": "string", "description": "Scope filter: global or project:xxx (e.g. project:codex-memory-pro). If both scope and project are provided, scope takes priority."},
            "mode": {
                "type": "string",
                "enum": ["auto", "memories", "documents", "hybrid"],
                "description": "Search mode. Default: auto.",
            },
            "limit": {"type": "integer", "description": "Max results (default 8)."},
        },
        "required": ["query"],
    },
}

_REMEMBER_SCHEMA = {
    "name": "cmp_remember",
    "description": (
        "Quick-persist a fact, preference, rule, or observation to codex-memory-pro. "
        "Enters an async refinement queue — processed by the daemon. "
        "Use for anything worth remembering across sessions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "Content to remember."},
            "category": {"type": "string", "description": "Category: preference, workflow, fact, etc."},
            "project": {"type": "string", "description": "Project ID for scoping."},
            "scope": {"type": "string", "description": "Scope: global or project:xxx. If both scope and project are provided, scope takes priority."},
        },
        "required": ["text"],
    },
}

_PROFILE_SCHEMA = {
    "name": "cmp_profile",
    "description": (
        "Get the user profile from codex-memory-pro. "
        "Returns preferences, behavioral patterns, and historical context. "
        "Use when you need to understand user characteristics or communication preferences."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Optional focus query to filter profile."},
            "project": {"type": "string", "description": "Project ID for scoping."},
            "scope": {"type": "string", "description": "Scope filter. Use global or project:xxx. If both scope and project are provided, scope takes priority."},
            "format": {"type": "string", "enum": ["text", "json"], "description": "Output format. Default: text."},
        },
        "required": [],
    },
}

_PROJECT_CONTEXT_SCHEMA = {
    "name": "cmp_project_context",
    "description": (
        "Get project-level memory view from codex-memory-pro. "
        "Returns project decisions, constraints, verified facts, and structured context. "
        "Use for project-scoped work to retrieve accumulated project knowledge."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "project": {"type": "string", "description": "Project ID (required)."},
            "query": {"type": "string", "description": "Optional focus query."},
            "level": {"type": "string", "enum": ["lite", "standard", "deep"], "description": "Detail level. Default: standard."},
            "format": {"type": "string", "enum": ["text", "json"], "description": "Output format. Default: text."},
        },
        "required": ["project"],
    },
}

_TASK_STATUS_SCHEMA = {
    "name": "cmp_task_status",
    "description": (
        "Query active task status in codex-memory-pro. "
        "Use for checkpoint recovery — returns task progress, continuation hints, and key variables."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "project": {"type": "string", "description": "Project ID for scoping."},
            "platform": {"type": "string", "description": "Platform filter (telegram, cli, etc.)."},
        },
        "required": [],
    },
}

_TASK_CREATE_SCHEMA = {
    "name": "cmp_task_create",
    "description": (
        "Create a structured task checkpoint in codex-memory-pro. "
        "Use for multi-step tasks (2+ steps). Returns a task ID for subsequent updates."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Task title."},
            "project": {"type": "string", "description": "Project ID for scoping."},
            "steps": {"type": "array", "items": {"type": "string"}, "description": "Task step list."},
        },
        "required": ["title"],
    },
}

_TASK_UPDATE_SCHEMA = {
    "name": "cmp_task_update",
    "description": (
        "Update a task step status in codex-memory-pro. "
        "Use to mark steps as completed/failed/in_progress and set continuation hints."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "taskId": {"type": "string", "description": "Task ID from task_create."},
            "step": {"type": "number", "description": "Step number (1-indexed)."},
            "status": {"type": "string", "enum": ["completed", "failed", "in_progress"], "description": "Step status."},
            "hint": {"type": "string", "description": "Continuation hint: what to do next + key variables."},
        },
        "required": ["taskId"],
    },
}

_TASK_COMPLETE_SCHEMA = {
    "name": "cmp_task_complete",
    "description": (
        "Archive a task as completed in codex-memory-pro. "
        "Call when all steps are done."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "taskId": {"type": "string", "description": "Task ID from task_create."},
            "summary": {"type": "string", "description": "Completion summary."},
        },
        "required": ["taskId"],
    },
}

_TASK_QUICK_SCHEMA = {
    "name": "cmp_task_quick",
    "description": (
        "Create a lightweight checkpoint for non-structured work in codex-memory-pro. "
        "Use for simple conversations that might need continuation (session disconnect, etc.)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "What you're currently doing."},
            "project": {"type": "string", "description": "Project ID for scoping."},
            "hint": {"type": "string", "description": "Current state + key variables + next step."},
        },
        "required": ["title"],
    },
}

ALL_TOOL_SCHEMAS = [
    _RECALL_SCHEMA,
    _SEARCH_SCHEMA,
    _REMEMBER_SCHEMA,
    _PROFILE_SCHEMA,
    _PROJECT_CONTEXT_SCHEMA,
    _TASK_STATUS_SCHEMA,
    _TASK_CREATE_SCHEMA,
    _TASK_UPDATE_SCHEMA,
    _TASK_COMPLETE_SCHEMA,
    _TASK_QUICK_SCHEMA,
]

# Mapping: plugin tool name → MCP tool name
_TOOL_NAME_MAP = {
    "cmp_recall": "memory_recall",
    "cmp_search": "memory_search",
    "cmp_remember": "memory_remember",
    "cmp_profile": "memory_profile",
    "cmp_project_context": "memory_project_context",
    "cmp_task_status": "task_status",
    "cmp_task_create": "task_create",
    "cmp_task_update": "task_update",
    "cmp_task_complete": "task_complete",
    "cmp_task_quick": "task_quick",
}


# ---------------------------------------------------------------------------
# Patterns for on_pre_compress heuristic extraction
# ---------------------------------------------------------------------------
_FACT_PATTERNS = [
    re.compile(r"(?:记住|记住这个|以后|务必|必须|不要|禁止|注意|切记)", re.IGNORECASE),
    re.compile(r"(?:偏好|prefer|习惯|习惯是|总是|从不|从不)"),
    re.compile(r"(?:规则|规范|约定|convention|workflow)"),
    re.compile(r"(?:地址|路径|端口|URL|host|endpoint)[是为：:]"),
]

_SKIP_CAPTURE_PATTERNS = [
    re.compile(r"(?:记住|remember|偏好|preference|规则|rule|约定|workflow)", re.IGNORECASE),
    re.compile(r"(?:bug|错误|报错|失败|修复|原因|根因|配置|路径|scope|project)", re.IGNORECASE),
]

_SKIP_MEMORY_WRITE_PATTERNS = [
    re.compile(r"(?:测试|test|临时|temporary|todo|稍后|回头)", re.IGNORECASE),
]

_VALID_SCOPE_RE = re.compile(r"^(?:global|project:[^\s]+)$")


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

class CodexMemoryProvider(MemoryProvider):
    """codex-memory-pro long-term memory via internal MCP client.

    Manages a private MCP server subprocess.  Automatic hooks handle the
    memory lifecycle; only 10 curated tools are exposed to the model.
    """

    def __init__(self) -> None:
        # --- MCP connection state ---
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._session = None  # MCP ClientSession
        self._stdio_cm = None  # stdio_client context manager (for cleanup)
        self._session_cm = None  # ClientSession context manager (for cleanup)
        self._connected = False
        self._connect_lock = threading.Lock()

        # --- Config ---
        self._server_script = _DEFAULT_SERVER_SCRIPT
        self._base_dir = _DEFAULT_BASE_DIR
        self._session_id = ""
        self._project = ""

        # --- State ---
        self._cron_skipped = False
        self._turn_count = 0
        self._initialized = False

        # --- Context cache (governance rules from memory_context_cache) ---
        self._context_cache = ""
        self._context_cache_lock = threading.Lock()
        self._context_cache_cadence = 10  # refresh every N turns
        self._last_cache_turn = -999

        # --- Thread pool for background operations ---
        self._sync_thread: Optional[threading.Thread] = None

        # --- Track last turn for session-end capture ---
        self._last_user = ""
        self._last_assistant = ""

    # -- Property -----------------------------------------------------------

    @property
    def name(self) -> str:
        return "codex-memory-pro"

    # -- Core lifecycle -----------------------------------------------------

    def is_available(self) -> bool:
        """Check prerequisites: MCP server script exists and node is available."""
        try:
            script = Path(self._server_script)
            if not script.exists():
                # Try reading from config.yaml mcp_servers (for backward compat)
                cfg_path = Path.home() / ".hermes" / "config.yaml"
                if cfg_path.exists():
                    try:
                        import yaml
                        with open(cfg_path) as f:
                            cfg = yaml.safe_load(f) or {}
                        mcp = cfg.get("mcp_servers", {}).get("codex-memory-pro", {})
                        if mcp.get("args"):
                            script = Path(mcp["args"][-1])
                    except Exception:
                        pass
                if not script.exists():
                    logger.debug("codex-memory-pro: server script not found at %s", self._server_script)
                    return False

            return shutil.which("node") is not None
        except Exception:
            return False

    def initialize(self, session_id: str, **kwargs) -> None:
        """Start MCP server subprocess and connect.

        Handles cron guard, config resolution, and pre-warming context cache.
        """
        # --- Cron guard ---
        agent_context = kwargs.get("agent_context", "")
        platform = kwargs.get("platform", "cli")
        if agent_context in ("cron", "flush") or platform == "cron":
            logger.debug("codex-memory-pro skipped: cron/flush context")
            self._cron_skipped = True
            return

        # --- Resolve config overrides ---
        self._session_id = session_id or ""  # only used for capture event tracing
        self._project = ""  # Empty — let CLI resolveContextualScope/buildRoutePayload infer
        hermes_home = kwargs.get("hermes_home", "")

        # Read plugin-specific config if present
        self._resolve_config(hermes_home)

        # Override project from kwargs/config if explicitly set
        # (e.g. config.yaml memory.codex_memory_pro.project)

        # --- Start event loop thread ---
        self._start_event_loop()

        # --- Connect to MCP server ---
        if not self._ensure_connection():
            logger.warning("codex-memory-pro: MCP connection failed, plugin inactive")
            self._cron_skipped = True
            return

        self._initialized = True

        # --- Pre-warm: load governance rules into context cache ---
        self._refresh_context_cache()

        logger.info(
            "codex-memory-pro plugin initialized (session=%s, project=%s)",
            self._session_id[:8] if self._session_id else "none",
            self._project or "global",
        )

    def _resolve_config(self, hermes_home: str) -> None:
        """Read plugin config from config.yaml under memory.codex_memory_pro."""
        try:
            from hermes_cli.config import load_config
            cfg = load_config()
            cmp_cfg = cfg.get("memory", {}).get("codex_memory_pro", {})

            if cmp_cfg.get("server_script"):
                self._server_script = cmp_cfg["server_script"]
            if cmp_cfg.get("base_dir"):
                self._base_dir = cmp_cfg["base_dir"]
            if cmp_cfg.get("project"):
                self._project = cmp_cfg["project"]
            if cmp_cfg.get("context_cache_cadence"):
                self._context_cache_cadence = int(cmp_cfg["context_cache_cadence"])
        except Exception as e:
            logger.debug("codex-memory-pro config resolution failed: %s", e)

    # -- Event loop management ----------------------------------------------

    def _start_event_loop(self) -> None:
        """Create and start a dedicated event loop in a daemon thread."""
        self._loop = asyncio.new_event_loop()

        def _run():
            asyncio.set_event_loop(self._loop)
            self._loop.run_forever()

        self._loop_thread = threading.Thread(target=_run, daemon=True, name="cmp-mcp-loop")
        self._loop_thread.start()

    def _stop_event_loop(self) -> None:
        """Stop the event loop thread."""
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._loop_thread and self._loop_thread.is_alive():
            self._loop_thread.join(timeout=5.0)
        self._loop = None
        self._loop_thread = None

    # -- MCP connection management ------------------------------------------

    async def _connect_async(self) -> bool:
        """Async: spawn MCP server and establish stdio connection."""
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError:
            logger.error("codex-memory-pro: 'mcp' package not installed")
            return False

        env = {**os.environ, "CODEX_MEMORY_BASE_DIR": self._base_dir}

        server_params = StdioServerParameters(
            command="node",
            args=[self._server_script],
            env=env,
        )

        try:
            self._stdio_cm = stdio_client(server_params)
            read_stream, write_stream = await self._stdio_cm.__aenter__()

            self._session_cm = ClientSession(read_stream, write_stream)
            session: ClientSession = await self._session_cm.__aenter__()
            await session.initialize()

            self._session = session
            self._connected = True
            logger.info("codex-memory-pro MCP server connected via stdio")
            return True
        except Exception as e:
            logger.error("codex-memory-pro MCP connection failed: %s", e)
            self._cleanup_connection()
            return False

    def _cleanup_connection(self) -> None:
        """Reset connection state without async (for error recovery)."""
        self._connected = False
        self._session = None
        self._stdio_cm = None
        self._session_cm = None

    async def _disconnect_async(self) -> None:
        """Async: cleanly close MCP session and terminate server process."""
        try:
            if self._session_cm is not None:
                await self._session_cm.__aexit__(None, None, None)
            if self._stdio_cm is not None:
                await self._stdio_cm.__aexit__(None, None, None)
        except Exception as e:
            logger.debug("codex-memory-pro disconnect error: %s", e)
        self._connected = False
        self._session = None
        self._stdio_cm = None
        self._session_cm = None

    def _ensure_connection(self) -> bool:
        """Ensure MCP connection is live. Thread-safe, with reconnection."""
        if self._connected and self._session is not None:
            return True

        with self._connect_lock:
            # Double-check after acquiring lock
            if self._connected and self._session is not None:
                return True

            if self._loop is None or not self._loop.is_running():
                logger.error("codex-memory-pro: event loop not running")
                return False

            try:
                future = asyncio.run_coroutine_threadsafe(
                    self._connect_async(), self._loop
                )
                return future.result(timeout=20.0)
            except Exception as e:
                logger.error("codex-memory-pro: connection attempt failed: %s", e)
                self._cleanup_connection()
                return False

    # -- MCP tool calling (sync wrapper around async) -----------------------

    def _call_tool(self, tool_name: str, args: dict = None,
                   timeout: float = 15.0) -> str:
        """Call an MCP tool synchronously via the event loop thread.

        Returns the text content from the tool result, or a JSON error.
        """
        if not self._ensure_connection():
            return json.dumps({"error": "codex-memory-pro not connected"})

        args = args or {}
        future = asyncio.run_coroutine_threadsafe(
            self._async_call_tool(tool_name, args), self._loop
        )

        try:
            return future.result(timeout=timeout + 2.0)
        except Exception as e:
            # Connection might have died — mark for reconnection
            logger.warning("codex-memory-pro tool %s call failed: %s", tool_name, e)
            self._connected = False
            self._session = None
            return json.dumps({"error": f"codex-memory-pro {tool_name}: {e}"})

    async def _async_call_tool(self, tool_name: str, args: dict) -> str:
        """Async: call MCP tool and extract text content from result."""
        try:
            result = await asyncio.wait_for(
                self._session.call_tool(tool_name, arguments=args),
                timeout=15.0,
            )
            # Extract text from MCP CallToolResult
            parts: list[str] = []
            if hasattr(result, "content"):
                for block in result.content:
                    if hasattr(block, "text") and block.text:
                        parts.append(block.text)
            return "\n".join(parts) if parts else json.dumps({"result": str(result)})
        except asyncio.TimeoutError:
            return json.dumps({"error": f"{tool_name} timed out"})
        except Exception as e:
            return json.dumps({"error": f"{tool_name}: {e}"})

    # =========================================================================
    # Automatic hooks (no model involvement)
    # =========================================================================

    def system_prompt_block(self) -> str:
        """Return static system prompt text about the active memory system.

        Prompt-cache friendly — dynamic governance/memory context is injected
        via prefetch(), not embedded into the system prompt.
        """
        if self._cron_skipped or not self._initialized:
            return ""

        return (
            "# codex-memory-pro Memory\n"
            "Active. Context (governance rules, relevant memories) is automatically "
            "injected before each turn via prefetch. Manual tools: cmp_recall, "
            "cmp_search, cmp_remember, cmp_profile, cmp_project_context, "
            "cmp_task_status, cmp_task_create, cmp_task_update, "
            "cmp_task_complete, cmp_task_quick."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Auto-inject relevant context before each API call.

        Calls memory_prepare (server-side smart debounce skips low-value queries).
        Returns context text to be injected as <memory-context>.
        """
        if self._cron_skipped or not self._initialized:
            return ""

        if not query or not query.strip():
            return ""

        # Build args for memory_prepare (no project — let CLI infer scope)
        args: dict[str, Any] = {"query": query}

        # Use shorter timeout for prefetch (blocks response generation)
        result = self._call_tool("memory_prepare", args, timeout=5.0)

        # Parse out error responses — return empty on failure (graceful degradation)
        try:
            parsed = json.loads(result)
            if "error" in parsed:
                logger.debug("codex-memory-pro prefetch error: %s", parsed["error"])
                return ""
        except (json.JSONDecodeError, TypeError):
            pass

        return result or ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Queue background prefetch for the next turn.

        memory_prepare has server-side debounce, so we also refresh the
        governance context cache in the background.
        """
        if self._cron_skipped or not self._initialized:
            return

        # Refresh context cache on cadence
        if (self._turn_count - self._last_cache_turn) >= self._context_cache_cadence:
            t = threading.Thread(
                target=self._refresh_context_cache, daemon=True, name="cmp-cache-refresh"
            )
            t.start()

    def sync_turn(self, user_content: str, assistant_content: str,
                  *, session_id: str = "") -> None:
        """Persist a completed turn to codex-memory-pro (non-blocking).

        Calls memory_capture to extract and persist durable insights.
        """
        if self._cron_skipped or not self._initialized:
            return

        # Track for session-end capture
        self._last_user = user_content
        self._last_assistant = assistant_content

        user_brief = self._truncate_capture_text(user_content)
        asst_brief = self._truncate_capture_text(assistant_content)

        if not user_brief.strip() and not asst_brief.strip():
            return

        if self._should_skip_sync_turn(user_brief, asst_brief):
            return

        args: dict[str, Any] = {
            "userText": user_brief,
            "assistantText": asst_brief,
        }  # No project — let CLI buildRoutePayload infer

        def _capture():
            try:
                self._call_tool("memory_capture", args, timeout=10.0)
            except Exception as e:
                logger.debug("codex-memory-pro sync_turn failed: %s", e)

        # Wait for previous sync to finish before starting a new one
        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5.0)

        self._sync_thread = threading.Thread(
            target=_capture, daemon=True, name="cmp-sync"
        )
        self._sync_thread.start()

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        """Track turn count for cadence logic."""
        self._turn_count = turn_number

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """Final capture on session end + flush pending syncs."""
        if self._cron_skipped or not self._initialized:
            return

        # Wait for pending sync
        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=10.0)

        # Final capture if we have a last turn
        last_user = self._truncate_capture_text(self._last_user)
        last_assistant = self._truncate_capture_text(self._last_assistant)
        if last_user.strip() or last_assistant.strip():
            if self._should_skip_sync_turn(last_user, last_assistant):
                return
            args: dict[str, Any] = {
                "userText": last_user,
                "assistantText": last_assistant,
            }  # No project — let CLI infer
            try:
                self._call_tool("memory_capture", args, timeout=10.0)
            except Exception as e:
                logger.debug("codex-memory-pro session-end capture failed: %s", e)

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        """Extract key facts from messages about to be compressed.

        Uses simple heuristics (pattern matching) to identify user statements
        that contain preferences, rules, corrections, or important facts.
        Returns text to include in the compression summary prompt.
        """
        if self._cron_skipped or not self._initialized:
            return ""

        facts: list[str] = []

        for msg in messages[-50:]:
            role = msg.get("role", "")
            content = ""
            if isinstance(msg.get("content"), str):
                content = msg["content"]
            elif isinstance(msg.get("content"), list):
                # Multi-part content (e.g. tool results)
                for part in msg["content"]:
                    if isinstance(part, dict) and part.get("type") == "text":
                        content += part.get("text", "") + " "

            if not content or not content.strip():
                continue

            # Only extract from user messages (assistant is our own output)
            if role != "user":
                continue

            content = content.strip()

            # Check if the message matches any fact patterns
            for pattern in _FACT_PATTERNS:
                if pattern.search(content):
                    # Extract the sentence(s) containing the match
                    sentences = re.split(r'[。！？\n]', content)
                    for sent in sentences:
                        if sent.strip() and pattern.search(sent):
                            facts.append(sent.strip())
                    break  # One match per message is enough

            if len(facts) >= 8:
                break  # Cap to avoid bloating compression prompt

        if not facts:
            return ""

        return (
            "## Key facts to preserve from compressed context\n"
            + "\n".join(f"- {f}" for f in facts)
        )

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        """Mirror built-in memory writes to codex-memory-pro.

        When the built-in memory tool writes (user preferences, facts),
        this hook persists the same content via memory_persist for
        cross-system consistency.
        """
        if self._cron_skipped or not self._initialized:
            return
        if action == "remove":
            return  # Don't mirror deletions (might cause orphaned memories)
        if not content or not content.strip():
            return
        if self._should_skip_memory_write(content):
            return

        args: dict[str, Any] = {"text": content}
        if target == "user":
            args["category"] = "preference"
        else:
            args["category"] = "fact"
        # No project — let CLI buildRoutePayload infer

        def _persist():
            try:
                self._call_tool("memory_persist", args, timeout=10.0)
            except Exception as e:
                logger.debug("codex-memory-pro memory mirror failed: %s", e)

        t = threading.Thread(target=_persist, daemon=True, name="cmp-memwrite")
        t.start()

    def on_delegation(self, task: str, result: str, *,
                      child_session_id: str = "", **kwargs) -> None:
        """Record subagent completion in codex-memory-pro.

        Persists the delegation task+result pair as a memory for
        cross-session continuity.
        """
        if self._cron_skipped or not self._initialized:
            return
        if not result or not result.strip():
            return

        # Compose a summary for persisting
        text = f"[Delegation] Task: {task[:500]}\nResult: {result[:1000]}"

        args: dict[str, Any] = {
            "text": text,
            "category": "workflow",
        }  # No project — let CLI infer

        def _record():
            try:
                self._call_tool("memory_remember", args, timeout=10.0)
            except Exception as e:
                logger.debug("codex-memory-pro delegation record failed: %s", e)

        t = threading.Thread(target=_record, daemon=True, name="cmp-delegation")
        t.start()

    # =========================================================================
    # Context cache management
    # =========================================================================

    def _refresh_context_cache(self) -> None:
        """Refresh the governance rules / context cache in the background."""
        if not self._initialized:
            return

        result = self._call_tool("memory_context_cache", {"format": "text"}, timeout=8.0)

        try:
            parsed = json.loads(result)
            if "error" in parsed:
                return
        except (json.JSONDecodeError, TypeError):
            pass

        if result and result.strip():
            with self._context_cache_lock:
                self._context_cache = result
            self._last_cache_turn = self._turn_count
            logger.debug("codex-memory-pro context cache refreshed (turn %d)", self._turn_count)

    def _truncate_capture_text(self, text: str, *, limit: int = 2000) -> str:
        if not text or len(text) <= limit:
            return text
        half = max((limit - 7) // 2, 1)
        return f"{text[:half]}\n...\n{text[-half:]}"

    def _should_skip_sync_turn(self, user_text: str, assistant_text: str) -> bool:
        user = (user_text or "").strip()
        assistant = (assistant_text or "").strip()
        if len(user) >= 10:
            return False
        if any(pattern.search(user) for pattern in _SKIP_CAPTURE_PATTERNS):
            return False
        if any(pattern.search(assistant) for pattern in _SKIP_CAPTURE_PATTERNS):
            return False
        return True

    def _should_skip_memory_write(self, content: str) -> bool:
        text = (content or "").strip()
        if not text:
            return True
        if len(text) >= 20:
            return False
        return any(pattern.search(text) for pattern in _SKIP_MEMORY_WRITE_PATTERNS)

    def _sanitize_tool_args(self, args: dict | None) -> dict:
        clean = dict(args or {})
        scope = clean.get("scope")
        if isinstance(scope, str) and scope and not _VALID_SCOPE_RE.match(scope):
            logger.warning("codex-memory-pro: invalid scope=%s, dropping", scope)
            clean.pop("scope", None)
        if clean.get("scope") and clean.get("project"):
            clean.pop("project", None)
        return clean

    # =========================================================================
    # Tool interface
    # =========================================================================

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Return the 10 curated tool schemas."""
        if self._cron_skipped:
            return []
        return list(ALL_TOOL_SCHEMAS)

    # Scope dimensions for Hermes:
    #   - global:   通用记忆（偏好、事实、规则）
    #   - project:<name>: 项目专属记忆（name = 项目目录名，如 codex-memory-pro）
    #   - project:Hermes: Hermes 自身相关记忆（默认）
    # Note: sessionId has been fully removed — it was causing scope locking to
    # ephemeral session UUIDs, filtering out all cross-session memories.

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        """Dispatch a tool call to the MCP server.

        Maps plugin tool names (cmp_*) to MCP tool names.
        """
        if self._cron_skipped:
            return tool_error("codex-memory-pro is not active (cron context).")

        mcp_name = _TOOL_NAME_MAP.get(tool_name)
        if not mcp_name:
            return tool_error(f"Unknown codex-memory-pro tool: {tool_name}")

        return self._call_tool(mcp_name, self._sanitize_tool_args(args))

    # =========================================================================
    # Shutdown
    # =========================================================================

    def shutdown(self) -> None:
        """Clean shutdown: flush syncs, close MCP connection, stop loop."""
        # Wait for pending sync
        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=10.0)

        # Close MCP connection
        if self._loop and self._loop.is_running():
            try:
                future = asyncio.run_coroutine_threadsafe(
                    self._disconnect_async(), self._loop
                )
                future.result(timeout=10.0)
            except Exception as e:
                logger.debug("codex-memory-pro shutdown error: %s", e)

        # Stop event loop
        self._stop_event_loop()
        self._initialized = False
        logger.info("codex-memory-pro plugin shut down")

    # =========================================================================
    # Config schema (for `hermes memory setup` wizard)
    # =========================================================================

    def get_config_schema(self) -> List[Dict[str, Any]]:
        """Return config fields for setup wizard."""
        return [
            {
                "key": "server_script",
                "description": "Path to MCP server script (mcp-server.mjs)",
                "default": _DEFAULT_SERVER_SCRIPT,
            },
            {
                "key": "base_dir",
                "description": "codex-memory-pro installation directory",
                "default": _DEFAULT_BASE_DIR,
            },
            {
                "key": "project",
                "description": "Default project ID for memory scoping (optional)",
            },
            {
                "key": "context_cache_cadence",
                "description": "Turns between governance cache refreshes (default 10)",
            },
        ]

    def save_config(self, values: dict, hermes_home: str) -> None:
        """Write plugin config to config.yaml under memory.codex_memory_pro."""
        # This is handled by Hermes config system when using memory.provider
        pass


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Register codex-memory-pro as a memory provider plugin."""
    ctx.register_memory_provider(CodexMemoryProvider())

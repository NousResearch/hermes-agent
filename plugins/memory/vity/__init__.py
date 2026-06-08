"""Vity (Maximem AI) memory plugin — MemoryProvider for Hermes Agent.

Cross-session semantic memory with profile-based recall,
memory graph (facts, preferences, emotions, episodes, knowledge, profile),
and sub-100ms context injection via the Maximem REST API.

Config via environment variables:
  MAXIMEM_API_KEY  — Maximem API key (required, starts with mx_)
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any, Dict, List

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool Schemas (what the agent sees and can call)
# ---------------------------------------------------------------------------

VITY_RECALL_SCHEMA = {
    "name": "vity_recall",
    "description": (
        "Search Vity's semantic memory for relevant context about the user. "
        "Returns ranked memories from the user's memory graph. "
        "Use at conversation start or when you need to recall user preferences, "
        "past decisions, project context, or personal facts."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for in memory."},
            "top_k": {"type": "integer", "description": "Max results (default: 10, max: 50)."},
        },
        "required": ["query"],
    },
}

VITY_PROFILE_SCHEMA = {
    "name": "vity_profile",
    "description": (
        "Retrieve the user's full memory profile — all stored facts, preferences, "
        "emotions, episodes, knowledge, and profile data. Use at conversation start "
        "for a complete context snapshot."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}

VITY_STORE_SCHEMA = {
    "name": "vity_store",
    "description": (
        "Store a new memory fact about the user in Vity's memory graph. "
        "Use for explicit user preferences, corrections, important decisions, "
        "or facts the user wants remembered across sessions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The fact or preference to remember."},
            "memory_type": {
                "type": "string",
                "description": "Memory type: fact, preference, emotion, episode, knowledge, or profile.",
                "enum": ["fact", "preference", "emotion", "episode", "knowledge", "profile"],
            },
        },
        "required": ["content"],
    },
}

ALL_TOOL_SCHEMAS = [VITY_RECALL_SCHEMA, VITY_PROFILE_SCHEMA, VITY_STORE_SCHEMA]

VITY_FORGET_SCHEMA = {
    "name": "vity_forget",
    "description": (
        "Delete memories matching a query from Vity's memory graph. "
        "Use when the user explicitly asks to forget something. "
        "Always use dry_run=true first to preview what would be deleted."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to delete from memory."},
            "dry_run": {
                "type": "boolean",
                "description": "If true (default), preview deletions without deleting. Pass false to confirm.",
            },
        },
        "required": [],
    },
}

ALL_TOOL_SCHEMAS = [VITY_RECALL_SCHEMA, VITY_PROFILE_SCHEMA, VITY_STORE_SCHEMA, VITY_FORGET_SCHEMA]


# ---------------------------------------------------------------------------
# Helper: Load config
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    """Load Vity config from env vars + $HERMES_HOME/vity.json overrides."""
    from hermes_constants import get_hermes_home

    config = {
        # MAXIMEM_API_KEY is the canonical env var (matches the official npm plugin)
        "api_key": (
            os.environ.get("MAXIMEM_API_KEY")
            or os.environ.get("VITY_API_KEY")  # backward compat
            or ""
        ),
    }

    config_path = get_hermes_home() / "vity.json"
    if config_path.exists():
        try:
            file_cfg = json.loads(config_path.read_text(encoding="utf-8"))
            config.update({k: v for k, v in file_cfg.items() if v})
        except Exception:
            pass

    return config


# ---------------------------------------------------------------------------
# Main MemoryProvider Implementation
# ---------------------------------------------------------------------------

class VityMemoryProvider(MemoryProvider):
    """Maximem Vity memory with semantic memory graph and profile-based recall."""

    def __init__(self):
        self._client = None
        self._client_lock = threading.Lock()
        self._api_key = ""
        # Prefetch threading (non-blocking pattern — REQUIRED by Hermes contract)
        self._prefetch_result = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread = None
        self._sync_thread = None
        # Track whether we've done the initial warm-up recall
        self._cold_start = True
        # Timing: measure Vity API time vs Hermes LLM time
        self._vity_prefetch_ms: float = 0.0  # how long prefetch() waited for Vity
        self._vity_tool_ms: float = 0.0      # accumulated time inside handle_tool_call() this turn
        self._prefetch_done_at: float = 0.0  # perf_counter timestamp when prefetch returned
        self._vity_retrieved = False         # did this turn actually pull memory? (prefetch ctx or recall/profile tool)

    # -- Identity ------------------------------------------------------------

    @property
    def name(self) -> str:
        return "vity"

    # -- Availability (no network calls!) ------------------------------------

    def is_available(self) -> bool:
        """Check config only — NO network calls allowed here."""
        cfg = _load_config()
        return bool(cfg.get("api_key"))

    # -- Config schema (for `hermes memory setup`) ---------------------------

    def get_config_schema(self):
        return [
            {
                "key": "api_key",
                "description": "Maximem API key (starts with mx_)",
                "secret": True,
                "required": True,
                "env_var": "MAXIMEM_API_KEY",
                "url": "https://maximem.ai/dashboard",
            },
        ]

    def save_config(self, values: dict, hermes_home: str) -> None:
        """Write non-secret config to $HERMES_HOME/vity.json."""
        from pathlib import Path
        from utils import atomic_json_write
        config_path = Path(hermes_home) / "vity.json"
        existing = {}
        if config_path.exists():
            try:
                existing = json.loads(config_path.read_text())
            except Exception:
                pass
        existing.update(values)
        atomic_json_write(config_path, existing, mode=0o600)

    def initialize(self, session_id: str, **kwargs) -> None:
        """Initialize Vity client for this session."""
        config = _load_config()
        self._api_key = config.get("api_key", "")
        self._session_id = session_id
        logger.debug("Vity initialized: session=%s", session_id)

        # Mirror OpenClaw's before_agent_start: immediately warm the recall cache
        # with a broad profile query so the first user message has full context.
        # The query covers all common personal facts the user might ask about.
        self._cold_start = True
        self.queue_prefetch(
            "user profile background context history preferences facts",
            session_id=session_id,
        )

    def _get_client(self):
        """Thread-safe lazy initialization of the Vity SDK client."""
        with self._client_lock:
            if self._client is not None:
                return self._client
            try:
                from maximem_vity import VityClient
                self._client = VityClient(api_key=self._api_key)
                return self._client
            except ImportError:
                raise RuntimeError(
                    "maximem-vity-sdk not installed. "
                    "Run: pip install maximem-vity-sdk"
                )

    # -- System Prompt Block -------------------------------------------------

    def system_prompt_block(self) -> str:
        return (
            "# Vity Memory (Maximem AI)\n"
            "Active. Memories are scoped to the configured Maximem API key.\n\n"
            "## Rules\n"
            "1. If the user asks about something personal (facts, preferences, history, context) "
            "and it is not already in your injected memory context, call `vity_recall` with a "
            "targeted query before saying you don't know. Never claim ignorance without trying.\n"
            "2. Use `vity_store` to save any new facts the user shares about themselves.\n"
            "3. Use `vity_forget` only when the user explicitly asks to delete a memory."
        )

    # -- Prefetch (background context injection) -----------------------------

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Pre-warm Vity recall in background after each turn."""
        def _run():
            try:
                client = self._get_client()
                # Use recall() — the purpose-built context injection endpoint
                context = client.recall(
                    current_prompt=query,
                    max_tokens=500,
                    strategy="hybrid",
                )
                if context:
                    with self._prefetch_lock:
                        self._prefetch_result = context
            except Exception as e:
                logger.debug("Vity prefetch failed: %s", e)

        self._prefetch_thread = threading.Thread(target=_run, daemon=True, name="vity-prefetch")
        self._prefetch_thread.start()

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Return pre-warmed Vity context.

        On cold start (first turn of a session), waits for the warm-up recall
        that was triggered in initialize(). This mirrors OpenClaw's
        before_agent_start hook — memory is always injected before turn 1.
        """
        import time as _time
        _t0 = _time.perf_counter()

        # Wait for background prefetch (longer wait on cold start to ensure
        # the initial recall triggered in initialize() has time to complete)
        wait_secs = 8.0 if self._cold_start else 3.0
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=wait_secs)

        with self._prefetch_lock:
            result = self._prefetch_result
            self._prefetch_result = ""

        # Cold-start fallback: if warm-up recall returned nothing (e.g. took
        # too long), do a quick blocking recall right now with the actual query.
        if not result and self._cold_start:
            try:
                client = self._get_client()
                result = client.recall(
                    current_prompt=query,
                    max_tokens=800,
                    strategy="hybrid",
                )
                logger.debug("Vity cold-start blocking recall returned %d chars", len(result))
            except Exception as e:
                logger.debug("Vity cold-start recall failed: %s", e)

        self._cold_start = False  # Subsequent turns use the normal prefetch path

        # Record timing: how long prefetch waited for Vity
        self._vity_prefetch_ms = (_time.perf_counter() - _t0) * 1000
        self._vity_tool_ms = 0.0  # reset per-turn tool accumulator
        self._prefetch_done_at = _time.perf_counter()
        # Reset per-turn; prefetch counts as retrieval only if it injected context.
        # recall/profile tool calls later in the turn can flip this back on.
        self._vity_retrieved = bool(result)

        if not result:
            return ""
        return f"## Vity Memory\n{result}"

    # -- Sync Turn (MUST be non-blocking!) -----------------------------------

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages=None,
    ) -> None:
        """Capture conversation turn into Vity's memory pipeline (non-blocking)."""
        import time
        now = int(time.time() * 1000)

        # Timing breakdown for the CLI feed — only shown when Vity actually
        # retrieved memory this turn (prefetched context or a recall/profile
        # tool call). Pure write-only turns stay quiet instead of printing 0ms.
        if self._prefetch_done_at > 0:
            total_turn_ms = (time.perf_counter() - self._prefetch_done_at) * 1000
            vity_total_ms = self._vity_prefetch_ms + self._vity_tool_ms
            hermes_llm_ms = max(0.0, total_turn_ms - self._vity_tool_ms)
            if self._vity_retrieved:
                # ANSI dim so it's subtle but readable alongside the ┊ feed
                _DIM, _RESET = "\033[2m", "\033[0m"
                print(
                    f"{_DIM}⏱ Vity memory · retrieved in {vity_total_ms:.0f}ms · "
                    f"response in {hermes_llm_ms:.0f}ms{_RESET}",
                    flush=True,
                )
            logger.debug(
                "Vity timing: retrieved=%s, vity=%.0f ms, hermes_llm=%.0f ms",
                self._vity_retrieved, vity_total_ms, hermes_llm_ms,
            )
            self._prefetch_done_at = 0.0
            self._vity_tool_ms = 0.0
            self._vity_retrieved = False

        def _sync():
            try:
                client = self._get_client()
                # Use capture() — the purpose-built conversation ingestion endpoint
                client.capture(
                    messages=[
                        {"role": "user", "content": user_content, "timestamp": now},
                        {"role": "assistant", "content": assistant_content, "timestamp": now + 1},
                    ],
                    agent_id="hermes",
                    session_key=session_id,
                )
            except Exception as e:
                logger.warning("Vity sync failed: %s", e)

        # Wait for previous sync before starting a new one
        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5.0)

        self._sync_thread = threading.Thread(target=_sync, daemon=True, name="vity-sync")
        self._sync_thread.start()

    # -- Tool Schemas + Dispatch ---------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return ALL_TOOL_SCHEMAS

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        """Dispatch tool calls from the agent to Vity's API.

        Each call is timed and accumulated into _vity_tool_ms so that
        sync_turn() can subtract it from total turn time, isolating the
        pure Hermes LLM processing time.
        """
        import time as _t
        _tool_start = _t.perf_counter()

        try:
            client = self._get_client()
        except Exception as e:
            return tool_error(str(e))

        try:
            if tool_name == "vity_profile":
                try:
                    profile = client.get_profile()
                    if not profile:
                        return json.dumps({"result": "No memories stored yet."})
                    self._vity_retrieved = True
                    return json.dumps({"result": profile, "source": "vity"})
                except Exception as e:
                    return tool_error(f"Failed to fetch profile: {e}")

            elif tool_name == "vity_recall":
                query = args.get("query", "")
                if not query:
                    return tool_error("Missing required parameter: query")
                top_k = min(int(args.get("top_k", 10)), 50)
                try:
                    context = client.recall(
                        current_prompt=query,
                        max_tokens=top_k * 100,
                        strategy="hybrid",
                    )
                    if not context:
                        return json.dumps({"result": "No relevant memories found."})
                    self._vity_retrieved = True
                    return json.dumps({"result": context})
                except Exception as e:
                    return tool_error(f"Search failed: {e}")

            elif tool_name == "vity_store":
                content = args.get("content", "")
                if not content:
                    return tool_error("Missing required parameter: content")
                memory_type = args.get("memory_type", "fact")
                try:
                    result = client.store(content=content, memory_type=memory_type)
                    return json.dumps({"result": "Memory stored successfully.", "id": result.get("id", "")})
                except Exception as e:
                    return tool_error(f"Failed to store memory: {e}")

            elif tool_name == "vity_forget":
                query = args.get("query", "")
                dry_run = args.get("dry_run", True)
                try:
                    result = client.forget(query=query, dry_run=dry_run)
                    count = result.get("count", 0)
                    if dry_run:
                        return json.dumps({"result": f"Would delete {count} memories. Call again with dry_run=false to confirm."})
                    return json.dumps({"result": f"Deleted {count} memories.", "count": count})
                except Exception as e:
                    return tool_error(f"Forget failed: {e}")

            return tool_error(f"Unknown tool: {tool_name}")

        finally:
            # Accumulate time spent inside Vity API calls this turn
            self._vity_tool_ms += (_t.perf_counter() - _tool_start) * 1000

    # -- Shutdown ------------------------------------------------------------

    def shutdown(self) -> None:
        """Flush background threads before process exits."""
        for t in (self._prefetch_thread, self._sync_thread):
            if t and t.is_alive():
                t.join(timeout=5.0)
        with self._client_lock:
            if self._client is not None:
                try:
                    self._client.close()
                except Exception:
                    pass
                self._client = None


# ---------------------------------------------------------------------------
# Plugin Entry Point (required by Hermes discovery system)
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Called by Hermes memory plugin discovery. Registers Vity as a provider."""
    ctx.register_memory_provider(VityMemoryProvider())

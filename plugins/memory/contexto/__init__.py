"""Contexto memory plugin — MemoryProvider backed by self-hosted Contexto.

Per-turn ingest of (user, assistant) pairs into the Contexto OSS memory
engine, semantic recall via /v1/search before each turn, and a
contexto_search tool for explicit lookups.

Agent slug is one-per-hermes-profile (derived from agent_identity).
userId is per-platform-user (gateway-supplied).

Config via environment:
  CONTEXTO_BASE_URL  — selfhost API base (default http://localhost:4010)
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


SEARCH_SCHEMA = {
    "name": "contexto_search",
    "description": (
        "Search the Contexto cognitive memory for content relevant to a query. "
        "Returns hits ranked across episodic, semantic, and procedural sectors."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to look up."},
            "max_results": {"type": "integer", "description": "Max snippets (default: 5)."},
        },
        "required": ["query"],
    },
}


class ContextoMemoryProvider(MemoryProvider):
    """Contexto-backed memory provider (self-hosted only)."""

    def __init__(self):
        self._client = None
        self._agent_slug = "hermes"
        self._user_id: str | None = None
        self._agent_context = "primary"
        self._prefetch_result = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread: threading.Thread | None = None
        self._sync_thread: threading.Thread | None = None

    @property
    def name(self) -> str:
        return "contexto"

    def is_available(self) -> bool:
        # Selfhost has no auth, so just check the package imports.
        try:
            import contexto  # noqa: F401
            return True
        except ImportError:
            return False

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "base_url",
                "description": "Contexto self-hosted API base URL",
                "default": "http://localhost:4010",
                "env_var": "CONTEXTO_BASE_URL",
            },
        ]

    def initialize(self, session_id: str, **kwargs) -> None:
        from contexto import ContextoClient

        base_url = os.environ.get("CONTEXTO_BASE_URL", "http://localhost:4010")
        self._client = ContextoClient(base_url=base_url)
        self._agent_slug = kwargs.get("agent_identity") or "hermes"
        # Prefer the gateway-supplied platform user_id (Telegram, Discord, etc.).
        # In CLI sessions there is none, so fall back to the agent slug itself —
        # this gives the contexto extractor a stable identity to anchor first-
        # person language to, which prevents the extractor from emitting bare
        # "User" / "I" subjects (or worse, drifting toward whichever named
        # entity is most salient in the conversation context).
        self._user_id = kwargs.get("user_id") or self._agent_slug
        self._agent_context = kwargs.get("agent_context") or "primary"
        # Only the primary agent registers the slug and writes — subagents,
        # cron, and flush share the parent's slug but must not ingest (their
        # turns are tool-output-heavy and would corrupt recall + blow the
        # extract timeout). Subagent RESULTS still reach memory via
        # on_delegation on the parent.
        if self._agent_context == "primary":
            self._client.register_agent(self._agent_slug, name=self._agent_slug)

    def system_prompt_block(self) -> str:
        scope = f"agent={self._agent_slug}"
        if self._user_id:
            scope += f", user={self._user_id}"
        return (
            "# Contexto Memory\n"
            f"Active. Scope: {scope}.\n"
            "Use contexto_search to look up prior knowledge by meaning. "
            "Recent context is auto-injected before each turn."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=3.0)
        with self._prefetch_lock:
            result = self._prefetch_result
            self._prefetch_result = ""
        if not result:
            return ""
        return f"## Contexto Memory\n{result}"

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        if self._agent_context != "primary":
            return

        def _run():
            # Daemon-thread errors print to stderr by default and pollute the
            # chat UI. Route them through the logger instead — error info still
            # lands in agent.log, just not in the user's terminal mid-stream.
            try:
                block = self._client.get_context_for_turn(
                    query, agent=self._agent_slug, user_id=self._user_id, max_results=5
                )
                with self._prefetch_lock:
                    self._prefetch_result = block
            except Exception as e:
                logger.warning("contexto prefetch failed: %s: %s", type(e).__name__, e)

        self._prefetch_thread = threading.Thread(target=_run, daemon=True, name="contexto-prefetch")
        self._prefetch_thread.start()

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        if self._agent_context != "primary":
            return

        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]

        def _sync():
            # Same logger redirection as _run above — keep daemon-thread
            # tracebacks out of the user's terminal. The error message
            # (including the response body, thanks to _raise_with_body in
            # the client) still surfaces in agent.log under "contexto sync".
            try:
                self._client.ingest(messages, agent=self._agent_slug, user_id=self._user_id)
            except Exception as e:
                logger.warning("contexto sync failed: %s: %s", type(e).__name__, e)

        if self._sync_thread and self._sync_thread.is_alive():
            # Give the previous extraction up to 60s before kicking off a new one;
            # otherwise we pile up overlapping ingests on slow turns.
            self._sync_thread.join(timeout=60.0)
        self._sync_thread = threading.Thread(target=_sync, daemon=True, name="contexto-sync")
        self._sync_thread.start()

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [SEARCH_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if tool_name != "contexto_search":
            return tool_error(f"Unknown tool: {tool_name}")
        query = args.get("query", "")
        if not query:
            return tool_error("Missing required parameter: query")
        max_results = min(int(args.get("max_results", 5)), 25)
        result = self._client.search(query, agent=self._agent_slug, user_id=self._user_id)
        items = (result or {}).get("workingMemory") or []
        snippets = [
            {
                "sector": it.get("sector"),
                "content": it.get("content", ""),
                "score": it.get("score", 0),
            }
            for it in items[:max_results]
        ]
        return json.dumps({"results": snippets, "count": len(snippets)})

    def on_delegation(self, task: str, result: str, *,
                      child_session_id: str = "", **kwargs) -> None:
        """Capture (delegation prompt, subagent result) as one parent turn.

        Subagents are silenced for ingest (their raw turns are noise), but
        the curated task→result pair represents real work done on the user's
        behalf and IS worth recalling. Tagged so extraction can see the
        delegation framing.
        """
        if self._agent_context != "primary" or not task or not result:
            return
        framed_user = f"[delegated subtask] {task}"
        framed_assistant = f"[delegation result] {result}"
        self.sync_turn(framed_user, framed_assistant, session_id=child_session_id)

    def shutdown(self) -> None:
        for t in (self._prefetch_thread, self._sync_thread):
            if t and t.is_alive():
                t.join(timeout=5.0)


def register(ctx) -> None:
    ctx.register_memory_provider(ContextoMemoryProvider())

"""Honcho memory plugin — MemoryProvider for Honcho AI-native memory.

Provides cross-session user modeling with dialectic Q&A, semantic search,
peer cards, and persistent conclusions via the Honcho SDK. Honcho provides AI-native cross-session user
modeling with dialectic Q&A, semantic search, peer cards, and conclusions.

The 4 tools (profile, search, context, conclude) are exposed through
the MemoryProvider interface.

Config: Uses the existing Honcho config chain:
  1. $HERMES_HOME/honcho.json (profile-scoped)
  2. ~/.honcho/config.json (legacy global)
  3. Environment variables
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool schemas (moved from tools/honcho_tools.py)
# ---------------------------------------------------------------------------

PROFILE_SCHEMA = {
    "name": "honcho_profile",
    "description": (
        "Retrieve the user's peer card from Honcho — a curated list of key facts "
        "about them (name, role, preferences, communication style, patterns). "
        "Fast, no LLM reasoning, minimal cost. "
        "Use this at conversation start or when you need a quick factual snapshot."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}

SEARCH_SCHEMA = {
    "name": "honcho_search",
    "description": (
        "Semantic search over Honcho's stored context about the user. "
        "Returns raw excerpts ranked by relevance — no LLM synthesis. "
        "Cheaper and faster than honcho_context. "
        "Good when you want to find specific past facts and reason over them yourself."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to search for in Honcho's memory.",
            },
            "max_tokens": {
                "type": "integer",
                "description": "Token budget for returned context (default 800, max 2000).",
            },
        },
        "required": ["query"],
    },
}

CONTEXT_SCHEMA = {
    "name": "honcho_context",
    "description": (
        "Ask Honcho a natural language question and get a synthesized answer. "
        "Uses Honcho's LLM (dialectic reasoning) — higher cost than honcho_profile or honcho_search. "
        "Can query about any peer: the user (default) or the AI assistant."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "A natural language question.",
            },
            "peer": {
                "type": "string",
                "description": "Which peer to query about: 'user' (default) or 'ai'.",
            },
        },
        "required": ["query"],
    },
}

CONCLUDE_SCHEMA = {
    "name": "honcho_conclude",
    "description": (
        "Write a conclusion about the user back to Honcho's memory. "
        "Conclusions are persistent facts that build the user's profile. "
        "Use when the user states a preference, corrects you, or shares "
        "something to remember across sessions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "conclusion": {
                "type": "string",
                "description": "A factual statement about the user to persist.",
            }
        },
        "required": ["conclusion"],
    },
}


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

class HonchoMemoryProvider(MemoryProvider):
    """Honcho AI-native memory with dialectic Q&A and persistent user modeling."""

    def __init__(self):
        self._manager = None   # HonchoSessionManager
        self._config = None    # HonchoClientConfig
        self._session_key = ""
        self._prefetch_result = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread: Optional[threading.Thread] = None
        self._sync_thread: Optional[threading.Thread] = None

    @property
    def name(self) -> str:
        return "honcho"

    def is_available(self) -> bool:
        """Check if Honcho is configured. No network calls."""
        try:
            from plugins.memory.honcho.client import HonchoClientConfig
            cfg = HonchoClientConfig.from_global_config()
            return cfg.enabled and bool(cfg.api_key or cfg.base_url)
        except Exception:
            return False

    def save_config(self, values, hermes_home):
        """Write config to $HERMES_HOME/honcho.json (Honcho SDK native format)."""
        import json
        from pathlib import Path
        config_path = Path(hermes_home) / "honcho.json"
        existing = {}
        if config_path.exists():
            try:
                existing = json.loads(config_path.read_text())
            except Exception:
                pass
        existing.update(values)
        config_path.write_text(json.dumps(existing, indent=2))

    def get_config_schema(self):
        return [
            {"key": "api_key", "description": "Honcho API key", "secret": True, "env_var": "HONCHO_API_KEY", "url": "https://app.honcho.dev"},
            {"key": "base_url", "description": "Honcho base URL", "default": "https://api.honcho.dev"},
        ]

    def initialize(self, session_id: str, **kwargs) -> None:
        """Initialize Honcho session manager."""
        try:
            from plugins.memory.honcho.client import HonchoClientConfig, get_honcho_client
            from plugins.memory.honcho.session import HonchoSessionManager

            cfg = HonchoClientConfig.from_global_config()
            if not cfg.enabled or not (cfg.api_key or cfg.base_url):
                logger.debug("Honcho not configured — plugin inactive")
                return

            self._config = cfg
            client = get_honcho_client(cfg)
            self._manager = HonchoSessionManager(
                honcho=client,
                config=cfg,
                context_tokens=cfg.context_tokens,
            )

            # Build session key from kwargs or session_id
            platform = kwargs.get("platform", "cli")
            user_id = kwargs.get("user_id", "")
            if user_id:
                self._session_key = f"{platform}:{user_id}"
            else:
                self._session_key = session_id

        except ImportError:
            logger.debug("honcho-ai package not installed — plugin inactive")
        except Exception as e:
            logger.warning("Honcho init failed: %s", e)
            self._manager = None

    def system_prompt_block(self) -> str:
        if not self._manager or not self._session_key:
            return ""
        return (
            "# Honcho Memory\n"
            "Active. AI-native cross-session user modeling.\n"
            "Use honcho_profile for a quick factual snapshot, "
            "honcho_search for raw excerpts, honcho_context for synthesized answers, "
            "honcho_conclude to save facts about the user."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Return prefetched dialectic context from background thread."""
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=3.0)
        with self._prefetch_lock:
            result = self._prefetch_result
            self._prefetch_result = ""
        if not result:
            return ""
        return f"## Honcho Context\n{result}"

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Fire a background dialectic query for the upcoming turn."""
        if not self._manager or not self._session_key or not query:
            return

        def _run():
            try:
                result = self._manager.dialectic_query(
                    self._session_key, query, peer="user"
                )
                if result and result.strip():
                    with self._prefetch_lock:
                        self._prefetch_result = result
            except Exception as e:
                logger.debug("Honcho prefetch failed: %s", e)

        self._prefetch_thread = threading.Thread(
            target=_run, daemon=True, name="honcho-prefetch"
        )
        self._prefetch_thread.start()

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Record the conversation turn in Honcho (non-blocking)."""
        if not self._manager or not self._session_key:
            return

        def _sync():
            try:
                session = self._manager.get_or_create_session(self._session_key)
                session.add_message("user", user_content[:4000])
                session.add_message("assistant", assistant_content[:4000])
                # Flush to Honcho API
                self._manager._flush_session(session)
            except Exception as e:
                logger.debug("Honcho sync_turn failed: %s", e)

        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5.0)
        self._sync_thread = threading.Thread(
            target=_sync, daemon=True, name="honcho-sync"
        )
        self._sync_thread.start()

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        """Mirror built-in user profile writes as Honcho conclusions."""
        if action != "add" or target != "user" or not content:
            return
        if not self._manager or not self._session_key:
            return

        def _write():
            try:
                self._manager.create_conclusion(self._session_key, content)
            except Exception as e:
                logger.debug("Honcho memory mirror failed: %s", e)

        t = threading.Thread(target=_write, daemon=True, name="honcho-memwrite")
        t.start()

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """Flush all pending messages to Honcho on session end."""
        if not self._manager:
            return
        # Wait for pending sync
        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=10.0)
        try:
            self._manager.flush_all()
        except Exception as e:
            logger.debug("Honcho session-end flush failed: %s", e)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [PROFILE_SCHEMA, SEARCH_SCHEMA, CONTEXT_SCHEMA, CONCLUDE_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        if not self._manager or not self._session_key:
            return json.dumps({"error": "Honcho is not active for this session."})

        try:
            if tool_name == "honcho_profile":
                card = self._manager.get_peer_card(self._session_key)
                if not card:
                    return json.dumps({"result": "No profile facts available yet."})
                return json.dumps({"result": card})

            elif tool_name == "honcho_search":
                query = args.get("query", "")
                if not query:
                    return json.dumps({"error": "Missing required parameter: query"})
                max_tokens = min(int(args.get("max_tokens", 800)), 2000)
                result = self._manager.search_context(
                    self._session_key, query, max_tokens=max_tokens
                )
                if not result:
                    return json.dumps({"result": "No relevant context found."})
                return json.dumps({"result": result})

            elif tool_name == "honcho_context":
                query = args.get("query", "")
                if not query:
                    return json.dumps({"error": "Missing required parameter: query"})
                peer = args.get("peer", "user")
                result = self._manager.dialectic_query(
                    self._session_key, query, peer=peer
                )
                return json.dumps({"result": result or "No result from Honcho."})

            elif tool_name == "honcho_conclude":
                conclusion = args.get("conclusion", "")
                if not conclusion:
                    return json.dumps({"error": "Missing required parameter: conclusion"})
                ok = self._manager.create_conclusion(self._session_key, conclusion)
                if ok:
                    return json.dumps({"result": f"Conclusion saved: {conclusion}"})
                return json.dumps({"error": "Failed to save conclusion."})

            return json.dumps({"error": f"Unknown tool: {tool_name}"})

        except Exception as e:
            logger.error("Honcho tool %s failed: %s", tool_name, e)
            return json.dumps({"error": f"Honcho {tool_name} failed: {e}"})

    def shutdown(self) -> None:
        for t in (self._prefetch_thread, self._sync_thread):
            if t and t.is_alive():
                t.join(timeout=5.0)
        # Flush any remaining messages
        if self._manager:
            try:
                self._manager.flush_all()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Register Honcho as a memory provider plugin."""
    ctx.register_memory_provider(HonchoMemoryProvider())

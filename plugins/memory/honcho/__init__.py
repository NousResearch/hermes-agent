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


ALL_TOOL_SCHEMAS = [PROFILE_SCHEMA, SEARCH_SCHEMA, CONTEXT_SCHEMA, CONCLUDE_SCHEMA]


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

        # B1: recall_mode — set during initialize from config
        self._recall_mode = "hybrid"  # "context", "tools", or "hybrid"

        # B4: First-turn context baking
        self._first_turn_context: Optional[str] = None
        self._first_turn_lock = threading.Lock()

        # B5: Cost-awareness turn counting and cadence
        self._turn_count = 0
        self._injection_frequency = "every-turn"  # or "first-turn"
        self._context_cadence = 1   # minimum turns between context API calls
        self._dialectic_cadence = 1  # minimum turns between dialectic API calls
        self._reasoning_level_cap: Optional[str] = None  # "minimal", "low", "mid", "high"
        self._last_context_turn = -999
        self._last_dialectic_turn = -999

        # Port #1957: lazy session init for tools-only mode
        self._session_initialized = False
        self._lazy_init_kwargs: Optional[dict] = None
        self._lazy_init_session_id: Optional[str] = None

        # Port #4053: cron guard — when True, plugin is fully inactive
        self._cron_skipped = False

    @property
    def name(self) -> str:
        return "honcho"

    def is_available(self) -> bool:
        """Check if Honcho is configured. No network calls."""
        try:
            from plugins.memory.honcho.client import HonchoClientConfig
            cfg = HonchoClientConfig.from_global_config()
            # Port #2645: baseUrl-only verification — api_key OR base_url suffices
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
            {"key": "baseUrl", "description": "Honcho base URL (for self-hosted)"},
        ]

    def post_setup(self, hermes_home: str, config: dict) -> None:
        """Run the full Honcho setup wizard after provider selection."""
        import types
        from plugins.memory.honcho.cli import cmd_setup
        cmd_setup(types.SimpleNamespace())

    def initialize(self, session_id: str, **kwargs) -> None:
        """Initialize Honcho session manager.

        Handles: cron guard, recall_mode, session name resolution,
        peer memory mode, SOUL.md ai_peer sync, memory file migration,
        and pre-warming context at init.
        """
        try:
            # ----- Port #4053: cron guard -----
            agent_context = kwargs.get("agent_context", "")
            platform = kwargs.get("platform", "cli")
            if agent_context in ("cron", "flush") or platform == "cron":
                logger.debug("Honcho skipped: cron/flush context (agent_context=%s, platform=%s)",
                             agent_context, platform)
                self._cron_skipped = True
                return

            from plugins.memory.honcho.client import HonchoClientConfig, get_honcho_client
            from plugins.memory.honcho.session import HonchoSessionManager

            cfg = HonchoClientConfig.from_global_config()
            if not cfg.enabled or not (cfg.api_key or cfg.base_url):
                logger.debug("Honcho not configured — plugin inactive")
                return

            # Override peer_name with gateway user_id for per-user memory scoping.
            # CLI sessions won't have user_id, so the config default is preserved.
            _gw_user_id = kwargs.get("user_id")
            if _gw_user_id:
                cfg.peer_name = _gw_user_id

            self._config = cfg

            # ----- B1: recall_mode from config -----
            self._recall_mode = cfg.recall_mode  # "context", "tools", or "hybrid"
            logger.debug("Honcho recall_mode: %s", self._recall_mode)

            # ----- B5: cost-awareness config -----
            try:
                raw = cfg.raw or {}
                self._injection_frequency = raw.get("injectionFrequency", "every-turn")
                self._context_cadence = int(raw.get("contextCadence", 1))
                self._dialectic_cadence = int(raw.get("dialecticCadence", 1))
                cap = raw.get("reasoningLevelCap")
                if cap and cap in ("minimal", "low", "mid", "high"):
                    self._reasoning_level_cap = cap
            except Exception as e:
                logger.debug("Honcho cost-awareness config parse error: %s", e)

            # ----- Port #1969: aiPeer sync from SOUL.md — REMOVED -----
            # SOUL.md is persona content, not identity config. aiPeer should
            # only come from honcho.json (host block or root) or the default.
            # See scratch/memory-plugin-ux-specs.md #10 for rationale.

            # ----- Port #1957: lazy session init for tools-only mode -----
            if self._recall_mode == "tools":
                # Defer actual session creation until first tool call
                self._lazy_init_kwargs = kwargs
                self._lazy_init_session_id = session_id
                # Still need a client reference for _ensure_session
                self._config = cfg
                logger.debug("Honcho tools-only mode — deferring session init until first tool call")
                return

            # ----- Eager init (context or hybrid mode) -----
            self._do_session_init(cfg, session_id, **kwargs)

        except ImportError:
            logger.debug("honcho-ai package not installed — plugin inactive")
        except Exception as e:
            logger.warning("Honcho init failed: %s", e)
            self._manager = None

    def _do_session_init(self, cfg, session_id: str, **kwargs) -> None:
        """Shared session initialization logic for both eager and lazy paths."""
        from plugins.memory.honcho.client import get_honcho_client
        from plugins.memory.honcho.session import HonchoSessionManager

        client = get_honcho_client(cfg)
        self._manager = HonchoSessionManager(
            honcho=client,
            config=cfg,
            context_tokens=cfg.context_tokens,
        )

        # ----- B3: resolve_session_name -----
        session_title = kwargs.get("session_title")
        self._session_key = (
            cfg.resolve_session_name(session_title=session_title, session_id=session_id)
            or session_id
            or "hermes-default"
        )
        logger.debug("Honcho session key resolved: %s", self._session_key)

        # Create session eagerly
        session = self._manager.get_or_create(self._session_key)
        self._session_initialized = True

        # ----- B6: Memory file migration (one-time, for new sessions) -----
        try:
            if not session.messages:
                from hermes_constants import get_hermes_home
                mem_dir = str(get_hermes_home() / "memories")
                self._manager.migrate_memory_files(self._session_key, mem_dir)
                logger.debug("Honcho memory file migration attempted for new session: %s", self._session_key)
        except Exception as e:
            logger.debug("Honcho memory file migration skipped: %s", e)

        # ----- B7: Pre-warming context at init -----
        if self._recall_mode in ("context", "hybrid"):
            try:
                self._manager.prefetch_context(self._session_key)
                self._manager.prefetch_dialectic(self._session_key, "What should I know about this user?")
                logger.debug("Honcho pre-warm threads started for session: %s", self._session_key)
            except Exception as e:
                logger.debug("Honcho pre-warm failed: %s", e)

    def _ensure_session(self) -> bool:
        """Lazily initialize the Honcho session (for tools-only mode).

        Returns True if the manager is ready, False otherwise.
        """
        if self._manager and self._session_initialized:
            return True
        if self._cron_skipped:
            return False
        if not self._config or not self._lazy_init_kwargs:
            return False

        try:
            self._do_session_init(
                self._config,
                self._lazy_init_session_id or "hermes-default",
                **self._lazy_init_kwargs,
            )
            # Clear lazy refs
            self._lazy_init_kwargs = None
            self._lazy_init_session_id = None
            return self._manager is not None
        except Exception as e:
            logger.warning("Honcho lazy session init failed: %s", e)
            return False

    def _format_first_turn_context(self, ctx: dict) -> str:
        """Format the prefetch context dict into a readable system prompt block."""
        parts = []

        rep = ctx.get("representation", "")
        if rep:
            parts.append(f"## User Representation\n{rep}")

        card = ctx.get("card", "")
        if card:
            parts.append(f"## User Peer Card\n{card}")

        ai_rep = ctx.get("ai_representation", "")
        if ai_rep:
            parts.append(f"## AI Self-Representation\n{ai_rep}")

        ai_card = ctx.get("ai_card", "")
        if ai_card:
            parts.append(f"## AI Identity Card\n{ai_card}")

        if not parts:
            return ""
        return "\n\n".join(parts)

    def system_prompt_block(self) -> str:
        """Return system prompt text, adapted by recall_mode.

        B4: On the FIRST call, fetch and bake the full Honcho context
        (user representation, peer card, AI representation, continuity synthesis).
        Subsequent calls return the cached block for prompt caching stability.
        """
        if self._cron_skipped:
            return ""
        if not self._manager or not self._session_key:
            # tools-only mode without session yet still returns a minimal block
            if self._recall_mode == "tools" and self._config:
                return (
                    "# Honcho Memory\n"
                    "Active (tools-only mode). Use honcho_profile, honcho_search, "
                    "honcho_context, and honcho_conclude tools to access user memory."
                )
            return ""

        # ----- B4: First-turn context baking -----
        first_turn_block = ""
        if self._recall_mode in ("context", "hybrid"):
            with self._first_turn_lock:
                if self._first_turn_context is None:
                    # First call — fetch and cache
                    try:
                        ctx = self._manager.get_prefetch_context(self._session_key)
                        self._first_turn_context = self._format_first_turn_context(ctx) if ctx else ""
                    except Exception as e:
                        logger.debug("Honcho first-turn context fetch failed: %s", e)
                        self._first_turn_context = ""
                first_turn_block = self._first_turn_context

        # ----- B1: adapt text based on recall_mode -----
        if self._recall_mode == "context":
            header = (
                "# Honcho Memory\n"
                "Active (context-injection mode). Relevant user context is automatically "
                "injected before each turn. No memory tools are available — context is "
                "managed automatically."
            )
        elif self._recall_mode == "tools":
            header = (
                "# Honcho Memory\n"
                "Active (tools-only mode). Use honcho_profile for a quick factual snapshot, "
                "honcho_search for raw excerpts, honcho_context for synthesized answers, "
                "honcho_conclude to save facts about the user. "
                "No automatic context injection — you must use tools to access memory."
            )
        else:  # hybrid
            header = (
                "# Honcho Memory\n"
                "Active (hybrid mode). Relevant context is auto-injected AND memory tools are available. "
                "Use honcho_profile for a quick factual snapshot, "
                "honcho_search for raw excerpts, honcho_context for synthesized answers, "
                "honcho_conclude to save facts about the user."
            )

        if first_turn_block:
            return f"{header}\n\n{first_turn_block}"
        return header

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Return prefetched dialectic context from background thread.

        B1: Returns empty when recall_mode is "tools" (no injection).
        B5: Respects injection_frequency — "first-turn" returns cached/empty after turn 0.
        Port #3265: Truncates to context_tokens budget.
        """
        if self._cron_skipped:
            return ""

        # B1: tools-only mode — no auto-injection
        if self._recall_mode == "tools":
            return ""

        # B5: injection_frequency — if "first-turn" and past first turn, return empty
        if self._injection_frequency == "first-turn" and self._turn_count > 0:
            return ""

        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=3.0)
        with self._prefetch_lock:
            result = self._prefetch_result
            self._prefetch_result = ""
        if not result:
            return ""

        # ----- Port #3265: token budget enforcement -----
        result = self._truncate_to_budget(result)

        return f"## Honcho Context\n{result}"

    def _truncate_to_budget(self, text: str) -> str:
        """Truncate text to fit within context_tokens budget if set."""
        if not self._config or not self._config.context_tokens:
            return text
        budget_chars = self._config.context_tokens * 4  # conservative char estimate
        if len(text) <= budget_chars:
            return text
        # Truncate at word boundary
        truncated = text[:budget_chars]
        last_space = truncated.rfind(" ")
        if last_space > budget_chars * 0.8:
            truncated = truncated[:last_space]
        return truncated + " …"

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Fire a background dialectic query for the upcoming turn.

        B5: Checks cadence before firing background threads.
        """
        if self._cron_skipped:
            return
        if not self._manager or not self._session_key or not query:
            return

        # B1: tools-only mode — no prefetch
        if self._recall_mode == "tools":
            return

        # B5: cadence check — skip if too soon since last dialectic call
        if self._dialectic_cadence > 1:
            if (self._turn_count - self._last_dialectic_turn) < self._dialectic_cadence:
                logger.debug("Honcho dialectic prefetch skipped: cadence %d, turns since last: %d",
                             self._dialectic_cadence, self._turn_count - self._last_dialectic_turn)
                return

        self._last_dialectic_turn = self._turn_count

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

        # Also fire context prefetch if cadence allows
        if self._context_cadence <= 1 or (self._turn_count - self._last_context_turn) >= self._context_cadence:
            self._last_context_turn = self._turn_count
            try:
                self._manager.prefetch_context(self._session_key, query)
            except Exception as e:
                logger.debug("Honcho context prefetch failed: %s", e)

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        """Track turn count for cadence and injection_frequency logic."""
        self._turn_count = turn_number

    @staticmethod
    def _chunk_message(content: str, limit: int) -> list[str]:
        """Split content into chunks that fit within the Honcho message limit.

        Splits at paragraph boundaries when possible, falling back to
        sentence boundaries, then word boundaries. Each continuation
        chunk is prefixed with "[continued] " so Honcho's representation
        engine can reconstruct the full message.
        """
        if len(content) <= limit:
            return [content]

        prefix = "[continued] "
        prefix_len = len(prefix)
        chunks = []
        remaining = content
        first = True
        while remaining:
            effective = limit if first else limit - prefix_len
            if len(remaining) <= effective:
                chunks.append(remaining if first else prefix + remaining)
                break

            segment = remaining[:effective]

            # Try paragraph break, then sentence, then word
            cut = segment.rfind("\n\n")
            if cut < effective * 0.3:
                cut = segment.rfind(". ")
                if cut >= 0:
                    cut += 2  # include the period and space
            if cut < effective * 0.3:
                cut = segment.rfind(" ")
            if cut < effective * 0.3:
                cut = effective  # hard cut

            chunk = remaining[:cut].rstrip()
            remaining = remaining[cut:].lstrip()
            if not first:
                chunk = prefix + chunk
            chunks.append(chunk)
            first = False

        return chunks

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Record the conversation turn in Honcho (non-blocking).

        Messages exceeding the Honcho API limit (default 25k chars) are
        split into multiple messages with continuation markers.
        """
        if self._cron_skipped:
            return
        if not self._manager or not self._session_key:
            return

        msg_limit = self._config.message_max_chars if self._config else 25000

        def _sync():
            try:
                session = self._manager.get_or_create(self._session_key)
                for chunk in self._chunk_message(user_content, msg_limit):
                    session.add_message("user", chunk)
                for chunk in self._chunk_message(assistant_content, msg_limit):
                    session.add_message("assistant", chunk)
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
        if self._cron_skipped:
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
        if self._cron_skipped:
            return
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
        """Return tool schemas, respecting recall_mode.

        B1: context-only mode hides all tools.
        """
        if self._cron_skipped:
            return []
        if self._recall_mode == "context":
            return []
        return list(ALL_TOOL_SCHEMAS)

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        """Handle a Honcho tool call, with lazy session init for tools-only mode."""
        if self._cron_skipped:
            return json.dumps({"error": "Honcho is not active (cron context)."})

        # Port #1957: ensure session is initialized for tools-only mode
        if not self._session_initialized:
            if not self._ensure_session():
                return json.dumps({"error": "Honcho session could not be initialized."})

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

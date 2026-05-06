"""obsidimem memory plugin — MemoryProvider backed by obsidi-mem API.

Provides hybrid search (pgvector + BM25), peer cards, session summarization,
synthesized answers, and observation storage via the obsidi-mem REST API.

Config: $HERMES_HOME/obsidimem.json
  api_base_url:                  "http://127.0.0.1:8000"
  observer_name:                 "hermes"    # agent peer name in obsidi-mem
  observed_name:                 "doug"      # user peer name in obsidi-mem
  recall_mode:                   "hybrid"    # context | tools | hybrid
  budget:                        1200        # context token budget
  timeout:                       60.0        # HTTP request timeout (seconds)
  trigger_dreamer_on_session_end: false      # POST /dreamer when session ends
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

RECALL_SCHEMA = {
    "name": "obsidimem_recall",
    "description": (
        "Search obsidi-mem for relevant memory context about the user. "
        "Returns peer card facts, recent observations, and semantic hits. "
        "Use when you need to recall what's known about a topic or past interactions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to search for in memory.",
            },
            "budget": {
                "type": "integer",
                "description": "Token budget for returned context (default: from config).",
            },
        },
        "required": ["query"],
    },
}

STORE_SCHEMA = {
    "name": "obsidimem_store",
    "description": (
        "Explicitly store an observation about the user in obsidi-mem. "
        "Use for important facts, preferences, or decisions worth preserving across sessions. "
        "The deriver also stores observations automatically after each turn — "
        "use this only for observations worth saving immediately."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The observation to store.",
            },
        },
        "required": ["content"],
    },
}

ANSWER_SCHEMA = {
    "name": "obsidimem_answer",
    "description": (
        "Ask obsidi-mem a natural language question and get an LLM-synthesized answer "
        "grounded in stored observations. Higher cost than obsidimem_recall. "
        "Use when you need a synthesized response rather than raw context excerpts."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "A natural language question to answer from memory.",
            },
        },
        "required": ["query"],
    },
}

ALL_TOOL_SCHEMAS = [RECALL_SCHEMA, STORE_SCHEMA, ANSWER_SCHEMA]


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

class ObsidimemProvider(MemoryProvider):
    """obsidi-mem memory provider — hybrid pgvector+BM25 search with peer cards."""

    def __init__(self):
        self._config: Optional[dict] = None
        self._client = None           # httpx.Client (sync)
        self._session_id: Optional[str] = None
        self._recall_mode = "hybrid"
        self._cron_skipped = False
        self._session_initialized = False

        # Prefetch state — background thread writes, prefetch() reads and clears
        self._prefetch_result = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread: Optional[threading.Thread] = None

        # Background sync (deriver) thread
        self._sync_thread: Optional[threading.Thread] = None
        self._write_thread: Optional[threading.Thread] = None

        # Lazy init state for tools-only mode
        self._lazy_init_session_id: Optional[str] = None
        self._lazy_init_kwargs: Optional[dict] = None

    @property
    def name(self) -> str:
        return "obsidimem"

    # -- Config helpers -------------------------------------------------------

    def _load_config(self, hermes_home: str) -> Optional[dict]:
        config_path = Path(hermes_home) / "obsidimem.json"
        if not config_path.exists():
            return None
        try:
            return json.loads(config_path.read_text())
        except Exception as e:
            logger.debug("obsidimem: config parse error: %s", e)
            return None

    def is_available(self) -> bool:
        """No network calls — just check for config with api_base_url."""
        import os
        hermes_home = os.environ.get("HERMES_HOME", str(Path.home() / ".hermes"))
        cfg = self._load_config(hermes_home)
        return bool(cfg and cfg.get("api_base_url"))

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {"key": "api_base_url", "description": "obsidi-mem API base URL", "required": True, "default": "http://127.0.0.1:8000"},
            {"key": "observer_name", "description": "Agent peer name in obsidi-mem (e.g. 'hermes')", "required": True, "default": "hermes"},
            {"key": "observed_name", "description": "User peer name in obsidi-mem (e.g. 'doug')", "required": True},
            {"key": "recall_mode", "description": "Recall mode", "choices": ["context", "tools", "hybrid"], "default": "hybrid"},
            {"key": "budget", "description": "Context token budget", "default": 1200},
            {"key": "timeout", "description": "HTTP timeout (seconds)", "default": 60.0},
            {"key": "trigger_dreamer_on_session_end", "description": "Trigger Dreamer when session ends", "default": False},
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        config_path = Path(hermes_home) / "obsidimem.json"
        existing = {}
        if config_path.exists():
            try:
                existing = json.loads(config_path.read_text())
            except Exception:
                pass
        existing.update(values)
        config_path.write_text(json.dumps(existing, indent=2))

    # -- Lifecycle ------------------------------------------------------------

    def initialize(self, session_id: str, **kwargs) -> None:
        try:
            agent_context = kwargs.get("agent_context", "")
            platform = kwargs.get("platform", "cli")
            if agent_context in ("cron", "flush") or platform == "cron":
                logger.debug("obsidimem: skipping (cron/flush context)")
                self._cron_skipped = True
                return

            import httpx

            hermes_home = kwargs.get("hermes_home", str(Path.home() / ".hermes"))
            cfg = self._load_config(hermes_home)
            if not cfg or not cfg.get("api_base_url"):
                logger.debug("obsidimem: no config found — plugin inactive")
                return

            self._config = cfg
            self._recall_mode = cfg.get("recall_mode", "hybrid")
            self._client = httpx.Client(
                base_url=cfg["api_base_url"].rstrip("/"),
                timeout=float(cfg.get("timeout", 60.0)),
            )

            # Ensure both peers exist (idempotent)
            self._ensure_peers()

            if self._recall_mode == "tools":
                # Defer session creation until first tool call
                self._lazy_init_session_id = session_id
                self._lazy_init_kwargs = kwargs
                logger.debug("obsidimem: tools-only mode — deferring session init")
                return

            # Eager session init for context/hybrid mode
            self._init_session(session_id)

            # Pre-warm context in background so turn 1 isn't empty
            if self._recall_mode in ("context", "hybrid"):
                t = threading.Thread(
                    target=self._do_prefetch,
                    args=("",),
                    daemon=True,
                    name="obsidimem-prewarm",
                )
                t.start()
                self._prefetch_thread = t

        except ImportError:
            logger.debug("obsidimem: httpx not installed — plugin inactive")
        except Exception as e:
            logger.warning("obsidimem: init failed: %s", e)
            self._client = None

    def _ensure_peers(self) -> None:
        """Get or create observer and observed peers (idempotent POST)."""
        try:
            for name in (self._config["observer_name"], self._config["observed_name"]):
                self._client.post("/memory/peers", json={"name": name})
        except Exception as e:
            logger.debug("obsidimem: peer ensure failed: %s", e)

    def _init_session(self, session_id: str) -> None:
        """Create an obsidi-mem session and store its UUID."""
        try:
            resp = self._client.post(
                "/memory/sessions",
                json={
                    "peer_name": self._config["observer_name"],
                    "metadata": {"hermes_session_id": session_id},
                },
            )
            resp.raise_for_status()
            self._session_id = resp.json()["id"]
            self._session_initialized = True
            logger.debug("obsidimem: session created %s", self._session_id)
        except Exception as e:
            logger.warning("obsidimem: session init failed: %s", e)

    def _ensure_session(self) -> bool:
        """Lazily initialize session for tools-only mode."""
        if self._session_initialized:
            return True
        if self._cron_skipped or not self._config or not self._client:
            return False
        try:
            self._ensure_peers()
            self._init_session(self._lazy_init_session_id or "hermes-default")
            self._lazy_init_session_id = None
            self._lazy_init_kwargs = None
            return self._session_initialized
        except Exception as e:
            logger.warning("obsidimem: lazy session init failed: %s", e)
            return False

    # -- Context injection ----------------------------------------------------

    def _do_prefetch(self, query: str) -> None:
        """Fetch context from API and write to prefetch cache."""
        try:
            params: Dict[str, Any] = {
                "observer": self._config["observer_name"],
                "observed": self._config["observed_name"],
                "budget": self._config.get("budget", 1200),
            }
            if query:
                params["query"] = query
            if self._session_id:
                params["session_id"] = self._session_id
            resp = self._client.get("/memory/context", params=params)
            resp.raise_for_status()
            formatted = self._format_context(resp.json())
            with self._prefetch_lock:
                self._prefetch_result = formatted
        except Exception as e:
            logger.debug("obsidimem: prefetch failed: %s", e)

    def _format_context(self, data: dict) -> str:
        """Format MemoryContextResponse dict into a readable block."""
        parts = []

        facts = data.get("contextual_facts", [])
        if facts:
            lines = [f["value"] for f in facts if f.get("value")]
            if lines:
                parts.append("## Context Facts\n" + "\n".join(f"- {l}" for l in lines))

        card = data.get("peer_card", [])
        if card:
            lines = [f.get("value") or f.get("content", "") for f in card if f.get("value") or f.get("content")]
            if lines:
                parts.append("## User Profile\n" + "\n".join(f"- {l}" for l in lines))

        recent = data.get("recent_observations", [])
        semantic = data.get("semantic_observations", [])
        seen = {o.get("id") for o in recent}
        obs = recent + [o for o in semantic if o.get("id") not in seen]
        if obs:
            lines = [o["content"] for o in obs if o.get("content")][:10]
            if lines:
                parts.append("## Memory\n" + "\n".join(f"- {l}" for l in lines))

        summary = data.get("session_summary")
        if summary:
            parts.append(f"## Session Summary\n{summary}")

        return "\n\n".join(parts)

    def system_prompt_block(self) -> str:
        if self._cron_skipped or not self._config:
            return ""
        if self._recall_mode == "context":
            return (
                "# obsidi-mem Memory\n"
                "Active (context-injection mode). Relevant memory is automatically "
                "injected before each turn. No memory tools are available."
            )
        elif self._recall_mode == "tools":
            return (
                "# obsidi-mem Memory\n"
                "Active (tools-only mode). Use obsidimem_recall to search memory, "
                "obsidimem_store to save an observation, obsidimem_answer for a synthesized answer."
            )
        else:
            return (
                "# obsidi-mem Memory\n"
                "Active (hybrid mode). Memory context is auto-injected before each turn "
                "and tools are available: obsidimem_recall (search), obsidimem_store (save), "
                "obsidimem_answer (synthesized answer)."
            )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if self._cron_skipped or self._recall_mode == "tools" or not self._config:
            return ""
        # Wait briefly for a running background thread
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=3.0)
        with self._prefetch_lock:
            result = self._prefetch_result
            self._prefetch_result = ""
        if result:
            return result
        # Synchronous fallback on first turn (prewarm thread missed or no-op query)
        self._do_prefetch(query)
        with self._prefetch_lock:
            result = self._prefetch_result
            self._prefetch_result = ""
        return result

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        if self._cron_skipped or self._recall_mode == "tools" or not self._config:
            return
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            return
        self._prefetch_thread = threading.Thread(
            target=self._do_prefetch,
            args=(query,),
            daemon=True,
            name="obsidimem-prefetch",
        )
        self._prefetch_thread.start()

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        **kwargs,
    ) -> None:
        if not new_session_id or self._cron_skipped or not self._config or not self._client:
            return

        for t in (self._sync_thread, self._prefetch_thread):
            if t and t.is_alive():
                t.join(timeout=10.0)

        if self._session_id:
            try:
                self._client.patch(f"/memory/sessions/{self._session_id}/end")
            except Exception as e:
                logger.debug("obsidimem: session switch end failed: %s", e)

            if reset and self._config.get("trigger_dreamer_on_session_end", False):
                try:
                    self._client.post(
                        "/dreamer",
                        json={
                            "observer_name": self._config["observer_name"],
                            "observed_name": self._config["observed_name"],
                        },
                    )
                except Exception as e:
                    logger.debug("obsidimem: session switch dreamer failed: %s", e)

        with self._prefetch_lock:
            self._prefetch_result = ""
        self._prefetch_thread = None
        self._session_id = None
        self._session_initialized = False

        if self._recall_mode == "tools":
            self._lazy_init_session_id = new_session_id
        else:
            self._init_session(new_session_id)

    # -- Write hooks ----------------------------------------------------------

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Post completed turn to the deriver pipeline (non-blocking)."""
        if self._cron_skipped or not self._config or not self._client:
            return

        observer_name = self._config["observer_name"]
        observed_name = self._config["observed_name"]
        messages = [
            {"role": "user", "content": user_content, "peer_name": observed_name},
            {"role": "assistant", "content": assistant_content, "peer_name": observer_name},
        ]
        payload: Dict[str, Any] = {
            "observer_name": observer_name,
            "observed_name": observed_name,
            "messages": messages,
        }
        if self._session_id:
            payload["session_id"] = self._session_id

        def _sync():
            try:
                self._client.post("/memory/deriver", json=payload)
            except Exception as e:
                logger.debug("obsidimem: sync_turn failed: %s", e)

        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5.0)
        self._sync_thread = threading.Thread(target=_sync, daemon=True, name="obsidimem-sync")
        self._sync_thread.start()

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        """Extract observations from messages about to be compressed/discarded."""
        if self._cron_skipped or not self._config or not self._client or not messages:
            return ""

        observer_name = self._config["observer_name"]
        observed_name = self._config["observed_name"]

        api_msgs = []
        for m in messages:
            role = m.get("role")
            content = m.get("content") or ""
            if role not in ("user", "assistant") or not content:
                continue
            peer_name = observed_name if role == "user" else observer_name
            api_msgs.append({"role": role, "content": content, "peer_name": peer_name})

        if not api_msgs:
            return ""

        payload: Dict[str, Any] = {
            "observer_name": observer_name,
            "observed_name": observed_name,
            "messages": api_msgs,
        }
        if self._session_id:
            payload["session_id"] = self._session_id

        def _compress_sync():
            try:
                self._client.post("/memory/deriver", json=payload)
            except Exception as e:
                if e.__class__.__name__ in {"ReadTimeout", "TimeoutException"}:
                    logger.warning("obsidimem: pre-compress derivation timed out: %s", e)
                else:
                    logger.debug("obsidimem: pre-compress derivation failed: %s", e)

        threading.Thread(
            target=_compress_sync,
            daemon=True,
            name="obsidimem-compress",
        ).start()
        return ""

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Mirror explicit /remember writes to obsidimem as level=explicit observations."""
        if action == "remove" or self._cron_skipped or not self._config or not self._client:
            return

        payload = {
            "observations": [{
                "observer_name": self._config["observer_name"],
                "observed_name": self._config["observed_name"],
                "content": f"[memory:{target}:{action}] {content}",
                "level": "explicit",
            }]
        }

        def _write():
            try:
                self._client.post("/memory/observations", json=payload)
            except Exception as e:
                logger.debug("obsidimem: on_memory_write failed: %s", e)

        self._write_thread = threading.Thread(target=_write, daemon=True, name="obsidimem-write")
        self._write_thread.start()

    def on_delegation(self, task: str, result: str, *, child_session_id: str = "", **kwargs) -> None:
        """Store delegated task+result as an explicit observation."""
        if self._cron_skipped or not self._config or not self._client:
            return

        content = f"Delegated task: {task}\nResult: {result}"
        payload = {
            "observations": [{
                "observer_name": self._config["observer_name"],
                "observed_name": self._config["observed_name"],
                "content": content,
                "level": "explicit",
            }]
        }

        def _delegation_sync():
            try:
                self._client.post("/memory/observations", json=payload)
            except Exception as e:
                if e.__class__.__name__ in {"ReadTimeout", "TimeoutException"}:
                    logger.warning("obsidimem: delegation observation sync timed out: %s", e)
                else:
                    logger.debug("obsidimem: delegation observation sync failed: %s", e)

        threading.Thread(
            target=_delegation_sync,
            daemon=True,
            name="obsidimem-delegation",
        ).start()

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """End the obsidi-mem session and optionally trigger Dreamer."""
        if self._cron_skipped or not self._config or not self._client:
            return
        for t in (self._sync_thread, self._prefetch_thread):
            if t and t.is_alive():
                t.join(timeout=10.0)
        if self._session_id:
            try:
                self._client.patch(f"/memory/sessions/{self._session_id}/end")
            except Exception as e:
                logger.debug("obsidimem: session end failed: %s", e)
        if self._config.get("trigger_dreamer_on_session_end", False):
            try:
                self._client.post(
                    "/dreamer",
                    json={
                        "observer_name": self._config["observer_name"],
                        "observed_name": self._config["observed_name"],
                    },
                )
            except Exception as e:
                logger.debug("obsidimem: dreamer trigger failed: %s", e)

    # -- Tool exposure --------------------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        if self._cron_skipped or self._recall_mode == "context":
            return []
        return list(ALL_TOOL_SCHEMAS)

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if self._cron_skipped:
            return tool_error("obsidimem is not active (cron context).")
        if not self._ensure_session():
            return tool_error("obsidimem session could not be initialized.")
        if not self._client or not self._config:
            return tool_error("obsidimem is not configured.")
        try:
            if tool_name == "obsidimem_recall":
                return self._tool_recall(args)
            elif tool_name == "obsidimem_store":
                return self._tool_store(args)
            elif tool_name == "obsidimem_answer":
                return self._tool_answer(args)
            return tool_error(f"Unknown tool: {tool_name}")
        except Exception as e:
            logger.error("obsidimem tool %s failed: %s", tool_name, e)
            return tool_error(f"obsidimem {tool_name} failed: {e}")

    def _tool_recall(self, args: dict) -> str:
        query = args.get("query", "").strip()
        if not query:
            return tool_error("Missing required parameter: query")
        params: Dict[str, Any] = {
            "observer": self._config["observer_name"],
            "observed": self._config["observed_name"],
            "query": query,
            "budget": args.get("budget") or self._config.get("budget", 1200),
        }
        if self._session_id:
            params["session_id"] = self._session_id
        resp = self._client.get("/memory/context", params=params)
        resp.raise_for_status()
        formatted = self._format_context(resp.json())
        return json.dumps({"result": formatted or "No relevant memory found."})

    def _tool_store(self, args: dict) -> str:
        content = args.get("content", "").strip()
        if not content:
            return tool_error("Missing required parameter: content")
        payload = {
            "observations": [{
                "observer_name": self._config["observer_name"],
                "observed_name": self._config["observed_name"],
                "content": content,
                "level": "explicit",
            }]
        }
        resp = self._client.post("/memory/observations", json=payload)
        resp.raise_for_status()
        return json.dumps({"result": f"Stored {len(resp.json())} observation(s)."})

    def _tool_answer(self, args: dict) -> str:
        query = args.get("query", "").strip()
        if not query:
            return tool_error("Missing required parameter: query")
        resp = self._client.post(
            "/memory/answer",
            json={
                "observer_name": self._config["observer_name"],
                "observed_name": self._config["observed_name"],
                "query": query,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return json.dumps({"result": data.get("answer") or "No answer synthesized."})

    # -- Shutdown -------------------------------------------------------------

    def shutdown(self) -> None:
        for t in (self._prefetch_thread, self._sync_thread, self._write_thread):
            if t and t.is_alive():
                t.join(timeout=5.0)
        if self._client:
            try:
                self._client.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Register obsidimem as a memory provider plugin."""
    ctx.register_memory_provider(ObsidimemProvider())

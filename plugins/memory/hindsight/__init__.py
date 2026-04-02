"""Hindsight memory plugin — MemoryProvider interface.

Long-term memory with knowledge graph, entity resolution, and multi-strategy
retrieval. Supports cloud (API key) and local (embedded PostgreSQL) modes.

Original PR #1811 by benfrank241, adapted to MemoryProvider ABC.

Config via environment variables:
  HINDSIGHT_API_KEY   — API key for Hindsight Cloud
  HINDSIGHT_BANK_ID   — memory bank identifier (default: hermes)
  HINDSIGHT_BUDGET    — recall budget: low/mid/high (default: mid)
  HINDSIGHT_API_URL   — API endpoint
  HINDSIGHT_MODE      — cloud or local (default: cloud)

Or via $HERMES_HOME/hindsight/config.json (profile-scoped), falling back to
~/.hindsight/config.json (legacy, shared) for backward compatibility.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
from typing import Any, Dict, List

from agent.memory_provider import MemoryProvider

logger = logging.getLogger(__name__)

_DEFAULT_API_URL = "https://api.hindsight.vectorize.io"
_VALID_BUDGETS = {"low", "mid", "high"}


# ---------------------------------------------------------------------------
# Thread helper (from original PR — avoids aiohttp event loop conflicts)
# ---------------------------------------------------------------------------

def _run_in_thread(fn, timeout: float = 30.0):
    result_q: queue.Queue = queue.Queue(maxsize=1)

    def _run():
        import asyncio
        asyncio.set_event_loop(None)
        try:
            result_q.put(("ok", fn()))
        except Exception as exc:
            result_q.put(("err", exc))

    t = threading.Thread(target=_run, daemon=True, name="hindsight-call")
    t.start()
    kind, value = result_q.get(timeout=timeout)
    if kind == "err":
        raise value
    return value


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

RETAIN_SCHEMA = {
    "name": "hindsight_retain",
    "description": (
        "Store information to long-term memory. Hindsight automatically "
        "extracts structured facts, resolves entities, and indexes for retrieval."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The information to store."},
            "context": {"type": "string", "description": "Short label (e.g. 'user preference', 'project decision')."},
        },
        "required": ["content"],
    },
}

RECALL_SCHEMA = {
    "name": "hindsight_recall",
    "description": (
        "Search long-term memory. Returns memories ranked by relevance using "
        "semantic search, keyword matching, entity graph traversal, and reranking."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
        },
        "required": ["query"],
    },
}

REFLECT_SCHEMA = {
    "name": "hindsight_reflect",
    "description": (
        "Synthesize a reasoned answer from long-term memories. Unlike recall, "
        "this reasons across all stored memories to produce a coherent response."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The question to reflect on."},
        },
        "required": ["query"],
    },
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    """Load config from profile-scoped path, legacy path, or env vars.

    Resolution order:
      1. $HERMES_HOME/hindsight/config.json  (profile-scoped)
      2. ~/.hindsight/config.json             (legacy, shared)
      3. Environment variables
    """
    from pathlib import Path
    from hermes_constants import get_hermes_home

    # Profile-scoped path (preferred)
    profile_path = get_hermes_home() / "hindsight" / "config.json"
    if profile_path.exists():
        try:
            return json.loads(profile_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    # Legacy shared path (backward compat)
    legacy_path = Path.home() / ".hindsight" / "config.json"
    if legacy_path.exists():
        try:
            return json.loads(legacy_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    return {
        "mode": os.environ.get("HINDSIGHT_MODE", "cloud"),
        "apiKey": os.environ.get("HINDSIGHT_API_KEY", ""),
        "banks": {
            "hermes": {
                "bankId": os.environ.get("HINDSIGHT_BANK_ID", "hermes"),
                "budget": os.environ.get("HINDSIGHT_BUDGET", "mid"),
                "enabled": True,
            }
        },
    }


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

class HindsightMemoryProvider(MemoryProvider):
    """Hindsight long-term memory with knowledge graph and multi-strategy retrieval."""

    def __init__(self):
        self._config = None
        self._api_key = None
        self._bank_id = "hermes"
        self._budget = "mid"
        self._mode = "cloud"
        self._prefetch_result = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread = None
        self._sync_thread = None

    @property
    def name(self) -> str:
        return "hindsight"

    def is_available(self) -> bool:
        try:
            cfg = _load_config()
            mode = cfg.get("mode", "cloud")
            if mode == "local":
                embed = cfg.get("embed", {})
                return bool(embed.get("llmApiKey") or os.environ.get("HINDSIGHT_LLM_API_KEY"))
            api_key = cfg.get("apiKey") or os.environ.get("HINDSIGHT_API_KEY", "")
            return bool(api_key)
        except Exception:
            return False

    def save_config(self, values, hermes_home):
        """Write config to $HERMES_HOME/hindsight/config.json."""
        import json
        from pathlib import Path
        config_dir = Path(hermes_home) / "hindsight"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / "config.json"
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
            {"key": "mode", "description": "Cloud API or local embedded mode", "default": "cloud", "choices": ["cloud", "local"]},
            {"key": "api_key", "description": "Hindsight Cloud API key", "secret": True, "env_var": "HINDSIGHT_API_KEY", "url": "https://app.hindsight.vectorize.io"},
            {"key": "bank_id", "description": "Memory bank identifier", "default": "hermes"},
            {"key": "budget", "description": "Recall thoroughness", "default": "mid", "choices": ["low", "mid", "high"]},
            {"key": "llm_provider", "description": "LLM provider for local mode", "default": "anthropic", "choices": ["anthropic", "openai", "groq", "ollama"]},
            {"key": "llm_api_key", "description": "LLM API key for local mode", "secret": True, "env_var": "HINDSIGHT_LLM_API_KEY"},
            {"key": "llm_model", "description": "LLM model for local mode", "default": "claude-haiku-4-5-20251001"},
        ]

    def _make_client(self):
        """Create a fresh Hindsight client (thread-safe)."""
        if self._mode == "local":
            from hindsight import HindsightEmbedded
            embed = self._config.get("embed", {})
            return HindsightEmbedded(
                profile=embed.get("profile", "hermes"),
                llm_provider=embed.get("llmProvider", ""),
                llm_api_key=embed.get("llmApiKey", ""),
                llm_model=embed.get("llmModel", ""),
            )
        from hindsight_client import Hindsight
        return Hindsight(api_key=self._api_key, timeout=30.0)

    def initialize(self, session_id: str, **kwargs) -> None:
        self._config = _load_config()
        self._mode = self._config.get("mode", "cloud")
        self._api_key = self._config.get("apiKey") or os.environ.get("HINDSIGHT_API_KEY", "")

        banks = self._config.get("banks", {}).get("hermes", {})
        self._bank_id = banks.get("bankId", "hermes")
        budget = banks.get("budget", "mid")
        self._budget = budget if budget in _VALID_BUDGETS else "mid"

        # Ensure bank exists
        try:
            client = _run_in_thread(self._make_client)
            _run_in_thread(lambda: client.create_bank(bank_id=self._bank_id, name=self._bank_id))
        except Exception:
            pass  # Already exists

    def system_prompt_block(self) -> str:
        return (
            f"# Hindsight Memory\n"
            f"Active. Bank: {self._bank_id}, budget: {self._budget}.\n"
            f"Use hindsight_recall to search, hindsight_reflect for synthesis, "
            f"hindsight_retain to store facts."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=3.0)
        with self._prefetch_lock:
            result = self._prefetch_result
            self._prefetch_result = ""
        if not result:
            return ""
        return f"## Hindsight Memory\n{result}"

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        def _run():
            try:
                client = self._make_client()
                resp = client.recall(bank_id=self._bank_id, query=query, budget=self._budget)
                if resp.results:
                    text = "\n".join(r.text for r in resp.results if r.text)
                    with self._prefetch_lock:
                        self._prefetch_result = text
            except Exception as e:
                logger.debug("Hindsight prefetch failed: %s", e)

        self._prefetch_thread = threading.Thread(target=_run, daemon=True, name="hindsight-prefetch")
        self._prefetch_thread.start()

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Retain conversation turn in background (non-blocking)."""
        combined = f"User: {user_content}\nAssistant: {assistant_content}"

        def _sync():
            try:
                _run_in_thread(
                    lambda: self._make_client().retain(
                        bank_id=self._bank_id, content=combined, context="conversation"
                    )
                )
            except Exception as e:
                logger.warning("Hindsight sync failed: %s", e)

        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5.0)
        self._sync_thread = threading.Thread(target=_sync, daemon=True, name="hindsight-sync")
        self._sync_thread.start()

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [RETAIN_SCHEMA, RECALL_SCHEMA, REFLECT_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        if tool_name == "hindsight_retain":
            content = args.get("content", "")
            if not content:
                return json.dumps({"error": "Missing required parameter: content"})
            context = args.get("context")
            try:
                _run_in_thread(
                    lambda: self._make_client().retain(
                        bank_id=self._bank_id, content=content, context=context
                    )
                )
                return json.dumps({"result": "Memory stored successfully."})
            except Exception as e:
                return json.dumps({"error": f"Failed to store memory: {e}"})

        elif tool_name == "hindsight_recall":
            query = args.get("query", "")
            if not query:
                return json.dumps({"error": "Missing required parameter: query"})
            try:
                resp = _run_in_thread(
                    lambda: self._make_client().recall(
                        bank_id=self._bank_id, query=query, budget=self._budget
                    )
                )
                if not resp.results:
                    return json.dumps({"result": "No relevant memories found."})
                lines = [f"{i}. {r.text}" for i, r in enumerate(resp.results, 1)]
                return json.dumps({"result": "\n".join(lines)})
            except Exception as e:
                return json.dumps({"error": f"Failed to search memory: {e}"})

        elif tool_name == "hindsight_reflect":
            query = args.get("query", "")
            if not query:
                return json.dumps({"error": "Missing required parameter: query"})
            try:
                resp = _run_in_thread(
                    lambda: self._make_client().reflect(
                        bank_id=self._bank_id, query=query, budget=self._budget
                    )
                )
                return json.dumps({"result": resp.text or "No relevant memories found."})
            except Exception as e:
                return json.dumps({"error": f"Failed to reflect: {e}"})

        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    def shutdown(self) -> None:
        for t in (self._prefetch_thread, self._sync_thread):
            if t and t.is_alive():
                t.join(timeout=5.0)


def register(ctx) -> None:
    """Register Hindsight as a memory provider plugin."""
    ctx.register_memory_provider(HindsightMemoryProvider())

"""Hindsight memory plugin — MemoryProvider interface.

Long-term memory with knowledge graph, entity resolution, and multi-strategy
retrieval. Supports cloud (API key) and local modes.

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

import asyncio
import json
import logging
import os
import threading
from typing import Any, Dict, List

from agent.memory_provider import MemoryProvider

logger = logging.getLogger(__name__)

_DEFAULT_API_URL = "https://api.hindsight.vectorize.io"
_DEFAULT_LOCAL_URL = "http://localhost:8888"
_VALID_BUDGETS = {"low", "mid", "high"}


# ---------------------------------------------------------------------------
# Dedicated event loop for Hindsight async calls (one per process, reused).
# Avoids creating ephemeral loops that leak aiohttp sessions.
# ---------------------------------------------------------------------------

_loop: asyncio.AbstractEventLoop | None = None
_loop_thread: threading.Thread | None = None
_loop_lock = threading.Lock()


def _get_loop() -> asyncio.AbstractEventLoop:
    """Return a long-lived event loop running on a background thread."""
    global _loop, _loop_thread
    with _loop_lock:
        if _loop is not None and _loop.is_running():
            return _loop
        _loop = asyncio.new_event_loop()

        def _run():
            asyncio.set_event_loop(_loop)
            _loop.run_forever()

        _loop_thread = threading.Thread(target=_run, daemon=True, name="hindsight-loop")
        _loop_thread.start()
        return _loop


def _run_sync(coro, timeout: float = 120.0):
    """Schedule *coro* on the shared loop and block until done."""
    loop = _get_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result(timeout=timeout)


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
        self._api_url = _DEFAULT_API_URL
        self._bank_id = "hermes"
        self._budget = "mid"
        self._mode = "cloud"
        self._recall_mode = "hybrid"  # "context", "tools", or "hybrid"
        self._client = None
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
                return True
            has_key = bool(cfg.get("apiKey") or os.environ.get("HINDSIGHT_API_KEY", ""))
            has_url = bool(cfg.get("api_url") or os.environ.get("HINDSIGHT_API_URL", ""))
            return has_key or has_url
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
            {"key": "api_url", "description": "Hindsight API URL", "default": _DEFAULT_API_URL, "when": {"mode": "cloud"}},
            {"key": "api_key", "description": "Hindsight Cloud API key", "secret": True, "env_var": "HINDSIGHT_API_KEY", "url": "https://ui.hindsight.vectorize.io", "when": {"mode": "cloud"}},
            {"key": "llm_provider", "description": "LLM provider for local mode", "default": "openai", "choices": ["openai", "anthropic", "gemini", "groq", "minimax", "ollama"], "when": {"mode": "local"}},
            {"key": "llm_api_key", "description": "LLM API key for local Hindsight", "secret": True, "env_var": "HINDSIGHT_LLM_API_KEY", "when": {"mode": "local"}},
            {"key": "llm_model", "description": "LLM model for local mode", "default": "gpt-4o-mini", "when": {"mode": "local"}},
            {"key": "recall_mode", "description": "Recall mode", "default": "hybrid", "choices": ["context", "tools", "hybrid"]},
        ]

    def _get_client(self):
        """Return the cached Hindsight client (created once, reused)."""
        if self._client is None:
            if self._mode == "local":
                from hindsight import HindsightEmbedded
                client = HindsightEmbedded(
                    profile=self._config.get("profile", "hermes"),
                    llm_provider=self._config.get("llm_provider", ""),
                    llm_api_key=self._config.get("llmApiKey") or os.environ.get("HINDSIGHT_LLM_API_KEY", ""),
                    llm_model=self._config.get("llm_model", ""),
                )
                # Disable __del__ to prevent "attached to a different loop"
                # errors during GC — we handle cleanup in shutdown() instead.
                client.__class__ = type(
                    "HindsightEmbedded", (HindsightEmbedded,), {"__del__": lambda self: None}
                )
                self._client = client
            else:
                from hindsight_client import Hindsight
                kwargs = {"base_url": self._api_url, "timeout": 30.0}
                if self._api_key:
                    kwargs["api_key"] = self._api_key
                self._client = Hindsight(**kwargs)
        return self._client

    def initialize(self, session_id: str, **kwargs) -> None:
        self._config = _load_config()
        self._mode = self._config.get("mode", "cloud")
        self._api_key = self._config.get("apiKey") or os.environ.get("HINDSIGHT_API_KEY", "")
        default_url = _DEFAULT_LOCAL_URL if self._mode == "local" else _DEFAULT_API_URL
        self._api_url = self._config.get("api_url") or os.environ.get("HINDSIGHT_API_URL", default_url)

        banks = self._config.get("banks", {}).get("hermes", {})
        self._bank_id = banks.get("bankId", "hermes")
        budget = banks.get("budget", "mid")
        self._budget = budget if budget in _VALID_BUDGETS else "mid"

        recall_mode = self._config.get("recall_mode", "hybrid")
        self._recall_mode = recall_mode if recall_mode in ("context", "tools", "hybrid") else "hybrid"

        logger.info("Hindsight initialized: mode=%s, api_url=%s, bank=%s, budget=%s, recall_mode=%s",
                     self._mode, self._api_url, self._bank_id, self._budget, self._recall_mode)

        # For local mode, start the embedded daemon in the background so it
        # doesn't block the chat. Redirect stdout/stderr to a log file to
        # prevent rich startup output from spamming the terminal.
        if self._mode == "local":
            def _start_daemon():
                import sys, traceback
                from pathlib import Path
                log_dir = Path(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))) / "logs"
                log_dir.mkdir(parents=True, exist_ok=True)
                log_file = open(log_dir / "hindsight-embed.log", "a")
                log_fd = log_file.fileno()
                # Redirect at both Python and OS level so subprocess output
                # (rich UI, uv, etc.) also goes to the log file.
                old_stdout_fd = os.dup(1)
                old_stderr_fd = os.dup(2)
                os.dup2(log_fd, 1)
                os.dup2(log_fd, 2)
                old_stdout, old_stderr = sys.stdout, sys.stderr
                sys.stdout, sys.stderr = log_file, log_file
                try:
                    self._get_client()._ensure_started()
                    log_file.write("\n=== Daemon started successfully ===\n")
                except Exception as e:
                    log_file.write(f"\n=== Daemon startup failed: {e} ===\n")
                    traceback.print_exc(file=log_file)
                finally:
                    log_file.flush()
                    sys.stdout, sys.stderr = old_stdout, old_stderr
                    os.dup2(old_stdout_fd, 1)
                    os.dup2(old_stderr_fd, 2)
                    os.close(old_stdout_fd)
                    os.close(old_stderr_fd)
                    log_file.close()

            t = threading.Thread(target=_start_daemon, daemon=True, name="hindsight-daemon-start")
            t.start()

    def system_prompt_block(self) -> str:
        if self._recall_mode == "context":
            return (
                f"# Hindsight Memory\n"
                f"Active (context mode). Bank: {self._bank_id}, budget: {self._budget}.\n"
                f"Relevant memories are automatically injected into context."
            )
        if self._recall_mode == "tools":
            return (
                f"# Hindsight Memory\n"
                f"Active (tools mode). Bank: {self._bank_id}, budget: {self._budget}.\n"
                f"Use hindsight_recall to search, hindsight_reflect for synthesis, "
                f"hindsight_retain to store facts."
            )
        return (
            f"# Hindsight Memory\n"
            f"Active. Bank: {self._bank_id}, budget: {self._budget}.\n"
            f"Relevant memories are automatically injected into context. "
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
        if self._recall_mode == "tools":
            return
        def _run():
            try:
                client = self._get_client()
                resp = _run_sync(client.arecall(bank_id=self._bank_id, query=query, budget=self._budget))
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
                client = self._get_client()
                _run_sync(client.aretain(
                    bank_id=self._bank_id, content=combined, context="conversation"
                ))
            except Exception as e:
                logger.warning("Hindsight sync failed: %s", e)

        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5.0)
        self._sync_thread = threading.Thread(target=_sync, daemon=True, name="hindsight-sync")
        self._sync_thread.start()

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        if self._recall_mode == "context":
            return []
        return [RETAIN_SCHEMA, RECALL_SCHEMA, REFLECT_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        try:
            client = self._get_client()
        except Exception as e:
            logger.warning("Hindsight client init failed: %s", e)
            return json.dumps({"error": f"Hindsight client unavailable: {e}"})

        if tool_name == "hindsight_retain":
            content = args.get("content", "")
            if not content:
                return json.dumps({"error": "Missing required parameter: content"})
            context = args.get("context")
            try:
                _run_sync(client.aretain(
                    bank_id=self._bank_id, content=content, context=context
                ))
                return json.dumps({"result": "Memory stored successfully."})
            except Exception as e:
                logger.warning("hindsight_retain failed: %s", e)
                return json.dumps({"error": f"Failed to store memory: {e}"})

        elif tool_name == "hindsight_recall":
            query = args.get("query", "")
            if not query:
                return json.dumps({"error": "Missing required parameter: query"})
            try:
                resp = _run_sync(client.arecall(
                    bank_id=self._bank_id, query=query, budget=self._budget
                ))
                if not resp.results:
                    return json.dumps({"result": "No relevant memories found."})
                lines = [f"{i}. {r.text}" for i, r in enumerate(resp.results, 1)]
                return json.dumps({"result": "\n".join(lines)})
            except Exception as e:
                logger.warning("hindsight_recall failed: %s", e)
                return json.dumps({"error": f"Failed to search memory: {e}"})

        elif tool_name == "hindsight_reflect":
            query = args.get("query", "")
            if not query:
                return json.dumps({"error": "Missing required parameter: query"})
            try:
                resp = _run_sync(client.areflect(
                    bank_id=self._bank_id, query=query, budget=self._budget
                ))
                return json.dumps({"result": resp.text or "No relevant memories found."})
            except Exception as e:
                logger.warning("hindsight_reflect failed: %s", e)
                return json.dumps({"error": f"Failed to reflect: {e}"})

        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    def shutdown(self) -> None:
        global _loop, _loop_thread
        for t in (self._prefetch_thread, self._sync_thread):
            if t and t.is_alive():
                t.join(timeout=5.0)
        if self._client is not None:
            try:
                if self._mode == "local":
                    # Mark as closed so __del__ becomes a no-op, then close
                    # the inner HTTP client on our event loop to avoid
                    # the "attached to a different loop" error.
                    self._client._closed = True
                    inner = getattr(self._client, "_client", None)
                    if inner is not None:
                        _run_sync(inner.aclose())
                        _run_sync(asyncio.sleep(0.25))
                else:
                    _run_sync(self._client.aclose())
                    _run_sync(asyncio.sleep(0.25))
            except Exception:
                pass
            self._client = None
        # Stop the background event loop so no tasks are pending at exit
        if _loop is not None and _loop.is_running():
            _loop.call_soon_threadsafe(_loop.stop)
            if _loop_thread is not None:
                _loop_thread.join(timeout=5.0)
            _loop = None
            _loop_thread = None


def register(ctx) -> None:
    """Register Hindsight as a memory provider plugin."""
    ctx.register_memory_provider(HindsightMemoryProvider())

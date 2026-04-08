"""PowerMem memory plugin — MemoryProvider interface.

Hybrid vector + full-text (+ optional graph) memory with LLM-based extraction,
deduplication, and Ebbinghaus-style retrieval weighting.

Config:
  - PowerMem settings live in $HERMES_HOME/.env (the plugin points
    POWERMEM_ENV_FILE there when that file exists).
  - Optional overrides: $HERMES_HOME/powermem.json (user_id, agent_id, enabled).

Environment (optional shortcuts, read by _load_config before JSON):
  POWERMEM_USER_ID   — default user id for CLI sessions (default: hermes-user)
  POWERMEM_AGENT_ID  — agent scope (default: hermes)
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)

_BREAKER_THRESHOLD = 5
_BREAKER_COOLDOWN_SECS = 120


def _load_config() -> dict:
    """Load Hermes-side powermem overrides from env + $HERMES_HOME/powermem.json."""
    from hermes_constants import get_hermes_home

    config: Dict[str, Any] = {"enabled": True}
    path = get_hermes_home() / "powermem.json"
    if path.exists():
        try:
            file_cfg = json.loads(path.read_text(encoding="utf-8"))
            for k, v in file_cfg.items():
                if v is not None and v != "":
                    config[k] = v
        except Exception:
            pass

    if "user_id" not in config:
        config["user_id"] = os.environ.get(
            "POWERMEM_USER_ID", os.environ.get("HERMES_POWERMEM_USER_ID", "hermes-user")
        )
    env_agent = os.environ.get("POWERMEM_AGENT_ID")
    if "agent_id" not in config and env_agent:
        config["agent_id"] = env_agent

    return config


def _powermem_installed() -> bool:
    return importlib.util.find_spec("powermem") is not None


PROFILE_SCHEMA = {
    "name": "powermem_profile",
    "description": (
        "List stored memories for this user (and agent scope). "
        "Use at conversation start for a quick snapshot; prefer powermem_search for targeted recall."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "limit": {
                "type": "integer",
                "description": "Max memories to return (default: 50, max: 200).",
            },
        },
        "required": [],
    },
}

SEARCH_SCHEMA = {
    "name": "powermem_search",
    "description": (
        "Search PowerMem by meaning (hybrid vector + text when configured). "
        "Returns ranked memory snippets with scores."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to look up."},
            "limit": {
                "type": "integer",
                "description": "Max results (default: 10, max: 50).",
            },
            "threshold": {
                "type": "number",
                "description": "Optional minimum quality/similarity threshold (0–1).",
            },
        },
        "required": ["query"],
    },
}

ADD_SCHEMA = {
    "name": "powermem_add",
    "description": (
        "Add an explicit memory (verbatim, no LLM extraction), matching PowerMem's Memory.add "
        "with infer=false. Use for corrections, preferences, or decisions the user states clearly."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The fact to store."},
        },
        "required": ["content"],
    },
}


class PowermemMemoryProvider(MemoryProvider):
    """PowerMem SDK-backed memory for Hermes."""

    def __init__(self) -> None:
        self._memory = None  # type: Optional[Any]
        self._mem_lock = threading.RLock()
        self._cfg: dict = {}
        self._user_id = "hermes-user"
        self._agent_id = "hermes"
        self._prefetch_result = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread: Optional[threading.Thread] = None
        self._sync_thread: Optional[threading.Thread] = None
        self._consecutive_failures = 0
        self._breaker_open_until = 0.0

    @property
    def name(self) -> str:
        return "powermem"

    def is_available(self) -> bool:
        """True when the powermem package is importable (no network I/O)."""
        return _powermem_installed()

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        path = Path(hermes_home) / "powermem.json"
        existing: Dict[str, Any] = {}
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                pass
        existing.update(values)
        existing.setdefault("enabled", True)
        path.write_text(json.dumps(existing, indent=2), encoding="utf-8")

    def get_config_schema(self):
        return [
            {
                "key": "user_id",
                "description": "Default user id for CLI sessions (gateway overrides with platform user)",
                "default": "hermes-user",
                "env_var": "POWERMEM_USER_ID",
            },
            {
                "key": "agent_id",
                "description": "Agent / profile scope for stored memories",
                "default": "hermes",
                "env_var": "POWERMEM_AGENT_ID",
            },
        ]

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
            self._breaker_open_until = time.monotonic() + _BREAKER_COOLDOWN_SECS
            logger.warning(
                "PowerMem circuit breaker: %d consecutive failures, pausing %ds.",
                self._consecutive_failures,
                _BREAKER_COOLDOWN_SECS,
            )

    def _apply_hermes_env_file(self, hermes_home: str) -> None:
        env_path = Path(hermes_home).expanduser() / ".env"
        if env_path.is_file():
            os.environ["POWERMEM_ENV_FILE"] = str(env_path.resolve())

    def _ensure_memory(self) -> Any:
        if self._memory is not None:
            return self._memory
        from powermem import create_memory

        self._memory = create_memory()
        return self._memory

    def initialize(self, session_id: str, **kwargs) -> None:
        self._cfg = _load_config()
        if self._cfg.get("enabled") is False:
            logger.info("PowerMem disabled via powermem.json (enabled: false)")
            self._memory = None
            return

        hermes_home = kwargs.get("hermes_home") or ""
        if hermes_home:
            self._apply_hermes_env_file(str(hermes_home))

        self._user_id = kwargs.get("user_id") or self._cfg.get("user_id", "hermes-user")
        # Explicit powermem.json / POWERMEM_AGENT_ID wins; else Hermes profile name.
        self._agent_id = (
            self._cfg.get("agent_id")
            or kwargs.get("agent_identity")
            or "hermes"
        )

        if not _powermem_installed():
            logger.warning("powermem package not installed")
            self._memory = None
            return

        try:
            self._ensure_memory()
        except Exception as e:
            logger.warning("PowerMem initialize failed: %s", e)
            self._memory = None

    def system_prompt_block(self) -> str:
        if not self._memory:
            return ""
        return (
            "# PowerMem\n"
            f"Active. user_id={self._user_id} agent_id={self._agent_id}.\n"
            "Use powermem_search for recall, powermem_add for explicit facts, "
            "powermem_profile to list recent memories."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=3.0)
        with self._prefetch_lock:
            result = self._prefetch_result
            self._prefetch_result = ""
        if not result:
            return ""
        return f"## PowerMem\n{result}"

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        if not self._memory or self._is_breaker_open() or not query.strip():
            return

        uid = self._user_id
        aid = self._agent_id

        def _run() -> None:
            try:
                mem = self._ensure_memory()
                with self._mem_lock:
                    out = mem.search(
                        query=query,
                        user_id=uid,
                        agent_id=aid,
                        limit=5,
                    )
                rows = out.get("results", []) if isinstance(out, dict) else []
                lines = []
                for r in rows:
                    text = r.get("memory") or r.get("content") or ""
                    if text:
                        sc = r.get("score", 0)
                        lines.append(f"- [{sc:.3f}] {text}")
                if lines:
                    with self._prefetch_lock:
                        self._prefetch_result = "\n".join(lines)
                self._record_success()
            except Exception as e:
                self._record_failure()
                logger.debug("PowerMem prefetch failed: %s", e)

        self._prefetch_thread = threading.Thread(
            target=_run, daemon=True, name="powermem-prefetch"
        )
        self._prefetch_thread.start()

    def sync_turn(
        self, user_content: str, assistant_content: str, *, session_id: str = ""
    ) -> None:
        if not self._memory or self._is_breaker_open():
            return

        uid = self._user_id
        aid = self._agent_id
        messages = [
            {"role": "user", "content": (user_content or "")[:8000]},
            {"role": "assistant", "content": (assistant_content or "")[:8000]},
        ]

        def _sync() -> None:
            try:
                mem = self._ensure_memory()
                with self._mem_lock:
                    mem.add(messages, user_id=uid, agent_id=aid, infer=True)
                self._record_success()
            except Exception as e:
                self._record_failure()
                logger.warning("PowerMem sync_turn failed: %s", e)

        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5.0)

        self._sync_thread = threading.Thread(
            target=_sync, daemon=True, name="powermem-sync"
        )
        self._sync_thread.start()

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        if action != "add" or not content or not self._memory:
            return

        uid = self._user_id
        aid = self._agent_id
        text = f"[Built-in {target}] {content}"

        def _run() -> None:
            try:
                mem = self._ensure_memory()
                with self._mem_lock:
                    mem.add(text, user_id=uid, agent_id=aid, infer=False)
            except Exception as e:
                logger.debug("PowerMem mirror builtin memory failed: %s", e)

        threading.Thread(target=_run, daemon=True, name="powermem-memwrite").start()

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [PROFILE_SCHEMA, SEARCH_SCHEMA, ADD_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        if self._is_breaker_open():
            return json.dumps(
                {
                    "error": (
                        "PowerMem temporarily paused after repeated errors; "
                        "will retry automatically."
                    )
                }
            )

        if not self._memory:
            return tool_error(
                "PowerMem not initialized — install powermem, configure $HERMES_HOME/.env, "
                "and check logs."
            )

        if tool_name == "powermem_profile":
            return self._tool_profile(args)
        if tool_name == "powermem_search":
            return self._tool_search(args)
        if tool_name == "powermem_add":
            return self._tool_add(args)
        return tool_error(f"Unknown tool: {tool_name}")

    def _tool_profile(self, args: dict) -> str:
        limit = min(int(args.get("limit", 50)), 200)
        try:
            mem = self._ensure_memory()
            with self._mem_lock:
                out = mem.get_all(
                    user_id=self._user_id,
                    agent_id=self._agent_id,
                    limit=limit,
                    order="desc",
                )
            self._record_success()
            rows = out.get("results", []) if isinstance(out, dict) else []
            if not rows:
                return json.dumps({"result": "No memories stored yet.", "count": 0})
            lines = []
            for r in rows:
                text = r.get("memory") or r.get("content") or ""
                if text:
                    lines.append(text)
            return json.dumps(
                {"result": "\n".join(f"- {l}" for l in lines), "count": len(lines)}
            )
        except Exception as e:
            self._record_failure()
            return tool_error(f"powermem_profile failed: {e}")

    def _tool_search(self, args: dict) -> str:
        query = (args.get("query") or "").strip()
        if not query:
            return tool_error("Missing required parameter: query")
        limit = min(int(args.get("limit", 10)), 50)
        threshold = args.get("threshold")
        th: Optional[float] = None
        if threshold is not None and threshold != "":
            try:
                th = float(threshold)
            except (TypeError, ValueError):
                th = None
        try:
            mem = self._ensure_memory()
            with self._mem_lock:
                out = mem.search(
                    query=query,
                    user_id=self._user_id,
                    agent_id=self._agent_id,
                    limit=limit,
                    threshold=th,
                )
            self._record_success()
            rows = out.get("results", []) if isinstance(out, dict) else []
            if not rows:
                return json.dumps({"results": [], "message": "No relevant memories found."})
            items = []
            for r in rows:
                text = r.get("memory") or r.get("content") or ""
                items.append(
                    {
                        "memory": text,
                        "score": r.get("score", 0),
                        "id": r.get("id") or r.get("memory_id"),
                    }
                )
            return json.dumps({"results": items, "count": len(items)})
        except Exception as e:
            self._record_failure()
            return tool_error(f"powermem_search failed: {e}")

    def _tool_add(self, args: dict) -> str:
        content = (args.get("content") or "").strip()
        if not content:
            return tool_error("Missing required parameter: content")
        try:
            mem = self._ensure_memory()
            with self._mem_lock:
                mem.add(
                    content,
                    user_id=self._user_id,
                    agent_id=self._agent_id,
                    infer=False,
                )
            self._record_success()
            return json.dumps({"result": "Stored.", "ok": True})
        except Exception as e:
            self._record_failure()
            return tool_error(f"powermem_add failed: {e}")

    def shutdown(self) -> None:
        for t in (self._prefetch_thread, self._sync_thread):
            if t and t.is_alive():
                t.join(timeout=5.0)
        with self._mem_lock:
            self._memory = None


def register(ctx) -> None:
    ctx.register_memory_provider(PowermemMemoryProvider())

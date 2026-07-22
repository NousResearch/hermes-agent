"""Mengram memory plugin — MemoryProvider interface.

Three memory types with server-side extraction: semantic facts, episodic
events, and procedural workflows that evolve from successes and failures
(failed runs record the violated assumption and derive preconditions that
ride along with recall). Cloud API or self-hosted (Apache 2.0).

Configuration
-------------
Secret (lives in $HERMES_HOME/.env or the environment):
  MENGRAM_API_KEY    — Mengram API key (free tier at https://mengram.io)

Behavioral settings (live in $HERMES_HOME/mengram.json, set via
`hermes memory setup`):
  base_url           — API base URL (default https://mengram.io; point at a
                       self-hosted deployment to keep data on your infra)
  user_id            — Canonical user identifier. When set, applied across
                       every gateway so the same human gets one merged memory.
                       When unset, the gateway-native id is used.
  sync_every_turns   — Turns to buffer before a background write (default 3).
                       Each flush costs one `add` against the Mengram plan
                       quota; buffering keeps free-tier usage sane.

The matching MENGRAM_BASE_URL / MENGRAM_USER_ID environment variables are
read as a fallback; mengram.json is the canonical home for non-secret
settings.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)

_BREAKER_THRESHOLD = 5
_BREAKER_COOLDOWN_SECS = 120
_PREFETCH_WAIT_SECS = 3
_HTTP_TIMEOUT_SECS = 10
_SYNC_EVERY_TURNS_DEFAULT = 3
_SYNC_MAX_BUFFER_CHARS = 12000
_TURN_SNIPPET_CHARS = 1500

# "default" — NOT a hermes-specific id — so CLI memory unifies with the
# user's existing Mengram memory from Claude Code, Cursor, and MCP (the
# cross-tool store is the point). Gateway-native ids (Telegram, Discord)
# still override via kwargs so multi-person gateways stay isolated.
_DEFAULT_USER_ID = "default"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    """Env vars provide defaults; $HERMES_HOME/mengram.json overrides."""
    from hermes_constants import get_hermes_home

    config: Dict[str, Any] = {
        "api_key": os.environ.get("MENGRAM_API_KEY", ""),
        "base_url": os.environ.get("MENGRAM_BASE_URL", "https://mengram.io"),
        "sync_every_turns": _SYNC_EVERY_TURNS_DEFAULT,
    }
    env_user_id = os.environ.get("MENGRAM_USER_ID")
    if env_user_id:
        config["user_id"] = env_user_id

    config_path = get_hermes_home() / "mengram.json"
    if config_path.exists():
        try:
            file_cfg = json.loads(config_path.read_text(encoding="utf-8"))
            config.update({k: v for k, v in file_cfg.items()
                           if v is not None and v != ""})
        except Exception:
            pass
    return config


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

SEARCH_SCHEMA = {
    "name": "mengram_search",
    "description": (
        "Search the user's long-term memory across all three types at once: "
        "semantic facts (preferences, people, projects), episodic events "
        "(what happened, decisions, outcomes), and procedural workflows "
        "(how the user does things). Use before answering anything that may "
        "depend on the user's history; vary the wording across calls for "
        "multi-part questions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
            "limit": {"type": "integer", "description": "Max results per memory type (default 5, max 20)."},
        },
        "required": ["query"],
    },
}

REMEMBER_SCHEMA = {
    "name": "mengram_remember",
    "description": (
        "Store lasting knowledge about the user. Mengram runs server-side "
        "extraction: facts, events, and workflows are pulled out, deduplicated "
        "against existing memory, and contradictions are resolved. Call when "
        "the user states a durable preference, decision, correction, or "
        "completes something worth recalling later. Skip transient chit-chat."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "What to remember, in natural language with full context."},
        },
        "required": ["content"],
    },
}

PROCEDURES_SCHEMA = {
    "name": "mengram_procedures",
    "description": (
        "Recall the user's learned workflows (procedural memory) relevant to a "
        "task: step sequences with version history, success/failure counts, and "
        "preconditions derived from past failures. Check BEFORE performing a "
        "multi-step task the user has likely done before — following a proven "
        "v3 beats re-deriving a v1. Respect any listed preconditions: they "
        "exist because an assumption once failed."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The task, e.g. 'deploy backend' or 'release npm package'."},
            "limit": {"type": "integer", "description": "Max workflows (default 3)."},
        },
        "required": ["query"],
    },
}


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

class MengramMemoryProvider(MemoryProvider):
    """Mengram memory: semantic + episodic + procedural with evolution."""

    def __init__(self):
        self._config: Optional[dict] = None
        self._api_key = ""
        self._base_url = "https://mengram.io"
        self._user_id = _DEFAULT_USER_ID
        self._channel = "cli"
        self._session_id = ""
        self._writes_enabled = True
        self._sync_every = _SYNC_EVERY_TURNS_DEFAULT
        # Turn buffer (quota-conscious: one add per N turns, not per turn)
        self._turn_buffer: List[str] = []
        self._turns_since_flush = 0
        self._buffer_lock = threading.Lock()
        # Prefetch state
        self._prefetch_thread: Optional[threading.Thread] = None
        self._prefetch_query = ""
        self._prefetch_result = ""
        self._prefetch_done = threading.Event()
        self._prefetch_lock = threading.Lock()
        # Profile cache for system_prompt_block
        self._profile_block = ""
        # Circuit breaker
        self._consecutive_failures = 0
        self._breaker_open_until = 0.0
        self._breaker_lock = threading.Lock()
        self._atexit_registered = False

    @property
    def name(self) -> str:
        return "mengram"

    # -- availability / config ------------------------------------------------

    def is_available(self) -> bool:
        return bool(_load_config().get("api_key"))

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {"key": "api_key", "description": "Mengram API key (free tier available)",
             "secret": True, "required": True, "env_var": "MENGRAM_API_KEY",
             "url": "https://mengram.io"},
            {"key": "base_url", "description": "API base URL (set for self-hosted; blank = cloud)",
             "required": False, "default": "https://mengram.io"},
            {"key": "user_id", "description": "Canonical user identifier (blank = per-gateway ids)",
             "required": False, "default": _DEFAULT_USER_ID},
            {"key": "sync_every_turns", "description": "Turns to buffer per background write (quota control)",
             "required": False, "default": str(_SYNC_EVERY_TURNS_DEFAULT)},
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        from pathlib import Path
        from utils import atomic_json_write
        config_path = Path(hermes_home) / "mengram.json"
        existing: Dict[str, Any] = {}
        if config_path.exists():
            try:
                existing = json.loads(config_path.read_text())
            except Exception:
                pass
        existing.update(values)
        atomic_json_write(config_path, existing, mode=0o600)

    # -- HTTP core -------------------------------------------------------------

    def _request(self, method: str, path: str, body: Optional[dict] = None) -> Optional[dict]:
        """Single API call with circuit breaker. Returns parsed JSON or None."""
        if self._is_breaker_open():
            return None
        url = f"{self._base_url.rstrip('/')}{path}"
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(
            url, data=data, method=method,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "User-Agent": "hermes-mengram-plugin/1.0.0",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT_SECS) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            self._record_success()
            return result
        except urllib.error.HTTPError as e:
            # 4xx are client/user errors (quota, bad input) — don't trip breaker
            if 400 <= e.code < 500:
                self._record_success()
                try:
                    return json.loads(e.read().decode("utf-8"))
                except Exception:
                    return {"error": f"HTTP {e.code}"}
            self._record_failure()
            logger.warning("Mengram API %s %s failed: HTTP %s", method, path, e.code)
            return None
        except Exception as e:
            self._record_failure()
            logger.warning("Mengram API %s %s failed: %s", method, path, e)
            return None

    def _is_breaker_open(self) -> bool:
        with self._breaker_lock:
            if self._consecutive_failures < _BREAKER_THRESHOLD:
                return False
            if time.monotonic() >= self._breaker_open_until:
                self._consecutive_failures = 0
                return False
            return True

    def _record_success(self) -> None:
        with self._breaker_lock:
            self._consecutive_failures = 0

    def _record_failure(self) -> None:
        with self._breaker_lock:
            self._consecutive_failures += 1
            tripped = self._consecutive_failures == _BREAKER_THRESHOLD
            if tripped:
                self._breaker_open_until = time.monotonic() + _BREAKER_COOLDOWN_SECS
        if tripped:
            logger.warning(
                "Mengram circuit breaker tripped after %d consecutive failures; "
                "pausing API calls for %ds.", _BREAKER_THRESHOLD, _BREAKER_COOLDOWN_SECS)

    # -- lifecycle ---------------------------------------------------------------

    def initialize(self, session_id: str, **kwargs) -> None:
        self._config = _load_config()
        self._api_key = self._config.get("api_key", "")
        self._base_url = self._config.get("base_url", "https://mengram.io")
        self._session_id = session_id
        configured = self._config.get("user_id")
        if configured == _DEFAULT_USER_ID:
            configured = None
        self._user_id = configured or kwargs.get("user_id") or _DEFAULT_USER_ID
        self._channel = kwargs.get("platform") or "cli"
        try:
            self._sync_every = max(1, int(self._config.get("sync_every_turns",
                                                           _SYNC_EVERY_TURNS_DEFAULT)))
        except (TypeError, ValueError):
            self._sync_every = _SYNC_EVERY_TURNS_DEFAULT
        # Cron/subagent contexts must not corrupt the user's memory with
        # system-prompt-shaped writes.
        self._writes_enabled = kwargs.get("agent_context", "primary") == "primary"

        # Warm the cognitive profile in the background so system_prompt_block
        # (called during prompt assembly) never blocks on the network.
        threading.Thread(target=self._warm_profile, daemon=True).start()

        if not self._atexit_registered:
            atexit.register(self.shutdown)
            self._atexit_registered = True

    def _warm_profile(self) -> None:
        result = self._request("GET", f"/v1/profile?sub_user_id={self._user_id}")
        prompt = (result or {}).get("system_prompt") or ""
        if prompt:
            self._profile_block = prompt[:2000]

    def system_prompt_block(self) -> str:
        block = (
            "# Mengram Memory\n"
            f"Active ({'cloud' if self._base_url.rstrip('/') == 'https://mengram.io' else 'self-hosted'}). "
            "Three memory types: semantic facts, episodic events, procedural "
            "workflows with failure-driven evolution. Use mengram_search before "
            "user-specific questions and mengram_procedures before multi-step "
            "tasks the user has done before.\n"
        )
        if self._profile_block:
            block += f"\n## What you remember about this user\n{self._profile_block}\n"
        return block

    # -- prefetch (background recall) ---------------------------------------------

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        if not query or not query.strip():
            return
        with self._prefetch_lock:
            self._prefetch_query = query.strip()[:500]
            self._prefetch_result = ""
            self._prefetch_done.clear()
            self._prefetch_thread = threading.Thread(
                target=self._do_prefetch, args=(self._prefetch_query,), daemon=True)
            self._prefetch_thread.start()

    def _do_prefetch(self, query: str) -> None:
        result = self._request("POST", "/v1/search/all", {
            "query": query, "limit": 5, "user_id": self._user_id,
        })
        text = self._format_recall(result) if result else ""
        with self._prefetch_lock:
            if query == self._prefetch_query:
                self._prefetch_result = text
                self._prefetch_done.set()

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        # Serve the backgrounded result if one is cooking; wait briefly.
        if self._prefetch_thread is not None:
            self._prefetch_done.wait(timeout=_PREFETCH_WAIT_SECS)
            with self._prefetch_lock:
                if self._prefetch_result:
                    out, self._prefetch_result = self._prefetch_result, ""
                    return out
        return ""

    @staticmethod
    def _format_recall(result: Dict[str, Any]) -> str:
        lines: List[str] = []
        for ent in (result.get("semantic") or [])[:5]:
            name = ent.get("entity", "?")
            for fact in (ent.get("facts") or [])[:3]:
                lines.append(f"- [fact] {name}: {fact}")
        for ep in (result.get("episodic") or [])[:3]:
            lines.append(f"- [event] {ep.get('summary', '')}")
        for proc in (result.get("procedural") or [])[:2]:
            steps = proc.get("steps") or []
            pre = (proc.get("metadata") or {}).get("preconditions") or []
            entry = f"- [workflow] {proc.get('name', '?')} (v{proc.get('version', 1)}, {len(steps)} steps)"
            if pre:
                entry += f" — verify first: {'; '.join(str(p) for p in pre[:2])}"
            lines.append(entry)
        if not lines:
            return ""
        return "[Mengram memory — relevant context]\n" + "\n".join(lines)

    # -- capture -------------------------------------------------------------------

    def sync_turn(self, user_content: str, assistant_content: str, *,
                  session_id: str = "", messages=None) -> None:
        if not self._writes_enabled:
            return
        snippet = ""
        if user_content and user_content.strip():
            snippet += f"User: {user_content.strip()[:_TURN_SNIPPET_CHARS]}\n"
        if assistant_content and assistant_content.strip():
            snippet += f"Assistant: {assistant_content.strip()[:_TURN_SNIPPET_CHARS]}\n"
        if not snippet:
            return
        flush_now = False
        with self._buffer_lock:
            self._turn_buffer.append(snippet)
            self._turns_since_flush += 1
            if (self._turns_since_flush >= self._sync_every
                    or sum(len(s) for s in self._turn_buffer) >= _SYNC_MAX_BUFFER_CHARS):
                flush_now = True
        if flush_now:
            threading.Thread(target=self._flush_buffer, daemon=True).start()

    def _flush_buffer(self) -> None:
        with self._buffer_lock:
            if not self._turn_buffer:
                return
            text = "\n".join(self._turn_buffer)
            self._turn_buffer = []
            self._turns_since_flush = 0
        self._request("POST", "/v1/add_text", {
            "text": text,
            "source": "hermes_agent",
            "run_id": self._session_id,
            "user_id": self._user_id,
            "metadata": {"channel": self._channel},
        })

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        """Flush buffered turns BEFORE compression discards them — memory
        keeps what the compaction summary loses."""
        if self._writes_enabled:
            self._flush_buffer()
        return ""

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        if self._writes_enabled:
            self._flush_buffer()

    def on_session_switch(self, new_session_id: str, *, parent_session_id: str = "",
                          reset: bool = False, rewound: bool = False, **kwargs) -> None:
        if reset and self._writes_enabled:
            self._flush_buffer()
        self._session_id = new_session_id

    def on_delegation(self, task: str, result: str, *,
                      child_session_id: str = "", **kwargs) -> None:
        if not self._writes_enabled:
            return
        with self._buffer_lock:
            self._turn_buffer.append(
                f"Delegated task: {task.strip()[:_TURN_SNIPPET_CHARS]}\n"
                f"Result: {result.strip()[:_TURN_SNIPPET_CHARS]}\n")
            self._turns_since_flush += 1

    def shutdown(self) -> None:
        try:
            if self._writes_enabled:
                self._flush_buffer()
        except Exception:
            pass

    # -- tools -----------------------------------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [SEARCH_SCHEMA, REMEMBER_SCHEMA, PROCEDURES_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if tool_name == "mengram_search":
            query = (args.get("query") or "").strip()
            if not query:
                return tool_error("query is required")
            limit = min(int(args.get("limit") or 5), 20)
            result = self._request("POST", "/v1/search/all", {
                "query": query, "limit": limit, "user_id": self._user_id,
            })
            if result is None:
                return tool_error("Mengram API unavailable (circuit breaker or network)")
            if result.get("error"):
                return tool_error(str(result.get("message") or result["error"]))
            text = self._format_recall(result)
            return json.dumps({"results": text or "No relevant memories found.",
                               "hint": result.get("hint", "")})

        if tool_name == "mengram_remember":
            content = (args.get("content") or "").strip()
            if not content:
                return tool_error("content is required")
            if not self._writes_enabled:
                return tool_error("writes are disabled in this agent context")
            result = self._request("POST", "/v1/add_text", {
                "text": content, "source": "hermes_agent",
                "run_id": self._session_id, "user_id": self._user_id,
            })
            if result is None:
                return tool_error("Mengram API unavailable (circuit breaker or network)")
            if result.get("error"):
                return tool_error(str(result.get("message") or result["error"]))
            return json.dumps({"status": "queued",
                               "detail": "Extraction runs server-side; facts appear within ~a minute."})

        if tool_name == "mengram_procedures":
            query = (args.get("query") or "").strip()
            if not query:
                return tool_error("query is required")
            limit = min(int(args.get("limit") or 3), 10)
            result = self._request(
                "GET",
                f"/v1/procedures/search?query={urllib.parse.quote(query)}"
                f"&limit={limit}&sub_user_id={urllib.parse.quote(self._user_id)}")
            if result is None:
                return tool_error("Mengram API unavailable (circuit breaker or network)")
            procs = result.get("results") or []
            if not procs:
                return json.dumps({"results": "No learned workflows match this task yet."})
            out = []
            for p in procs:
                steps = [f"{s.get('step', i + 1)}. {s.get('action', '')}"
                         for i, s in enumerate(p.get("steps") or [])]
                out.append({
                    "name": p.get("name"),
                    "version": p.get("version", 1),
                    "success_count": p.get("success_count", 0),
                    "fail_count": p.get("fail_count", 0),
                    "preconditions": (p.get("metadata") or {}).get("preconditions") or [],
                    "steps": steps,
                })
            return json.dumps({"workflows": out})

        raise NotImplementedError(f"Provider mengram does not handle tool {tool_name}")

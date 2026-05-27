"""Atlas memory plugin — MemoryProvider interface.

Backs Hermes long-term memory with Atlas, Blake's RDF-grounded personal
knowledge substrate. Recall is sourced from Atlas's /v1/memory/hermes/read
endpoint (active facts with confidence + life-context); writes go to
/v1/memory/hermes/write (classified, provenance-tracked RDF triples).

This is the Hermes-side adapter for army-of-one Plan 011-C.2. The transport
contract was defined by Plan 012 (memory_routes.py in army-of-one). Unlike
mem0/supermemory (server-side turn extraction), Atlas writes are EXPLICIT
facts: the agent decides what's worth remembering and stores it verbatim via
the atlas_remember tool, or the built-in memory tool mirrors writes through
the on_memory_write hook.

Design (mirrors mem0 provider patterns):
  - Non-blocking prefetch via background thread + cache
  - Circuit breaker: pause API calls after consecutive failures
  - Graceful degradation: every Atlas failure is swallowed; MemoryManager
    guarantees a failing provider never blocks the agent or the built-in
    provider.

Config via environment variables:
  ATLAS_BASE_URL       — Atlas API base URL (required; e.g.
                         http://atlas.agentic-stack.internal:8000 in cloud,
                         http://localhost:8000 locally)
  ATLAS_BEARER_TOKEN   — Bearer for LAN/VPC auth (optional for localhost)
  ATLAS_AGENT_NAME     — Agent identifier for fact attribution (default: hermes)
  ATLAS_MAX_AGE_DAYS   — Exclude facts older than N days (default: 90)

Or via $HERMES_HOME/atlas.json.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Dict, List

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)

# Circuit breaker: after this many consecutive failures, pause API calls
# for _BREAKER_COOLDOWN_SECS to avoid hammering a down Atlas.
_BREAKER_THRESHOLD = 5
_BREAKER_COOLDOWN_SECS = 120

# Conservative timeouts — Atlas read is fast (in-process SPARQL) but the
# network hop (Cloud Map within the VPC) adds latency. Match Plan 012 spec.
_READ_TIMEOUT_SECS = 2.0
_WRITE_TIMEOUT_SECS = 3.0


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    """Load config from env vars, with $HERMES_HOME/atlas.json overrides."""
    from hermes_constants import get_hermes_home

    config = {
        "base_url": os.environ.get("ATLAS_BASE_URL", ""),
        "token": os.environ.get("ATLAS_BEARER_TOKEN", ""),
        "agent_name": os.environ.get("ATLAS_AGENT_NAME", "hermes"),
        "max_age_days": int(os.environ.get("ATLAS_MAX_AGE_DAYS", "90")),
    }

    config_path = get_hermes_home() / "atlas.json"
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

RECALL_SCHEMA = {
    "name": "atlas_recall",
    "description": (
        "Retrieve stored facts about Blake from Atlas — preferences, people, "
        "projects, decisions, and context drawn from his unified knowledge "
        "substrate. Returns active facts ranked by recency + confidence. "
        "Use at conversation start or when you need durable context."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}

REMEMBER_SCHEMA = {
    "name": "atlas_remember",
    "description": (
        "Store a durable fact in Atlas. Stored verbatim with provenance and "
        "confidence bootstrapping. Use for explicit preferences, corrections, "
        "relationships, or decisions worth recalling across sessions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The fact to store."},
            "target": {
                "type": "string",
                "enum": ["user", "memory"],
                "description": "'user' for facts about Blake; 'memory' for agent self-notes. Default 'user'.",
            },
            "life_context": {
                "type": "string",
                "enum": ["work", "personal", "health", "education",
                         "finance", "civic", "hobby", "brand", "spiritual"],
                "description": "Optional life-domain tag.",
            },
        },
        "required": ["content"],
    },
}


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

class AtlasMemoryProvider(MemoryProvider):
    """Atlas RDF-grounded long-term memory provider."""

    def __init__(self):
        self._config = None
        self._base_url = ""
        self._token = ""
        self._agent_name = "hermes"
        self._max_age_days = 90
        self._session_id = ""
        self._prefetch_result = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread = None
        self._sync_thread = None
        # Circuit breaker state
        self._consecutive_failures = 0
        self._breaker_open_until = 0.0

    @property
    def name(self) -> str:
        return "atlas"

    def is_available(self) -> bool:
        # No network call here — just config presence (per ABC contract).
        cfg = _load_config()
        return bool(cfg.get("base_url"))

    def save_config(self, values, hermes_home):
        """Write non-secret config to $HERMES_HOME/atlas.json."""
        from pathlib import Path
        config_path = Path(hermes_home) / "atlas.json"
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
            {"key": "base_url", "description": "Atlas API base URL", "required": True,
             "default": "http://localhost:8000", "env_var": "ATLAS_BASE_URL"},
            {"key": "token", "description": "Atlas bearer token (required for non-localhost)",
             "secret": True, "env_var": "ATLAS_BEARER_TOKEN"},
            {"key": "agent_name", "description": "Agent identifier for fact attribution",
             "default": "hermes"},
            {"key": "max_age_days", "description": "Exclude facts older than N days",
             "default": "90"},
        ]

    # -- HTTP helpers --------------------------------------------------------

    def _headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self._token:
            h["Authorization"] = f"Bearer {self._token}"
        return h

    def _is_breaker_open(self) -> bool:
        if self._consecutive_failures < _BREAKER_THRESHOLD:
            return False
        if time.monotonic() >= self._breaker_open_until:
            self._consecutive_failures = 0
            return False
        return True

    def _record_success(self):
        self._consecutive_failures = 0

    def _record_failure(self):
        self._consecutive_failures += 1
        if self._consecutive_failures >= _BREAKER_THRESHOLD:
            self._breaker_open_until = time.monotonic() + _BREAKER_COOLDOWN_SECS
            logger.warning(
                "Atlas circuit breaker tripped after %d consecutive failures. "
                "Pausing API calls for %ds.",
                self._consecutive_failures, _BREAKER_COOLDOWN_SECS,
            )

    def _fetch_facts(self) -> list:
        """GET /v1/memory/hermes/read — returns list of fact dicts. Raises on error."""
        import httpx
        url = f"{self._base_url.rstrip('/')}/v1/memory/hermes/read"
        resp = httpx.get(
            url,
            params={"agent": self._agent_name, "max_age_days": self._max_age_days},
            headers=self._headers(),
            timeout=_READ_TIMEOUT_SECS,
        )
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, list) else []

    @staticmethod
    def _format_facts(facts: list) -> str:
        """Render Atlas facts as a compact bullet list for prompt injection."""
        lines = []
        for f in facts:
            key = f.get("key", "")
            value = f.get("value", "")
            if not value:
                continue
            ctx = f.get("life_context")
            tag = f" [{ctx}]" if ctx else ""
            if key and key != value:
                lines.append(f"- {key}: {value}{tag}")
            else:
                lines.append(f"- {value}{tag}")
        return "\n".join(lines)

    # -- Lifecycle -----------------------------------------------------------

    def initialize(self, session_id: str, **kwargs) -> None:
        self._config = _load_config()
        self._base_url = self._config.get("base_url", "")
        self._token = self._config.get("token", "")
        self._agent_name = self._config.get("agent_name", "hermes")
        self._max_age_days = int(self._config.get("max_age_days", 90))
        self._session_id = session_id

    def system_prompt_block(self) -> str:
        return (
            "# Atlas Memory\n"
            "Active — RDF-grounded long-term memory across Blake's life domains.\n"
            "Use atlas_recall to fetch stored facts, atlas_remember to store a "
            "durable fact worth recalling across sessions."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=_READ_TIMEOUT_SECS + 0.5)
        with self._prefetch_lock:
            result = self._prefetch_result
            self._prefetch_result = ""
        if not result:
            return ""
        return f"## Atlas Memory\n{result}"

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        if self._is_breaker_open():
            return

        def _run():
            try:
                facts = self._fetch_facts()
                if facts:
                    formatted = self._format_facts(facts)
                    with self._prefetch_lock:
                        self._prefetch_result = formatted
                self._record_success()
            except Exception as e:
                self._record_failure()
                logger.debug("Atlas prefetch failed: %s", e)

        self._prefetch_thread = threading.Thread(target=_run, daemon=True, name="atlas-prefetch")
        self._prefetch_thread.start()

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """No-op: Atlas does NOT do turn-level fact extraction. Facts are written
        explicitly via atlas_remember or mirrored from built-in memory via
        on_memory_write. This avoids polluting the RDF store with raw turns."""
        return

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [RECALL_SCHEMA, REMEMBER_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        if self._is_breaker_open():
            return json.dumps({
                "error": "Atlas temporarily unavailable (consecutive failures). Will retry automatically."
            })

        if tool_name == "atlas_recall":
            try:
                facts = self._fetch_facts()
                self._record_success()
                if not facts:
                    return json.dumps({"result": "No facts stored in Atlas yet."})
                return json.dumps({"result": self._format_facts(facts), "count": len(facts)})
            except Exception as e:
                self._record_failure()
                return tool_error(f"Atlas recall failed: {e}")

        elif tool_name == "atlas_remember":
            content = args.get("content", "")
            if not content:
                return tool_error("Missing required parameter: content")
            target = args.get("target", "user")
            life_context = args.get("life_context")
            try:
                self._write_fact(
                    target=target, action="add", content=content,
                    life_context=life_context,
                )
                self._record_success()
                return json.dumps({"result": "Fact stored in Atlas."})
            except Exception as e:
                self._record_failure()
                return tool_error(f"Atlas write failed: {e}")

        return tool_error(f"Unknown tool: {tool_name}")

    def _write_fact(self, *, target: str, action: str, content: str,
                    old_text: str | None = None, life_context: str | None = None) -> None:
        """POST /v1/memory/hermes/write. Raises on error."""
        import httpx
        url = f"{self._base_url.rstrip('/')}/v1/memory/hermes/write"
        body = {
            "target": target,
            "action": action,
            "content": content,
            "agent": self._agent_name,
            "session_id": self._session_id or None,
        }
        if old_text:
            body["old_text"] = old_text
        if life_context:
            body["life_context"] = life_context
        resp = httpx.post(url, json=body, headers=self._headers(), timeout=_WRITE_TIMEOUT_SECS)
        resp.raise_for_status()

    def on_memory_write(self, action: str, target: str, content: str, metadata=None) -> None:
        """Mirror built-in memory writes into Atlas (non-blocking, best-effort).

        When Hermes's built-in memory tool writes a fact, echo it to Atlas so
        the RDF store stays in sync with the flat memory.md. Atlas-side
        failures are swallowed — the built-in write already succeeded.
        """
        if self._is_breaker_open():
            return
        # Atlas targets are 'user' | 'memory'; map unknown targets to 'memory'.
        atlas_target = target if target in ("user", "memory") else "memory"
        # Atlas actions are 'add' | 'replace' | 'remove'.
        atlas_action = action if action in ("add", "replace", "remove") else "add"
        old_text = (metadata or {}).get("old_text") if metadata else None
        if atlas_action in ("replace", "remove") and not old_text:
            # Can't satisfy Atlas's contract without old_text — downgrade to add.
            atlas_action = "add"

        def _mirror():
            try:
                self._write_fact(
                    target=atlas_target, action=atlas_action,
                    content=content, old_text=old_text,
                )
                self._record_success()
            except Exception as e:
                self._record_failure()
                logger.debug("Atlas memory-write mirror failed: %s", e)

        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=_WRITE_TIMEOUT_SECS + 1.0)
        self._sync_thread = threading.Thread(target=_mirror, daemon=True, name="atlas-mirror")
        self._sync_thread.start()

    def shutdown(self) -> None:
        for t in (self._prefetch_thread, self._sync_thread):
            if t and t.is_alive():
                t.join(timeout=_WRITE_TIMEOUT_SECS + 1.0)


def register(ctx) -> None:
    """Register Atlas as a memory provider plugin."""
    ctx.register_memory_provider(AtlasMemoryProvider())

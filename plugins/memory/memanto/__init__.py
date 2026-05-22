"""Memanto memory agent plugin using the MemoryProvider interface.

Memanto (https://memanto.ai) is a *memory agent*: typed long-term memory with
confidence + provenance, semantic recall, and RAG-style answers, backed by the
Moorcheh vector platform. Unlike a passive "memory layer", every namespace in
Memanto is a first-class **agent** (``memanto agent create/activate``), so this
provider maps one Hermes identity to one Memanto agent.

What it gives Hermes:
  * Auto-recall — relevant memories injected before each turn (prefetch).
  * Turn capture — conversation turns persisted as ``event`` memories.
  * Explicit tools — ``memanto_remember`` / ``memanto_recall`` / ``memanto_answer``.
  * Built-in memory mirroring — Hermes ``memory`` writes echoed into Memanto.

The Memanto SDK (``memanto``) is imported lazily so this module loads even when
the package isn't installed; ``is_available()`` then returns False and the
provider stays inert.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)

# -- Defaults -----------------------------------------------------------------

_DEFAULT_AGENT_ID = "hermes-{identity}"
_DEFAULT_PATTERN = "tool"
_VALID_PATTERNS = ("support", "project", "tool")
_DEFAULT_MAX_RECALL_RESULTS = 10
_DEFAULT_CONFIDENCE = 0.85
_CAPTURE_CONFIDENCE = 0.6
_MIN_CAPTURE_LENGTH = 10
_MAX_TITLE_LENGTH = 100
_MAX_AGENT_ID_LENGTH = 64

# Memory taxonomy mirrored from memanto.app.constants.VALID_MEMORY_TYPES so the
# tool schema matches what the backend validates. Kept as a literal list to
# avoid importing memanto at module-import time.
_VALID_MEMORY_TYPES = (
    "fact", "preference", "goal", "decision", "artifact", "learning", "event",
    "instruction", "relationship", "context", "observation", "commitment", "error",
)

_TRIVIAL_RE = re.compile(
    r"^(ok|okay|thanks|thank you|got it|sure|yes|no|yep|nope|k|ty|thx|np)\.?$",
    re.IGNORECASE,
)
_CONTEXT_STRIP_RE = re.compile(
    r"<memanto-memory>[\s\S]*?</memanto-memory>\s*", re.DOTALL
)


def _default_config() -> dict:
    return {
        "agent_id": _DEFAULT_AGENT_ID,
        "pattern": _DEFAULT_PATTERN,
        "auto_recall": True,
        "auto_capture": True,
        "auto_create": True,
        "mirror_memory_writes": True,
        "max_recall_results": _DEFAULT_MAX_RECALL_RESULTS,
        "min_confidence": None,
        "session_duration_hours": None,
    }


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y", "on"}:
            return True
        if lowered in {"false", "0", "no", "n", "off"}:
            return False
    return default


def _sanitize_agent_id(raw: str) -> str:
    """Coerce to Memanto's id charset (letters, digits, ``-``, ``_``)."""
    cleaned = re.sub(r"[^a-zA-Z0-9_-]", "-", raw or "")
    cleaned = re.sub(r"-+", "-", cleaned).strip("-")
    cleaned = cleaned[:_MAX_AGENT_ID_LENGTH].strip("-")
    return cleaned or "hermes"


def _detect_memory_type(text: str) -> str:
    lowered = text.lower()
    if re.search(r"prefer|like|love|hate|want", lowered):
        return "preference"
    if re.search(r"decided|will use|going with|chose", lowered):
        return "decision"
    if re.search(r"\bis\b|\bare\b|\bhas\b|\bhave\b", lowered):
        return "fact"
    return "observation"


def _load_memanto_config(hermes_home: str) -> dict:
    config = _default_config()
    config_path = Path(hermes_home) / "memanto.json"
    if config_path.exists():
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                config.update({k: v for k, v in raw.items() if v is not None})
        except Exception:
            logger.debug("Failed to parse %s", config_path, exc_info=True)

    # agent_id is kept raw here — the {identity} template is resolved (and the
    # result sanitized) in initialize() once agent_identity is known.
    config["agent_id"] = str(config.get("agent_id") or _DEFAULT_AGENT_ID).strip() or _DEFAULT_AGENT_ID
    pattern = str(config.get("pattern") or _DEFAULT_PATTERN).strip().lower()
    config["pattern"] = pattern if pattern in _VALID_PATTERNS else _DEFAULT_PATTERN
    config["auto_recall"] = _as_bool(config.get("auto_recall"), True)
    config["auto_capture"] = _as_bool(config.get("auto_capture"), True)
    config["auto_create"] = _as_bool(config.get("auto_create"), True)
    config["mirror_memory_writes"] = _as_bool(config.get("mirror_memory_writes"), True)
    try:
        config["max_recall_results"] = max(1, min(100, int(config.get("max_recall_results", _DEFAULT_MAX_RECALL_RESULTS))))
    except Exception:
        config["max_recall_results"] = _DEFAULT_MAX_RECALL_RESULTS
    if config.get("min_confidence") is not None:
        try:
            config["min_confidence"] = max(0.0, min(1.0, float(config["min_confidence"])))
        except Exception:
            config["min_confidence"] = None
    if config.get("session_duration_hours") is not None:
        try:
            config["session_duration_hours"] = max(1, min(24 * 30, int(config["session_duration_hours"])))
        except Exception:
            config["session_duration_hours"] = None
    return config


def _save_memanto_config(values: dict, hermes_home: str) -> None:
    config_path = Path(hermes_home) / "memanto.json"
    existing: dict = {}
    if config_path.exists():
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                existing = raw
        except Exception:
            existing = {}
    existing.update(values)
    config_path.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _clean_text_for_capture(text: str) -> str:
    return _CONTEXT_STRIP_RE.sub("", text or "").strip()


def _is_trivial_message(text: str) -> bool:
    return bool(_TRIVIAL_RE.match((text or "").strip()))


def _memory_score(raw: dict) -> Optional[float]:
    """Coalesce the score key Memanto uses across endpoints."""
    score = raw.get("score")
    if score is None:
        score = raw.get("similarity_score")
    try:
        return float(score) if score is not None else None
    except (TypeError, ValueError):
        return None


def _format_recall_block(memories: List[dict], max_results: int) -> str:
    """Render recall hits into a context block for prefetch injection."""
    lines = []
    for item in (memories or [])[:max_results]:
        content = (item.get("content") or item.get("title") or "").strip()
        if not content:
            continue
        prefix_bits = []
        mem_type = item.get("type")
        if mem_type:
            prefix_bits.append(f"[{mem_type}]")
        score = _memory_score(item)
        if score is not None:
            prefix_bits.append(f"[{round(score * 100)}%]")
        prefix = " ".join(prefix_bits)
        lines.append(f"- {prefix} {content}".strip())
    if not lines:
        return ""
    intro = (
        "Background from your Memanto memory agent. Use it silently when relevant; "
        "do not force memories into the conversation."
    )
    body = "## Relevant Memories\n" + "\n".join(lines)
    return f"<memanto-memory>\n{intro}\n\n{body}\n</memanto-memory>"


class _MemantoClient:
    """Thin wrapper over Memanto's ``SdkClient`` pinned to one agent.

    Mirrors the MCP integration's lifecycle: ensure the agent exists (creating
    it on first use when ``auto_create``) and activate a session once, lazily,
    guarded by a lock. A permanent-for-session failure flag stops us from
    re-hitting the network on every turn when the backend is unreachable.
    """

    def __init__(self, api_key: str, agent_id: str, *, pattern: str = "tool",
                 auto_create: bool = True, session_duration_hours: Optional[int] = None):
        from memanto.cli.client.sdk_client import SdkClient

        self._agent_id = agent_id
        self._pattern = pattern
        self._auto_create = auto_create
        self._session_duration_hours = session_duration_hours
        self._client = SdkClient(api_key=api_key)
        self._lock = threading.Lock()
        self._ready = False
        self._failed = False

    @property
    def agent_id(self) -> str:
        return self._agent_id

    def ensure_session(self) -> None:
        if self._ready:
            return
        with self._lock:
            if self._ready:
                return
            if self._failed:
                raise RuntimeError("Memanto session activation previously failed this session")
            from memanto.app.utils.errors import (
                AgentAlreadyExistsError,
                AgentNotFoundError,
            )
            try:
                try:
                    self._client.get_agent(self._agent_id)
                except AgentNotFoundError:
                    if not self._auto_create:
                        raise
                    try:
                        self._client.create_agent(
                            self._agent_id,
                            pattern=self._pattern,
                            description="Created by Hermes",
                        )
                    except AgentAlreadyExistsError:
                        pass  # race: another caller created it
                self._client.activate_agent(
                    self._agent_id,
                    duration_hours=self._session_duration_hours,
                )
                self._ready = True
            except Exception:
                self._failed = True
                raise

    def remember(self, *, memory_type: str, title: str, content: str,
                 confidence: float, tags: Optional[List[str]] = None,
                 source: str = "hermes", provenance: str = "explicit_statement") -> dict:
        self.ensure_session()
        return self._client.remember(
            agent_id=self._agent_id,
            memory_type=memory_type,
            title=title,
            content=content,
            confidence=confidence,
            tags=tags or [],
            source=source,
            provenance=provenance,
        )

    def recall(self, query: str, *, limit: int, type: Optional[List[str]] = None,
               min_confidence: Optional[float] = None) -> List[dict]:
        self.ensure_session()
        result = self._client.recall(
            agent_id=self._agent_id,
            query=query,
            limit=limit,
            type=type,
            min_confidence=min_confidence,
        )
        return result.get("memories", [])

    def answer(self, question: str, *, limit: Optional[int] = None) -> dict:
        self.ensure_session()
        return self._client.answer(
            agent_id=self._agent_id,
            question=question,
            limit=limit,
        )


# -- Tool schemas -------------------------------------------------------------

REMEMBER_SCHEMA = {
    "name": "memanto_remember",
    "description": "Store a durable fact, preference, decision, goal, or instruction in the Memanto memory agent for future recall.",
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The memory itself — one atomic, self-contained statement."},
            "type": {
                "type": "string",
                "enum": list(_VALID_MEMORY_TYPES),
                "description": "Semantic memory type. Use 'preference' for likes/styles, 'fact' for stable claims, 'decision' for choices, 'goal' for objectives, 'instruction' for directives.",
            },
            "tags": {"type": "array", "items": {"type": "string"}, "description": "Optional lowercase tags for later filtering."},
            "confidence": {"type": "number", "description": "How sure you are this is true, 0.0-1.0. Use 1.0 only for explicit statements."},
        },
        "required": ["content"],
    },
}

RECALL_SCHEMA = {
    "name": "memanto_recall",
    "description": "Search the Memanto memory agent by semantic similarity. Use this FIRST before asking the user to repeat information.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Natural-language search query."},
            "limit": {"type": "integer", "description": "Maximum results to return, 1 to 100."},
            "type": {
                "type": "array",
                "items": {"type": "string", "enum": list(_VALID_MEMORY_TYPES)},
                "description": "Optional type filter, e.g. ['preference'].",
            },
        },
        "required": ["query"],
    },
}

ANSWER_SCHEMA = {
    "name": "memanto_answer",
    "description": "Ask a natural-language question and get an answer grounded ONLY in the Memanto memory agent's stored memories (RAG). Prefer over memanto_recall when you need a synthesized answer rather than a ranked list.",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "The question to answer from memory."},
            "limit": {"type": "integer", "description": "Number of context memories to retrieve, 1 to 100."},
        },
        "required": ["question"],
    },
}


class MemantoMemoryProvider(MemoryProvider):
    """Memanto-backed memory provider for Hermes."""

    def __init__(self):
        self._config = _default_config()
        self._api_key = ""
        self._client: Optional[_MemantoClient] = None
        self._agent_id = "hermes"
        self._hermes_home = ""
        self._auto_recall = True
        self._auto_capture = True
        self._mirror_memory_writes = True
        self._max_recall_results = _DEFAULT_MAX_RECALL_RESULTS
        self._min_confidence: Optional[float] = None
        self._write_enabled = True
        self._active = False
        self._warmup_thread: Optional[threading.Thread] = None
        self._sync_thread: Optional[threading.Thread] = None
        self._write_thread: Optional[threading.Thread] = None

    @property
    def name(self) -> str:
        return "memanto"

    def is_available(self) -> bool:
        if not os.environ.get("MOORCHEH_API_KEY", "").strip():
            return False
        try:
            __import__("memanto")
            return True
        except Exception:
            return False

    def get_config_schema(self):
        return [
            {
                "key": "api_key",
                "description": "Moorcheh API key (powers Memanto)",
                "secret": True,
                "required": True,
                "env_var": "MOORCHEH_API_KEY",
                "url": "https://console.moorcheh.ai/api-keys",
            },
            {
                "key": "agent_id",
                "description": "Memory agent id / namespace ({identity} expands to the profile name)",
                "secret": False,
                "required": False,
                "default": _DEFAULT_AGENT_ID,
            },
        ]

    def save_config(self, values, hermes_home):
        sanitized = dict(values or {})
        # Keep the {identity} template intact; only sanitize concrete ids.
        if "agent_id" in sanitized and "{identity}" not in str(sanitized["agent_id"]):
            sanitized["agent_id"] = _sanitize_agent_id(str(sanitized["agent_id"]))
        if "pattern" in sanitized:
            pattern = str(sanitized["pattern"]).strip().lower()
            sanitized["pattern"] = pattern if pattern in _VALID_PATTERNS else _DEFAULT_PATTERN
        _save_memanto_config(sanitized, hermes_home)

    def initialize(self, session_id: str, **kwargs) -> None:
        from hermes_constants import get_hermes_home

        self._hermes_home = kwargs.get("hermes_home") or str(get_hermes_home())
        self._config = _load_memanto_config(self._hermes_home)
        self._api_key = os.environ.get("MOORCHEH_API_KEY", "").strip()

        # Resolve the agent id: env override > config, with {identity} template.
        identity = kwargs.get("agent_identity", "default") or "default"
        raw_id = os.environ.get("MEMANTO_AGENT_ID", "").strip() or self._config["agent_id"]
        self._agent_id = _sanitize_agent_id(raw_id.replace("{identity}", identity))

        self._auto_recall = self._config["auto_recall"]
        self._auto_capture = self._config["auto_capture"]
        self._mirror_memory_writes = self._config["mirror_memory_writes"]
        self._max_recall_results = self._config["max_recall_results"]
        self._min_confidence = self._config["min_confidence"]

        agent_context = kwargs.get("agent_context", "")
        self._write_enabled = agent_context not in {"cron", "flush", "subagent"}

        self._active = False
        self._client = None
        if not self._api_key:
            return
        try:
            self._client = _MemantoClient(
                self._api_key,
                self._agent_id,
                pattern=self._config["pattern"],
                auto_create=self._config["auto_create"],
                session_duration_hours=self._config["session_duration_hours"],
            )
            self._active = True
        except Exception:
            logger.warning("Memanto initialization failed", exc_info=True)
            self._client = None
            self._active = False
            return

        # Warm up the session in the background so the first turn's recall does
        # not also pay agent-create + activate latency.
        self._warmup_thread = threading.Thread(
            target=self._warmup, daemon=True, name="memanto-warmup"
        )
        self._warmup_thread.start()

    def _warmup(self) -> None:
        try:
            self._client.ensure_session()
        except Exception:
            logger.debug("Memanto warmup activation failed", exc_info=True)

    def system_prompt_block(self) -> str:
        if not self._active:
            return ""
        return (
            "# Memanto Memory Agent\n"
            f"Active. Memory agent: {self._agent_id}.\n"
            "Use memanto_recall to look up stored memories, memanto_remember to save durable "
            "facts/preferences/decisions/goals, and memanto_answer for a synthesized answer "
            "grounded in memory."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not self._active or not self._auto_recall or not self._client or not query.strip():
            return ""
        try:
            memories = self._client.recall(
                query[:2000],
                limit=self._max_recall_results,
                min_confidence=self._min_confidence,
            )
            return _format_recall_block(memories, self._max_recall_results)
        except Exception:
            logger.debug("Memanto prefetch failed", exc_info=True)
            return ""

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        if not self._active or not self._auto_capture or not self._write_enabled or not self._client:
            return
        clean_user = _clean_text_for_capture(user_content)
        clean_assistant = _clean_text_for_capture(assistant_content)
        if len(clean_user) < _MIN_CAPTURE_LENGTH or len(clean_assistant) < _MIN_CAPTURE_LENGTH:
            return
        if _is_trivial_message(clean_user):
            return

        title = clean_user[:_MAX_TITLE_LENGTH - 3] + "..." if len(clean_user) > _MAX_TITLE_LENGTH else clean_user
        content = f"User: {clean_user}\n\nAssistant: {clean_assistant}"

        def _run():
            try:
                self._client.remember(
                    memory_type="event",
                    title=title,
                    content=content,
                    confidence=_CAPTURE_CONFIDENCE,
                    tags=["conversation"],
                    source="hermes",
                    provenance="observed",
                )
            except Exception:
                logger.debug("Memanto sync_turn failed", exc_info=True)

        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=2.0)
        self._sync_thread = threading.Thread(target=_run, daemon=True, name="memanto-sync")
        self._sync_thread.start()

    def on_memory_write(self, action: str, target: str, content: str,
                        metadata: Optional[Dict[str, Any]] = None) -> None:
        if not self._active or not self._write_enabled or not self._mirror_memory_writes or not self._client:
            return
        if action != "add" or not (content or "").strip():
            return
        clean = content.strip()
        title = clean[:_MAX_TITLE_LENGTH - 3] + "..." if len(clean) > _MAX_TITLE_LENGTH else clean
        mem_type = "preference" if target == "user" else _detect_memory_type(clean)

        def _run():
            try:
                self._client.remember(
                    memory_type=mem_type,
                    title=title,
                    content=clean,
                    confidence=0.9,
                    tags=["hermes-memory", target],
                    source="hermes-memory",
                    provenance="explicit_statement",
                )
            except Exception:
                logger.debug("Memanto on_memory_write failed", exc_info=True)

        if self._write_thread and self._write_thread.is_alive():
            self._write_thread.join(timeout=2.0)
        self._write_thread = threading.Thread(target=_run, daemon=False, name="memanto-memory-write")
        self._write_thread.start()

    def shutdown(self) -> None:
        for attr_name in ("_warmup_thread", "_sync_thread", "_write_thread"):
            thread = getattr(self, attr_name, None)
            if thread and thread.is_alive():
                thread.join(timeout=5.0)
            setattr(self, attr_name, None)

    # -- Tools ----------------------------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [REMEMBER_SCHEMA, RECALL_SCHEMA, ANSWER_SCHEMA]

    def _tool_remember(self, args: dict) -> str:
        content = str(args.get("content") or "").strip()
        if not content:
            return tool_error("content is required")
        mem_type = str(args.get("type") or "").strip()
        if mem_type not in _VALID_MEMORY_TYPES:
            mem_type = _detect_memory_type(content)
        tags = args.get("tags") or []
        if not isinstance(tags, list):
            tags = []
        try:
            confidence = float(args.get("confidence", _DEFAULT_CONFIDENCE))
        except (TypeError, ValueError):
            confidence = _DEFAULT_CONFIDENCE
        confidence = max(0.0, min(1.0, confidence))
        title = content[:_MAX_TITLE_LENGTH - 3] + "..." if len(content) > _MAX_TITLE_LENGTH else content
        try:
            result = self._client.remember(
                memory_type=mem_type,
                title=title,
                content=content,
                confidence=confidence,
                tags=[str(t) for t in tags],
                source="hermes-tool",
                provenance="explicit_statement",
            )
            return json.dumps({
                "saved": True,
                "memory_id": result.get("memory_id"),
                "agent_id": self._agent_id,
                "type": mem_type,
            })
        except Exception as exc:
            return tool_error(f"Failed to store memory: {exc}")

    def _tool_recall(self, args: dict) -> str:
        query = str(args.get("query") or "").strip()
        if not query:
            return tool_error("query is required")
        try:
            limit = max(1, min(100, int(args.get("limit", self._max_recall_results) or self._max_recall_results)))
        except (TypeError, ValueError):
            limit = self._max_recall_results
        type_filter = args.get("type")
        if type_filter is not None and not isinstance(type_filter, list):
            type_filter = [type_filter]
        try:
            memories = self._client.recall(
                query, limit=limit, type=type_filter, min_confidence=self._min_confidence
            )
            formatted = []
            for item in memories:
                entry: Dict[str, Any] = {
                    "id": item.get("id"),
                    "type": item.get("type"),
                    "content": item.get("content") or item.get("title", ""),
                }
                score = _memory_score(item)
                if score is not None:
                    entry["score"] = round(score * 100)
                formatted.append(entry)
            return json.dumps({"results": formatted, "count": len(formatted)})
        except Exception as exc:
            return tool_error(f"Recall failed: {exc}")

    def _tool_answer(self, args: dict) -> str:
        question = str(args.get("question") or "").strip()
        if not question:
            return tool_error("question is required")
        limit = args.get("limit")
        if limit is not None:
            try:
                limit = max(1, min(100, int(limit)))
            except (TypeError, ValueError):
                limit = None
        try:
            result = self._client.answer(question, limit=limit)
            sources = result.get("sources", []) or []
            return json.dumps({
                "answer": result.get("answer", ""),
                "sources": sources,
                "count": len(sources),
            })
        except Exception as exc:
            return tool_error(f"Answer failed: {exc}")

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if not self._active or not self._client:
            return tool_error("Memanto is not configured")
        if tool_name == "memanto_remember":
            return self._tool_remember(args)
        if tool_name == "memanto_recall":
            return self._tool_recall(args)
        if tool_name == "memanto_answer":
            return self._tool_answer(args)
        return tool_error(f"Unknown tool: {tool_name}")


def register(ctx):
    ctx.register_memory_provider(MemantoMemoryProvider())

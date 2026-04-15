"""Nowledge Mem memory provider for Hermes Agent.

Implements the MemoryProvider ABC to give Hermes native access to the
user's cross-tool knowledge graph. Replaces the generic MCP connection
with lifecycle hooks: automatic Working Memory injection, per-turn
recall, user-profile mirroring, and native tool names.

Transport: ``nmem`` CLI only. The CLI handles server URL, API key, and
remote access. If ``nmem`` is not installed, the provider disables
gracefully.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider

try:
    from tools.registry import tool_error as _registry_tool_error
except ImportError:
    _registry_tool_error = None

try:
    from tools.registry import tool_result as _registry_tool_result
except ImportError:
    _registry_tool_result = None

from .client import NowledgeMemClient

logger = logging.getLogger(__name__)

GUIDANCE = """\
# Nowledge Mem
Cross-tool personal knowledge graph. Knowledge saved here persists across \
all tools and sessions.

**Nowledge Mem vs Hermes memory:** Hermes memory stores Hermes-specific \
facts (env details, tool quirks). Nowledge Mem stores cross-tool knowledge: \
decisions, procedures, and learnings. Ask yourself: "Would this matter in a \
Claude Code, Cursor, or Codex session?" If yes, save to Nowledge Mem.

Save proactively when the conversation produces a decision, procedure, \
learning, or important context. One strong memory is better than three \
weak ones.

**Upsert by ID:** pass a stable id to nmem_save to create-or-update in one \
call. If a memory with that ID already exists it is updated; otherwise a new \
one is created. Use this when you have a natural key (for example a topic or \
decision name) instead of the search-then-update dance.

Without an id, search first (nmem_search); if existing knowledge covers the \
topic, use nmem_update rather than create a duplicate."""

_SEARCH = {
    "name": "nmem_search",
    "description": "Search stored memories. Supports label filtering and deep search mode.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query."},
            "limit": {"type": "integer", "description": "Max results, 1-20 (default 10)."},
            "filter_labels": {
                "type": "string",
                "description": "Comma-separated labels to filter by.",
            },
            "mode": {
                "type": "string",
                "description": "'normal' (fast, default) or 'deep' (graph-enhanced).",
            },
        },
        "required": ["query"],
    },
}

_SAVE = {
    "name": "nmem_save",
    "description": (
        "Save a decision, insight, procedure, or learning. Pass id to upsert: "
        "update if that ID exists, create if not. Without id, search first to avoid duplicates."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "Memory content."},
            "id": {
                "type": "string",
                "description": "Stable ID for upsert. Omit to auto-generate.",
            },
            "title": {"type": "string", "description": "Descriptive title."},
            "importance": {
                "type": "number",
                "description": "0.0-1.0. 0.8+ major, 0.5-0.7 useful, 0.3-0.4 minor.",
            },
            "labels": {
                "type": "string",
                "description": "Comma-separated labels.",
            },
            "unit_type": {
                "type": "string",
                "description": "fact, preference, decision, plan, procedure, learning, context, or event.",
            },
            "event_date": {
                "type": "string",
                "description": "When it happened (YYYY, YYYY-MM, or YYYY-MM-DD).",
            },
        },
        "required": ["content"],
    },
}

_UPDATE = {
    "name": "nmem_update",
    "description": "Refine an existing memory. Search first to find the memory_id.",
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {"type": "string", "description": "Memory ID from search results."},
            "content": {"type": "string", "description": "Updated content (omit to keep current)."},
            "title": {"type": "string", "description": "Updated title (omit to keep current)."},
            "importance": {"type": "number", "description": "Updated importance (omit to keep current)."},
        },
        "required": ["memory_id"],
    },
}

_DELETE = {
    "name": "nmem_delete",
    "description": "Delete one or more memories. Also removes associated relationships and entity links.",
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {"type": "string", "description": "Single memory ID to delete."},
            "memory_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Multiple memory IDs for bulk delete.",
            },
        },
        "required": [],
    },
}

_THREAD_SEARCH = {
    "name": "nmem_thread_search",
    "description": "Search past conversations. Returns threads with matched message snippets.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query."},
            "limit": {"type": "integer", "description": "Max threads, 1-50 (default 10)."},
            "source": {
                "type": "string",
                "description": "Filter by source (for example 'claude-code' or 'hermes').",
            },
        },
        "required": ["query"],
    },
}

_THREAD_MESSAGES = {
    "name": "nmem_thread_messages",
    "description": "Fetch messages from a thread. Use after nmem_thread_search.",
    "parameters": {
        "type": "object",
        "properties": {
            "thread_id": {"type": "string", "description": "Thread ID."},
            "offset": {"type": "integer", "description": "Messages to skip (default 0)."},
            "limit": {"type": "integer", "description": "Max messages, 1-100 (default 50)."},
        },
        "required": ["thread_id"],
    },
}

ALL_TOOL_SCHEMAS = [
    _SEARCH,
    _SAVE,
    _UPDATE,
    _DELETE,
    _THREAD_SEARCH,
    _THREAD_MESSAGES,
]


def tool_error(message: Any, **extra: Any) -> str:
    """Return Hermes-style JSON error payloads across old and new releases."""
    if _registry_tool_error is not None:
        return _registry_tool_error(message, **extra)

    payload = {"error": str(message)}
    if extra:
        payload.update(extra)
    return json.dumps(payload, ensure_ascii=False)


def tool_result(data: Any = None, **kwargs: Any) -> str:
    """Return Hermes-style JSON result payloads across old and new releases."""
    if _registry_tool_result is not None:
        return _registry_tool_result(data, **kwargs)

    if data is not None:
        if kwargs:
            raise ValueError("tool_result accepts either data or kwargs, not both")
        return json.dumps(data, ensure_ascii=False)
    return json.dumps(kwargs, ensure_ascii=False)


class NowledgeMemProvider(MemoryProvider):
    """Nowledge Mem as a native Hermes memory provider."""

    def __init__(self) -> None:
        self._client: Optional[NowledgeMemClient] = None
        self._working_memory = ""
        self._cron_skipped = False
        self._bg_threads: List[threading.Thread] = []
        self._resolved_space: str | None = None
        self._session_id = ""
        self._saved_message_count = 0

    @property
    def name(self) -> str:
        return "nowledge-mem"

    def is_available(self) -> bool:
        """Check if nmem CLI is installed. No network calls."""
        return NowledgeMemClient.is_available()

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        agent_context = kwargs.get("agent_context", "")
        platform = kwargs.get("platform", "cli")

        if agent_context in ("cron", "flush") or platform == "cron":
            logger.debug(
                "Nowledge Mem skipped: cron/flush context (agent_context=%s, platform=%s)",
                agent_context,
                platform,
            )
            self._cron_skipped = True
            return

        config = self._load_config(kwargs.get("hermes_home", ""))
        raw_timeout = config.get("timeout", 30)
        try:
            timeout = int(raw_timeout or 30)
        except (TypeError, ValueError):
            logger.warning(
                "Invalid timeout in nowledge-mem.json: %r; falling back to 30s",
                raw_timeout,
            )
            timeout = 30
        if timeout <= 0:
            timeout = 30
        self._resolved_space = self._resolve_space(config, kwargs)
        self._client = NowledgeMemClient(timeout=timeout, space=self._resolved_space)
        self._session_id = session_id or ""
        self._saved_message_count = 0

        if not self._client.health():
            logger.warning("Nowledge Mem not reachable via nmem CLI")
            self._client = None
            return

        try:
            wm = self._client.working_memory()
            self._working_memory = self._format_working_memory(wm)
        except Exception as error:
            logger.debug("Working memory fetch failed: %s", error)
            self._working_memory = ""

        logger.info(
            "Nowledge Mem provider initialized (CLI transport)",
            extra={"space": self._resolved_space or "default"},
        )

    def system_prompt_block(self) -> str:
        if self._cron_skipped:
            return ""

        parts = [GUIDANCE]
        if self._resolved_space:
            parts.append(f"## Active Space\n\n{self._resolved_space}")
        if self._working_memory:
            parts.append(f"## Working Memory\n\n{self._working_memory}")
        elif self._client:
            parts.append("## Working Memory\n\nNo briefing available yet. The knowledge graph is connected.")
        else:
            parts.append(
                "## Working Memory\n\nNowledge Mem server is not reachable. Tools are unavailable until the server is running."
            )
        return "\n\n".join(parts)

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if self._cron_skipped or not self._client:
            return ""
        if not query or len(query.strip()) < 10:
            return ""

        try:
            result = self._client.search(query.strip()[:200], limit=5)
            return self._format_prefetch(result)
        except Exception as error:
            logger.debug("Nowledge Mem prefetch failed: %s", error)
            return ""

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        if self._cron_skipped or not self._client:
            return
        if action != "add" or target != "user" or not content:
            return

        client = self._client

        def _mirror() -> None:
            try:
                client.save(
                    content,
                    title="User profile (synced from Hermes)",
                    labels=["hermes-profile"],
                    importance=0.5,
                )
                logger.debug("Mirrored user-profile write to Nowledge Mem")
            except Exception as error:
                logger.debug("Mirror write failed (non-fatal): %s", error)

        thread = threading.Thread(target=_mirror, daemon=True, name="nmem-mirror")
        thread.start()
        self._bg_threads.append(thread)

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        if self._cron_skipped or not self._client:
            return ""
        return (
            "Cross-session knowledge is stored in Nowledge Mem. Decisions and insights "
            "from compressed messages can be recovered via nmem_search. Preserve references "
            "to memory IDs if any were mentioned."
        )

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        if self._cron_skipped or not self._client:
            return
        session_id = (self._session_id or "").strip()
        if not session_id:
            return

        cleaned_messages = self._clean_session_messages(messages)
        if not cleaned_messages:
            return

        saved_count = int(self._saved_message_count or 0)
        if saved_count > len(cleaned_messages):
            saved_count = 0

        if saved_count <= 0:
            title = self._build_thread_title(cleaned_messages)
            try:
                self._client.import_thread(
                    session_id,
                    cleaned_messages,
                    title=title or None,
                    source="hermes",
                )
            except Exception as error:
                logger.warning(
                    "Nowledge Mem session import failed for %s: %s",
                    session_id,
                    error,
                )
                return
            self._saved_message_count = len(cleaned_messages)
            return

        delta = cleaned_messages[saved_count:]
        if not delta:
            return
        try:
            self._client.append_thread(session_id, delta)
        except Exception as error:
            logger.warning(
                "Nowledge Mem session append failed for %s: %s",
                session_id,
                error,
            )
            return
        self._saved_message_count = len(cleaned_messages)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        if self._cron_skipped:
            return []
        return list(ALL_TOOL_SCHEMAS)

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs: Any) -> str:
        if not self._client:
            return tool_error("Nowledge Mem is not connected.", success=False)

        try:
            result = self._dispatch(tool_name, args)
            return tool_result(self._normalize_tool_result(result))
        except Exception as error:
            logger.error("Tool %s failed: %s", tool_name, error)
            return tool_error(f"{tool_name} failed: {error}", success=False, tool=tool_name)

    def _dispatch(self, tool_name: str, args: Dict[str, Any]) -> Any:
        assert self._client is not None
        client = self._client

        if tool_name == "nmem_search":
            return client.search(
                args.get("query", ""),
                limit=args.get("limit", 10),
                filter_labels=self._parse_csv(args.get("filter_labels")),
                mode=args.get("mode"),
            )
        if tool_name == "nmem_save":
            return client.save(
                args["content"],
                memory_id=args.get("id"),
                title=args.get("title"),
                importance=args.get("importance"),
                labels=self._parse_csv(args.get("labels")),
                unit_type=args.get("unit_type"),
                event_date=args.get("event_date"),
            )
        if tool_name == "nmem_update":
            return client.update(
                args["memory_id"],
                content=args.get("content"),
                title=args.get("title"),
                importance=args.get("importance"),
            )
        if tool_name == "nmem_delete":
            memory_ids = self._normalize_id_list(args.get("memory_ids"))
            if memory_ids:
                return client.delete_many(memory_ids)
            if args.get("memory_id"):
                return client.delete(args["memory_id"])
            raise ValueError("nmem_delete requires memory_id or memory_ids")
        if tool_name == "nmem_thread_search":
            return client.thread_search(
                args.get("query", ""),
                limit=args.get("limit", 10),
                source=args.get("source"),
            )
        if tool_name == "nmem_thread_messages":
            return client.thread_messages(
                args["thread_id"],
                limit=args.get("limit", 50),
                offset=args.get("offset", 0),
            )
        raise ValueError(f"Unknown tool: {tool_name}")

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "timeout",
                "description": "Request timeout in seconds",
                "default": "30",
            },
            {
                "key": "space",
                "description": "Optional default space name for this Hermes provider",
                "default": "",
            },
            {
                "key": "space_by_identity",
                "description": 'Optional JSON object mapping Hermes identities to space names, for example {"research":"Research Agent"}',
                "default": "",
            },
            {
                "key": "space_template",
                "description": "Optional template like research-{identity}; used when space is empty",
                "default": "",
            }
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        config_path = Path(hermes_home) / "nowledge-mem.json"
        existing: dict = {}
        if config_path.exists():
            try:
                existing = json.loads(config_path.read_text())
            except Exception:
                pass
        existing.update(values)
        config_path.write_text(json.dumps(existing, indent=2))
        logger.info("Nowledge Mem config saved to %s", config_path)

    def shutdown(self) -> None:
        for thread in self._bg_threads:
            if thread.is_alive():
                thread.join(timeout=3.0)
        self._bg_threads.clear()
        self._resolved_space = None
        self._session_id = ""
        self._saved_message_count = 0

    @staticmethod
    def _parse_csv(value: Any) -> Optional[List[str]]:
        return NowledgeMemProvider._to_str_list(value)

    @staticmethod
    def _normalize_id_list(value: Any) -> Optional[List[str]]:
        return NowledgeMemProvider._to_str_list(value)

    @staticmethod
    def _to_str_list(value: Any) -> Optional[List[str]]:
        if value is None:
            return None
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        if isinstance(value, (list, tuple, set)):
            parsed = [str(item).strip() for item in value if str(item).strip()]
            return parsed or None

        text = str(value).strip()
        return [text] if text else None

    @staticmethod
    def _normalize_tool_result(result: Any) -> Any:
        if isinstance(result, dict):
            if "success" not in result and "error" not in result:
                return {"success": True, **result}
            return result
        return {"success": True, "result": result}

    @staticmethod
    def _load_config(hermes_home: str) -> Dict[str, Any]:
        if hermes_home:
            config_path = Path(hermes_home) / "nowledge-mem.json"
            if config_path.exists():
                try:
                    cfg = json.loads(config_path.read_text())
                    if isinstance(cfg, dict):
                        return cfg
                except Exception as err:
                    logger.debug(
                        "Failed to parse nowledge-mem config %s under hermes_home=%s: %s",
                        config_path,
                        hermes_home,
                        err,
                    )
        return {}

    @staticmethod
    def _resolve_space(config: Dict[str, Any], kwargs: Dict[str, Any]) -> str | None:
        if "space" in config:
            raw_space = config.get("space")
            if isinstance(raw_space, str):
                return raw_space.strip()

        raw_identity = kwargs.get("agent_identity")
        identity = str(raw_identity or "").strip()
        identity_map = config.get("space_by_identity")
        if isinstance(identity_map, str):
            try:
                identity_map = json.loads(identity_map)
            except Exception:
                identity_map = None
        if identity and isinstance(identity_map, dict):
            mapped = identity_map.get(identity)
            if isinstance(mapped, str) and mapped.strip():
                return mapped.strip()

        template = str(config.get("space_template") or "").strip()
        if identity and template:
            resolved = template.replace("{identity}", identity)
            resolved = " ".join(resolved.split()).strip()
            if resolved:
                return resolved

        env_space = (os.environ.get("NMEM_SPACE") or "").strip()
        if env_space:
            return env_space

        legacy_env_space = (os.environ.get("NMEM_SPACE_ID") or "").strip()
        if legacy_env_space:
            return legacy_env_space

        return None

    @staticmethod
    def _format_working_memory(wm: Any) -> str:
        if not wm:
            return ""
        if isinstance(wm, dict):
            content = wm.get("content", "") or wm.get("briefing", "")
            if isinstance(content, str) and content.strip():
                return content.strip()
            return json.dumps(wm, indent=2, ensure_ascii=False)
        if isinstance(wm, str):
            return wm.strip()
        return str(wm)

    @staticmethod
    def _format_prefetch(result: Any) -> str:
        if not result or not isinstance(result, dict):
            return ""
        memories = result.get("memories", result.get("results", []))
        if not memories:
            return ""

        lines: List[str] = []
        for memory in memories[:5]:
            score = memory.get("score", memory.get("relevance_score", 0))
            if score < 0.3:
                continue
            title = memory.get("title", "Untitled")
            content = (memory.get("content", "") or "")[:300]
            labels = memory.get("labels", [])
            label_str = f" [{', '.join(labels)}]" if labels else ""
            lines.append(f"- **{title}**{label_str}: {content}")

        if not lines:
            return ""
        return "## Recalled from Nowledge Mem\n" + "\n".join(lines)

    @staticmethod
    def _clean_session_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        cleaned: List[Dict[str, str]] = []
        for message in messages or []:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or "").strip()
            if role not in {"user", "assistant"}:
                continue
            content = NowledgeMemProvider._extract_message_text(message.get("content"))
            if not content:
                continue
            cleaned.append({"role": role, "content": content})
        return cleaned

    @staticmethod
    def _extract_message_text(content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = [NowledgeMemProvider._extract_message_text(item) for item in content]
            return "\n".join(part for part in parts if part).strip()
        if isinstance(content, dict):
            if "text" in content:
                return NowledgeMemProvider._extract_message_text(content.get("text"))
            if "content" in content:
                return NowledgeMemProvider._extract_message_text(content.get("content"))
            return ""
        return str(content).strip()

    @staticmethod
    def _build_thread_title(messages: List[Dict[str, str]]) -> str:
        for message in messages:
            if message.get("role") != "user":
                continue
            first_line = next(
                (line.strip() for line in message.get("content", "").splitlines() if line.strip()),
                "",
            )
            if first_line:
                return first_line[:80]
        return "Hermes session"

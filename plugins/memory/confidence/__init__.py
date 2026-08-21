"""Confidence memory provider.

A local, profile-scoped provider that stores durable memory as atomic statements
with layer/confidence/source/TTL metadata.  It complements the built-in
MEMORY.md/USER.md stores instead of replacing them.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from agent.memory_provider import MemoryProvider
from hermes_constants import get_hermes_home
from tools.registry import tool_error

from .schemas import Confidence, Layer, MemorySource, Scope, SourceKind, now_utc
from .store import ConfidenceMemoryStore

logger = logging.getLogger(__name__)

CONFIDENCE_MEMORY_SCHEMA = {
    "name": "confidence_memory",
    "description": (
        "Layered long-term memory with confidence, source excerpts, TTL, and injection policy. "
        "Use for reviewing, adding, confirming, searching, and deleting confidence-scored memories. "
        "Tentative memories are internal hints only and are never injected into shared-destination output."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "list", "search", "confirm", "delete", "refresh"],
            },
            "id": {"type": "string", "description": "Memory id for confirm/delete."},
            "statement": {"type": "string", "description": "Atomic memory statement for add."},
            "layer": {
                "type": "string",
                "enum": ["profile", "ongoing_theme", "unresolved_question"],
            },
            "confidence": {
                "type": "string",
                "enum": ["confirmed", "inferred", "tentative"],
            },
            "source_kind": {
                "type": "string",
                "enum": ["user_stated", "user_confirmed", "document", "activity_pattern", "inference"],
            },
            "source_ref": {"type": "string"},
            "source_excerpt": {"type": "string"},
            "ttl": {"type": "string", "description": "Optional TTL like 14d, 60d, 180d."},
            "query": {"type": "string", "description": "Search query."},
            "include_inactive": {"type": "boolean"},
            "limit": {"type": "integer"},
        },
        "required": ["action"],
    },
}


class ConfidenceMemoryProvider(MemoryProvider):
    """Local SQLite provider for confidence-scored durable memory."""

    def __init__(self, config: dict | None = None):
        self._config = config or self._load_config()
        self._store: ConfidenceMemoryStore | None = None
        self._session_id = ""
        self._profile_limit = int(self._config.get("profile_limit", 15))

    @property
    def name(self) -> str:
        return "confidence"

    def is_available(self) -> bool:
        return True

    def _load_config(self) -> dict:
        try:
            from hermes_cli.config import cfg_get, load_config_readonly
            config = load_config_readonly()
            return cfg_get(config, "memory", "confidence", default={}) or {}
        except Exception:
            return {}

    def get_config_schema(self):
        return [
            {"key": "db_path", "description": "SQLite DB path", "default": "$HERMES_HOME/confidence_memory.db"},
            {"key": "profile_limit", "description": "Max profile memories injected", "default": "15"},
        ]

    def save_config(self, values, hermes_home):
        config_path = Path(hermes_home) / "config.yaml"
        try:
            import yaml
            from hermes_cli.config import read_user_config_raw
            existing = read_user_config_raw(config_path)
            existing.setdefault("memory", {})
            existing["memory"].setdefault("confidence", {})
            existing["memory"]["confidence"].update(values)
            with open(config_path, "w", encoding="utf-8") as handle:
                yaml.dump(existing, handle, default_flow_style=False, allow_unicode=True)
        except Exception:
            logger.debug("Failed to save confidence memory config", exc_info=True)

    def initialize(self, session_id: str, **kwargs) -> None:
        hermes_home = str(kwargs.get("hermes_home") or get_hermes_home())
        db_path = self._config.get("db_path") or "$HERMES_HOME/confidence_memory.db"
        if isinstance(db_path, str):
            db_path = db_path.replace("$HERMES_HOME", hermes_home).replace("${HERMES_HOME}", hermes_home)
        self._store = ConfidenceMemoryStore(db_path)
        self._session_id = session_id
        self._profile_limit = int(self._config.get("profile_limit", 15))

    def system_prompt_block(self) -> str:
        return (
            "# Confidence Memory\n"
            "Active. Memories have confidence labels: [confirmed] facts, [inferred] soft guidance, "
            "and [tentative] internal hints only. Tentative memories must never be stated as fact, "
            "cited as a reason, or written to shared destinations. If current user input conflicts "
            "with memory, the current user input wins."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not self._store:
            return ""
        try:
            items = self._store.select_for_injection(query=query or "", profile_limit=self._profile_limit)
            if not items:
                return ""
            formatted = self._store.format_for_prompt(items)
            if not formatted:
                return ""
            return "## Confidence Memory Context\n" + formatted
        except Exception:
            logger.debug("confidence memory prefetch failed", exc_info=True)
            return ""

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [CONFIDENCE_MEMORY_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if tool_name != "confidence_memory":
            return tool_error(f"Unknown tool: {tool_name}")
        if not self._store:
            return tool_error("confidence memory provider is not initialized")
        action = args.get("action")
        try:
            if action == "add":
                return self._handle_add(args)
            if action == "list":
                return self._handle_list(args)
            if action == "search":
                return self._handle_search(args)
            if action == "confirm":
                return self._handle_confirm(args)
            if action == "delete":
                return self._handle_delete(args)
            if action == "refresh":
                self._store.refresh_statuses()
                return json.dumps({"success": True})
            return tool_error(f"Unknown action: {action}")
        except Exception as exc:
            return tool_error(str(exc))

    def _source_from_args(self, args: Dict[str, Any], *, default_kind: SourceKind = SourceKind.INFERENCE) -> MemorySource:
        return MemorySource(
            kind=SourceKind(args.get("source_kind") or default_kind.value),
            observed_at=now_utc(),
            ref=args.get("source_ref") or self._session_id,
            excerpt=args.get("source_excerpt") or "",
        )

    def _handle_add(self, args: Dict[str, Any]) -> str:
        statement = (args.get("statement") or "").strip()
        if not statement:
            return tool_error("statement is required for add")
        item_id = self._store.add(
            statement=statement,
            layer=Layer(args.get("layer") or Layer.ONGOING_THEME.value),
            confidence=Confidence(args.get("confidence") or Confidence.TENTATIVE.value),
            sources=[self._source_from_args(args)],
            ttl=args.get("ttl") or "",
            scope=Scope.INJECTION,
        )
        item = self._store.get(item_id)
        return json.dumps({"success": True, "id": item_id, "scope": item.scope, "confidence": item.confidence})

    def _serialize_item(self, item) -> dict:
        return {
            "id": item.id,
            "layer": item.layer,
            "statement": item.statement,
            "confidence": item.confidence,
            "status": item.status,
            "scope": item.scope,
            "ttl": item.ttl,
            "sources": [source.to_json() for source in item.sources],
            "supersededBy": item.superseded_by,
        }

    def _handle_list(self, args: Dict[str, Any]) -> str:
        include = bool(args.get("include_inactive", False))
        limit = int(args.get("limit") or 50)
        items = self._store.list_items(include_inactive=include)[:limit]
        return json.dumps({"success": True, "items": [self._serialize_item(item) for item in items]}, ensure_ascii=False)

    def _handle_search(self, args: Dict[str, Any]) -> str:
        query = args.get("query") or ""
        limit = int(args.get("limit") or 10)
        items = self._store.search(query, include_inactive=bool(args.get("include_inactive", False)), limit=limit)
        return json.dumps({"success": True, "items": [self._serialize_item(item) for item in items]}, ensure_ascii=False)

    def _handle_confirm(self, args: Dict[str, Any]) -> str:
        item_id = args.get("id") or ""
        if not item_id:
            return tool_error("id is required for confirm")
        self._store.confirm(item_id, self._source_from_args(args, default_kind=SourceKind.USER_CONFIRMED))
        return json.dumps({"success": True, "id": item_id})

    def _handle_delete(self, args: Dict[str, Any]) -> str:
        item_id = args.get("id") or ""
        if not item_id:
            return tool_error("id is required for delete")
        self._store.delete(item_id)
        return json.dumps({"success": True, "id": item_id})

    def shutdown(self) -> None:
        if self._store:
            try:
                self._store.close()
            finally:
                self._store = None


# Plugin discovery finds MemoryProvider subclasses automatically, but keeping
# this explicit function makes the directory easy for humans to inspect.
def register_memory_provider() -> ConfidenceMemoryProvider:
    return ConfidenceMemoryProvider()

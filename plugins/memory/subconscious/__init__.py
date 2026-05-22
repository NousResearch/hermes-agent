"""Local structured subconscious memory provider for Hermes."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

from .consolidate import run_consolidation
from .store import LAYERS, SubconsciousStore

_TOOL_SCHEMA = {
    "name": "subconscious",
    "description": (
        "Local structured subconscious memory: status/search/hybrid_search/add/link/related/skill_candidates/consolidate/metrics/expire/conflicts "
        "across working, episodic, semantic, and procedural layers. No network or API keys."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["status", "search", "hybrid_search", "add", "link", "related", "skill_candidates", "consolidate", "metrics", "expire", "conflicts"]},
            "layer": {"type": "string", "enum": sorted(LAYERS)},
            "query": {"type": "string"},
            "content": {"type": "string"},
            "tags": {"type": "array", "items": {"type": "string"}},
            "limit": {"type": "integer"},
            "topic_id": {"type": "string"},
            "chat_id": {"type": "string"},
            "platform": {"type": "string"},
            "source_id": {"type": "integer"},
            "target_id": {"type": "integer"},
            "memory_id": {"type": "integer"},
            "relation": {"type": "string"},
            "weight": {"type": "number"},
            "dry_run": {"type": "boolean"},
        },
        "required": ["action"],
    },
}


class SubconsciousMemoryProvider(MemoryProvider):
    def __init__(self) -> None:
        self._store: SubconsciousStore | None = None
        self._hermes_home = ""
        self._session_id = ""
        self._platform = ""
        self._active = False

    @property
    def name(self) -> str:
        return "subconscious"

    def is_available(self) -> bool:
        return True

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id or ""
        self._hermes_home = kwargs.get("hermes_home") or ""
        if not self._hermes_home:
            from hermes_constants import get_hermes_home
            self._hermes_home = str(get_hermes_home())
        self._platform = kwargs.get("platform") or ""
        db_path = Path(self._hermes_home) / "subconscious" / "subconscious.db"
        self._store = SubconsciousStore(db_path)
        self._active = kwargs.get("agent_context", "primary") == "primary"

    def system_prompt_block(self) -> str:
        return (
            "Subconscious memory provider is available: use it for local structured "
            "recall/status across working, episodic, semantic, and procedural layers."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not self._store or not query.strip():
            return ""
        rows = self._store.hybrid_search(query, limit=6)
        if not rows:
            return ""
        lines = ["## Subconscious Recall"]
        for row in rows:
            lines.append(f"- {row['layer']}: {row['summary'] or row['content']}")
        return "\n".join(lines)

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        if not self._active or not self._store:
            return
        text = (user_content or "").strip()
        if any(marker in text.lower() for marker in ("remember", "одобряю", "карт бланш", "do not", "don't", "always", "never")):
            self._store.add_memory(
                "working",
                text[:1200],
                summary=text[:220],
                tags=["turn", "candidate"],
                source="sync_turn",
                session_id=session_id or self._session_id,
                confidence=0.4,
                metadata={"platform": self._platform},
            )

    def on_memory_write(self, action: str, target: str, content: str, metadata: Dict[str, Any] | None = None) -> None:
        if not self._active or not self._store or action not in {"add", "replace"}:
            return
        layer = "procedural" if target == "memory" and any(k in content.lower() for k in ("workflow", "run ", "command", "skill")) else "semantic"
        self._store.add_memory(
            layer,
            content,
            summary=content[:220],
            tags=["built-in-memory", target],
            source="memory_tool",
            session_id=(metadata or {}).get("session_id", self._session_id),
            confidence=0.8,
            metadata=metadata or {},
        )

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        if not self._store:
            return ""
        rows = self._store.search("pending unresolved todo blocker next", layer="working", limit=5)
        if not rows:
            return ""
        return "Subconscious working memory to preserve:\n" + "\n".join(f"- {r['summary'] or r['content']}" for r in rows)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [_TOOL_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if tool_name != "subconscious":
            return tool_error(f"Unknown subconscious tool: {tool_name}")
        action = args.get("action")
        try:
            if action == "status":
                return json.dumps({"success": True, **self._require_store().status()}, ensure_ascii=False)
            if action == "search":
                rows = self._require_store().search(
                    args.get("query", ""),
                    layer=args.get("layer"),
                    limit=args.get("limit", 8),
                    topic_id=args.get("topic_id"),
                    chat_id=args.get("chat_id"),
                    platform=args.get("platform"),
                )
                return json.dumps({"success": True, "results": rows}, ensure_ascii=False)
            if action == "hybrid_search":
                rows = self._require_store().hybrid_search(
                    args.get("query", ""),
                    layer=args.get("layer"),
                    limit=args.get("limit", 8),
                    topic_id=args.get("topic_id"),
                    chat_id=args.get("chat_id"),
                    platform=args.get("platform"),
                )
                return json.dumps({"success": True, "results": rows}, ensure_ascii=False)
            if action == "add":
                metadata = {
                    key: args[key]
                    for key in ("topic_id", "chat_id", "platform")
                    if args.get(key) not in (None, "")
                }
                row_id = self._require_store().add_memory(
                    args.get("layer") or "semantic",
                    args.get("content") or "",
                    tags=args.get("tags") or ["manual"],
                    source="subconscious_tool",
                    session_id=self._session_id,
                    confidence=0.7,
                    metadata=metadata,
                )
                return json.dumps({"success": True, "id": row_id}, ensure_ascii=False)
            if action == "link":
                edge_id = self._require_store().add_edge(
                    int(args.get("source_id") or 0),
                    int(args.get("target_id") or 0),
                    relation=args.get("relation") or "related",
                    weight=float(args.get("weight") or 0.5),
                    source="subconscious_tool",
                    metadata={"platform": self._platform},
                )
                return json.dumps({"success": True, "edge_id": edge_id}, ensure_ascii=False)
            if action == "related":
                rows = self._require_store().related(int(args.get("memory_id") or 0), limit=args.get("limit", 8))
                return json.dumps({"success": True, "results": rows}, ensure_ascii=False)
            if action == "skill_candidates":
                rows = self._require_store().skill_candidates(limit=args.get("limit", 20))
                return json.dumps({"success": True, "results": rows}, ensure_ascii=False)
            if action == "consolidate":
                result = run_consolidation(self._hermes_home, dry_run=bool(args.get("dry_run", False)))
                return json.dumps(result, ensure_ascii=False)
            if action == "metrics":
                result = self._require_store().capture_metrics_snapshot()
                return json.dumps({"success": True, **result}, ensure_ascii=False)
            if action == "expire":
                result = self._require_store().expire_stale_memories()
                return json.dumps(result, ensure_ascii=False)
            if action == "conflicts":
                result = self._require_store().detect_conflicts()
                return json.dumps(result, ensure_ascii=False)
            return tool_error("action must be one of: status, search, hybrid_search, add, link, related, skill_candidates, consolidate, metrics, expire, conflicts")
        except Exception as exc:
            return tool_error(str(exc))

    def _require_store(self) -> SubconsciousStore:
        if not self._store:
            raise RuntimeError("subconscious provider not initialized")
        return self._store

    def shutdown(self) -> None:
        if self._store:
            self._store.close()
            self._store = None


def register(ctx) -> None:
    ctx.register_memory_provider(SubconsciousMemoryProvider())

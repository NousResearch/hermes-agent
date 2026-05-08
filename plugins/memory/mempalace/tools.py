"""Tool dispatch and result formatting for the MemPalace Hermes plugin."""

from __future__ import annotations

import json
import logging
import re
import time
from difflib import SequenceMatcher
from typing import Any

from .errors import MemPalaceBackendError, MemPalaceToolError
from .events import MESSAGE_KIND_EXPLICIT_MEMORY, SOURCE_TOOL
from .schemas import TOOL_SCHEMAS

logger = logging.getLogger(__name__)


class MemPalaceToolsMixin:
    """Tool schemas, handlers, and search/result helpers."""

    _CONVERSATION_MESSAGE_KINDS = {"user_message", "assistant_message"}
    _MEMORY_PRIORITY_KINDS = {
        "explicit_memory",
        "builtin_memory_write",
        "compressed_context",
        "session_summary",
    }

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return list(TOOL_SCHEMAS)

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        """Dispatch tool calls to MemPalace functions."""
        try:
            return json.dumps(self._dispatch(tool_name, args))
        except MemPalaceToolError as exc:
            return json.dumps(exc.to_dict())
        except MemPalaceBackendError as exc:
            logger.error("MemPalace backend error (%s): %s", tool_name, exc)
            return json.dumps(exc.to_dict())
        except Exception as exc:
            logger.error("MemPalace tool error (%s): %s", tool_name, exc)
            return json.dumps(
                MemPalaceToolError(str(exc), {"tool_name": tool_name}).to_dict()
            )

    def _dispatch(self, tool_name: str, args: dict) -> Any:
        if tool_name != "mempalace_status" and self._collection is None:
            raise MemPalaceBackendError(
                "MemPalace is not initialized", {"tool_name": tool_name}
            )

        if tool_name == "mempalace_memorize":
            return self._do_memorize(args)
        if tool_name == "mempalace_search":
            return self._do_search(args)
        if tool_name == "mempalace_recall":
            return self._do_recall(args)
        if tool_name == "mempalace_forget":
            return self._do_forget(args)
        if tool_name == "mempalace_status":
            return self._do_status()
        raise MemPalaceToolError(f"Unknown tool: {tool_name}", {"tool_name": tool_name})

    def _do_memorize(self, args: dict) -> dict:
        content = str(args.get("content", "") or "").strip()
        if not content:
            raise MemPalaceToolError("content is required")

        room = self._resolve_room(args.get("room"))
        memory_type = str(args.get("memory_type", "factual") or "factual")
        importance = float(args.get("importance", 0.7))
        importance = max(0.0, min(1.0, importance))
        chunk_index = int(time.time() * 1000) % 1000000

        try:
            memory_id = self._store_memory(
                room=room,
                content=content,
                source_file=f"memorize_{memory_type}",
                chunk_index=chunk_index,
                source=SOURCE_TOOL,
                message_kind=MESSAGE_KIND_EXPLICIT_MEMORY,
                memory_type=memory_type,
                importance=importance,
            )
            return {
                "success": True,
                "memory_id": memory_id,
                "message": f"Stored in MemPalace wing='{self._wing}', room='{room}'",
                "memory_type": memory_type,
                "importance": importance,
            }
        except Exception as exc:
            raise MemPalaceBackendError(
                "Failed to store memory", {"cause": str(exc), "room": room}
            ) from exc

    def _do_search(self, args: dict) -> dict:
        query = str(args.get("query", "") or "").strip()
        if not query:
            raise MemPalaceToolError("query is required")

        room = self._resolve_room(args.get("room")) if args.get("room") else None
        requested = int(args.get("top_k", self._n_results))
        top_k = min(max(requested, 1), self._tool_max_results)
        query_limit = self._expanded_fetch_limit(top_k)

        try:
            where_filter = self._build_where(room)
            result = self._collection.query(
                query_texts=[query],
                n_results=query_limit,
                where=where_filter,
            )
            return self._format_search_result(result, raw=True, limit=top_k)
        except Exception as exc:
            raise MemPalaceBackendError(
                "Search failed", {"cause": str(exc), "query": query, "room": room}
            ) from exc

    def _do_recall(self, args: dict) -> dict:
        room = self._resolve_room(args.get("room")) if args.get("room") else None
        requested = int(args.get("n_results", 5))
        n = min(max(requested, 1), self._tool_max_results)
        fetch_limit = self._expanded_fetch_limit(n)

        try:
            where_filter = self._build_where(room)
            result = self._collection.get(
                where=where_filter,
                limit=fetch_limit,
                include=["documents", "metadatas"],
            )

            docs = result.get("documents", [])
            metas = result.get("metadatas", [])
            ids = result.get("ids", [])
            items = []
            for i, doc in enumerate(docs):
                meta = metas[i] if i < len(metas) else {}
                items.append(
                    {
                        "id": ids[i] if i < len(ids) else "",
                        "content": doc,
                        "metadata": meta,
                        "distance": 0.0,
                    }
                )

            return {"results": self._prepare_results(items, limit=n)}
        except Exception as exc:
            raise MemPalaceBackendError(
                "Recall failed", {"cause": str(exc), "room": room}
            ) from exc

    def _do_forget(self, args: dict) -> dict:
        memory_id = str(args.get("memory_id", "") or "").strip()
        if not memory_id:
            raise MemPalaceToolError("memory_id is required")

        try:
            self._collection.delete(ids=[memory_id])
            return {"success": True, "memory_id": memory_id}
        except Exception as exc:
            raise MemPalaceBackendError(
                "Failed to delete memory", {"cause": str(exc), "memory_id": memory_id}
            ) from exc

    def _do_status(self) -> dict:
        try:
            status: dict[str, Any] = {
                "palace_path": self._palace_path,
                "wing": self._wing,
                "collection_name": getattr(self, "_collection_name", ""),
                "kg_enabled": self._kg_enabled,
                "session_id": self._session_id,
                "user_id": self._user_id,
                "platform": getattr(self, "_platform", "default"),
                "tool_max_results": getattr(self, "_tool_max_results", None),
                "room_strategy": getattr(self._config, "room_strategy", None),
            }

            if self._collection is not None:
                try:
                    status["total_memories"] = self._collection.count()
                except Exception:
                    status["total_memories"] = "unknown"

            if self._kg is not None:
                try:
                    status["knowledge_graph"] = self._kg.stats()
                except Exception:
                    status["knowledge_graph"] = "error"

            return status
        except Exception as exc:
            raise MemPalaceBackendError(
                "Status check failed", {"cause": str(exc)}
            ) from exc

    def _raw_search(
        self, query: str, n_results: int = 5, room: str | None = None
    ) -> dict:
        effective_room = self._resolve_room(room) if room else None
        target = min(max(n_results, 1), self._tool_max_results)
        return self._collection.query(
            query_texts=[query],
            n_results=self._expanded_fetch_limit(target),
            where=self._build_where(effective_room),
        )

    def _build_where(self, room: str | None = None) -> dict[str, Any] | None:
        conditions: list[dict[str, Any]] = []
        if self._wing:
            conditions.append({"wing": self._wing})
        if room:
            conditions.append({"room": room})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    def _format_search_result(
        self, result: Any, raw: bool = False, limit: int | None = None
    ) -> dict[str, Any] | str:
        if not isinstance(result, dict):
            prepared = self._prepare_results(
                result if isinstance(result, list) else [], limit=limit
            )
            if raw:
                return {"results": prepared}
            return self._render_results(prepared)

        ids = result.get("ids", [[]])[0] if result.get("ids") else []
        docs = result.get("documents", [[]])[0] if result.get("documents") else []
        metas = result.get("metadatas", [[]])[0] if result.get("metadatas") else []
        distances = result.get("distances", [[]])[0] if result.get("distances") else []

        results_list = []
        for i, doc in enumerate(docs):
            meta = metas[i] if i < len(metas) else {}
            distance = distances[i] if i < len(distances) else None
            results_list.append(
                {
                    "id": ids[i] if i < len(ids) else "",
                    "content": doc,
                    "metadata": meta,
                    "distance": distance,
                }
            )

        prepared = self._prepare_results(results_list, limit=limit)
        if raw:
            return {"results": prepared}
        return self._render_results(prepared)

    def _prepare_results(
        self, results_list: list[dict[str, Any]], limit: int | None = None
    ) -> list[dict[str, Any]]:
        """Deduplicate near-identical hits and suppress noisy conversation records."""
        if not results_list:
            return []

        ranked = sorted(results_list, key=self._result_rank_key)
        unique: list[dict[str, Any]] = []
        seen_keys: set[str] = set()
        normalized_seen: list[str] = []

        for item in ranked:
            content = str(item.get("content", "") or "").strip()
            if not content:
                continue

            normalized = self._normalize_content(content)
            if not normalized:
                continue

            coarse_key = normalized[:180]
            if coarse_key in seen_keys:
                continue
            if any(
                self._looks_like_duplicate(normalized, prior)
                for prior in normalized_seen
            ):
                continue

            seen_keys.add(coarse_key)
            normalized_seen.append(normalized)
            unique.append(item)

        preferred = [
            item for item in unique if not self._is_low_signal_conversation(item)
        ]
        selected = preferred if preferred else unique[:1]
        if limit is not None:
            return selected[: max(limit, 0)]
        return selected

    def _render_results(self, results_list: list[dict[str, Any]]) -> str:
        if not results_list:
            return ""

        lines = ["[MemPalace Memory]", ""]
        for i, r in enumerate(results_list[:5], 1):
            content = r.get("content", "")
            meta = r.get("metadata", {})
            role = meta.get("role", "") if isinstance(meta, dict) else ""
            ts = meta.get("created_at", "") if isinstance(meta, dict) else ""

            if content:
                prefix = f"[{role}] " if role else ""
                ts_str = f" ({ts[:16]})" if ts else ""
                lines.append(f"{i}. {prefix}{content[:200]}{ts_str}")

        return "\n".join(lines)

    def _expanded_fetch_limit(self, requested: int) -> int:
        requested = min(max(requested, 1), self._tool_max_results)
        expanded = max(requested, min(self._tool_max_results, requested * 3))
        return min(expanded, self._tool_max_results)

    def _result_rank_key(
        self, item: dict[str, Any]
    ) -> tuple[int, int, float, float, str]:
        meta = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        message_kind = str(meta.get("message_kind", "") or "")
        is_low_signal = self._is_low_signal_conversation(item)
        priority_bucket = 0 if message_kind in self._MEMORY_PRIORITY_KINDS else 1
        low_signal_bucket = 1 if is_low_signal else 0
        importance = float(meta.get("importance", 0.0) or 0.0)
        distance = item.get("distance")
        distance_value = (
            float(distance) if isinstance(distance, (int, float)) else 999999.0
        )
        created_at = str(meta.get("created_at", "") or "")
        return (
            priority_bucket,
            low_signal_bucket,
            -importance,
            distance_value,
            created_at,
        )

    def _is_low_signal_conversation(self, item: dict[str, Any]) -> bool:
        meta = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        message_kind = str(meta.get("message_kind", "") or "")
        if message_kind not in self._CONVERSATION_MESSAGE_KINDS:
            return False

        if meta.get("memory_type"):
            return False
        if float(meta.get("importance", 0.0) or 0.0) >= 0.85:
            return False
        if str(meta.get("source", "") or "") in {
            "memory",
            "compression",
            "session_end",
            "tool",
        }:
            return False
        return True

    def _normalize_content(self, content: str) -> str:
        normalized = content.casefold()
        normalized = re.sub(r"\s+", " ", normalized)
        normalized = re.sub(r"[`*_#>\-]+", " ", normalized)
        normalized = re.sub(r"[^\w\s\u4e00-\u9fff]", "", normalized)
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized.strip()

    def _looks_like_duplicate(self, left: str, right: str) -> bool:
        if left == right:
            return True
        shorter, longer = sorted((left, right), key=len)
        if shorter and longer.startswith(shorter[: min(len(shorter), 120)]):
            if len(shorter) >= 40:
                return True
        if len(shorter) >= 40 and SequenceMatcher(a=left, b=right).ratio() >= 0.92:
            return True
        return False

"""The LCM context engine.

``LCMEngine`` implements the ``ContextEngine`` ABC (``agent/context_engine``):
it owns the session's compaction policy, and it is the only class in this
plugin that touches host types.  Everything else is plain storage/logic.

Behaviour that the acceptance criteria hang on:

* **Opt-in.**  Nothing here auto-activates.  The user sets ``context.engine:
  lcm``; otherwise the built-in ``ContextCompressor`` is used.  Selecting an
  engine that fails to import or construct leaves the host on the built-in
  path (``plugins.context_engine.load_context_engine`` returns ``None``).
* **Fail-open everywhere.**  Every host-facing hook is wrapped so a storage
  error degrades to a no-op instead of breaking a turn.  ``compress`` returns
  the caller's own list unchanged on any internal failure.
* **Referential compaction.**  ``compress`` records the compactable range,
  writes a lineage-bearing node, and substitutes a bounded digest that names
  the node.  The raw messages are never deleted.
* **Bounded recall.**  The agent-facing tools page; they never assemble a
  whole transcript.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agent.context_engine import ContextEngine

from . import compaction, recall
from .config import LCMConfig, load_lcm_config
from .storage import LCMStore, default_db_path

logger = logging.getLogger(__name__)

ENGINE_NAME = "lcm"

# Tool schemas the engine injects into the agent's tool list.
_TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "name": "lcm_search",
        "description": (
            "Search THIS session's retained raw messages (bounded, current-session "
            "only). Returns at most a few matches with their message_id. Use it to "
            "locate a detail that was compacted out of the live context. For broad "
            "cross-session history search use session_search instead."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search text."},
                "limit": {
                    "type": "integer",
                    "description": "Maximum matches (clamped to the configured ceiling).",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "lcm_expand",
        "description": (
            "Expand one LCM summary node's lineage into bounded pages of the raw "
            "messages it absorbed, or read a single stored message by id. Returns "
            "one page plus has_more/next_page."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "node_id": {"type": "string", "description": "Summary node id from a context summary."},
                "message_id": {"type": "string", "description": "Stored message id."},
                "page": {"type": "integer", "description": "Lineage page index (node_id only)."},
                "page_size": {"type": "integer", "description": "Page size (clamped)."},
                "max_chars": {
                    "type": "integer",
                    "description": "Per-message body cap; clamped to a hard maximum.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "lcm_page",
        "description": (
            "Walk this session's retained raw messages one bounded page at a time, "
            "ascending by sequence. Pass next_cursor back as cursor to continue. The "
            "session is never returned whole in one call."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "cursor": {"type": "integer", "description": "Sequence offset from next_cursor."},
                "page_size": {"type": "integer", "description": "Page size (clamped)."},
            },
            "required": [],
        },
    },
    {
        "name": "lcm_status",
        "description": (
            "LCM diagnostics: retained-message and summary-node counts, schema "
            "version, effective recall bounds, and the active recall policy."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
]

_TOOL_NAMES = frozenset(schema["name"] for schema in _TOOL_SCHEMAS)


class LCMEngine(ContextEngine):
    """DAG-based, opt-in context engine with lineage-preserving compaction."""

    # Routine automatic compaction is background maintenance here; keep it out
    # of the user-visible lifecycle stream (warnings/errors/manual still show).
    emit_automatic_compaction_status = False

    def __init__(
        self,
        config: Optional[LCMConfig] = None,
        store: Optional[LCMStore] = None,
        *,
        context_length: int = 0,
    ) -> None:
        self._config = config or load_lcm_config()
        self._store_override = store
        self._store: Optional[LCMStore] = None

        self.threshold_percent = self._config.threshold_percent
        self.protect_first_n = self._config.protect_first_n
        self.protect_last_n = self._config.protect_last_n
        self.context_length = int(context_length or 0)
        self.threshold_tokens = int(self.context_length * self.threshold_percent)

        self._session_id: Optional[str] = None
        self._last_node_id: Optional[str] = None
        self._last_compaction: Optional[Dict[str, Any]] = None
        self._ingest_failures = 0

    # -- identity ----------------------------------------------------------

    @property
    def name(self) -> str:
        return ENGINE_NAME

    def is_available(self) -> bool:
        """Whether the engine can run on this host (SQLite + its storage dir)."""
        try:
            self._ensure_store()
            return self._store is not None
        except Exception:
            return False

    # -- store / session lifecycle ----------------------------------------

    def _db_path(self) -> Path:
        if self._config.db_path:
            return Path(self._config.db_path).expanduser()
        try:
            from hermes_constants import get_hermes_home

            home = Path(get_hermes_home())
        except Exception:  # pragma: no cover - host always provides it
            home = Path.home() / ".hermes"
        return default_db_path(home)

    def _ensure_store(self) -> Optional[LCMStore]:
        if self._store is not None and self._store.is_open:
            return self._store
        if self._store_override is not None:
            self._store = self._store_override.open()
        else:
            self._store = LCMStore(
                self._db_path(),
                max_page_size=self._config.max_page_size,
                page_size=self._config.page_size,
                body_chars=self._config.body_chars,
                max_search_results=self._config.max_search_results,
            ).open()
        return self._store

    def bind_session_state(self, session_db: Any = None, session_id: str = "") -> None:
        """Optional host hook; the engine keeps its own store, so this is a no-op."""
        if session_id:
            self._session_id = str(session_id)

    def on_session_start(self, session_id: str, **kwargs: Any) -> None:
        """Open the store for this session."""
        try:
            self._session_id = str(session_id)
            self._ensure_store()
        except Exception as exc:  # pragma: no cover - fail-open
            logger.warning("LCM: session start failed (%s); running without storage", exc)

    def on_session_end(self, session_id: str, messages: List[Dict[str, Any]] = None) -> None:
        """Flush any final tail, then close the store."""
        try:
            if messages and self._config.ingest_on_turn_complete:
                self._ingest(str(session_id), messages)
        except Exception as exc:  # pragma: no cover - fail-open
            logger.debug("LCM: final ingest failed: %s", exc)
        finally:
            try:
                if self._store is not None:
                    self._store.close()
            except Exception:  # pragma: no cover
                pass

    def on_session_reset(self) -> None:
        """Host hook (``/new``, ``/reset``): reset counters, keep the store."""
        super().on_session_reset()
        self._last_node_id = None
        self._last_compaction = None

    # -- ingestion ---------------------------------------------------------

    def _ingest(self, session_id: str, messages: List[Dict[str, Any]]) -> List[str]:
        store = self._ensure_store()
        if store is None:
            return ["" for _ in messages]
        return store.sync_transcript(session_id, messages)

    def on_turn_complete(
        self, messages: List[Dict[str, Any]], usage: Dict[str, Any] = None, **kwargs: Any
    ) -> None:
        """Durably record the finished turn's tail (observation only).

        This is the ingestion mirror for an engine that already owns
        compaction: every raw message is retained as it arrives, so the store
        is complete even for turns that never trigger a compaction pass.
        """
        if not self._config.ingest_on_turn_complete:
            return
        session_id = kwargs.get("session_id") or self._session_id
        if not session_id or not messages:
            return
        try:
            self._ingest(str(session_id), messages)
        except Exception as exc:
            self._ingest_failures += 1
            logger.debug("LCM: turn ingest failed: %s", exc)

    # -- token accounting --------------------------------------------------

    def update_from_response(self, usage: Dict[str, Any]) -> None:
        if not isinstance(usage, dict):
            return
        prompt = usage.get("prompt_tokens", usage.get("input_tokens"))
        completion = usage.get("completion_tokens", usage.get("output_tokens"))
        total = usage.get("total_tokens")
        if isinstance(prompt, (int, float)):
            self.last_prompt_tokens = int(prompt)
        if isinstance(completion, (int, float)):
            self.last_completion_tokens = int(completion)
        if isinstance(total, (int, float)):
            self.last_total_tokens = int(total)
        elif isinstance(prompt, (int, float)) and isinstance(completion, (int, float)):
            self.last_total_tokens = int(prompt) + int(completion)

    def should_compress(self, prompt_tokens: int = None) -> bool:
        tokens = self.last_prompt_tokens if prompt_tokens is None else prompt_tokens
        if not isinstance(tokens, (int, float)) or self.threshold_tokens <= 0:
            return False
        return tokens >= self.threshold_tokens

    def update_model(
        self,
        model: str,
        context_length: int,
        base_url: str = "",
        api_key: str = "",
        provider: str = "",
        api_mode: str = "",
    ) -> None:
        super().update_model(
            model=model,
            context_length=context_length,
            base_url=base_url,
            api_key=api_key,
            provider=provider,
            api_mode=api_mode,
        )

    # -- compaction --------------------------------------------------------

    def compress(
        self,
        messages: List[Dict[str, Any]],
        current_tokens: Optional[int] = None,
        focus_topic: Optional[str] = None,
        force: bool = False,
        memory_context: str = "",
    ) -> List[Dict[str, Any]]:
        """Replace the compactable middle with a bounded, lineage-bearing digest.

        Returns *messages* unchanged on any internal failure — a broken engine
        must never be worse than no engine at all.
        """
        try:
            return self._compress(messages, focus_topic=focus_topic)
        except Exception as exc:
            logger.warning("LCM: compaction failed (%s); leaving context uncompacted", exc)
            return messages

    def _compress(
        self, messages: List[Dict[str, Any]], *, focus_topic: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        if not isinstance(messages, list) or not messages:
            return messages
        if not all(isinstance(m, dict) for m in messages):
            return messages

        plan = compaction.plan_compaction(
            messages,
            protect_first_n=self.protect_first_n,
            protect_last_n=self.protect_last_n,
        )
        if plan is None:
            return messages
        head_end, tail_start = plan

        session_id = self._session_id or "unknown-session"
        store = self._ensure_store()
        if store is None:
            return messages

        # 1. Retain every raw message first (idempotent append), so the range we
        #    are about to compact is already durable before we reference it.
        #    The returned ids are aligned to transcript position, which is what
        #    makes the lineage below exact.
        ids = store.sync_transcript(session_id, messages)

        middle = messages[head_end:tail_start]
        middle_ids = ids[head_end:tail_start]

        sources: List[Tuple[str, str]] = []
        prior_nodes: List[str] = []
        for message, message_id in zip(middle, middle_ids):
            node_ref = compaction.summary_node_id(message)
            if node_ref:
                # Descending the DAG: the superseded node is the true ancestor,
                # not the rendered marker message that stands in for it.
                sources.append(("node", node_ref))
                prior_nodes.append(node_ref)
            elif message_id:
                sources.append(("message", message_id))
        if not sources:
            return messages

        level = 1
        if prior_nodes:
            parent_levels = [
                node["level"] for node in (store.node(ref) for ref in prior_nodes) if node
            ]
            level = (max(parent_levels) if parent_levels else 1) + 1

        # The id is derived from the sources, so it is known before the summary
        # text exists — the text can therefore cite the node that will hold it.
        node_id = store.resolve_node_id(session_id, level, sources)
        summary = compaction.build_summary(
            middle,
            node_id=node_id,
            level=level,
            max_chars=self._config.summary_chars,
            page_size=self._config.max_page_size,
            prior_node_ids=prior_nodes,
        )
        store.create_node(
            session_id,
            level=level,
            summary=summary,
            sources=sources,
            node_id=node_id,
        )

        marker = compaction.build_summary_message(
            node_id=node_id,
            level=level,
            summary=summary,
            role=compaction.choose_summary_role(messages[:head_end], messages[tail_start:]),
        )

        store.mark_compacted(
            session_id, [sid for kind, sid in sources if kind == "message"]
        )

        self.compression_count += 1
        self._last_node_id = node_id
        self._last_compaction = {
            "node_id": node_id,
            "level": level,
            "source_count": len(sources),
            "messages_before": len(messages),
            "messages_after": head_end + 1 + (len(messages) - tail_start),
            "focus_topic": focus_topic or "",
        }

        return messages[:head_end] + [marker] + messages[tail_start:]

    # -- tools -------------------------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [dict(schema) for schema in _TOOL_SCHEMAS]

    def handle_tool_call(self, name: str, args: Dict[str, Any], **kwargs: Any) -> str:
        """Dispatch an LCM tool call; always returns a JSON string."""
        try:
            if name not in _TOOL_NAMES:
                return json.dumps({"error": f"unknown LCM tool: {name}"})
            store = self._ensure_store()
            session_id = str(kwargs.get("session_id") or self._session_id or "")
            if store is None:
                return json.dumps({"error": "LCM store unavailable"})
            if not session_id:
                return json.dumps(
                    {"error": "no active session for LCM recall; open a session first"}
                )
            payload = self._dispatch_tool(name, store, session_id, args or {})
            return json.dumps(payload, default=str)
        except Exception as exc:
            logger.warning("LCM: tool %s failed: %s", name, exc)
            return json.dumps({"error": f"LCM tool failed: {exc}"})

    def _dispatch_tool(
        self, name: str, store: LCMStore, session_id: str, args: Dict[str, Any]
    ) -> Dict[str, Any]:
        if name == "lcm_search":
            return recall.search(
                store, session_id, str(args.get("query", "")), limit=args.get("limit")
            )
        if name == "lcm_expand":
            return recall.expand(
                store,
                session_id,
                node_id=args.get("node_id"),
                message_id=args.get("message_id"),
                page=int(args.get("page") or 0),
                page_size=args.get("page_size"),
                max_chars=args.get("max_chars"),
            )
        if name == "lcm_page":
            return recall.page(
                store,
                session_id,
                cursor=int(args.get("cursor") or 0),
                page_size=args.get("page_size"),
            )
        return recall.status(store, session_id)

    # -- diagnostics -------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        status = super().get_status()
        status.update(
            {
                "engine": ENGINE_NAME,
                "session_id": self._session_id,
                "last_node_id": self._last_node_id,
                "last_compaction": self._last_compaction,
                "ingest_failures": self._ingest_failures,
                "recall_policy": recall.policy(),
            }
        )
        return status

    def diagnostics(self) -> Dict[str, Any]:
        """Extended diagnostics (store shape + bounds); used by ``/lcm``."""
        base = self.get_status()
        try:
            store = self._ensure_store()
            if store is None:
                base["store"] = {"open": False}
                return base
            base["store"] = {
                "open": store.is_open,
                "db_path": str(store.db_path),
                "bounds": {
                    "page_size": store.page_size,
                    "max_page_size": store.max_page_size,
                    "max_search_results": store.max_search_results,
                    "body_chars": store.body_chars,
                },
                "stats": store.stats(self._session_id or ""),
            }
        except Exception as exc:
            base["store"] = {"error": str(exc)}
        return base


def build_engine(config: Optional[LCMConfig] = None) -> LCMEngine:
    """Construct an engine instance (used by ``register`` and by tests)."""
    return LCMEngine(config=config)

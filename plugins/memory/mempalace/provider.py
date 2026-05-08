"""Provider implementation for the MemPalace Hermes memory plugin."""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

from agent.memory_provider import MemoryProvider

# MemPalace imports are optional — guard at module level to allow
# graceful degradation when mempalace is not installed.
try:
    from mempalace.knowledge_graph import KnowledgeGraph  # type: ignore[unresolved-import]
    from mempalace.palace import get_collection  # type: ignore[unresolved-import]

    _MEMPALACE_AVAILABLE = True
except ImportError:
    KnowledgeGraph = None
    get_collection = None  # type: ignore[assignment]
    _MEMPALACE_AVAILABLE = False

from .collections import resolve_collection_name, resolve_room
from .config import (
    DEFAULT_ENABLE_KG,
    DEFAULT_N_RESULTS,
    DEFAULT_TOOL_MAX_RESULTS,
    DEFAULT_WING,
    MemPalaceConfig,
    load_mempalace_config,
)
from .events import (
    MESSAGE_KIND_ASSISTANT_MESSAGE,
    MESSAGE_KIND_USER_MESSAGE,
    SOURCE_SYNC_TURN,
)
from .hooks import MemPalaceHooksMixin
from .store import build_memory_item, upsert_memory_item
from .tools import MemPalaceToolsMixin
from .writer import WriteQueue

logger = logging.getLogger(__name__)


class MemPalaceMemoryProvider(MemPalaceHooksMixin, MemPalaceToolsMixin, MemoryProvider):
    """MemPalace local-first AI memory provider."""

    def __init__(self):
        self._collection = None
        self._kg: Any | None = None
        self._queue: WriteQueue | None = None
        self._palace_path = ""
        self._wing = DEFAULT_WING
        self._n_results = DEFAULT_N_RESULTS
        self._tool_max_results = DEFAULT_TOOL_MAX_RESULTS
        self._kg_enabled = DEFAULT_ENABLE_KG
        self._session_id = ""
        self._user_id = "default"
        self._agent_id = "hermes"
        self._platform = "default"
        self._config = MemPalaceConfig()
        self._runtime_ctx: dict[str, Any] = {}
        self._collection_name = ""
        self._lock = threading.Lock()
        self._thread_factory = threading.Thread
        self._prefetch_result = ""
        self._prefetch_thread: threading.Thread | None = None
        self._recent_messages: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return "mempalace"

    def is_available(self) -> bool:
        return _MEMPALACE_AVAILABLE

    def get_config_schema(self) -> list[dict[str, Any]]:
        return [
            {
                "key": "palace_path",
                "description": "Root directory for MemPalace data (ChromaDB + knowledge graph). Defaults to ~/.hermes/mempalace/",
                "default": "",
                "required": False,
            },
            {
                "key": "wing",
                "description": "Wing name for organizing memories (default: conversations)",
                "default": "conversations",
                "required": False,
            },
            {
                "key": "n_results",
                "description": "Default semantic search result count (default: 5)",
                "default": 5,
                "required": False,
            },
            {
                "key": "tool_max_results",
                "description": "Upper bound for tool-driven result counts (default: 20)",
                "default": 20,
                "required": False,
            },
            {
                "key": "enable_kg",
                "description": "Enable knowledge graph support when available",
                "default": True,
                "required": False,
            },
            {
                "key": "collection_name",
                "description": "Explicit Chroma collection name. Overrides collection_template when set.",
                "default": "",
                "required": False,
            },
            {
                "key": "collection_template",
                "description": "Collection naming template using {user_id}, {platform}, {session_id}, {agent_id}.",
                "default": "hermes-{platform}-{user_id}",
                "required": False,
            },
            {
                "key": "room_strategy",
                "description": "How rooms are derived when callers do not specify a room.",
                "default": "platform_session",
                "required": False,
            },
            {
                "key": "fixed_room",
                "description": "Room name used when room_strategy=fixed.",
                "default": "memory",
                "required": False,
            },
        ]

    def initialize(self, session_id: str, **kwargs) -> None:
        if not _MEMPALACE_AVAILABLE:
            raise RuntimeError(
                "mempalace package is not installed. "
                "Install it with: pip install mempalace"
            )
        self._session_id = session_id
        cfg = load_mempalace_config(
            kwargs.get("config", {}),
            hermes_home=str(kwargs.get("hermes_home", "") or ""),
        )
        self._config = cfg
        self._wing = cfg.wing
        self._n_results = cfg.n_results
        self._tool_max_results = cfg.tool_max_results
        self._kg_enabled = cfg.enable_kg
        self._palace_path = cfg.palace_path
        os.makedirs(self._palace_path, exist_ok=True)

        self._user_id = kwargs.get("user_id", "default") or "default"
        self._agent_id = kwargs.get("agent_id", "hermes") or "hermes"
        self._platform = kwargs.get("platform", "default") or "default"
        self._runtime_ctx = {
            "session_id": session_id,
            "user_id": self._user_id,
            "agent_id": self._agent_id,
            "platform": self._platform,
        }

        self._collection_name = resolve_collection_name(cfg, self._runtime_ctx)
        if get_collection is None:
            raise RuntimeError("mempalace package is not installed")
        self._collection = get_collection(
            palace_path=self._palace_path,
            collection_name=self._collection_name,
        )
        logger.info(
            "MemPalace collection '%s' initialized at %s",
            self._collection_name,
            self._palace_path,
        )

        if self._kg_enabled:
            kg_path = os.path.join(self._palace_path, "knowledge_graph.db")
            if KnowledgeGraph is None:
                raise RuntimeError("mempalace knowledge graph is not installed")
            self._kg = KnowledgeGraph(db_path=kg_path)
            logger.info("MemPalace knowledge graph initialized at %s", kg_path)
        else:
            self._kg = None

        self._queue = WriteQueue(
            self._collection, self._agent_id, thread_factory=self._thread_factory
        )
        logger.info("MemPalace initialized successfully for session %s", session_id)

    def system_prompt_block(self) -> str:
        return (
            "# MemPalace Memory\n"
            "MemPalace is active for long-term memory. Use mempalace_memorize to store "
            "important facts, preferences, and decisions. Use mempalace_search to find "
            "relevant memories. MemPalace uses semantic search — it finds related concepts "
            "even when exact words don't match."
        )

    def shutdown(self) -> None:
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=3.0)
        if self._queue:
            self._queue.shutdown()
        if self._kg:
            try:
                self._kg.close()
            except Exception:
                pass
        logger.info("MemPalace shutdown complete")

    def sync_turn(
        self, user_content: str, assistant_content: str, *, session_id: str = ""
    ) -> None:
        if not self._queue:
            return

        effective_session = session_id or self._session_id
        room = self._resolve_room(session_id=effective_session)
        items = []

        if user_content and user_content.strip():
            items.append(
                self._queue_item(
                    room=room,
                    content=user_content.strip(),
                    source_file="conversation",
                    chunk_index=0,
                    source=SOURCE_SYNC_TURN,
                    message_kind=MESSAGE_KIND_USER_MESSAGE,
                    session_id=effective_session,
                )
            )

        if assistant_content and assistant_content.strip():
            items.append(
                self._queue_item(
                    room=room,
                    content=assistant_content.strip(),
                    source_file="conversation",
                    chunk_index=1,
                    source=SOURCE_SYNC_TURN,
                    message_kind=MESSAGE_KIND_ASSISTANT_MESSAGE,
                    session_id=effective_session,
                )
            )

        if items:
            self._queue.enqueue(items)

    def _runtime_context(self, *, session_id: str | None = None) -> dict[str, Any]:
        runtime_ctx = dict(self._runtime_ctx)
        if session_id:
            runtime_ctx["session_id"] = session_id
        return runtime_ctx

    def _resolve_room(
        self, explicit_room: str | None = None, *, session_id: str | None = None
    ) -> str:
        runtime_ctx = self._runtime_context(session_id=session_id)
        return resolve_room(self._config, runtime_ctx, explicit_room=explicit_room)

    def _queue_item(
        self,
        *,
        room: str,
        content: str,
        source_file: str,
        chunk_index: int,
        source: str,
        message_kind: str,
        session_id: str | None = None,
        memory_type: str | None = None,
        importance: float | None = None,
    ) -> dict[str, Any]:
        return build_memory_item(
            runtime_ctx=self._runtime_context(session_id=session_id),
            wing=self._wing,
            room=room,
            content=content,
            source_file=source_file,
            chunk_index=chunk_index,
            source=source,
            message_kind=message_kind,
            agent_id=self._agent_id,
            memory_type=memory_type,
            importance=importance,
        )

    def _store_memory(
        self,
        *,
        room: str,
        content: str,
        source_file: str,
        chunk_index: int,
        source: str,
        message_kind: str,
        session_id: str | None = None,
        memory_type: str | None = None,
        importance: float | None = None,
    ) -> str:
        item = self._queue_item(
            room=room,
            content=content,
            source_file=source_file,
            chunk_index=chunk_index,
            source=source,
            message_kind=message_kind,
            session_id=session_id,
            memory_type=memory_type,
            importance=importance,
        )
        return upsert_memory_item(self._collection, item, self._agent_id)

"""Runtime session manager for Hermes' native RetainDB integration."""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from datetime import datetime, timezone
from typing import Any

from retaindb_integration.client import RetainDBClient, RetainDBClientConfig
from retaindb_integration.identity import ResolvedRetainDBIdentity, RetainDBIdentityResolver
from retaindb_integration.recall import build_retaindb_overlay
from retaindb_integration.write_queue import DurableRetainDBWriteQueue

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class RetainDBSessionManager:
    """Turn-time recall + durable write-behind manager for a Hermes session."""

    def __init__(
        self,
        client: RetainDBClient | None = None,
        config: RetainDBClientConfig | None = None,
        runtime_identity: dict[str, Any] | None = None,
    ):
        self._config = config or RetainDBClientConfig.from_global_config()
        self._client = client or RetainDBClient(self._config)
        self._runtime_identity = dict(runtime_identity or {})
        self._identity_resolver = RetainDBIdentityResolver(self._config)
        self._write_queue = DurableRetainDBWriteQueue(self._client, self._config)
        self._prefetch_cache: dict[str, dict[str, Any]] = {}
        self._prefetch_lock = threading.Lock()

    @property
    def config(self) -> RetainDBClientConfig:
        return self._config

    def set_runtime_identity(self, runtime_identity: dict[str, Any] | None) -> None:
        self._runtime_identity = dict(runtime_identity or {})

    def resolve_identity(self, session_id: str) -> ResolvedRetainDBIdentity:
        runtime = dict(self._runtime_identity)
        runtime.setdefault("session_id", session_id)
        return self._identity_resolver.resolve(
            session_id=session_id,
            runtime_identity=runtime,
        )

    def connection_status(self) -> dict[str, Any]:
        try:
            projects = self._client.list_projects()
            return {"ok": True, "projects": projects}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def get_profile(self, session_id: str) -> dict[str, Any]:
        identity = self.resolve_identity(session_id)
        return self._client.get_user_profile(
            project=identity.project,
            user_id=identity.user_id,
            include_pending=True,
            timeout_ms=self._config.prefetch_timeout_ms,
        )

    def search(self, session_id: str, query: str, *, top_k: int = 8) -> dict[str, Any]:
        identity = self.resolve_identity(session_id)
        return self._client.search_memories(
            project=identity.project,
            query=query,
            user_id=identity.user_id,
            session_id=identity.session_id,
            agent_id=identity.agent_id,
            top_k=top_k,
            include_pending=True,
            timeout_ms=self._config.prefetch_timeout_ms,
        )

    def remember(
        self,
        session_id: str,
        content: str,
        *,
        memory_type: str = "factual",
        importance: float = 0.6,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        identity = self.resolve_identity(session_id)
        return self._client.add_memory(
            project=identity.project,
            content=content,
            memory_type=memory_type,
            user_id=identity.user_id,
            session_id=identity.session_id,
            agent_id=identity.agent_id,
            importance=importance,
            metadata=metadata or {},
            write_mode="sync",
            timeout_ms=max(2500, self._config.prefetch_timeout_ms),
        )

    def forget(self, memory_id: str) -> dict[str, Any]:
        return self._client.delete_memory(memory_id)

    def get_context(
        self,
        session_id: str,
        query: str,
        *,
        local_entries: list[str] | None = None,
        recent_texts: list[str] | None = None,
    ) -> dict[str, Any]:
        identity = self.resolve_identity(session_id)
        query_result = self._client.query(
            project=identity.project,
            query=query,
            user_id=identity.user_id,
            session_id=identity.session_id,
            agent_id=identity.agent_id,
            include_memories=True,
            max_tokens=self._config.context_tokens,
            timeout_ms=self._config.prefetch_timeout_ms,
        )
        profile = self._client.get_user_profile(
            project=identity.project,
            user_id=identity.user_id,
            include_pending=True,
            timeout_ms=self._config.prefetch_timeout_ms,
        )
        overlay = build_retaindb_overlay(
            profile=profile,
            query_result=query_result,
            local_entries=local_entries,
            recent_texts=recent_texts,
        )
        return {
            "identity": identity,
            "profile": profile,
            "query": query_result,
            "context": overlay,
        }

    def _prefetch_worker(
        self,
        session_id: str,
        query: str,
        local_entries: list[str] | None,
        recent_texts: list[str] | None,
        generation: str,
    ) -> None:
        result_payload: dict[str, Any]
        try:
            result_payload = self.get_context(
                session_id,
                query,
                local_entries=local_entries,
                recent_texts=recent_texts,
            )
        except Exception as exc:
            logger.debug("RetainDB prefetch failed for %s: %s", session_id, exc)
            result_payload = {"context": "", "error": str(exc)}

        with self._prefetch_lock:
            cache = self._prefetch_cache.get(session_id)
            if not cache or cache.get("generation") != generation:
                return
            cache["result"] = result_payload.get("context", "")
            cache["details"] = result_payload
            cache["event"].set()

        if self._config.debug_recall_trace and result_payload.get("context"):
            logger.info(
                "RetainDB recall trace session=%s profile_memories=%s query_results=%s",
                session_id,
                len((result_payload.get("profile") or {}).get("memories") or []),
                len((result_payload.get("query") or {}).get("results") or []),
            )

    def prefetch_context(
        self,
        session_id: str,
        query: str,
        *,
        local_entries: list[str] | None = None,
        recent_texts: list[str] | None = None,
    ) -> None:
        generation = hashlib.sha1(
            f"{session_id}:{query}:{_now_iso()}".encode("utf-8")
        ).hexdigest()
        event = threading.Event()
        with self._prefetch_lock:
            self._prefetch_cache[session_id] = {
                "generation": generation,
                "event": event,
                "result": "",
                "details": {},
            }
        thread = threading.Thread(
            target=self._prefetch_worker,
            args=(session_id, query, local_entries, recent_texts, generation),
            name=f"retaindb-prefetch-{session_id}",
            daemon=True,
        )
        thread.start()

    def pop_context_result(self, session_id: str, *, wait_ms: int | None = None) -> str:
        with self._prefetch_lock:
            cache = self._prefetch_cache.get(session_id)
        if not cache:
            return ""
        event: threading.Event = cache["event"]
        timeout = max(0.0, float(wait_ms or self._config.prefetch_timeout_ms) / 1000.0)
        if not event.wait(timeout=timeout):
            with self._prefetch_lock:
                self._prefetch_cache.pop(session_id, None)
            return ""
        with self._prefetch_lock:
            final = self._prefetch_cache.pop(session_id, None) or {}
        return str(final.get("result") or "")

    def enqueue_turn(
        self,
        session_id: str,
        user_content: str,
        assistant_content: str,
        *,
        message_index: int,
        turn_id: str,
    ) -> None:
        identity = self.resolve_identity(session_id)
        messages = [
            {"role": "user", "content": user_content, "timestamp": _now_iso()},
            {"role": "assistant", "content": assistant_content, "timestamp": _now_iso()},
        ]
        payload_checksum = hashlib.sha1(
            json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        self._write_queue.enqueue(
            identity,
            turn_id=turn_id,
            message_index=message_index,
            payload_checksum=payload_checksum,
            messages=messages,
        )

        if self._config.write_frequency == "turn":
            self.flush_session(session_id)

    def save_user_observation(self, session_id: str, content: str) -> dict[str, Any]:
        return self.remember(
            session_id,
            content,
            memory_type="factual",
            importance=0.7,
            metadata={"source": "hermes.memory_tool"},
        )

    def flush_session(self, session_id: str) -> None:
        identity = self.resolve_identity(session_id)
        self._write_queue.flush_session(identity)

    def flush_all(self) -> None:
        self._write_queue.flush_all()

    def shutdown(self) -> None:
        self._write_queue.shutdown()

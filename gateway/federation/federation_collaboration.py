"""Federation collaboration layer — cross-device memory sync and distributed search.

Memory Sync:
- When a device adds/updates/deletes a memory entry, broadcast to all peers
- Peers apply the change idempotently (same content = no-op)
- Conflict resolution: latest timestamp wins

Distributed Search:
- Query session_search across all federation peers
- Aggregate results with deduplication
- Return unified search results from all devices

Protocol messages:
- MEMORY_SYNC: memory entry broadcast
- SEARCH_QUERY: distributed search request
- SEARCH_RESULT: search response from a peer
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from gateway.federation.federation_protocol import FedMessage, MessageType

logger = logging.getLogger(__name__)


# ========================================================================
# Memory sync
# ========================================================================

@dataclass
class MemoryEntry:
    """A single memory entry to sync across devices."""

    node_id: str
    content: str
    target: str = "memory"  # memory | user
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    version: int = 1

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "content": self.content,
            "target": self.target,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryEntry":
        return cls(**d)


class FederationMemorySync:
    """Synchronize memory entries across federation peers.

    Usage:
        sync = FederationMemorySync(
            device_id="my-device",
            adapter=federation_adapter,
            hermes_home=Path.home() / ".hermes",
        )
        await sync.start()

        # When memory changes locally, notify sync:
        sync.on_local_memory_change(node_id, content, target)
    """

    def __init__(
        self,
        device_id: str,
        adapter: Any,  # FederationAdapter
        hermes_home: Optional[Path] = None,
    ):
        self.device_id = device_id
        self.adapter = adapter
        self.hermes_home = hermes_home or Path.home() / ".hermes"
        self._local_memories: Dict[str, MemoryEntry] = {}
        self._running = False
        self._sync_interval = 60  # seconds between full sync broadcasts
        self._on_apply: Optional[Callable[[MemoryEntry], None]] = None

    async def start(self) -> None:
        """Start memory sync — register handlers and load local memories."""
        self._running = True
        self._load_local_memories()
        logger.info(
            "Federation memory sync: started (device=%s, entries=%d)",
            self.device_id, len(self._local_memories),
        )

    async def stop(self) -> None:
        """Stop memory sync."""
        self._running = False
        logger.info("Federation memory sync: stopped")

    def _load_local_memories(self) -> None:
        """Load all local memory entries for sync."""
        for target_dir in ["memory", "user"]:
            mem_path = self.hermes_home / "memories" / f"{target_dir}.md"
            if not mem_path.exists():
                continue
            try:
                content = mem_path.read_text(encoding="utf-8")
                # Parse markdown entries (simplified — each ## is a node)
                lines = content.split("\n")
                current_id = None
                current_content: list[str] = []

                for line in lines:
                    if line.startswith("## "):
                        # Save previous entry
                        if current_id and current_content:
                            self._local_memories[current_id] = MemoryEntry(
                                node_id=current_id,
                                content="\n".join(current_content).strip(),
                                target=target_dir,
                            )
                        current_id = line[3:].strip()
                        current_content = []
                    elif current_id:
                        current_content.append(line)

                # Save last entry
                if current_id and current_content:
                    self._local_memories[current_id] = MemoryEntry(
                        node_id=current_id,
                        content="\n".join(current_content).strip(),
                        target=target_dir,
                    )
            except Exception as e:
                logger.warning("Federation: failed to load memories: %s", e)

    async def on_local_memory_change(
        self, node_id: str, content: str, target: str = "memory",
    ) -> None:
        """Called when a memory entry changes locally — broadcast to peers."""
        entry = MemoryEntry(
            node_id=node_id,
            content=content,
            target=target,
            updated_at=time.time(),
        )
        self._local_memories[node_id] = entry

        # Broadcast to all peers
        msg = FedMessage(
            msg_type=MessageType.MEMORY_SYNC.value,
            sender_id=self.device_id,
            payload={
                "action": "update",
                "entry": entry.to_dict(),
            },
        )
        await self.adapter.send(msg)
        logger.debug(
            "Federation memory: synced entry %s to peers", node_id,
        )

    async def handle_memory_sync(self, msg: FedMessage) -> None:
        """Handle incoming MEMORY_SYNC message from a peer."""
        action = msg.payload.get("action", "")
        sender = msg.sender_id

        if sender == self.device_id:
            return  # Ignore self

        if action == "update":
            entry_data = msg.payload.get("entry", {})
            if not entry_data:
                return

            remote = MemoryEntry.from_dict(entry_data)

            # Check if we already have this entry
            local = self._local_memories.get(remote.node_id)
            if local and local.updated_at >= remote.updated_at:
                # Our version is newer or same — skip
                return

            # Apply remote entry (idempotent — write if different)
            self._apply_remote_entry(remote)
            self._local_memories[remote.node_id] = remote

            # Re-broadcast to other peers (gossip)
            if self.adapter.get_peer_count() > 1:
                await self.adapter.send(FedMessage(
                    msg_type=MessageType.MEMORY_SYNC.value,
                    sender_id=self.device_id,
                    target_id=None,  # broadcast to others
                    payload=msg.payload,
                ))

        elif action == "delete":
            node_id = msg.payload.get("node_id", "")
            self._local_memories.pop(node_id, None)

    def _apply_remote_entry(self, entry: MemoryEntry) -> None:
        """Apply a remote memory entry to local storage."""
        mem_path = self.hermes_home / "memories" / f"{entry.target}.md"
        mem_path.parent.mkdir(parents=True, exist_ok=True)

        if not mem_path.exists():
            # Create new file
            mem_path.write_text(f"## {entry.node_id}\n{entry.content}\n", encoding="utf-8")
            logger.info(
                "Federation memory: created new entry %s", entry.node_id,
            )
            return

        # Update existing file
        content = mem_path.read_text(encoding="utf-8")
        marker = f"## {entry.node_id}"

        if marker in content:
            # Entry exists — update in place
            sections = content.split("## ")
            new_sections = []
            for section in sections:
                if section.startswith(entry.node_id):
                    # Replace this section's content
                    lines = section.split("\n", 1)
                    new_sections.append(f"{lines[0]}\n{entry.content}\n")
                else:
                    new_sections.append(f"## {section}")

            mem_path.write_text("".join(new_sections), encoding="utf-8")
            logger.debug(
                "Federation memory: updated entry %s", entry.node_id,
            )
        else:
            # Entry doesn't exist — append
            with open(mem_path, "a", encoding="utf-8") as f:
                f.write(f"## {entry.node_id}\n{entry.content}\n")
            logger.info(
                "Federation memory: added remote entry %s", entry.node_id,
            )

    def set_on_apply(self, callback: Callable[[MemoryEntry], None]) -> None:
        """Set callback for when a remote memory is applied locally."""
        self._on_apply = callback

    @property
    def entry_count(self) -> int:
        """Number of synced memory entries."""
        return len(self._local_memories)


# ========================================================================
# Distributed search
# ========================================================================

@dataclass
class SearchResult:
    """A single search result from a peer device."""

    device_id: str
    session_id: int
    session_title: str
    snippet: str
    score: float = 0.0
    timestamp: float = 0.0


class FederationDistributedSearch:
    """Query session_search across all federation peers.

    Usage:
        search = FederationDistributedSearch(device_id="my-device", adapter=...)
        results = await search.search("auth refactor", limit=5)
    """

    def __init__(
        self,
        device_id: str,
        adapter: Any,  # FederationAdapter
        request_timeout: float = 10.0,
    ):
        self.device_id = device_id
        self.adapter = adapter
        self.request_timeout = request_timeout
        self._pending_queries: Dict[str, Dict[str, list]] = {}  # query_id -> {device_id: [results]}
        self._running = False

    async def start(self) -> None:
        """Start distributed search — register handler."""
        self._running = True
        logger.info("Federation distributed search: started")

    async def stop(self) -> None:
        """Stop distributed search."""
        self._running = False

    async def search(
        self,
        query: str,
        limit: int = 10,
        sort: str = "newest",
        profile: Optional[str] = None,
    ) -> List[SearchResult]:
        """Execute a distributed search across all peers.

        Returns aggregated results from all devices, deduplicated and sorted.
        """
        import uuid
        query_id = str(uuid.uuid4())[:8]
        peer_count = self.adapter.get_peer_count()

        if peer_count == 0:
            logger.info("Federation search: no peers connected, local only")
            return []

        # Initialize query collector
        self._pending_queries[query_id] = {}

        # Broadcast search request to all peers
        msg = FedMessage(
            msg_type=MessageType.SEARCH_QUERY.value,
            sender_id=self.device_id,
            payload={
                "query_id": query_id,
                "query": query,
                "limit": limit,
                "sort": sort,
                "profile": profile,
            },
        )
        await self.adapter.send(msg)

        # Wait for responses
        await asyncio.sleep(self.request_timeout)

        # Aggregate results
        all_results: list[SearchResult] = []
        seen_sessions: set[tuple[str, int]] = set()

        for device_results in self._pending_queries[query_id].values():
            for r in device_results:
                key = (r.get("source", ""), r.get("session_id", 0))
                if key not in seen_sessions:
                    seen_sessions.add(key)
                    all_results.append(SearchResult(
                        device_id=r.get("device_id", "unknown"),
                        session_id=r.get("session_id", 0),
                        session_title=r.get("session_title", ""),
                        snippet=r.get("snippet", ""),
                        score=r.get("score", 0.0),
                        timestamp=r.get("timestamp", 0.0),
                    ))

        # Sort by score descending, then by timestamp
        all_results.sort(key=lambda r: (-r.score, -r.timestamp))

        # Cleanup
        del self._pending_queries[query_id]

        logger.info(
            "Federation search: query '%s' returned %d results from %d devices",
            query, len(all_results), len(self._pending_queries.get(query_id, {})),
        )

        return all_results[:limit]

    async def handle_search_query(self, msg: FedMessage) -> None:
        """Handle incoming SEARCH_QUERY — execute local search and respond."""
        query_id = msg.payload.get("query_id", "")
        query = msg.payload.get("query", "")
        limit = msg.payload.get("limit", 10)
        sort = msg.payload.get("sort", "newest")
        profile = msg.payload.get("profile")

        if not query_id or not query:
            return

        # Execute local search
        results = await self._execute_local_search(query, limit, sort, profile)

        # Send results back to requester
        response = FedMessage(
            msg_type=MessageType.SEARCH_RESULT.value,
            sender_id=self.device_id,
            target_id=msg.sender_id,
            payload={
                "query_id": query_id,
                "device_id": self.device_id,
                "results": results,
                "result_count": len(results),
            },
        )
        await self.adapter.send(response)
        logger.debug(
            "Federation search: responded to query %s with %d results",
            query_id, len(results),
        )

    async def handle_search_result(self, msg: FedMessage) -> None:
        """Handle incoming SEARCH_RESULT — collect results."""
        query_id = msg.payload.get("query_id", "")
        if query_id and query_id in self._pending_queries:
            device_id = msg.payload.get("device_id", "unknown")
            results = msg.payload.get("results", [])
            self._pending_queries[query_id][device_id] = results

    async def _execute_local_search(
        self,
        query: str,
        limit: int,
        sort: str,
        profile: Optional[str],
    ) -> list[dict]:
        """Execute session_search locally and return results."""
        try:
            from tools.session_search_tool import session_search

            kwargs = {"query": query, "limit": limit}
            if profile:
                kwargs["profile"] = profile
            if sort == "newest":
                kwargs["sort"] = "newest"
            elif sort == "oldest":
                kwargs["sort"] = "oldest"
            result = session_search(**kwargs)

            if not isinstance(result, dict) or "sessions" not in result:
                return []
            sessions = result["sessions"]
            if not isinstance(sessions, list):
                return []

            results = []
            for session in sessions:
                if not isinstance(session, dict):
                    continue
                snippet = session.get("snippet", "") or ""
                if not snippet and session.get("bookend_start"):
                    snippet = str(session["bookend_start"])[:200]

                results.append({
                    "device_id": self.device_id,
                    "session_id": session.get("session_id", 0),
                    "session_title": session.get("title", ""),
                    "snippet": snippet[:200] if snippet else "",
                    "score": 1.0,
                    "timestamp": session.get("when", 0.0),
                    "source": session.get("source", "local"),
                })

            return results

        except Exception as e:
            logger.warning("Federation search: local search failed: %s", e)
            return []

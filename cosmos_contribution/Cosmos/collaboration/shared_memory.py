"""
cosmos Shared Memory Pool - Team Knowledge Bases

Novel Approaches:
1. Namespace Isolation - Separate team and personal memories
2. Merge Strategies - Intelligent conflict resolution
3. Sync Protocols - Real-time memory synchronization
4. Access Logging - Audit trail for compliance
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Any, Callable
import json
import hashlib

from loguru import logger


class MemoryPermission(Enum):
    """Permission levels for memory access."""
    NONE = 0
    READ = 1
    WRITE = 2
    ADMIN = 3


class MergeStrategy(Enum):
    """Strategies for merging conflicting memories."""
    SKIP = "skip"           # Keep existing
    OVERWRITE = "overwrite" # Replace with new
    KEEP_NEWER = "keep_newer"
    KEEP_BOTH = "keep_both"
    MERGE_CONTENT = "merge_content"


@dataclass
class MemoryAccess:
    """Record of memory access."""
    user_id: str
    memory_id: str
    action: str  # "read", "write", "delete"
    timestamp: datetime = field(default_factory=datetime.now)
    details: dict = field(default_factory=dict)


@dataclass
class SharedMemory:
    """A memory item in the shared pool."""
    id: str
    content: str
    namespace: str  # Team or pool name

    # Ownership
    created_by: str
    created_at: datetime = field(default_factory=datetime.now)
    modified_by: Optional[str] = None
    modified_at: Optional[datetime] = None

    # Metadata
    tags: list[str] = field(default_factory=list)
    embedding: Optional[list[float]] = None
    metadata: dict = field(default_factory=dict)

    # Versioning
    version: int = 1
    previous_versions: list[dict] = field(default_factory=list)

    # Access control
    permission_overrides: dict[str, MemoryPermission] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "namespace": self.namespace,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat(),
            "tags": self.tags,
            "version": self.version,
        }


@dataclass
class MemoryPool:
    """A shared memory pool for a team."""
    id: str
    name: str
    description: str = ""

    # Members
    owner_id: str = ""
    member_permissions: dict[str, MemoryPermission] = field(default_factory=dict)

    # Settings
    default_permission: MemoryPermission = MemoryPermission.READ
    merge_strategy: MergeStrategy = MergeStrategy.KEEP_NEWER
    max_memories: int = 10000

    # Stats
    memory_count: int = 0
    total_size_bytes: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    last_modified: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "owner": self.owner_id,
            "members": len(self.member_permissions),
            "memory_count": self.memory_count,
        }


class SharedMemoryPool:
    """
    Shared memory pool for team collaboration.

    Features:
    - Multiple isolated namespaces
    - Fine-grained access control
    - Real-time synchronization
    - Conflict resolution
    """

    def __init__(
        self,
        embed_fn: Optional[Callable] = None,
        storage_backend: Optional[Any] = None,
    ):
        self.embed_fn = embed_fn
        self.storage = storage_backend

        self.pools: dict[str, MemoryPool] = {}
        self.memories: dict[str, dict[str, SharedMemory]] = {}  # pool_id -> {memory_id -> memory}
        self.access_log: list[MemoryAccess] = []

        self._subscribers: dict[str, list[Callable]] = {}
        self._lock = asyncio.Lock()

    async def create_pool(
        self,
        name: str,
        owner_id: str,
        description: str = "",
        default_permission: MemoryPermission = MemoryPermission.READ,
    ) -> MemoryPool:
        """Create a new shared memory pool."""
        pool_id = hashlib.sha256(f"{name}{owner_id}{datetime.now().isoformat()}".encode()).hexdigest()[:16]

        pool = MemoryPool(
            id=pool_id,
            name=name,
            description=description,
            owner_id=owner_id,
            default_permission=default_permission,
        )

        pool.member_permissions[owner_id] = MemoryPermission.ADMIN

        async with self._lock:
            self.pools[pool_id] = pool
            self.memories[pool_id] = {}

        logger.info(f"Created memory pool: {name} ({pool_id})")
        return pool

    async def add_member(
        self,
        pool_id: str,
        user_id: str,
        permission: MemoryPermission,
        added_by: str,
    ) -> bool:
        """Add a member to a pool."""
        if pool_id not in self.pools:
            return False

        pool = self.pools[pool_id]

        # Check admin permission
        if pool.member_permissions.get(added_by) != MemoryPermission.ADMIN:
            if added_by != pool.owner_id:
                return False

        async with self._lock:
            pool.member_permissions[user_id] = permission

        logger.info(f"Added {user_id} to pool {pool_id} with {permission.name}")
        return True

    async def remove_member(
        self,
        pool_id: str,
        user_id: str,
        removed_by: str,
    ) -> bool:
        """Remove a member from a pool."""
        if pool_id not in self.pools:
            return False

        pool = self.pools[pool_id]

        # Check permission
        if pool.member_permissions.get(removed_by) != MemoryPermission.ADMIN:
            if removed_by != pool.owner_id:
                return False

        # Can't remove owner
        if user_id == pool.owner_id:
            return False

        async with self._lock:
            if user_id in pool.member_permissions:
                del pool.member_permissions[user_id]

        return True

    async def check_permission(
        self,
        pool_id: str,
        user_id: str,
        required: MemoryPermission,
        memory_id: Optional[str] = None,
    ) -> bool:
        """Check if user has required permission."""
        if pool_id not in self.pools:
            return False

        pool = self.pools[pool_id]

        # Check memory-specific override
        if memory_id and memory_id in self.memories.get(pool_id, {}):
            memory = self.memories[pool_id][memory_id]
            if user_id in memory.permission_overrides:
                return memory.permission_overrides[user_id].value >= required.value

        # Check pool permission
        user_permission = pool.member_permissions.get(user_id, pool.default_permission)

        return user_permission.value >= required.value

    async def store(
        self,
        pool_id: str,
        content: str,
        user_id: str,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict] = None,
    ) -> Optional[SharedMemory]:
        """Store a memory in a shared pool."""
        if not await self.check_permission(pool_id, user_id, MemoryPermission.WRITE):
            logger.warning(f"Permission denied: {user_id} cannot write to {pool_id}")
            return None

        memory_id = hashlib.sha256(f"{content[:50]}{datetime.now().isoformat()}".encode()).hexdigest()[:16]

        # Get embedding
        embedding = None
        if self.embed_fn:
            try:
                if asyncio.iscoroutinefunction(self.embed_fn):
                    embedding = await self.embed_fn(content)
                else:
                    embedding = self.embed_fn(content)
            except Exception as e:
                logger.warning(f"Embedding failed: {e}")

        memory = SharedMemory(
            id=memory_id,
            content=content,
            namespace=pool_id,
            created_by=user_id,
            tags=tags or [],
            embedding=embedding,
            metadata=metadata or {},
        )

        async with self._lock:
            if pool_id not in self.memories:
                self.memories[pool_id] = {}

            self.memories[pool_id][memory_id] = memory
            self.pools[pool_id].memory_count += 1
            self.pools[pool_id].total_size_bytes += len(content.encode())
            self.pools[pool_id].last_modified = datetime.now()

        # Log access
        self._log_access(user_id, memory_id, "write", {"pool_id": pool_id})

        # Notify subscribers
        await self._notify_subscribers(pool_id, "store", memory)

        return memory

    async def retrieve(
        self,
        pool_id: str,
        memory_id: str,
        user_id: str,
    ) -> Optional[SharedMemory]:
        """Retrieve a specific memory."""
        if not await self.check_permission(pool_id, user_id, MemoryPermission.READ):
            return None

        if pool_id not in self.memories:
            return None

        memory = self.memories[pool_id].get(memory_id)

        if memory:
            self._log_access(user_id, memory_id, "read", {"pool_id": pool_id})

        return memory

    async def search(
        self,
        pool_id: str,
        query: str,
        user_id: str,
        top_k: int = 10,
        filter_tags: Optional[list[str]] = None,
    ) -> list[SharedMemory]:
        """Search memories in a pool."""
        if not await self.check_permission(pool_id, user_id, MemoryPermission.READ):
            return []

        if pool_id not in self.memories:
            return []

        memories = list(self.memories[pool_id].values())

        # Filter by tags
        if filter_tags:
            memories = [m for m in memories if any(t in m.tags for t in filter_tags)]

        # Score by query match
        query_lower = query.lower()
        scored = []

        for memory in memories:
            score = 0.0

            # Content match
            if query_lower in memory.content.lower():
                score += 0.5

            # Tag match
            for tag in memory.tags:
                if query_lower in tag.lower():
                    score += 0.2

            if score > 0:
                scored.append((memory, score))

        # Sort by score
        scored.sort(key=lambda x: x[1], reverse=True)

        self._log_access(user_id, "", "search", {"pool_id": pool_id, "query": query})

        return [m for m, _ in scored[:top_k]]

    async def update(
        self,
        pool_id: str,
        memory_id: str,
        user_id: str,
        content: Optional[str] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict] = None,
    ) -> Optional[SharedMemory]:
        """Update a memory."""
        if not await self.check_permission(pool_id, user_id, MemoryPermission.WRITE, memory_id):
            return None

        if pool_id not in self.memories or memory_id not in self.memories[pool_id]:
            return None

        async with self._lock:
            memory = self.memories[pool_id][memory_id]

            # Save version
            memory.previous_versions.append({
                "content": memory.content,
                "modified_by": memory.modified_by,
                "modified_at": memory.modified_at.isoformat() if memory.modified_at else None,
                "version": memory.version,
            })

            # Update fields
            if content is not None:
                memory.content = content
            if tags is not None:
                memory.tags = tags
            if metadata is not None:
                memory.metadata.update(metadata)

            memory.modified_by = user_id
            memory.modified_at = datetime.now()
            memory.version += 1

        self._log_access(user_id, memory_id, "update", {"pool_id": pool_id})

        await self._notify_subscribers(pool_id, "update", memory)

        return memory

    async def delete(
        self,
        pool_id: str,
        memory_id: str,
        user_id: str,
    ) -> bool:
        """Delete a memory."""
        if not await self.check_permission(pool_id, user_id, MemoryPermission.WRITE, memory_id):
            return False

        if pool_id not in self.memories or memory_id not in self.memories[pool_id]:
            return False

        async with self._lock:
            memory = self.memories[pool_id][memory_id]
            del self.memories[pool_id][memory_id]
            self.pools[pool_id].memory_count -= 1
            self.pools[pool_id].total_size_bytes -= len(memory.content.encode())

        self._log_access(user_id, memory_id, "delete", {"pool_id": pool_id})

        await self._notify_subscribers(pool_id, "delete", {"memory_id": memory_id})

        return True

    async def merge_pools(
        self,
        source_pool_id: str,
        target_pool_id: str,
        user_id: str,
        strategy: MergeStrategy = MergeStrategy.KEEP_NEWER,
    ) -> dict:
        """Merge memories from source pool into target."""
        # Check permissions
        if not await self.check_permission(source_pool_id, user_id, MemoryPermission.READ):
            return {"success": False, "error": "No read access to source"}

        if not await self.check_permission(target_pool_id, user_id, MemoryPermission.WRITE):
            return {"success": False, "error": "No write access to target"}

        stats = {"added": 0, "updated": 0, "skipped": 0, "conflicts": 0}

        source_memories = self.memories.get(source_pool_id, {})
        target_memories = self.memories.get(target_pool_id, {})

        for memory_id, memory in source_memories.items():
            if memory_id in target_memories:
                # Conflict
                stats["conflicts"] += 1
                target = target_memories[memory_id]

                if strategy == MergeStrategy.SKIP:
                    stats["skipped"] += 1
                    continue

                elif strategy == MergeStrategy.OVERWRITE:
                    await self.update(
                        target_pool_id, memory_id, user_id,
                        content=memory.content,
                        tags=memory.tags,
                        metadata=memory.metadata,
                    )
                    stats["updated"] += 1

                elif strategy == MergeStrategy.KEEP_NEWER:
                    if memory.modified_at and target.modified_at:
                        if memory.modified_at > target.modified_at:
                            await self.update(
                                target_pool_id, memory_id, user_id,
                                content=memory.content,
                            )
                            stats["updated"] += 1
                        else:
                            stats["skipped"] += 1
                    else:
                        stats["skipped"] += 1

                elif strategy == MergeStrategy.KEEP_BOTH:
                    # Create new memory with different ID
                    await self.store(
                        target_pool_id,
                        memory.content,
                        user_id,
                        tags=memory.tags + ["merged"],
                        metadata={**memory.metadata, "source_id": memory_id},
                    )
                    stats["added"] += 1

            else:
                # No conflict, add new
                new_memory = SharedMemory(
                    id=memory_id,
                    content=memory.content,
                    namespace=target_pool_id,
                    created_by=user_id,
                    tags=memory.tags,
                    embedding=memory.embedding,
                    metadata=memory.metadata,
                )

                async with self._lock:
                    self.memories[target_pool_id][memory_id] = new_memory
                    self.pools[target_pool_id].memory_count += 1

                stats["added"] += 1

        return {"success": True, "stats": stats}

    def subscribe(
        self,
        pool_id: str,
        callback: Callable,
    ):
        """Subscribe to pool changes."""
        if pool_id not in self._subscribers:
            self._subscribers[pool_id] = []
        self._subscribers[pool_id].append(callback)

    def unsubscribe(
        self,
        pool_id: str,
        callback: Callable,
    ):
        """Unsubscribe from pool changes."""
        if pool_id in self._subscribers:
            self._subscribers[pool_id] = [
                c for c in self._subscribers[pool_id] if c != callback
            ]

    async def _notify_subscribers(
        self,
        pool_id: str,
        event: str,
        data: Any,
    ):
        """Notify subscribers of pool changes."""
        if pool_id not in self._subscribers:
            return

        for callback in self._subscribers[pool_id]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event, data)
                else:
                    callback(event, data)
            except Exception as e:
                logger.error(f"Subscriber callback failed: {e}")

    def _log_access(
        self,
        user_id: str,
        memory_id: str,
        action: str,
        details: dict,
    ):
        """Log memory access."""
        access = MemoryAccess(
            user_id=user_id,
            memory_id=memory_id,
            action=action,
            details=details,
        )
        self.access_log.append(access)

        # Keep log bounded
        if len(self.access_log) > 10000:
            self.access_log = self.access_log[-5000:]

    async def get_access_log(
        self,
        pool_id: str,
        user_id: str,
        limit: int = 100,
    ) -> list[MemoryAccess]:
        """Get access log for a pool (requires admin)."""
        if not await self.check_permission(pool_id, user_id, MemoryPermission.ADMIN):
            return []

        return [
            a for a in self.access_log
            if a.details.get("pool_id") == pool_id
        ][-limit:]

    def get_stats(self) -> dict:
        """Get shared memory statistics."""
        return {
            "total_pools": len(self.pools),
            "total_memories": sum(len(m) for m in self.memories.values()),
            "total_access_logs": len(self.access_log),
            "pools": {
                pid: pool.to_dict()
                for pid, pool in self.pools.items()
            },
        }

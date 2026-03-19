"""
cosmos Sharding Module - Distributed Memory Interface

Provides interfaces for sharded memory storage to support:
- Horizontal scaling
- Distributed memory shards
- Load balancing
"""

import hashlib
from typing import Any, Protocol, runtime_checkable

@runtime_checkable
class Shardable(Protocol):
    """Protocol for components that can be sharded."""
    async def get_shard(self, key: str) -> str: ...

class ShardManager:
    """
    Manages distribution of keys across multiple shards.
    
    Current implementation: Directory-based sharding (local).
    Future: Network-based sharding (distributed).
    """

    def __init__(self, num_shards: int = 4):
        self.num_shards = num_shards

    def get_shard_id(self, key: str) -> int:
        """Determines which shard a key belongs to using consistent hashing."""
        hash_val = int(hashlib.sha256(key.encode()).hexdigest(), 16)
        return hash_val % self.num_shards

    def get_shard_path(self, base_path: str, shard_id: int) -> str:
        """Get the physical path for a shard ID."""
        return f"{base_path}/shard_{shard_id}"

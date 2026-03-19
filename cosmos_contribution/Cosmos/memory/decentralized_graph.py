"""
COSMOS DECENTRALIZED MEMORY GRAPH
=================================

A reimagined memory architecture that transcends centralized dict storage.
This implements local sharding and dynamic cache invalidation across the Swarm.

Features:
1. Node-Edge Graph Architecture
2. Local Sharding per Agent
3. Dynamic Timestamp Cache Invalidation
4. Critical Metadata P2P Synchronization
"""

import asyncio
import hashlib
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger

@dataclass
class MemoryEdge:
    """A semantic relationship between two memory nodes."""
    target_node_id: str
    relationship_type: str  # e.g., "depends_on", "relates_to", "solves"
    weight: float = 1.0


@dataclass
class MemoryNode:
    """A single fragment of memory in the decentralized graph."""
    id: str
    content: str
    owner_agent: str
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    version: int = 1
    
    # Context/Embeddings
    tags: Set[str] = field(default_factory=set)
    embedding: Optional[List[float]] = None
    
    # Graph Topology
    edges: List[MemoryEdge] = field(default_factory=list)
    
    def get_metadata(self) -> Dict[str, Any]:
        """Provides lightweight metadata for P2P synchronization."""
        return {
            "id": self.id,
            "owner": self.owner_agent,
            "version": self.version,
            "updated_at": self.updated_at.isoformat(),
            "tags": list(self.tags)
        }


class DecentralizedGraph:
    """
    Local Shard of the Global Memory Graph.
    Each agent instantiates this to hold their local memories and proxy global metadata.
    """

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        
        # Local Shard (Actual Content)
        self.local_nodes: Dict[str, MemoryNode] = {}
        
        # Global Registry (Metadata Only - for Cache Invalidation)
        self.global_metadata_cache: Dict[str, Dict[str, Any]] = {}
        
        self._lock = asyncio.Lock()
        logger.info(f"[MEMORY GRAPH] Local Shard initialized for Agent: {agent_id}")

    async def store_node(self, content: str, tags: List[str] = None) -> MemoryNode:
        """Store a new memory node in the local shard."""
        node_id = hashlib.blake2b(f"{content[:100]}{datetime.now().isoformat()}".encode(), digest_size=16).hexdigest()
        
        node = MemoryNode(
            id=node_id,
            content=content,
            owner_agent=self.agent_id,
            tags=set(tags or [])
        )
        
        async with self._lock:
            self.local_nodes[node_id] = node
            # Instantly update local cache of global metadata
            self.global_metadata_cache[node_id] = node.get_metadata()
            
        logger.debug(f"[MEMORY GRAPH] Node {node_id} stored locally by {self.agent_id}")
        return node
        
    async def ingest_metadata_sync(self, sync_payloads: List[Dict[str, Any]]):
        """
        Ingest a P2P broadcast of metadata to update the local cache.
        If a remote node's version > local cache version, local cache is invalidated/updated.
        """
        invalidations = 0
        async with self._lock:
            for payload in sync_payloads:
                node_id = payload["id"]
                remote_version = payload["version"]
                
                local_meta = self.global_metadata_cache.get(node_id)
                if not local_meta or remote_version > local_meta.get("version", 0):
                    self.global_metadata_cache[node_id] = payload
                    invalidations += 1
                    
                    # If we held the actual content locally (because we mirrored it previously),
                    # it is now inherently stale. A fetch requires re-download.
                    if node_id in self.local_nodes and self.local_nodes[node_id].owner_agent != self.agent_id:
                        del self.local_nodes[node_id]
                        
        if invalidations > 0:
            logger.debug(f"[MEMORY GRAPH] {self.agent_id} processed {invalidations} cache invalidations.")

    async def link_nodes(self, source_id: str, target_id: str, relationship: str, weight: float = 1.0):
        """Create a semantic edge between nodes."""
        async with self._lock:
            if source_id in self.local_nodes:
                self.local_nodes[source_id].edges.append(MemoryEdge(target_id, relationship, weight))
                self.local_nodes[source_id].updated_at = datetime.now()
                self.local_nodes[source_id].version += 1
                return True
        return False

    async def query_local_graph(self, query_tags: List[str]) -> List[MemoryNode]:
        """Query local shard based on tags."""
        results = []
        async with self._lock:
            for node in self.local_nodes.values():
                if any(tag in node.tags for tag in query_tags):
                    results.append(node)
        return results

    def get_sync_payload(self) -> List[Dict[str, Any]]:
        """Generate payload of local nodes to broadcast to the P2P unified router."""
        return [node.get_metadata() for node in self.local_nodes.values()]


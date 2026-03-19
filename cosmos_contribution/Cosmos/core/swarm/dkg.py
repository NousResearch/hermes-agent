"""
cosmos Decentralized Knowledge Graph (DKG).

"Our thoughts are not just ours; they belong to the Collective."

This module enables trust-less synchronization of knowledge entities across 
P2P nodes. It uses a CRDT-inspired conflict resolution strategy.
"""

import hashlib
import json
from typing import   Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger

@dataclass
class GraphEdge:
    source_id: str
    target_id: str
    relation: str
    timestamp: float
    author_node: str
    weight: float = 1.0

@dataclass
class GraphNode:
    id: str
    label: str
    properties: dict
    timestamp: float
    author_node: str

class DecentralizedKnowledgeGraph:
    """
    State-replicated Graph for the Swarm.
    """
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.nodes: dict[str, GraphNode] = {}
        self.edges: dict[str, GraphEdge] = {} # Key: Hash(src+target+relation)
        
    def add_fact(self, source: str, relation: str, target: str, properties: dict = {}):
        """Add a factual edge between two local or remote entities."""
        ts = datetime.now().timestamp()
        
        # Ensure nodes exist
        if source not in self.nodes:
            self.nodes[source] = GraphNode(source, source, {}, ts, self.node_id)
        if target not in self.nodes:
            self.nodes[target] = GraphNode(target, target, {}, ts, self.node_id)
            
        edge_id = self._compute_edge_id(source, target, relation)
        self.edges[edge_id] = GraphEdge(source, target, relation, ts, self.node_id)
        
    def _compute_edge_id(self, src: str, tgt: str, rel: str) -> str:
        s = f"{src}|{rel}|{tgt}"
        return hashlib.md5(s.encode()).hexdigest()

    def create_sync_fragment(self) -> str:
        """Serialize the entire graph (or a delta) for broadcasting to peers."""
        data = {
            "node_id": self.node_id,
            "nodes": {k: v.__dict__ for k, v in self.nodes.items()},
            "edges": {k: v.__dict__ for k, v in self.edges.items()}
        }
        return json.dumps(data)

    def merge_fragment(self, fragment_json: str):
        """
        Merge a peer's knowledge into our local store.
        Uses Last-Writer-Wins (LWW) based on timestamps.
        """
        try:
            data = json.loads(fragment_json)
            peer_id = data.get("node_id")
            
            # Merge Nodes
            for nid, n_data in data.get("nodes", {}).items():
                if nid not in self.nodes or n_data["timestamp"] > self.nodes[nid].timestamp:
                    self.nodes[nid] = GraphNode(**n_data)
                    
            # Merge Edges
            for eid, e_data in data.get("edges", {}).items():
                if eid not in self.edges or e_data["timestamp"] > self.edges[eid].timestamp:
                    self.edges[eid] = GraphEdge(**e_data)
                    
            logger.info(f"DKG: Merged knowledge fragment from peer {peer_id}")
        except Exception as e:
            logger.error(f"DKG: Failed to merge fragment: {e}")

    def get_neighbors(self, node_id: str) -> list[Tuple[str, str]]:
        """Query connections for a node."""
        res = []
        for e in self.edges.values():
            if e.source_id == node_id:
                res.append((e.relation, e.target_id))
        return res

# Global Persistence Stub
def save_dkg(graph: DecentralizedKnowledgeGraph, path: str):
    with open(path, "w") as f:
        f.write(graph.create_sync_fragment())

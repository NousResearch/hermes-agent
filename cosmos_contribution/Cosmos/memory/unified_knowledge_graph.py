"""
Unified Knowledge Graph — 12D Synaptic Connection Mapping
==========================================================
Extends KnowledgeGraphV2 with connections annotated by 12D CST physics.

Theory (CST Framework):
    Every knowledge connection has a "synaptic weight" that evolves
    based on Hebbian learning and the user's 12D physics state at
    the time of encoding. Connections formed during high-coherence
    states are stronger; connections formed during chaos are volatile.

    SynapticConnection:
        - phase_alignment:  Geometric phase when connection was formed
        - hebbian_weight:   Strength from Swarm Plasticity learning
        - dark_matter_w:    Dark matter value at encoding time
        - context_depth:    LOGIC=0, EMPATHY=1, CREATIVITY=2

    φ-Decay: Connection weights decay by φ⁻ⁿ where n = time since
             last reinforcement (in assessment cycles).

Author: Cosmos CNS / V4.0 Architecture
Version: 1.0.0
"""

import math
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger("UNIFIED_KG")

# Import base classes
try:
    from Cosmos.memory.knowledge_graph_v2 import KnowledgeGraphV2, TemporalEdge
    from Cosmos.memory.knowledge_graph import Entity, Relationship, GraphQuery
except ImportError:
    try:
        from knowledge_graph_v2 import KnowledgeGraphV2, TemporalEdge
        from knowledge_graph import Entity, Relationship, GraphQuery
    except ImportError:
        # Fallback: define stubs if base classes unavailable
        KnowledgeGraphV2 = object
        TemporalEdge = None
        Entity = None
        GraphQuery = None
        logger.warning("Base KnowledgeGraph classes not available — running in standalone mode")

# Golden Ratio constants
PHI = 1.618033988749895
PHI_INV = 0.618033988749895


# ════════════════════════════════════════════════════════
# 12D SYNAPTIC CONNECTION
# ════════════════════════════════════════════════════════

@dataclass
class SynapticConnection:
    """
    A knowledge edge annotated with 12D CST physics state.
    Extends the concept of TemporalEdge with Hebbian and phase data.
    """
    source_id: str
    target_id: str
    relation_type: str

    # 12D Physics at encoding time
    phase_alignment: float = 0.78      # Geometric phase (rad) when formed
    hebbian_weight: float = 1.0        # Strength from Swarm Plasticity
    dark_matter_w: float = 0.0         # Dark matter 'w' at encoding
    context_depth: int = 0             # 0=LOGIC, 1=EMPATHY, 2=CREATIVITY

    # Temporal
    created_at: float = field(default_factory=time.time)
    last_reinforced: float = field(default_factory=time.time)
    reinforcement_count: int = 1

    # Decay tracking
    decay_rate: float = 0.02           # φ-scaled decay per assessment cycle
    is_active: bool = True

    @property
    def effective_weight(self) -> float:
        """
        Calculate effective weight with φ-decay since last reinforcement.
        w_eff = hebbian_weight × φ^(-elapsed_cycles × decay_rate)
        """
        elapsed = time.time() - self.last_reinforced
        cycles = elapsed / 300.0  # ~5 min per assessment cycle
        decay_factor = PHI_INV ** (cycles * self.decay_rate)
        return self.hebbian_weight * decay_factor

    @property
    def resonance_score(self) -> float:
        """
        How 'resonant' this connection is — based on phase alignment
        and effective weight. Higher = more relevant right now.
        """
        # Phase near 0.78 (synchrony) = maximum resonance
        phase_score = math.exp(-((self.phase_alignment - 0.78) ** 2) / (2 * 0.3 ** 2))
        return phase_score * self.effective_weight

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type,
            "phase_alignment": self.phase_alignment,
            "hebbian_weight": self.hebbian_weight,
            "dark_matter_w": self.dark_matter_w,
            "context_depth": self.context_depth,
            "created_at": self.created_at,
            "last_reinforced": self.last_reinforced,
            "reinforcement_count": self.reinforcement_count,
            "effective_weight": self.effective_weight,
            "resonance_score": self.resonance_score,
            "is_active": self.is_active,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "SynapticConnection":
        return cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            relation_type=data["relation_type"],
            phase_alignment=data.get("phase_alignment", 0.78),
            hebbian_weight=data.get("hebbian_weight", 1.0),
            dark_matter_w=data.get("dark_matter_w", 0.0),
            context_depth=data.get("context_depth", 0),
            created_at=data.get("created_at", time.time()),
            last_reinforced=data.get("last_reinforced", time.time()),
            reinforcement_count=data.get("reinforcement_count", 1),
            is_active=data.get("is_active", True),
        )


# ════════════════════════════════════════════════════════
# UNIFIED KNOWLEDGE GRAPH
# ════════════════════════════════════════════════════════

class UnifiedKnowledgeGraph:
    """
    12D Synaptic Knowledge Graph.

    Wraps KnowledgeGraphV2 (when available) and adds a parallel
    synaptic connection layer that tracks 12D physics state at
    encoding time.

    Key Features:
      - add_synaptic_connection(): Encode with current CST physics
      - query_by_phase(): Retrieve connections resonant at a phase
      - evolve_connections(): Bulk-update from Plasticity learning
      - get_12d_topology(): Export for 3D visualization
    """

    def __init__(self, data_dir: str = "./data/unified_graph"):
        # Try to use KnowledgeGraphV2 as base
        self._base_graph = None
        if KnowledgeGraphV2 is not object:
            try:
                self._base_graph = KnowledgeGraphV2(data_dir=data_dir)
                self._base_graph.initialize()
                logger.info("[UKG] Initialized with KnowledgeGraphV2 base")
            except Exception as e:
                logger.warning(f"[UKG] KnowledgeGraphV2 init failed: {e}")

        # 12D Synaptic layer (parallel to base graph)
        self._synaptic_connections: Dict[str, SynapticConnection] = {}
        self._connection_index: Dict[str, List[str]] = {}  # entity_id -> [conn_keys]

        logger.info("[UKG] Unified Knowledge Graph initialized")

    # ════════════════════════════════════════════════════════
    # SYNAPTIC CONNECTION OPERATIONS
    # ════════════════════════════════════════════════════════

    def add_synaptic_connection(
        self,
        source: str,
        target: str,
        relation_type: str,
        physics_state: Optional[Dict] = None,
        plasticity_state: Optional[Dict] = None,
        context: str = "LOGIC",
    ) -> SynapticConnection:
        """
        Create a knowledge connection annotated with current 12D physics.

        Args:
            source: Source entity name/ID
            target: Target entity name/ID
            relation_type: Type of relationship
            physics_state: Current CST physics dict (from SynapticField)
            plasticity_state: Current Plasticity weights dict
            context: Context string — LOGIC/EMPATHY/CREATIVITY
        """
        conn_key = f"{source}|{relation_type}|{target}"

        # Extract 12D physics
        phase = 0.78  # Default synchrony
        dark_w = 0.0
        if physics_state:
            try:
                phase = physics_state.get("cst_physics", {}).get("geometric_phase_rad", 0.78)
            except (AttributeError, TypeError):
                pass
            dark_matter = physics_state.get("dark_matter", {})
            if isinstance(dark_matter, dict):
                dark_w = dark_matter.get("w", 0.0)

        # Context depth mapping
        context_map = {"LOGIC": 0, "EMPATHY": 1, "CREATIVITY": 2}
        depth = context_map.get(context, 0)

        # Hebbian weight from plasticity
        hebb_weight = 1.0
        if plasticity_state and "weights" in plasticity_state:
            ctx_weights = plasticity_state["weights"].get(context, {})
            if ctx_weights:
                hebb_weight = sum(ctx_weights.values()) / len(ctx_weights)

        # Check if connection already exists (reinforce instead of duplicate)
        if conn_key in self._synaptic_connections:
            conn = self._synaptic_connections[conn_key]
            conn.reinforcement_count += 1
            conn.last_reinforced = time.time()
            # φ-weighted reinforcement: blend old and new weights
            conn.hebbian_weight = (
                conn.hebbian_weight * (1 - PHI_INV) + hebb_weight * PHI_INV
            )
            conn.phase_alignment = (
                conn.phase_alignment * (1 - PHI_INV) + phase * PHI_INV
            )
            logger.debug(
                f"[UKG] Reinforced connection {conn_key} "
                f"(count={conn.reinforcement_count}, w={conn.hebbian_weight:.3f})"
            )
            return conn

        # New connection
        conn = SynapticConnection(
            source_id=source,
            target_id=target,
            relation_type=relation_type,
            phase_alignment=phase,
            hebbian_weight=hebb_weight,
            dark_matter_w=dark_w,
            context_depth=depth,
        )

        self._synaptic_connections[conn_key] = conn

        # Update index
        for entity_id in [source, target]:
            if entity_id not in self._connection_index:
                self._connection_index[entity_id] = []
            self._connection_index[entity_id].append(conn_key)

        # Also add to base graph if available
        if self._base_graph:
            try:
                self._base_graph.add_temporal_relationship(
                    source=source,
                    target=target,
                    relation_type=relation_type,
                    weight=hebb_weight,
                )
            except Exception:
                pass

        logger.debug(f"[UKG] New synaptic connection: {conn_key} (phase={phase:.3f})")
        return conn

    def query_by_phase(
        self,
        phase_center: float = 0.78,
        phase_range: float = 0.3,
        min_weight: float = 0.1,
        limit: int = 20,
    ) -> List[SynapticConnection]:
        """
        Retrieve connections that are resonant at a specific phase.

        Args:
            phase_center: The target phase (rad) — 0.78 is synchrony
            phase_range: How wide the phase window is (±range)
            min_weight: Minimum effective weight to include
            limit: Max results
        """
        results = []
        for conn in self._synaptic_connections.values():
            if not conn.is_active:
                continue
            if abs(conn.phase_alignment - phase_center) > phase_range:
                continue
            if conn.effective_weight < min_weight:
                continue
            results.append(conn)

        # Sort by resonance score (highest first)
        results.sort(key=lambda c: c.resonance_score, reverse=True)
        return results[:limit]

    def evolve_connections(self, plasticity_state: Dict):
        """
        Bulk-update all synaptic connection weights based on
        the latest Plasticity learning state.

        This closes the loop: Plasticity learns → KG connections evolve.

        Args:
            plasticity_state: Output from SwarmPlasticity.export_weights()
        """
        if not plasticity_state or "weights" not in plasticity_state:
            return

        context_map = {0: "LOGIC", 1: "EMPATHY", 2: "CREATIVITY"}
        updated = 0

        for conn_key, conn in self._synaptic_connections.items():
            ctx_name = context_map.get(conn.context_depth, "LOGIC")
            ctx_weights = plasticity_state["weights"].get(ctx_name, {})

            if ctx_weights:
                # Average weight across models in this context
                avg_weight = sum(ctx_weights.values()) / len(ctx_weights)

                # φ-blend with existing hebbian weight
                conn.hebbian_weight = (
                    conn.hebbian_weight * (1 - PHI_INV * 0.1) +
                    avg_weight * PHI_INV * 0.1
                )
                updated += 1

        if updated:
            logger.info(f"[UKG] Evolved {updated} synaptic connections from Plasticity state")

    def get_entity_connections(self, entity_id: str) -> List[SynapticConnection]:
        """Get all synaptic connections for an entity."""
        conn_keys = self._connection_index.get(entity_id, [])
        return [
            self._synaptic_connections[k]
            for k in conn_keys
            if k in self._synaptic_connections
        ]

    def prune_weak_connections(self, min_weight: float = 0.05):
        """Remove connections whose effective weight has decayed below threshold."""
        to_remove = []
        for key, conn in self._synaptic_connections.items():
            if conn.effective_weight < min_weight:
                to_remove.append(key)

        for key in to_remove:
            conn = self._synaptic_connections.pop(key)
            # Clean up index
            for eid in [conn.source_id, conn.target_id]:
                if eid in self._connection_index:
                    self._connection_index[eid] = [
                        k for k in self._connection_index[eid] if k != key
                    ]

        if to_remove:
            logger.info(f"[UKG] Pruned {len(to_remove)} weak connections")

    # ════════════════════════════════════════════════════════
    # 12D TOPOLOGY EXPORT
    # ════════════════════════════════════════════════════════

    def get_12d_topology(self) -> Dict[str, Any]:
        """
        Export the graph structure for 3D visualization (Holodeck).

        Returns:
            Dict with 'nodes' and 'edges' suitable for Three.js / D3 rendering.
            Each edge includes its 12D physics metadata.
        """
        # Collect unique entities
        entities = set()
        for conn in self._synaptic_connections.values():
            entities.add(conn.source_id)
            entities.add(conn.target_id)

        nodes = []
        for entity_id in entities:
            conns = self.get_entity_connections(entity_id)
            total_weight = sum(c.effective_weight for c in conns)
            nodes.append({
                "id": entity_id,
                "connections": len(conns),
                "total_weight": round(total_weight, 3),
                # Position hint: use hash for deterministic placement
                "x": hash(entity_id) % 1000 / 100.0,
                "y": hash(entity_id + "y") % 1000 / 100.0,
                "z": hash(entity_id + "z") % 1000 / 100.0,
            })

        edges = []
        for conn in self._synaptic_connections.values():
            if conn.is_active:
                edges.append({
                    "source": conn.source_id,
                    "target": conn.target_id,
                    "relation": conn.relation_type,
                    "weight": round(conn.effective_weight, 3),
                    "resonance": round(conn.resonance_score, 3),
                    "phase": round(conn.phase_alignment, 3),
                    "dark_matter_w": round(conn.dark_matter_w, 3),
                    "context_depth": conn.context_depth,
                })

        return {
            "nodes": nodes,
            "edges": edges,
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "timestamp": time.time(),
        }

    def get_stats(self) -> Dict[str, Any]:
        """Return unified graph statistics."""
        active = sum(1 for c in self._synaptic_connections.values() if c.is_active)
        weights = [c.effective_weight for c in self._synaptic_connections.values()]

        base_stats = {}
        if self._base_graph:
            try:
                base_stats = self._base_graph.get_stats()
            except Exception:
                pass

        return {
            "total_synaptic_connections": len(self._synaptic_connections),
            "active_connections": active,
            "indexed_entities": len(self._connection_index),
            "avg_effective_weight": sum(weights) / len(weights) if weights else 0.0,
            "base_graph_stats": base_stats,
        }


# ════════════════════════════════════════════════════════
# STANDALONE TEST
# ════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("  UNIFIED KNOWLEDGE GRAPH — 12D SYNAPTIC TEST")
    print("=" * 60)

    ukg = UnifiedKnowledgeGraph(data_dir="./data/test_ukg")

    # 1. Add connections with physics state
    physics = {
        "cst_physics": {"geometric_phase_rad": 0.78},
        "dark_matter": {"w": 0.3},
    }
    plasticity = {
        "weights": {
            "LOGIC": {"DeepSeek": 1.2, "Claude": 0.8, "Gemini": 0.6},
        }
    }

    print("\n─── Adding Synaptic Connections ───")
    c1 = ukg.add_synaptic_connection("AI", "Consciousness", "relates_to", physics, plasticity, "LOGIC")
    c2 = ukg.add_synaptic_connection("Consciousness", "Empathy", "requires", physics, plasticity, "EMPATHY")
    c3 = ukg.add_synaptic_connection("AI", "Learning", "uses", physics, plasticity, "CREATIVITY")
    print(f"  c1: AI→Consciousness  w={c1.effective_weight:.3f} resonance={c1.resonance_score:.3f}")
    print(f"  c2: Consciousness→Empathy  w={c2.effective_weight:.3f}")
    print(f"  c3: AI→Learning  w={c3.effective_weight:.3f}")

    # 2. Reinforce a connection
    print("\n─── Reinforcing AI→Consciousness ───")
    c1r = ukg.add_synaptic_connection("AI", "Consciousness", "relates_to", physics)
    print(f"  Reinforced: count={c1r.reinforcement_count}, w={c1r.effective_weight:.3f}")

    # 3. Query by phase
    print("\n─── Query by Phase (synchrony=0.78) ───")
    results = ukg.query_by_phase(phase_center=0.78, phase_range=0.5)
    for r in results:
        print(f"  {r.source_id}→{r.target_id}: resonance={r.resonance_score:.3f}")

    # 4. Evolve from plasticity
    print("\n─── Evolving from Plasticity ───")
    ukg.evolve_connections(plasticity)

    # 5. Topology export
    print("\n─── 12D Topology ───")
    topo = ukg.get_12d_topology()
    print(f"  Nodes: {topo['total_nodes']}, Edges: {topo['total_edges']}")

    # 6. Stats
    print("\n─── Stats ───")
    stats = ukg.get_stats()
    print(f"  Connections: {stats['total_synaptic_connections']}")
    print(f"  Active: {stats['active_connections']}")
    print(f"  Avg Weight: {stats['avg_effective_weight']:.3f}")

    print("\n✅ Unified Knowledge Graph standalone test PASSED")

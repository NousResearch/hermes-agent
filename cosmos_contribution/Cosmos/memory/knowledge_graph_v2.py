"""
cosmos Knowledge Graph v2 - Enhanced with Temporal Edges and Entity Resolution

Q1 2025 Feature: Enhanced Knowledge Graph
- Temporal edge tracking (relationships over time)
- Automated entity resolution and merging
- 3D graph visualization data support
- Relationship evolution tracking
"""

import asyncio
import json
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Any
from collections import defaultdict
import numpy as np

from loguru import logger

# Import base classes from existing knowledge graph
from Cosmos.memory.knowledge_graph import (
    Entity, Relationship, GraphQuery, KnowledgeGraph
)


@dataclass
class TemporalEdge:
    """A relationship with temporal information."""
    source_id: str
    target_id: str
    relation_type: str

    # Temporal data
    start_time: datetime
    end_time: Optional[datetime] = None  # None = still active
    is_active: bool = True

    # Evolution tracking
    weight_history: list[tuple[datetime, float]] = field(default_factory=list)
    current_weight: float = 1.0

    # Metadata
    evidence_timeline: list[dict] = field(default_factory=list)  # Timestamped evidence
    confidence: float = 0.5
    stability_score: float = 0.5  # How stable this relationship is

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "is_active": self.is_active,
            "weight_history": [
                (t.isoformat(), w) for t, w in self.weight_history
            ],
            "current_weight": self.current_weight,
            "evidence_timeline": self.evidence_timeline,
            "confidence": self.confidence,
            "stability_score": self.stability_score,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TemporalEdge":
        weight_history = [
            (datetime.fromisoformat(t), w) for t, w in data.get("weight_history", [])
        ]
        return cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            relation_type=data["relation_type"],
            start_time=datetime.fromisoformat(data["start_time"]),
            end_time=datetime.fromisoformat(data["end_time"]) if data.get("end_time") else None,
            is_active=data.get("is_active", True),
            weight_history=weight_history,
            current_weight=data.get("current_weight", 1.0),
            evidence_timeline=data.get("evidence_timeline", []),
            confidence=data.get("confidence", 0.5),
            stability_score=data.get("stability_score", 0.5),
        )


@dataclass
class EntityCluster:
    """A cluster of entities that might be the same."""
    id: str
    canonical_id: str  # The main entity ID
    member_ids: list[str]
    similarity_scores: dict[str, float]  # member_id -> similarity to canonical
    merged_at: Optional[datetime] = None
    merge_confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "canonical_id": self.canonical_id,
            "member_ids": self.member_ids,
            "similarity_scores": self.similarity_scores,
            "merged_at": self.merged_at.isoformat() if self.merged_at else None,
            "merge_confidence": self.merge_confidence,
        }


@dataclass
class EntityResolutionCandidate:
    """A candidate pair for entity resolution."""
    entity1_id: str
    entity2_id: str
    similarity_score: float
    match_reasons: list[str]
    suggested_action: str  # "merge", "link", "ignore"


@dataclass
class TemporalQuery:
    """Query with temporal constraints."""
    query: str
    time_point: Optional[datetime] = None  # State at specific time
    time_range: Optional[tuple[datetime, datetime]] = None  # State over range
    include_inactive: bool = False  # Include ended relationships
    track_evolution: bool = False  # Include weight history


class KnowledgeGraphV2(KnowledgeGraph):
    """
    Extended knowledge graph with temporal edges and entity resolution.

    New Features:
    - Temporal relationship tracking
    - Entity resolution and merging
    - Relationship evolution analysis
    - 3D visualization data
    """

    def __init__(
        self,
        data_dir: str = "./data/graph_v2",
        max_nodes: int = 10000,
        auto_link_threshold: float = 0.75,
        resolution_threshold: float = 0.85,
    ):
        super().__init__(data_dir, max_nodes, auto_link_threshold)

        self.resolution_threshold = resolution_threshold

        # Temporal edges storage
        self.temporal_edges: dict[str, TemporalEdge] = {}  # edge_id -> TemporalEdge

        # Entity resolution
        self.entity_clusters: dict[str, EntityCluster] = {}
        self.canonical_mapping: dict[str, str] = {}  # entity_id -> canonical_id

        # Resolution candidates
        self.resolution_queue: list[EntityResolutionCandidate] = []

        # Edge indices
        self.edges_by_source: dict[str, set[str]] = defaultdict(set)
        self.edges_by_target: dict[str, set[str]] = defaultdict(set)
        self.edges_by_time: dict[str, set[str]] = defaultdict(set)  # "YYYY-MM" -> edge_ids

        self._v2_initialized = False

    async def initialize(self):
        """Initialize V2 features."""
        await super().initialize()

        if self._v2_initialized:
            return

        await self._load_temporal_data()
        await self._load_resolution_data()

        self._v2_initialized = True
        logger.info(f"Knowledge Graph V2 initialized with {len(self.temporal_edges)} temporal edges")

    async def add_temporal_relationship(
        self,
        source: str,
        target: str,
        relation_type: str,
        weight: float = 1.0,
        start_time: Optional[datetime] = None,
        evidence: Optional[str] = None,
        confidence: float = 0.5,
    ) -> TemporalEdge:
        """
        Add a relationship with temporal tracking.

        Args:
            source: Source entity name or ID
            target: Target entity name or ID
            relation_type: Type of relationship
            weight: Relationship strength
            start_time: When relationship started (default: now)
            evidence: Text evidence for the relationship
            confidence: Confidence in the relationship

        Returns:
            TemporalEdge object
        """
        async with self._lock:
            source_id = self._resolve_entity(source)
            target_id = self._resolve_entity(target)

            if not source_id or not target_id:
                raise ValueError(f"Entity not found: {source if not source_id else target}")

            # Check for existing edge
            edge_key = f"{source_id}|{target_id}|{relation_type}"
            existing_edge_id = None

            for eid, edge in self.temporal_edges.items():
                if (edge.source_id == source_id and
                    edge.target_id == target_id and
                    edge.relation_type == relation_type and
                    edge.is_active):
                    existing_edge_id = eid
                    break

            now = datetime.now()
            start = start_time or now

            if existing_edge_id:
                # Update existing edge
                edge = self.temporal_edges[existing_edge_id]
                edge.weight_history.append((now, edge.current_weight))
                edge.current_weight = (edge.current_weight + weight) / 2  # Rolling average
                edge.confidence = (edge.confidence + confidence) / 2

                if evidence:
                    edge.evidence_timeline.append({
                        "timestamp": now.isoformat(),
                        "evidence": evidence[:500],
                    })

                # Update stability score
                edge.stability_score = self._calculate_stability(edge)

                await self._save_temporal_edge(edge)
                return edge

            # Create new edge
            edge_id = f"te_{hashlib.md5(edge_key.encode()).hexdigest()[:12]}"

            edge = TemporalEdge(
                source_id=source_id,
                target_id=target_id,
                relation_type=relation_type,
                start_time=start,
                current_weight=weight,
                weight_history=[(start, weight)],
                confidence=confidence,
            )

            if evidence:
                edge.evidence_timeline.append({
                    "timestamp": start.isoformat(),
                    "evidence": evidence[:500],
                })

            self.temporal_edges[edge_id] = edge

            # Update indices
            self.edges_by_source[source_id].add(edge_id)
            self.edges_by_target[target_id].add(edge_id)
            self.edges_by_time[start.strftime("%Y-%m")].add(edge_id)

            # Also add to base graph
            await super().add_relationship(source_id, target_id, relation_type, weight, evidence)

            await self._save_temporal_edge(edge)
            return edge

    async def end_relationship(
        self,
        source: str,
        target: str,
        relation_type: str,
        end_time: Optional[datetime] = None,
    ) -> Optional[TemporalEdge]:
        """Mark a relationship as ended."""
        async with self._lock:
            source_id = self._resolve_entity(source)
            target_id = self._resolve_entity(target)

            for eid, edge in self.temporal_edges.items():
                if (edge.source_id == source_id and
                    edge.target_id == target_id and
                    edge.relation_type == relation_type and
                    edge.is_active):

                    edge.end_time = end_time or datetime.now()
                    edge.is_active = False

                    await self._save_temporal_edge(edge)
                    return edge

            return None

    async def query_at_time(
        self,
        query: str,
        time_point: datetime,
        max_entities: int = 10,
    ) -> GraphQuery:
        """Query the graph state at a specific point in time."""
        # Get base query results
        result = await super().query(query, max_entities)

        # Filter relationships to those active at time_point
        active_relationships = []

        for rel in result.relationships:
            # Find temporal edge
            for edge in self.temporal_edges.values():
                if (edge.source_id == rel.source_id and
                    edge.target_id == rel.target_id and
                    edge.relation_type == rel.relation_type):

                    # Check if active at time_point
                    if edge.start_time <= time_point:
                        if edge.end_time is None or edge.end_time >= time_point:
                            active_relationships.append(rel)
                    break

        result.relationships = active_relationships
        return result

    async def get_relationship_evolution(
        self,
        source: str,
        target: str,
        relation_type: Optional[str] = None,
    ) -> list[dict]:
        """Get the evolution of a relationship over time."""
        source_id = self._resolve_entity(source)
        target_id = self._resolve_entity(target)

        evolutions = []

        for edge in self.temporal_edges.values():
            if edge.source_id == source_id and edge.target_id == target_id:
                if relation_type is None or edge.relation_type == relation_type:
                    evolutions.append({
                        "relation_type": edge.relation_type,
                        "start_time": edge.start_time.isoformat(),
                        "end_time": edge.end_time.isoformat() if edge.end_time else None,
                        "is_active": edge.is_active,
                        "weight_history": [
                            {"time": t.isoformat(), "weight": w}
                            for t, w in edge.weight_history
                        ],
                        "current_weight": edge.current_weight,
                        "stability_score": edge.stability_score,
                    })

        return evolutions

    def _calculate_stability(self, edge: TemporalEdge) -> float:
        """Calculate how stable a relationship is over time."""
        if len(edge.weight_history) < 2:
            return 0.5

        weights = [w for _, w in edge.weight_history]

        # Calculate variance
        variance = np.var(weights)

        # Lower variance = more stable
        stability = 1.0 / (1.0 + variance)

        # Bonus for longevity
        if edge.start_time:
            days_active = (datetime.now() - edge.start_time).days
            longevity_bonus = min(0.2, days_active / 365 * 0.2)
            stability += longevity_bonus

        return min(1.0, stability)

    # Entity Resolution Methods

    async def find_resolution_candidates(
        self,
        min_similarity: Optional[float] = None,
    ) -> list[EntityResolutionCandidate]:
        """Find entities that might be the same."""
        threshold = min_similarity or self.resolution_threshold
        candidates = []

        entities_list = list(self.entities.values())

        for i, e1 in enumerate(entities_list):
            for e2 in entities_list[i + 1:]:
                similarity, reasons = await self._calculate_entity_similarity(e1, e2)

                if similarity >= threshold:
                    candidates.append(EntityResolutionCandidate(
                        entity1_id=e1.id,
                        entity2_id=e2.id,
                        similarity_score=similarity,
                        match_reasons=reasons,
                        suggested_action="merge" if similarity >= 0.95 else "link",
                    ))

        # Sort by similarity
        candidates.sort(key=lambda c: c.similarity_score, reverse=True)

        self.resolution_queue = candidates
        return candidates

    async def _calculate_entity_similarity(
        self,
        e1: Entity,
        e2: Entity,
    ) -> tuple[float, list[str]]:
        """Calculate similarity between two entities."""
        scores = []
        reasons = []

        # Name similarity
        name_sim = self._string_similarity(e1.name.lower(), e2.name.lower())
        if name_sim > 0.8:
            scores.append(name_sim)
            reasons.append(f"Similar names ({name_sim:.2f})")

        # Type match
        if e1.entity_type == e2.entity_type:
            scores.append(0.3)
            reasons.append("Same entity type")

        # Embedding similarity
        if e1.embedding and e2.embedding:
            emb_sim = self._cosine_similarity(
                np.array(e1.embedding),
                np.array(e2.embedding)
            )
            if emb_sim > 0.7:
                scores.append(emb_sim)
                reasons.append(f"Similar embeddings ({emb_sim:.2f})")

        # Property overlap
        if e1.properties and e2.properties:
            common_keys = set(e1.properties.keys()) & set(e2.properties.keys())
            if common_keys:
                prop_matches = sum(
                    1 for k in common_keys
                    if e1.properties[k] == e2.properties[k]
                )
                if prop_matches > 0:
                    scores.append(prop_matches / len(common_keys) * 0.5)
                    reasons.append(f"Matching properties ({prop_matches})")

        overall = np.mean(scores) if scores else 0.0
        return float(overall), reasons

    def _string_similarity(self, s1: str, s2: str) -> float:
        """Calculate string similarity using Levenshtein-like approach."""
        if s1 == s2:
            return 1.0

        # Simple character-level Jaccard
        set1 = set(s1)
        set2 = set(s2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union > 0 else 0.0

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    async def merge_entities(
        self,
        entity1_id: str,
        entity2_id: str,
        keep_id: Optional[str] = None,
    ) -> Entity:
        """
        Merge two entities into one.

        Args:
            entity1_id: First entity ID
            entity2_id: Second entity ID
            keep_id: Which ID to keep as canonical (default: older one)

        Returns:
            Merged entity
        """
        async with self._lock:
            e1 = self.entities.get(entity1_id)
            e2 = self.entities.get(entity2_id)

            if not e1 or not e2:
                raise ValueError("Entity not found")

            # Determine canonical
            if keep_id:
                canonical = e1 if e1.id == keep_id else e2
                secondary = e2 if e1.id == keep_id else e1
            else:
                # Keep the older one
                canonical = e1 if e1.created_at <= e2.created_at else e2
                secondary = e2 if e1.created_at <= e2.created_at else e1

            # Merge properties
            for key, value in secondary.properties.items():
                if key not in canonical.properties:
                    canonical.properties[key] = value

            # Merge mention count
            canonical.mention_count += secondary.mention_count

            # Update last mentioned
            canonical.last_mentioned = max(
                canonical.last_mentioned,
                secondary.last_mentioned
            )

            # Merge embedding (average)
            if canonical.embedding and secondary.embedding:
                canonical.embedding = (
                    (np.array(canonical.embedding) + np.array(secondary.embedding)) / 2
                ).tolist()

            # Update relationships to point to canonical
            await self._redirect_relationships(secondary.id, canonical.id)

            # Create cluster record
            cluster_id = f"cluster_{canonical.id}"
            cluster = EntityCluster(
                id=cluster_id,
                canonical_id=canonical.id,
                member_ids=[canonical.id, secondary.id],
                similarity_scores={secondary.id: 1.0},
                merged_at=datetime.now(),
                merge_confidence=1.0,
            )
            self.entity_clusters[cluster_id] = cluster
            self.canonical_mapping[secondary.id] = canonical.id

            # Remove secondary entity
            del self.entities[secondary.id]
            self.name_to_id.pop(secondary.name.lower(), None)

            await self._save_entity(canonical)
            await self._save_resolution_data()

            logger.info(f"Merged entity {secondary.id} into {canonical.id}")
            return canonical

    async def _redirect_relationships(self, old_id: str, new_id: str):
        """Redirect all relationships from old entity to new."""
        # Update temporal edges
        for edge_id, edge in list(self.temporal_edges.items()):
            if edge.source_id == old_id:
                edge.source_id = new_id
                self.edges_by_source[old_id].discard(edge_id)
                self.edges_by_source[new_id].add(edge_id)

            if edge.target_id == old_id:
                edge.target_id = new_id
                self.edges_by_target[old_id].discard(edge_id)
                self.edges_by_target[new_id].add(edge_id)

            await self._save_temporal_edge(edge)

        # Update base graph
        if self.graph is not None:
            # Get all edges involving old_id
            if old_id in self.graph:
                # Redirect outgoing edges
                for _, target, data in list(self.graph.out_edges(old_id, data=True)):
                    self.graph.add_edge(new_id, target, **data)

                # Redirect incoming edges
                for source, _, data in list(self.graph.in_edges(old_id, data=True)):
                    self.graph.add_edge(source, new_id, **data)

                self.graph.remove_node(old_id)

    def get_canonical_id(self, entity_id: str) -> str:
        """Get the canonical ID for an entity (handling merges)."""
        return self.canonical_mapping.get(entity_id, entity_id)

    # 3D Visualization Data

    def get_3d_graph_data(
        self,
        max_nodes: int = 100,
        include_inactive: bool = False,
    ) -> dict:
        """
        Get graph data formatted for 3D visualization.

        Returns data suitable for force-directed 3D graph libraries.
        """
        nodes = []
        links = []

        # Get most important entities
        sorted_entities = sorted(
            self.entities.values(),
            key=lambda e: e.mention_count,
            reverse=True
        )[:max_nodes]

        entity_ids = {e.id for e in sorted_entities}

        # Format nodes
        for entity in sorted_entities:
            nodes.append({
                "id": entity.id,
                "name": entity.name,
                "type": entity.entity_type,
                "size": min(20, 5 + entity.mention_count),  # Node size
                "color": self._get_type_color(entity.entity_type),
                "properties": entity.properties,
            })

        # Format links from temporal edges
        for edge in self.temporal_edges.values():
            if edge.source_id in entity_ids and edge.target_id in entity_ids:
                if include_inactive or edge.is_active:
                    links.append({
                        "source": edge.source_id,
                        "target": edge.target_id,
                        "type": edge.relation_type,
                        "weight": edge.current_weight,
                        "active": edge.is_active,
                        "stability": edge.stability_score,
                        "start_time": edge.start_time.isoformat(),
                        "end_time": edge.end_time.isoformat() if edge.end_time else None,
                    })

        return {
            "nodes": nodes,
            "links": links,
            "metadata": {
                "total_entities": len(self.entities),
                "total_edges": len(self.temporal_edges),
                "displayed_entities": len(nodes),
                "displayed_edges": len(links),
            }
        }

    def _get_type_color(self, entity_type: str) -> str:
        """Get color for entity type in visualization."""
        colors = {
            "person": "#4CAF50",
            "concept": "#2196F3",
            "code": "#FF9800",
            "file": "#9C27B0",
            "url": "#00BCD4",
            "tool": "#F44336",
            "default": "#757575",
        }
        return colors.get(entity_type.lower(), colors["default"])

    # Persistence

    async def _save_temporal_edge(self, edge: TemporalEdge):
        """Save temporal edge to disk."""
        edges_dir = self.data_dir / "temporal_edges"
        edges_dir.mkdir(parents=True, exist_ok=True)

        edge_id = hashlib.md5(
            f"{edge.source_id}|{edge.target_id}|{edge.relation_type}".encode()
        ).hexdigest()[:12]

        edge_file = edges_dir / f"{edge_id}.json"
        edge_file.write_text(json.dumps(edge.to_dict(), indent=2), encoding='utf-8')

    async def _save_entity(self, entity: Entity):
        """Save entity to disk."""
        entities_dir = self.data_dir / "entities"
        entities_dir.mkdir(parents=True, exist_ok=True)

        entity_file = entities_dir / f"{entity.id}.json"
        entity_file.write_text(json.dumps(entity.to_dict(), indent=2), encoding='utf-8')

    async def _save_resolution_data(self):
        """Save entity resolution data."""
        resolution_file = self.data_dir / "resolution.json"
        data = {
            "clusters": {k: v.to_dict() for k, v in self.entity_clusters.items()},
            "canonical_mapping": self.canonical_mapping,
        }
        resolution_file.write_text(json.dumps(data, indent=2), encoding='utf-8')

    async def _load_temporal_data(self):
        """Load temporal edges from disk."""
        edges_dir = self.data_dir / "temporal_edges"
        if not edges_dir.exists():
            return

        for edge_file in edges_dir.glob("*.json"):
            try:
                data = json.loads(edge_file.read_text(encoding='utf-8'))
                edge = TemporalEdge.from_dict(data)

                edge_id = edge_file.stem
                self.temporal_edges[edge_id] = edge

                self.edges_by_source[edge.source_id].add(edge_id)
                self.edges_by_target[edge.target_id].add(edge_id)
                self.edges_by_time[edge.start_time.strftime("%Y-%m")].add(edge_id)
            except Exception as e:
                logger.error(f"Failed to load temporal edge {edge_file}: {e}")

    async def _load_resolution_data(self):
        """Load entity resolution data."""
        resolution_file = self.data_dir / "resolution.json"
        if not resolution_file.exists():
            return

        try:
            data = json.loads(resolution_file.read_text(encoding='utf-8'))

            for k, v in data.get("clusters", {}).items():
                self.entity_clusters[k] = EntityCluster(**{
                    **v,
                    "merged_at": datetime.fromisoformat(v["merged_at"]) if v.get("merged_at") else None,
                })

            self.canonical_mapping = data.get("canonical_mapping", {})
        except Exception as e:
            logger.error(f"Failed to load resolution data: {e}")

    def get_stats(self) -> dict:
        """Get extended statistics."""
        base_stats = super().get_stats()
        base_stats.update({
            "temporal_edges": len(self.temporal_edges),
            "active_edges": sum(1 for e in self.temporal_edges.values() if e.is_active),
            "entity_clusters": len(self.entity_clusters),
            "merged_entities": len(self.canonical_mapping),
            "resolution_candidates": len(self.resolution_queue),
        })
        return base_stats

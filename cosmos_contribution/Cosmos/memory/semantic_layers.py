"""
cosmos Semantic Memory Layers - Concept Hierarchy and Knowledge Distillation

Q1 2025 Feature: Semantic Memory Layers
- Automatic concept hierarchy extraction
- Abstract knowledge distillation
- Cross-domain connection discovery
- Multi-level abstraction
"""

import asyncio
import json
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Any, Callable
from collections import defaultdict
from enum import Enum
import numpy as np

from loguru import logger


class AbstractionLevel(Enum):
    """Levels of semantic abstraction."""
    INSTANCE = 0      # Specific facts (e.g., "Python 3.11 released in Oct 2022")
    CATEGORY = 1      # Categories (e.g., "Python is a programming language")
    CONCEPT = 2       # Concepts (e.g., "Programming languages are tools for computation")
    PRINCIPLE = 3     # Principles (e.g., "Abstraction enables complexity management")
    DOMAIN = 4        # Domains (e.g., "Software Engineering")


@dataclass
class SemanticConcept:
    """A concept at any abstraction level."""
    id: str
    name: str
    level: AbstractionLevel
    description: str

    # Hierarchy
    parent_id: Optional[str] = None
    children_ids: list[str] = field(default_factory=list)

    # Properties
    properties: dict = field(default_factory=dict)
    examples: list[str] = field(default_factory=list)
    counter_examples: list[str] = field(default_factory=list)

    # Embeddings
    embedding: Optional[list[float]] = None
    centroid_embedding: Optional[list[float]] = None  # Average of all instances

    # Statistics
    instance_count: int = 0
    first_seen: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    confidence: float = 0.5

    # Cross-domain connections
    related_domains: list[str] = field(default_factory=list)
    analogies: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "level": self.level.value,
            "description": self.description,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "properties": self.properties,
            "examples": self.examples[:10],
            "counter_examples": self.counter_examples[:5],
            "instance_count": self.instance_count,
            "first_seen": self.first_seen.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "confidence": self.confidence,
            "related_domains": self.related_domains,
            "analogies": self.analogies[:5],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SemanticConcept":
        return cls(
            id=data["id"],
            name=data["name"],
            level=AbstractionLevel(data["level"]),
            description=data["description"],
            parent_id=data.get("parent_id"),
            children_ids=data.get("children_ids", []),
            properties=data.get("properties", {}),
            examples=data.get("examples", []),
            counter_examples=data.get("counter_examples", []),
            instance_count=data.get("instance_count", 0),
            first_seen=datetime.fromisoformat(data.get("first_seen", datetime.now().isoformat())),
            last_updated=datetime.fromisoformat(data.get("last_updated", datetime.now().isoformat())),
            confidence=data.get("confidence", 0.5),
            related_domains=data.get("related_domains", []),
            analogies=data.get("analogies", []),
        )


@dataclass
class DomainCluster:
    """A cluster of related concepts forming a domain."""
    id: str
    name: str
    description: str
    concept_ids: list[str] = field(default_factory=list)
    centroid_embedding: Optional[list[float]] = None
    keywords: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "concept_ids": self.concept_ids,
            "keywords": self.keywords,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class CrossDomainConnection:
    """A connection between concepts across domains."""
    source_concept_id: str
    target_concept_id: str
    connection_type: str  # "analogy", "generalization", "specialization", "contrast"
    explanation: str
    strength: float
    discovered_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "source_concept_id": self.source_concept_id,
            "target_concept_id": self.target_concept_id,
            "connection_type": self.connection_type,
            "explanation": self.explanation,
            "strength": self.strength,
            "discovered_at": self.discovered_at.isoformat(),
        }


class SemanticLayerSystem:
    """
    Multi-level semantic memory with concept hierarchies.

    Features:
    - Automatic concept hierarchy extraction
    - Abstract knowledge distillation from instances
    - Cross-domain connection discovery
    - Analogy finding
    """

    def __init__(
        self,
        data_dir: str = "./data/semantic",
        max_concepts: int = 10000,
        abstraction_threshold: float = 0.7,
        cross_domain_threshold: float = 0.6,
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.max_concepts = max_concepts
        self.abstraction_threshold = abstraction_threshold
        self.cross_domain_threshold = cross_domain_threshold

        # Storage
        self.concepts: dict[str, SemanticConcept] = {}
        self.domains: dict[str, DomainCluster] = {}
        self.connections: list[CrossDomainConnection] = []

        # Indices
        self.name_to_id: dict[str, str] = {}
        self.concepts_by_level: dict[AbstractionLevel, set[str]] = defaultdict(set)
        self.concepts_by_parent: dict[str, set[str]] = defaultdict(set)

        # Embedding function
        self.embed_fn: Optional[Callable] = None

        # LLM function for distillation (set externally)
        self.llm_fn: Optional[Callable] = None

        self._lock = asyncio.Lock()
        self._initialized = False

    async def initialize(self):
        """Load existing semantic data."""
        if self._initialized:
            return

        await self._load_from_disk()
        self._initialized = True
        logger.info(f"Semantic layers initialized with {len(self.concepts)} concepts, {len(self.domains)} domains")

    async def add_instance(
        self,
        content: str,
        context: Optional[str] = None,
        domain_hint: Optional[str] = None,
    ) -> SemanticConcept:
        """
        Add a new instance and integrate into concept hierarchy.

        This will:
        1. Create instance-level concept
        2. Find or create parent category
        3. Update hierarchy statistics
        4. Trigger distillation if threshold reached
        """
        async with self._lock:
            # Get embedding
            embedding = await self._get_embedding(content)

            # Create instance concept
            instance_id = f"inst_{hashlib.md5(content[:100].encode()).hexdigest()[:12]}"

            instance = SemanticConcept(
                id=instance_id,
                name=content[:100],
                level=AbstractionLevel.INSTANCE,
                description=content,
                embedding=embedding,
            )

            # Find best parent category
            parent = await self._find_or_create_parent(instance, context)
            if parent:
                instance.parent_id = parent.id
                parent.children_ids.append(instance.id)
                parent.instance_count += 1
                parent.last_updated = datetime.now()

                # Add as example
                if len(parent.examples) < 10:
                    parent.examples.append(content[:200])

                # Update centroid
                await self._update_centroid(parent)

                # Check if we should distill higher
                if parent.instance_count >= 5 and parent.level.value < AbstractionLevel.PRINCIPLE.value:
                    await self._attempt_abstraction(parent)

            # Store
            self.concepts[instance_id] = instance
            self.concepts_by_level[AbstractionLevel.INSTANCE].add(instance_id)
            if parent:
                self.concepts_by_parent[parent.id].add(instance_id)

            # Check for cross-domain connections
            if domain_hint:
                await self._discover_cross_domain_connections(instance, domain_hint)

            await self._save_concept(instance)
            if parent:
                await self._save_concept(parent)

            return instance

    async def _find_or_create_parent(
        self,
        instance: SemanticConcept,
        context: Optional[str] = None,
    ) -> Optional[SemanticConcept]:
        """Find the best parent category for an instance."""
        if not instance.embedding:
            return None

        instance_vec = np.array(instance.embedding)

        # Find most similar category
        best_parent = None
        best_similarity = self.abstraction_threshold

        for concept_id, concept in self.concepts.items():
            if concept.level.value < AbstractionLevel.CATEGORY.value:
                continue

            if concept.centroid_embedding:
                centroid_vec = np.array(concept.centroid_embedding)
                similarity = self._cosine_similarity(instance_vec, centroid_vec)

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_parent = concept

        # If no good parent found, create one
        if not best_parent:
            category_name = await self._generate_category_name(instance.description, context)
            if category_name:
                best_parent = await self._create_concept(
                    name=category_name,
                    level=AbstractionLevel.CATEGORY,
                    description=f"Category for: {instance.name}",
                    embedding=instance.embedding,
                )

        return best_parent

    async def _create_concept(
        self,
        name: str,
        level: AbstractionLevel,
        description: str,
        embedding: Optional[list[float]] = None,
        parent_id: Optional[str] = None,
    ) -> SemanticConcept:
        """Create a new concept."""
        concept_id = f"concept_{level.name.lower()}_{hashlib.md5(name.encode()).hexdigest()[:10]}"

        concept = SemanticConcept(
            id=concept_id,
            name=name,
            level=level,
            description=description,
            embedding=embedding,
            centroid_embedding=embedding,
            parent_id=parent_id,
        )

        self.concepts[concept_id] = concept
        self.name_to_id[name.lower()] = concept_id
        self.concepts_by_level[level].add(concept_id)

        if parent_id:
            self.concepts_by_parent[parent_id].add(concept_id)
            if parent_id in self.concepts:
                self.concepts[parent_id].children_ids.append(concept_id)

        await self._save_concept(concept)

        return concept

    async def _attempt_abstraction(self, concept: SemanticConcept):
        """Try to create a higher-level abstraction."""
        if concept.level.value >= AbstractionLevel.PRINCIPLE.value:
            return

        next_level = AbstractionLevel(concept.level.value + 1)

        # Check if we already have a parent at the next level
        if concept.parent_id and concept.parent_id in self.concepts:
            parent = self.concepts[concept.parent_id]
            if parent.level.value >= next_level.value:
                return

        # Use LLM to generate abstraction
        abstraction = await self._distill_abstraction(concept)
        if abstraction:
            parent = await self._create_concept(
                name=abstraction["name"],
                level=next_level,
                description=abstraction["description"],
                embedding=concept.centroid_embedding,
            )

            concept.parent_id = parent.id
            parent.children_ids.append(concept.id)
            parent.instance_count = concept.instance_count

            await self._save_concept(concept)
            await self._save_concept(parent)

    async def _distill_abstraction(self, concept: SemanticConcept) -> Optional[dict]:
        """Use LLM to distill a higher-level abstraction."""
        if not self.llm_fn:
            # Fallback: simple pattern
            return {
                "name": f"Abstract_{concept.name[:30]}",
                "description": f"Higher-level concept encompassing: {concept.description[:100]}",
            }

        prompt = f"""Given these examples of a concept:

Concept: {concept.name}
Description: {concept.description}
Examples:
{chr(10).join(f'- {ex}' for ex in concept.examples[:5])}

Generate a higher-level abstraction (more general principle or pattern).
Return JSON: {{"name": "...", "description": "..."}}"""

        try:
            result = await self.llm_fn(prompt)
            if isinstance(result, str):
                return json.loads(result)
            return result
        except Exception as e:
            logger.error(f"Distillation failed: {e}")
            return None

    async def _generate_category_name(
        self,
        instance_description: str,
        context: Optional[str] = None,
    ) -> Optional[str]:
        """Generate a category name for an instance."""
        if not self.llm_fn:
            # Simple heuristic: extract key phrase
            words = instance_description.split()
            if len(words) >= 2:
                return f"{words[0].capitalize()} {words[1]}"
            return instance_description[:50]

        prompt = f"""What category does this belong to?

Instance: {instance_description[:200]}
{f'Context: {context}' if context else ''}

Return only the category name (2-4 words)."""

        try:
            result = await self.llm_fn(prompt)
            return result.strip()[:50] if result else None
        except Exception as e:
            logger.error(f"Category generation failed: {e}")
            return None

    async def _update_centroid(self, concept: SemanticConcept):
        """Update the centroid embedding for a concept."""
        child_embeddings = []

        for child_id in concept.children_ids:
            if child_id in self.concepts:
                child = self.concepts[child_id]
                if child.embedding:
                    child_embeddings.append(child.embedding)

        if child_embeddings:
            centroid = np.mean(child_embeddings, axis=0)
            concept.centroid_embedding = centroid.tolist()

    async def _discover_cross_domain_connections(
        self,
        concept: SemanticConcept,
        source_domain: str,
    ):
        """Find connections to concepts in other domains."""
        if not concept.embedding:
            return

        concept_vec = np.array(concept.embedding)

        for domain_id, domain in self.domains.items():
            if domain.name == source_domain:
                continue

            # Check each concept in the domain
            for other_id in domain.concept_ids:
                if other_id not in self.concepts:
                    continue

                other = self.concepts[other_id]
                if not other.embedding:
                    continue

                other_vec = np.array(other.embedding)
                similarity = self._cosine_similarity(concept_vec, other_vec)

                if similarity >= self.cross_domain_threshold:
                    # Found a cross-domain connection
                    connection = CrossDomainConnection(
                        source_concept_id=concept.id,
                        target_concept_id=other.id,
                        connection_type="analogy",
                        explanation=f"Similar concepts across {source_domain} and {domain.name}",
                        strength=float(similarity),
                    )
                    self.connections.append(connection)

                    # Update concept analogies
                    concept.analogies.append({
                        "target_id": other.id,
                        "target_name": other.name,
                        "domain": domain.name,
                        "strength": float(similarity),
                    })

    async def get_concept_hierarchy(
        self,
        concept_id: str,
        depth: int = 3,
    ) -> dict:
        """Get the hierarchy around a concept."""
        if concept_id not in self.concepts:
            return {}

        concept = self.concepts[concept_id]

        # Get ancestors
        ancestors = []
        current = concept
        while current.parent_id and len(ancestors) < depth:
            if current.parent_id in self.concepts:
                parent = self.concepts[current.parent_id]
                ancestors.append({
                    "id": parent.id,
                    "name": parent.name,
                    "level": parent.level.name,
                })
                current = parent
            else:
                break

        # Get descendants
        def get_children(c_id: str, remaining_depth: int) -> list:
            if remaining_depth <= 0 or c_id not in self.concepts:
                return []

            c = self.concepts[c_id]
            children = []
            for child_id in c.children_ids[:10]:  # Limit children
                if child_id in self.concepts:
                    child = self.concepts[child_id]
                    children.append({
                        "id": child.id,
                        "name": child.name,
                        "level": child.level.name,
                        "children": get_children(child_id, remaining_depth - 1),
                    })
            return children

        descendants = get_children(concept_id, depth)

        return {
            "concept": concept.to_dict(),
            "ancestors": ancestors,
            "descendants": descendants,
        }

    async def find_analogies(
        self,
        concept_id: str,
        max_results: int = 5,
    ) -> list[dict]:
        """Find analogous concepts across domains."""
        if concept_id not in self.concepts:
            return []

        concept = self.concepts[concept_id]

        # Return stored analogies
        analogies = concept.analogies[:max_results]

        # If not enough, search for more
        if len(analogies) < max_results and concept.embedding:
            concept_vec = np.array(concept.embedding)

            candidates = []
            for other_id, other in self.concepts.items():
                if other_id == concept_id:
                    continue
                if other.level != concept.level:
                    continue
                if not other.embedding:
                    continue

                other_vec = np.array(other.embedding)
                similarity = self._cosine_similarity(concept_vec, other_vec)

                if similarity >= 0.5:
                    candidates.append({
                        "target_id": other.id,
                        "target_name": other.name,
                        "level": other.level.name,
                        "similarity": float(similarity),
                    })

            # Sort by similarity
            candidates.sort(key=lambda x: x["similarity"], reverse=True)
            analogies.extend(candidates[:max_results - len(analogies)])

        return analogies

    async def query_semantic(
        self,
        query: str,
        min_level: AbstractionLevel = AbstractionLevel.INSTANCE,
        max_results: int = 10,
    ) -> list[SemanticConcept]:
        """Query semantic concepts."""
        query_embedding = await self._get_embedding(query)
        if not query_embedding:
            # Fall back to text search
            results = []
            query_lower = query.lower()
            for concept in self.concepts.values():
                if concept.level.value >= min_level.value:
                    if query_lower in concept.name.lower() or query_lower in concept.description.lower():
                        results.append(concept)
            return results[:max_results]

        query_vec = np.array(query_embedding)

        # Score all concepts at or above min_level
        scored = []
        for concept in self.concepts.values():
            if concept.level.value < min_level.value:
                continue

            embedding = concept.centroid_embedding or concept.embedding
            if embedding:
                similarity = self._cosine_similarity(query_vec, np.array(embedding))
                scored.append((concept, similarity))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [c for c, _ in scored[:max_results]]

    async def create_domain(
        self,
        name: str,
        description: str,
        seed_concepts: Optional[list[str]] = None,
    ) -> DomainCluster:
        """Create a new domain cluster."""
        domain_id = f"domain_{hashlib.md5(name.encode()).hexdigest()[:10]}"

        domain = DomainCluster(
            id=domain_id,
            name=name,
            description=description,
            concept_ids=seed_concepts or [],
        )

        # Update centroid if concepts provided
        if seed_concepts:
            embeddings = []
            for cid in seed_concepts:
                if cid in self.concepts and self.concepts[cid].embedding:
                    embeddings.append(self.concepts[cid].embedding)

            if embeddings:
                domain.centroid_embedding = np.mean(embeddings, axis=0).tolist()

        self.domains[domain_id] = domain
        await self._save_domain(domain)

        return domain

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    async def _get_embedding(self, text: str) -> Optional[list[float]]:
        """Get embedding for text."""
        if not self.embed_fn:
            return None
        try:
            if asyncio.iscoroutinefunction(self.embed_fn):
                return await self.embed_fn(text)
            return self.embed_fn(text)
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return None

    async def _save_concept(self, concept: SemanticConcept):
        """Save concept to disk."""
        concepts_dir = self.data_dir / "concepts" / concept.level.name.lower()
        concepts_dir.mkdir(parents=True, exist_ok=True)

        concept_file = concepts_dir / f"{concept.id}.json"
        concept_file.write_text(json.dumps(concept.to_dict(), indent=2), encoding='utf-8')

    async def _save_domain(self, domain: DomainCluster):
        """Save domain to disk."""
        domains_dir = self.data_dir / "domains"
        domains_dir.mkdir(parents=True, exist_ok=True)

        domain_file = domains_dir / f"{domain.id}.json"
        domain_file.write_text(json.dumps(domain.to_dict(), indent=2), encoding='utf-8')

    async def _load_from_disk(self):
        """Load all data from disk."""
        # Load concepts
        concepts_dir = self.data_dir / "concepts"
        if concepts_dir.exists():
            for concept_file in concepts_dir.rglob("*.json"):
                try:
                    data = json.loads(concept_file.read_text(encoding='utf-8'))
                    concept = SemanticConcept.from_dict(data)
                    self.concepts[concept.id] = concept
                    self.name_to_id[concept.name.lower()] = concept.id
                    self.concepts_by_level[concept.level].add(concept.id)
                    if concept.parent_id:
                        self.concepts_by_parent[concept.parent_id].add(concept.id)
                except Exception as e:
                    logger.error(f"Failed to load concept {concept_file}: {e}")

        # Load domains
        domains_dir = self.data_dir / "domains"
        if domains_dir.exists():
            for domain_file in domains_dir.glob("*.json"):
                try:
                    data = json.loads(domain_file.read_text(encoding='utf-8'))
                    domain = DomainCluster(**{k: v for k, v in data.items() if k != "created_at"})
                    domain.created_at = datetime.fromisoformat(data.get("created_at", datetime.now().isoformat()))
                    self.domains[domain.id] = domain
                except Exception as e:
                    logger.error(f"Failed to load domain {domain_file}: {e}")

    def get_stats(self) -> dict:
        """Get semantic layer statistics."""
        return {
            "total_concepts": len(self.concepts),
            "total_domains": len(self.domains),
            "concepts_by_level": {
                level.name: len(ids) for level, ids in self.concepts_by_level.items()
            },
            "cross_domain_connections": len(self.connections),
        }

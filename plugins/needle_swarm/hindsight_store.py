"""hindsight_store.py - Retain/consolidate/recall memory layer inspired by Hindsight.

Every resolved request gets retained as a RawFact.
consolidate() clusters raw facts by embedding similarity into deduplicated Observations with
proof counts and decaying freshness.
recall() queries the observation layer rather than raw facts.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .needle_ops import cosine_similarity, embed_text

logger = logging.getLogger(__name__)


@dataclass
class RawFact:
    fact_id: str
    content: str
    source_type: str  # "DISPATCH", "SWARM", "PLAYBOOK", "ESCALATION"
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: List[float] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RawFact:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Observation:
    obs_id: str
    summary: str
    proof_count: int
    raw_fact_ids: List[str]
    embedding: List[float]
    freshness: float = 1.0
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Observation:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class HindsightStore:
    """Hindsight memory layer with retain, consolidate, and recall."""

    def __init__(self, store_dir: str | Path):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.facts_file = self.store_dir / "raw_facts.json"
        self.obs_file = self.store_dir / "observations.json"
        self.raw_facts: List[RawFact] = []
        self.observations: List[Observation] = []
        self.load()

    def load(self) -> None:
        if self.facts_file.exists():
            try:
                data = json.loads(self.facts_file.read_text(encoding="utf-8"))
                self.raw_facts = [RawFact.from_dict(d) for d in data]
            except Exception as e:
                logger.warning("Failed to load raw facts: %s", e)
                self.raw_facts = []

        if self.obs_file.exists():
            try:
                data = json.loads(self.obs_file.read_text(encoding="utf-8"))
                self.observations = [Observation.from_dict(d) for d in data]
            except Exception as e:
                logger.warning("Failed to load observations: %s", e)
                self.observations = []

    def save(self) -> None:
        self.facts_file.write_text(json.dumps([f.to_dict() for f in self.raw_facts], indent=2), encoding="utf-8")
        self.obs_file.write_text(json.dumps([o.to_dict() for o in self.observations], indent=2), encoding="utf-8")

    def retain(self, content: str, source_type: str = "DISPATCH", metadata: Optional[Dict[str, Any]] = None) -> RawFact:
        """Cheap retain fired on every resolved request."""
        fact_id = f"fact_{int(time.time()*1000)}_{len(self.raw_facts)+1}"
        emb = embed_text(content)
        fact = RawFact(
            fact_id=fact_id,
            content=content,
            source_type=source_type,
            metadata=metadata or {},
            embedding=emb,
            created_at=time.time(),
        )
        self.raw_facts.append(fact)
        self.save()
        return fact

    def consolidate(self, similarity_threshold: float = 0.75) -> int:
        """Consolidate unclustered raw facts into deduplicated observations."""
        now = time.time()
        # Decay existing observations
        for obs in self.observations:
            days_passed = (now - obs.updated_at) / 86400.0
            obs.freshness = max(0.1, obs.freshness * math.exp(-0.05 * days_passed))

        processed_fact_ids = set()
        for obs in self.observations:
            processed_fact_ids.update(obs.raw_fact_ids)

        unprocessed = [f for f in self.raw_facts if f.fact_id not in processed_fact_ids]
        if not unprocessed:
            self.save()
            return 0

        new_obs_count = 0
        for fact in unprocessed:
            matched_obs = None
            best_sim = 0.0
            for obs in self.observations:
                sim = cosine_similarity(fact.embedding, obs.embedding)
                if sim > best_sim and sim >= similarity_threshold:
                    best_sim = sim
                    matched_obs = obs

            if matched_obs:
                matched_obs.proof_count += 1
                matched_obs.raw_fact_ids.append(fact.fact_id)
                matched_obs.freshness = min(2.0, matched_obs.freshness + 0.2)
                matched_obs.updated_at = now
            else:
                obs_id = f"obs_{int(now*1000)}_{len(self.observations)+1}"
                new_obs = Observation(
                    obs_id=obs_id,
                    summary=fact.content,
                    proof_count=1,
                    raw_fact_ids=[fact.fact_id],
                    embedding=fact.embedding,
                    freshness=1.0,
                    updated_at=now,
                )
                self.observations.append(new_obs)
                new_obs_count += 1

        self.save()
        return new_obs_count

    def recall(self, query: str, top_k: int = 5, min_score: float = 0.4) -> List[Dict[str, Any]]:
        """Query consolidated observations layer."""
        if not self.observations:
            return []

        q_emb = embed_text(query)
        scored = []
        for obs in self.observations:
            sim = cosine_similarity(q_emb, obs.embedding)
            score = sim * obs.freshness * (1.0 + 0.1 * math.log(obs.proof_count))
            if sim >= min_score:
                scored.append((obs, sim, score))

        scored.sort(key=lambda x: x[2], reverse=True)
        results = []
        for obs, sim, score in scored[:top_k]:
            results.append({
                "obs_id": obs.obs_id,
                "summary": obs.summary,
                "proof_count": obs.proof_count,
                "freshness": round(obs.freshness, 2),
                "similarity": round(sim, 3),
                "final_score": round(score, 3),
            })
        return results

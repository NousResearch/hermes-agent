"""registry.py - SkillMeta and SkillRegistry for Needle Swarm catalog.

Disk-backed catalog of tiny, fine-tuned Needle skill models.
Tracks metadata, tool schemas, use/escalation stats, and pending training examples.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SkillMeta:
    """Metadata for a registered Needle skill."""

    skill_id: str
    name: str
    description: str
    kind: str  # "ACTION", "MEMORY", or "PLAYBOOK"
    size_mb: float = 12.0
    version: int = 1
    tool_schema: Optional[Dict[str, Any]] = None
    tools: List[str] = field(default_factory=list)
    clusters: List[str] = field(default_factory=list)
    weights_path: Optional[str] = None
    corpus_path: Optional[str] = None
    use_count: int = 0
    escalation_count: int = 0
    success_count: int = 0
    confidence_avg: float = 0.0
    eval_score: float = 0.0
    pending_examples: List[Dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SkillMeta:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class SkillRegistry:
    """Disk-backed catalog of Needle skills."""

    def __init__(self, catalog_dir: str | Path):
        self.catalog_dir = Path(catalog_dir)
        self.catalog_dir.mkdir(parents=True, exist_ok=True)
        self.meta_file = self.catalog_dir / "skills.json"
        self._skills: Dict[str, SkillMeta] = {}
        self.load()

    def load(self) -> None:
        if self.meta_file.exists():
            try:
                data = json.loads(self.meta_file.read_text(encoding="utf-8"))
                self._skills = {
                    sid: SkillMeta.from_dict(sdata) for sid, sdata in data.items()
                }
            except Exception as e:
                logger.warning("Failed to load skill registry from %s: %s", self.meta_file, e)
                self._skills = {}

    def save(self) -> None:
        data = {sid: meta.to_dict() for sid, meta in self._skills.items()}
        self.meta_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def register(self, meta: SkillMeta) -> SkillMeta:
        meta.updated_at = time.time()
        self._skills[meta.skill_id] = meta
        self.save()
        return meta

    def unregister(self, skill_id: str) -> Optional[SkillMeta]:
        meta = self._skills.pop(skill_id, None)
        if meta:
            self.save()
        return meta

    def get(self, skill_id: str) -> Optional[SkillMeta]:
        return self._skills.get(skill_id)

    def list_skills(self, cluster: Optional[str] = None, kind: Optional[str] = None) -> List[SkillMeta]:
        res = list(self._skills.values())
        if cluster:
            res = [s for s in res if cluster in s.clusters]
        if kind:
            res = [s for s in res if s.kind.upper() == kind.upper()]
        return res

    def record_usage(self, skill_id: str, confidence: float, escalated: bool = False, success: bool = True) -> None:
        meta = self.get(skill_id)
        if not meta:
            return
        meta.use_count += 1
        if escalated:
            meta.escalation_count += 1
        if success:
            meta.success_count += 1
        # Exponential moving average for confidence
        alpha = 0.2
        meta.confidence_avg = (1 - alpha) * meta.confidence_avg + alpha * confidence
        meta.updated_at = time.time()
        self.save()

    def add_pending_example(self, skill_id: str, example: Dict[str, Any]) -> None:
        meta = self.get(skill_id)
        if not meta:
            return
        meta.pending_examples.append(example)
        meta.updated_at = time.time()
        self.save()

    def clear_pending_examples(self, skill_id: str) -> List[Dict[str, Any]]:
        meta = self.get(skill_id)
        if not meta:
            return []
        examples = meta.pending_examples
        meta.pending_examples = []
        meta.updated_at = time.time()
        self.save()
        return examples

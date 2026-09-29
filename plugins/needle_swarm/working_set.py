"""working_set.py - WorkingSet loaded in-memory subset budgeted by MB.

LRU-evicts unpinned skills when memory exceeds budget.
Supports exclusive cluster switching so swarm fan-out queries only relevant skills.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Set

from .needle_ops import LoadedNeedleInstance
from .registry import SkillMeta, SkillRegistry

logger = logging.getLogger(__name__)


class WorkingSet:
    """In-memory active working set of Needle skills budgeted by MB."""

    def __init__(self, registry: SkillRegistry, max_mb: float = 128.0):
        self.registry = registry
        self.max_mb = max_mb
        self._loaded: Dict[str, LoadedNeedleInstance] = {}
        self._pinned: Set[str] = set()
        self._last_used: Dict[str, float] = {}
        self._active_cluster: Optional[str] = None

    def current_mb(self) -> float:
        total = 0.0
        for sid in self._loaded:
            meta = self.registry.get(sid)
            if meta:
                total += meta.size_mb
            else:
                total += 12.0
        return total

    def is_loaded(self, skill_id: str) -> bool:
        return skill_id in self._loaded

    def get(self, skill_id: str) -> Optional[LoadedNeedleInstance]:
        if skill_id in self._loaded:
            self._last_used[skill_id] = time.time()
            return self._loaded[skill_id]
        return None

    def load_skill(self, skill_id: str, pin: bool = False) -> Optional[LoadedNeedleInstance]:
        if skill_id in self._loaded:
            self._last_used[skill_id] = time.time()
            if pin:
                self._pinned.add(skill_id)
            return self._loaded[skill_id]

        meta = self.registry.get(skill_id)
        if not meta:
            logger.warning("Attempted to load unregistered skill: %s", skill_id)
            return None

        # Check budget and evict unpinned LRU skills if needed
        needed_mb = meta.size_mb
        self._ensure_budget(needed_mb)

        instance = LoadedNeedleInstance(
            skill_id=meta.skill_id,
            kind=meta.kind,
            tools=meta.tools,
            weights_path=meta.weights_path,
        )
        self._loaded[skill_id] = instance
        self._last_used[skill_id] = time.time()
        if pin:
            self._pinned.add(skill_id)
        logger.debug("Loaded skill %s (%s MB). Total: %.1f/%.1f MB", skill_id, meta.size_mb, self.current_mb(), self.max_mb)
        return instance

    def unload_skill(self, skill_id: str) -> bool:
        if skill_id in self._loaded:
            del self._loaded[skill_id]
            self._pinned.discard(skill_id)
            self._last_used.pop(skill_id, None)
            logger.debug("Unloaded skill %s. Total: %.1f MB", skill_id, self.current_mb())
            return True
        return False

    def load_cluster(self, cluster_id: str, exclusive: bool = True) -> List[str]:
        """Load all skills belonging to a cluster.

        If exclusive=True, unloads unpinned skills from previous cluster.
        """
        if exclusive and self._active_cluster != cluster_id:
            # Unload unpinned skills
            for sid in list(self._loaded.keys()):
                if sid not in self._pinned:
                    self.unload_skill(sid)

        self._active_cluster = cluster_id
        cluster_skills = self.registry.list_skills(cluster=cluster_id)
        loaded_ids = []
        for meta in cluster_skills:
            if self.load_skill(meta.skill_id):
                loaded_ids.append(meta.skill_id)
        return loaded_ids

    def get_loaded_skills(self) -> List[LoadedNeedleInstance]:
        return list(self._loaded.values())

    def get_loaded_ids(self) -> List[str]:
        return list(self._loaded.keys())

    def _ensure_budget(self, needed_mb: float) -> None:
        while self.current_mb() + needed_mb > self.max_mb:
            # Find candidate unpinned skill with oldest _last_used
            candidates = [sid for sid in self._loaded if sid not in self._pinned]
            if not candidates:
                logger.warning("WorkingSet MB budget exceeded (%.1f > %.1f MB), but all loaded skills are pinned", self.current_mb() + needed_mb, self.max_mb)
                break
            lru_sid = min(candidates, key=lambda s: self._last_used.get(s, 0.0))
            self.unload_skill(lru_sid)

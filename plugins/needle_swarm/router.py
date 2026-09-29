"""router.py - Router picks relevant skill(s) via embedding match against description.

Uses Needle's embed call (needle_ops.embed_text) to match utterances against skill descriptions.
Falls through to escalation when score is below threshold or ambiguous.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from .needle_ops import cosine_similarity, embed_text
from .registry import SkillMeta, SkillRegistry
from .working_set import WorkingSet

logger = logging.getLogger(__name__)


class Router:
    """Embedding-based router matching utterances to skill descriptions."""

    def __init__(self, registry: SkillRegistry, working_set: WorkingSet, threshold: float = 0.55):
        self.registry = registry
        self.working_set = working_set
        self.threshold = threshold
        self._desc_embeddings: dict[str, List[float]] = {}

    def _get_embedding(self, skill_id: str, text: str) -> List[float]:
        if skill_id not in self._desc_embeddings:
            self._desc_embeddings[skill_id] = embed_text(text)
        return self._desc_embeddings[skill_id]

    def route(
        self,
        utterance: str,
        active_ids: Optional[List[str]] = None,
        threshold_override: Optional[float] = None,
    ) -> Tuple[Optional[str], float, List[Tuple[str, float]]]:
        """Route an utterance to the best-matching loaded or active skill.

        Returns: (best_skill_id_or_None, best_score, all_ranked_scores)
        """
        thresh = threshold_override if threshold_override is not None else self.threshold
        candidate_ids = active_ids or self.working_set.get_loaded_ids()
        if not candidate_ids:
            # Fall back to all registered skills
            candidate_ids = [s.skill_id for s in self.registry.list_skills()]

        query_emb = embed_text(utterance)
        scores: List[Tuple[str, float]] = []

        for sid in candidate_ids:
            meta = self.registry.get(sid)
            if not meta:
                continue
            skill_emb = self._get_embedding(sid, meta.description)
            sim = cosine_similarity(query_emb, skill_emb)
            scores.append((sid, sim))

        scores.sort(key=lambda x: x[1], reverse=True)

        if not scores:
            return None, 0.0, []

        best_id, best_score = scores[0]
        if best_score < thresh:
            logger.debug("Routing score %.3f below threshold %.3f -> escalate", best_score, thresh)
            return None, best_score, scores

        return best_id, best_score, scores

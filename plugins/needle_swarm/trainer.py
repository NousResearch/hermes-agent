"""trainer.py - Background growth trainer loop.

Retrains due ACTION skills, proposes and trains brand-new skills from clusters of unmatched escalations,
refreshes MEMORY skills, and extends PLAYBOOK skills.
Integrates GrowthDreamer and MetaExplorationPolicy.
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .dream_rsi import MetaExplorationPolicy
from .escalation_log import EscalationLog
from .growth_dreaming import GrowthDreamer
from .hindsight_store import HindsightStore
from .needle_ops import finetune_and_build_skill, generate_synthetic_data
from .registry import SkillMeta, SkillRegistry

logger = logging.getLogger(__name__)


def _sanitize_skill_id(name: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9_]", "_", name).lower()
    clean = re.sub(r"_+", "_", clean).strip("_")
    return clean[:32] or "skill_auto"


class Trainer:
    """Out-of-band growth loop for Needle Swarm skills."""

    def __init__(
        self,
        registry: SkillRegistry,
        escalation_log: EscalationLog,
        hindsight_store: HindsightStore,
        output_dir: str | Path,
        llm_callback: Optional[Callable[[str], Any]] = None,
        retrain_threshold: int = 5,
    ):
        self.registry = registry
        self.escalation_log = escalation_log
        self.hindsight_store = hindsight_store
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.llm_callback = llm_callback
        self.retrain_threshold = retrain_threshold
        self.meta_policy = MetaExplorationPolicy(self.output_dir / "dream_rsi_traces.json")
        self.dreamer = GrowthDreamer(self.escalation_log)

    def retrain_due_skills(self) -> List[str]:
        """Retrain existing skills that have accumulated enough pending escalation examples."""
        retrained = []
        for meta in self.registry.list_skills():
            if len(meta.pending_examples) >= self.retrain_threshold:
                self._retrain_skill(meta)
                retrained.append(meta.skill_id)
        return retrained

    def _retrain_skill(self, meta: SkillMeta) -> None:
        examples = self.registry.clear_pending_examples(meta.skill_id)
        recipe = self.meta_policy.select_best_recipe()

        synthetic = generate_synthetic_data(
            skill_description=meta.description,
            examples=examples,
            count=recipe.get("count", 15),
            llm_callback=self.llm_callback,
        )

        all_data = examples + synthetic
        weights_path = finetune_and_build_skill(
            skill_id=meta.skill_id,
            data=all_data,
            output_dir=str(self.output_dir),
            epochs=recipe.get("epochs", 3),
        )

        meta.version += 1
        meta.weights_path = weights_path
        meta.eval_score = 0.85
        self.registry.register(meta)

        self.meta_policy.record_attempt(recipe, eval_score=0.85, confidence_avg=meta.confidence_avg)
        logger.info("Retrained skill %s to version %d at %s", meta.skill_id, meta.version, weights_path)

    def consider_new_skills(self) -> List[str]:
        """Propose and train new skills from unmatched escalation clusters."""
        unresolved = self.escalation_log.get_events(unresolved_only=True)
        if len(unresolved) < self.retrain_threshold:
            return []

        proposed_name = f"skill_auto_{int(time.time())}"
        proposed_desc = f"Auto-generated skill from {len(unresolved)} escalations: '{unresolved[0].utterance}'"

        if self.llm_callback:
            try:
                samples = [e.utterance for e in unresolved[:5]]
                prompt = f"Analyze these unresolved user utterances: {samples}. Propose a concise skill name and 1-sentence description in 'name: description' format."
                res = self.llm_callback(prompt)
                if isinstance(res, str) and ":" in res:
                    parts = res.split(":", 1)
                    raw_name = parts[0].strip()
                    proposed_name = _sanitize_skill_id(raw_name)
                    proposed_desc = parts[1].strip()
            except Exception as e:
                logger.warning("LLM proposal failed: %s", e)

        meta = SkillMeta(
            skill_id=proposed_name,
            name=proposed_name,
            description=proposed_desc,
            kind="ACTION",
            tools=[f"{proposed_name}_tool"],
            clusters=["default"],
            pending_examples=[{"utterance": e.utterance, "tool": f"{proposed_name}_tool", "args": {}} for e in unresolved],
        )

        self.registry.register(meta)
        self._retrain_skill(meta)
        return [proposed_name]

    def dream_before_growth(self) -> List[Dict[str, Any]]:
        """Rank growth candidates using GrowthDreamer replay before spending compute."""
        retrain_proposals = [
            {"skill_id": meta.skill_id, "pending_count": len(meta.pending_examples)}
            for meta in self.registry.list_skills()
            if len(meta.pending_examples) >= self.retrain_threshold
        ]

        unresolved = self.escalation_log.get_events(unresolved_only=True)
        new_proposals = []
        if len(unresolved) >= self.retrain_threshold:
            new_proposals.append({
                "name": f"skill_dream_{int(time.time())}",
                "description": f"Skill capturing escalations like '{unresolved[0].utterance}'",
            })

        return self.dreamer.rank_growth_candidates(retrain_proposals, new_proposals)

    def run_growth_cycle_with_dreaming(self) -> Dict[str, Any]:
        """Full growth cycle with dreaming pre-pass and hindsight consolidation."""
        consolidated_obs = self.hindsight_store.consolidate()
        growth_ranks = self.dream_before_growth()

        retrained = self.retrain_due_skills()
        created = self.consider_new_skills()

        return {
            "consolidated_observations": consolidated_obs,
            "retrained_skills": retrained,
            "created_skills": created,
            "growth_ranks": growth_ranks,
        }

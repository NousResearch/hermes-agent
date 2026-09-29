"""orchestrator.py - Entry point for local (Bonsai) and cloud LLMs.

Handles single dispatch (`handle_single`), swarm fan-out (`handle_swarm`), mode switches,
growth cycles, and escalation logging.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .escalation_log import EscalationLog
from .hindsight_store import HindsightStore
from .registry import SkillMeta, SkillRegistry
from .router import Router
from .swarm import SwarmDispatcher, SwarmResult
from .trainer import Trainer
from .working_set import WorkingSet

logger = logging.getLogger(__name__)


class Orchestrator:
    """Main entry point for Needle Swarm orchestration."""

    def __init__(
        self,
        data_dir: str | Path,
        brain_llm_callback: Optional[Callable[[str], str]] = None,
        max_working_mb: float = 128.0,
        confidence_threshold: float = 0.55,
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.registry = SkillRegistry(self.data_dir / "catalog")
        self.working_set = WorkingSet(self.registry, max_mb=max_working_mb)
        self.router = Router(self.registry, self.working_set, threshold=confidence_threshold)
        self.swarm_dispatcher = SwarmDispatcher(self.working_set)

        self.escalation_log = EscalationLog(self.data_dir / "escalations.json")
        self.hindsight_store = HindsightStore(self.data_dir / "hindsight")

        self.brain_llm_callback = brain_llm_callback
        self.trainer = Trainer(
            registry=self.registry,
            escalation_log=self.escalation_log,
            hindsight_store=self.hindsight_store,
            output_dir=self.data_dir / "artifacts",
            llm_callback=brain_llm_callback,
        )

    def enter_mode(self, mode_cluster: str, exclusive: bool = True) -> List[str]:
        """Switch active loaded cluster."""
        return self.working_set.load_cluster(mode_cluster, exclusive=exclusive)

    def handle_single(
        self,
        utterance: str,
        threshold_override: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Single-dispatch mode (e.g. coding mode).

        Routes utterance to best skill. Escalates to brain LLM if confidence is low.
        """
        best_id, confidence, ranked = self.router.route(utterance, threshold_override=threshold_override)

        if best_id:
            inst = self.working_set.get(best_id) or self.working_set.load_skill(best_id)
            if inst:
                if inst.kind == "ACTION":
                    tool, args, conf = inst.dispatch(utterance)
                    res = {"tool": tool, "args": args}
                elif inst.kind == "PLAYBOOK":
                    proc, conf = inst.extract(utterance)
                    res = {"procedure": proc}
                else:
                    tool, args, conf = inst.dispatch(utterance)
                    res = {"result": tool, "args": args}

                self.registry.record_usage(best_id, confidence=conf, escalated=False, success=True)
                self.escalation_log.log(
                    utterance=utterance,
                    routed_skill_id=best_id,
                    confidence=conf,
                    resolved_by="NEEDLE",
                    outcome="SUCCESS",
                    resolved_action=res,
                )
                self.hindsight_store.retain(f"Single dispatch via {best_id}: {utterance} -> {res}", source_type="DISPATCH")

                return {
                    "status": "AUTO_RESOLVED",
                    "skill_id": best_id,
                    "confidence": conf,
                    "action": res,
                }

        # Escalation path to Brain LLM (Bonsai or Cloud model)
        logger.info("Single dispatch escalated for query: '%s' (best score: %.3f)", utterance, confidence)
        llm_resolution = None
        if self.brain_llm_callback:
            try:
                llm_prompt = f"Resolve user request that no specialist skill matched: '{utterance}'"
                llm_resolution = self.brain_llm_callback(llm_prompt)
            except Exception as e:
                logger.warning("Brain LLM callback failed: %s", e)

        resolved_act = {"llm_response": llm_resolution or "Escalated to Brain LLM"}

        if best_id:
            self.registry.add_pending_example(best_id, {"utterance": utterance, "resolved_action": resolved_act})
            self.registry.record_usage(best_id, confidence=confidence, escalated=True, success=False)

        self.escalation_log.log(
            utterance=utterance,
            routed_skill_id=best_id,
            confidence=confidence,
            resolved_by="CLOUD_LLM" if self.brain_llm_callback else "BONSAI",
            outcome="ESCALATED",
            resolved_action=resolved_act,
        )
        self.hindsight_store.retain(f"Escalated request: '{utterance}' resolved by Brain LLM: {resolved_act}", source_type="ESCALATION")

        return {
            "status": "ESCALATED",
            "best_candidate": best_id,
            "confidence": confidence,
            "brain_resolution": resolved_act,
        }

    def handle_swarm(self, query: str) -> SwarmResult:
        """Swarm fan-out mode (e.g. security defense, research).

        Dispatches query simultaneously to all loaded skills in active cluster.
        """
        swarm_res = self.swarm_dispatcher.dispatch_swarm(query)
        self.hindsight_store.retain(f"Swarm fan-out for '{query}' across {len(self.working_set.get_loaded_ids())} skills", source_type="SWARM")
        return swarm_res

    def run_growth_cycle(self) -> Dict[str, Any]:
        """Trigger background trainer growth cycle."""
        return self.trainer.run_growth_cycle_with_dreaming()

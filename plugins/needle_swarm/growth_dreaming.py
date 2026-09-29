"""growth_dreaming.py - GrowthDreamer replaying escalation logs.

Replays history in the escalation log without fine-tuning compute to rank candidate retrains and
hypothetical new skill proposals.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from .escalation_log import EscalationEvent, EscalationLog
from .needle_ops import cosine_similarity, embed_text

logger = logging.getLogger(__name__)


class GrowthDreamer:
    """Dream-RSI replayer over escalation log history."""

    def __init__(self, escalation_log: EscalationLog):
        self.escalation_log = escalation_log

    def replay_thresholds(
        self,
        candidate_thresholds: List[float] = [0.4, 0.5, 0.55, 0.6, 0.7, 0.8],
    ) -> Dict[float, Dict[str, Any]]:
        """Replay alternate confidence thresholds against recorded events."""
        events = self.escalation_log.get_events()
        results: Dict[float, Dict[str, Any]] = {}

        for thresh in candidate_thresholds:
            auto_resolved = 0
            misfired = 0
            escalated = 0

            for ev in events:
                if ev.confidence >= thresh:
                    auto_resolved += 1
                    if ev.outcome in ("MISFIRE", "FAILED"):
                        misfired += 1
                else:
                    escalated += 1

            risk = (misfired / auto_resolved) if auto_resolved > 0 else 0.0
            results[thresh] = {
                "auto_resolved": auto_resolved,
                "misfired": misfired,
                "escalated": escalated,
                "misfire_risk": round(risk, 3),
            }

        return results

    def replay_hypothetical_skill(
        self,
        proposed_description: str,
        match_threshold: float = 0.55,
    ) -> Dict[str, Any]:
        """Evaluate how many unresolved escalations a proposed new skill description would capture."""
        unresolved = self.escalation_log.get_events(unresolved_only=True)
        if not unresolved:
            return {"proposed_description": proposed_description, "captured_count": 0, "coverage_pct": 0.0}

        prop_emb = embed_text(proposed_description)
        captured = []

        for ev in unresolved:
            sim = cosine_similarity(prop_emb, ev.embedding)
            if sim >= match_threshold:
                captured.append((ev, sim))

        pct = (len(captured) / len(unresolved)) * 100.0
        return {
            "proposed_description": proposed_description,
            "captured_count": len(captured),
            "total_unresolved": len(unresolved),
            "coverage_pct": round(pct, 1),
            "captured_event_ids": [ev.event_id for ev, _ in captured],
        }

    def rank_growth_candidates(
        self,
        retrain_proposals: List[Dict[str, Any]],
        new_skill_proposals: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Mix retrains and new-skill proposals into one evidence-ranked list."""
        ranked: List[Dict[str, Any]] = []

        for p in retrain_proposals:
            skill_id = p["skill_id"]
            events = self.escalation_log.get_events(skill_id=skill_id)
            score = len(events) * 2.0 + sum(1 for e in events if e.outcome == "MISFIRE") * 3.0
            ranked.append({
                "type": "RETRAIN",
                "target_id": skill_id,
                "dream_score": score,
                "evidence_count": len(events),
                "details": p,
            })

        for p in new_skill_proposals:
            desc = p.get("description", "")
            eval_res = self.replay_hypothetical_skill(desc)
            score = eval_res["captured_count"] * 2.5
            ranked.append({
                "type": "NEW_SKILL",
                "target_id": p.get("name", "new_skill"),
                "dream_score": score,
                "evidence_count": eval_res["captured_count"],
                "details": p,
            })

        ranked.sort(key=lambda x: x["dream_score"], reverse=True)
        return ranked

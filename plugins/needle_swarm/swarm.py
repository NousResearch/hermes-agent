"""swarm.py - SwarmDispatcher for thread-pool fan-out to active skills.

Dispatches input to all loaded skills simultaneously in swarm fan-out mode.
Merges candidate tool calls, retrieved snippets, and matched playbooks into a SwarmResult
that Bonsai or cloud LLMs can consume.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .working_set import WorkingSet

logger = logging.getLogger(__name__)


@dataclass
class SwarmResult:
    """Merged signal bundle from swarm fan-out."""

    query: str
    action_signals: List[Dict[str, Any]] = field(default_factory=list)
    memory_signals: List[Dict[str, Any]] = field(default_factory=list)
    playbook_signals: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)

    def as_bonsai_context(self) -> str:
        """Format signal bundle into LLM context snippet."""
        lines = [f"=== Swarm Signals for Query: '{self.query}' ==="]

        if self.action_signals:
            lines.append("\n-- Candidate Tool Dispatch Signals --")
            for sig in self.action_signals:
                lines.append(
                    f"  [Skill: {sig['skill_id']}] Tool: {sig['tool']} (conf: {sig['confidence']:.2f}) Args: {sig['args']}"
                )

        if self.memory_signals:
            lines.append("\n-- Memory / RAG Context Snippets --")
            for sig in self.memory_signals:
                lines.append(f"  [Corpus: {sig['skill_id']}] {sig['snippet']}")

        if self.playbook_signals:
            lines.append("\n-- Matched Playbooks / Procedures --")
            for sig in self.playbook_signals:
                lines.append(
                    f"  [Playbook: {sig['skill_id']}] Conf: {sig['confidence']:.2f} Action: {sig['procedure'].get('recommended_action')}"
                )
                for step in sig['procedure'].get("checklist", []):
                    lines.append(f"    - {step}")

        if not (self.action_signals or self.memory_signals or self.playbook_signals):
            lines.append("\n(No active specialist skills returned high-confidence signals)")

        return "\n".join(lines)


class SwarmDispatcher:
    """Parallel fan-out dispatcher over loaded skills."""

    def __init__(self, working_set: WorkingSet, max_workers: int = 8):
        self.working_set = working_set
        self.max_workers = max_workers

    def dispatch_swarm(self, query: str, target_skill_ids: Optional[List[str]] = None) -> SwarmResult:
        if target_skill_ids:
            skill_ids = target_skill_ids
        else:
            skill_ids = self.working_set.get_loaded_ids()

        result = SwarmResult(query=query)
        if not skill_ids:
            return result

        def _evaluate_skill(sid: str) -> Tuple[str, str, Any, float, Optional[Exception]]:
            inst = self.working_set.get(sid)
            if not inst:
                return sid, "UNKNOWN", None, 0.0, ValueError("Skill not loaded")

            try:
                if inst.kind == "ACTION":
                    tool, args, conf = inst.dispatch(query)
                    return sid, "ACTION", {"tool": tool, "args": args}, conf, None
                elif inst.kind == "PLAYBOOK":
                    proc, conf = inst.extract(query)
                    return sid, "PLAYBOOK", proc, conf, None
                elif inst.kind == "MEMORY":
                    # Simple RAG retrieval simulation
                    snippet = f"Relevant snippet from {sid} for '{query}'"
                    return sid, "MEMORY", {"snippet": snippet}, 0.8, None
                else:
                    tool, args, conf = inst.dispatch(query)
                    return sid, inst.kind, {"tool": tool, "args": args}, conf, None
            except Exception as e:
                return sid, inst.kind, None, 0.0, e

        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(skill_ids))) as executor:
            future_to_sid = {executor.submit(_evaluate_skill, sid): sid for sid in skill_ids}
            for future in as_completed(future_to_sid):
                sid = future_to_sid[future]
                try:
                    sid_res, kind, payload, conf, err = future.result()
                    if err:
                        logger.warning("Swarm skill %s failed: %s", sid, err)
                        result.errors.append({"skill_id": sid, "error": str(err)})
                        continue

                    if kind == "ACTION" and payload:
                        result.action_signals.append({
                            "skill_id": sid_res,
                            "tool": payload["tool"],
                            "args": payload["args"],
                            "confidence": conf,
                        })
                    elif kind == "PLAYBOOK" and payload:
                        result.playbook_signals.append({
                            "skill_id": sid_res,
                            "procedure": payload,
                            "confidence": conf,
                        })
                    elif kind == "MEMORY" and payload:
                        result.memory_signals.append({
                            "skill_id": sid_res,
                            "snippet": payload["snippet"],
                            "confidence": conf,
                        })
                except Exception as e:
                    result.errors.append({"skill_id": sid, "error": str(e)})

        return result

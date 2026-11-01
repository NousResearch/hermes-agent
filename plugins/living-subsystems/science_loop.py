"""ScienceLoop: hypothesis -> experiment -> evaluation (retain/discard/modify)."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .base import Subsystem, new_id, now, result

GOALS_FILE = "goals.json"
OUTCOMES = ("success", "failure", "inconclusive")
VERDICTS = ("retain", "discard", "modify")


class ScienceLoop(Subsystem):
    def _mutate(self, hypothesis_id: str, fn) -> Dict[str, Any]:
        with self.lock(GOALS_FILE):
            data = self.load(GOALS_FILE)
            hyp = data.get("hypotheses", {}).get(hypothesis_id)
            if hyp is None:
                return result(False, f"Unknown hypothesis: {hypothesis_id}")
            fn(hyp)
            hyp["updated"] = now()
            self.save(GOALS_FILE, data)
            return result(True, "ok", hypothesis=hyp)

    def add_hypothesis(self, description: str, category: str = "default") -> Dict[str, Any]:
        hyp = {"id": new_id("hyp"), "description": description, "category": category, "status": "open",
               "experiments": [], "evaluation": None, "created": now(), "updated": now()}
        with self.lock(GOALS_FILE):
            data = self.load(GOALS_FILE)
            data.setdefault("hypotheses", {})[hyp["id"]] = hyp
            self.save(GOALS_FILE, data)
        return result(True, "Hypothesis added", hypothesis=hyp)

    def record_experiment(self, hypothesis_id: str, outcome_text: str, outcome: str = "inconclusive") -> Dict[str, Any]:
        if outcome not in OUTCOMES:
            return result(False, f"outcome must be one of {OUTCOMES}")
        return self._mutate(hypothesis_id, lambda h: h["experiments"].append(
            {"result": outcome_text, "outcome": outcome, "ts": now()}))

    def evaluate_hypothesis(self, hypothesis_id: str, verdict: str,
                            evaluation_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if verdict not in VERDICTS:
            return result(False, f"verdict must be one of {VERDICTS}")

        def apply(h: Dict[str, Any]) -> None:
            h["status"] = {"retain": "retained", "discard": "discarded", "modify": "open"}[verdict]
            h["evaluation"] = {"verdict": verdict, "data": evaluation_data or {}, "ts": now()}
        return self._mutate(hypothesis_id, apply)

    def _hypotheses(self) -> Dict[str, Dict[str, Any]]:
        return self.load(GOALS_FILE).get("hypotheses", {})

    def status(self) -> Dict[str, Any]:
        counts: Dict[str, int] = {}
        for h in self._hypotheses().values():
            counts[h["status"]] = counts.get(h["status"], 0) + 1
        return result(True, f"{sum(counts.values())} hypotheses", counts=counts)

    def run(self, **kwargs: Any) -> Dict[str, Any]:
        """List open hypotheses that have experiments awaiting an evaluation."""
        ready = [h["id"] for h in self._hypotheses().values() if h["status"] == "open" and h["experiments"]]
        return result(True, f"{len(ready)} hypotheses ready for evaluation", ready=ready)

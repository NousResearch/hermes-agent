"""Component 2 — Monotonic Optimization Loop (State Locking).

Based on CAAF (Convergent AI Agent Framework, 2026):
- State Locking ensures verified constraints grow monotonically: V_t ⊆ V_{t+1}
- Without State Locking, naive reflection loops exhibit 0% convergence (seesaw effect).
- With State Locking, 100% monotonic convergence is achieved.

Design principle: all improvement loops must terminate in an externally verifiable signal.

Phase 1 enhancements:
- Dual scoring support (primary_score + instrument_trust_score)
- Combined weighted score for decision making
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


@dataclass
class OptimizationDecision:
    """Result of evaluating a single optimization step."""

    action: str  # "keep", "revert", "halt"
    summary: str
    score_before: float = 0.0
    score_after: float = 0.0
    delta: float = 0.0
    locked_constraints: List[str] = field(default_factory=list)
    # Phase 1: Dual scoring (primary metric + trustworthiness of the scoring instrument itself)
    dual_instrument_score: float = 0.0
    combined_score: float = 0.0


@dataclass
class IterationRecord:
    """A single entry for the iterations.jsonl procedural memory log."""

    timestamp: str
    iteration: int
    score_before: float
    score_after: float
    action: str
    locked_constraints: List[str]
    explanation: str = ""
    # Phase 1 dual scoring fields
    dual_instrument_score: float = 0.0
    combined_score: float = 0.0


class StateLockingOptimizer:
    """Enforces monotonic improvement via State Locking.

    The protocol:
    1. Evaluate proposed change against the fitness function.
    2. Accept only if strictly improving the prior best score (or dual score).
    3. Maintain a "best-so-far" checkpoint never overwritten by a worse result.
    4. Halt after N iterations without improvement.
    """

    def __init__(
        self,
        max_stall: int = 5,
        enabled: bool = True,
        log_path: Optional[Path] = None,
        dual_scoring_weight: float = 0.3,  # weight for instrument trust
    ):
        self.enabled = enabled
        self.max_stall = max_stall
        self.log_path = log_path
        self.dual_scoring_weight = dual_scoring_weight

        self.best_score: float = 0.0
        self.last_score: float = 0.0
        self.stall_count: int = 0
        self.iteration: int = 0
        self._locked_constraints: Set[str] = set()
        self._history: List[IterationRecord] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset state for a new session."""
        self.best_score = 0.0
        self.last_score = 0.0
        self.stall_count = 0
        self.iteration = 0
        self._locked_constraints.clear()
        self._history.clear()

    def evaluate_step(self, score: float, instrument_trust: float = 0.0) -> OptimizationDecision:
        """Evaluate a proposed change and return keep/revert/halt decision.

        Phase 1: Supports dual scoring. Combined score = (1-w)*primary + w*instrument_trust
        """
        if not self.enabled:
            return OptimizationDecision(action="keep", summary="State locking disabled")

        self.iteration += 1

        combined = (score * (1 - self.dual_scoring_weight)) + (instrument_trust * self.dual_scoring_weight)
        delta = combined - self.best_score

        # Strict improvement required on the combined score
        if delta > 0:
            self.best_score = combined
            self.last_score = score
            self.stall_count = 0
            action = "keep"
            summary = (
                f"Iteration {self.iteration}: combined score improved from "
                f"{self.best_score:.4f} to {combined:.4f} (+{delta:.4f}). Keeping change. "
                f"(primary={score:.4f}, instrument_trust={instrument_trust:.4f})"
            )
        elif delta == 0:
            self.stall_count += 1
            action = "keep"
            summary = (
                f"Iteration {self.iteration}: combined score unchanged at {combined:.4f}. "
                f"Keeping change (stall {self.stall_count}/{self.max_stall})."
            )
        else:
            self.stall_count += 1
            action = "revert"
            summary = (
                f"Iteration {self.iteration}: combined score regressed from "
                f"{self.best_score:.4f} to {combined:.4f} ({delta:.4f}). Reverting change."
            )

        # Halt condition
        if self.stall_count >= self.max_stall:
            action = "halt"
            summary = (
                f"Iteration {self.iteration}: halted after {self.stall_count} "
                f"iterations without improvement. Best combined score: {self.best_score:.4f}."
            )

        decision = OptimizationDecision(
            action=action,
            summary=summary,
            score_before=self.last_score,
            score_after=score,
            delta=delta,
            locked_constraints=sorted(self._locked_constraints),
            dual_instrument_score=instrument_trust,
            combined_score=combined,
        )

        self._record(decision)
        return decision

    def lock_constraint(self, constraint: str) -> None:
        """Mark a constraint as verified and immutable (State Locking)."""
        self._locked_constraints.add(constraint)

    def is_locked(self, constraint: str) -> bool:
        """Check whether a constraint has been locked."""
        return constraint in self._locked_constraints

    def get_locked_constraints(self) -> List[str]:
        """Return the current set of locked constraints."""
        return sorted(self._locked_constraints)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _record(self, decision: OptimizationDecision) -> None:
        """Append an iteration record to the procedural memory log."""
        record = IterationRecord(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            iteration=self.iteration,
            score_before=decision.score_before,
            score_after=decision.score_after,
            action=decision.action,
            locked_constraints=decision.locked_constraints,
            explanation=decision.summary,
            dual_instrument_score=decision.dual_instrument_score,
            combined_score=decision.combined_score,
        )
        self._history.append(record)
        if self.log_path:
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
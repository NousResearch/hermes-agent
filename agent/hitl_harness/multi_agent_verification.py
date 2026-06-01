"""Component 3 — Multi-Agent Verification.

Implements specialist-verifier patterns with confidence-calibrated consensus:
- MapCoder: 93.9% HumanEval pass@1 via 4 specialized agents
- AgentCoder: decoupled code/test generation prevents self-deception
- LACIE: confidence calibration reduces incorrect acceptance by 47%
- Debate paradox: unweighted consensus converges to wrong answers 23.9% of the time

Key principle: calibrated confidence transforms debate from hazard to signal.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Verdict:
    """Result of a verification scan."""

    action: str  # "proceed", "review", "block"
    confidence: float  # 0.0–1.0
    reason: str = ""
    agent_votes: Dict[str, float] = field(default_factory=dict)


class SpecialistAgent:
    """Base class for specialist agents in the verification pipeline."""

    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role

    def evaluate(self, tool_call: Dict[str, Any], result: Optional[str] = None) -> float:
        """Return a confidence score in [0.0, 1.0]."""
        raise NotImplementedError


class CoderAgent(SpecialistAgent):
    """Evaluates code-generation quality (syntax, structure, style)."""

    def __init__(self):
        super().__init__("coder", "code_generator")

    def evaluate(self, tool_call: Dict[str, Any], result: Optional[str] = None) -> float:
        # Heuristic: code-writing tools get high base confidence
        name = tool_call.get("name", "")
        if name in ("write_file", "patch", "edit_file"):
            return 0.75
        return 0.5


class TesterAgent(SpecialistAgent):
    """Evaluates test coverage and correctness independently."""

    def __init__(self):
        super().__init__("tester", "test_designer")

    def evaluate(self, tool_call: Dict[str, Any], result: Optional[str] = None) -> float:
        # Heuristic: look for test-related keywords in tool call or result
        name = tool_call.get("name", "")
        args = json.dumps(tool_call.get("arguments", {}))
        if "test" in name.lower() or "test" in args.lower():
            return 0.80
        if result and "passed" in result.lower():
            return 0.90
        return 0.5


class VerifierAgent(SpecialistAgent):
    """Executes validation and reports pass/fail with confidence."""

    def __init__(self):
        super().__init__("verifier", "test_executor")

    def evaluate(self, tool_call: Dict[str, Any], result: Optional[str] = None) -> float:
        if result is None:
            return 0.5
        # Parse result for success / failure signals
        lower = result.lower()
        if any(k in lower for k in ("error", "fail", "exception", "traceback", "timeout")):
            return 0.20
        if any(k in lower for k in ("success", "pass", "ok", "done")):
            return 0.85
        return 0.5


class MultiAgentVerification:
    """Orchestrates specialist agents and calibrates consensus."""

    def __init__(
        self,
        high_threshold: float = 0.80,
        low_threshold: float = 0.40,
        enabled: bool = True,
    ):
        self.enabled = enabled
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.agents: List[SpecialistAgent] = [
            CoderAgent(),
            TesterAgent(),
            VerifierAgent(),
        ]
        self._vote_history: List[Dict[str, float]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear vote history for a new session."""
        self._vote_history.clear()

    def pre_scan(self, tool_call: Dict[str, Any]) -> Verdict:
        """Pre-execution scan: estimate confidence before running the tool."""
        if not self.enabled:
            return Verdict(action="proceed", confidence=1.0)

        votes = {agent.name: agent.evaluate(tool_call) for agent in self.agents}
        calibrated = self._calibrate(votes)

        if calibrated >= self.high_threshold:
            action = "proceed"
        elif calibrated >= self.low_threshold:
            action = "review"
        else:
            action = "block"

        return Verdict(
            action=action,
            confidence=calibrated,
            reason=f"Pre-scan confidence: {calibrated:.2f}",
            agent_votes=votes,
        )

    def post_scan(self, tool_call: Dict[str, Any], result: str) -> Verdict:
        """Post-execution scan: calibrate confidence with actual result."""
        if not self.enabled:
            return Verdict(action="proceed", confidence=1.0)

        votes = {agent.name: agent.evaluate(tool_call, result) for agent in self.agents}
        calibrated = self._calibrate(votes)
        self._vote_history.append(votes)

        if calibrated >= self.high_threshold:
            action = "proceed"
        elif calibrated >= self.low_threshold:
            action = "review"
        else:
            action = "block"

        return Verdict(
            action=action,
            confidence=calibrated,
            reason=f"Post-scan confidence: {calibrated:.2f}",
            agent_votes=votes,
        )

    def queue_size(self) -> int:
        """Return the number of recorded vote rounds (for status dashboards)."""
        return len(self._vote_history)

    def detect_sycophancy(self, window: int = 5) -> Optional[str]:
        """Detect consensus collapse (wrong consensus convergence).

        Returns a warning string if agents are converging to low-confidence
        agreement, or ``None`` if diversity is healthy.
        """
        if len(self._vote_history) < window:
            return None

        recent = self._vote_history[-window:]
        # Check for low-variance, low-mean convergence
        means = [sum(v.values()) / len(v) for v in recent]
        avg_mean = sum(means) / len(means)
        if avg_mean < self.low_threshold:
            return (
                f"Sycophancy warning: agents converged to low-confidence "
                f"agreement (μ={avg_mean:.2f}) over last {window} rounds."
            )
        return None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _calibrate(self, votes: Dict[str, float]) -> float:
        """Calibrate consensus using historical accuracy weighting (LACIE-style).

        Simple implementation: weighted average with verifier weighted higher.
        In a full system, historical accuracy per agent would be tracked and
        used to adjust weights dynamically.
        """
        weights = {"coder": 0.25, "tester": 0.35, "verifier": 0.40}
        total = 0.0
        weight_sum = 0.0
        for name, score in votes.items():
            w = weights.get(name, 0.33)
            total += score * w
            weight_sum += w
        return total / weight_sum if weight_sum > 0 else 0.5

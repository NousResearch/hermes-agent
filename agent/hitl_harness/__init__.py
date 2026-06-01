"""Human-in-the-Loop (HITL) Agent Harness for Hermes Agent.

Implements the five-component reference architecture from the HITL research report:

1. Fitness Function Wizard      — intent → executable GOAL.md
2. Monotonic Optimization Loop  — State Locking for convergent self-improvement
3. Multi-Agent Verification     — specialist-verifier with confidence calibration
4. Layered Observability        — 4-layer explanation stack
5. Graduated Autonomy Gate      — L0–L4 autonomy with cognitive forcing functions

Phase 1 additions (research-backed):
- deepeval_adapter: native agentic metrics
- iterations_logger: first-class iterations.jsonl with dual scoring
- Dual scoring primitive in StateLockingOptimizer

Usage::

    from agent.hitl_harness import HITLHarness

    harness = HITLHarness(agent, config)
    harness.on_iteration_start(messages)
    ...
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.hitl_harness.deepeval_adapter import DeepevalAdapter, default_adapter
from agent.hitl_harness.fitness_wizard import FitnessFunctionWizard
from agent.hitl_harness.graduated_autonomy import AutonomyLevel, GraduatedAutonomyGate
from agent.hitl_harness.iterations_logger import IterationsLogger, IterationLogEntry
from agent.hitl_harness.layered_observability import LayeredObservability
from agent.hitl_harness.multi_agent_verification import MultiAgentVerification
from agent.hitl_harness.fitness_engine import FitnessEngine, FitnessResult
from agent.hitl_harness.swe_bench_adapter import SWEBenchAdapter
from agent.hitl_harness.telemetry_feedback import TelemetryFeedback

from agent.hitl_harness.state_locking import (
    IterationRecord,
    OptimizationDecision,
    StateLockingOptimizer,
)


@dataclass
class HITLHarnessConfig:
    """Runtime configuration for the HITL harness."""

    enabled: bool = True
    autonomy_level: int = 1  # L0–L4
    state_locking: bool = True
    multi_agent_verification: bool = True
    observability_layer: int = 1  # default visible layer (1–4)
    fitness_wizard: bool = True
    iterations_log_path: str = "~/.hermes/iterations.jsonl"
    max_iterations_without_improvement: int = 5
    confidence_threshold_high: float = 0.80
    confidence_threshold_low: float = 0.40
    dual_scoring: bool = True
    include_deepeval: bool = True

    @classmethod
    def from_agent_config(cls, config: Dict[str, Any]) -> "HITLHarnessConfig":
        hitl = config.get("hitl", {}) if isinstance(config, dict) else {}
        return cls(
            enabled=hitl.get("enabled", True),
            autonomy_level=hitl.get("autonomy_level", 1),
            state_locking=hitl.get("state_locking", True),
            multi_agent_verification=hitl.get("multi_agent_verification", True),
            observability_layer=hitl.get("observability_layer", 1),
            fitness_wizard=hitl.get("fitness_wizard", True),
            iterations_log_path=hitl.get("iterations_log_path", "~/.hermes/iterations.jsonl"),
            max_iterations_without_improvement=hitl.get("max_iterations_without_improvement", 5),
            confidence_threshold_high=hitl.get("confidence_threshold_high", 0.80),
            confidence_threshold_low=hitl.get("confidence_threshold_low", 0.40),
            dual_scoring=hitl.get("dual_scoring", True),
            include_deepeval=hitl.get("include_deepeval", True),
        )


class HITLHarness:
    """Central orchestrator for the HITL agent harness.

    Phase 1: Now wires deepeval_adapter, iterations_logger, and dual scoring
    into the iteration lifecycle.
    """

    def __init__(
        self,
        agent: Any,
        config: Optional[HITLHarnessConfig] = None,
    ):
        self.agent = agent
        self.cfg = config or HITLHarnessConfig()
        self._iterations_log_path = Path(self.cfg.iterations_log_path).expanduser()
        self._iterations_log_path.parent.mkdir(parents=True, exist_ok=True)

        # Component 1 — Fitness Function Wizard
        self.fitness_wizard = FitnessFunctionWizard(
            goals_dir=Path("~/.hermes/goals").expanduser(),
            enabled=self.cfg.fitness_wizard,
        )

        # Component 2 — Monotonic Optimization Loop (with dual scoring)
        self.optimizer = StateLockingOptimizer(
            max_stall=self.cfg.max_iterations_without_improvement,
            enabled=self.cfg.state_locking,
            log_path=self._iterations_log_path,
        )

        # Phase 1: Dedicated iterations logger (first-class artifact)
        self.iterations_logger = IterationsLogger(
            log_path=self._iterations_log_path,
            session_id="current",
        )

        # Component 3 — Multi-Agent Verification
        self.verification = MultiAgentVerification(
            high_threshold=self.cfg.confidence_threshold_high,
            low_threshold=self.cfg.confidence_threshold_low,
            enabled=self.cfg.multi_agent_verification,
        )

        # Component 4 — Layered Observability
        self.observability = LayeredObservability(
            default_layer=self.cfg.observability_layer,
            log_path=self._iterations_log_path,
        )

        # Component 5 — Graduated Autonomy Gate
        self.autonomy = GraduatedAutonomyGate(
            level=AutonomyLevel(self.cfg.autonomy_level),
            enabled=True,
        )

        # Phase 1: deepeval adapter
        self.deepeval = default_adapter if self.cfg.include_deepeval else None

        # Session-scoped state
        self._session_fitness_score: Optional[float] = None
        self._iteration_count = 0

    # ------------------------------------------------------------------
    # Lifecycle hooks (called from run_agent.py / model_tools.py)
    # ------------------------------------------------------------------

    def on_session_start(self, session_id: str, system_message: str) -> None:
        if not self.cfg.enabled:
            return
        self._iteration_count = 0
        self._session_fitness_score = None
        self.optimizer.reset()
        self.verification.reset()
        self.observability.start_session(session_id)
        self.iterations_logger.session_id = session_id

        goal = self.fitness_wizard.load_goal(session_id)
        if goal:
            self.observability.emit(
                layer=1,
                message=f"Loaded fitness function: {goal.get('name', 'unnamed')}",
                metadata={"goal_id": goal.get("id")},
            )

    def on_iteration_start(self, messages: List[Dict[str, Any]]) -> None:
        if not self.cfg.enabled:
            return
        self._iteration_count += 1
        self.observability.emit(
            layer=4,
            message=f"Iteration {self._iteration_count} started",
            metadata={"message_count": len(messages)},
        )

    def pre_tool_call(self, tool_call: Dict[str, Any]) -> Optional[str]:
        if not self.cfg.enabled:
            return None
        block_reason = self.autonomy.check_tool_permission(tool_call)
        if block_reason:
            self.observability.emit(
                layer=2,
                message=f"Tool blocked by autonomy gate: {block_reason}",
                metadata={"tool": tool_call.get("name"), "reason": block_reason},
            )
            return block_reason
        verdict = self.verification.pre_scan(tool_call)
        if verdict.action == "block":
            self.observability.emit(
                layer=2,
                message=f"Tool blocked by verification: {verdict.reason}",
                metadata={"tool": tool_call.get("name"), "confidence": verdict.confidence},
            )
            return verdict.reason
        return None

    def post_tool_call(self, tool_call: Dict[str, Any], result: str) -> str:
        if not self.cfg.enabled:
            return result
        verdict = self.verification.post_scan(tool_call, result)
        self.observability.emit(
            layer=3,
            message=f"Tool {tool_call.get('name')} completed",
            metadata={
                "tool": tool_call.get("name"),
                "confidence": verdict.confidence,
                "action": verdict.action,
            },
        )
        if verdict.confidence is not None:
            try:
                parsed = json.loads(result)
                if isinstance(parsed, dict):
                    parsed["_hitl_verification"] = {
                        "confidence": round(verdict.confidence, 4),
                        "action": verdict.action,
                    }
                    return json.dumps(parsed)
            except Exception:
                pass
        return result

    def on_iteration_end(self, response: Dict[str, Any], trajectory: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Called after the LLM response is received.

        Phase 1 wiring:
        - Uses enhanced evaluate_current_state (dual scores + deepeval)
        - Passes instrument_trust to StateLockingOptimizer
        - Logs via first-class IterationsLogger
        """
        if not self.cfg.enabled:
            return response

        # Build a minimal trajectory if none provided
        if trajectory is None:
            trajectory = {
                "messages": response.get("messages", []),
                "tool_calls": response.get("tool_calls", []),
                "final_response": str(response.get("content", "")),
            }

        # Phase 1: Get dual scores from wizard (now supports deepeval)
        fitness_result = self.fitness_wizard.evaluate_current_state(
            self.agent,
            deepeval_adapter=self.deepeval,
            trajectory=trajectory,
        )

        if fitness_result:
            primary = fitness_result["primary_score"]
            instrument = fitness_result["instrument_trust_score"]

            # State Locking with dual scoring
            decision = self.optimizer.evaluate_step(primary, instrument_trust=instrument)

            # First-class logging
            self.iterations_logger.log_iteration(
                primary_score=primary,
                instrument_trust_score=instrument,
                action=decision.action,
                iteration=self._iteration_count,
                locked_constraints=decision.locked_constraints,
                explanation=decision.summary,
            )

            self.observability.emit(
                layer=1,
                message=decision.summary,
                metadata={
                    "primary_score": primary,
                    "instrument_trust_score": instrument,
                    "combined_score": fitness_result["combined_score"],
                    "best_score": self.optimizer.best_score,
                    "action": decision.action,
                },
            )

            if decision.action == "halt":
                response["_hitl_halt"] = True
                response["_hitl_halt_reason"] = decision.summary

        return response

    def on_session_end(self) -> None:
        if not self.cfg.enabled:
            return
        self.observability.end_session()
        self._flush_iterations_log()

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _flush_iterations_log(self) -> None:
        pass

    def get_status(self) -> Dict[str, Any]:
        return {
            "enabled": self.cfg.enabled,
            "autonomy_level": self.autonomy.level.name,
            "iteration_count": self._iteration_count,
            "best_fitness_score": self.optimizer.best_score,
            "last_fitness_score": self.optimizer.last_score,
            "stall_count": self.optimizer.stall_count,
            "verification_queue_size": self.verification.queue_size(),
            "observability_events": self.observability.event_count(),
            "goal_loaded": self.fitness_wizard.current_goal is not None,
            "dual_scoring_enabled": self.cfg.dual_scoring,
        }

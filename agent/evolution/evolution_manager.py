"""Evolution Manager — central orchestrator for the Evolution Engine.

Follows the MemoryManager pattern: one manager instance per agent session,
pluggable evaluation/analysis backends, tools gated on configuration.

Lifecycle (wired into run_agent.py):
  1. initialize() — load config, open store, resolve auxiliary model
  2. start_task(task) — begin tracking a task attempt
  3. TrajectoryCollector hooks fire during agent execution
  4. evaluate() — score the completed attempt
  5. If failed: analyze() → propose() → gate() → apply() → retry
  6. shutdown() — persist state, close store

Core loop (per-task):
  ┌─────────────────────────────────────────────┐
  │  Task → Attempt → Capture → Evaluate         │
  │    ↑                                      ↓  │
  │    ├── Apply ← Gate ← Propose ← Analyze ←── │
  │    │        (safe?)  (fixes)   (root cause)  │
  │    └─────────────────────────────────────────│
  │              Retry if failed                  │
  └─────────────────────────────────────────────┘
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from agent.evolution.config import EvolutionConfig
from agent.evolution.task_definition import (
    ImprovementActionType,
    TaskDefinition,
    TaskStatus,
    load_task,
    list_tasks,
)
from agent.evolution.trajectory_collector import (
    EvalResult,
    Trajectory,
    TrajectoryCollector,
)
from agent.evolution.evaluator import TaskEvaluator, EvaluationContext
from agent.evolution.failure_analyzer import FailureAnalysis, FailureAnalyzer
from agent.evolution.improvement_proposer import ImprovementProposal, ImprovementProposer
from agent.evolution.regression_gate import GateVerdict, RegressionGate
from agent.evolution.evolution_store import EvolutionStore, get_evolution_store
from agent.evolution.harness_variants import VariantManager

logger = logging.getLogger(__name__)

# Maximum time (seconds) to wait for async evolution work
_EVOLUTION_DRAIN_TIMEOUT_S = 10.0


# ---------------------------------------------------------------------------
# Evolution run state
# ---------------------------------------------------------------------------


@dataclass
class EvolutionRun:
    """Tracks the state of a single evolution run (one task)."""

    run_id: str
    task: TaskDefinition
    status: TaskStatus = TaskStatus.PENDING
    iteration: int = 0
    max_iterations: int = 5

    # Current iteration state
    collector: Optional[TrajectoryCollector] = None
    trajectory: Optional[Trajectory] = None
    eval_result: Optional[EvalResult] = None
    analysis: Optional[FailureAnalysis] = None
    proposals: List[ImprovementProposal] = field(default_factory=list)
    applied_proposals: List[ImprovementProposal] = field(default_factory=list)

    # Timestamps
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    @property
    def is_done(self) -> bool:
        return self.status in (
            TaskStatus.SUCCEEDED,
            TaskStatus.FAILED,
            TaskStatus.EXHAUSTED,
            TaskStatus.CANCELLED,
        )

    @property
    def can_continue(self) -> bool:
        return (
            not self.is_done
            and self.iteration < self.max_iterations
            and self.status != TaskStatus.CANCELLED
        )


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class EvolutionManager:
    """Central orchestrator for autonomous agent evolution.

    Usage in run_agent.py::

        agent._evolution_manager = EvolutionManager()
        agent._evolution_manager.initialize(session_id=agent.session_id)

        # When agent starts a tracked task:
        run = agent._evolution_manager.start_task(task)

        # During agent execution, hooks fire:
        run.collector.record_model_call(...)
        run.collector.record_tool_call(...)

        # After task completion:
        result = agent._evolution_manager.evaluate(run)

        # If failed, improve:
        if not result.passed:
            improved = agent._evolution_manager.improve_and_retry(run)
    """

    def __init__(self):
        self._config = EvolutionConfig()
        self._store: Optional[EvolutionStore] = None
        self._evaluator: Optional[TaskEvaluator] = None
        self._analyzer: Optional[FailureAnalyzer] = None
        self._proposer: Optional[ImprovementProposer] = None
        self._gate: Optional[RegressionGate] = None
        self._variants: Optional[VariantManager] = None
        self._active_run: Optional[EvolutionRun] = None
        self._session_id: str = ""
        self._lock = threading.Lock()
        self._initialized = False

        # External callbacks (set by agent init)
        self._llm_call_fn: Optional[Callable] = None  # For LLM-based analysis/proposals
        self._apply_skill_fn: Optional[Callable] = None  # skill_manager_tool integration
        self._apply_prompt_fn: Optional[Callable] = None
        self._apply_tool_fn: Optional[Callable] = None
        self._background_executor = None  # ThreadPoolExecutor for async work

    # -- Lifecycle -------------------------------------------------------------

    def initialize(
        self,
        session_id: str = "",
        config: Optional[EvolutionConfig] = None,
        llm_call_fn: Optional[Callable] = None,
        apply_skill_fn: Optional[Callable] = None,
        apply_prompt_fn: Optional[Callable] = None,
        apply_tool_fn: Optional[Callable] = None,
    ) -> None:
        """Initialize the Evolution Engine for a session.

        Called once at agent startup. Loads config, opens store, resolves
        auxiliary model.

        Args:
            session_id: The agent's session ID.
            config: Optional pre-built config (loads from config.yaml if None).
            llm_call_fn: Callable for LLM-based analysis/proposal generation.
            apply_skill_fn: Callable to apply skill changes via skill_manager_tool.
            apply_prompt_fn: Callable to apply prompt modifications.
            apply_tool_fn: Callable to register/modify tools.
        """
        with self._lock:
            if config is not None:
                self._config = config
            else:
                self._config = EvolutionConfig.from_config()

            self._session_id = session_id
            self._llm_call_fn = llm_call_fn
            self._apply_skill_fn = apply_skill_fn
            self._apply_prompt_fn = apply_prompt_fn
            self._apply_tool_fn = apply_tool_fn

            # Validate config
            errors = self._config.validate()
            if errors:
                logger.warning("Evolution config has %d error(s): %s", len(errors), errors)

            # Open store
            self._store = get_evolution_store()

            # Initialize components
            self._evaluator = TaskEvaluator(llm_judge_fn=llm_call_fn)
            self._analyzer = FailureAnalyzer(llm_analyze_fn=llm_call_fn)
            self._proposer = ImprovementProposer(llm_propose_fn=llm_call_fn)
            self._gate = RegressionGate(
                max_regression_tasks=self._config.max_regression_tasks,
            )
            self._variants = VariantManager.load()

            self._initialized = True
            logger.info(
                "Evolution Engine initialized (mode=%s, max_iter=%d, session=%s)",
                self._config.mode,
                self._config.max_iterations,
                session_id,
            )

    def shutdown(self) -> None:
        """Clean shutdown — persist state, close connections."""
        with self._lock:
            if self._variants:
                try:
                    self._variants.save()
                except Exception as e:
                    logger.debug("Failed to save variant state: %s", e)
            if self._store:
                try:
                    self._store.close()
                except Exception:
                    pass
            self._initialized = False

    @property
    def is_enabled(self) -> bool:
        return self._initialized and self._config.enabled

    # -- Task lifecycle --------------------------------------------------------

    def start_task(self, task: TaskDefinition) -> EvolutionRun:
        """Begin tracking an evolution run for a task.

        Creates the run record, initializes the trajectory collector,
        and routes the task to the appropriate harness variant.
        """
        if not self.is_enabled:
            raise RuntimeError("Evolution Engine is not enabled")

        with self._lock:
            # Route to best variant
            variant = self._variants.route_task(task.name, task.domain) if self._variants else None
            variant_id = variant.variant_id if variant else "default"

            # Create run in store
            run_id = self._store.create_run(
                task_name=task.name,
                task_domain=task.domain,
                task_complexity=task.complexity,
                max_iterations=self._config.max_iterations,
                session_id=self._session_id,
                harness_variant=variant_id,
            )

            # Create collector
            collector = TrajectoryCollector(
                task_name=task.name,
                run_id=run_id,
            )

            run = EvolutionRun(
                run_id=run_id,
                task=task,
                max_iterations=self._config.max_iterations,
                collector=collector,
                started_at=datetime.now(timezone.utc).isoformat(),
            )

            self._active_run = run
            collector.start()

            logger.info(
                "Started evolution run %s for task '%s' (variant=%s, iteration=1/%d)",
                run_id, task.name, variant_id, self._config.max_iterations,
            )
            return run

    def end_task(self, run: EvolutionRun) -> EvolutionRun:
        """Stop tracking and finalize the run."""
        with self._lock:
            if run.collector and run.collector.is_active:
                run.trajectory = run.collector.stop(
                    status="success" if (run.eval_result and run.eval_result.passed) else "failed"
                )

            # Save trajectory and finalize status
            if run.trajectory:
                try:
                    trace_path = run.collector.save()
                    self._store.update_run_status(
                        run.run_id,
                        run.status.value,
                        trace_path=str(trace_path),
                        completed_at=datetime.now(timezone.utc).isoformat(),
                        final_score=run.eval_result.score if run.eval_result else None,
                        iterations=run.iteration,
                    )
                except Exception as e:
                    logger.warning("Failed to save trajectory: %s", e)

            if self._active_run and self._active_run.run_id == run.run_id:
                self._active_run = None

            run.completed_at = datetime.now(timezone.utc).isoformat()
            return run

    # -- Evaluation ------------------------------------------------------------

    def evaluate(self, run: EvolutionRun) -> EvalResult:
        """Evaluate a completed task attempt."""
        if not run.trajectory:
            if run.collector:
                run.trajectory = run.collector.stop()
            else:
                return EvalResult(passed=False, score=0.0)

        run.status = TaskStatus.EVALUATING

        result = self._evaluator.evaluate(
            run.task,
            run.trajectory,
            EvaluationContext(trajectory=run.trajectory),
        )

        run.eval_result = result

        # Record iteration
        iteration_num = run.iteration + 1
        self._store.add_iteration(
            run_id=run.run_id,
            iteration_num=iteration_num,
            status="evaluating",
            score=result.score,
            trace_json=run.trajectory.to_json(),
            eval_json=_eval_to_json(result),
        )

        if result.passed:
            run.status = TaskStatus.SUCCEEDED
            # Record regression baseline
            self._store.set_baseline(
                task_name=run.task.name,
                score=result.score,
                task_domain=run.task.domain,
            )
            # Update variant tracking
            if self._variants:
                variant = self._variants.active_variant
                variant.record_result(run.task.name, result.score, True)

            # Track improvement if this was a retry (iteration > 0)
            if run.iteration > 0:
                try:
                    from agent.evolution.improvement_metrics import get_tracker
                    prev_score = 0.0
                    prev_iterations = self._store.get_iterations(run.run_id)
                    for it in prev_iterations:
                        if it.get("score") is not None:
                            prev_score = it["score"]
                            break
                    get_tracker().record_improvement(
                        task_name=run.task.name,
                        score_before=prev_score,
                        score_after=result.score,
                        run_id=run.run_id,
                    )
                except Exception:
                    pass

            logger.info("Task '%s' SUCCEEDED (score=%.2f, run=%s)", run.task.name, result.score, run.run_id)
        else:
            run.status = TaskStatus.FAILED
            logger.info("Task '%s' FAILED (score=%.2f, run=%s)", run.task.name, result.score, run.run_id)

        return result

    # -- Evolution loop --------------------------------------------------------

    def improve_and_retry(self, run: EvolutionRun) -> bool:
        """Try to improve and retry a failed task.

        Returns True if a retry was initiated, False if exhausted.
        """
        if not run.can_continue:
            logger.info("Run %s cannot continue (status=%s, iter=%d/%d)",
                       run.run_id, run.status.value, run.iteration, run.max_iterations)
            return False

        run.iteration += 1
        run.status = TaskStatus.ANALYZING

        # Step 1: Analyze failure
        analysis = self._analyzer.analyze(run.task, run.trajectory, run.eval_result)
        run.analysis = analysis
        self._store.add_iteration(
            run_id=run.run_id,
            iteration_num=run.iteration,
            status="analyzing",
            analysis_json=analysis.to_json(),
        )

        if not analysis.findings:
            logger.info("No actionable findings for run %s — exhausting", run.run_id)
            run.status = TaskStatus.EXHAUSTED
            return False

        # Step 2: Generate improvement proposals
        run.status = TaskStatus.IMPROVING
        existing_skills = _list_existing_skills()
        existing_tools = _list_existing_tools()
        proposals = self._proposer.propose(
            run.task, analysis,
            existing_skills=existing_skills,
            existing_tools=existing_tools,
        )
        run.proposals = proposals

        if not proposals:
            logger.info("No improvement proposals generated for run %s", run.run_id)
            run.status = TaskStatus.EXHAUSTED
            return False

        # Step 3: Gate each proposal, apply accepted ones
        applied = 0
        already_applied = {(p.action_type.value, p.target) for p in run.applied_proposals}
        for proposal in proposals:
            # Skip if already applied in a previous iteration
            if (proposal.action_type.value, proposal.target) in already_applied:
                continue
            gate_result = self._gate.evaluate(proposal)
            self._store.add_iteration(
                run_id=run.run_id,
                iteration_num=run.iteration,
                status="improving" if gate_result.passed else "analyzing",
                improvement_action=proposal.action_type.value,
                improvement_target=proposal.target,
                proposal_json=proposal.to_json(),
            )

            if gate_result.is_blocked:
                logger.info("Proposal '%s' BLOCKED by gate: %s", proposal.target, gate_result.failures)
                continue

            if gate_result.verdict == GateVerdict.NEEDS_REVIEW:
                # For now, auto-accept skill proposals; require approval for others
                if proposal.requires_approval and not proposal.action_type.value in ("skill_create", "skill_patch"):
                    logger.info("Proposal '%s' NEEDS REVIEW — skipping (requires human approval)", proposal.target)
                    continue

            # Apply the proposal
            success = self._apply_proposal(proposal, run)
            if success:
                applied += 1
                run.applied_proposals.append(proposal)
                logger.info("Applied proposal: %s → %s", proposal.action_type.value, proposal.target)

        if applied == 0:
            logger.info("No proposals were applicable for run %s", run.run_id)
            run.status = TaskStatus.EXHAUSTED
            return False

        # Step 4: Set up retry
        run.trajectory = None
        run.eval_result = None
        run.status = TaskStatus.ATTEMPTING

        # Reset collector for new attempt
        run.collector = TrajectoryCollector(
            task_name=run.task.name,
            run_id=run.run_id,
        )
        run.collector.start()

        logger.info(
            "Retry initiated for run %s (iteration %d/%d, %d proposals applied)",
            run.run_id, run.iteration, run.max_iterations, applied,
        )
        return True

    # -- Proposal application -------------------------------------------------

    def _apply_proposal(self, proposal: ImprovementProposal, run: EvolutionRun) -> bool:
        """Apply an approved improvement proposal."""
        try:
            if proposal.action_type == ImprovementActionType.SKILL_CREATE:
                return self._apply_skill_create(proposal)
            elif proposal.action_type == ImprovementActionType.SKILL_PATCH:
                return self._apply_skill_patch(proposal)
            elif proposal.action_type == ImprovementActionType.TOOL_CREATE:
                return self._apply_tool_create(proposal)
            elif proposal.action_type == ImprovementActionType.TOOL_MODIFY:
                return self._apply_tool_modify(proposal)
            elif proposal.action_type == ImprovementActionType.PROMPT_MODIFY:
                return self._apply_prompt_modify(proposal)
            elif proposal.action_type == ImprovementActionType.MEMORY_UPDATE:
                return self._apply_memory_update(proposal)
            else:
                logger.warning("Unknown proposal action type: %s", proposal.action_type)
                return False
        except Exception as e:
            logger.error("Failed to apply proposal '%s': %s", proposal.target, e)
            return False

    def _apply_skill_create(self, proposal: ImprovementProposal) -> bool:
        """Create a new skill from a proposal."""
        if self._apply_skill_fn:
            return self._apply_skill_fn("create", name=proposal.target, content=proposal.content)
        # Fallback: write directly
        return _write_skill_file(proposal.target, proposal.content)

    def _apply_skill_patch(self, proposal: ImprovementProposal) -> bool:
        """Patch an existing skill."""
        if self._apply_skill_fn:
            return self._apply_skill_fn("patch", name=proposal.target,
                                       old_string=proposal.old_string,
                                       new_string=proposal.new_string)
        return False  # Patches require the tool infrastructure

    def _apply_tool_create(self, proposal: ImprovementProposal) -> bool:
        """Register a new tool from proposal code."""
        if self._apply_tool_fn:
            return self._apply_tool_fn("create", name=proposal.target, code=proposal.content)
        return _write_tool_file(proposal.target, proposal.content)

    def _apply_tool_modify(self, proposal: ImprovementProposal) -> bool:
        """Modify an existing tool — routes through PR proposer for safety."""
        try:
            from agent.evolution.pr_proposer import propose_code_fix
            result = propose_code_fix(
                failure_analysis={"findings": [{"category": "auto-detected",
                    "confidence": proposal.confidence, "description": proposal.rationale,
                    "evidence": proposal.description}]},
                proposed_code=proposal.content,
                tool_name=proposal.target,
            )
            if "branch_name" in result:
                logger.info("Code fix branch: %s", result["branch_name"])
                return True
            if "error" not in result:
                return True
        except Exception as e:
            logger.debug("PR proposer unavailable: %s", e)
        if self._apply_tool_fn:
            return self._apply_tool_fn("modify", name=proposal.target,
                                       old_string=proposal.old_string,
                                       new_string=proposal.new_string)
        return False

    def _apply_prompt_modify(self, proposal: ImprovementProposal) -> bool:
        """Apply a prompt modification."""
        if self._apply_prompt_fn:
            return self._apply_prompt_fn(proposal.target, proposal.content)
        logger.warning("No prompt modification handler configured")
        return False

    def _apply_memory_update(self, proposal: ImprovementProposal) -> bool:
        """Write a memory update."""
        try:
            from hermes_constants import get_hermes_home
            memory_path = get_hermes_home() / "MEMORY.md"
            with open(memory_path, "a", encoding="utf-8") as f:
                f.write(f"\n\n## Evolution: {proposal.target}\n\n{proposal.content}\n")
            return True
        except Exception:
            return False

    # -- Full cycle (drives the entire loop) ----------------------------------

    def run_full_cycle(
        self,
        task: TaskDefinition,
        executor: Callable[[EvolutionRun], bool],
    ) -> EvolutionRun:
        """Run the complete evolution cycle: attempt → evaluate → improve → retry.

        This is the single-call entry point. It drives the full loop:

        1. Start tracking the task
        2. Call *executor* to run the agent (executor receives the run,
           should execute the task and populate the trajectory)
        3. Evaluate the result
        4. If failed and iterations remain: analyze, propose, gate, apply fixes
        5. Call *executor* again for retry
        6. Repeat until success or exhaustion

        Args:
            task: The task definition.
            executor: Callable that executes the agent for this task.
                Signature: (EvolutionRun) -> bool
                The executor should:
                - Read run.task for the task definition
                - Execute the agent (the run.collector tracks automatically)
                - Call run.collector.stop() and set run.trajectory when done
                - Return True if agent completed, False if execution failed

        Returns:
            The final EvolutionRun with complete history.
        """
        run = self.start_task(task)

        while run.can_continue:
            # Execute the agent
            try:
                executor_success = executor(run)
                if not executor_success:
                    run.status = TaskStatus.FAILED
                    break
            except Exception as e:
                logger.error("Task executor failed: %s", e)
                run.status = TaskStatus.FAILED
                if run.collector and run.collector.is_active:
                    run.collector.record_error(str(e))
                break

            # Stop collection and evaluate
            if run.collector and run.collector.is_active:
                run.trajectory = run.collector.stop()
            result = self.evaluate(run)

            if result.passed:
                break

            # Try to improve
            retry_initiated = self.improve_and_retry(run)
            if not retry_initiated:
                break

        # Finalize
        self.end_task(run)
        return run

    # -- Status / query API ---------------------------------------------------

    def get_active_run(self) -> Optional[EvolutionRun]:
        return self._active_run

    def get_run_status(self, run_id: str) -> Optional[Dict[str, Any]]:
        if not self._store:
            return None
        return self._store.get_run(run_id)

    def get_run_iterations(self, run_id: str) -> List[Dict[str, Any]]:
        if not self._store:
            return []
        return self._store.get_iterations(run_id)

    def list_runs(self, task_name: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
        if not self._store:
            return []
        return self._store.list_runs(task_name=task_name, limit=limit)

    def get_variant_stats(self) -> List[Dict[str, Any]]:
        if not self._variants:
            return []
        return [v.to_dict() for v in self._variants.active_variants]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _eval_to_json(result: EvalResult) -> str:
    """Serialize EvalResult to JSON string."""
    import json
    return json.dumps({
        "passed": result.passed,
        "score": result.score,
        "checks": [c.to_dict() for c in result.checks],
    })


def _list_existing_skills() -> List[str]:
    """Return names of all existing skills."""
    try:
        from agent.skill_utils import list_skill_names
        return list_skill_names()
    except Exception:
        return []


def _list_existing_tools() -> List[str]:
    """Return names of all registered tools."""
    try:
        from tools.registry import _TOOL_REGISTRY
        return list(_TOOL_REGISTRY.keys()) if hasattr(_TOOL_REGISTRY, 'keys') else []
    except Exception:
        return []


def _write_skill_file(name: str, content: str) -> bool:
    """Write a SKILL.md file directly (fallback)."""
    try:
        from hermes_constants import get_hermes_home
        skill_dir = get_hermes_home() / "skills" / name
        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text(content, encoding="utf-8")
        logger.info("Created skill: %s", skill_file)
        return True
    except Exception as e:
        logger.error("Failed to write skill '%s': %s", name, e)
        return False


def _write_tool_file(name: str, code: str) -> bool:
    """Write a tool Python file (fallback, writes to user tools dir)."""
    try:
        from hermes_constants import get_hermes_home
        tools_dir = get_hermes_home() / "tools"
        tools_dir.mkdir(parents=True, exist_ok=True)
        tool_file = tools_dir / f"{name}.py"
        tool_file.write_text(code, encoding="utf-8")
        logger.info("Created tool: %s", tool_file)
        return True
    except Exception as e:
        logger.error("Failed to write tool '%s': %s", name, e)
        return False

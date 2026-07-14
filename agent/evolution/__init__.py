"""Hermes Agent Evolution Engine (HAEE).

A built-in autonomous evaluation and self-improvement system for Hermes Agent.
Inspired by HarnessX (AEGIS engine), SIA (3-agent loop), and the Darwin Gödel
Machine (self-modifying agents).

Quick start::

    from agent.evolution import EvolutionManager, EvolutionConfig

    config = EvolutionConfig.from_config()
    manager = EvolutionManager()
    manager.initialize(session_id="...", config=config)

    # Define a task
    from agent.evolution.task_definition import TaskDefinition, SuccessCriterion

    task = TaskDefinition(
        name="fix-bug-123",
        description="Fix the login redirect bug",
        success_criteria=[
            SuccessCriterion(type="test_pass", command="pytest tests/test_login.py"),
        ],
    )

    # Run the task with evolution tracking
    run = manager.start_task(task)
    # ... agent attempts task, hooks capture trajectory ...
    result = manager.evaluate(run)

    if not result.passed:
        # Analyze failure and propose improvements
        manager.improve_and_retry(run)

Architecture:
  - EvolutionManager: Central orchestrator (MemoryManager pattern)
  - TaskDefinition: YAML-based task specification with success criteria
  - TrajectoryCollector: Captures full execution traces
  - TaskEvaluator: Multi-method evaluation (tests, files, content, LLM judge)
  - FailureAnalyzer: Root-cause analysis (AEGIS Digester)
  - ImprovementProposer: Generates concrete fixes (AEGIS Evolver)
  - RegressionGate: Seesaw constraint + deterministic safety checks
  - HarnessVariants: Variant isolation for conflicting improvements
  - EvolutionStore: SQLite persistence for evolution history
"""

from agent.evolution.config import EvolutionConfig
from agent.evolution.evolution_manager import EvolutionManager, EvolutionRun
from agent.evolution.evolution_store import EvolutionStore, get_evolution_store
from agent.evolution.task_definition import (
    ImprovementActionType,
    SuccessCriterion,
    SuccessCriterionType,
    TaskDefinition,
    TaskStatus,
    list_tasks,
    load_task,
    save_task,
    delete_task,
)
from agent.evolution.trajectory_collector import (
    EvalCheck,
    EvalResult,
    Trajectory,
    TrajectoryCollector,
    TraceStep,
    list_traces,
    load_trace,
)
from agent.evolution.evaluator import EvaluationContext, TaskEvaluator, quick_evaluate
from agent.evolution.failure_analyzer import (
    FailureAnalysis,
    FailureAnalyzer,
    FailureCategory,
    FailureFinding,
)
from agent.evolution.improvement_proposer import ImprovementProposal, ImprovementProposer
from agent.evolution.regression_gate import GateResult, GateVerdict, RegressionGate
from agent.evolution.harness_variants import HarnessVariant, VariantManager
from agent.evolution.auxiliary_llm import EvolutionLLMClient, get_evolution_llm
from agent.evolution.atropos_export import (
    ATROPOS_FORMAT_VERSION,
    export_run,
    export_all_runs,
    export_run_to_jsonl,
    get_export_stats,
)
from agent.evolution.conversation_observer import (
    ConversationObserver,
    get_observer,
)
from agent.evolution.improvement_metrics import (
    ImprovementTracker,
    ImprovementRecord,
    LearnedFailurePredictor,
    TrajectoryFeatures,
    get_tracker,
)

__all__ = [
    # Config
    "EvolutionConfig",
    # Manager
    "EvolutionManager",
    "EvolutionRun",
    # Store
    "EvolutionStore",
    "get_evolution_store",
    # Tasks
    "TaskDefinition",
    "TaskStatus",
    "SuccessCriterion",
    "SuccessCriterionType",
    "ImprovementActionType",
    "list_tasks",
    "load_task",
    "save_task",
    "delete_task",
    # Trajectory
    "Trajectory",
    "TrajectoryCollector",
    "TraceStep",
    "EvalCheck",
    "EvalResult",
    "list_traces",
    "load_trace",
    # Evaluation
    "TaskEvaluator",
    "EvaluationContext",
    "quick_evaluate",
    # Analysis
    "FailureAnalyzer",
    "FailureAnalysis",
    "FailureFinding",
    "FailureCategory",
    # Improvement
    "ImprovementProposer",
    "ImprovementProposal",
    # Safety
    "RegressionGate",
    "GateResult",
    "GateVerdict",
    # Variants
    "VariantManager",
    "HarnessVariant",
    # Auxiliary LLM
    "EvolutionLLMClient",
    "get_evolution_llm",
    # Atropos Export
    "ATROPOS_FORMAT_VERSION",
    "export_run",
    "export_all_runs",
    "export_run_to_jsonl",
    "get_export_stats",
    # Conversation Observer
    "ConversationObserver",
    "get_observer",
    # Improvement Metrics
    "ImprovementTracker",
    "ImprovementRecord",
    "LearnedFailurePredictor",
    "TrajectoryFeatures",
    "get_tracker",
]

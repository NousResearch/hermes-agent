"""SamAgent conductor package: fan-out gate, plan card, loop detector, and L0–L4 verification."""
from __future__ import annotations

from samagent.conductor.engine import (
    FanoutDecision,
    LoopDetector,
    PlanCard,
    SamAgentConductor,
    build_plan_card,
    evaluate_fanout_gate,
)
from samagent.conductor.speed import (
    SPEED_BUDGET_TARGETS,
    SpeedProfileReport,
    evaluate_speed_profile,
)
from samagent.conductor.verify import (
    LayerResult,
    VerificationReport,
    VerificationRunner,
    scan_directory_security,
)
from samagent.conductor.worktree_integrator import (
    IntegrationReport,
    WorktreeModuleRun,
    WorktreeSwarmIntegrator,
    ensure_git_repo_initialized,
)

__all__ = [
    "FanoutDecision",
    "IntegrationReport",
    "LayerResult",
    "LoopDetector",
    "PlanCard",
    "SPEED_BUDGET_TARGETS",
    "SamAgentConductor",
    "SpeedProfileReport",
    "VerificationReport",
    "VerificationRunner",
    "WorktreeModuleRun",
    "WorktreeSwarmIntegrator",
    "build_plan_card",
    "ensure_git_repo_initialized",
    "evaluate_fanout_gate",
    "evaluate_speed_profile",
    "scan_directory_security",
]

"""
Hermes Agent Core Modules

Modules:
- task_outcome: TaskOutcome recording system for self-optimization
- error_detector: Error pattern detection from repeated failures
- phase_gate: Phase-based quality gates for workflow orchestration
- skillify: Auto-generate SKILL.md files from solved problems

Inspired by ClaudeCodeFramework's SelfOptimization and WorkflowOrchestrator systems.
"""

from .task_outcome import TaskOutcome, TaskOutcomeStore
from .error_detector import ErrorPatternDetector
from .phase_gate import PhaseGate, PhaseGateResult
from .skillify import skillify_solution

__all__ = [
    "TaskOutcome",
    "TaskOutcomeStore",
    "ErrorPatternDetector",
    "PhaseGate",
    "PhaseGateResult",
    "skillify_solution",
]

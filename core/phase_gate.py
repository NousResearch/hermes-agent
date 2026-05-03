"""
PhaseGate - Quality Gate System for Workflow Orchestration

Inspired by ClaudeCodeFramework's WorkflowOrchestrator V2.
Provides phase-based quality gates to validate workflow transitions
and ensure required criteria are met before proceeding.

This module is INDEPENDENT and does not integrate with delegate_tool.py.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class GateCheckType(str, Enum):
    """Types of checks that can be performed at a gate."""
    MUST_HAVES = "must_haves"
    AUTOMATED_VERIFY = "automated_verify"
    CONTEXT_BUDGET = "context_budget"


@dataclass
class GateCheck:
    """
    Defines a single check to be performed at a phase gate.
    
    Attributes:
        type: The category of check (must_haves, automated_verify, context_budget)
        required: List of required items that must be present (for must_haves)
        tasks: Task specification or pattern to verify completion (for automated_verify)
        budget: Maximum allowed value (for context_budget, e.g., token count)
    """
    type: GateCheckType
    required: Optional[List[str]] = None
    tasks: Optional[str] = None
    budget: Optional[float] = None


@dataclass
class PhaseGateResult:
    """Result of validating a phase gate."""
    phase: str
    passed: bool
    checks: Dict[str, bool] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


class PhaseGate:
    """
    Quality gate system for validating phase transitions.
    
    A PhaseGate consists of multiple checks that must all pass before
    a workflow can proceed to the next phase. Checks are categorized
    into must-haves (required items), automated verification (task
    completion), and context budget (resource limits).
    
    Example:
        gate = PhaseGate("plan", [
            GateCheck(GateCheckType.MUST_HAVES, required=["goals", "constraints"]),
            GateCheck(GateCheckType.CONTEXT_BUDGET, budget=100000),
        ])
        result = gate.validate({"goals": ["...], "context_tokens": 50000})
    """
    
    def __init__(self, phase: str, checks: List[GateCheck]):
        """
        Initialize a PhaseGate.
        
        Args:
            phase: Name/identifier of the phase this gate guards
            checks: List of GateCheck objects defining validation criteria
        """
        self.phase = phase
        self.checks = checks
    
    def _check_must_haves(self, check: GateCheck, context: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Validate must-have items are present in context.
        
        Args:
            check: GateCheck with required list
            context: Validation context dictionary
            
        Returns:
            Tuple of (passed, error_message)
        """
        if not check.required:
            return True, None
        
        missing = []
        for item in check.required:
            if item not in context or context[item] is None:
                missing.append(item)
        
        if missing:
            return False, f"Missing must-haves: {', '.join(missing)}"
        return True, None
    
    def _check_automated_verify(self, check: GateCheck, context: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Verify task completion based on tasks specification.
        
        The tasks field can specify patterns like "all_completed", "any_completed",
        or a specific task name. The context should contain task state information.
        
        Args:
            check: GateCheck with tasks specification
            context: Validation context with task information
            
        Returns:
            Tuple of (passed, error_message)
        """
        if not check.tasks:
            return True, None
        
        tasks_spec = check.tasks.lower()
        
        # Check for task state in context
        task_state = context.get("task_state", {})
        completed_tasks = context.get("completed_tasks", [])
        
        if tasks_spec == "all_completed":
            if not completed_tasks:
                return False, "No tasks have been completed"
            # Verify all required tasks are done
            required = check.required or []
            pending = [t for t in required if t not in completed_tasks]
            if pending:
                return False, f"Pending tasks: {', '.join(pending)}"
        elif tasks_spec == "any_completed":
            if not completed_tasks:
                return False, "At least one task should be completed"
        elif tasks_spec == "verified":
            # Context-level verification flag
            if not context.get("verified", False):
                return False, "Verification check failed"
        else:
            # Treat as specific task name
            if tasks_spec not in completed_tasks:
                return False, f"Required task not completed: {tasks_spec}"
        
        return True, None
    
    def _check_context_budget(self, check: GateCheck, context: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Validate context stays within budget limits.
        
        Args:
            check: GateCheck with budget limit
            context: Validation context with usage metrics
            
        Returns:
            Tuple of (passed, error_message)
        """
        if check.budget is None:
            return True, None
        
        # Check various context size metrics
        context_tokens = context.get("context_tokens", 0)
        messages_count = context.get("message_count", 0)
        history_tokens = context.get("history_tokens", 0)
        
        # Determine which metric to check (prefer explicit token count)
        actual_usage = context_tokens or history_tokens
        
        if actual_usage > check.budget:
            return False, (
                f"Context budget exceeded: {actual_usage} > {check.budget} "
                f"(budget)"
            )
        
        # Also check message count if specified in context
        max_messages = context.get("max_messages")
        if max_messages and messages_count > max_messages:
            return False, (
                f"Message count exceeded: {messages_count} > {max_messages}"
            )
        
        return True, None
    
    def validate(self, context: Dict[str, Any]) -> PhaseGateResult:
        """
        Validate all checks against the given context.
        
        Args:
            context: Dictionary containing validation data such as:
                - present_items: List of items that exist
                - completed_tasks: List of completed task identifiers
                - context_tokens: Current token count
                - history_tokens: Historical token count
                - verified: Boolean for verification checks
                - task_state: State information for tasks
                
        Returns:
            PhaseGateResult with pass/fail status and detailed results
        """
        result = PhaseGateResult(phase=self.phase, passed=True)
        
        for check in self.checks:
            check_name = f"{check.type.value}"
            passed = False
            error = None
            
            if check.type == GateCheckType.MUST_HAVES:
                passed, error = self._check_must_haves(check, context)
            elif check.type == GateCheckType.AUTOMATED_VERIFY:
                passed, error = self._check_automated_verify(check, context)
            elif check.type == GateCheckType.CONTEXT_BUDGET:
                passed, error = self._check_context_budget(check, context)
            
            result.checks[check_name] = passed
            if error:
                result.errors.append(error)
            
            if not passed:
                result.passed = False
        
        return result


# -----------------------------------------------------------------------------
# Preset Gates
# -----------------------------------------------------------------------------

PLAN_GATE = PhaseGate(
    phase="plan",
    checks=[
        GateCheck(
            type=GateCheckType.MUST_HAVES,
            required=["goals", "scope"]
        ),
        GateCheck(
            type=GateCheckType.AUTOMATED_VERIFY,
            tasks="all_completed",
            required=["requirements_gathered", "architecture_outlined"]
        ),
        GateCheck(
            type=GateCheckType.CONTEXT_BUDGET,
            budget=80000
        ),
    ]
)

IMPLEMENT_GATE = PhaseGate(
    phase="implement",
    checks=[
        GateCheck(
            type=GateCheckType.MUST_HAVES,
            required=["code_changes", "tests"]
        ),
        GateCheck(
            type=GateCheckType.AUTOMATED_VERIFY,
            tasks="all_completed",
            required=["implementation_complete", "tests_written"]
        ),
        GateCheck(
            type=GateCheckType.CONTEXT_BUDGET,
            budget=120000
        ),
    ]
)

VERIFY_GATE = PhaseGate(
    phase="verify",
    checks=[
        GateCheck(
            type=GateCheckType.MUST_HAVES,
            required=["test_results", "verification_artifacts"]
        ),
        GateCheck(
            type=GateCheckType.AUTOMATED_VERIFY,
            tasks="verified"
        ),
        GateCheck(
            type=GateCheckType.CONTEXT_BUDGET,
            budget=60000
        ),
    ]
)


# -----------------------------------------------------------------------------
# Convenience Factory
# -----------------------------------------------------------------------------

def create_gate(phase: str, must_haves: Optional[List[str]] = None,
                verify_tasks: Optional[List[str]] = None,
                budget: Optional[float] = None) -> PhaseGate:
    """
    Factory function to create a custom PhaseGate.
    
    Args:
        phase: Name of the phase
        must_haves: List of required context keys
        verify_tasks: List of tasks that must be completed
        budget: Maximum context token budget
        
    Returns:
        Configured PhaseGate instance
    """
    checks = []
    
    if must_haves:
        checks.append(GateCheck(
            type=GateCheckType.MUST_HAVES,
            required=must_haves
        ))
    
    if verify_tasks:
        checks.append(GateCheck(
            type=GateCheckType.AUTOMATED_VERIFY,
            tasks="all_completed",
            required=verify_tasks
        ))
    
    if budget is not None:
        checks.append(GateCheck(
            type=GateCheckType.CONTEXT_BUDGET,
            budget=budget
        ))
    
    return PhaseGate(phase=phase, checks=checks)

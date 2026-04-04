"""Lightweight Workflow Engine — DAG-based task sequences on top of cron.

Provides Workflow/Step/Condition primitives for multi-step agent workflows
that go beyond single cron jobs. Steps can run agents, execute skills,
check conditions, or wait for approval.

Stored in ~/.hermes/cron/workflows.json alongside jobs.json.

Inspired by agno's workflow.py Step/Condition/Parallel primitives,
but intentionally lightweight (~200 lines vs agno's 349KB).

Usage:
    workflow = Workflow(
        name="morning-analysis",
        steps=[
            Step(name="check-portfolio", action="run_agent",
                 prompt="Check my portfolio positions"),
            Step(name="analyze-drops", action="run_agent",
                 prompt="For any position down >5%, run technical analysis",
                 condition=Condition(field="check-portfolio.output",
                                     op="contains", value="down")),
            Step(name="alert", action="run_agent",
                 prompt="Send me a summary of findings",
                 depends_on=["analyze-drops"]),
        ],
    )
"""

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

CRON_DIR = get_hermes_home() / "cron"
WORKFLOWS_FILE = CRON_DIR / "workflows.json"


# ============================================================================
# Data model
# ============================================================================

@dataclass
class Condition:
    """A condition that must be met for a step to execute."""
    field: str = ""       # "step_name.output" or "step_name.status"
    op: str = "equals"    # equals, not_equals, contains, not_contains, exists
    value: str = ""       # Expected value

    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Evaluate this condition against the workflow context."""
        parts = self.field.split(".", 1)
        if len(parts) != 2:
            return False
        step_name, attr = parts
        step_result = context.get(step_name, {})
        actual = step_result.get(attr, "")
        actual_str = str(actual).lower()
        value_lower = self.value.lower()

        if self.op == "equals":
            return actual_str == value_lower
        elif self.op == "not_equals":
            return actual_str != value_lower
        elif self.op == "contains":
            return value_lower in actual_str
        elif self.op == "not_contains":
            return value_lower not in actual_str
        elif self.op == "exists":
            return bool(actual)
        return False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Condition":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Step:
    """A single step in a workflow."""
    name: str = ""
    action: str = "run_agent"  # run_agent, run_skill, check_condition
    prompt: str = ""           # Prompt for run_agent
    skill: str = ""            # Skill name for run_skill
    condition: Optional[Condition] = None  # Skip step if condition fails
    depends_on: List[str] = field(default_factory=list)  # Step names
    timeout_seconds: int = 300
    # Runtime state
    status: str = "pending"    # pending, running, completed, failed, skipped
    output: str = ""
    error: str = ""
    duration_seconds: float = 0.0
    started_at: float = 0.0
    completed_at: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.condition:
            d["condition"] = self.condition.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Step":
        cond = d.pop("condition", None)
        step = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        if cond and isinstance(cond, dict):
            step.condition = Condition.from_dict(cond)
        return step


@dataclass
class Workflow:
    """A multi-step workflow definition."""
    id: str = ""
    name: str = ""
    description: str = ""
    schedule: str = ""  # Cron expression (e.g., "0 7 * * 1-5") or empty for manual
    enabled: bool = True
    next_run_at: float = 0.0
    steps: List[Step] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_run_at: float = 0.0
    # Execution state
    status: str = "idle"  # idle, running, completed, failed
    run_context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "schedule": self.schedule,
            "enabled": self.enabled,
            "next_run_at": self.next_run_at,
            "steps": [s.to_dict() for s in self.steps],
            "created_at": self.created_at,
            "last_run_at": self.last_run_at,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Workflow":
        steps_data = d.pop("steps", [])
        wf = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        wf.steps = [Step.from_dict(s) for s in steps_data]
        return wf


# ============================================================================
# Workflow execution
# ============================================================================

def execute_workflow(workflow: Workflow) -> Dict[str, Any]:
    """Execute a workflow sequentially, respecting conditions and dependencies.

    Returns a result dict with step outcomes.
    """
    workflow.status = "running"
    workflow.last_run_at = time.time()
    context: Dict[str, Any] = {}

    for step in workflow.steps:
        # Check dependencies
        unmet = [d for d in step.depends_on if context.get(d, {}).get("status") != "completed"]
        if unmet:
            step.status = "skipped"
            step.error = f"Unmet dependencies: {', '.join(unmet)}"
            context[step.name] = {"status": "skipped", "output": "", "error": step.error}
            continue

        # Check condition
        if step.condition and not step.condition.evaluate(context):
            step.status = "skipped"
            step.error = f"Condition not met: {step.condition.field} {step.condition.op} {step.condition.value}"
            context[step.name] = {"status": "skipped", "output": "", "error": step.error}
            continue

        # Execute
        step.status = "running"
        step.started_at = time.time()

        try:
            if step.action == "run_agent":
                output = _execute_agent_step(step, context)
            elif step.action == "run_skill":
                output = _execute_skill_step(step, context)
            elif step.action == "check_condition":
                output = "Condition check step (always passes)"
            else:
                output = f"Unknown action: {step.action}"

            step.status = "completed"
            step.output = output
        except Exception as e:
            step.status = "failed"
            step.error = str(e)
            output = ""

        step.completed_at = time.time()
        step.duration_seconds = step.completed_at - step.started_at

        context[step.name] = {
            "status": step.status,
            "output": step.output,
            "error": step.error,
        }

    # Determine overall status
    failed_steps = [s for s in workflow.steps if s.status == "failed"]
    completed_steps = [s for s in workflow.steps if s.status == "completed"]
    workflow.status = "failed" if failed_steps else "completed"

    return {
        "workflow_id": workflow.id,
        "workflow_name": workflow.name,
        "status": workflow.status,
        "steps": [s.to_dict() for s in workflow.steps],
        "completed": len(completed_steps),
        "failed": len(failed_steps),
        "skipped": len([s for s in workflow.steps if s.status == "skipped"]),
        "total_duration": sum(s.duration_seconds for s in workflow.steps),
    }


def _execute_agent_step(step: Step, context: Dict[str, Any]) -> str:
    """Execute a run_agent step by running Hermes with the prompt."""
    # Build context from prior steps
    context_parts = []
    for dep_name in step.depends_on:
        dep = context.get(dep_name, {})
        if dep.get("output"):
            context_parts.append(f"[{dep_name}]: {dep['output'][:500]}")

    full_prompt = step.prompt
    if context_parts:
        full_prompt = f"{step.prompt}\n\nPrior step results:\n" + "\n".join(context_parts)

    # Use cron's run_job infrastructure for execution
    try:
        from cron.scheduler import _run_agent_with_prompt
        output = _run_agent_with_prompt(full_prompt, timeout=step.timeout_seconds)
        return output
    except ImportError:
        # Fallback: run via subprocess
        return _run_agent_subprocess(full_prompt, step.timeout_seconds)


def _execute_skill_step(step: Step, context: Dict[str, Any]) -> str:
    """Execute a run_skill step."""
    return f"Skill execution placeholder: {step.skill}"


def _run_agent_subprocess(prompt: str, timeout: int) -> str:
    """Run hermes agent as a subprocess and capture output."""
    import subprocess
    try:
        result = subprocess.run(
            ["hermes", "chat", "--once", "-q", prompt],
            capture_output=True, text=True, timeout=timeout,
        )
        return result.stdout.strip() or result.stderr.strip()
    except subprocess.TimeoutExpired:
        return f"[timeout after {timeout}s]"
    except FileNotFoundError:
        return "[hermes CLI not found]"


# ============================================================================
# Persistence (workflows.json)
# ============================================================================

def load_workflows() -> List[Workflow]:
    """Load all workflows from disk."""
    if not WORKFLOWS_FILE.exists():
        return []
    try:
        data = json.loads(WORKFLOWS_FILE.read_text(encoding="utf-8"))
        return [Workflow.from_dict(w) for w in data]
    except (json.JSONDecodeError, Exception) as e:
        logger.warning("Failed to load workflows: %s", e)
        return []


def save_workflows(workflows: List[Workflow]) -> None:
    """Save all workflows to disk."""
    CRON_DIR.mkdir(parents=True, exist_ok=True)
    data = [w.to_dict() for w in workflows]
    WORKFLOWS_FILE.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def get_workflow(workflow_id: str) -> Optional[Workflow]:
    """Get a workflow by ID."""
    for wf in load_workflows():
        if wf.id == workflow_id:
            return wf
    return None


def save_workflow(workflow: Workflow) -> None:
    """Add or update a workflow."""
    workflows = load_workflows()
    found = False
    for i, wf in enumerate(workflows):
        if wf.id == workflow.id:
            workflows[i] = workflow
            found = True
            break
    if not found:
        workflows.append(workflow)
    save_workflows(workflows)


def delete_workflow(workflow_id: str) -> bool:
    """Delete a workflow by ID. Returns True if found and deleted."""
    workflows = load_workflows()
    initial = len(workflows)
    workflows = [w for w in workflows if w.id != workflow_id]
    if len(workflows) < initial:
        save_workflows(workflows)
        return True
    return False


# ============================================================================
# Scheduling integration (called by cron/scheduler.py tick())
# ============================================================================

def get_due_workflows() -> List[Workflow]:
    """Return workflows that are due to run based on their schedule."""
    now = time.time()
    due = []
    for wf in load_workflows():
        if not wf.enabled or not wf.schedule:
            continue
        if wf.status == "running":
            continue  # Already executing
        if wf.next_run_at <= 0:
            # First run — compute from schedule
            next_time = _compute_next_run(wf.schedule)
            if next_time and next_time <= now:
                due.append(wf)
        elif wf.next_run_at <= now:
            due.append(wf)
    return due


def advance_workflow_schedule(workflow_id: str) -> None:
    """Advance a workflow's next_run_at to the next occurrence."""
    workflows = load_workflows()
    for wf in workflows:
        if wf.id == workflow_id and wf.schedule:
            next_time = _compute_next_run(wf.schedule)
            wf.next_run_at = next_time or 0.0
            break
    save_workflows(workflows)


def tick_workflows() -> int:
    """Check and run all due workflows. Called by cron/scheduler.py tick().

    Returns number of workflows executed.
    """
    due = get_due_workflows()
    if not due:
        return 0

    executed = 0
    for wf in due:
        try:
            # Advance schedule BEFORE execution (crash-safe)
            advance_workflow_schedule(wf.id)

            result = execute_workflow(wf)

            # Update workflow state
            wf.last_run_at = time.time()
            wf.status = result.get("status", "completed")
            save_workflow(wf)

            logger.info(
                "Workflow '%s' completed: %d/%d steps",
                wf.name, result.get("completed", 0), result.get("total_tasks", len(wf.steps)),
            )
            executed += 1
        except Exception as e:
            logger.warning("Workflow '%s' failed: %s", wf.name, e)
            try:
                wf.status = "failed"
                save_workflow(wf)
            except Exception:
                pass

    return executed


def _compute_next_run(cron_expr: str) -> Optional[float]:
    """Compute the next run time from a cron expression. Returns epoch float or None."""
    try:
        from croniter import croniter
        from hermes_time import now as _hermes_now
        cron = croniter(cron_expr, _hermes_now())
        return cron.get_next(float)
    except Exception:
        return None

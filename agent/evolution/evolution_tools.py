"""Model-facing tools for the Evolution Engine.

These tools are exposed to the agent model so it can:
  - Define new evaluation tasks
  - Check evolution status
  - Request improvement for a failed task
  - View evolution history and performance trends

Tools are registered via the standard Hermes tool registry and are
gated on ``evolution.enabled`` in config.yaml.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.evolution.task_definition import (
    TaskDefinition,
    SuccessCriterion,
    SuccessCriterionType,
    list_tasks,
    load_task,
    save_task,
    delete_task,
    get_task_dir,
)
from agent.evolution.evolution_manager import EvolutionManager
from agent.evolution.evolution_store import get_evolution_store

logger = logging.getLogger(__name__)

# Tool name prefix to namespace evolution tools
TOOL_PREFIX = "evolution_"


# ---------------------------------------------------------------------------
# Tool: evolution_define_task
# ---------------------------------------------------------------------------


def evolution_define_task(
    name: str,
    description: str,
    success_criteria: List[Dict[str, Any]],
    domain: str = "general",
    complexity: int = 1,
    environment: Optional[Dict[str, Any]] = None,
    tools_allowed: Optional[List[str]] = None,
    tools_denied: Optional[List[str]] = None,
    timeout_seconds: int = 600,
    max_turns: int = 30,
) -> str:
    """Define a new evaluation task for the Evolution Engine.

    Tasks are used by the agent to benchmark its own performance and
    drive autonomous improvement. When the agent fails a task, the
    Evolution Engine analyzes the failure and proposes targeted fixes.

    Args:
        name: Unique task name (lowercase, hyphenated). Used as the filename.
        description: What the task requires the agent to do.
        success_criteria: List of criteria to evaluate success. Each criterion has:
            - type: "test_pass" | "file_exists" | "content_match" | "command_output" | "llm_judge" | "manual"
            - command: (for test_pass/command_output) Shell command to run
            - path: (for file_exists/content_match) File path to check
            - pattern: (for content_match) Regex pattern to match
            - expected_output: (for command_output) Expected substring in output
            - rubric: (for llm_judge) Scoring rubric text
            - weight: Relative weight in composite score (default 1.0)
        domain: Task domain (e.g., "software-development", "data-science")
        complexity: Task complexity on 1-14 scale
        environment: Dict with cwd, sandbox type, env vars
        tools_allowed: List of tool names the agent may use
        tools_denied: List of tool names the agent may NOT use
        timeout_seconds: Maximum task duration
        max_turns: Maximum conversation turns

    Returns:
        Confirmation message with the saved task path.
    """
    try:
        # Parse criteria
        criteria = []
        for c in success_criteria:
            try:
                ctype = SuccessCriterionType(c["type"])
            except (KeyError, ValueError):
                return f"Error: Invalid criterion type '{c.get('type')}'. Must be one of: {[t.value for t in SuccessCriterionType]}"
            criteria.append(SuccessCriterion(
                type=ctype,
                command=c.get("command"),
                path=c.get("path"),
                pattern=c.get("pattern"),
                expected_output=c.get("expected_output"),
                rubric=c.get("rubric"),
                weight=float(c.get("weight", 1.0)),
            ))

        task = TaskDefinition(
            name=name,
            description=description,
            success_criteria=criteria,
            domain=domain,
            complexity=complexity,
            environment=environment or {},
            tools_allowed=tools_allowed or [],
            tools_denied=tools_denied or [],
            timeout_seconds=timeout_seconds,
            max_turns=max_turns,
        )

        # Validate
        errors = task.validate()
        if errors:
            return f"Task validation failed:\n" + "\n".join(f"  - {e}" for e in errors)

        # Save
        path = save_task(task)
        return (
            f"Task '{name}' defined successfully.\n"
            f"  Path: {path}\n"
            f"  Domain: {domain}\n"
            f"  Complexity: {complexity}/14\n"
            f"  Criteria: {len(criteria)} ({', '.join(c.type.value for c in criteria)})\n"
            f"  Timeout: {timeout_seconds}s, Max turns: {max_turns}\n"
            f"\nRun this task with `evolution_run {name}`"
        )
    except Exception as e:
        logger.error("Failed to define task '%s': %s", name, e)
        return f"Error defining task: {e}"


# ---------------------------------------------------------------------------
# Tool: evolution_run
# ---------------------------------------------------------------------------


def evolution_run(
    task_name: str,
    evolution_manager: Optional[EvolutionManager] = None,
) -> str:
    """Run an evolution-tracked task attempt.

    The agent attempts the specified task while the Evolution Engine
    captures the full trajectory. After completion, the task is
    evaluated against its success criteria.

    If the agent fails, use evolution_improve to analyze and fix.

    Args:
        task_name: Name of the task to run (from evolution_define_task or list_tasks)

    Returns:
        Status message indicating the task should be attempted.
    """
    task = load_task(task_name)
    if task is None:
        available = [t.name for t in list_tasks()]
        return f"Task '{task_name}' not found. Available tasks: {', '.join(available) if available else '(none)'}"

    if evolution_manager and evolution_manager.is_enabled:
        try:
            run = evolution_manager.start_task(task)
            return (
                f"Evolution run started: {run.run_id}\n"
                f"  Task: {task.name}\n"
                f"  Description: {task.description}\n"
                f"  Success criteria ({len(task.success_criteria)}):\n" +
                "\n".join(
                    f"    - [{c.type.value}] {c.command or c.path or c.rubric or 'manual'}"
                    for c in task.success_criteria
                ) +
                f"\n  Timeout: {task.timeout_seconds}s | Max turns: {task.max_turns}\n"
                f"\nProceed with the task. The Evolution Engine is tracking your progress."
            )
        except Exception as e:
            logger.error("Failed to start evolution run: %s", e)
            return f"Error starting evolution run: {e}"

    # Evolution not enabled — just describe the task
    return (
        f"Task: {task.name}\n"
        f"Description: {task.description}\n"
        f"Success criteria ({len(task.success_criteria)}):\n" +
        "\n".join(
            f"  - [{c.type.value}] {c.command or c.path or c.rubric or 'manual'}"
            for c in task.success_criteria
        ) +
        f"\n\nProceed with the task. (Evolution tracking is disabled — enable it in config.yaml with evolution.enabled: true)"
    )


# ---------------------------------------------------------------------------
# Tool: evolution_list_tasks
# ---------------------------------------------------------------------------


def evolution_list_tasks(
    domain: Optional[str] = None,
) -> str:
    """List all defined evolution tasks.

    Args:
        domain: Optional domain filter (e.g., "software-development")

    Returns:
        Formatted list of tasks with their status and complexity.
    """
    tasks = list_tasks()
    if domain:
        tasks = [t for t in tasks if t.domain == domain]

    if not tasks:
        return f"No evolution tasks defined{' for domain: ' + domain if domain else ''}.\n\nDefine tasks with: evolution_define_task"

    lines = [f"Evolution Tasks ({len(tasks)} total):", ""]
    for t in tasks:
        criteria_types = ", ".join(c.type.value for c in t.success_criteria)
        lines.append(
            f"  {t.name}  [{t.domain}]  complexity={t.complexity}/14  "
            f"criteria={criteria_types}  timeout={t.timeout_seconds}s"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool: evolution_status
# ---------------------------------------------------------------------------


def evolution_status(
    run_id: Optional[str] = None,
) -> str:
    """Show evolution engine status and recent runs.

    Args:
        run_id: Optional specific run ID to inspect in detail.

    Returns:
        Status overview or detailed run information.
    """
    store = get_evolution_store()

    if run_id:
        run = store.get_run(run_id)
        if not run:
            return f"Run '{run_id}' not found."
        iterations = store.get_iterations(run_id)
        lines = [
            f"Evolution Run: {run['run_id']}",
            f"  Task: {run['task_name']} ({run.get('task_domain', 'general')})",
            f"  Status: {run['status']}",
            f"  Score: {run.get('final_score', 'N/A')}",
            f"  Iterations: {run.get('iterations', 0)}/{run.get('max_iterations', 5)}",
            f"  Created: {run.get('created_at', 'N/A')}",
            f"  Completed: {run.get('completed_at', 'N/A')}",
            f"",
            f"  Iterations:",
        ]
        for it in iterations:
            lines.append(
                f"    #{it['iteration_num']}: {it['status']} "
                f"(score={it.get('score', 'N/A')}, "
                f"action={it.get('improvement_action', 'none')})"
            )
        if run.get("summary"):
            lines.append(f"\n  Summary: {run['summary']}")
        return "\n".join(lines)

    # Overview
    recent = store.list_runs(limit=10)
    if not recent:
        return "No evolution runs recorded yet.\n\nEnable evolution in config.yaml and define tasks with: evolution_define_task"

    succeeded = sum(1 for r in recent if r["status"] == "succeeded")
    failed = sum(1 for r in recent if r["status"] == "failed")
    exhausted = sum(1 for r in recent if r["status"] == "exhausted")

    lines = [
        f"Evolution Engine Status",
        f"  Recent runs: {len(recent)} (succeeded: {succeeded}, failed: {failed}, exhausted: {exhausted})",
        f"",
        f"  Recent:",
    ]
    for r in recent[:10]:
        status_icon = {"succeeded": "✅", "failed": "❌", "exhausted": "⚠️", "pending": "⏳"}.get(r["status"], "❓")
        lines.append(
            f"    {status_icon} {r['task_name']}: {r['status']} "
            f"(score={r.get('final_score', 'N/A')}, iter={r.get('iterations', 0)})"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool schemas for registry
# ---------------------------------------------------------------------------


def get_evolution_tool_schemas() -> List[Dict[str, Any]]:
    """Return tool schemas for the model-facing evolution tools.

    These are registered with the tool registry when evolution is enabled.
    """
    return [
        {
            "name": f"{TOOL_PREFIX}define_task",
            "description": "Define a new evaluation task for the Evolution Engine. Tasks benchmark agent performance and drive autonomous improvement.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Unique task name (lowercase-hyphenated). Used as the filename.",
                    },
                    "description": {
                        "type": "string",
                        "description": "What the task requires the agent to do.",
                    },
                    "success_criteria": {
                        "type": "array",
                        "description": "List of success criteria. Each criterion has type, optional command/path/pattern/rubric, and weight.",
                        "items": {"type": "object"},
                    },
                    "domain": {
                        "type": "string",
                        "description": "Task domain (e.g., software-development, data-science).",
                    },
                    "complexity": {
                        "type": "integer",
                        "description": "Task complexity on 1-14 scale.",
                    },
                    "environment": {
                        "type": "object",
                        "description": "Dict with cwd, sandbox type, env vars.",
                    },
                    "tools_allowed": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tool names the agent may use.",
                    },
                    "tools_denied": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tool names the agent may NOT use.",
                    },
                    "timeout_seconds": {
                        "type": "integer",
                        "description": "Maximum task duration in seconds.",
                    },
                    "max_turns": {
                        "type": "integer",
                        "description": "Maximum conversation turns.",
                    },
                },
                "required": ["name", "description", "success_criteria"],
            },
        },
        {
            "name": f"{TOOL_PREFIX}list_tasks",
            "description": "List all defined evolution tasks, optionally filtered by domain.",
            "parameters": {
                "type": "object",
                "properties": {
                    "domain": {
                        "type": "string",
                        "description": "Optional domain filter.",
                    },
                },
            },
        },
        {
            "name": f"{TOOL_PREFIX}status",
            "description": "Show Evolution Engine status, recent runs, and performance trends.",
            "parameters": {
                "type": "object",
                "properties": {
                    "run_id": {
                        "type": "string",
                        "description": "Optional specific run ID to inspect in detail.",
                    },
                },
            },
        },
    ]

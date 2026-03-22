"""
Workflow Recorder — record, save, and replay agent workflows.

A workflow captures a sequence of user prompts and their expected behavior
so they can be replayed later as automated multi-step tasks.

Workflows are stored as YAML files in ~/.hermes/workflows/.

Usage (CLI / Gateway):
    /workflow record <name>       Start recording
    /workflow stop                Stop recording and save
    /workflow run <name>          Replay a saved workflow
    /workflow list                List saved workflows
    /workflow show <name>         Show workflow steps
    /workflow delete <name>       Delete a workflow
"""

import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class WorkflowStep:
    """A single step in a workflow."""

    prompt: str
    description: str = ""
    expect_tool: str = ""  # optional: expected tool call (informational)


@dataclass
class Workflow:
    """A recorded workflow with metadata."""

    name: str
    description: str = ""
    steps: List[WorkflowStep] = field(default_factory=list)
    created_at: float = 0.0
    last_run_at: float = 0.0
    run_count: int = 0
    tags: List[str] = field(default_factory=list)


def _get_workflows_dir() -> Path:
    """Return the workflows directory, creating it if needed."""
    hermes_home = Path(os.getenv("HERMES_HOME", Path.home() / ".hermes"))
    workflows_dir = hermes_home / "workflows"
    workflows_dir.mkdir(parents=True, exist_ok=True)
    return workflows_dir


def _sanitize_name(name: str) -> str:
    """Sanitize a workflow name for use as a filename."""
    clean = re.sub(r"[^\w\-]", "-", name.strip().lower())
    clean = re.sub(r"-+", "-", clean).strip("-")
    return clean[:60] or "unnamed"


def _workflow_path(name: str) -> Path:
    """Return the path for a workflow file."""
    return _get_workflows_dir() / f"{_sanitize_name(name)}.yaml"


def save_workflow(workflow: Workflow) -> Path:
    """Save a workflow to disk as YAML."""
    import yaml

    path = _workflow_path(workflow.name)
    data = {
        "name": workflow.name,
        "description": workflow.description,
        "created_at": workflow.created_at or time.time(),
        "last_run_at": workflow.last_run_at,
        "run_count": workflow.run_count,
        "tags": workflow.tags,
        "steps": [
            {
                "prompt": s.prompt,
                **({"description": s.description} if s.description else {}),
                **({"expect_tool": s.expect_tool} if s.expect_tool else {}),
            }
            for s in workflow.steps
        ],
    }
    path.write_text(
        yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    logger.info("Saved workflow '%s' (%d steps) to %s", workflow.name, len(workflow.steps), path)
    return path


def load_workflow(name: str) -> Optional[Workflow]:
    """Load a workflow from disk."""
    import yaml

    path = _workflow_path(name)
    if not path.is_file():
        return None

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as e:
        logger.warning("Failed to load workflow '%s': %s", name, e)
        return None

    steps = []
    for s in data.get("steps", []):
        if isinstance(s, dict) and s.get("prompt"):
            steps.append(WorkflowStep(
                prompt=s["prompt"],
                description=s.get("description", ""),
                expect_tool=s.get("expect_tool", ""),
            ))
        elif isinstance(s, str):
            steps.append(WorkflowStep(prompt=s))

    return Workflow(
        name=data.get("name", name),
        description=data.get("description", ""),
        steps=steps,
        created_at=data.get("created_at", 0),
        last_run_at=data.get("last_run_at", 0),
        run_count=data.get("run_count", 0),
        tags=data.get("tags", []),
    )


def list_workflows() -> List[Workflow]:
    """List all saved workflows."""
    workflows = []
    for path in sorted(_get_workflows_dir().glob("*.yaml")):
        name = path.stem
        wf = load_workflow(name)
        if wf:
            workflows.append(wf)
    return workflows


def delete_workflow(name: str) -> bool:
    """Delete a workflow file."""
    path = _workflow_path(name)
    if path.is_file():
        path.unlink()
        logger.info("Deleted workflow '%s'", name)
        return True
    return False


class WorkflowRecorder:
    """Records user prompts during a conversation into a workflow."""

    def __init__(self, name: str, description: str = ""):
        self.workflow = Workflow(
            name=name,
            description=description,
            created_at=time.time(),
        )
        self._recording = True

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def step_count(self) -> int:
        return len(self.workflow.steps)

    def record_step(self, user_prompt: str, tool_names: Optional[List[str]] = None):
        """Record a user prompt as a workflow step."""
        if not self._recording:
            return
        # Skip slash commands
        if user_prompt.strip().startswith("/"):
            return

        step = WorkflowStep(
            prompt=user_prompt.strip(),
            expect_tool=", ".join(tool_names) if tool_names else "",
        )
        self.workflow.steps.append(step)
        logger.debug("Recorded workflow step %d: %s", len(self.workflow.steps), user_prompt[:50])

    def stop(self) -> Workflow:
        """Stop recording and return the workflow."""
        self._recording = False
        return self.workflow

    def save(self) -> Path:
        """Stop recording and save to disk."""
        self._recording = False
        return save_workflow(self.workflow)


class WorkflowRunner:
    """Replays a workflow by feeding prompts to the agent sequentially."""

    def __init__(self, workflow: Workflow):
        self.workflow = workflow
        self.current_step = 0
        self._running = False
        self._cancelled = False

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def progress(self) -> str:
        return f"{self.current_step}/{len(self.workflow.steps)}"

    def cancel(self):
        """Cancel the running workflow."""
        self._cancelled = True

    def run(
        self,
        send_prompt: Callable[[str], str],
        on_step_start: Optional[Callable[[int, WorkflowStep], None]] = None,
        on_step_done: Optional[Callable[[int, WorkflowStep, str], None]] = None,
    ) -> List[Dict[str, Any]]:
        """Run the workflow by sending each prompt to the agent.

        Args:
            send_prompt: Function that sends a prompt and returns the response.
            on_step_start: Called before each step with (index, step).
            on_step_done: Called after each step with (index, step, response).

        Returns:
            List of step results with prompt, response, duration, and status.
        """
        self._running = True
        self._cancelled = False
        results = []

        for i, step in enumerate(self.workflow.steps):
            if self._cancelled:
                results.append({
                    "step": i + 1,
                    "prompt": step.prompt,
                    "response": "",
                    "status": "cancelled",
                    "duration": 0,
                })
                break

            self.current_step = i + 1

            if on_step_start:
                on_step_start(i, step)

            t0 = time.time()
            try:
                response = send_prompt(step.prompt)
                duration = time.time() - t0
                results.append({
                    "step": i + 1,
                    "prompt": step.prompt,
                    "response": response,
                    "status": "ok",
                    "duration": duration,
                })
            except KeyboardInterrupt:
                results.append({
                    "step": i + 1,
                    "prompt": step.prompt,
                    "response": "",
                    "status": "interrupted",
                    "duration": time.time() - t0,
                })
                break
            except Exception as e:
                duration = time.time() - t0
                results.append({
                    "step": i + 1,
                    "prompt": step.prompt,
                    "response": str(e),
                    "status": "error",
                    "duration": duration,
                })
                logger.warning("Workflow step %d failed: %s", i + 1, e)

            if on_step_done:
                on_step_done(i, step, results[-1].get("response", ""))

        self._running = False

        # Update run stats
        self.workflow.last_run_at = time.time()
        self.workflow.run_count += 1
        save_workflow(self.workflow)

        return results


def format_workflow_list(workflows: List[Workflow]) -> str:
    """Format a list of workflows for display."""
    if not workflows:
        return "No saved workflows. Use /workflow record <name> to create one."

    lines = [f"Saved workflows ({len(workflows)}):"]
    for wf in workflows:
        steps = len(wf.steps)
        runs = wf.run_count
        desc = f" - {wf.description}" if wf.description else ""
        lines.append(f"  {wf.name} ({steps} steps, {runs} runs){desc}")
    return "\n".join(lines)


def format_workflow_detail(workflow: Workflow) -> str:
    """Format a workflow's steps for display."""
    lines = [
        f"Workflow: {workflow.name}",
    ]
    if workflow.description:
        lines.append(f"Description: {workflow.description}")
    lines.append(f"Steps: {len(workflow.steps)}, Runs: {workflow.run_count}")
    lines.append("")
    for i, step in enumerate(workflow.steps, 1):
        tool_hint = f"  [{step.expect_tool}]" if step.expect_tool else ""
        prompt_preview = step.prompt[:100] + ("..." if len(step.prompt) > 100 else "")
        lines.append(f"  {i}. {prompt_preview}{tool_hint}")
    return "\n".join(lines)

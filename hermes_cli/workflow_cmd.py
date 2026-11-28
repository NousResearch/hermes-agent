"""``hermes workflow`` handlers: validate a manifest, inspect its checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from hermes_cli.workflow_manifest import Workflow, WorkflowError, WorkflowState, next_steps, resume_state_path


def cmd_workflow(args: argparse.Namespace) -> None:
    action = getattr(args, "workflow_action", None)
    manifest_path = Path(getattr(args, "manifest", ""))

    try:
        workflow = Workflow.load(manifest_path)
    except (OSError, ValueError, WorkflowError) as exc:
        raise SystemExit(f"invalid workflow manifest: {exc}")

    if action == "validate":
        print(f"Valid workflow '{workflow.name}': {len(workflow.steps)} steps, "
              f"max_iterations={workflow.max_iterations}, max_children={workflow.max_children}")
        return

    if action == "status":
        state = WorkflowState.load(resume_state_path(manifest_path))
        completed = [s.id for s in workflow.steps if s.id in state.completed]
        ready = [s.id for s in next_steps(workflow, state)]
        print(f"Workflow '{workflow.name}'")
        print(f"  completed: {', '.join(completed) or '(none)'}")
        print(f"  ready:     {', '.join(ready) or '(none)'}")
        return

    raise SystemExit(f"unknown workflow action: {action}")

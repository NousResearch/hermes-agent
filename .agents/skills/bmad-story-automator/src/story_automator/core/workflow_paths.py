from __future__ import annotations

from dataclasses import dataclass

from story_automator.core.runtime_policy import load_effective_policy, step_contract


@dataclass(frozen=True)
class WorkflowPaths:
    skill: str = ""
    workflow: str = ""
    instructions: str = ""
    checklist: str = ""
    template: str = ""


def _paths_for_step(step: str, project_root: str | None = None) -> WorkflowPaths:
    files = (step_contract(load_effective_policy(project_root), step).get("assets") or {}).get("files") or {}
    return WorkflowPaths(
        skill=str(files.get("skill") or ""),
        workflow=str(files.get("workflow") or ""),
        instructions=str(files.get("instructions") or ""),
        checklist=str(files.get("checklist") or ""),
        template=str(files.get("template") or ""),
    )


def create_story_workflow_paths(project_root: str | None = None) -> WorkflowPaths:
    return _paths_for_step("create", project_root)


def dev_story_workflow_paths(project_root: str | None = None) -> WorkflowPaths:
    return _paths_for_step("dev", project_root)


def retrospective_workflow_paths(project_root: str | None = None) -> WorkflowPaths:
    return _paths_for_step("retro", project_root)


def review_workflow_paths(project_root: str | None = None) -> WorkflowPaths:
    return _paths_for_step("review", project_root)


def testarch_automate_workflow_paths(project_root: str | None = None) -> WorkflowPaths:
    return _paths_for_step("auto", project_root)

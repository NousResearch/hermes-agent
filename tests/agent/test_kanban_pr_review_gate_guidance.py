"""Regression coverage for the Kanban → PR → reviewer → user merge gate."""

from pathlib import Path

from agent.prompt_builder import KANBAN_GUIDANCE


ROOT = Path(__file__).resolve().parents[2]


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_kanban_guidance_requires_pr_review_gate_without_merge():
    guidance = KANBAN_GUIDANCE

    assert "branch, commit" in guidance
    assert "open a PR" in guidance
    assert "separate reviewer Kanban task" in guidance
    assert "do not merge or enable auto-merge" in guidance
    assert "final merge decision belongs to the user" in guidance
    assert "complete the implementation task with structured PR handoff" in guidance
    assert "Do NOT block the" in guidance
    assert "blocked parent deadlocks" in guidance


def test_bundled_skills_document_pr_review_gate():
    worker = _read("skills/devops/kanban-worker/SKILL.md")
    orchestrator = _read("skills/devops/kanban-orchestrator/SKILL.md")
    pr_workflow = _read("skills/github/github-pr-workflow/SKILL.md")

    for text in (worker, orchestrator, pr_workflow):
        assert "auto-merge" in text
        assert "reviewer" in text
        assert "merge decision" in text

    assert "blocked parent keeps the reviewer task in `todo`" in worker
    assert "completes the implementation task" in orchestrator
    assert "structured PR handoff metadata" in orchestrator
    assert "main/default profile should route" in orchestrator
    assert "explicit\ninstruction" in pr_workflow


def test_generated_bundled_skill_docs_document_pr_review_gate():
    worker_doc = _read(
        "website/docs/user-guide/skills/bundled/devops/devops-kanban-worker.md"
    )
    orchestrator_doc = _read(
        "website/docs/user-guide/skills/bundled/devops/devops-kanban-orchestrator.md"
    )
    pr_workflow_doc = _read(
        "website/docs/user-guide/skills/bundled/github/github-github-pr-workflow.md"
    )

    for text in (worker_doc, orchestrator_doc, pr_workflow_doc):
        assert "auto-merge" in text
        assert "reviewer" in text
        assert "merge decision" in text

    assert "## Repository code-change workflow" in worker_doc
    assert "blocked parent keeps the reviewer task in `todo`" in worker_doc
    assert "## User-requested repository changes" in orchestrator_doc
    assert "main/default profile should route" in orchestrator_doc
    assert "structured PR handoff metadata" in orchestrator_doc
    assert "### Kanban-gated agent work" in pr_workflow_doc
    assert "deadlocks review in `todo`" in pr_workflow_doc


def test_repo_docs_and_pr_template_carry_gate():
    agents = _read("AGENTS.md")
    contributing = _read("CONTRIBUTING.md")
    template = _read(".github/PULL_REQUEST_TEMPLATE.md")

    assert "Kanban/PR/review gate" in agents
    assert "user decides whether and when to merge" in contributing
    assert "Reviewer Kanban task created after this PR exists" in template
    assert "do not block it with review-required" in template

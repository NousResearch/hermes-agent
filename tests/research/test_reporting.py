"""Tests for research.reporting."""

from research.models import ResearchProject
from research.reporting import render_inbox_markdown, render_iteration_report


def test_render_iteration_report_basic():
    project = ResearchProject(
        project_id="proj_test",
        name="My Project",
        cwd="/tmp",
        workspace_dir="/tmp/ws",
        phase="training",
        spec_level="partial_spec",
    )
    status = {"ideas": 2, "literature_notes": 5, "hypotheses": 1, "experiments": 1, "active_run_ids": ["run_a"]}
    report = render_iteration_report(project, status, "All going well", next_steps=["Evaluate"])
    assert "My Project" in report
    assert "training" in report
    assert "All going well" in report
    assert "Evaluate" in report
    assert "run_a" in report


def test_render_iteration_report_with_runs():
    project = ResearchProject(
        project_id="proj_test",
        name="Test",
        cwd="/tmp",
        workspace_dir="/tmp/ws",
    )
    status = {"ideas": 0, "literature_notes": 0, "hypotheses": 0, "experiments": 0, "active_run_ids": []}
    runs = [{"run_id": "run_1", "status": "completed", "method": "sft", "base_model": "llama-8b"}]
    report = render_iteration_report(project, status, "Done", run_rows=runs)
    assert "run_1" in report
    assert "completed" in report
    assert "sft" in report


def test_render_inbox_markdown():
    md = render_inbox_markdown("Run Complete", "All metrics passed", details="f1=0.95", next_steps=["Deploy"])
    assert "# Run Complete" in md
    assert "All metrics passed" in md
    assert "f1=0.95" in md
    assert "Deploy" in md


def test_render_inbox_markdown_minimal():
    md = render_inbox_markdown("Title", "Summary only")
    assert "# Title" in md
    assert "Summary only" in md
    assert "Details" not in md
    assert "Next Steps" not in md

"""Tests for research.loop_policy."""

from research.loop_policy import (
    bootstrap_experiment_template,
    bootstrap_hypothesis_template,
    bootstrap_idea_candidates,
    compose_continuation_prompt,
    compose_monitor_prompt,
    detect_spec_level,
)
from research.models import ResearchProject


def test_detect_spec_level_zero_short():
    assert detect_spec_level("") == "zero_spec"
    assert detect_spec_level("do something cool") == "zero_spec"


def test_detect_spec_level_zero_no_markers():
    assert detect_spec_level("I want to explore new research directions in a creative way for my project") == "zero_spec"


def test_detect_spec_level_partial():
    assert detect_spec_level("I want to fine-tune a model using sft on my dataset") == "partial_spec"
    assert detect_spec_level("Build a dpo alignment pipeline for safety improvements") == "partial_spec"


def test_detect_spec_level_full():
    brief = (
        "Fine-tune the base model llama with lora on my dataset with learning rate 1e-4 "
        "and batch size 32 for 3 epochs then evaluate on the benchmark"
    )
    assert detect_spec_level(brief) == "full_spec"


def test_bootstrap_idea_candidates_count():
    ideas = bootstrap_idea_candidates("proj_test", "explore new methods")
    assert len(ideas) == 3
    # All candidates start as unchosen; the caller picks one
    assert all(not i.chosen for i in ideas)


def test_bootstrap_idea_candidates_ids_unique():
    ideas = bootstrap_idea_candidates("proj_test", "explore")
    ids = {i.idea_id for i in ideas}
    assert len(ids) == 3


def test_bootstrap_hypothesis_template():
    h = bootstrap_hypothesis_template("proj_test", chosen_idea_id="idea_abc")
    assert h.linked_idea_id == "idea_abc"
    assert h.status == "draft"
    assert h.statement != ""


def test_bootstrap_experiment_template():
    exp = bootstrap_experiment_template("proj_test", "hyp_abc", method="dpo")
    assert exp.hypothesis_id == "hyp_abc"
    assert exp.method == "dpo"
    assert len(exp.evaluation_plan) > 0


def test_compose_continuation_prompt():
    project = ResearchProject(
        project_id="proj_test",
        name="My Research",
        cwd="/tmp",
        workspace_dir="/tmp/ws",
    )
    status_summary = {"ideas": 3, "runs": []}
    prompt = compose_continuation_prompt(project, status_summary, "session ended", "evaluate_results")
    assert "proj_test" in prompt
    assert "My Research" in prompt
    assert "evaluate_results" in prompt
    assert "session ended" in prompt


def test_compose_monitor_prompt():
    project = ResearchProject(
        project_id="proj_test",
        name="My Research",
        cwd="/tmp",
        workspace_dir="/tmp/ws",
    )
    prompt = compose_monitor_prompt(project, "run_abc")
    assert "run_abc" in prompt
    assert "proj_test" in prompt
    assert "monitor" in prompt.lower()

"""Tests for research.models."""

from research.models import (
    ExperimentSpec,
    Hypothesis,
    InboxItem,
    LoopCheckpoint,
    ResearchIdea,
    ResearchProject,
    RunSpec,
    make_id,
    now_utc,
    slugify,
)


def test_slugify_basic():
    assert slugify("My Cool Project") == "my-cool-project"


def test_slugify_special_chars():
    assert slugify("Hello!@#World") == "hello-world"


def test_slugify_empty_falls_back():
    assert slugify("", default="fallback") == "fallback"


def test_slugify_only_special_chars():
    assert slugify("!@#$%^") == "project"


def test_make_id_prefix():
    result = make_id("proj")
    assert result.startswith("proj_")
    assert len(result) == len("proj_") + 10


def test_make_id_unique():
    ids = {make_id("test") for _ in range(100)}
    assert len(ids) == 100


def test_now_utc_format():
    ts = now_utc()
    assert "T" in ts
    assert "+00:00" in ts
    # No microseconds
    assert "." not in ts


def test_research_project_defaults():
    p = ResearchProject(name="Test", cwd="/tmp", workspace_dir="/tmp/ws")
    assert p.project_id.startswith("proj_")
    assert p.phase == "bootstrap"
    assert p.status == "active"
    assert p.spec_level == "partial_spec"
    assert p.active_run_ids == []


def test_research_project_extra_fields_allowed():
    p = ResearchProject(name="Test", cwd="/tmp", workspace_dir="/tmp/ws", custom_field="hello")
    assert p.custom_field == "hello"


def test_research_idea_defaults():
    idea = ResearchIdea(title="Idea 1", summary="An idea")
    assert idea.idea_id.startswith("idea_")
    assert idea.chosen is False
    assert idea.source == "agent"


def test_hypothesis_defaults():
    h = Hypothesis(statement="X > Y", success_metric="accuracy", stop_condition="divergence")
    assert h.hypothesis_id.startswith("hyp_")
    assert h.status == "draft"


def test_run_spec_defaults():
    rs = RunSpec(project_id="proj_test", base_model="llama-8b", dataset_path="/data.jsonl")
    assert rs.run_id.startswith("run_")
    assert rs.method == "sft"
    assert rs.batch_size == 32
    assert rs.lora_rank == 32
    assert rs.status == "pending"


def test_run_spec_serialization():
    rs = RunSpec(project_id="proj_test", base_model="llama-8b", dataset_path="/data.jsonl")
    d = rs.model_dump(mode="json")
    assert d["project_id"] == "proj_test"
    restored = RunSpec.model_validate(d)
    assert restored.run_id == rs.run_id


def test_experiment_spec_defaults():
    e = ExperimentSpec(
        title="Exp 1",
        hypothesis_id="hyp_abc",
        objective="test something",
        success_metric="f1 > 0.9",
        stop_condition="budget exceeded",
    )
    assert e.experiment_id.startswith("exp_")
    assert e.method == "sft"
    assert e.status == "draft"


def test_loop_checkpoint_defaults():
    c = LoopCheckpoint(project_id="proj_test", reason="session end", summary="paused")
    assert c.checkpoint_id.startswith("ckpt_")
    assert c.next_action == ""


def test_inbox_item_defaults():
    item = InboxItem(project_id="proj_test", title="Update", summary="Run finished")
    assert item.item_id.startswith("inbox_")
    assert item.status == "unread"

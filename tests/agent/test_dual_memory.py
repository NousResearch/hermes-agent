from pathlib import Path

import pytest

from agent.dual_memory import (
    PARA_BUCKETS,
    PersonalWorkspace,
    ProceduralMemory,
    SkillDraft,
    WorkspaceItem,
    default_procedural_skills_root,
    default_workspace_root,
    filter_workspace_candidate,
    route_item,
)


@pytest.fixture()
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def test_default_paths_are_profile_scoped(hermes_home):
    assert default_workspace_root() == hermes_home / "personal_workspace"
    assert default_procedural_skills_root() == hermes_home / "skills" / "procedural-memory"


def test_workspace_init_creates_para_manifests(hermes_home):
    workspace = PersonalWorkspace()
    workspace.initialize()

    for bucket in PARA_BUCKETS:
        manifest = hermes_home / "personal_workspace" / bucket / "_manifest.md"
        assert manifest.exists()
        assert f"# {bucket} Manifest" in manifest.read_text(encoding="utf-8")


def test_route_item_uses_para_state_machine_signals():
    assert route_item("Ship mobile app", "deadline next Friday") == "Projects"
    assert route_item("Writing habit", "ongoing responsibility") == "Areas"
    assert route_item("OAuth reference notes", "Reusable reference material") == "Resources"
    assert route_item("Old launch plan", "completed and archived") == "Archives"


def test_workspace_write_refreshes_manifest_and_retrieves_top_file(hermes_home):
    workspace = PersonalWorkspace()
    path = workspace.write_item(
        WorkspaceItem(
            title="Literature Review Plan",
            content="Paper notes about retrieval augmented generation and memory routing.",
            bucket="Projects",
            summary="RAG memory routing research plan",
            tags=["research", "rag"],
        )
    )

    assert path == hermes_home / "personal_workspace" / "Projects" / "literature-review-plan.md"
    manifest = path.parent / "_manifest.md"
    manifest_text = manifest.read_text(encoding="utf-8")
    assert "Literature Review Plan" in manifest_text
    assert "tags: research, rag" in manifest_text

    results = workspace.retrieve("rag routing", top_k=1)
    assert len(results) == 1
    assert results[0].record.title == "Literature Review Plan"
    assert "retrieval augmented generation" in results[0].content


def test_workspace_append_preserves_existing_file(hermes_home):
    workspace = PersonalWorkspace()
    first = workspace.write_item(
        WorkspaceItem(
            title="Course Notes",
            content="Initial notes about the course project with a concrete deadline.",
            bucket="Projects",
        )
    )
    second = workspace.write_item(
        WorkspaceItem(
            title="Course Notes",
            content="Added notes about the grading rubric and final deliverable.",
            bucket="Projects",
        ),
        mode="append",
    )

    assert second == first
    text = first.read_text(encoding="utf-8")
    assert "Initial notes" in text
    assert "Added notes" in text
    assert "## Update" in text


def test_filter_workspace_candidate_rejects_transient_content():
    assert not filter_workspace_candidate("ok thanks")
    assert not filter_workspace_candidate("short")
    assert filter_workspace_candidate(
        "This project decision is durable: use the PARA workspace for user-visible research notes."
    )


def test_procedural_memory_writes_skill_markdown(hermes_home):
    procedural = ProceduralMemory()
    path = procedural.write_skill(
        SkillDraft(
            name="Dataset Triage",
            description="Triage user Obsidian datasets for dual-memory experiments.",
            triggers=["A task asks to evaluate a personal knowledge-base dataset."],
            steps=[
                "Inventory projects, courses, and literature notes.",
                "Map each note to Projects, Areas, Resources, or Archives.",
                "Record failure cases for later workflow refinement.",
            ],
            constraints=["Do not overwrite user-authored notes."],
            recovery=["If classification is ambiguous, keep the note in Resources and add a backlink."],
            source="Distilled from a successful dataset preparation trajectory.",
        )
    )

    assert path == hermes_home / "skills" / "procedural-memory" / "dataset-triage" / "SKILL.md"
    text = path.read_text(encoding="utf-8")
    assert "name: dataset-triage" in text
    assert "## When To Use" in text
    assert "## Procedure" in text
    assert "Do not overwrite user-authored notes." in text


def test_procedural_memory_refuses_accidental_overwrite(hermes_home):
    procedural = ProceduralMemory()
    draft = SkillDraft(
        name="Repeatable Flow",
        description="A reusable flow.",
        triggers=["When the same flow succeeds repeatedly."],
        steps=["Run the known-good steps."],
    )
    procedural.write_skill(draft)
    with pytest.raises(FileExistsError):
        procedural.write_skill(draft)

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_MD = REPO_ROOT / "skills" / "research" / "cyber-vc-analyst" / "SKILL.md"
WEBSITE_DOC = (
    REPO_ROOT
    / "website"
    / "docs"
    / "user-guide"
    / "skills"
    / "bundled"
    / "research"
    / "research-cyber-vc-analyst.md"
)
SLACK_WORKFLOW = (
    REPO_ROOT
    / "skills"
    / "research"
    / "cyber-vc-analyst"
    / "references"
    / "slack-workflow.md"
)
PRIVATE_BLUEPRINT = (
    REPO_ROOT
    / "skills"
    / "research"
    / "cyber-vc-analyst"
    / "references"
    / "private-repo-blueprint.md"
)
RESEARCH_DEPTH = (
    REPO_ROOT
    / "skills"
    / "research"
    / "cyber-vc-analyst"
    / "references"
    / "research-depth.md"
)
RESEARCH_STATE = (
    REPO_ROOT
    / "skills"
    / "research"
    / "cyber-vc-analyst"
    / "references"
    / "research-state.md"
)
WORKFLOW_PHASES = (
    REPO_ROOT
    / "skills"
    / "research"
    / "cyber-vc-analyst"
    / "references"
    / "workflow-phases.md"
)


def test_skill_and_docs_keep_theme_output_root_aligned():
    expected = "default: 3.Areas/cyber futures frontier"
    skill_text = SKILL_MD.read_text(encoding="utf-8")
    docs_text = WEBSITE_DOC.read_text(encoding="utf-8")

    assert expected in skill_text
    assert expected in docs_text


def test_skill_references_slack_workflow_and_private_repo_blueprint():
    skill_text = SKILL_MD.read_text(encoding="utf-8")

    assert "references/slack-workflow.md" in skill_text
    assert "references/private-repo-blueprint.md" in skill_text
    assert "references/research-depth.md" in skill_text
    assert "references/research-state.md" in skill_text
    assert "references/workflow-phases.md" in skill_text


def test_slack_workflow_documents_thread_safe_invocation():
    slack_text = SLACK_WORKFLOW.read_text(encoding="utf-8")

    assert "!cyber-vc-analyst" in slack_text
    assert "/hermes cyber-vc-analyst" in slack_text
    assert "Slack blocks them there" in slack_text


def test_private_repo_blueprint_includes_docs_schemas_fixtures_and_release_notes():
    blueprint_text = PRIVATE_BLUEPRINT.read_text(encoding="utf-8")

    for token in ("docs/", "schemas/", "fixtures/", "release-notes/"):
        assert token in blueprint_text


def test_slack_workflow_documents_compare_and_triage_shapes():
    slack_text = SLACK_WORKFLOW.read_text(encoding="utf-8")

    assert "!cyber-vc-analyst compare Red Access Security vs Noma Security" in slack_text
    assert "!cyber-vc-analyst triage <company>" in slack_text
    assert "### Compare mode" in slack_text
    assert "### Triage mode" in slack_text


def test_private_repo_blueprint_includes_compare_and_triage_assets():
    blueprint_text = PRIVATE_BLUEPRINT.read_text(encoding="utf-8")

    for token in (
        "compare-analysis.schema.yaml",
        "triage-analysis.schema.yaml",
        "compare-red-access-vs-noma.input.yaml",
        "triage-red-access.input.yaml",
    ):
        assert token in blueprint_text


def test_shared_references_document_depth_state_and_phases():
    depth_text = RESEARCH_DEPTH.read_text(encoding="utf-8")
    state_text = RESEARCH_STATE.read_text(encoding="utf-8")
    phases_text = WORKFLOW_PHASES.read_text(encoding="utf-8")

    for token in ("quick", "standard", "deep"):
        assert token in depth_text
    for token in ("research-state", "completed phases", "next recommended step"):
        assert token in state_text
    for token in ("company", "theme", "compare", "triage", "competitors"):
        assert token in phases_text

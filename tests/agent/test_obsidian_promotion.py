from dataclasses import replace

import pytest

from agent.obsidian_promotion import (
    ObsidianPromotionAction,
    ObsidianPromotionCandidate,
    plan_obsidian_promotion,
    write_obsidian_promotion,
)


def test_forge_dev_synthesis_defaults_to_dev_dry_run():
    candidate = ObsidianPromotionCandidate(
        title="Hermes Memory Governance Gate",
        content=(
            "Reusable Hermes development synthesis: memory candidates are reviewed "
            "before durable storage. Keep Obsidian curated and Honcho runtime-only."
        ),
        source_type="dev_synthesis",
        source_path="docs/memory-governance-gate.md",
        profile="forge",
        project="hermes-agent",
        tags=("hermes", "memory-governance"),
        created_at="2026-05-12T00:00:00Z",
    )

    plan = plan_obsidian_promotion(candidate)

    assert plan.action == ObsidianPromotionAction.DEV_SYNTHESIS
    assert plan.target_relative_path == "Dev/hermes-memory-governance-gate.md"
    assert plan.requires_approval is True
    assert plan.approval_flag == "approved=True"
    assert plan.duplicate_candidates == []
    assert "dry-run" in plan.rationale
    assert "source_path: \"docs/memory-governance-gate.md\"" in plan.frontmatter_preview
    assert "OPENAI_API_KEY" not in plan.markdown_preview


def test_raw_evidence_plan_detects_duplicate_path_and_title(tmp_path):
    vault = tmp_path / "vault"
    raw_dir = vault / "Raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "hermes-source-evidence.md").write_text(
        "---\n"
        "title: \"Hermes Source Evidence\"\n"
        "type: raw\n"
        "---\n"
        "# Hermes Source Evidence\n",
        encoding="utf-8",
    )
    (raw_dir / "older-note.md").write_text(
        "---\n"
        "title: \"Hermes Source Evidence\"\n"
        "type: raw\n"
        "---\n",
        encoding="utf-8",
    )

    plan = plan_obsidian_promotion(
        {
            "title": "Hermes Source Evidence",
            "content": "Source evidence: run logs show the targeted tests passed.",
            "source_type": "source_evidence",
            "source_url": "https://example.com/evidence",
            "profile": "forge",
            "tags": ["evidence"],
            "created_at": "2026-05-12",
        },
        vault_path=vault,
    )

    assert plan.action == ObsidianPromotionAction.RAW_EVIDENCE
    assert plan.target_relative_path == "Raw/hermes-source-evidence.md"
    assert plan.duplicate_candidates == [
        "Raw/hermes-source-evidence.md",
        "Raw/older-note.md",
    ]
    assert "source_url: \"https://example.com/evidence\"" in plan.frontmatter_preview


@pytest.mark.parametrize(
    "title, content, source_type, source_path",
    [
        (
            "Full conversation transcript",
            "Complete transcript of every user and assistant message.",
            "full_transcript",
            None,
        ),
        ("debug-log", "Temporary debug log from a failing run.", "debug_log", None),
        ("plan", "Active plan for current implementation work.", "plan", "plan.md"),
        ("workspace dump", "Raw workspace dump with intermediate tool outputs.", "dump", None),
    ],
)
def test_forbidden_artifacts_are_rejected(title, content, source_type, source_path):
    plan = plan_obsidian_promotion(
        ObsidianPromotionCandidate(
            title=title,
            content=content,
            source_type=source_type,
            source_path=source_path,
        )
    )

    assert plan.action == ObsidianPromotionAction.REJECT
    assert plan.target_relative_path is None
    assert plan.requires_approval is False
    assert "not eligible" in plan.rationale


def test_vault_profile_is_not_treated_as_obsidian_governance_owner():
    plan = plan_obsidian_promotion(
        ObsidianPromotionCandidate(
            title="Stablecoin Counterparty Risk Summary",
            content=(
                "Project final summary: strategy and risk decisions for stablecoin "
                "counterparty monitoring."
            ),
            source_type="project_summary",
            profile="vault",
            project="risk-monitoring",
        )
    )

    assert plan.action == ObsidianPromotionAction.PROJECT_SUMMARY
    assert plan.target_relative_path == "Projects/stablecoin-counterparty-risk-summary.md"
    assert not plan.target_relative_path.startswith("90. setting/")
    assert "vault profile is investment/risk/strategy" in plan.rationale


def test_secret_values_are_redacted_from_previews():
    plan = plan_obsidian_promotion(
        ObsidianPromotionCandidate(
            title="Credential incident",
            content="OPENAI_API_KEY=sk-testsecretvalue1234567890 should never be copied.",
            source_type="source_evidence",
            source_path="/tmp/leaked.env",
            created_at="2026-05-12",
        )
    )

    assert plan.action == ObsidianPromotionAction.REJECT
    assert "sk-testsecretvalue" not in plan.frontmatter_preview
    assert "sk-testsecretvalue" not in plan.markdown_preview
    assert "[REDACTED]" in plan.markdown_preview


def test_write_requires_explicit_approval_and_stays_inside_vault(tmp_path):
    vault = tmp_path / "vault"
    plan = plan_obsidian_promotion(
        ObsidianPromotionCandidate(
            title="Hermes Promotion Pipeline",
            content="Reusable development synthesis for safe dry-run Obsidian promotion.",
            source_type="dev_synthesis",
            profile="forge",
        ),
        vault_path=vault,
    )

    with pytest.raises(PermissionError):
        write_obsidian_promotion(plan, vault_path=vault)

    written = write_obsidian_promotion(plan, vault_path=vault, approved=True)
    assert written == vault / "Dev" / "hermes-promotion-pipeline.md"
    assert written.read_text(encoding="utf-8") == plan.markdown_preview

    with pytest.raises(FileExistsError):
        write_obsidian_promotion(plan, vault_path=vault, approved=True)

    traversal_plan = replace(plan, target_relative_path="../outside.md")
    with pytest.raises(ValueError):
        write_obsidian_promotion(traversal_plan, vault_path=vault, approved=True)

    absolute_plan = replace(plan, target_relative_path=str(tmp_path / "outside.md"))
    with pytest.raises(ValueError):
        write_obsidian_promotion(absolute_plan, vault_path=vault, approved=True)

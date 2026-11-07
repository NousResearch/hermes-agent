"""Contract tests for the White Knight upstream-audit workflow."""

from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[2]
SKILL = REPO / "optional-skills/software-development/hermes-upstream-audit/SKILL.md"


def _skill() -> str:
    return SKILL.read_text(encoding="utf-8")


def _frontmatter() -> dict:
    text = _skill()
    assert text.startswith("---\n")
    end = text.index("\n---\n", 4)
    parsed = yaml.safe_load(text[4:end])
    assert isinstance(parsed, dict)
    return parsed


def test_skill_has_valid_frontmatter_and_optional_scope():
    frontmatter = _frontmatter()

    assert frontmatter["name"] == "hermes-upstream-audit"
    assert len(frontmatter["description"]) <= 60
    assert frontmatter["description"].endswith(".")
    assert frontmatter["platforms"] == ["linux", "macos", "windows"]
    assert "github" in frontmatter["metadata"]["hermes"]["related_skills"]


def test_workflow_freezes_and_reuses_both_canonical_snapshots():
    text = _skill()

    assert "complete open-issue snapshot" in text
    assert "complete open-PR snapshot" in text
    assert "Reuse these exact snapshots for the whole audit" in text
    assert "base SHA" in text
    assert "pagination" in text


def test_workflow_requires_local_causal_correlation_and_explicit_unknowns():
    text = _skill()

    for phrase in (
        "local reproduction or direct applicability",
        "exact source owner",
        "root-cause classification",
        "unresolved disagreements and missing evidence",
        "unknowns, limitations, and stop conditions",
        "intentional design",
    ):
        assert phrase in text


def test_workflow_separates_repair_review_and_publication():
    text = _skill()

    assert "Do not open a public PR from the repair step" in text
    assert "independent review" in text
    assert "immediately before opening one neutral public PR" in text
    assert "Never approve, merge, force-push" in text


def test_workflow_fails_closed_on_incomplete_or_competing_evidence():
    text = _skill()

    for phrase in (
        "create no repair card",
        "An open competing PR is a rejection",
        "A changed head",
        "missing receipt",
        "partial, rate-limited, or ambiguous snapshot",
    ):
        assert phrase in text

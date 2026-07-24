"""Hermetic contract tests for the bundled hermes-loop skill."""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_DIR = REPO_ROOT / "skills" / "autonomous-ai-agents" / "hermes-loop"
SKILL_MD = SKILL_DIR / "SKILL.md"
PROTOCOL = SKILL_DIR / "references" / "protocol.md"
SPEC = SKILL_DIR / "references" / "spec-orchestrate.md"
BUILD = SKILL_DIR / "references" / "build.md"
REVIEW = SKILL_DIR / "references" / "review.md"
TEMPLATES = SKILL_DIR / "templates"


def _read(path: Path) -> str:
    assert path.is_file(), f"missing {path}"
    return path.read_text(encoding="utf-8")


def test_skill_layout_and_frontmatter():
    text = _read(SKILL_MD)
    assert text.startswith("---")
    assert "name: hermes-loop" in text
    assert "description:" in text
    assert (SKILL_DIR / "references").is_dir()
    assert (SKILL_DIR / "templates").is_dir()
    for name in ("packet.md", "build-handoff.md", "review-verdict.md"):
        assert (TEMPLATES / name).is_file()


def test_manual_freeze_and_triage_rules():
    proto = _read(PROTOCOL)
    skill = _read(SKILL_MD)
    blob = "\n".join([proto, skill, _read(SPEC)])
    assert "auto_decompose: false" in blob
    assert "triage" in proto.lower()
    assert "specify" in blob.lower()
    assert "decompose" in blob.lower()
    assert (
        "never self-freeze" in blob.lower()
        or "never self-freezes" in blob.lower()
        or "self-freeze" in blob.lower()
    )


def test_no_kernel_review_status_for_factory_graph():
    blob = "\n".join(_read(p) for p in (PROTOCOL, SPEC, BUILD, REVIEW, SKILL_MD))
    assert "kernel" in blob.lower() and "review" in blob.lower()
    assert "ordinary" in blob.lower()
    assert "set status to `review`" not in blob.lower()
    assert "status: review" not in blob.lower()


def test_no_agent_merge():
    blob = "\n".join(_read(p) for p in (PROTOCOL, BUILD, REVIEW, SKILL_MD))
    assert "never" in blob.lower() and "merge" in blob.lower()
    assert (
        "auto-merge" in blob.lower()
        or "auto merge" in blob.lower()
        or "enable auto-merge" in blob.lower()
    )


def test_sha_and_packet_evidence_fields():
    proto = _read(PROTOCOL)
    handoff = _read(TEMPLATES / "build-handoff.md")
    verdict = _read(TEMPLATES / "review-verdict.md")
    assert "packet version" in handoff.lower()
    assert "full commit sha" in handoff.lower()
    assert "FULL_SHA" in verdict or "full commit sha" in verdict.lower()
    assert "invalidat" in proto.lower()
    assert "not configured" in proto.lower() or "not configured" in verdict.lower()


def test_optional_labels_are_not_authority():
    proto = _read(PROTOCOL)
    assert "optional" in proto.lower()
    assert "label" in proto.lower()
    assert "authority" in proto.lower() or "never grant" in proto.lower()


def test_reviewer_fixer_mode_off():
    rev = _read(REVIEW)
    assert (
        "fixer mode off" in rev.lower()
        or "do not fix" in rev.lower()
    )
    assert "push" in rev.lower()

"""Regression checks for cross-skill routing in the arxiv skill."""

from pathlib import Path


SKILL = (
    Path(__file__).resolve().parents[2]
    / "skills"
    / "research"
    / "arxiv"
    / "SKILL.md"
)


def test_local_pdf_processing_routes_to_consolidated_pdf_skill() -> None:
    text = SKILL.read_text(encoding="utf-8")
    assert "For local PDF processing, see the `pdf` skill" in text
    assert "`ocr-and-documents`" not in text
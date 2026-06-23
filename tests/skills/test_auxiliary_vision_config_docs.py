"""Regression guards for skill/docs references to auxiliary vision config."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SEARCH_ROOTS = (
    REPO_ROOT / "skills",
    REPO_ROOT / "optional-skills",
    REPO_ROOT / "website" / "docs",
    REPO_ROOT / "docs",
)


def test_skills_and_docs_do_not_reference_removed_computer_vision_config_key():
    """Issue #51150: docs/skills must point at auxiliary.vision, not a dead key."""
    stale = []
    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".md", ".mdx"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if "auxiliary.computer_vision" in text or "computer_vision" in text:
                stale.append(str(path.relative_to(REPO_ROOT)))

    assert stale == []

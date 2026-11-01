"""Static checks for the deepagents skill (optional-skills/autonomous-ai-agents)."""
from pathlib import Path

import yaml

SKILL_PATH = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "autonomous-ai-agents"
    / "deepagents"
    / "SKILL.md"
)


def _load():
    content = SKILL_PATH.read_text(encoding="utf-8")
    assert content.startswith("---")
    end = content.index("\n---\n", 3)
    fm = yaml.safe_load(content[3:end])
    body = content[end + 5 :]
    return fm, body


def test_frontmatter_shape():
    fm, _ = _load()
    assert fm["name"] == "deepagents"
    assert len(fm["description"]) <= 60
    assert fm["description"].rstrip().endswith(".")
    assert fm["platforms"] == ["linux", "macos", "windows"]
    assert "deepagents" not in fm["description"].lower().replace(" deep agents", "")


def test_related_skills_are_siblings():
    fm, _ = _load()
    related = fm["metadata"]["hermes"]["related_skills"]
    for name in related:
        assert name in {"claude-code", "codex", "opencode", "hermes-agent"}


def test_body_covers_required_sections():
    _, body = _load()
    for heading in (
        "## When to Use",
        "## Prerequisites",
        "## How to Run",
        "## Pitfalls",
        "## Verification",
    ):
        assert heading in body, f"missing section: {heading}"


def test_no_machine_local_paths():
    _, body = _load()
    assert "C:\\Users\\nextbrain" not in body
    assert "/home/tony" not in body


def test_mentions_free_model_pitfall():
    _, body = _load()
    assert "insufficient_credits_for_paid_model" in body
    assert ":free" in body

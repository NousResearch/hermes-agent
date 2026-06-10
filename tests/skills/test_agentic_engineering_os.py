from pathlib import Path
import re

import yaml


ROOT = Path(__file__).resolve().parents[2]
SKILL_NAMES = ["code-structure", "gpt-loop", "code-simplifier"]


def _load_skill(name: str):
    path = ROOT / "skills" / "software-development" / name / "SKILL.md"
    content = path.read_text(encoding="utf-8")
    match = re.search(r"\n---\s*\n", content[4:])
    assert match, f"frontmatter not closed for {name}"
    frontmatter = yaml.safe_load(content[4 : match.start() + 4])
    body = content[match.end() + 4 :]
    return path, content, frontmatter, body


def test_agentic_engineering_skills_have_valid_frontmatter():
    for name in SKILL_NAMES:
        path, content, frontmatter, body = _load_skill(name)
        assert path.exists()
        assert content.startswith("---\n")
        assert frontmatter["name"] == name
        assert frontmatter["description"]
        assert len(frontmatter["description"]) <= 1024
        assert frontmatter["version"] == "1.0.0"
        assert frontmatter["author"] == "Hermes Agent"
        assert frontmatter["license"] == "MIT"
        assert "metadata" in frontmatter
        assert body.strip()


def test_agentic_engineering_skills_use_modern_sections():
    required_sections = [
        "## When to Use",
        "## Prerequisites",
        "## How to Run",
        "## Quick Reference",
        "## Procedure",
        "## Pitfalls",
        "## Verification",
    ]
    for name in SKILL_NAMES:
        _, _, _, body = _load_skill(name)
        for section in required_sections:
            assert section in body, f"{section} missing from {name}"


def test_agentic_engineering_os_is_documented_in_always_on_context():
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert "## Agentic Engineering OS" in agents
    for name in SKILL_NAMES:
        assert name in agents
    assert ".references/" in agents


def test_local_references_are_ignored_except_readme_and_gitkeep():
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert ".references/*" in gitignore
    assert "!.references/README.md" in gitignore
    assert "!.references/.gitkeep" in gitignore
    assert (ROOT / ".references" / "README.md").exists()
    assert (ROOT / ".references" / ".gitkeep").exists()

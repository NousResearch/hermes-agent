from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


REQUIRED_SKILLS = {
    "skills/autonomous-ai-agents/hermes-self-improvement/SKILL.md": [
        "Context refresh loop",
        "Planner / worker / reviewer orchestration",
        "GitHub / repo automation gates",
        "Daily research improvement loop",
    ],
    "skills/software-development/context-refresh-loop/SKILL.md": [
        "Trigger points",
        "Refresh format",
        "Rehydrate before acting",
    ],
    "skills/software-development/orchestration-roles/SKILL.md": [
        "Role contract",
        "Worktree-first rule",
        "Hard gates",
    ],
    "skills/github/github-flow-automation/SKILL.md": [
        "Safe default flow",
        "Review gate checklist",
        "CI fix loop",
    ],
    "skills/research/daily-improvement-loop/SKILL.md": [
        "Daily job output",
        "Implementation gate",
        "Durable learning",
    ],
}


def _frontmatter_and_body(path: Path):
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---\n")
    _, frontmatter, body = text.split("---", 2)
    return yaml.safe_load(frontmatter), body


def test_self_improvement_skill_bundle_is_repo_backed_and_parseable():
    for rel_path, sections in REQUIRED_SKILLS.items():
        path = ROOT / rel_path
        assert path.exists(), rel_path
        frontmatter, body = _frontmatter_and_body(path)
        assert frontmatter["name"]
        assert frontmatter["description"]
        assert frontmatter["version"] == "1.0.0"
        assert frontmatter["metadata"]["hermes"]["tags"]
        for section in sections:
            assert section in body, f"{section!r} missing from {rel_path}"


def test_self_improvement_workflows_keep_core_narrow():
    """Self-improvement should be encoded as skills/gates, not new core tools."""
    changed_skill_paths = [Path(path) for path in REQUIRED_SKILLS]
    assert all(path.parts[0] == "skills" for path in changed_skill_paths)
    assert "new core tool" not in (ROOT / "skills/autonomous-ai-agents/hermes-self-improvement/SKILL.md").read_text(encoding="utf-8").lower()

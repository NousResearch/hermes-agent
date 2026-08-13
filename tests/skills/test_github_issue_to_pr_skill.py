"""Tests for the github-issue-to-pr bundled skill."""
import re
from pathlib import Path

import yaml

SKILL_PATH = (
    Path(__file__).resolve().parents[2]
    / "skills"
    / "github"
    / "github-issue-to-pr"
    / "SKILL.md"
)


def _frontmatter_and_body():
    content = SKILL_PATH.read_text(encoding="utf-8")
    assert content.startswith("---")
    m = re.search(r"\n---\s*\n", content[3:])
    assert m, "frontmatter must close with ---"
    fm = yaml.safe_load(content[3 : m.start() + 3])
    body = content[m.end() + 3 :]
    return fm, body


def test_skill_file_exists():
    assert SKILL_PATH.is_file()


def test_frontmatter_required_fields():
    fm, _ = _frontmatter_and_body()
    for field in ("name", "description", "version", "author", "license", "platforms"):
        assert field in fm, f"missing frontmatter field: {field}"
    assert fm["name"] == "github-issue-to-pr"
    hermes = fm["metadata"]["hermes"]
    assert hermes["tags"]
    assert "related_skills" in hermes


def test_description_hardline():
    fm, _ = _frontmatter_and_body()
    desc = fm["description"]
    assert len(desc) <= 60, f"description is {len(desc)} chars; hardline is 60"
    assert desc.endswith(".")


def test_author_credits_human_first():
    fm, _ = _frontmatter_and_body()
    assert not fm["author"].startswith("Hermes Agent"), "human contributor must be credited first"
    assert "benbarclay" in fm["author"]


def test_related_skills_resolve_in_repo():
    fm, _ = _frontmatter_and_body()
    repo_root = SKILL_PATH.parents[3]
    for name in fm["metadata"]["hermes"]["related_skills"]:
        hits = (
            list(repo_root.glob(f"skills/*/{name}/SKILL.md"))
            + list(repo_root.glob(f"optional-skills/*/{name}/SKILL.md"))
            + list(repo_root.glob(f"skills/*/*/{name}/SKILL.md"))
        )
        assert hits, f"related_skills entry does not resolve in-repo: {name}"


def test_body_structure_and_size():
    _, body = _frontmatter_and_body()
    for section in ("## When to Use", "## Procedure", "## Pitfalls", "## Verification"):
        assert section in body, f"missing section: {section}"
    assert len(SKILL_PATH.read_text(encoding="utf-8")) <= 100_000


def test_no_machine_local_paths():
    content = SKILL_PATH.read_text(encoding="utf-8")
    assert "/home/" not in content
    assert not re.search(r"[A-Z]:\\\\Users", content)


def test_steps_have_completion_criteria():
    _, body = _frontmatter_and_body()
    steps = re.findall(r"^### \d+\..*?(?=^### \d+\.|^## )", body, re.MULTILINE | re.DOTALL)
    assert len(steps) >= 6
    for step in steps:
        assert "Done when" in step, f"step missing completion criterion: {step[:60]!r}"


def test_core_disciplines_present():
    """The learnings folded in from maintainer practice must survive edits."""
    _, body = _frontmatter_and_body()
    assert "--comments" in body, "must read the full issue thread"
    assert "pr list --search" in body, "must sweep for duplicate PRs before coding"
    assert re.search(r"git log -p -S", body), "must check design intent via history"
    assert "sabotage" in body.lower() or "FAILS" in body, "must prove the regression test bites"
    assert "sibling" in body, "must fix the class, not the site"
    assert "dispatches CI" in body, "must open the PR immediately after work exists"


def test_not_a_router_skill():
    """Steps must carry their own procedure, not just route to sibling skills."""
    _, body = _frontmatter_and_body()
    steps = re.findall(r"^### \d+\..*?(?=^### \d+\.|^## )", body, re.MULTILINE | re.DOTALL)
    routing = [s for s in steps if re.match(r"^### \d+\.[^\n]*\n+Load `", s)]
    assert len(routing) == 0, "steps must not open by delegating to another skill"

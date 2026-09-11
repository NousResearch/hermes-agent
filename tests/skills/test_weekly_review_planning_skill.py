"""Tests for the weekly-review-planning skill and blueprint->skill wiring."""
import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_PATH = (
    REPO_ROOT / "skills" / "productivity" / "weekly-review-planning" / "SKILL.md"
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
    assert fm["name"] == "weekly-review-planning"


def test_description_hardline():
    fm, _ = _frontmatter_and_body()
    desc = fm["description"]
    assert len(desc) <= 60, f"description is {len(desc)} chars; hardline is 60"
    assert desc.endswith(".")


def test_author_credits_human_first():
    fm, _ = _frontmatter_and_body()
    assert not fm["author"].startswith("Hermes Agent")
    assert "benbarclay" in fm["author"]


def test_related_skills_resolve_in_repo():
    fm, _ = _frontmatter_and_body()
    for name in fm["metadata"]["hermes"]["related_skills"]:
        hits = (
            list(REPO_ROOT.glob(f"skills/*/{name}/SKILL.md"))
            + list(REPO_ROOT.glob(f"optional-skills/*/{name}/SKILL.md"))
            + list(REPO_ROOT.glob(f"skills/*/*/{name}/SKILL.md"))
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


def test_steps_have_completion_criteria():
    _, body = _frontmatter_and_body()
    steps = re.findall(r"^### \d+\..*?(?=^### \d+\.|^## )", body, re.MULTILINE | re.DOTALL)
    assert len(steps) >= 6
    for step in steps:
        assert "Done when" in step, f"step missing completion criterion: {step[:60]!r}"


def _skill_dir_exists(name: str) -> bool:
    return bool(
        list(REPO_ROOT.glob(f"skills/*/{name}/SKILL.md"))
        + list(REPO_ROOT.glob(f"skills/*/*/{name}/SKILL.md"))
    )


def test_blueprint_loads_this_skill():
    from cron.blueprint_catalog import CATALOG

    bp = next(b for b in CATALOG if b.key == "weekly-review")
    assert "weekly-review-planning" in bp.skills
    assert "weekly-review-planning" in bp.prompt_template


def test_every_blueprint_skill_resolves_in_repo():
    """Invariant: any skill a blueprint loads must exist as a bundled skill."""
    from cron.blueprint_catalog import CATALOG

    for bp in CATALOG:
        for skill_name in bp.skills:
            assert _skill_dir_exists(skill_name), (
                f"blueprint {bp.key!r} loads nonexistent skill {skill_name!r}"
            )


def test_task_skill_blueprints_are_wired():
    """The recurring-task blueprints must load their procedure skills."""
    from cron.blueprint_catalog import CATALOG

    expected = {
        "morning-brief": "google-workspace",
        "important-mail": "email-inbox-triage",
        "weekly-review": "weekly-review-planning",
        "price-watch": "product-price-monitor",
    }
    by_key = {b.key: b for b in CATALOG}
    for key, skill_name in expected.items():
        assert key in by_key, f"blueprint {key!r} missing from catalog"
        assert skill_name in by_key[key].skills, (
            f"blueprint {key!r} must load skill {skill_name!r}"
        )

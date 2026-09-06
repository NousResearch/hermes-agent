"""Tests for the competitor-news-monitor skill and competitor-watch blueprint."""
import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_PATH = (
    REPO_ROOT / "skills" / "research" / "competitor-news-monitor" / "SKILL.md"
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
    assert fm["name"] == "competitor-news-monitor"


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


def test_no_phantom_skill_references():
    content = SKILL_PATH.read_text(encoding="utf-8")
    assert "change-monitor-and-notify" not in content, "phantom skill ref must be gone"


def test_setup_tick_split():
    _, body = _frontmatter_and_body()
    assert "Setup (foreground, once)" in body
    assert "Tick (each scheduled run)" in body
    assert "cronjob(action=" in body, "must wire scheduling through the cronjob tool"


def test_coverage_honesty_discipline():
    _, body = _frontmatter_and_body()
    assert "unknown coverage" in body, "source failure != no news"
    assert "cutoff advance" in body.replace("advances", "advance").replace(
        "advanced", "advance"
    ), "cutoff must only advance on success"


def test_steps_have_completion_criteria():
    _, body = _frontmatter_and_body()
    steps = re.findall(r"^### \d+\..*?(?=^### \d+\.|^## )", body, re.MULTILINE | re.DOTALL)
    assert len(steps) >= 5
    for step in steps:
        assert "Done when" in step, f"step missing completion criterion: {step[:60]!r}"


def test_no_machine_local_paths():
    content = SKILL_PATH.read_text(encoding="utf-8")
    assert "/home/" not in content


def test_competitor_watch_blueprint_registered():
    from cron.blueprint_catalog import CATALOG

    bp = next((b for b in CATALOG if b.key == "competitor-watch"), None)
    assert bp is not None, "competitor-watch blueprint missing from catalog"
    assert "competitor-news-monitor" in bp.skills
    slot_names = {s.name for s in bp.slots}
    assert {"companies", "categories", "time", "recurrence", "deliver"} <= slot_names
    assert "[SILENT]" in bp.prompt_template
    assert "{companies}" in bp.prompt_template and "{categories}" in bp.prompt_template


def test_every_blueprint_skill_resolves_in_repo():
    from cron.blueprint_catalog import CATALOG

    for bp in CATALOG:
        for skill_name in bp.skills:
            hits = list(REPO_ROOT.glob(f"skills/*/{skill_name}/SKILL.md")) + list(
                REPO_ROOT.glob(f"skills/*/*/{skill_name}/SKILL.md")
            )
            assert hits, f"blueprint {bp.key!r} loads nonexistent skill {skill_name!r}"

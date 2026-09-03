"""Tests for the product-price-monitor skill and its price-watch blueprint."""
import re
from pathlib import Path

import yaml

SKILL_PATH = (
    Path(__file__).resolve().parents[2]
    / "skills"
    / "productivity"
    / "product-price-monitor"
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
    assert fm["name"] == "product-price-monitor"


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
    repo_root = SKILL_PATH.parents[3]
    for name in fm["metadata"]["hermes"]["related_skills"]:
        hits = (
            list(repo_root.glob(f"skills/*/{name}/SKILL.md"))
            + list(repo_root.glob(f"optional-skills/*/{name}/SKILL.md"))
            + list(repo_root.glob(f"skills/*/*/{name}/SKILL.md"))
        )
        assert hits, f"related_skills entry does not resolve in-repo: {name}"


def test_no_phantom_connectors_in_prose():
    _, body = _frontmatter_and_body()
    assert "flight-research" not in body, "no phantom 'flight-research' skill references"


def test_setup_tick_split():
    """The skill must separate one-time setup from the recurring cron tick."""
    _, body = _frontmatter_and_body()
    assert "Setup (foreground, once)" in body
    assert "Tick (each scheduled run)" in body
    assert "cronjob(action=" in body, "must wire scheduling through the cronjob tool"
    assert "Do not schedule until one foreground fetch works" in body


def test_state_discipline_present():
    _, body = _frontmatter_and_body()
    assert "never overwrite the last good observation" in body
    assert "fingerprint" in body


def test_steps_have_completion_criteria():
    _, body = _frontmatter_and_body()
    steps = re.findall(r"^### \d+\..*?(?=^### \d+\.|^## )", body, re.MULTILINE | re.DOTALL)
    assert len(steps) >= 5
    for step in steps:
        assert "Done when" in step, f"step missing completion criterion: {step[:60]!r}"


def test_no_machine_local_paths():
    content = SKILL_PATH.read_text(encoding="utf-8")
    assert not re.search(r"/home/(?!.*price-watches)", content)
    assert "/home/bb" not in content


def test_price_watch_blueprint_registered():
    from cron.blueprint_catalog import CATALOG

    bp = next((b for b in CATALOG if b.key == "price-watch"), None)
    assert bp is not None, "price-watch blueprint missing from catalog"
    assert "product-price-monitor" in bp.skills, "blueprint must load the skill"
    slot_names = {s.name for s in bp.slots}
    assert {"item", "condition", "interval_h", "deliver"} <= slot_names
    assert "[SILENT]" in bp.prompt_template, "silent path must be explicit"
    assert "{item}" in bp.prompt_template and "{condition}" in bp.prompt_template


def test_price_watch_blueprint_schedule_resolves():
    from cron.blueprint_catalog import CATALOG

    bp = next(b for b in CATALOG if b.key == "price-watch")
    interval_slot = next(s for s in bp.slots if s.name == "interval_h")
    for opt in interval_slot.options:
        expr = bp.schedule_template.format(interval_h=opt)
        fields = expr.split()
        assert len(fields) == 5, f"invalid cron expr: {expr}"
        assert fields[1] == f"*/{opt}"

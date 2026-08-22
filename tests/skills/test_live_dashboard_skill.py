"""Tests for the live-dashboard skill and its live-dashboard blueprint.

Inspired by Energy's (getenergy.com) natural-language live dashboards —
describe what you want to see in one sentence, get a persistent
self-refreshing status page fed by email/web/file sources.
"""
import re
from pathlib import Path

import yaml

SKILL_PATH = (
    Path(__file__).resolve().parents[2]
    / "skills"
    / "productivity"
    / "live-dashboard"
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
    assert fm["name"] == "live-dashboard"


def test_description_hardline():
    fm, _ = _frontmatter_and_body()
    desc = fm["description"]
    assert len(desc) <= 60, f"description is {len(desc)} chars; hardline is 60"
    assert desc.endswith(".")


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


def test_setup_tick_split():
    """The skill must separate one-time setup from the recurring cron tick."""
    _, body = _frontmatter_and_body()
    assert "Setup (foreground, once)" in body
    assert "Tick (each scheduled run)" in body
    assert "cronjob(action=" in body, "must wire scheduling through the cronjob tool"


def test_state_discipline_present():
    """State-file source of truth + stale-read handling must be explicit."""
    _, body = _frontmatter_and_body()
    assert "dashboard.json" in body
    assert "source of truth" in body
    assert "last-known-good" in body or "last good value" in body


def test_source_verification_before_scheduling():
    _, body = _frontmatter_and_body()
    assert "Only after step 3 succeeded" in body
    assert "one bounded foreground read" in body


def test_silent_path_explicit():
    _, body = _frontmatter_and_body()
    assert "[SILENT]" in body, "no-change ticks must stay silent"


def test_steps_have_completion_criteria():
    _, body = _frontmatter_and_body()
    steps = re.findall(r"^### \d+\..*?(?=^### \d+\.|^## )", body, re.MULTILINE | re.DOTALL)
    assert len(steps) >= 6
    for step in steps:
        assert "Done when" in step, f"step missing completion criterion: {step[:60]!r}"


def test_html_is_projection_not_state():
    _, body = _frontmatter_and_body()
    assert "never hand-edit HTML state" in body
    assert "self-contained HTML" in body


def test_live_dashboard_blueprint_registered():
    from cron.blueprint_catalog import CATALOG

    bp = next((b for b in CATALOG if b.key == "live-dashboard"), None)
    assert bp is not None, "live-dashboard blueprint missing from catalog"
    assert "live-dashboard" in bp.skills, "blueprint must load the skill"
    slot_names = {s.name for s in bp.slots}
    assert {"purpose", "sources", "time", "recurrence", "deliver"} <= slot_names
    assert "[SILENT]" in bp.prompt_template, "silent path must be explicit"
    assert "{purpose}" in bp.prompt_template and "{sources}" in bp.prompt_template


def test_live_dashboard_blueprint_fills():
    """fill_blueprint must produce a valid cron job kwargs dict."""
    from cron.blueprint_catalog import CATALOG, fill_blueprint

    bp = next(b for b in CATALOG if b.key == "live-dashboard")
    job = fill_blueprint(
        bp,
        {
            "purpose": "team visa applications",
            "sources": "email threads and the case-status site",
            "time": "07:30",
            "recurrence": "weekdays",
            "deliver": "origin",
        },
    )
    assert "team visa applications" in job["prompt"]
    assert "email threads and the case-status site" in job["prompt"]
    fields = job["schedule"].split()
    assert len(fields) == 5, f"invalid cron expr: {job['schedule']}"
    assert fields[0] == "30" and fields[1] == "7"
    assert fields[4] == "1-5"

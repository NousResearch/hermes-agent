"""Tests for the live-dashboard optional skill.

Inspired by Energy's (getenergy.com) natural-language live dashboards —
describe what you want to see in one sentence, get a persistent
self-refreshing status page fed by email/web/file sources.
"""
import re
from pathlib import Path

import yaml

SKILL_PATH = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
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


def test_frontmatter_required_fields():
    fm, _ = _frontmatter_and_body()
    for field in ("name", "description", "version", "author", "license", "platforms"):
        assert field in fm, f"missing frontmatter field: {field}"
    assert fm["name"] == "live-dashboard"


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
    assert "never hand-edit HTML state" in body
    assert "[SILENT]" in body, "no-change ticks must stay silent"


def test_source_verification_before_scheduling():
    _, body = _frontmatter_and_body()
    assert "Only after step 3 succeeded" in body
    assert "one bounded foreground read" in body


def test_steps_have_completion_criteria():
    _, body = _frontmatter_and_body()
    steps = re.findall(r"^### \d+\..*?(?=^### \d+\.|^## )", body, re.MULTILINE | re.DOTALL)
    assert len(steps) >= 7
    for step in steps:
        assert "Done when" in step, f"step missing completion criterion: {step[:60]!r}"


def test_desktop_preview_with_path_fallback():
    """Desktop sessions render in the preview pane; everything else gets the file path.
    The Hermes home directory is never hardcoded in prose the agent executes."""
    _, body = _frontmatter_and_body()
    assert 'desktop_preview(action="open"' in body
    assert "report the absolute path" in body
    assert "~/.hermes" not in body


def test_frontmatter_blueprint_is_a_valid_installed_blueprint():
    """The install-time suggestion rides the skills-pipeline blueprint block, not a
    hard-wired catalog entry (an optional skill may not be installed)."""
    from tools.blueprints import blueprint_to_job_spec, parse_blueprint

    spec = parse_blueprint(SKILL_PATH.read_text(encoding="utf-8"))
    assert spec is not None and spec.skill_name == "live-dashboard"
    job = blueprint_to_job_spec(spec)
    assert job["skills"] == ["live-dashboard"]
    assert len(job["schedule"].split()) == 5, f"invalid cron expr: {job['schedule']}"
    assert "[SILENT]" in job["prompt"] and "~/.hermes" not in job["prompt"]

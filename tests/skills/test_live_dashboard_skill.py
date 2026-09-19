"""Tests for the live-dashboard optional skill."""
import re
from pathlib import Path

import hermes_yaml as yaml

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


def test_procedure_is_structured_setup_tick_show():
    """The skill splits into setup / scheduled tick / render phases, each phase is a
    numbered-step procedure, and every step states a completion criterion."""
    _, body = _frontmatter_and_body()
    phases = re.findall(r"^## Procedure — (.+)$", body, re.MULTILINE)
    assert len(phases) == 3, phases
    steps = re.findall(r"^### \d+\..*?(?=^### \d+\.|^## )", body, re.MULTILINE | re.DOTALL)
    assert len(steps) >= 7
    for step in steps:
        assert "Done when" in step, f"step missing completion criterion: {step[:60]!r}"
    for heading in ("## When to Use", "## Prerequisites", "## Pitfalls", "## Verification"):
        assert heading in body, heading


def test_tools_wired_and_home_not_hardcoded():
    """Scheduling goes through `cronjob`, desktop rendering through `desktop_preview`,
    no-change ticks stay silent, and the Hermes home path is resolved, never assumed."""
    _, body = _frontmatter_and_body()
    assert re.search(r"cronjob\(action=\"create\"", body)
    assert re.search(r"desktop_preview\(action=\"open\"", body)
    assert "[SILENT]" in body
    assert "dashboard.json" in body
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
"""Tests for the live-dashboard optional skill."""
from pathlib import Path

SKILL_PATH = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "productivity"
    / "live-dashboard"
    / "SKILL.md"
)


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

"""Tests for the google-workspace daily-brief reference."""
import re
from pathlib import Path

GWS_DIR = (
    Path(__file__).resolve().parents[2]
    / "skills"
    / "productivity"
    / "google-workspace"
)
REF_PATH = GWS_DIR / "references" / "daily-brief.md"
SKILL_PATH = GWS_DIR / "SKILL.md"


def test_reference_exists():
    assert REF_PATH.is_file()


def test_skill_md_points_at_reference():
    content = SKILL_PATH.read_text(encoding="utf-8")
    assert "references/daily-brief.md" in content, "SKILL.md must list the reference"
    refs_section = content.split("## References", 1)[1].split("##", 1)[0]
    assert "daily-brief" in refs_section


def test_reference_carries_load_trigger():
    content = REF_PATH.read_text(encoding="utf-8")
    assert "morning brief" in content.lower(), "reference must state when to load it"


def test_reference_credits_contributor():
    content = REF_PATH.read_text(encoding="utf-8")
    assert "benbarclay" in content


def test_steps_have_completion_criteria():
    content = REF_PATH.read_text(encoding="utf-8")
    steps = re.findall(r"^### \d+\..*?(?=^### \d+\.|^## )", content, re.MULTILINE | re.DOTALL)
    assert len(steps) >= 5
    for step in steps:
        assert "Done when" in step, f"step missing completion criterion: {step[:60]!r}"


def test_core_disciplines_present():
    content = REF_PATH.read_text(encoding="utf-8")
    assert "half-open window" in content or "[day_start, next_day_start)" in content
    assert "no preparation found" in content
    assert "not authorization to mutate" in content


def test_no_machine_local_paths():
    content = REF_PATH.read_text(encoding="utf-8")
    assert "/home/" not in content


def test_no_standalone_daily_brief_skill():
    """The brief ships as a reference, not a sibling skill."""
    repo_root = GWS_DIR.parents[2]
    assert not (repo_root / "skills" / "productivity" / "google-workspace-daily-brief").exists()

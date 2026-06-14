from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DCF_SKILL = REPO_ROOT / "optional-skills" / "finance" / "dcf-model" / "SKILL.md"


def test_dcf_skill_uses_resolved_recalc_helper_path():
    """DCF instructions must not execute recalc.py from the workspace CWD."""
    text = DCF_SKILL.read_text(encoding="utf-8")

    assert "python recalc.py" not in text
    assert "python <excel_author_skill_dir>/scripts/recalc.py" in text
    assert "skill_dir" in text

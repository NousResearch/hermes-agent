"""Regression coverage for documented script paths in bundled skills."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _skill(path: str) -> str:
    return (ROOT / path / "SKILL.md").read_text(encoding="utf-8")


def test_skill_script_references_identify_their_real_location() -> None:
    """Do not present repo tools or project examples as files shipped by a skill."""
    research = _skill("optional-skills/research/research-paper-writing")
    authoring = _skill("skills/software-development/hermes-agent-skill-authoring")
    desktop_dom = _skill("skills/software-development/inspecting-hermes-desktop-dom")
    debugpy = _skill("skills/software-development/python-debugpy")
    debugging = _skill("skills/software-development/systematic-debugging")

    assert "your-method/scripts/reproduce_table1.sh" in research
    assert "your-method/scripts/make_figure2.py" in research
    assert "To reproduce Table 1: `bash scripts/reproduce_table1.sh`" not in research
    assert "To reproduce Figure 2: `python scripts/make_figure2.py`" not in research

    assert "From the Hermes source checkout root" in authoring
    assert "./website/scripts/generate-skill-docs.py" in authoring
    assert (ROOT / "scripts/run_tests.sh").is_file()
    assert (ROOT / "website/scripts/generate-skill-docs.py").is_file()

    assert "apps/desktop/scripts/eval.mjs" in desktop_dom
    assert "apps/desktop/scripts/perf/lib/cdp.mjs" in desktop_dom
    assert "apps/desktop/scripts/profile-typing-lag.md" in desktop_dom
    assert (ROOT / "apps/desktop/scripts/eval.mjs").is_file()
    assert (ROOT / "apps/desktop/scripts/perf/lib/cdp.mjs").is_file()
    assert (ROOT / "apps/desktop/scripts/profile-typing-lag.md").is_file()

    assert "from the Hermes source checkout root" in debugpy
    assert "scripted repro you created in the working project" in debugging
    assert "python scripts/repro_bug.py" not in debugging

from io import StringIO

from rich.console import Console

from hermes_cli.skills_hub import do_validate


def test_do_validate_prints_bundled_skill_issues(tmp_path):
    skill_dir = tmp_path / "productivity" / "slides"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        """---
name: slides
related_skills: [missing-helper]
---

Run `python scripts/missing.py`.
""",
        encoding="utf-8",
    )
    out = StringIO()
    console = Console(file=out, force_terminal=False, color_system=None, width=140)

    count = do_validate(bundled=True, bundled_dir=tmp_path, console=console)

    output = out.getvalue()
    assert count == 2
    assert "Bundled skill validation found 2 issue(s)" in output
    assert "slides" in output
    assert "missing-file-reference" in output
    assert "unknown-related-skill" in output


def test_do_validate_requires_scope(tmp_path):
    out = StringIO()
    console = Console(file=out, force_terminal=False, color_system=None)

    count = do_validate(bundled=False, bundled_dir=tmp_path, console=console)

    assert count == 0
    assert "--bundled" in out.getvalue()


def test_do_validate_can_run_selected_checks(tmp_path):
    skill_dir = tmp_path / "productivity" / "slides"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        """---
name: slides
related_skills: [missing-helper]
---

Run `python scripts/missing.py`.
""",
        encoding="utf-8",
    )
    out = StringIO()
    console = Console(file=out, force_terminal=False, color_system=None, width=140)

    count = do_validate(
        bundled=True,
        bundled_dir=tmp_path,
        checks=["related-skills"],
        console=console,
    )

    output = out.getvalue()
    assert count == 1
    assert "unknown-related-skill" in output
    assert "missing-file-reference" not in output

"""The classic CLI banner should make an actually-empty skills state actionable."""

from hermes_cli.banner import _banner_skill_lines


def _lines(skills_by_category, *, enabled=True):
    return _banner_skill_lines(skills_by_category, enabled, dim="dim", text="text")


def test_empty_skills_banner_names_browse_command():
    lines = _lines({})

    assert len(lines) == 1
    assert "No skills installed yet" in lines[0]
    assert "hermes skills browse" in lines[0]


def test_populated_skills_banner_does_not_show_empty_hint():
    lines = _lines({"general": ["note-taking"]})

    assert all("hermes skills browse" not in line for line in lines)


def test_disabled_skills_toolset_keeps_disabled_copy():
    lines = _lines({}, enabled=False)

    assert len(lines) == 1
    assert "Skills toolset disabled" in lines[0]
    assert "hermes skills browse" not in lines[0]

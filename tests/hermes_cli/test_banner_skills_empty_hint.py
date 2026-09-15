"""Empty-skills banner points the user at the skills catalog.

The welcome banner always renders an "Available Skills" section. When nothing is
installed, a bare "no skills" message leaves the user with no next step, so the
line must name the action that fixes it: `hermes skills browse`.
"""

from hermes_cli.banner import _banner_skill_lines


def _skill_lines(skills_by_category, skills_enabled=True):
    return _banner_skill_lines(
        skills_by_category, skills_enabled, dim="x", text="y"
    )


def test_empty_skills_names_the_browse_action():
    """Zero skills: the banner line must name the `hermes skills browse` action."""
    lines = _skill_lines({})
    assert lines, "expected an empty-state line for zero installed skills"
    assert "skills browse" in "".join(lines)


def test_skill_list_never_shows_the_empty_hint():
    """The hint only applies when there really are no skills installed."""
    lines = _skill_lines({"general": ["note-taking"]})
    assert "skills browse" not in "".join(lines)

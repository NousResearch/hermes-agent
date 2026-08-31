"""The session_platforms frontmatter gate (skills-index slim PR).

A skill whose metadata.hermes.session_platforms names gateway channels is
hidden from the skills index on every other channel; unknown platform
fails OPEN (offline builds/tests must not hide skills).
"""
from agent.prompt_builder import _skill_should_show
from agent.skill_utils import extract_skill_conditions


def _conds(platforms):
    return extract_skill_conditions(
        {"metadata": {"hermes": {"session_platforms": platforms}}}
    )


class TestSessionPlatformGate:
    def test_hidden_on_other_channel(self):
        assert _skill_should_show(_conds(["teams", "cron"]), {"terminal"}, set(), "desktop") is False
        assert _skill_should_show(_conds(["teams", "cron"]), {"terminal"}, set(), "telegram") is False

    def test_shown_on_named_channel(self):
        assert _skill_should_show(_conds(["teams", "cron"]), {"terminal"}, set(), "teams") is True
        assert _skill_should_show(_conds(["teams", "cron"]), {"terminal"}, set(), "cron") is True

    def test_case_insensitive(self):
        assert _skill_should_show(_conds(["Teams"]), {"terminal"}, set(), "TEAMS") is True

    def test_unknown_platform_fails_open(self):
        assert _skill_should_show(_conds(["teams"]), {"terminal"}, set(), None) is True
        assert _skill_should_show(_conds(["teams"]), {"terminal"}, set(), "") is True

    def test_empty_list_visible_everywhere(self):
        assert _skill_should_show(_conds([]), {"terminal"}, set(), "desktop") is True

    def test_gate_runs_even_without_tool_info(self):
        # The channel gate is independent of tool-filtering backward compat.
        assert _skill_should_show(_conds(["teams"]), None, None, "desktop") is False

    def test_teams_meeting_pipeline_carries_the_gate(self):
        from pathlib import Path
        import re, yaml

        p = Path(__file__).resolve().parents[2] / "skills" / "productivity" / "teams-meeting-pipeline" / "SKILL.md"
        content = p.read_text(encoding="utf-8")
        m = re.search(r"\n---\s*\n", content[3:])
        fm = yaml.safe_load(content[3 : m.start() + 3])
        assert fm["metadata"]["hermes"]["session_platforms"] == ["teams", "cron"]

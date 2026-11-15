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

    def test_current_session_platform_hint_reads_session_source(self, monkeypatch):
        from agent.prompt_builder import _current_session_platform_hint
        from gateway.session_context import clear_session_vars, set_session_vars

        monkeypatch.delenv("HERMES_PLATFORM", raising=False)
        monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)
        monkeypatch.delenv("HERMES_SESSION_SOURCE", raising=False)
        tokens = set_session_vars(source="desktop")
        try:
            assert _current_session_platform_hint() == "desktop"
            assert _skill_should_show(_conds(["teams", "cron"]), {"terminal"}, set(), _current_session_platform_hint()) is False
        finally:
            clear_session_vars(tokens)

    def test_skip_first_secret_skips_remaining_missing_entries(self, monkeypatch):
        from tools import skills_tool
        from tools.skills_tool_setup import _capture_required_environment_variables

        asked = []
        def _cancel_first(var_name, prompt, metadata=None):
            asked.append(var_name)
            return {"success": True, "stored_as": var_name, "validated": False, "skipped": True}

        with skills_tool.bind_turn_secret_capture(_cancel_first):
            res = _capture_required_environment_variables(
                "teams-meeting-pipeline",
                [
                    {"name": "MSGRAPH_TENANT_ID", "prompt": "Enter value for MSGRAPH_TENANT_ID"},
                    {"name": "MSGRAPH_CLIENT_ID", "prompt": "Enter value for MSGRAPH_CLIENT_ID"},
                    {"name": "MSGRAPH_CLIENT_SECRET", "prompt": "Enter value for MSGRAPH_CLIENT_SECRET"},
                ],
            )
        assert asked == ["MSGRAPH_TENANT_ID"]
        assert res["missing_names"] == ["MSGRAPH_TENANT_ID", "MSGRAPH_CLIENT_ID", "MSGRAPH_CLIENT_SECRET"]
        assert res["setup_skipped"] is True

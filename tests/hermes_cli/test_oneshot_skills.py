"""Regression tests: -z/--oneshot must honor -s/--skills (#31548, #65119).

The oneshot path builds its AIAgent directly (bypassing HermesCLI), so the
--skills preload has to be forwarded explicitly through its dedicated stable
prompt field. These tests pin the forwarding contract and the
partial-success semantics shared with normal CLI chat.
"""

import pytest

from hermes_cli.oneshot import _build_preloaded_skills_prompt, _normalize_skills

class TestNormalizeSkills:
    def test_none_and_empty(self):
        assert _normalize_skills(None) == []
        assert _normalize_skills("") == []
        assert _normalize_skills([]) == []

    def test_comma_separated_string(self):
        assert _normalize_skills("a,b") == ["a", "b"]

    def test_repeated_flags_deduped_order_preserved(self):
        assert _normalize_skills(["b", "a", "b"]) == ["b", "a"]

class TestBuildPreloadedSkillsPrompt:
    def test_no_skills_returns_none(self):
        assert _build_preloaded_skills_prompt(None) is None

    def test_all_missing_raises(self, monkeypatch):
        import agent.skill_commands as sc

        monkeypatch.setattr(
            sc, "build_preloaded_skills_prompt",
            lambda parsed, **kw: ("", [], list(parsed)),
        )
        with pytest.raises(ValueError, match="Unknown skill"):
            _build_preloaded_skills_prompt("not-a-skill")

    def test_partial_success_returns_prompt(self, monkeypatch):
        import agent.skill_commands as sc

        monkeypatch.setattr(
            sc, "build_preloaded_skills_prompt",
            lambda parsed, **kw: ("PROMPT", ["good"], ["bad"]),
        )
        assert _build_preloaded_skills_prompt(["good", "bad"]) == "PROMPT"


def test_oneshot_forwards_preloaded_skills_to_stable_prompt_field(monkeypatch):
    import hermes_cli.oneshot as oneshot_mod

    captured = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.session_id = "session"

        def run_conversation(self, *_args, **_kwargs):
            return {"final_response": "ok", "session_id": self.session_id}

        def close(self):
            pass

    monkeypatch.setattr(oneshot_mod, "_create_session_db_for_oneshot", lambda: None)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"model": {"default": "m", "provider": "openai"}})
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", lambda **_kw: {
        "api_key": "key", "base_url": None, "provider": "openai", "api_mode": "chat", "credential_pool": None,
    })
    monkeypatch.setattr("hermes_cli.tools_config._get_platform_tools", lambda *_args: [])
    monkeypatch.setattr("hermes_cli.mcp_startup.ensure_mcp_discovery_before_agent_build", lambda **_kw: None)
    monkeypatch.setattr(oneshot_mod, "_build_preloaded_skills_prompt", lambda _skills: "SKILL PROMPT")
    monkeypatch.setattr("run_agent.AIAgent", FakeAgent)

    oneshot_mod._run_agent("hello", skills="skill")

    assert captured["preloaded_skills_prompt"] == "SKILL PROMPT"
    assert "ephemeral_system_prompt" not in captured

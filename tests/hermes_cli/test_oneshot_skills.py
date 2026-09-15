"""Regression tests: -z/--oneshot must honor -s/--skills (#31548, #65119).

The oneshot path builds its AIAgent directly (bypassing HermesCLI), so the
--skills preload has to be forwarded explicitly and injected via
``ephemeral_system_prompt``. These tests pin the forwarding contract and the
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

    def test_loaded_prompt_returned(self, monkeypatch):
        import agent.skill_commands as sc

        monkeypatch.setattr(
            sc, "build_preloaded_skills_prompt",
            lambda parsed, **kw: ("SKILL CONTENT", ["s"], []),
        )
        assert _build_preloaded_skills_prompt("s") == "SKILL CONTENT"

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

    def test_advisory_missing_skill_warns_instead_of_killing_the_run(self, monkeypatch, caplog):
        """The dispatcher's injected review skill must never be the reason a worker dies at INIT.

        Regression: a review card carrying no skills of its own reached the preload path with the
        injected ``sdlc-review`` as the ONLY name, nothing loaded, and the path raised
        ``Unknown skill(s): sdlc-review`` — every review run died before its first tool call.
        """
        import logging

        import agent.skill_commands as sc

        monkeypatch.setattr(
            sc, "build_preloaded_skills_prompt",
            lambda parsed, **kw: ("", [], list(parsed)),
        )
        monkeypatch.setenv("HERMES_KANBAN_ADVISORY_SKILLS", "sdlc-review")

        with caplog.at_level(logging.WARNING):
            assert _build_preloaded_skills_prompt("sdlc-review") is None
        assert "sdlc-review" in caplog.text

    def test_requested_missing_skill_still_fails_loudly_beside_an_advisory_one(self, monkeypatch):
        """Only the harness-injected name is advisory: a genuinely requested typo still raises."""
        import agent.skill_commands as sc

        monkeypatch.setattr(
            sc, "build_preloaded_skills_prompt",
            lambda parsed, **kw: ("", [], list(parsed)),
        )
        monkeypatch.setenv("HERMES_KANBAN_ADVISORY_SKILLS", "sdlc-review")

        with pytest.raises(ValueError, match=r"Unknown skill\(s\): typo-skill"):
            _build_preloaded_skills_prompt(["sdlc-review", "typo-skill"])

    def test_advisory_name_matches_by_lookup_path_leaf(self, monkeypatch):
        """``is_advisory_skill`` accepts a path-shaped identifier for the same skill."""
        import agent.skill_commands as sc

        monkeypatch.setenv("HERMES_KANBAN_ADVISORY_SKILLS", "sdlc-review")
        assert sc.is_advisory_skill("devops/sdlc-review") is True
        assert sc.is_advisory_skill("sdlc-review") is True
        assert sc.is_advisory_skill("other-skill") is False

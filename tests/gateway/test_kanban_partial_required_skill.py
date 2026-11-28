"""Regression for NousResearch/hermes-agent#154555: review needs its required skill.

An optional card skill that loads must not mask a disabled review-lane skill:
the worker has not made a model call yet, so it can still refuse deterministically.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_review_skill_missing_with_another_skill_loaded_refuses_before_model(tmp_path, monkeypatch):
    from agent.skill_commands import build_preloaded_skills_prompt
    from gateway.session_kanban import _run_task_turns

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for relative, name in (("owned-skill", "owned-skill"),
                           ("devops/sdlc-review", "sdlc-review")):
        directory = home / "skills" / relative
        directory.mkdir(parents=True)
        (directory / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: Test skill\n---\nInstructions.\n",
            encoding="utf-8",
        )
    (home / "config.yaml").write_text(
        "skills:\n  disabled:\n    - sdlc-review\n", encoding="utf-8",
    )
    pins = ["owned-skill", "sdlc-review"]
    _prompt, loaded, missing = build_preloaded_skills_prompt(pins, task_id="review-session")
    assert "owned-skill" in loaded
    assert "sdlc-review" in missing

    def unexpected_model_call(*_args, **_kwargs):
        raise AssertionError("review ran without its required sdlc-review skill")

    agent = SimpleNamespace(session_id="review-session", run_conversation=unexpected_model_call)
    with pytest.raises(ValueError, match="skill not found: sdlc-review"):
        _run_task_turns(
            agent, {"text": "Review this change"}, [],
            {"skills": pins, "profile": "reviewer", "context": ""},
        )

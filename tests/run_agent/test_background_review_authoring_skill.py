"""Background-review integration for bundled skill-authoring guidance."""

from __future__ import annotations

from types import SimpleNamespace

from agent import background_review as br
from agent.skill_authoring_guidance import SkillAuthoringGuidance


def _guidance(
    *,
    skill: str = "CURRENT-V2-SKILL",
    contract: str | None = "CURRENT-V2-CONTRACT",
) -> SkillAuthoringGuidance:
    return SkillAuthoringGuidance(
        skill_content=skill,
        contract_content=contract,
    )


def _agent() -> SimpleNamespace:
    return SimpleNamespace(
        platform="test",
        _cached_system_prompt="STALE-SYSTEM-PROMPT-WITH-OLD-AUTHORING",
        _MEMORY_REVIEW_PROMPT="review memory",
        _SKILL_REVIEW_PROMPT="review skills",
        _COMBINED_REVIEW_PROMPT="review both",
    )


def test_missing_authoring_skill_keeps_existing_prompt(monkeypatch):
    monkeypatch.setattr(
        br,
        "load_bundled_skill_authoring_guidance",
        lambda platform=None: None,
    )

    assert br._attach_authoring_skill(
        "existing skill-review prompt",
        review_skills=True,
        platform="test",
    ) == "existing skill-review prompt"


def test_memory_only_review_does_not_load_or_add_authoring_tokens(monkeypatch):
    def _loader_must_not_run(platform=None):
        raise AssertionError("memory-only review must not load authoring guidance")

    monkeypatch.setattr(
        br,
        "load_bundled_skill_authoring_guidance",
        _loader_must_not_run,
    )

    _, prompt = br.spawn_background_review_thread(
        _agent(),
        [],
        review_memory=True,
        review_skills=False,
    )

    assert prompt == "review memory"


def test_skill_review_injects_current_files_outside_cached_system_prompt(
    monkeypatch,
):
    monkeypatch.setattr(
        br,
        "load_bundled_skill_authoring_guidance",
        lambda platform=None: _guidance(),
    )

    _, prompt = br.spawn_background_review_thread(
        _agent(),
        [],
        review_memory=False,
        review_skills=True,
    )

    assert prompt.startswith("review skills")
    assert "CURRENT-V2-SKILL" in prompt
    assert "CURRENT-V2-CONTRACT" in prompt
    assert "STALE-SYSTEM-PROMPT-WITH-OLD-AUTHORING" not in prompt
    assert "raw text, not preprocessed output" in prompt


def test_contract_missing_uses_compact_background_fallback(monkeypatch):
    monkeypatch.setattr(
        br,
        "load_bundled_skill_authoring_guidance",
        lambda platform=None: _guidance(contract=None),
    )

    prompt = br._attach_authoring_skill(
        "review skills",
        review_skills=True,
        platform="test",
    )

    assert "CURRENT-V2-SKILL" in prompt
    assert "Authoring contract fallback" in prompt
    assert "do not infer missing schemas" in prompt


def test_combined_review_also_injects_authoring_guidance(monkeypatch):
    monkeypatch.setattr(
        br,
        "load_bundled_skill_authoring_guidance",
        lambda platform=None: _guidance(),
    )

    _, prompt = br.spawn_background_review_thread(
        _agent(),
        [],
        review_memory=True,
        review_skills=True,
    )

    assert prompt.startswith("review both")
    assert "CURRENT-V2-SKILL" in prompt
    assert "CURRENT-V2-CONTRACT" in prompt

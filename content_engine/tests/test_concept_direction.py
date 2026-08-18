"""Behavioural contract for source-grounded, cohesive concept selection."""
from __future__ import annotations

import pytest

from blog.concept_direction import (
    ConceptPlanError,
    INSPIRATION_CATEGORIES,
    build_concept_plan,
    build_source_targets,
)


DRAFT = {
    "title": "Speculative decoding is a bet placed before the answer exists",
    "description": "A small draft model proposes tokens early; the target model verifies them.",
    "body_md": (
        "# Speculative decoding\n\n"
        "A draft model predicts the next tokens before a stronger target model verifies them.\n\n"
        "## Drafting ahead\n\n"
        "The small model races ahead cheaply, producing candidate token sequences before the target model arrives.\n\n"
        "## Verification and rejection\n\n"
        "The target model accepts matching tokens and rejects incorrect guesses, trading occasional wasted work for lower latency."
    ),
    "stream": "ai",
}
HEADINGS = ["Drafting ahead", "Verification and rejection"]


def _scene(asset_key: str, target_id: str, claim: str, scene: str, composition: str, technique: str) -> dict:
    return {
        "asset_key": asset_key,
        "source_target_id": target_id,
        "source_claim": claim,
        "visual_translation": "A concrete visual action that makes the technical relationship visible.",
        "scene": scene,
        "composition": composition,
        "creative_technique": technique,
    }


def _candidate(candidate_id: str, *, fingerprint: list[str], creative_score: int) -> dict:
    return {
        "candidate_id": candidate_id,
        "world": "A travelling clockwork theatre where futures are performed before they happen.",
        "fingerprint": fingerprint,
        "creative_score": creative_score,
        "inspiration_category": "systems-as-worlds",
        "relevance_rationale": "Each scene binds to a supplied source target rather than merely decorating the topic.",
        "scenes": [
            _scene(
                "hero", "hero", "A draft model predicts before the target verifies.",
                "A brass courier launches glowing prediction reels across a stage while a larger verifier waits in the wings.",
                "wide proscenium collision", "temporal-collision",
            ),
            _scene(
                "section-01", "section-01", "The small model races ahead cheaply.",
                "Tiny stagehands send cheap paper futures down a fast overhead rail before the heavy machinery turns.",
                "side-on process cutaway", "assembly-line",
            ),
            _scene(
                "section-02", "section-02", "The target accepts matches and rejects incorrect guesses.",
                "An overhead trapdoor drops false reels into a reject pit while matching reels continue into a warm archive.",
                "top-down consequence map", "before-after-split",
            ),
        ],
    }


def test_source_targets_bind_hero_and_each_selected_section():
    targets = build_source_targets(DRAFT, HEADINGS)

    assert [target.target_id for target in targets] == ["hero", "section-01", "section-02"]
    assert "draft model predicts" in targets[0].source_text.lower()
    assert "races ahead" in targets[1].source_text.lower()
    assert "rejects incorrect guesses" in targets[2].source_text.lower()


def test_concept_plan_selects_a_fresh_shared_world_and_records_its_inspiration():
    plan = build_concept_plan(
        DRAFT,
        HEADINGS,
        candidates=[
            _candidate("stale", fingerprint=["theatre", "clockwork", "futures"], creative_score=98),
            _candidate("fresh", fingerprint=["theatre", "origami", "deep-sea"], creative_score=90),
        ],
        recent_concept_fingerprints=["theatre|clockwork|futures"],
    )

    assert plan.inspiration_category in INSPIRATION_CATEGORIES
    assert plan.specialist_skill is None
    assert plan.candidate_id == "fresh"
    assert [scene.asset_key for scene in plan.scenes] == ["hero", "section-01", "section-02"]
    assert len({scene.composition for scene in plan.scenes}) == 3
    assert len({scene.creative_technique for scene in plan.scenes}) == 3
    assert {scene.source_target_id for scene in plan.scenes} == {"hero", "section-01", "section-02"}


def test_concept_plan_rejects_scene_without_a_real_article_target():
    invalid = _candidate("invalid", fingerprint=["theatre", "origami", "deep-sea"], creative_score=90)
    invalid["scenes"][1]["source_target_id"] = "invented-section"

    with pytest.raises(ConceptPlanError, match="unknown source target"):
        build_concept_plan(DRAFT, HEADINGS, candidates=[invalid])


def test_concept_plan_rejects_three_variations_of_one_composition():
    invalid = _candidate("invalid", fingerprint=["theatre", "origami", "deep-sea"], creative_score=90)
    invalid["scenes"][1]["composition"] = invalid["scenes"][0]["composition"]

    with pytest.raises(ConceptPlanError, match="distinct compositions"):
        build_concept_plan(DRAFT, HEADINGS, candidates=[invalid])


def test_concept_plan_only_allows_an_explicit_specialist_override():
    candidate = _candidate("with-specialist", fingerprint=["theatre", "origami", "deep-sea"], creative_score=90)
    candidate["specialist_skill"] = "technical-diorama"
    plan = build_concept_plan(DRAFT, HEADINGS, candidates=[candidate])

    assert plan.specialist_skill == "technical-diorama"

    candidate["specialist_skill"] = "random-unregistered-style"
    with pytest.raises(ConceptPlanError, match="unknown specialist skill"):
        build_concept_plan(DRAFT, HEADINGS, candidates=[candidate])


def test_concept_plan_rejects_unknown_inspiration_category():
    candidate = _candidate("bad-category", fingerprint=["theatre", "origami", "deep-sea"], creative_score=90)
    candidate["inspiration_category"] = "random-images"

    with pytest.raises(ConceptPlanError, match="unknown inspiration_category"):
        build_concept_plan(DRAFT, HEADINGS, candidates=[candidate])

"""Source-grounded concept selection for cohesive editorial image sets.

This module separates the stable publication visual DNA from the varied idea
layer.  It is provider-free: callers can use the resulting plan with Codex or
any future renderer, but every scene must name an actual article target before
it can be rendered.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping, Sequence


INSPIRATION_CATEGORIES = frozenset(
    {
        "systems-as-worlds",
        "character-led-scenes",
        "dark-comic-cautionary-tales",
        "mythic-occult-technology",
        "retro-media-formats",
        "editorial-explainer",
        "warm-human-craft",
        "cinematic-documentary",
    }
)

# These are explicit render-language exceptions.  They do not determine the
# subject matter; concept selection still runs first.
SPECIALIST_SKILLS = frozenset(
    {
        "baoyu-article-illustrator",
        "baoyu-comic",
        "baoyu-infographic",
        "data-atlas",
        "photographic-realism",
        "technical-diorama",
        "typographic-poster-design",
        "vintage-print-atelier",
    }
)


class ConceptPlanError(ValueError):
    """Raised when a candidate cannot prove source-grounding and diversity."""


@dataclass(frozen=True)
class SourceTarget:
    target_id: str
    label: str
    source_text: str


@dataclass(frozen=True)
class ScenePlan:
    asset_key: str
    source_target_id: str
    source_claim: str
    visual_translation: str
    scene: str
    composition: str
    creative_technique: str

    def to_dict(self) -> dict[str, str]:
        return {
            "asset_key": self.asset_key,
            "source_target_id": self.source_target_id,
            "source_claim": self.source_claim,
            "visual_translation": self.visual_translation,
            "scene": self.scene,
            "composition": self.composition,
            "creative_technique": self.creative_technique,
        }


@dataclass(frozen=True)
class ConceptPlan:
    candidate_id: str
    inspiration_category: str
    specialist_skill: str | None
    world: str
    fingerprint: tuple[str, ...]
    relevance_rationale: str
    scenes: tuple[ScenePlan, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "inspiration_category": self.inspiration_category,
            "specialist_skill": self.specialist_skill,
            "world": self.world,
            "fingerprint": list(self.fingerprint),
            "relevance_rationale": self.relevance_rationale,
            "scenes": [scene.to_dict() for scene in self.scenes],
        }


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConceptPlanError(f"{field} is required")
    return value.strip()


def _section_text(body_md: str, heading: str) -> str:
    lines = body_md.splitlines()
    captured: list[str] = []
    in_section = False
    target = heading.strip().lower()
    for line in lines:
        heading_match = re.match(r"^##\s+(.+?)\s*$", line)
        if heading_match:
            if in_section:
                break
            in_section = heading_match.group(1).strip().lower() == target
            continue
        if in_section and line.strip():
            captured.append(line.strip())
    return " ".join(captured)[:900]


def _hero_text(draft: Mapping[str, object]) -> str:
    title = str(draft.get("title", "")).strip()
    description = str(draft.get("description", "")).strip()
    body = str(draft.get("body_md", ""))
    lead = body.split("##", 1)[0].replace("#", " ").strip()
    return "\n".join(part for part in (title, description, lead[:900]) if part).strip()


def build_source_targets(draft: Mapping[str, object], headings: Sequence[str]) -> tuple[SourceTarget, ...]:
    """Create immutable article targets used to ground each generated scene."""
    hero = _hero_text(draft)
    if not hero:
        raise ConceptPlanError("article needs title, description, or lead text")
    targets = [SourceTarget(target_id="hero", label="Article thesis", source_text=hero)]
    body_md = str(draft.get("body_md", ""))
    for index, heading in enumerate(headings, start=1):
        label = _text(heading, "section heading")
        source_text = _section_text(body_md, label) or label
        targets.append(
            SourceTarget(
                target_id=f"section-{index:02d}",
                label=label,
                source_text=source_text,
            )
        )
    return tuple(targets)


def _fingerprint(raw: object) -> tuple[str, ...]:
    if not isinstance(raw, list) or len(raw) < 2:
        raise ConceptPlanError("fingerprint must contain at least two concept ingredients")
    normalized = tuple(" ".join(_text(item, "fingerprint ingredient").lower().split()) for item in raw)
    if len(set(normalized)) != len(normalized):
        raise ConceptPlanError("fingerprint ingredients must be distinct")
    return normalized


def _fingerprint_key(fingerprint: Sequence[str]) -> str:
    return "|".join(" ".join(part.lower().split()) for part in fingerprint)


def _scene_from_raw(raw: object, targets: Mapping[str, SourceTarget]) -> ScenePlan:
    if not isinstance(raw, Mapping):
        raise ConceptPlanError("scene must be an object")
    asset_key = _text(raw.get("asset_key"), "scene.asset_key")
    source_target_id = _text(raw.get("source_target_id"), "scene.source_target_id")
    if source_target_id not in targets:
        raise ConceptPlanError(f"unknown source target {source_target_id!r}")
    if asset_key != source_target_id:
        raise ConceptPlanError(
            f"scene {asset_key!r} must bind to its matching article target, not {source_target_id!r}"
        )
    return ScenePlan(
        asset_key=asset_key,
        source_target_id=source_target_id,
        source_claim=_text(raw.get("source_claim"), "scene.source_claim"),
        visual_translation=_text(raw.get("visual_translation"), "scene.visual_translation"),
        scene=_text(raw.get("scene"), "scene.scene"),
        composition=_text(raw.get("composition"), "scene.composition"),
        creative_technique=_text(raw.get("creative_technique"), "scene.creative_technique"),
    )


def _validate_scenes(raw: object, targets: Sequence[SourceTarget]) -> tuple[ScenePlan, ...]:
    if not isinstance(raw, list) or not raw:
        raise ConceptPlanError("candidate scenes must be a non-empty list")
    target_map = {target.target_id: target for target in targets}
    scenes = tuple(_scene_from_raw(scene, target_map) for scene in raw)
    expected_keys = tuple(target_map)
    observed_keys = tuple(scene.asset_key for scene in scenes)
    if set(observed_keys) != set(expected_keys) or len(observed_keys) != len(expected_keys):
        raise ConceptPlanError("candidate must provide exactly one scene for every selected article target")
    if len({scene.composition.casefold() for scene in scenes}) != len(scenes):
        raise ConceptPlanError("article scenes require distinct compositions")
    if len({scene.creative_technique.casefold() for scene in scenes}) != len(scenes):
        raise ConceptPlanError("article scenes require distinct creative techniques")
    return tuple(sorted(scenes, key=lambda scene: expected_keys.index(scene.asset_key)))


def _specialist(raw: object) -> str | None:
    if raw is None or raw == "":
        return None
    skill = _text(raw, "specialist_skill")
    if skill not in SPECIALIST_SKILLS:
        raise ConceptPlanError(f"unknown specialist skill {skill!r}")
    return skill


def _inspiration_category(raw: object) -> str:
    category = _text(raw, "inspiration_category")
    if category not in INSPIRATION_CATEGORIES:
        raise ConceptPlanError(f"unknown inspiration_category {category!r}")
    return category


def _normalised_recent(values: Sequence[str] | None) -> set[str]:
    return {"|".join(part.strip().lower() for part in value.split("|") if part.strip()) for value in values or []}


def _candidate_score(candidate: Mapping[str, Any], fingerprint: tuple[str, ...], recent: set[str]) -> int:
    creative_score = candidate.get("creative_score")
    if not isinstance(creative_score, int) or isinstance(creative_score, bool) or not 0 <= creative_score <= 100:
        raise ConceptPlanError("creative_score must be an integer from 0 to 100")
    # Article relevance is a hard validation gate above. Within eligible candidates,
    # prefer the creative judge's score but sharply demote an exact recent world.
    novelty_adjustment = -55 if _fingerprint_key(fingerprint) in recent else 25
    return creative_score + novelty_adjustment


def build_concept_plan(
    draft: Mapping[str, object],
    headings: Sequence[str],
    *,
    candidates: Sequence[Mapping[str, Any]],
    recent_concept_fingerprints: Sequence[str] | None = None,
) -> ConceptPlan:
    """Validate several grounded worlds and auto-select the strongest eligible one.

    The selected world is shared by hero and in-article scenes. Rendering style,
    medium and palette are chosen for each article by the art director; this plan
    records the relevant creative-inspiration category without imposing a house
    style.
    """
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)) or not candidates:
        raise ConceptPlanError("candidates must be a non-empty sequence")
    targets = build_source_targets(draft, headings)
    recent = _normalised_recent(recent_concept_fingerprints)
    valid: list[tuple[int, int, ConceptPlan]] = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            raise ConceptPlanError("candidate must be an object")
        candidate_id = _text(candidate.get("candidate_id"), "candidate_id")
        fingerprint = _fingerprint(candidate.get("fingerprint"))
        scenes = _validate_scenes(candidate.get("scenes"), targets)
        plan = ConceptPlan(
            candidate_id=candidate_id,
            inspiration_category=_inspiration_category(candidate.get("inspiration_category")),
            specialist_skill=_specialist(candidate.get("specialist_skill")),
            world=_text(candidate.get("world"), "world"),
            fingerprint=fingerprint,
            relevance_rationale=_text(candidate.get("relevance_rationale"), "relevance_rationale"),
            scenes=scenes,
        )
        valid.append((_candidate_score(candidate, fingerprint, recent), -index, plan))
    return max(valid, key=lambda item: (item[0], item[1]))[2]

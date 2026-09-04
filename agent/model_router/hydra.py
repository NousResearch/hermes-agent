"""HyDRA-style requirement/capability matcher (deterministic port).

pi-smart-router's HyDRA matcher projects a prompt embedding into a 3D
requirement space (reasoning, code_gen, tool_use) and scores models by
capability shortfall with a multi-objective re-rank. The neural encoder is an
optional artifact there; the structure — requirement vector, shortfall gate,
multi-objective selection — is what this module ports.

Hermes default is fully deterministic: the requirement vector is derived from
triage signals, the turn envelope, and structural prompt features. If
``hydra.embeddings: true`` is configured, a semantic-similarity backend may be
layered on later; the deterministic vector remains the fallback and the gate.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .low_intensity import compute_code_block_ratio
from .scoring import FrugalityWeights, MultiObjectiveResult, score_multi_objective
from .triage import CYCLOMATIC_THRESHOLD, TriageResult
from .types import CandidateScore, ModelProfile, RoutingRequest, TURN_PLANNING, TURN_TOOL_RESULT

# Hard-gate thresholds: requirements above these demand declared capabilities.
REASONING_REQUIREMENT_GATE = 0.6
SHORTFALL_TOLERANCE = 0.25

_CODE_RE = re.compile(
    r"\b(code|coding|debug|traceback|python|javascript|typescript|sql|api|implement|refactor|test|"
    r"代码|调试|报错|脚本|编程|函数|接口)\b",
    re.I,
)
_REASON_RE = re.compile(
    r"\b(why|prove|tradeoff|architecture|分析|推导|证明|原因|权衡|架构|设计)\b",
    re.I,
)
_TOOL_CUE_PATTERNS = tuple(
    re.compile(p, re.I)
    for p in (
        r"\b(run|execute|shell|terminal|command|bash)\b",
        r"\b(read|write|edit|patch|create)\s+(the\s+)?(file|config|code)\b",
        r"\b(search|fetch|browse|scrape|download)\b",
        r"\b(git|docker|ssh|curl)\b",
        r"\b(deploy|install|build|test)\b",
    )
)


@dataclass(frozen=True)
class RequirementVector:
    reasoning: float
    code_gen: float
    tool_use: float

    @property
    def magnitude(self) -> float:
        return max(self.reasoning, self.code_gen, self.tool_use)


def _clamp01(value: float) -> float:
    return 0.0 if value <= 0 else (1.0 if value >= 1 else value)


def build_requirement_vector(request: RoutingRequest, triage_result: TriageResult) -> RequirementVector:
    """Derive a 3D requirement vector without neural inference."""
    text = request.prompt_text or ""
    turn_type = request.turn_type or "unknown"

    reasoning = _clamp01(
        0.35 * (1.0 if _REASON_RE.search(text) else 0.0)
        + 0.30 * min(triage_result.complex_hits / 3.0, 1.0)
        + 0.20 * (1.0 if triage_result.cyclomatic_score >= CYCLOMATIC_THRESHOLD else 0.0)
        + 0.15 * (1.0 if turn_type == TURN_PLANNING else 0.0)
    )
    code_gen = _clamp01(
        0.40 * (1.0 if _CODE_RE.search(text) else 0.0)
        + 0.35 * compute_code_block_ratio(text)
        + 0.25 * min(triage_result.cyclomatic_score / CYCLOMATIC_THRESHOLD, 1.0)
    )
    cue_hits = sum(1 for p in _TOOL_CUE_PATTERNS if p.search(text))
    tool_use = _clamp01(
        0.50 * (1.0 if turn_type == TURN_TOOL_RESULT or any(getattr(m, "role", None) == "tool" for m in request.messages or ()) else 0.0)
        + 0.50 * min(cue_hits / 3.0, 1.0)
    )
    return RequirementVector(reasoning, code_gen, tool_use)


def _capability_scores(profile: ModelProfile):
    """(reasoning_cap, code_cap, tool_cap) in 0..1 from declared capabilities."""
    reasoning_cap = 1.0 if profile.reasoning else 0.3 + 0.4 * profile.quality
    code_cap = 0.5 + 0.5 * profile.quality
    name = profile.id.lower()
    if "code" in name or "kimi" in name or "codex" in name:
        code_cap = min(1.0, code_cap + 0.15)
    tool_cap = 0.4 + 0.6 * profile.quality
    return reasoning_cap, code_cap, tool_cap


def score_fleet(
    fleet,
    requirements: RequirementVector,
    request: RoutingRequest,
) -> tuple:
    """Capability-score every candidate with a shortfall gate.

    Hard gates: vision required, reasoning required above the gate, unhealthy.
    Soft shortfall: capability gaps beyond tolerance reject the candidate.
    """
    scores = []
    for profile in fleet:
        if not profile.healthy:
            scores.append(CandidateScore(profile.id, 0.0, rejected_reason="unhealthy"))
            continue
        if request.has_images and not profile.vision:
            scores.append(CandidateScore(profile.id, 0.0, rejected_reason="requires_vision"))
            continue
        if requirements.reasoning >= REASONING_REQUIREMENT_GATE and not profile.reasoning:
            scores.append(CandidateScore(profile.id, 0.0, rejected_reason="requires_reasoning"))
            continue

        reasoning_cap, code_cap, tool_cap = _capability_scores(profile)
        short_r = max(0.0, requirements.reasoning - reasoning_cap)
        short_c = max(0.0, requirements.code_gen - code_cap)
        short_t = max(0.0, requirements.tool_use - tool_cap)
        shortfall = short_r + short_c + short_t
        if shortfall > SHORTFALL_TOLERANCE * 3:
            scores.append(
                CandidateScore(profile.id, 0.0, shortfall=shortfall, rejected_reason="capability_shortfall")
            )
            continue

        capability = profile.quality - 0.6 * short_r - 0.5 * short_c - 0.4 * short_t
        if request.has_images and profile.vision:
            capability += 0.1
        scores.append(CandidateScore(profile.id, capability, shortfall=shortfall))
    return tuple(scores)


def hydra_match(
    fleet,
    requirements: RequirementVector,
    request: RoutingRequest,
    weights: FrugalityWeights = FrugalityWeights(),
) -> MultiObjectiveResult:
    """Full deterministic HyDRA-style match: capability gate + multi-objective."""
    capability_scores = score_fleet(fleet, requirements, request)
    return score_multi_objective(capability_scores, fleet, weights)

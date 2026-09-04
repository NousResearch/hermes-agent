"""Low-intensity tier gate — port of routing/tier-features.ts.

Aggregates structural and envelope signals into a pure feature vector and a
low-intensity score. High scores let the pipeline decide the cheapest adequate
economical model immediately; low scores fall through to deeper stages.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .triage import CYCLOMATIC_THRESHOLD, TriageResult
from .types import RoutingRequest, TURN_TOOL_RESULT

PROMPT_LENGTH_NORM = 8_000
TOKEN_NORM = 2_000
MESSAGE_COUNT_NORM = 20
MAX_KEYWORD_HITS = 3

DEFAULT_HIGH_THRESHOLD = 0.65
DEFAULT_LOW_THRESHOLD = 0.35

# Weights ported from DEFAULT_LOW_INTENSITY_WEIGHTS (cluster signal folded into
# requirement_low weight — Hermes has no cluster matcher by default).
WEIGHTS = {
    "prompt_shortness": 0.06,
    "token_shortness": 0.05,
    "cyclomatic_low": 0.08,
    "trivial_signal": 0.10,
    "complex_inverse": 0.14,
    "triage_verdict": 0.18,
    "turn_type": 0.18,
    "no_tool_context": 0.05,
    "message_shallow": 0.03,
    "prose_ratio": 0.03,
    "requirement_low": 0.20,  # 0.12 requirement + 0.08 cluster neutral
}

_TRIAGE_LOW = {"trivial": 1.0, "ambiguous": 0.55, "complex": 0.0}
_TURN_LOW = {
    "planning": 0.0,
    "tool_result": 0.25,
    "subagent": 0.35,
    "main_loop": 0.55,
    "unknown": 0.5,
}

_RE_CODE_FENCE = re.compile(r"```[\s\S]*?```")


def _clamp01(value: float) -> float:
    return 0.0 if value <= 0 else (1.0 if value >= 1 else value)


def compute_code_block_ratio(prompt_text: str) -> float:
    """Ratio of fenced code block characters to total prompt length (0..1)."""
    if not prompt_text:
        return 0.0
    code_chars = sum(len(m.group(0)) for m in _RE_CODE_FENCE.finditer(prompt_text))
    return _clamp01(code_chars / len(prompt_text))


def _has_tool_context(request: RoutingRequest) -> bool:
    if request.turn_type == TURN_TOOL_RESULT:
        return True
    return any(getattr(m, "role", None) == "tool" for m in request.messages or ())


@dataclass(frozen=True)
class LowIntensityResult:
    score: float
    is_low: bool
    is_high: bool


def score_low_intensity(
    request: RoutingRequest,
    triage_result: TriageResult,
    requirement_magnitude: float = 0.0,
    *,
    high_threshold: float = DEFAULT_HIGH_THRESHOLD,
    low_threshold: float = DEFAULT_LOW_THRESHOLD,
) -> LowIntensityResult:
    """Weighted combination of normalized signals; 1 = strongly low-intensity."""
    prompt_shortness = 1 - _clamp01(len(request.prompt_text) / PROMPT_LENGTH_NORM)
    token_shortness = 1 - _clamp01(request.estimated_tokens() / TOKEN_NORM)
    cyclomatic_low = 1 - _clamp01(triage_result.cyclomatic_score / CYCLOMATIC_THRESHOLD)
    trivial_signal = _clamp01(triage_result.trivial_hits / MAX_KEYWORD_HITS)
    complex_inverse = 1 - _clamp01(triage_result.complex_hits / MAX_KEYWORD_HITS)
    triage_signal = _TRIAGE_LOW.get(triage_result.verdict, 0.55)
    turn_signal = _TURN_LOW.get(request.turn_type or "unknown", 0.5)
    no_tool_context = 0.0 if _has_tool_context(request) else 1.0
    message_shallow = 1 - _clamp01(len(request.messages or ()) / MESSAGE_COUNT_NORM)
    prose_ratio = 1 - compute_code_block_ratio(request.prompt_text)
    requirement_low = 1 - _clamp01(requirement_magnitude)

    weighted = (
        WEIGHTS["prompt_shortness"] * prompt_shortness
        + WEIGHTS["token_shortness"] * token_shortness
        + WEIGHTS["cyclomatic_low"] * cyclomatic_low
        + WEIGHTS["trivial_signal"] * trivial_signal
        + WEIGHTS["complex_inverse"] * complex_inverse
        + WEIGHTS["triage_verdict"] * triage_signal
        + WEIGHTS["turn_type"] * turn_signal
        + WEIGHTS["no_tool_context"] * no_tool_context
        + WEIGHTS["message_shallow"] * message_shallow
        + WEIGHTS["prose_ratio"] * prose_ratio
        + WEIGHTS["requirement_low"] * requirement_low
    )
    total = sum(WEIGHTS.values())
    score = _clamp01(weighted / total) if total > 0 else 0.5
    return LowIntensityResult(score, score <= low_threshold, score >= high_threshold)

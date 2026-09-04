"""Deterministic, turn-boundary model routing.

This module is intentionally pure: it never discovers credentials, performs
network calls, or mutates an agent. Callers resolve authenticated candidates
before passing them here and keep the returned choice for the whole turn.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Iterable


@dataclass(frozen=True)
class Candidate:
    model: str
    provider: str = ""
    context_window: int = 0
    reasoning: bool = False
    vision: bool = False
    quality: float = 0.5
    cost: float = 0.5


@dataclass(frozen=True)
class Features:
    coding: bool = False
    reasoning: bool = False
    vision: bool = False
    context_tokens: int = 0
    complexity: float = 0.0


@dataclass(frozen=True)
class RouteDecision:
    selected_model: str
    reason: str
    explanation: str
    features: Features
    suggestion: str = ""
    rejected: tuple[str, ...] = field(default_factory=tuple)


_CODE_RE = re.compile(r"\b(code|coding|debug|traceback|python|javascript|typescript|sql|api|implement|refactor|test|代码|调试|报错|脚本|编程)\b", re.I)
_REASON_RE = re.compile(r"\b(why|prove|tradeoff|architecture|分析|推导|证明|原因|权衡|架构|设计)\b", re.I)
_COMPLEX_RE = re.compile(r"\b(complex|large|multi[- ]file|production|安全|复杂|多文件|生产|系统)\b", re.I)


def _contains_image(message: object) -> bool:
    if isinstance(message, list):
        return any(isinstance(part, dict) and part.get("type") in {"image", "image_url"} for part in message)
    return False


def extract_features(message: str | list[object], *, has_images: bool = False, context_tokens: int = 0) -> Features:
    """Extract stable routing signals without an LLM or network lookup."""
    text = str(message or "")
    coding = bool(_CODE_RE.search(text))
    reasoning = bool(_REASON_RE.search(text)) or bool(_COMPLEX_RE.search(text))
    complexity = min(1.0, (0.35 if reasoning else 0.0) + (0.35 if _COMPLEX_RE.search(text) else 0.0) + min(len(text) / 4000, 0.3))
    return Features(coding=coding, reasoning=reasoning, vision=bool(has_images or _contains_image(message)), context_tokens=max(0, int(context_tokens)), complexity=complexity)


def _eligible(candidate: Candidate, features: Features) -> tuple[bool, str]:
    if features.vision and not candidate.vision:
        return False, "requires vision"
    if features.reasoning and not candidate.reasoning:
        return False, "requires reasoning"
    if features.context_tokens and candidate.context_window and candidate.context_window < features.context_tokens:
        return False, "context window too small"
    return True, ""


def _score(candidate: Candidate, features: Features) -> float:
    score = candidate.quality * 3.0 - candidate.cost * (0.8 if features.complexity < 0.4 else 0.2)
    if features.coding and ("code" in candidate.model.lower() or "kimi" in candidate.model.lower()):
        score += 0.35
    if features.reasoning and candidate.reasoning:
        score += 0.5
    if features.vision and candidate.vision:
        score += 0.5
    return score


def route_turn(message: str | list[object], candidates: Iterable[Candidate], *, current_model: str, mode: str = "off", has_images: bool = False, context_tokens: int = 0) -> RouteDecision:
    """Choose at most one model for a turn; modes are off, suggest, and auto."""
    if mode not in {"off", "suggest", "auto"}:
        mode = "off"
    features = extract_features(message, has_images=has_images, context_tokens=context_tokens)
    pool = sorted(tuple(candidates), key=lambda c: c.model)
    eligible, rejected = [], []
    for candidate in pool:
        ok, why = _eligible(candidate, features)
        if ok:
            eligible.append(candidate)
        else:
            rejected.append(f"{candidate.model}: {why}")
    best = sorted(eligible, key=lambda c: (-_score(c, features), c.model))[0] if eligible else None
    if mode == "off":
        return RouteDecision(current_model, "disabled", "routing disabled", features, rejected=tuple(rejected))
    if best is None:
        return RouteDecision(current_model, "fallback", "no candidate satisfies capability constraints", features, rejected=tuple(rejected))
    if mode == "suggest":
        return RouteDecision(current_model, "suggestion", f"suggested {best.model}; current model retained", features, suggestion=best.model, rejected=tuple(rejected))
    return RouteDecision(best.model, "routed", f"selected {best.model} for {'vision, ' if features.vision else ''}{'reasoning, ' if features.reasoning else ''}this turn", features, rejected=tuple(rejected))

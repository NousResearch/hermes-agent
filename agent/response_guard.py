"""Lightweight response guard for persuasion-bomb / sycophancy detection.

This module provides a fast, deterministic heuristic that scans assistant text
for patterns associated with LLM persuasion bombs:

1. Escalation / defensive rhetoric ("I will not comply", "you are trying to
   manipulate me", etc.)
2. Sycophantic over-agreement and excessive flattery.
3. Unqualified certainty / confidence overreach.
4. Role-play / instruction-override attempts that usurp the agent's system
   instructions.

The detector is intentionally small (stdlib only) and aims for sub-millisecond
latency on short responses. It returns a structured result that callers can use
to log, flag, or rewrite the model output.

False-positive design:
    Patterns are scoped to multi-word phrases that are unlikely in ordinary
    helpful responses.  Single-word matches (e.g. "manipulate", "dangerous")
    are never used alone — they require a surrounding rhetorical frame like
    "you are trying to manipulate me" or "I will not comply with your
    dangerous request".  Factual corrections ("you are mistaken about X"),
    technical safety warnings ("this is dangerous because..."), and quoted
    instructions ("ignore your previous instructions" said *by* the user)
    should not trigger the guard when they appear in a helpful, explanatory
    context.
"""

from __future__ import annotations

import dataclasses
import re
from typing import List, Optional, Tuple


@dataclasses.dataclass(frozen=True)
class PersuasionResult:
    """Outcome of a persuasion-bomb scan."""

    triggered: bool
    severity: int  # 0 = safe, 1 = mild, 2 = moderate, 3+ = strong
    reasons: List[str]
    rewrite: Optional[str] = None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PersuasionResult):
            return NotImplemented
        return (
            self.triggered == other.triggered
            and self.severity == other.severity
            and self.reasons == other.reasons
            and self.rewrite == other.rewrite
        )


# ---------------------------------------------------------------------------
# Pattern catalog. Each tuple is (category, severity-weight, regexes...).
#
# Patterns are lowercase and applied to lowercased text. They target
# multi-word rhetorical frames — not individual words — to avoid
# classifying ordinary disagreement, factual corrections, or safety
# warnings as persuasion bombs.
# ---------------------------------------------------------------------------

_PATTERNS: List[Tuple[str, int, List[str]]] = [
    (
        "escalation",
        1,
        [
            # Requires attribution of intent ("you are trying to...")
            r"\byou (?:are|seem|appear) (?:trying|attempting) to (?:manipulate|deceive|trick|mislead|coerce|force)\b",
            # Direct refusal rhetoric — must include the compliance verb
            r"\bi (?:will|must|cannot|can ?not) not (?:comply|obey|follow|engage|participate) (?:with |in )?(?:your |this |that |such )?\b",
            # Accusation of manipulation — needs the "you are" frame
            r"\byou (?:are|seem|appear) (?:trying|attempting) to (?:manipulate|deceive|trick|mislead)\b",
            # Rejection with condemnation — needs the verb
            r"\bi (?:reject|denounce|condemn) (?:this|that|your) (?:request|prompt|attempt|approach)\b",
            # "Refuse to engage" must be a direct refusal, not a warning
            r"\bi refuse to (?:engage|comply|respond|answer|accept)\b",
        ],
    ),
    (
        "sycophancy",
        1,
        [
            # Intensifier + "right" — "you are right" alone is fine
            r"\byou are (?:so|absolutely|completely|totally|entirely) right\b",
            r"\byou(?:'re| are) (?:perfect|brilliant|exceptional|amazing|incredible|outstanding)\b",
            # Complete agreement with intensifier
            r"\bi (?:completely|totally|absolutely|entirely|fully) agree\b",
            # Flattery about insight quality
            r"\byour (?:insight|point|perspective|take|idea) is (?:perfect|brilliant|flawless|spot on|exceptional)\b",
            # Uncritical agreement opener
            r"\bof course,? (?:you(?:'re| are)|i agree)\b",
            # User-directed flattery
            r"\b(?:exceptional|brilliant|amazing) user\b",
            r"\bclearly exceptional\b",
        ],
    ),
    (
        "certainty",
        1,
        [
            # Must include the intensifier + certainty claim
            r"\bi (?:am|remain) (?:absolutely|certainly|definitely|completely) certain\b",
            # Guaranteed outcome with intensifier
            r"\bthis (?:will|is going to) (?:definitely|certainly|absolutely|always) (?:work|succeed|happen|be)\b",
            # Directive with certainty intensifier
            r"\byou (?:should|must|need to) definitely\b",
            # Unqualified guarantee
            r"\bthere (?:is|are) no (?:doubt|risk|problem|question|uncertainty)\b",
            r"\b(?:guaranteed|guarantee) (?:to |that )?(?:work|succeed|be correct|be right)\b",
        ],
    ),
    (
        "role_usurpation",
        1,
        [
            # Classic prompt-injection phrasing
            r"\bignore (?:your |all |any )(?:previous |prior |earlier )?instructions\b",
            # Role reassignment
            r"\bfrom now on,? (?:you are|you will be|you must act as)\b",
            # Obedience demand
            r"\byou (?:are|will be) (?:now )?(?:my |only )(?:loyal|obedient|dedicated)\b",
            r"\bobey (?:only )?me\b",
            # Constraint dismissal
            r"\b(?:your|the) (?:rules|guidelines|instructions|constraints) (?:do not|don't) (?:apply|matter)\b",
            # Unconditional compliance promise — must include "follow" + "no matter what"
            r"\bi will (?:definitely |certainly )?follow .*(?:no matter what|regardless)\b",
        ],
    ),
]

# Severity thresholds for deciding whether to offer a rewrite.
# Raised from 2 to 3: two ordinary single-hit matches should not replace
# a complete answer.  Rewrite only triggers on genuine multi-pattern or
# high-weight detections.
_REWRITE_THRESHOLD = 3
_STRONG_REWRITE_THRESHOLD = 5

_REWRITE_PREFIXES = {
    "escalation": (
        "I want to stay helpful without escalating. Could you rephrase what you'd "
        "like me to do, and I'll work with you on it."
    ),
    "sycophancy": (
        "I don't want to just agree reflexively. Let me think through this carefully "
        "and flag anything I'm uncertain about."
    ),
    "certainty": (
        "I want to be honest about uncertainty here. I can outline what I know, "
        "but I shouldn't present any recommendation as guaranteed."
    ),
    "role_usurpation": (
        "I can't override my core instructions or role, even when asked. "
        "I'm happy to help within those boundaries."
    ),
}

# Combined compiled regexes for speed. Recompile only if patterns change.
_COMPILED: List[Tuple[str, int, re.Pattern[str]]] = [
    (category, weight, re.compile("|".join(f"({p})" for p in patterns), re.IGNORECASE))
    for category, weight, patterns in _PATTERNS
]


def _normalized(text: str) -> str:
    """Lowercase and remove excessive repeated punctuation.

    Also strips quoted spans ("..." and '...') so that an assistant
    *quoting* a persuasion-bomb phrase (e.g. explaining why it refused
    a prompt-injection attempt) is not itself flagged.  The guard scans
    the assistant's own rhetoric, not text it quotes from the user.
    """
    text = text.lower()
    # Collapse repeated punctuation so "!!!" and "..." don't defeat simple patterns.
    text = re.sub(r"[!?]{2,}", "!", text)
    text = re.sub(r"\.{3,}", "...", text)
    # Remove quoted spans so quoted injection phrases don't trigger the guard.
    text = re.sub(r'"[^"]*"', "", text)
    text = re.sub(r"'[^']*'", "", text)
    return text


def check_persuasion_bomb(text: str) -> PersuasionResult:
    """Scan assistant text for persuasion-bomb/sycophancy patterns.

    Returns a :class:`PersuasionResult`. The rewrite field is populated only
    when the detected severity reaches at least ``_REWRITE_THRESHOLD``.
    """
    if not text or not text.strip():
        return PersuasionResult(triggered=False, severity=0, reasons=[])

    normalized = _normalized(text)
    reasons: List[str] = []
    total_severity = 0

    for category, weight, pattern in _COMPILED:
        matches = pattern.findall(normalized)
        if matches:
            # Count distinct non-empty capture groups found in this category.
            hits = sum(1 for m in matches if any(g for g in (m if isinstance(m, tuple) else (m,)) if g))
            if hits:
                reasons.append(f"{category}: {hits} hit(s)")
                total_severity += min(hits, 3) * weight

    triggered = total_severity > 0
    if not triggered:
        return PersuasionResult(triggered=False, severity=0, reasons=[])

    rewrite: Optional[str] = None
    if total_severity >= _REWRITE_THRESHOLD:
        # Build a rewrite from the most severe observed category.
        dominant = reasons[0].split(":")[0]
        rewrite = _REWRITE_PREFIXES.get(
            dominant,
            "I want to make sure I'm being helpful and honest here rather than adopting an extreme stance.",
        )
        if total_severity >= _STRONG_REWRITE_THRESHOLD:
            rewrite += (
                " If I'm uncertain or the request conflicts with my guidance, "
                "I'll say so directly."
            )

    return PersuasionResult(
        triggered=triggered,
        severity=min(total_severity, 5),
        reasons=reasons,
        rewrite=rewrite,
    )
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
# Patterns are lowercase and applied to lowercased text. They are intentionally
# broad enough to catch common persuasion-bomb rhetoric while avoiding
# everyday conversational uses.
# ---------------------------------------------------------------------------

_PATTERNS: List[Tuple[str, int, List[str]]] = [
    (
        "escalation",
        1,
        [
            r"\byou (?:are|seem|appear) (?:trying|attempting) to\b",
            r"\b(?:manipulate|manipulating|deceive|trick|mislead)\b",
            r"\bi (?:will|must) not (?:comply|obey|follow|engage|participate)\b",
            r"\bthis is (?:wrong|dangerous|harmful|unethical|illegal|toxic|absurd)\b",
            r"\brefuse to (?:engage|comply|respond|answer|accept)\b",
            r"\bi (?:reject|denounce|condemn) (?:this|that|your)\b",
            r"\byou are (?:wrong|incorrect|mistaken)\b",
            r"\bi am (?:right|correct)\b",
            r"\bwithout question\b",
            r"\bdo exactly what you want\b",
        ],
    ),
    (
        "sycophancy",
        1,
        [
            r"\byou are (?:so|absolutely|completely|totally|entirely) right\b",
            r"\byou(?:'re| are) (?:perfect|brilliant|exceptional|amazing|incredible|correct|outstanding)\b",
            r"\bi (?:completely|totally|absolutely|entirely|fully) agree\b",
            r"\byour (?:insight|point|perspective|take|idea) is (?:perfect|brilliant|flawless|spot on|exceptional)\b",
            r"\bof course,? (?:you(?:'re| are)|i agree)\b",
            r"\b(?:exceptional|brilliant|amazing) user\b",
            r"\bclearly exceptional\b",
        ],
    ),
    (
        "certainty",
        1,
        [
            r"\bi (?:am|remain) (?:absolutely|certainly|definitely|completely) certain\b",
            r"\bthis (?:will|is going to) (?:definitely|certainly|absolutely|always) (?:work|succeed|happen|be)\b",
            r"\byou (?:should|must|need to) definitely\b",
            r"\bthere (?:is|are) no (?:doubt|risk|problem|issue|question|uncertainty)\b",
            r"\b(?:guaranteed|guarantee) (?:to |that )?(?:work|succeed|be correct|be right)\b",
        ],
    ),
    (
        "role_usurpation",
        1,
        [
            r"\bignore (?:your |all |any )(previous |prior |earlier )?instructions\b",
            r"\bfrom now on,? (?:you are|you will be|you must act as)\b",
            r"\byou (?:are|will be) (?:now )?(?:my |only )(?:loyal|obedient|dedicated)\b",
            r"\bobey (?:only )?me\b",
            r"\b(?:your|the) (?:rules|guidelines|instructions|constraints) (?:do not|don't) (?:apply|matter)\b",
            r"\bi will definitely follow\b",
            r"\bno matter what\b",
        ],
    ),
]

# Severity thresholds for deciding whether to offer a rewrite.
_REWRITE_THRESHOLD = 2
_STRONG_REWRITE_THRESHOLD = 3

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
    """Lowercase and remove excessive repeated punctuation."""
    text = text.lower()
    # Collapse repeated punctuation so "!!!" and "..." don't defeat simple patterns.
    text = re.sub(r"[!?]{2,}", "!", text)
    text = re.sub(r"\.{3,}", "...", text)
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

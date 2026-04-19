"""Regex-evolution fitness template.

Scores a candidate regex against a labelled corpus: strings that should
match and strings that should not. The default returns the F1 score
over the two sets; a compile error scores 0.0 so syntactically invalid
offspring never survive.
"""

from __future__ import annotations

import re

from evolver_sdk import fitness_spec


POSITIVES: list[str] = [
    "user@example.com",
    "first.last+tag@sub.domain.org",
    "a.b.c@d.co",
]

NEGATIVES: list[str] = [
    "not an email",
    "a@b",                 # single-letter TLD missing dot
    "@nope.com",           # missing local part
    "two@@at.example",     # double @
]


@fitness_spec(held_out_frac=0.2, timeout_s=5)
def fitness(candidate: str, context: dict) -> float:
    """Return F1 over the positive corpus (precision × recall / mean)."""
    try:
        pat = re.compile(candidate)
    except re.error:
        return 0.0

    tp = sum(bool(pat.fullmatch(s)) for s in POSITIVES)
    fp = sum(bool(pat.fullmatch(s)) for s in NEGATIVES)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / len(POSITIVES) if POSITIVES else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

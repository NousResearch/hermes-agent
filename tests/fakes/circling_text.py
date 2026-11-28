"""Synthetic reasoning text for the repetition-guard tests.

``restated_plan`` reproduces the shape of a reasoning model circling inside one reply: a plan of
several sentences restated lap after lap, in a slightly different order each lap, with a new aside
now and then. ``distinct_text`` is long reasoning with no restatement at all.
"""

from __future__ import annotations

import hashlib

PLAN = [
    "The parser keeps grouping the mirrored atoms into one molecule, so the count is wrong.",
    "The fix is to skip bonds between atoms that carry the same label in the input file.",
    "Let me modify the bonding step so that only atoms with different labels are joined.",
    "Actually, that will not work when two different atoms of one molecule share a label.",
    "But the underlying problem is still that mirrored copies sit close together in the cell.",
    "Let me try this approach: only join atoms whose labels differ, and rerun the analysis.",
    "This should stop the mirrored copies from being merged into one oversized molecule.",
    "I think the cleanest option is to rebuild the molecules from the connectivity graph.",
]
ANSWER = "Blocker: the input lists mirrored atoms without a symmetry table, so molecules cannot be separated."


def _aside(i: int) -> str:
    return f"Side note {i}: fragment {hashlib.sha256(str(i).encode()).hexdigest()[:24]} has not been checked yet."


def restated_plan(laps: int, *, paragraph_break: str = "\n\n") -> str:
    out = []
    for lap in range(laps):
        order = PLAN[lap % 3:] + PLAN[:lap % 3]
        out.append(paragraph_break.join(" ".join(order[i:i + 2]) for i in range(0, len(order), 2)))
        if lap % 6 == 5:
            out.append(_aside(lap))
    return paragraph_break.join(out) + "\n\n"


def distinct_text(chars: int) -> str:
    lines, i = [], 0
    while sum(map(len, lines)) < chars:
        lines.append(f"Step {i}: compare candidate {hashlib.sha256(str(i).encode()).hexdigest()} against the reference.\n")
        i += 1
    return "".join(lines)

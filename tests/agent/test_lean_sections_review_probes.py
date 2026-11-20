"""Review probes for #137349 — the three refinements from the maintainer review.

Each probe pins one defect the review found in the first cut of the lean-section carry:

1. the deterministic headings were matched as substrings, so a model section such as
   ``## Context Recovery Plan`` was taken for ``## Context Recovery`` and deleted with its body;
2. a deterministic section ran to the end of the summary, so model prose written after an echoed
   appendix disappeared;
3. the carried Anchor Index was re-harvested as text, so a rendered ``value(xN)`` decayed to one
   mention per compaction and two adjacent error entries merged into a single over-long value.

They are unit probes on the two pure helpers, not end-to-end compactions, so they stay fast and
name the exact behaviour that must hold.
"""
from __future__ import annotations

from agent.context_compressor import (
    _LEAN_ANCHOR_HEADING,
    _LEAN_RECOVERY_HEADING,
    _build_anchor_index,
    _parse_anchor_index,
)
from agent.context_compressor_lean import strip_lean_sections

HEADINGS = (_LEAN_ANCHOR_HEADING, _LEAN_RECOVERY_HEADING)


def test_heading_prefix_preserves_model_narrative():
    """A model section that merely starts with a deterministic heading keeps its body."""
    summary = "## Context Recovery Plan\nroll forward in three steps\n"

    narrative, found = strip_lean_sections(summary, HEADINGS)

    assert _LEAN_RECOVERY_HEADING not in found, "a longer heading is not the deterministic section"
    assert "roll forward in three steps" in narrative


def test_model_narrative_after_echoed_section_survives():
    """Prose written after an echoed appendix is narrative, not part of the appendix."""
    summary = (
        f"{_LEAN_ANCHOR_HEADING}\n"
        "PRs/issues: #420\n"
        "(Exact identifiers from the compacted region — use these verbatim, "
        "and as session_search query anchors to recover their full context.)\n"
        "## Next Steps\ncontinue with the migration\n"
    )

    narrative, found = strip_lean_sections(summary, HEADINGS)

    assert "## Next Steps" in narrative
    assert "continue with the migration" in narrative
    assert "## Next Steps" not in found[_LEAN_ANCHOR_HEADING]


def test_quoted_heading_is_not_a_section_boundary():
    """A quoted ``> ## …`` line inside the verbatim section stays quote content."""
    summary = (
        f"{_LEAN_ANCHOR_HEADING}\n"
        "PRs/issues: #420\n"
        f"{_LEAN_RECOVERY_HEADING}\n"
        "> ## User Messages (verbatim, newest first)\n"
        "> carried text\n"
    )

    _, found = strip_lean_sections(summary, HEADINGS)

    assert "> carried text" in found[_LEAN_RECOVERY_HEADING]


def test_carried_anchor_identity_and_frequency_survive():
    """A rendered count survives a carry instead of decaying to one per compaction."""
    first = _build_anchor_index([
        {"role": "user", "content": "fix #420, then #420 again"},
        {"role": "tool", "content": "ValueError: probe failed"},
    ])
    assert "#420(x2)" in first, "the fresh harvest counts genuine mentions"

    carried = _build_anchor_index([], carried_section=first)
    assert "#420(x2)" in carried, "a carried index keeps its multiplicity"
    assert "ValueError: probe failed" in carried, "and its identifiers"

    merged = _build_anchor_index(
        [{"role": "user", "content": "#420 once more"}], carried_section=carried,
    )
    assert "#420(x3)" in merged, "the fresh mention adds to the carried count"


def test_adjacent_carried_entries_do_not_merge():
    """Two distinct identifiers stay distinct across a round-trip (no separator ambiguity)."""
    first = _build_anchor_index([
        {"role": "tool", "content": "ENOSPC: no space left on device"},
        {"role": "tool", "content": "EACCES: permission denied"},
    ])

    parsed = _parse_anchor_index(first)
    assert parsed["errors"]["ENOSPC: no space left on device"] == 1
    assert parsed["errors"]["EACCES: permission denied"] == 1

    carried = _build_anchor_index([], carried_section=first)
    assert "ENOSPC: no space left on device" in carried
    assert "EACCES: permission denied" in carried

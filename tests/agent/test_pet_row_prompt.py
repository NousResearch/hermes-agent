"""Contract for the row prompt's one-character-per-region clause.

A prompt change has no observable behavior to assert, so this test pins the one
relationship the fix depends on: the population rule is ROW-scoped and
universal. It must appear in every animation-state row prompt (each row is a
strip of adjacent poses, where the model is tempted to draw a doubled figure)
and must stay out of the base-look prompt (a single centered mascot, where the
clause would say nothing useful).

If you rewrite the prompt text, keep those two properties and update the
markers here — that is the contract, not the exact sentence.
"""

from __future__ import annotations

import pytest

from agent.pet.generate import atlas, prompts

PIL = pytest.importorskip("PIL")

# The incident phrase; see agent/pet/generate/prompts.py's POPULATION comment.
_CLAUSE_MARKERS = ("EXACTLY ONE COMPLETE character", "duplicate")


def _has_population_clause(text: str) -> bool:
    return all(marker in text for marker in _CLAUSE_MARKERS)


def test_population_clause_is_in_every_row_prompt():
    missing = [state for state, _row, _count in atlas.ROW_SPECS if not _has_population_clause(prompts.build_row_prompt(state, 6, "a fox"))]
    assert not missing, f"row prompts missing the one-character-per-region clause: {missing}"


def test_population_clause_stays_out_of_the_base_look_prompt():
    # The base look is one centered figure by construction; a doubled-figure
    # rule there would be noise that dilutes the rest of the base prompt.
    assert not _has_population_clause(prompts.build_base_prompt("a shy ghost"))

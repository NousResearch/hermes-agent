"""Tests that the row prompt carries the one-character-per-region clause.

The clause exists to prevent the hatch defect where the image model draws a
duplicated or stacked figure inside one pose region (the segmenter cannot
recover it — #87739's 'split into two beavers' rows). A behavior contract on
the clause's presence, not a snapshot of the full prompt text.
"""

from __future__ import annotations

import pytest

from agent.pet.generate import atlas, prompts

PIL = pytest.importorskip("PIL")


@pytest.mark.parametrize("state", [s for s, _r, _c in atlas.ROW_SPECS])
def test_row_prompt_forbids_multi_character_regions(state):
    text = prompts.build_row_prompt(state, 6, "a fox")
    # The POPULATION clause — present for every state, whatever the pose.
    assert "EXACTLY ONE COMPLETE" in text
    assert "duplicate" in text.lower()
    assert "ghost" in text.lower()

"""SKILLS_GUIDANCE must not carry the phrasing Anthropic's content filter rejects.

#82154: on a subscription OAuth credential, Anthropic's server-side content
filter rejected the first sentence of the built-in ``SKILLS_GUIDANCE`` prompt
and surfaced the rejection as ``HTTP 400 "You're out of extra usage."`` —
a billing-shaped message that sent users to buy quota they did not need.

Bisected against the live API against the full 71,721-char assembled prompt:
that sentence alone reproduced the 400, and removing it alone cleared it.
Size (20 KB of filler → 200) and the ``system[0]`` identity gate (a 429, not a
400) were both ruled out.

These tests pin the reword. They deliberately assert on the *trigger substrings*
rather than on an exact replacement string, so a future rewording is free to
change the prose as long as it does not reintroduce the rejected phrasing or
drop the behaviour the sentence exists to produce.
"""

from __future__ import annotations

import re

import pytest

from agent.prompt_builder import SKILLS_GUIDANCE


# Substrings unique to the rejected sentence. The bisect showed the trigger
# survives removal of the "(5+ tool calls)" clause, so the clause alone is not
# a sufficient guard — the surrounding phrasing is pinned too.
REJECTED_FRAGMENTS = (
    "After completing a complex task",
    "5+ tool calls",
    "fixing a tricky error",
    "save the approach as a",
    "so you can reuse it next time",
)


class TestRejectedPhrasingIsGone:
    @pytest.mark.parametrize("fragment", REJECTED_FRAGMENTS)
    def test_trigger_fragment_absent(self, fragment):
        assert fragment not in SKILLS_GUIDANCE, (
            f"{fragment!r} is part of the phrasing Anthropic's content filter "
            "rejects on subscription OAuth tokens (#82154)"
        )

    def test_first_sentence_is_the_verified_reword(self):
        # The reporter verified this replacement returns 200 where the original
        # returned 400. Pin the first line so a refactor can't silently revert it.
        first_line = SKILLS_GUIDANCE.split("\n", 1)[0]
        assert first_line == (
            "When you work out a non-trivial workflow, record it with skill_manage "
            "for future reuse."
        )


class TestBehaviourIsPreserved:
    """The reword must not cost the prompt its meaning — it still has to tell
    the model to record workflows as skills and to patch stale ones."""

    def test_still_instructs_recording_a_workflow_as_a_skill(self):
        first_line = SKILLS_GUIDANCE.split("\n", 1)[0].lower()
        assert "skill_manage" in first_line
        assert "workflow" in first_line
        assert "reuse" in first_line

    def test_patch_stale_skills_sentence_untouched(self):
        assert "skill_manage(action='patch')" in SKILLS_GUIDANCE
        assert "Skills that aren't maintained become liabilities." in SKILLS_GUIDANCE

    def test_skill_safety_rule_block_untouched(self):
        # Guarded independently by tests/agent/test_ghost_skill_pruning.py; asserted
        # here too so a reword of the guidance can't quietly take the block with it.
        assert "## Skill Safety Rule" in SKILLS_GUIDANCE
        for rule in ("UNAVAILABLE", "RELOAD", "WAIT", "DEDUP"):
            assert rule in SKILLS_GUIDANCE

    def test_real_newlines_and_line_count_preserved(self):
        # test_ghost_skill_pruning.py asserts count("\n") >= 6; the reword must
        # not drop a line separator on its way past that bound.
        assert "\\n" not in SKILLS_GUIDANCE
        assert SKILLS_GUIDANCE.count("\n") >= 6


class TestGuidanceReachesTheSystemPrompt:
    def test_guidance_is_wired_into_tool_guidance(self):
        # A reword is worthless if the constant stopped being appended. Assert the
        # wiring rather than trusting the constant in isolation.
        import inspect

        import agent.system_prompt as system_prompt

        source = inspect.getsource(system_prompt)
        assert re.search(r"tool_guidance\.append\(\s*SKILLS_GUIDANCE\s*\)", source)

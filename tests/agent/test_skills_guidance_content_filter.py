"""SKILLS_GUIDANCE must not carry the phrasing Anthropic's content filter rejects.

#82154: on a subscription OAuth credential, Anthropic's server-side content
filter rejected the first sentence of the built-in ``SKILLS_GUIDANCE`` prompt
and surfaced the rejection as ``HTTP 400 "You're out of extra usage."`` —
a billing-shaped message that sent users to buy quota they did not need.

Bisected against the live API against the full 71,721-char assembled prompt:
that sentence alone reproduced the 400, and removing it alone cleared it.
Size (20 KB of filler → 200) and the ``system[0]`` identity gate (a 429, not a
400) were both ruled out.

The remaining tests check that the guidance renders (real newlines) and is only
emitted when ``skill_manage`` is available — never its exact wording.
"""

from __future__ import annotations


from agent.prompt_builder import SKILLS_GUIDANCE


class TestBehaviourIsPreserved:
    """The reword must not cost the prompt its meaning — it still has to tell
    the model to record workflows as skills and to patch stale ones."""


    def test_real_newlines_preserved(self):
        """The block must contain REAL newlines (not escaped backslash-n
        literals) so the safety-rule heading renders as a heading."""
        assert chr(10) in SKILLS_GUIDANCE
        assert (chr(92) + 'n') not in SKILLS_GUIDANCE


class TestGuidanceReachesTheSystemPrompt:
    def test_guidance_is_wired_into_tool_guidance(self):
        # A reword is worthless if the constant stopped being emitted. Assert the
        # wiring behaviorally rather than trusting the constant in isolation.
        from types import SimpleNamespace

        import agent.system_prompt as system_prompt

        agent = SimpleNamespace(valid_tool_names={"skill_manage"}, _kanban_worker_guidance="")
        assert SKILLS_GUIDANCE in (system_prompt._tool_guidance_block(agent) or "")
        agent.valid_tool_names = {"terminal"}
        assert SKILLS_GUIDANCE not in (system_prompt._tool_guidance_block(agent) or "")

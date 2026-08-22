"""
Test that the MoA reference system prompt frames the advisor role safely.

Related issues: #61452 (advisors fabricating tool execution). The heavy
anti-imitation lifting is now structural — the digest advisory view removes
the tool-call transcript advisors used to imitate — so the prompt is short
and must NOT reintroduce the old confabulation license.
"""

from agent.moa_loop import _REFERENCE_SYSTEM_PROMPT


def test_reference_system_prompt_prohibits_claiming_execution():
    """The prompt must state the advisor executes nothing and never claims to.

    This addresses #61452 where reference models were fabricating tool
    execution in their text output.
    """
    prompt_lower = _REFERENCE_SYSTEM_PROMPT.lower()

    assert "you have no tools" in prompt_lower or "cannot call tools" in prompt_lower, \
        "Prompt must explicitly state that reference models have no tools"

    assert "execute nothing" in prompt_lower or "do not execute" in prompt_lower, \
        "Prompt must explicitly state that reference models execute nothing"

    assert "never claim" in prompt_lower or "never imply" in prompt_lower, \
        "Prompt must warn against claiming/implying execution"

    # Fabricated action logs / invented results are the observed failure mode.
    assert "action logs" in prompt_lower or "tool syntax" in prompt_lower, \
        "Prompt must prohibit writing action logs / tool syntax"
    assert "invent" in prompt_lower, \
        "Prompt must prohibit inventing results"


def test_reference_system_prompt_no_confabulation_license():
    """The old prompt told advisors to "assume any referenced files ... exist",
    which licensed invented verification ("SUCCESS — the fix is working").
    It must stay gone, replaced by an ask-the-agent-to-check rule.
    """
    prompt_lower = _REFERENCE_SYSTEM_PROMPT.lower()

    assert "assume any referenced" not in prompt_lower, \
        "The confabulation license must not return"
    assert "do not invent the answer" in prompt_lower, \
        "Prompt must direct advisors to name what the agent should check"


def test_reference_system_prompt_structure():
    """
    Verify the reference system prompt has a clear structure.

    A well-structured prompt helps models follow instructions better.
    """
    # Prompt should not be empty
    assert len(_REFERENCE_SYSTEM_PROMPT) > 100, \
        "Reference system prompt should be substantive"

    # Should have multiple paragraphs (structured guidance)
    assert _REFERENCE_SYSTEM_PROMPT.count("\n\n") >= 2, \
        "Prompt should be structured with multiple sections"

    # Should contain the word "advisor" (defines role)
    assert "advisor" in _REFERENCE_SYSTEM_PROMPT.lower(), \
        "Prompt should clearly define the advisor role"

    # It should describe the digest the advisor is reading.
    assert "digest" in _REFERENCE_SYSTEM_PROMPT.lower(), \
        "Prompt should describe the digest format the advisor receives"


def test_reference_system_prompt_premise_audit():
    """Advisors must audit the user's premise, not obey it.

    A false-premise request ("fix the inverted logic" when nothing is
    inverted) had an advisor proposing to break correct code. The prompt
    directs advisors to flag contradicted assumptions instead.
    """
    prompt_lower = _REFERENCE_SYSTEM_PROMPT.lower()
    assert "audit the request" in prompt_lower
    assert "contradicts an assumption" in prompt_lower

"""
Test that the MoA reference system prompt contains explicit warnings
against claiming tool execution.

Related issue: #61452
"""

from agent.moa_loop import _REFERENCE_SYSTEM_PROMPT


def test_reference_system_prompt_prohibits_claiming_execution():
    """
    Verify that the reference system prompt contains explicit warnings
    against claiming tool execution.

    The prompt should:
    1. State that reference models cannot execute anything
    2. Warn against claiming/implying execution
    3. Provide bad/good examples

    This addresses #61452 where reference models were fabricating
    tool execution in their text output.
    """
    prompt_lower = _REFERENCE_SYSTEM_PROMPT.lower()

    # Critical constraints
    assert "you cannot call tools" in prompt_lower or "you do not execute" in prompt_lower, \
        "Prompt must explicitly state that reference models cannot execute"

    assert "never claim" in prompt_lower or "never imply" in prompt_lower, \
        "Prompt must warn against claiming/implying execution"

    # Check for examples (helps models understand what NOT to do)
    assert "bad:" in prompt_lower or "avoid:" in prompt_lower, \
        "Prompt should provide negative examples"

    # Specific action verbs that should NOT appear as claimed actions
    # (these are common patterns of hallucinated execution)
    forbidden_patterns = [
        "i ran", "i executed", "i downloaded", "i accessed",
        "i checked", "i called", "i browsed"
    ]

    # The prompt should mention these as bad examples
    # (i.e., in the context of what to avoid, not as instruction)
    has_any_forbidden = any(
        f"bad: \"{pattern}" in _REFERENCE_SYSTEM_PROMPT.lower() or
        f"avoid \"{pattern}" in _REFERENCE_SYSTEM_PROMPT.lower()
        for pattern in forbidden_patterns
    )

    # At least one bad example pattern should exist
    assert has_any_forbidden or "examples" in _REFERENCE_SYSTEM_PROMPT.lower(), \
        "Prompt should contain examples of what to avoid"


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


def test_slot_ref_system_stock_then_hint_then_evidence_order():
    """Composition order: stock framing, prompt_hint, then retrieved evidence."""
    from agent.moa_loop import _slot_ref_system

    slot = {
        "prompt_hint": "act as the steelman",
        "context_command": ["echo", "EVIDENCE_BLOCK"],
    }
    prompt = _slot_ref_system(slot, [{"role": "user", "content": "q"}])
    assert prompt.startswith(_REFERENCE_SYSTEM_PROMPT)
    assert prompt.index("act as the steelman") < prompt.index("EVIDENCE_BLOCK")
    assert "Retrieved material for this turn" in prompt


def test_slot_ref_system_failed_command_renders_sentinel_not_silence():
    """A broken retriever renders as retrieval failure, never an empty record."""
    from agent.moa_loop import _slot_ref_system

    prompt = _slot_ref_system(
        {"context_command": ["sh", "-c", "exit 3"]},
        [{"role": "user", "content": "q"}],
    )
    assert "RETRIEVAL FAILED this turn" in prompt
    assert "the tool broke, not the record" in prompt


def test_slot_ref_system_prompt_replaces_stock_and_keeps_evidence():
    """prompt replaces the stock framing; evidence is still appended last."""
    from agent.moa_loop import _slot_ref_system

    prompt = _slot_ref_system(
        {"prompt": "FULL_REPLACEMENT", "context_command": ["echo", "EVIDENCE_BLOCK"]},
        [{"role": "user", "content": "q"}],
    )
    assert prompt.startswith("FULL_REPLACEMENT")
    assert "EVIDENCE_BLOCK" in prompt
    assert _REFERENCE_SYSTEM_PROMPT not in prompt


def test_slot_ref_system_reuses_precomputed_context_block(monkeypatch):
    """A precomputed context_block must not re-run the slot command.

    The context_command is a subprocess; resolving the same slot twice per
    reference would double retrieval latency and side effects at fan-out.
    """
    from agent import moa_loop

    calls = []
    original = moa_loop._slot_context_block

    def counting(slot, ref_messages=None):
        calls.append(1)
        return original(slot, ref_messages)

    monkeypatch.setattr(moa_loop, "_slot_context_block", counting)

    prompt = moa_loop._slot_ref_system(
        {"context_command": ["echo", "EVIDENCE_BLOCK"]},
        [{"role": "user", "content": "q"}],
        context_block="PRECOMPUTED",
    )
    assert "PRECOMPUTED" in prompt
    assert calls == []

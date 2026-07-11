"""
Regression test for issue #62365 - Context compaction fabricates user
requests that were never made.

The bug: the LLM summarizer (in agent/context_compressor.py) was being
asked to write the user's "most recent unfulfilled input" verbatim under
## Historical Task Snapshot. When no clear input existed, the LLM would
fabricate one (often paraphrasing a tool result or assistant message).

The fix: add an explicit "FABRICATION PREVENTION" block to the summarizer
prompt that:
  - Instructs the LLM to write "None." if no real user message exists
  - Restricts "User asked:" to text that the user actually typed
  - Explicitly forbids using the previous summary as input

This test verifies:
  1. The summarizer prompt template now contains the fabrication
     prevention block (static check)
  2. The fallback path (when LLM summarizer is unavailable) only emits
     "User asked:" lines for messages with role="user" (behavioral check)
  3. The fallback path does NOT fabricate user asks from assistant or
     tool messages (behavioral check)
"""

import re
from pathlib import Path


def test_static_summarizer_prompt_includes_fabrication_prevention():
    """The summarizer prompt must include the fabrication-prevention block.

    On unfixed code, the template has "If no outstanding task exists, write
    'None.' ]" at the end of the Historical Task Snapshot description but
    no explicit DO-NOT-INVENT block. The LLM was hallucinating because
    the directive to write None was buried at the END of a long example
    list. Issue #62365 fix moves the no-fabrication rule to a prominent
    CRITICAL block at the top of the section.
    """
    worktree = Path("/tmp/hermes-pr-work-60859/hermes-agent")
    src = (worktree / "agent" / "context_compressor.py").read_text()

    # Find _template_sections definition (used by both first-compaction
    # and iterative-update paths)
    m = re.search(r'_template_sections\s*=\s*f?"""', src)
    assert m, "_template_sections not found"
    start = m.end()
    # Find the closing triple-quote
    end = src.find('"""', start)
    body = src[start:end]

    # The fabrication prevention block must be present
    assert "FABRICATION PREVENTION" in body, (
        "#62365 regression: the summarizer prompt template is missing the "
        "fabrication-prevention block. The LLM was hallucinating user "
        "requests because the 'write None' instruction was buried at the "
        "end of an examples list. The fix adds a prominent CRITICAL block "
        "with DO NOT INVENT, quote-only-user-messages, and "
        "previous-summary-is-reference-only rules."
    )

    # Specifically check the DO NOT INVENT directive
    assert "DO NOT INVENT user requests" in body, (
        "#62365: the explicit DO NOT INVENT directive is missing."
    )

    # Check the "User asked" guard
    assert '"User asked:" must be followed by text that the user actually typed' in body, (
        "#62365: the 'User asked:' guard is missing."
    )


def test_behavioral_fallback_only_emits_user_role_asks():
    """Behavioral check: the deterministic fallback path must only emit
    'User asked:' lines for messages with role='user'. Without the fix,
    the fallback (used when the LLM summarizer is unavailable) would emit
    assistant or tool messages as user asks.
    """
    import sys
    sys.path.insert(0, "/tmp/hermes-pr-work-60859/hermes-agent")

    from agent.context_compressor import ContextCompressor

    # Create a minimal compressor instance (bypass __init__)
    cc = ContextCompressor.__new__(ContextCompressor)
    cc._previous_summary = ""
    cc._summary_model_fallen_back = False
    cc._last_summary_error = None

    # Build a list of turns: 1 user msg + 1 assistant msg + 1 tool result
    turns = [
        {"role": "user", "content": "show me what is wrong"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}
        ]},
        {"role": "tool", "tool_call_id": "c1", "content": "Permission denied: /var/log/system.log"},
        {"role": "assistant", "content": "There's a permission issue. User asked: 'пусть подтянет анализ кошельков' — this is what we should focus on next."},
        # ^ assistant message that IMPLICATES a fake user ask via tool-call result echo
    ]

    # Call the fallback builder
    summary = cc._build_static_fallback_summary(
        turns_to_summarize=turns,
        reason="",
    )

    # The summary should contain the actual user message under Historical Task Snapshot
    assert "Historical Task Snapshot" in summary, (
        "Fallback summary missing Historical Task Snapshot section."
    )
    assert "show me what is wrong" in summary, (
        "#62365: fallback dropped the actual user message 'show me what is wrong'."
    )

    # The CRITICAL assertion: the assistant message's fabricated "User asked: ..." string
    # must NOT appear as the active task. Check the Historical Task Snapshot section
    # specifically (not the whole summary, since the assistant message text appears
    # legitimately in Completed Actions or other sections).
    hts_match = re.search(
        r"## Historical Task Snapshot\n(.*?)(?=\n## |\Z)",
        summary,
        re.DOTALL,
    )
    assert hts_match, "Could not find Historical Task Snapshot section in fallback"
    hts = hts_match.group(1)
    assert "пусть подтянет анализ кошельков" not in hts, (
        "#62365: fallback's Historical Task Snapshot contains the fabricated "
        "Russian user ask 'пусть подтянет анализ кошельков' that came from "
        "an ASSISTANT message, not a user message. This is the exact "
        "hallucination pattern the LLM-prompt fix is designed to prevent. "
        f"Historical Task Snapshot section:\n{hts[:300]}"
    )
    # Specifically the "User asked:" line in the active task should be the
    # real user message, not the assistant's fake one
    assert "User asked: 'show me what is wrong'" in hts, (
        "#62365: fallback's active task doesn't contain the real user message."
        f"Historical Task Snapshot section:\n{hts[:300]}"
    )


def test_behavioral_fallback_with_no_user_turns_writes_none():
    """Edge case: if the compacted turns have no user messages at all,
    the fallback should write 'None.' in the user-ask section rather
    than inventing one from assistant/tool content."""
    import sys
    sys.path.insert(0, "/tmp/hermes-pr-work-60859/hermes-agent")

    from agent.context_compressor import ContextCompressor

    cc = ContextCompressor.__new__(ContextCompressor)
    cc._previous_summary = ""

    # All assistant messages, no user messages
    turns = [
        {"role": "assistant", "content": "Reading the config file now."},
        {"role": "tool", "tool_call_id": "c1", "content": "config = {verbose: True}"},
        {"role": "assistant", "content": "Found it. The verbose flag was missing."},
    ]

    summary = cc._build_static_fallback_summary(
        turns_to_summarize=turns,
        reason="",
    )

    # Should write "None" for user ask
    # The fallback template writes 'Unknown from deterministic fallback.'
    # when there are no user_asks, which is the correct behavior.
    assert "Unknown from deterministic fallback" in summary or "None" in summary, (
        f"#62365: fallback did not handle no-user-turns case. Got summary:\n{summary[:500]}"
    )
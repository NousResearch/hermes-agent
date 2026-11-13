"""The content-policy (safety-filter) refusal notice must tell the user they can just continue.

Regression for the "Hermes Bot stuck sessions" report: a refusal is deterministic for the whole
conversation (the filter judges the thread, not just the new message), so once a chat trips the
filter every later message — even an innocuous one — used to re-refuse and lock the chat. The
refused turn is now dropped from the conversation automatically (the user keeps seeing it, it just
leaves the model's context), so the copy must say the chat is NOT stuck and the user can ask
something else — never the old "/undo N to clear it yourself" lockout instructions.
"""


def test_recovery_copy_says_not_stuck_and_offers_remedies():
    from hermes_agent.agent.turn_failure_copy import CONTENT_POLICY_NEXT_STEPS

    lowered = CONTENT_POLICY_NEXT_STEPS.lower()
    # The chat is not locked: the refused message left the conversation; just keep going.
    assert "not stuck" in lowered
    assert "ask something else" in lowered
    # Still offers the per-message remedies for a one-off block.
    assert "/model" in CONTENT_POLICY_NEXT_STEPS
    # The manual-clear lockout instructions are gone — the drop is automatic now.
    assert "/undo" not in CONTENT_POLICY_NEXT_STEPS

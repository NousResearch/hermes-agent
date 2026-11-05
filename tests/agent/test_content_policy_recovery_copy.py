"""The content-policy (safety-filter) refusal guidance must point at the escape from a locked chat.

Regression for the "Hermes Bot stuck sessions" report: a refusal is deterministic for the whole
conversation (the filter judges the thread, not just the new message), so once a chat trips the
filter every later message — even an innocuous one — re-refuses. "Reword your message" is useless
there; the only recovery is clearing the recent context. The user-facing next-steps copy now names
the model-free escape (/undo, repeated) that works on every surface.
"""


def test_recovery_copy_names_undo():
    from agent.turn_failure_copy import CONTENT_POLICY_NEXT_STEPS

    # The model-free escape, reachable on every surface (CLI, TUI, Desktop Bot Chat, messaging).
    assert "/undo" in CONTENT_POLICY_NEXT_STEPS
    # Still offers the per-message remedies for a one-off block.
    assert "/model" in CONTENT_POLICY_NEXT_STEPS

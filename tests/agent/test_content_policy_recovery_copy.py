"""A safety refusal notice must not promise that a later request will succeed."""


def test_recovery_copy_explains_context_without_suggesting_bypass():
    from agent.turn_failure_copy import content_policy_copy

    notice = content_policy_copy(label="Example provider", summary="Provider policy rejection")
    assert "Provider policy rejection" in notice
    assert "earlier conversation context" in notice.lower()
    assert "may still be refused" in notice
    assert "/model" not in notice
    assert "not stuck" not in notice

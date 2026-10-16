from agent.prompt_builder import PLATFORM_HINTS


def test_slack_prompt_requires_direct_creation_under_generated_delivery_root():
    hint = PLATFORM_HINTS["slack"]

    assert "$HERMES_HOME/cache/generated" in hint
    assert "create" in hint.lower()
    assert "directly" in hint.lower()
    assert "cache/scratch" in hint
    assert "Downloads" in hint
    assert "copy" in hint.lower()

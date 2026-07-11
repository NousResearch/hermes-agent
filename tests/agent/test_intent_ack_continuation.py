"""Model-sovereignty contract: no keyword-based intent continuation."""
import agent.agent_runtime_helpers as helpers


def test_runtime_has_no_keyword_intent_ack_classifier():
    assert not hasattr(helpers, "looks_like_codex_intermediate_ack")
    assert not hasattr(helpers, "intent_ack_continuation_mode")
    assert not hasattr(helpers, "intent_ack_continuation_enabled")

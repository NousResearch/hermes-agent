from plugins.conversation_posture import (
    ORDINARY_CHAT_CANARY_SAMPLES,
    POSTURE_CANARY_VERDICT,
    run_canary,
)


def test_conversation_posture_canary_is_hermes_owned():
    verdict = run_canary()

    assert verdict["ok"] is True
    assert verdict["profile"] == "conversation_to_commitment"
    assert verdict["ordinary_chat_samples"] == ORDINARY_CHAT_CANARY_SAMPLES
    assert POSTURE_CANARY_VERDICT["ok"] is True


def test_ordinary_chat_samples_avoid_diagnostic_ceremony_terms():
    forbidden = ("workflow", "process", "tool call", "mcp", "runbook", "pipeline")

    assert ORDINARY_CHAT_CANARY_SAMPLES
    for sample in ORDINARY_CHAT_CANARY_SAMPLES:
        text = sample["assistant"].lower()
        for term in forbidden:
            assert term not in text

from gateway.response_filters import (
    is_background_notification_silent_ack,
    is_intentional_silence_agent_result,
    is_intentional_silence_response,
)


def test_exact_silence_tokens_are_intentional_silence():
    for token in ("[SILENT]", " SILENT ", "NO_REPLY", "no reply"):
        assert is_intentional_silence_response(token)


def test_edge_punctuation_silence_tokens_are_intentional_silence():
    for token in (".NO_REPLY", "*NO_REPLY*", " .NO_REPLY ", "*[SILENT]*", "NO_REPLY."):
        assert is_intentional_silence_response(token)


def test_blank_and_prose_mentions_are_not_silence():
    assert not is_intentional_silence_response("")
    assert not is_intentional_silence_response("Use NO_REPLY when no answer is needed.")
    assert not is_intentional_silence_response("The reply was [SILENT], intentionally.")
    assert not is_intentional_silence_response("😄 NO_REPLY")
    assert not is_intentional_silence_response("[SILENT")


def test_failed_agent_result_never_counts_as_intentional_silence():
    assert is_intentional_silence_agent_result({"failed": False}, "NO_REPLY")
    assert not is_intentional_silence_agent_result({"failed": True}, "NO_REPLY")


def test_background_notification_silent_ack_is_suppressed():
    # Given: a successful background-notification turn chose intentional silence.
    result = {"final_response": "NO_REPLY", "failed": False}

    # When: the gateway classifies its delivery outcome.
    suppressed = is_background_notification_silent_ack(result, is_background_notification=True)

    # Then: the acknowledgment is consumed without delivery.
    assert suppressed is True


def test_background_notification_substantive_result_is_delivered():
    # Given: the notification surfaced a new build failure.
    result = {"final_response": "Build failed for the first time: test_api", "failed": False}

    # When: the gateway classifies its delivery outcome.
    suppressed = is_background_notification_silent_ack(result, is_background_notification=True)

    # Then: substantive information remains deliverable.
    assert suppressed is False


def test_ordinary_user_acknowledgment_is_never_background_suppressed():
    # Given: an ordinary user turn received short acknowledgment-like prose.
    result = {
        "final_response": "이미 처리한 작업이라 재실행할 건 없습니다.",
        "failed": False,
    }

    # When: the turn has no background-notification provenance.
    suppressed = is_background_notification_silent_ack(result, is_background_notification=False)

    # Then: the new background-only guard cannot hide the user reply.
    assert suppressed is False


def test_background_guard_preserves_empty_and_failed_responses():
    # Given: outcomes that existing gateway recovery paths must surface.
    results = (
        {"final_response": "(empty)", "failed": False},
        {"final_response": "NO_REPLY", "failed": True},
    )

    # When: they originate from a background-notification turn.
    suppressed = [
        is_background_notification_silent_ack(
            result,
            is_background_notification=True,
        )
        for result in results
    ]

    # Then: neither the model-failure sentinel nor an explicit failure is hidden.
    assert suppressed == [False, False]

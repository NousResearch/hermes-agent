from gateway.response_filters import (
    should_suppress_delivery,
    validated_delivery_outcome,
)


def _result(*, action="suppress", reason="nothing new", failed=False):
    return {
        "failed": failed,
        "turn_id": "turn-1",
        "delivery_outcome": {
            "action": action,
            "reason": reason,
            "turn_id": "turn-1",
        },
    }


def test_exact_same_turn_structured_suppress_is_executed():
    assert validated_delivery_outcome(_result()) == {
        "action": "suppress",
        "reason": "nothing new",
        "turn_id": "turn-1",
    }
    assert should_suppress_delivery(_result()) is True
    assert should_suppress_delivery(_result(action="deliver")) is False


def test_failed_or_unknown_turn_status_always_delivers():
    assert should_suppress_delivery(_result(failed=True)) is False
    unknown = _result()
    unknown.pop("failed")
    assert should_suppress_delivery(unknown) is False


def test_stale_or_malformed_outcome_always_delivers():
    stale = _result()
    stale["delivery_outcome"]["turn_id"] = "turn-0"
    assert should_suppress_delivery(stale) is False

    extra = _result()
    extra["delivery_outcome"]["extra"] = True
    assert should_suppress_delivery(extra) is False

    empty_reason = _result(reason="   ")
    assert should_suppress_delivery(empty_reason) is False


def test_response_text_has_no_delivery_authority():
    for text in ("[SILENT]", "NO_REPLY", ".NO_REPLY", "ordinary report"):
        result = {
            "failed": False,
            "turn_id": "turn-1",
            "delivery_outcome": None,
            "final_response": text,
        }
        assert should_suppress_delivery(result) is False

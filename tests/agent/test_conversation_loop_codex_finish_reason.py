from types import SimpleNamespace

from agent.conversation_loop import _codex_response_finish_reason


def test_codex_incomplete_content_filter_maps_to_content_filter_dict():
    response = SimpleNamespace(
        status="incomplete",
        incomplete_details={"reason": "content_filter"},
    )

    assert _codex_response_finish_reason(response) == "content_filter"


def test_codex_incomplete_content_filter_maps_to_content_filter_object():
    response = SimpleNamespace(
        status="incomplete",
        incomplete_details=SimpleNamespace(reason="content_filter"),
    )

    assert _codex_response_finish_reason(response) == "content_filter"


def test_codex_incomplete_length_reasons_still_map_to_length():
    for reason in ("max_output_tokens", "length"):
        response = SimpleNamespace(
            status="incomplete",
            incomplete_details={"reason": reason},
        )
        assert _codex_response_finish_reason(response) == "length"


def test_codex_completed_response_maps_to_stop():
    response = SimpleNamespace(status="completed", incomplete_details=None)

    assert _codex_response_finish_reason(response) == "stop"

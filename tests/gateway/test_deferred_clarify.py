import json


def test_deferred_marker_round_trips_as_provider_valid_tool_result():
    from gateway.extensions.deferred_clarify import (
        DEFERRED_CLARIFY_KIND,
        is_deferred_clarify_result,
        make_deferred_marker,
        parse_deferred_marker,
    )

    marker = make_deferred_marker("cld_123")
    payload = json.loads(marker)

    assert payload["status"] == "deferred"
    assert payload["kind"] == DEFERRED_CLARIFY_KIND
    assert payload["interaction_id"] == "cld_123"
    assert is_deferred_clarify_result(marker) is True
    assert parse_deferred_marker(marker) == "cld_123"


def test_recovery_prompt_includes_question_answer_and_instruction():
    from gateway.extensions.deferred_clarify import build_recovery_prompt

    prompt = build_recovery_prompt(
        question="Deploy where?",
        answer="staging",
    )

    assert "previous Hermes turn" in prompt
    assert "Deploy where?" in prompt
    assert "staging" in prompt
    assert "Do not ask the same clarification again" in prompt


def test_marker_parser_ignores_malformed_or_other_tool_results():
    from gateway.extensions.deferred_clarify import (
        is_deferred_clarify_result,
        parse_deferred_marker,
    )

    assert parse_deferred_marker("not json") is None
    assert parse_deferred_marker('{"status":"ok"}') is None
    assert is_deferred_clarify_result('{"status":"ok"}') is False


def test_finds_marker_inside_actual_clarify_tool_result_envelope():
    from agent.tool_dispatch_helpers import make_tool_result_message
    from gateway.extensions.deferred_clarify import (
        find_deferred_clarify_interaction_id,
        make_deferred_marker,
    )
    from tools.clarify_tool import clarify_tool

    tool_result = clarify_tool(
        question="Deploy where?",
        choices=["staging", "production"],
        callback=lambda _question, _choices: make_deferred_marker("cld_executor_123"),
    )
    messages = [make_tool_result_message("clarify", tool_result, "call_clarify")]

    assert (
        find_deferred_clarify_interaction_id(messages, {"call_clarify"})
        == "cld_executor_123"
    )


def test_deferred_detection_is_constrained_to_clarify_tool_envelope():
    from gateway.extensions.deferred_clarify import (
        find_deferred_clarify_interaction_id,
        make_deferred_marker,
    )
    from tools.clarify_tool import clarify_tool

    marker = make_deferred_marker("cld_other_tool")
    clarify_envelope = clarify_tool(
        question="Deploy where?",
        callback=lambda _question, _choices: marker,
    )

    assert (
        find_deferred_clarify_interaction_id(
            [
                {
                    "role": "tool",
                    "name": "terminal",
                    "tool_call_id": "call_1",
                    "content": clarify_envelope,
                }
            ],
            {"call_1"},
        )
        is None
    )
    assert (
        find_deferred_clarify_interaction_id(
            [
                {
                    "role": "tool",
                    "name": "clarify",
                    "tool_call_id": "call_1",
                    "content": marker,
                }
            ],
            {"call_1"},
        )
        is None
    )

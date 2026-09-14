from copy import deepcopy

from agent.responses_replay_identity import enforce_responses_replay_identity_closure


def _message(item_id: str = "msg_1", text: str = "answer"):
    return {
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "phase": "final_answer",
        "id": item_id,
        "content": [{"type": "output_text", "text": text}],
    }


def _reasoning():
    return {
        "type": "reasoning",
        "encrypted_content": "sealed",
        "summary": [],
    }


def test_drops_message_id_linked_to_reasoning():
    request = {"input": [_reasoning(), _message()]}

    sanitized = enforce_responses_replay_identity_closure(request)

    assert "id" not in sanitized["input"][1]
    assert sanitized["input"][1]["phase"] == "final_answer"
    assert sanitized["input"][1]["content"] == [
        {"type": "output_text", "text": "answer"}
    ]


def test_reasoning_dependency_stays_open_through_function_call():
    request = {
        "input": [
            _reasoning(),
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "read_file",
                "arguments": "{}",
            },
            _message("msg_commentary", "still working"),
        ]
    }

    sanitized = enforce_responses_replay_identity_closure(request)

    assert "id" not in sanitized["input"][2]


def test_tool_output_closes_reasoning_dependency_group():
    request = {
        "input": [
            _reasoning(),
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "read_file",
                "arguments": "{}",
            },
            {"type": "function_call_output", "call_id": "call_1", "output": "ok"},
            _message("msg_independent"),
        ]
    }

    sanitized = enforce_responses_replay_identity_closure(request)

    assert sanitized["input"][-1]["id"] == "msg_independent"


def test_keeps_independent_message_id_without_reasoning_dependency():
    request = {"input": [_message("msg_independent")]}

    sanitized = enforce_responses_replay_identity_closure(request)

    assert sanitized is request
    assert sanitized["input"][0]["id"] == "msg_independent"


def test_drop_all_mode_removes_message_ids_when_reasoning_replay_is_disabled():
    request = {"input": [_message("msg_unknown_dependency")]}

    sanitized = enforce_responses_replay_identity_closure(
        request, drop_all_message_ids=True
    )

    assert "id" not in sanitized["input"][0]


def test_does_not_mutate_original_request():
    request = {"input": [_reasoning(), _message()]}
    original = deepcopy(request)

    enforce_responses_replay_identity_closure(request)

    assert request == original

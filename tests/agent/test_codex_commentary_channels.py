"""Commentary is an assistant message, not a reasoning summary or a final answer."""

from copy import deepcopy
from types import SimpleNamespace as NS

import pytest

from agent.codex_responses_adapter import (
    _chat_messages_to_responses_input,
    _normalize_codex_response,
)


def _message(text, phase, item_id="msg_progress"):
    return NS(
        type="message", role="assistant", id=item_id, phase=phase,
        status="completed", content=[NS(type="output_text", text=text)],
    )


def _reasoning(text="Reasoning summary."):
    return NS(
        type="reasoning", id="rs_summary", status="completed",
        encrypted_content="opaque-replay-state",
        summary=[NS(type="summary_text", text=text)],
    )


@pytest.mark.parametrize("issuer", [None, "codex_backend", "github_responses", "xai_responses"])
def test_mixed_response_keeps_commentary_out_of_both_reasoning_and_final(issuer):
    response = NS(status="completed", output=[
        _reasoning(),
        _message("I'll inspect the file.", "commentary"),
        _message("Analysis summary.", "analysis", "msg_analysis"),
        _message("The file is correct.", "final_answer", "msg_final"),
    ])
    original = deepcopy(response)

    message, finish_reason = _normalize_codex_response(response, issuer_kind=issuer)

    assert finish_reason == "stop"
    assert message.content == "The file is correct."
    assert message.reasoning == "Reasoning summary.\n\nAnalysis summary."
    assert message.reasoning_content is None
    assert message.reasoning_details is None
    assert message.codex_message_items == [
        {
            "type": "message", "role": "assistant", "status": "completed",
            "id": item.id, "phase": item.phase,
            "content": [{"type": "output_text", "text": item.content[0].text}],
        }
        for item in response.output if item.type == "message"
    ]
    assert message.codex_reasoning_items[0]["encrypted_content"] == "opaque-replay-state"
    assert response == original


@pytest.mark.parametrize("phase", ["commentary", " Commentary ", "COMMENTARY"])
def test_commentary_only_is_visible_sidecar_not_reasoning_or_final(phase):
    text = "I'll inspect the file."
    message, finish_reason = _normalize_codex_response(
        NS(status="completed", output=[_message(text, phase)], output_text=text),
        issuer_kind="codex_backend",
    )

    assert finish_reason == "incomplete"
    assert message.content == ""
    assert message.reasoning is None
    assert message.codex_reasoning_items is None
    assert message.codex_message_items[0]["phase"] == "commentary"
    assert message.codex_message_items[0]["content"][0]["text"] == text


def test_normalized_commentary_and_encrypted_state_replay_without_reclassification():
    message, _ = _normalize_codex_response(NS(status="completed", output=[
        _reasoning(),
        _message("I'll inspect the file.", "commentary"),
        _message("The file is correct.", "final_answer", "msg_final"),
    ]), issuer_kind="codex_backend")
    history = [
        {"role": "user", "content": "Check the file."},
        {"role": "assistant", **vars(message)},
    ]
    original = deepcopy(history)

    replay = _chat_messages_to_responses_input(history, current_issuer_kind="codex_backend")

    assert replay == [
        {"role": "user", "content": "Check the file."},
        {
            "type": "reasoning", "encrypted_content": "opaque-replay-state",
            "summary": [{"type": "summary_text", "text": "Reasoning summary."}],
        },
        *message.codex_message_items,
    ]
    assert history == original


def test_actual_reasoning_is_not_promoted_based_on_user_facing_wording():
    message, finish_reason = _normalize_codex_response(
        NS(status="completed", output=[_reasoning("The answer is 42.")]),
        issuer_kind="codex_backend",
    )

    assert finish_reason == "incomplete"
    assert message.content == ""
    assert message.reasoning == "The answer is 42."
    assert message.codex_message_items is None


def test_xai_response_marker_in_commentary_does_not_turn_progress_into_final():
    text = "I will inspect <response>the file</response> next."
    message, finish_reason = _normalize_codex_response(
        NS(status="completed", output=[_message(text, "commentary")]),
        issuer_kind="xai_responses",
    )

    assert finish_reason == "incomplete"
    assert message.content == ""
    assert message.reasoning is None
    assert message.codex_message_items[0]["content"][0]["text"] == text

"""Message-scoped stream phase/terminal evidence, with vendor-boundary event fixtures."""
from copy import deepcopy
import json
from types import SimpleNamespace as NS

import pytest

from agent.codex_runtime import _consume_codex_event_stream
from agent.codex_responses_adapter import _normalize_codex_response


def msg(item_id="msg_a", text="Checking.", phase="commentary", status="completed"):
    result = {"type": "message", "role": "assistant", "id": item_id, "status": status,
              "content": [{"type": "output_text", "text": text}]}
    if phase is not None:
        result["phase"] = phase
    return result


def added(item, index=0):
    return {"type": "response.output_item.added", "output_index": index, "item": item}


def done(item, index=0):
    return {"type": "response.output_item.done", "output_index": index, "item": item}


def delta(text, item_id="msg_a", index=0):
    return {"type": "response.output_text.delta", "item_id": item_id, "output_index": index, "delta": text}


def terminal(output=None, kind="completed", **fields):
    return {"type": "response." + kind, "response": {"status": kind, "output": output, **fields}}


def consume(events, *, raw=False):
    original = deepcopy(events)
    wire = events if raw else json.loads(json.dumps(events), object_hook=lambda values: NS(**values))
    before = deepcopy(wire)
    text, commentary, reasoning = [], [], []
    response = _consume_codex_event_stream(iter(wire), model="test-model", on_text_delta=text.append,
                                          on_commentary_message=commentary.append, on_reasoning_delta=reasoning.append)
    assert wire == before
    assert events == original
    return response, text, commentary, reasoning


@pytest.mark.parametrize("raw", [False, True])
def test_phase_from_added_survives_missing_phase_on_done(raw):
    response, text, commentary, _ = consume([
        added(msg(text="", status="in_progress")), delta("Checking."),
        done(msg(phase=None)), terminal(),
    ], raw=raw)
    result, finish = _normalize_codex_response(response, issuer_kind="codex_backend")
    assert finish == "incomplete"
    assert result.content == ""
    assert result.codex_message_items[0]["phase"] == "commentary"
    assert text == []
    assert commentary == ["Checking."]


def test_interleaved_item_does_not_steal_commentary_phase():
    response, text, commentary, _ = consume([
        added(msg(text="", status="in_progress")),
        added({"type": "reasoning", "id": "rs_other"}, 1),
        delta("Checking."), done(msg()), terminal(),
    ])
    assert text == []
    assert commentary == ["Checking."]
    assert _normalize_codex_response(response, issuer_kind="codex_backend")[1] == "incomplete"


def test_parallel_commentary_buffers_and_output_order_are_item_scoped():
    response, text, commentary, _ = consume([
        added(msg("msg_a", "", status="in_progress"), 0), delta("A", "msg_a", 0),
        added(msg("msg_b", "", status="in_progress"), 1), delta("B", "msg_b", 1),
        delta("1", "msg_a", 0), delta("2", "msg_b", 1),
        done(msg("msg_b", "B2"), 1), done(msg("msg_a", "A1"), 0), terminal(),
    ])
    assert commentary == ["B2", "A1"]
    assert text == []
    result, _ = _normalize_codex_response(response, issuer_kind="codex_backend")
    assert [item["id"] for item in result.codex_message_items] == ["msg_a", "msg_b"]


def test_index_only_delta_resolves_announced_message_identity():
    event = delta("Checking.")
    event.pop("item_id")
    _, text, commentary, _ = consume([added(msg(text="")), added({"type": "reasoning", "id": "rs_other"}, 1), event, done(msg()), terminal()])
    assert text == []
    assert commentary == ["Checking."]


def test_unknown_explicit_item_id_does_not_inherit_another_phase():
    _, text, _, reasoning = consume([
        added(msg(text="", phase="analysis")),
        delta("Answer", "msg_unknown", 2),
        done(msg("msg_unknown", "Answer", "final_answer"), 2), terminal(),
    ])
    assert text == ["Answer"]
    assert reasoning == []


def test_complete_commentary_text_beats_a_truncated_delta_buffer():
    _, _, commentary, _ = consume([added(msg(text="")), delta("Check"), done(msg(text="Checking all paths.")), terminal()])
    assert commentary == ["Checking all paths."]


@pytest.mark.parametrize("raw", [False, True])
def test_valid_terminal_output_recovers_missing_final_without_duplicates(raw):
    progress = msg()
    final = msg("msg_final", "The answer is 42.", "final_answer")
    response, _, commentary, _ = consume([added(msg(text="")), delta("Checking."), done(progress), terminal([progress, final])], raw=raw)
    result, finish = _normalize_codex_response(response, issuer_kind="codex_backend")
    assert finish == "stop"
    assert result.content == "The answer is 42."
    assert len(result.codex_message_items) == 2
    assert commentary == ["Checking."]


def test_terminal_metadata_restores_phase_omitted_from_done():
    response, _, commentary, _ = consume([done(msg(phase=None)), terminal([msg()])])
    result, finish = _normalize_codex_response(response, issuer_kind="codex_backend")
    assert finish == "incomplete"
    assert result.content == ""
    assert len(result.codex_message_items) == 1
    assert commentary == ["Checking."]


@pytest.mark.parametrize("malformed", [None, [], "broken", {}, [None, 1, {}]])
def test_malformed_terminal_output_does_not_erase_done_items(malformed):
    response, _, _, _ = consume([done(msg(text="42", phase="final_answer")), terminal(malformed)])
    result, finish = _normalize_codex_response(response)
    assert finish == "stop"
    assert result.content == "42"


def test_terminal_snapshot_does_not_overwrite_completed_item_text():
    response, _, _, _ = consume([done(msg(text="Original answer", phase="final_answer")), terminal([msg(text="Conflicting copy", phase="final_answer")])])
    result, finish = _normalize_codex_response(response)
    assert finish == "stop"
    assert result.content == "Original answer"
    assert len(result.codex_message_items) == 1


@pytest.mark.parametrize("with_done", [False, True])
def test_eof_without_terminal_retains_partial_but_never_claims_completion(with_done):
    events = [delta("Partial answer.")]
    if with_done:
        events.append(done(msg(text="Partial answer.", phase="final_answer")))
    response, _, _, _ = consume(events)
    result, finish = _normalize_codex_response(response, issuer_kind="codex_backend")
    assert response.status != "completed"
    assert finish == "incomplete"
    assert result.content == "Partial answer."


def test_commentary_deltas_without_done_keep_their_phase():
    response, text, commentary, _ = consume([added(msg(text="", status="in_progress")), delta("Checking."), terminal()])
    result, finish = _normalize_codex_response(response, issuer_kind="codex_backend")
    assert finish == "incomplete"
    assert result.content == ""
    assert result.codex_message_items[0]["phase"] == "commentary"
    assert text == []
    assert commentary == ["Checking."]


@pytest.mark.parametrize("kind", ["incomplete", "failed"])
def test_terminal_event_type_supplies_absent_payload_status(kind):
    response, _, _, _ = consume([done(msg(text="Partial", phase="final_answer")), {"type": "response." + kind, "response": {}}])
    assert response.status == kind


def test_pending_function_call_is_not_settled_on_eof():
    tool = {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "terminal", "arguments": "{"}
    with pytest.raises(RuntimeError, match="terminal response"):
        consume([added(tool)])


def test_successful_terminal_still_settles_an_announced_function_call():
    tool = {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "terminal", "arguments": "{}"}
    response, _, _, _ = consume([added(tool), terminal()])
    result, finish = _normalize_codex_response(response)
    assert finish == "tool_calls"
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].function.arguments == "{}"


def test_soft_incomplete_envelope_with_real_final_keeps_azure_compatibility():
    response, _, _, _ = consume([done(msg(text="Complete answer", phase="final_answer")), terminal(kind="completed", output=None, status="incomplete")])
    assert _normalize_codex_response(response)[1] == "stop"


def test_reasoning_callback_remains_separate():
    _, text, commentary, reasoning = consume([
        added(msg(text="", phase="analysis")), delta("Internal summary."),
        done(msg(text="Internal summary.", phase="analysis")), terminal(),
    ])
    assert reasoning == ["Internal summary."]
    assert text == []
    assert commentary == []



def test_legacy_unidentified_done_is_not_duplicated_by_terminal_snapshot():
    item = msg(text="Complete answer", phase="final_answer")
    item.pop("id")
    response, _, _, _ = consume([
        {"type": "response.output_item.done", "item": item}, terminal([item]),
    ])
    result, finish = _normalize_codex_response(response)
    assert finish == "stop"
    assert result.content == "Complete answer"
    assert len(result.codex_message_items) == 1


def test_index_only_message_identity_merges_terminal_snapshot_once():
    item = msg()
    item.pop("id")
    response, _, commentary, _ = consume([added(item), done(item), terminal([item])])
    result, finish = _normalize_codex_response(response, issuer_kind="codex_backend")
    assert finish == "incomplete"
    assert len(result.codex_message_items) == 1
    assert commentary == ["Checking."]


def test_terminal_only_unidentified_final_can_be_recovered():
    item = msg(text="42", phase="final_answer")
    item.pop("id")
    response, _, _, _ = consume([terminal([item])])
    result, finish = _normalize_codex_response(response)
    assert finish == "stop"
    assert result.content == "42"
    assert len(result.codex_message_items) == 1


def test_terminal_backfill_does_not_promote_a_user_message():
    answer = msg(text="42", phase="final_answer")
    alien = msg("msg_user", "Not an assistant answer", "final_answer")
    alien["role"] = "user"
    response, _, _, _ = consume([done(answer), terminal([answer, alien])])
    result, finish = _normalize_codex_response(response)
    assert finish == "stop"
    assert result.content == "42"
    assert len(result.codex_message_items) == 1


def test_real_sdk_message_objects_keep_phase_without_mutation():
    from openai.types.responses import ResponseOutputMessage, ResponseOutputText

    part = ResponseOutputText.model_construct(type="output_text", text="Checking.", annotations=[])
    announced = ResponseOutputMessage.model_construct(type="message", role="assistant", id="msg_sdk", status="in_progress", content=[], phase="commentary")
    completed = ResponseOutputMessage.model_construct(type="message", role="assistant", id="msg_sdk", status="completed", content=[part])
    before = completed.model_dump()
    commentary = []
    response = _consume_codex_event_stream(iter([
        NS(type="response.output_item.added", output_index=0, item=announced),
        NS(type="response.output_text.delta", item_id="msg_sdk", output_index=0, delta="Checking."),
        NS(type="response.output_item.done", output_index=0, item=completed),
        NS(type="response.completed", response=NS(status="completed")),
    ]), model="test-model", on_commentary_message=commentary.append)
    result, finish = _normalize_codex_response(response, issuer_kind="codex_backend")
    assert finish == "incomplete"
    assert result.codex_message_items[0]["phase"] == "commentary"
    assert commentary == ["Checking."]
    assert completed.model_dump() == before

# This test must pass on the unmodified codecrc (before the fix) and document
# the exact regression: a single Responses function_call surfaced as two
# same-id items (one with arguments, one with empty ``{}``) was executed twice,
# emitting an empty-argument tool call that ran with default/no args.
"""Regression: same-id function_call must yield exactly one tool call.

luna/sol responses can announce a function_call through multiple same-id events
(a bare ``output_item.added`` with empty arguments followed by a completed
``output_item.done``, or two ``output_item.done`` frames for the same id, one
empty).  Hermes previously kept every same-id copy, so one logical tool call
became two ``tool_calls`` entries — the empty-argument twin reaching the tool
executor (e.g. ``read_file`` on the workdir) and erroring.

This test pins the assembler to emit one item per call id, preferring the copy
that carries non-empty arguments.
"""

from types import SimpleNamespace

import pytest

from agent.codex_responses_adapter import _normalize_codex_response
from agent.codex_runtime import _CodexResponseAssembler


def _ev(t: str, **kw) -> SimpleNamespace:
    return SimpleNamespace(type=t, **kw)


def _assembler():
    return _CodexResponseAssembler(
        model="gpt-5.6-luna",
        on_text_delta=None,
        on_reasoning_delta=None,
        on_commentary_message=None,
        on_first_delta=None,
    )


def _finish(a: _CodexResponseAssembler):
    a.feed(_ev("response.completed", response=SimpleNamespace(id="r", usage=None, status="completed")))
    return a.result().output


def _calls(output):
    return [
        (getattr(it, "id", None), getattr(it, "arguments", None))
        for it in output
        if "function_call" in str(getattr(it, "type", ""))
    ]


def test_done_plus_empty_added_same_id_single_call():
    a = _assembler()
    a.feed(_ev("response.output_item.added",
               item=SimpleNamespace(type="function_call", id="fc_1", name="x", arguments=""), output_index=0))
    a.feed(_ev("response.function_call_arguments.delta", item_id="fc_1", delta='{"a":1}'))
    a.feed(_ev("response.output_item.done",
               item=SimpleNamespace(type="function_call", id="fc_1", name="x", arguments='{"a":1}'), output_index=0))
    calls = _calls(_finish(a))
    assert len(calls) == 1, calls
    assert calls[0] == ("fc_1", '{"a":1}')


def test_two_done_frames_same_id_empty_then_full_single_call():
    a = _assembler()
    a.feed(_ev("response.output_item.done",
               item=SimpleNamespace(type="function_call", id="fc_1", name="x", arguments="{}"), output_index=0))
    a.feed(_ev("response.output_item.done",
               item=SimpleNamespace(type="function_call", id="fc_1", name="x", arguments='{"a":1}'), output_index=1))
    calls = _calls(_finish(a))
    assert len(calls) == 1, calls
    assert calls[0] == ("fc_1", '{"a":1}')


def test_two_done_frames_same_id_distinct_calls_preserved():
    """Different ids (and genuinely different calls) are never merged."""
    a = _assembler()
    a.feed(_ev("response.output_item.done",
               item=SimpleNamespace(type="function_call", id="fc_1", name="x", arguments='{"a":1}'), output_index=0))
    a.feed(_ev("response.output_item.done",
               item=SimpleNamespace(type="function_call", id="fc_2", name="y", arguments='{"b":2}'), output_index=1))
    calls = _calls(_finish(a))
    assert len(calls) == 2, calls


def test_all_empty_duplicates_keeps_first():
    a = _assembler()
    a.feed(_ev("response.output_item.done",
               item=SimpleNamespace(type="function_call", id="fc_1", name="x", arguments="{}"), output_index=0))
    a.feed(_ev("response.output_item.done",
               item=SimpleNamespace(type="function_call", id="fc_1", name="x", arguments="{}"), output_index=1))
    calls = _calls(_finish(a))
    assert len(calls) == 1, calls
    assert calls[0][0] == "fc_1"


def test_message_items_untouched():
    a = _assembler()
    a.feed(_ev("response.output_item.added",
               item=SimpleNamespace(type="function_call", id="fc_1", name="x", arguments=""), output_index=0))
    a.feed(_ev("response.output_item.done",
               item=SimpleNamespace(type="function_call", id="fc_1", name="x", arguments='{"a":1}'), output_index=0))
    a.feed(_ev("response.output_item.done",
               item=SimpleNamespace(type="message", role="assistant", status="completed",
                                    content=[SimpleNamespace(type="output_text", text="hi")]), output_index=1))
    out = _finish(a)
    kinds = [str(getattr(it, "type", "")) for it in out]
    assert kinds.count("message") == 1
    assert len(_calls(out)) == 1


# --- Non-streamed normalize path (agent/transports/codex.py -> _normalize_codex_response) ---
#
# The same provider behavior surfaces identically on the non-streamed route: one logical
# function_call arrives as two same-id ``response.output`` items (a ``{}`` twin plus the real
# one).  _OutputScan.scan would emit one tool_call per item; the dedupe collapses them.


def _resp(items):
    return SimpleNamespace(id="r", status="completed", output=items, usage=None, error=None, output_text=None)


def test_normalize_empty_then_full_same_id_single_call():
    empty = SimpleNamespace(type="function_call", id="fc_1", call_id="call_1", name="read_file",
                            status="completed", arguments="{}")
    full = SimpleNamespace(type="function_call", id="fc_1", call_id="call_1", name="read_file",
                           status="completed", arguments='{"path":"x"}')
    msg, _ = _normalize_codex_response(_resp([empty, full]))
    calls = [(getattr(t, "id", None), getattr(getattr(t, "function", None), "arguments", None))
             for t in msg.tool_calls]
    assert len(calls) == 1, calls
    assert calls[0] == ("call_1", '{"path":"x"}')  # keeps the populated twin


def test_normalize_full_then_empty_same_id_keeps_full():
    empty = SimpleNamespace(type="function_call", id="fc_1", call_id="call_1", name="read_file",
                            status="completed", arguments="{}")
    full = SimpleNamespace(type="function_call", id="fc_1", call_id="call_1", name="read_file",
                           status="completed", arguments='{"path":"x"}')
    msg, _ = _normalize_codex_response(_resp([full, empty]))
    calls = [(getattr(t, "id", None), getattr(getattr(t, "function", None), "arguments", None))
             for t in msg.tool_calls]
    assert len(calls) == 1, calls
    assert calls[0] == ("call_1", '{"path":"x"}')


def test_normalize_distinct_calls_preserved():
    fc1 = SimpleNamespace(type="function_call", id="fc_1", call_id="call_1", name="a",
                          status="completed", arguments='{"k":1}')
    fc2 = SimpleNamespace(type="function_call", id="fc_2", call_id="call_2", name="b",
                          status="completed", arguments='{"k":2}')
    msg, _ = _normalize_codex_response(_resp([fc1, fc2]))
    assert len(msg.tool_calls) == 2


def _normalized_calls(items, streamed):
    if streamed:
        assembler = _assembler()
        for index, item in enumerate(items):
            assembler.feed(_ev("response.output_item.done", item=item, output_index=index))
        items = _finish(assembler)
    message, finish_reason = _normalize_codex_response(_resp(items))
    assert finish_reason == "tool_calls"
    return [(call.id, call.function.name, call.function.arguments) for call in message.tool_calls]


@pytest.mark.parametrize("streamed", [False, True], ids=["nonstreamed", "streamed"])
@pytest.mark.parametrize("second_arguments", ['{"path":"second"}', "{}"])
def test_distinct_call_ids_survive_reused_item_id(streamed, second_arguments):
    items = [
        SimpleNamespace(type="function_call", id="fc_shared", call_id="call_a",
                        name="read_file", arguments='{"path":"first"}'),
        SimpleNamespace(type="function_call", id="fc_shared", call_id="call_b",
                        name="read_file", arguments=second_arguments),
    ]
    assert _normalized_calls(items, streamed) == [
        ("call_a", "read_file", '{"path":"first"}'),
        ("call_b", "read_file", second_arguments),
    ]


@pytest.mark.parametrize("streamed", [False, True], ids=["nonstreamed", "streamed"])
@pytest.mark.parametrize("reverse", [False, True], ids=["full-first", "empty-first"])
@pytest.mark.parametrize("second_fields,collapse", [
    pytest.param({"id": "fc_rotated"}, True, id="stable-call-id"),
    pytest.param({"name": "skills_list"}, False, id="different-tool-name"),
    pytest.param({"call_id": "call_b"}, False, id="different-call-id"),
    pytest.param({"call_id": None, "id": "call_a"}, False, id="separate-id-namespaces"),
])
def test_only_matching_tool_identity_loses_empty_twin(streamed, reverse, second_fields, collapse):
    full = SimpleNamespace(type="function_call", id="fc_shared", call_id="call_a",
                           name="read_file", arguments='{"path":"first"}')
    empty = SimpleNamespace(**{
        "type": "function_call", "id": "fc_shared", "call_id": "call_a",
        "name": "read_file", "arguments": "{}", **second_fields,
    })
    items = [empty, full] if reverse else [full, empty]
    expected = [("call_a", "read_file", full.arguments)]
    if not collapse:
        empty_message, _ = _normalize_codex_response(_resp([empty]))
        empty_call = empty_message.tool_calls[0]
        empty_result = (empty_call.id, empty_call.function.name, "{}")
        expected = [empty_result, *expected] if reverse else [*expected, empty_result]
    assert _normalized_calls(items, streamed) == expected


@pytest.mark.parametrize("streamed", [False, True], ids=["nonstreamed", "streamed"])
@pytest.mark.parametrize("reverse", [False, True], ids=["incomplete-first", "completed-first"])
@pytest.mark.parametrize("status", ["queued", "in_progress", "incomplete", " IN_PROGRESS "])
@pytest.mark.parametrize("arguments", ['{"path":"unfinished"}', "{}"])
def test_incomplete_twin_cannot_suppress_completed_call(streamed, reverse, status, arguments):
    incomplete = SimpleNamespace(type="function_call", id="fc_shared", call_id="call_a",
                                 name="read_file", status=status, arguments=arguments)
    completed = SimpleNamespace(type="function_call", id="fc_shared", call_id="call_a",
                                name="read_file", status="completed", arguments="{}")
    items = [completed, incomplete] if reverse else [incomplete, completed]
    assert _normalized_calls(items, streamed) == [("call_a", "read_file", "{}")]


@pytest.mark.parametrize("streamed", [False, True], ids=["nonstreamed", "streamed"])
@pytest.mark.parametrize("reverse", [False, True], ids=["dict-first", "object-first"])
@pytest.mark.parametrize("dict_arguments,object_arguments", [
    ('{"path":"dict"}', "{}"),
    ("{}", "{}"),
    ("{}", '{"path":"object"}'),
])
def test_dict_twin_cannot_suppress_object_call(streamed, reverse, dict_arguments, object_arguments):
    dictionary = dict(type="function_call", id="fc_shared", call_id="call_a",
                      name="read_file", status="completed", arguments=dict_arguments)
    item = SimpleNamespace(type="function_call", id="fc_shared", call_id="call_a",
                           name="read_file", status="completed", arguments=object_arguments)
    items = [item, dictionary] if reverse else [dictionary, item]
    assert _normalized_calls(items, streamed) == [("call_a", "read_file", object_arguments)]

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
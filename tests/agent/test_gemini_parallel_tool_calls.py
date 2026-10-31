"""Parallel same-name tool calls must stay separate calls when streamed.

Regression: Gemini sends every ``functionCall`` in its own SSE event, usually
as ``parts[0]``, so three ``read_file`` calls shared
``(part_index, name, thought_signature)``. The stream translator keyed its
tool-call slots on that tuple, so calls 2 and 3 were treated as continuations of
call 1 and their arguments were appended to it, producing one tool call whose
arguments were three glued JSON objects::

    {"path": "a"}{"path": "b"}{"path": "c"}

Nothing can parse that, so the sanitizer replaced the arguments with ``{}``,
the loop read that as a truncated tool call, retried 4 times and gave up with
"Response truncated due to output length limit" — three file reads lost and a
misleading error shown to the user.
"""

from __future__ import annotations

import json
from typing import Any

from agent.gemini_native_adapter import translate_stream_event

MODEL = "gemini-2.5-flash"


def _event(*parts: dict, finish: str = "") -> dict:
    candidate: dict[str, Any] = {"content": {"parts": list(parts)}}
    if finish:
        candidate["finishReason"] = finish
    return {"candidates": [candidate]}


def _tool_deltas(chunks):
    out = []
    for chunk in chunks:
        for choice in chunk.choices:
            for delta in choice.delta.tool_calls or []:
                out.append(delta)
    return out


def test_parallel_same_name_calls_become_separate_tool_calls():
    slots: dict = {}
    paths = ["/a/billing-invoice-email.ts", "/a/billing-mail.ts", "/a/email-copy.ts"]
    deltas = []
    for path in paths:
        deltas.extend(
            _tool_deltas(
                translate_stream_event(
                    _event({"functionCall": {"name": "read_file", "args": {"path": path}}}),
                    model=MODEL,
                    tool_call_indices=slots,
                )
            )
        )

    assert [d.index for d in deltas] == [0, 1, 2]
    assert len({d.id for d in deltas}) == 3
    assert [json.loads(d.function.arguments)["path"] for d in deltas] == paths
    # Every argument payload must be a single, complete JSON document — never
    # two objects glued together.
    for delta in deltas:
        assert not delta.function.arguments.endswith("}{")


def test_prefix_sharing_but_distinct_calls_are_not_glued():
    slots: dict = {}
    deltas = []
    for path in ("/a", "/a/b"):
        deltas.extend(
            _tool_deltas(
                translate_stream_event(
                    _event({"functionCall": {"name": "read_file", "args": {"path": path}}}),
                    model=MODEL,
                    tool_call_indices=slots,
                )
            )
        )

    assert [d.index for d in deltas] == [0, 1]
    assert json.loads(deltas[1].function.arguments) == {"path": "/a/b"}


def test_parallel_calls_finish_reason_still_reports_tool_calls():
    slots: dict = {}
    translate_stream_event(
        _event({"functionCall": {"name": "read_file", "args": {"path": "/a"}}}),
        model=MODEL,
        tool_call_indices=slots,
    )
    chunks = translate_stream_event(
        _event(
            {"functionCall": {"name": "read_file", "args": {"path": "/b"}}},
            finish="STOP",
        ),
        model=MODEL,
        tool_call_indices=slots,
    )
    assert chunks[-1].choices[0].finish_reason == "tool_calls"


def test_repeated_identical_call_event_is_still_deduped():
    slots: dict = {}
    event = _event({"functionCall": {"name": "read_file", "args": {"path": "/a"}}})
    first = _tool_deltas(translate_stream_event(event, model=MODEL, tool_call_indices=slots))
    second = _tool_deltas(translate_stream_event(event, model=MODEL, tool_call_indices=slots))

    assert first[0].index == second[0].index == 0
    assert first[0].id == second[0].id
    assert second[0].function.arguments == ""

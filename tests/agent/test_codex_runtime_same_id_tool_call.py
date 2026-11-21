"""Regression: one Responses function call surfaced as two same-identity items runs once.

Copilot Responses backends (gpt-5.6-luna / gpt-5.6-sol) can emit one logical call as a
populated item plus an empty ``{}`` twin with the same call identity: in a non-streamed
``response.output`` or as repeated ``output_item.done`` frames. Hermes executed both and
the empty twin failed (``read_file`` on the workdir). Rotating stream item ids and stale
settled announcements are fixed in the assembler by #135285
(tests/agent/test_codex_stream_rotating_item_ids.py); this file pins the shared normalizer
that both routes reach through ``_normalize_codex_response``.
"""

import subprocess
import sys
import textwrap
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


def _normalized_message(items, streamed):
    if streamed:
        assembler = _assembler()
        for index, item in enumerate(items):
            assembler.feed(_ev("response.output_item.done", item=item, output_index=index))
        items = _finish(assembler)
    message, finish_reason = _normalize_codex_response(_resp(items))
    assert finish_reason == "tool_calls"
    return message


def _normalized_calls(items, streamed):
    message = _normalized_message(items, streamed)
    return [(call.id, call.function.name, call.function.arguments) for call in message.tool_calls]


@pytest.mark.parametrize("streamed", [False, True], ids=["nonstreamed", "streamed"])
@pytest.mark.parametrize("arguments,expected", [
    (("{}", '{"a":1}'), '{"a":1}'),
    (('{"a":1}', "{}"), '{"a":1}'),
    (("{}", "{}"), "{}"),
], ids=["empty-then-full", "full-then-empty", "all-empty"])
def test_repeated_frames_without_call_id_collapse(streamed, arguments, expected):
    items = [SimpleNamespace(type="function_call", id="fc_1", name="x", status="completed", arguments=value)
             for value in arguments]
    assert [(name, args) for _, name, args in _normalized_calls(items, streamed)] == [("x", expected)]


@pytest.mark.parametrize("streamed", [False, True], ids=["nonstreamed", "streamed"])
@pytest.mark.parametrize("call_ids,item_ids", [
    (("call_a",), ("fc_shared",)),
    (("call_a", "call_b"), ("fc_shared", "fc_shared")),
    (("call_a", "call_b", "call_c"), ("fc_shared", "fc_shared", "fc_a")),
    (("call_a", "call_b", "call_c"), ("fc_shared", "fc_shared", "fc_a|fc_c")),
], ids=["single", "shared-alias", "derived-alias-collision", "expanded-alias-collision"])
def test_call_identities_survive_history_and_next_request(streamed, call_ids, item_ids):
    from agent.agent_runtime_helpers import sanitize_api_messages
    from agent.chat_completion_helpers import _assistant_tool_call_dict
    from agent.codex_responses_adapter import _chat_messages_to_responses_input
    from run_agent import AIAgent

    items = [SimpleNamespace(type="function_call", id=item_id,
                             call_id=call_id, name="read_file", arguments='{"path":"' + call_id + '"}')
             for call_id, item_id in zip(call_ids, item_ids)]
    # A genuine repeated frame must still collapse before storage.
    message = _normalized_message([*items, items[0]], streamed)
    stored_calls = [_assistant_tool_call_dict(AIAgent, call, index)
                    for index, call in enumerate(message.tool_calls)]
    history = [
        {"role": "user", "content": "Read the files"},
        {"role": "assistant", "content": "", "tool_calls": stored_calls},
        *[{"role": "tool", "tool_call_id": call_id, "name": "read_file", "content": call_id}
          for call_id in call_ids],
    ]
    sanitized = sanitize_api_messages(history)
    assert [tc["id"] for msg in sanitized for tc in msg.get("tool_calls", [])] == list(call_ids)
    assert [msg["tool_call_id"] for msg in sanitized if msg["role"] == "tool"] == list(call_ids)
    if len(call_ids) == 1:
        assert stored_calls[0]["response_item_id"] == "fc_shared"
    wire = _chat_messages_to_responses_input(sanitized)
    calls = [item for item in wire if item.get("type") == "function_call"]
    outputs = [item for item in wire if item.get("type") == "function_call_output"]
    assert [item["call_id"] for item in calls] == list(call_ids)
    assert [item["call_id"] for item in outputs] == list(call_ids)
    assert len({call["response_item_id"] for call in stored_calls}) == len(call_ids)
    assert [item["arguments"] for item in calls] == [item.arguments for item in items]
    assert [item["output"] for item in outputs] == list(call_ids)


@pytest.mark.parametrize("streamed", [False, True], ids=["nonstreamed", "streamed"])
def test_expanded_call_id_alias_search_terminates_through_next_request(streamed):
    # Bound the real pipeline: an alias retaining '|fc_x' can reserve the same
    # expanded prefix forever, even as its numeric suffix grows.
    code = textwrap.dedent("""
        import runpy
        import sys
        from types import SimpleNamespace
        from agent.agent_runtime_helpers import sanitize_api_messages
        from agent.chat_completion_helpers import _assistant_tool_call_dict
        from agent.codex_responses_adapter import _chat_messages_to_responses_input
        from run_agent import AIAgent

        helpers = runpy.run_path(sys.argv[1])
        call_ids = ['call_a|fc_x', 'call_b', 'call_c']
        item_ids = ['fc_shared', 'fc_shared', 'fc_a']
        items = [SimpleNamespace(type='function_call', id=item_id, call_id=call_id,
                                 name='read_file', arguments='{"path":"' + call_id + '"}')
                 for call_id, item_id in zip(call_ids, item_ids)]
        message = helpers['_normalized_message'](items, sys.argv[2] == 'True')
        assert [call.id for call in message.tool_calls] == call_ids
        stored = [_assistant_tool_call_dict(AIAgent, call, index)
                  for index, call in enumerate(message.tool_calls)]
        assert stored[2]['response_item_id'] == 'fc_a'
        history = [
            {'role': 'user', 'content': 'Read the files'},
            {'role': 'assistant', 'content': '', 'tool_calls': stored},
            *[{'role': 'tool', 'tool_call_id': call_id, 'name': 'read_file', 'content': call_id}
              for call_id in call_ids],
        ]
        sanitized = sanitize_api_messages(history)
        assert [tc['id'] for msg in sanitized for tc in msg.get('tool_calls', [])] == call_ids
        wire = _chat_messages_to_responses_input(sanitized)
        calls = [item for item in wire if item.get('type') == 'function_call']
        outputs = [item for item in wire if item.get('type') == 'function_call_output']
        assert [item['call_id'] for item in calls] == call_ids
        assert [item['arguments'] for item in calls] == [item.arguments for item in items]
        assert [item['output'] for item in outputs] == call_ids
    """)
    result = subprocess.run(
        [sys.executable, "-c", code, __file__, str(streamed)],
        capture_output=True, text=True, timeout=8, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


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
@pytest.mark.parametrize("item_type", ["function_call", "custom_tool_call"])
@pytest.mark.parametrize("dict_arguments,object_arguments", [
    ('{"path":"dict"}', "{}"),
    ("{}", "{}"),
    ("{}", '{"path":"object"}'),
])
def test_dict_twin_cannot_suppress_object_call(streamed, reverse, item_type, dict_arguments, object_arguments):
    field = "arguments" if item_type == "function_call" else "input"
    dictionary = dict(type=item_type, id="fc_shared", call_id="call_a",
                      name="read_file", status="completed", **{field: dict_arguments})
    item = SimpleNamespace(type=item_type, id="fc_shared", call_id="call_a",
                           name="read_file", status="completed", **{field: object_arguments})
    items = [item, dictionary] if reverse else [dictionary, item]
    assert _normalized_calls(items, streamed) == [("call_a", "read_file", object_arguments)]


@pytest.mark.parametrize("streamed", [False, True], ids=["nonstreamed", "streamed"])
@pytest.mark.parametrize("reverse", [False, True], ids=["full-first", "empty-first"])
@pytest.mark.parametrize("status", [None, "completed", "queued", "in_progress", "incomplete", " IN_PROGRESS "])
@pytest.mark.parametrize("full_input", ['{"path":"custom"}', "{}"])
def test_custom_call_twins_use_input_at_every_scannable_status(streamed, reverse, status, full_input):
    full = SimpleNamespace(type="custom_tool_call", id="fc_shared", call_id="call_a",
                           name="read_file", status=status, input=full_input)
    empty = SimpleNamespace(type="custom_tool_call", id="fc_shared", call_id="call_a",
                            name="read_file", status=status, input="{}")
    other = SimpleNamespace(type="custom_tool_call", id="fc_other", call_id="call_b",
                            name="skills_list", status="completed", input="{}")
    items = [empty, other, full] if reverse else [full, other, empty]
    assert _normalized_calls(items, streamed) == [
        ("call_a", "read_file", full_input), ("call_b", "skills_list", "{}"),
    ]


@pytest.mark.parametrize("streamed", [False, True], ids=["nonstreamed", "streamed"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("status", ["queued", "in_progress", "incomplete", " IN_PROGRESS "])
def test_ignored_function_cannot_suppress_custom_call(streamed, reverse, status):
    ignored = SimpleNamespace(type="function_call", id="fc_shared", call_id="call_a",
                              name="read_file", status=status, arguments='{"path":"ignored"}')
    custom = SimpleNamespace(type="custom_tool_call", id="fc_shared", call_id="call_a",
                             name="read_file", status=status, input="{}")
    items = [custom, ignored] if reverse else [ignored, custom]
    assert _normalized_calls(items, streamed) == [("call_a", "read_file", "{}")]


@pytest.mark.parametrize("streamed", [False, True], ids=["nonstreamed", "streamed"])
@pytest.mark.parametrize("reverse", [False, True], ids=["full-first", "empty-first"])
@pytest.mark.parametrize("item_type", ["function_call", "custom_tool_call"])
@pytest.mark.parametrize("identity_field", ["call_id", "id"])
@pytest.mark.parametrize("padded_twin", ["full", "empty"])
def test_whitespace_identity_twins_collapse(streamed, reverse, item_type, identity_field, padded_twin):
    field = "arguments" if item_type == "function_call" else "input"
    identity = "call_a" if identity_field == "call_id" else "item_a"
    common = {"type": item_type, "id": "fc_shared", "name": "read_file", identity_field: identity}
    full = SimpleNamespace(**{**common, field: '{"path":"full"}'})
    empty = SimpleNamespace(**{**common, field: "{}"})
    setattr(full if padded_twin == "full" else empty, identity_field, f" \t{identity}\n")
    items = [empty, full] if reverse else [full, empty]
    assert _normalized_calls(items, streamed) == [(identity, "read_file", '{"path":"full"}')]
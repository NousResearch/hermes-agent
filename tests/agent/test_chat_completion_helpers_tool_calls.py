"""Slot assignment of streamed tool-call deltas whose id changes at a reused index."""

from types import SimpleNamespace

from agent.chat_completion_helpers_tool_calls import _ToolCallAccumulator


def _calls(*deltas):
    acc = _ToolCallAccumulator()
    for tc_id, name, args in deltas:
        acc.feed(SimpleNamespace(index=0, id=tc_id, function=SimpleNamespace(name=name, arguments=args)))
    return [(e["function"]["name"], e["function"]["arguments"]) for _, e in sorted(acc.materialize().items())]


def test_nameless_fragments_with_fresh_ids_stay_one_call():
    # LM Studio / Kimi: fresh id on every argument fragment, name only on the first (#5254).
    assert _calls(
        ("701601222", "terminal", ""),
        ("tool-call-1", None, '{"com'),
        ("tool-call-2", None, 'mand": '),
        ("tool-call-3", None, '"ls"}'),
    ) == [("terminal", '{"command": "ls"}')]


def test_same_name_fresh_id_splits_only_after_complete_arguments():
    # A name resent on each fragment continues the call (MiniMax); a same-name delta after
    # complete arguments is a second parallel call (Ollama); a different name always splits.
    assert _calls(
        ("mm-1", "terminal", '{"command": '),
        ("mm-2", "terminal", '"ls"}'),
        ("mm-3", "terminal", '{"command": "pwd"}'),
        ("mm-4", "read_file", "{}"),
    ) == [("terminal", '{"command": "ls"}'), ("terminal", '{"command": "pwd"}'), ("read_file", "{}")]

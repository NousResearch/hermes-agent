"""Repeated tool call_id across turns must not replay as duplicate (400).

A session that ran the same tool on two separate turns can store the same
tool ``call_id`` twice (observed as ``terminal:0`` on both turns). Replaying
both pairs verbatim makes the Responses payload carry the same ``call_id``
twice and the provider rejects the whole request with::

    400 Duplicate function_call_output for call_id 'terminal:0'

The converter must uniquify repeats per occurrence while keeping each
``function_call`` paired with its own ``function_call_output``.
"""

from collections import Counter

from agent.codex_responses_adapter import _chat_messages_to_responses_input


def _two_turn_duplicate_messages(call_id="terminal:0"):
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "call_id": call_id,
                    "function": {"name": "terminal", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": call_id, "content": "first result"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "call_id": call_id,
                    "function": {"name": "terminal", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": call_id, "content": "second result"},
    ]


def test_repeated_call_id_replays_unique_and_paired():
    """Each replayed call_id appears exactly twice: one call + one output."""
    items = _chat_messages_to_responses_input(
        _two_turn_duplicate_messages("terminal:0")
    )

    calls = [i["call_id"] for i in items if i.get("type") == "function_call"]
    outputs = [i["call_id"] for i in items if i.get("type") == "function_call_output"]

    assert len(calls) == 2
    assert len(outputs) == 2
    # First occurrence keeps the stored id (prompt-cache prefix safe).
    assert calls[0] == "terminal:0"
    assert outputs[0] == "terminal:0"
    # The repeat must be uniquified, and the second pair must still match.
    assert calls[1] == outputs[1]
    assert calls[1] != calls[0]
    # Wire invariant: each call_id has exactly one call + one output.
    assert Counter(calls + outputs)[calls[0]] == 2
    assert Counter(calls + outputs)[calls[1]] == 2
    # Dense suffix sequence: the repeat is the second occurrence, so it
    # takes `__r2` — not a skipped `__r3`.
    assert calls[1] == "terminal:0__r2"


def test_natural_suffixed_id_collision_stays_unique():
    """A stored pair whose real id is already `terminal:0__r2` must not
    collide with the generated suffix for the second `terminal:0` repeat
    (andrexibiza review on #102640): the fresh-id search must skip occupied
    candidates until it finds one actually free, in bounded-range order."""
    messages = [
        # Natural pair: real stored id happens to be terminal:0__r2.
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "call_id": "terminal:0__r2",
                    "function": {"name": "terminal", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "terminal:0__r2", "content": "natural"},
        # First terminal:0 pair.
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "call_id": "terminal:0",
                    "function": {"name": "terminal", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "terminal:0", "content": "first"},
        # Second terminal:0 pair — its generated suffix must not land on
        # the natural terminal:0__r2 already on the wire.
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "call_id": "terminal:0",
                    "function": {"name": "terminal", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "terminal:0", "content": "second"},
    ]

    items = _chat_messages_to_responses_input(messages)

    calls = [i["call_id"] for i in items if i.get("type") == "function_call"]
    outputs = [
        i["call_id"] for i in items if i.get("type") == "function_call_output"
    ]

    assert len(calls) == len(outputs) == 3
    # No duplicate wire ids — the generated repeat skipped the occupied
    # natural suffix instead of colliding with it.
    assert len(set(calls)) == 3
    # Pairs still matched.
    assert calls == outputs
    # The natural pair keeps its exact stored id.
    assert "terminal:0__r2" in calls
    assert calls.count("terminal:0__r2") == 1


def test_out_of_order_output_pairs_with_later_call():
    """A tool result stored BEFORE its assistant call (imported/foreign
    history) must pair with that later call, not collide with it."""
    messages = [
        {"role": "tool", "tool_call_id": "terminal:0", "content": "early"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "call_id": "terminal:0",
                    "function": {"name": "terminal", "arguments": "{}"},
                }
            ],
        },
    ]

    items = _chat_messages_to_responses_input(messages)

    calls = [i["call_id"] for i in items if i.get("type") == "function_call"]
    outputs = [
        i["call_id"] for i in items if i.get("type") == "function_call_output"
    ]

    assert len(calls) == len(outputs) == 1
    assert calls[0] == outputs[0] == "terminal:0"


def test_repeated_call_id_outputs_stay_in_order():
    """Nth output pairs with Nth call even with interleaved user text."""
    messages = _two_turn_duplicate_messages("terminal:0")
    messages.insert(2, {"role": "user", "content": "keep going"})

    items = _chat_messages_to_responses_input(messages)

    calls = [i["call_id"] for i in items if i.get("type") == "function_call"]
    outputs = [i["call_id"] for i in items if i.get("type") == "function_call_output"]

    assert len(calls) == len(outputs) == 2
    assert calls[0] == outputs[0]
    assert calls[1] == outputs[1]
    assert calls[0] != calls[1]

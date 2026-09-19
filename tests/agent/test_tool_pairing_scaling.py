"""Bound pairing work by history size without wall-clock timing assertions."""

import pytest

from agent import agent_runtime_helpers as helpers


def _exchange(index):
    call = {
        "id": f"item_{index}",
        "call_id": f"call_{index}",
        "response_item_id": f"fc_{index}",
        "type": "function",
        "function": {"name": "read_file", "arguments": "{}"},
    }
    return [
        {"role": "assistant", "content": "", "tool_calls": [call]},
        {"role": "tool", "tool_call_id": f"call_{index}|fc_{index}", "content": "ok"},
    ]


@pytest.mark.parametrize("pair_count", [64, 256])
def test_positional_pairing_bounds_history_reads(pair_count):
    class CountedHistory(list):
        items_read = 0

        def __iter__(self):
            for item in super().__iter__():
                self.items_read += 1
                yield item

        def __getitem__(self, key):
            value = super().__getitem__(key)
            # A slice copies every selected reference before the caller can break.
            self.items_read += len(value) if isinstance(key, slice) else 1
            return value

    paired = [msg for i in range(pair_count) for msg in _exchange(i)]
    displaced_call, displaced_result = _exchange("displaced")
    boundary = {"role": "user", "content": "next turn"}
    messages = CountedHistory(paired + [displaced_call, boundary, displaced_result])

    repaired, repairs = helpers._prune_unanswered_tool_calls(messages)

    assert messages.items_read <= 8 * len(messages)
    assert repairs == 1
    expected = paired + [boundary, displaced_result]
    assert repaired == expected
    assert all(actual is original for actual, original in zip(repaired, expected))


@pytest.mark.parametrize("pair_count", [64, 256])
def test_global_pairing_bounds_id_comparisons(pair_count, monkeypatch):
    intersections = 0
    original_variants = helpers.tool_call_id_variants

    class CountedVariants(frozenset):
        def __and__(self, other):
            nonlocal intersections
            intersections += 1
            return super().__and__(other)

    monkeypatch.setattr(
        helpers, "tool_call_id_variants",
        lambda call: CountedVariants(original_variants(call)),
    )
    messages = [msg for i in range(pair_count) for msg in _exchange(i)]
    unanswered, _ = _exchange("missing")
    missing_call = unanswered["tool_calls"][0]
    orphan = {"role": "tool", "tool_call_id": "orphan", "content": "unmatched"}
    messages.extend([unanswered, orphan])

    call_ids, result_ids, orphaned, missing = helpers._classify_tool_call_orphans(messages)

    assert intersections <= 4 * (pair_count + 1)
    assert orphaned == [orphan] and orphaned[0] is orphan
    assert missing == [missing_call] and missing[0] is missing_call
    assert call_ids == set().union(*(
        original_variants(call)
        for message in messages for call in message.get("tool_calls", [])
    ))
    assert result_ids == set().union(*(
        helpers.tool_result_id_variants(message["tool_call_id"])
        for message in messages if message["role"] == "tool"
    ))

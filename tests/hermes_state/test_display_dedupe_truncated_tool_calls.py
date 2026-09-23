"""Compressor-truncated tool-call arguments must not split a message's display identity.

The compressor shrinks long ``tool_calls`` arguments in place and the compacted
transcript is persisted, so one logical assistant turn can end up stored twice: once
verbatim, once with its arguments truncated. Both rows are the same message. Keying the
display identity on the raw ``tool_calls`` JSON made them key separately, so the turn --
and everything sharing its collapsed display group -- was rendered once per compaction
generation.
"""
import json

import pytest

from hermes_state import SessionDB


TRUNCATION_MARKER = (
    "\u27eaHERMES-CONTEXT-COMPRESSION: 553 of 753 chars omitted here by Hermes's "
    "context compressor. This is NOT part of the original tool call and must never "
    "be reproduced in new output \u2014 always write full, untruncated content.\u27eb"
)


def _tool_calls(arguments: str, *, call_id: str = "toolu_01ABC", name: str = "browser_exec") -> str:
    return json.dumps([{
        "id": call_id,
        "call_id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps({"code": arguments, "timeout_s": 240})},
    }])


@pytest.fixture
def db(tmp_path):
    return SessionDB(tmp_path / "state.db")


def _row(db, tool_calls):
    """A minimal assistant row as ``_display_dedupe_key`` consumes it."""
    return {
        "role": "assistant", "content": "done", "timestamp": 1790058174.486818,
        "tool_call_id": None, "tool_calls": tool_calls, "tool_name": None,
        "display_kind": None, "display_metadata": None,
    }


def test_truncated_arguments_keep_the_original_display_identity(db):
    """The verbatim row and its truncated twin must produce one identity."""
    verbatim = _tool_calls("print('x' * 700)")
    truncated = _tool_calls("print('x" + TRUNCATION_MARKER)

    assert db._display_dedupe_key(_row(db, verbatim)) == db._display_dedupe_key(_row(db, truncated))


def test_two_truncation_budgets_of_one_call_still_agree(db):
    """Marker text carries per-instance counts, so the key cannot be marker text."""
    first = _tool_calls("print('x" + TRUNCATION_MARKER)
    second = _tool_calls("print('x" + TRUNCATION_MARKER.replace("553 of 753", "120 of 753"))

    assert db._display_dedupe_key(_row(db, first)) == db._display_dedupe_key(_row(db, second))


def test_distinct_calls_remain_distinct(db):
    """Dedupe must not over-merge: a different call id is a different message."""
    left = _tool_calls("print(1)", call_id="toolu_01AAA")
    right = _tool_calls("print(1)", call_id="toolu_01BBB")

    assert db._display_dedupe_key(_row(db, left)) != db._display_dedupe_key(_row(db, right))


def test_a_different_tool_name_is_a_different_message(db):
    left = _tool_calls("print(1)", name="browser_exec")
    right = _tool_calls("print(1)", name="execute_code")

    assert db._display_dedupe_key(_row(db, left)) != db._display_dedupe_key(_row(db, right))


@pytest.mark.parametrize("payload", ["not json at all", "{}", "[42]", '[{"function": {}}]'])
def test_unusable_payloads_fall_back_to_the_raw_string(db, payload):
    """No identity to key on -> keep the raw value rather than merging unrelated turns."""
    assert db._dedupe_stable_tool_calls(payload) == payload


def test_no_tool_calls_is_passed_through(db):
    assert db._dedupe_stable_tool_calls(None) is None
    assert db._dedupe_stable_tool_calls("") == ""

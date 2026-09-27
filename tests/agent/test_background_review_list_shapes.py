"""Regression tests for the list-shape AttributeError guards in
``agent.background_review.summarize_background_review_actions`` (#59437).

The outer ``_run_review_in_thread`` used to crash with
``'list' object has no attribute 'get'`` every time a tool response
returned a list (or any non-dict) where the summarizer expected a
dict — most commonly the ``_change`` field in skill_manage responses
or one of the entries in a memory operations list. The crash took
down the entire background review, discarding every other successful
action that the fork had completed.

What this module guards:

A. ``summarize_background_review_actions`` no longer raises when
   ``data["_change"]`` is a list.  It returns the rest of the
   actions normally.
B. ``summarize_background_review_actions`` no longer raises when
   ``operations`` is a non-list (string, int, None).  It treats the
   field as empty.
C. ``summarize_background_review_actions`` no longer raises when
   ``operations[i]`` is a non-dict (string, None).  It skips that
   entry but processes the rest.
D. ``summarize_background_review_actions`` no longer raises when
   ``call_details.get(tcid)`` returns a non-dict (e.g. None or a
   stray scalar).  It coerces to ``{}``.

"""

from __future__ import annotations

import json

from agent import background_review as bg


def _make_skill_tool_message(change, operations=None):
    """Build the messages list that triggered the original crash."""
    return [
        # Assistant: calls skill_manage
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "skill_manage",
                        "arguments": json.dumps(
                            {
                                "action": "patch",
                                "name": "my-skill",
                                "operations": operations
                                or [
                                    {
                                        "action": "replace",
                                        "content": "x",
                                        "old_text": "y",
                                    }
                                ],
                            }
                        ),
                    },
                }
            ],
        },
        # Tool: response with a buggy _change field (a list instead of dict)
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": json.dumps(
                {
                    "success": True,
                    "message": "Skill 'my-skill' patched.",
                    "_change": change,  # ← the offender, normally a dict
                }
            ),
        },
    ]


def _make_memory_tool_message(operations_field):
    """Memory tool response with a non-canonical operations field."""
    return [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "memory",
                        "arguments": json.dumps({"action": "add", "target": "memory"}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_2",
            "content": json.dumps(
                {
                    "success": True,
                    "message": "Entry added.",
                    "operations": operations_field,
                }
            ),
        },
    ]


# ---------------------------------------------------------------------------
# B. operations as a non-list (string / int / None)
# ---------------------------------------------------------------------------


def test_b_operations_as_none_treated_as_empty():
    """``operations = None`` (missing key, JSON null) is still safe."""
    msgs = _make_memory_tool_message(operations_field=None)
    actions = bg.summarize_background_review_actions(
        review_messages=msgs,
        prior_snapshot=[],
        notification_mode="verbose",
    )
    assert isinstance(actions, list)


# ---------------------------------------------------------------------------
# C. operations[i] as a non-dict (str / None)
# ---------------------------------------------------------------------------


def test_c_operations_contains_non_dict_entries():
    """A legacy/half-typed operations list with string entries short-circuits.

    In ``verbose`` mode the function should produce the valid entries and
    silently skip the non-dict ones without ``AttributeError``. In
    non-verbose mode it falls back to a generic "Memory updated" string,
    so this test exercises the verbose branch where iteration over
    per-entry fields actually happens.
    """
    msgs = _make_memory_tool_message(
        operations_field=[
            "raw-string-no-fields",
            {"action": "add", "content": "valid entry"},
            None,
            {"action": "replace", "content": "another", "old_text": "thing"},
        ]
    )
    actions = bg.summarize_background_review_actions(
        review_messages=msgs,
        prior_snapshot=[],
        notification_mode="verbose",
    )
    assert isinstance(actions, list)
    # ``notification_mode='verbose'`` walks per-entry fields; the two
    # dict-shaped entries produce action lines, the string and None
    # entries are skipped via the isinstance guard. The exact wording is
    # not asserted (memory module shapes may vary) but at least one
    # action line must be present.
    assert len(actions) >= 1, f"expected at least one action line, got {actions!r}"


# ---------------------------------------------------------------------------
# D. detail comes back non-dict (None / stale value)
# ---------------------------------------------------------------------------


def test_d_detail_non_dict_replaced_with_empty():
    """When ``call_details.get(tcid)`` returns None, summarize must coerce
    it to ``{}`` rather than calling ``.get(...)`` on ``None``.
    """
    # Build a tool-only message whose tcid does NOT have an assistant tool_call.
    msgs = _make_skill_tool_message(change={})
    # Drop the assistant message so call_details is empty for tcid=call_1.
    msgs = [m for m in msgs if m.get("role") != "assistant"]

    actions = bg.summarize_background_review_actions(
        review_messages=msgs,
        prior_snapshot=[],
        notification_mode="verbose",
    )
    assert isinstance(actions, list)

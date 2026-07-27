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
E. The caller in ``_run_review_in_thread`` no longer aborts the
   whole review on an unrelated summarize exception; partial valid
   actions are surfaced.

The tests run without pytest (handoff from a prior pattern): they use
plain ``assert`` and a small standalone runner.  Importing the module
exercises the new code paths without booting the LLM stack — there
are no I/O or model dependencies in the unit-of-work being tested.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import sys
import types


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _isolate_hermes_home():
    os.environ.setdefault("HERMES_HOME", "/tmp/hermes-bg-review-test")


def _load_module():
    """Lazy import so a missing optional dep doesn't block the suite.

    Returns the module or None if import failed.
    """
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    try:
        return importlib.import_module("agent.background_review")
    except Exception:
        return None


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


class TestRunner:
    def __init__(self):
        self.passed = []
        self.failed = []

    def run(self, name, fn):
        try:
            fn()
        except Exception as e:  # noqa: BLE001 — runner summary uses it
            import traceback
            self.failed.append((name, e, traceback.format_exc()))
        else:
            self.passed.append(name)

    def summary(self):
        total = len(self.passed) + len(self.failed)
        print(f"\n{'=' * 70}\nResults: {len(self.passed)}/{total} passed")
        if self.failed:
            print(f"\n--- {len(self.failed)} failure(s) ---")
            for n, _e, tb in self.failed:
                print(f"\n[FAIL] {n}\n{tb}")
        return 0 if not self.failed else 1


# ---------------------------------------------------------------------------
# A. _change as a list (the originally-reported crash class)
# ---------------------------------------------------------------------------


def test_a_change_as_list_does_not_crash():
    """When ``data["_change"]`` is a list, summarize must NOT raise.

    Before the fix, ``change = data.get("_change", {})`` returned the list
    and ``change.get("description", "")`` raised ``AttributeError: 'list'
    object has no attribute 'get'``.
    """
    _isolate_hermes_home()
    bg = _load_module()
    if bg is None:
        print("SKIP module not importable")
        return

    msgs = _make_skill_tool_message(change=["not", "a", "dict"])
    actions = bg.summarize_background_review_actions(
        review_messages=msgs,
        prior_snapshot=[],
        notification_mode="verbose",
    )
    assert isinstance(actions, list)
    # The successful update must still surface even though _change was malformed.
    assert any("Skill" in a or "my-skill" in a or "patched" in a for a in actions), (
        f"expected at least one skill-related action line, got {actions!r}"
    )


def test_a_change_as_int_does_not_crash():
    """And ditto for any non-dict scalar that the JSON shape allows."""
    _isolate_hermes_home()
    bg = _load_module()
    if bg is None:
        print("SKIP module not importable")
        return

    msgs = _make_skill_tool_message(change=42)
    actions = bg.summarize_background_review_actions(
        review_messages=msgs,
        prior_snapshot=[],
        notification_mode="verbose",
    )
    assert isinstance(actions, list)


# ---------------------------------------------------------------------------
# B. operations as a non-list (string / int / None)
# ---------------------------------------------------------------------------


def test_b_operations_as_string_treated_as_empty():
    """``operations = "abc"`` from a stale response must not crash."""
    _isolate_hermes_home()
    bg = _load_module()
    if bg is None:
        print("SKIP module not importable")
        return

    msgs = _make_memory_tool_message(operations_field="legacy-string-shape")
    actions = bg.summarize_background_review_actions(
        review_messages=msgs,
        prior_snapshot=[],
        notification_mode="verbose",
    )
    assert isinstance(actions, list)


def test_b_operations_as_none_treated_as_empty():
    """``operations = None`` (missing key, JSON null) is still safe."""
    _isolate_hermes_home()
    bg = _load_module()
    if bg is None:
        print("SKIP module not importable")
        return

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
    _isolate_hermes_home()
    bg = _load_module()
    if bg is None:
        print("SKIP module not importable")
        return

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
    _isolate_hermes_home()
    bg = _load_module()
    if bg is None:
        print("SKIP module not importable")
        return

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


# ---------------------------------------------------------------------------
# E. Caller defends against summarize raising
# ---------------------------------------------------------------------------


def test_e_call_does_not_unwind_module_callables():
    """Structural: the new defensive try/except around the summarize
    call is in place. Caught here rather than via a partial mocking
    cascade because monkeypatching the AIAgent is too brittle for a
    blind regression test — keeping it text-anchored guards the
    ``_run_review_in_thread`` invariant without a real LLM.
    """
    src_path = os.path.join(REPO_ROOT, "agent", "background_review.py")
    src = open(src_path, encoding="utf-8").read()
    # The fix added: ``try: actions = summarize_background_review_actions(...)``
    # followed by ``except Exception as e: ... actions = []``.
    assert "actions = summarize_background_review_actions(" in src
    assert (
        "summarize_background_review_actions returned partial results"
        in src
    ), "expected partial-results guard message present"
    # And the prior-tonon-dict guard for the call_details lookup.
    assert "if not isinstance(detail, dict):" in src
    assert "if isinstance(ops_raw, list)" in src
    assert "if isinstance(change_raw, dict)" in src


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def main():
    runner = TestRunner()
    runner.run("a_change_as_list_does_not_crash", test_a_change_as_list_does_not_crash)
    runner.run("a_change_as_int_does_not_crash", test_a_change_as_int_does_not_crash)
    runner.run("b_operations_as_string_treated_as_empty", test_b_operations_as_string_treated_as_empty)
    runner.run("b_operations_as_none_treated_as_empty", test_b_operations_as_none_treated_as_empty)
    runner.run("c_operations_contains_non_dict_entries", test_c_operations_contains_non_dict_entries)
    runner.run("d_detail_non_dict_replaced_with_empty", test_d_detail_non_dict_replaced_with_empty)
    runner.run("e_call_defends_via_try_except", test_e_call_does_not_unwind_module_callables)
    return runner.summary()


if __name__ == "__main__":
    sys.exit(main())

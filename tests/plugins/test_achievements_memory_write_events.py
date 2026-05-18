"""Regression coverage for ``memory_write_events`` (issue #26927).

Before the fix, ``memory_write_events`` was computed with
``_count_tool(tool_sequence, "mnemosyne_remember", "memory")`` —
where ``_count_tool`` substring-matches needles against tool names.
The ``"memory"`` needle therefore matched every call to the built-in
``memory`` tool, regardless of whether the model passed
``action=add``, ``action=replace``, or ``action=remove``.  The metric
degenerated into an exact copy of ``memory_events``, and the Memory
Palace achievement always showed the same progress as Memory Keeper.

The fix:

* Count ``mnemosyne_remember`` by name (it's a write-only tool from
  the external Mnemosyne plugin — no args inspection needed).
* For the built-in ``memory`` tool, inspect each tool call's
  ``arguments`` and only count ``action in {"add", "replace"}`` as
  a write.  ``remove`` deletes content rather than persisting new
  content, so it is excluded.
* Drop the broken ``"memory"`` substring needle.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


PLUGIN_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "plugins"
    / "hermes-achievements"
    / "dashboard"
    / "plugin_api.py"
)


@pytest.fixture
def plugin_api():
    spec = importlib.util.spec_from_file_location(
        "plugin_api_memory_write_events", PLUGIN_MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _memory_call(action: str, *, target: str = "memory", content: str = "x"):
    """Build an OpenAI-style tool call with JSON-encoded arguments."""
    import json

    return {
        "function": {
            "name": "memory",
            "arguments": json.dumps(
                {"action": action, "target": target, "content": content}
            ),
        },
    }


def _memory_call_with_dict_args(action: str):
    """Some adapters parse ``arguments`` before storage — handle both."""
    return {
        "function": {"name": "memory"},
        "name": "memory",
        "arguments": {"action": action, "target": "memory", "content": "x"},
    }


# ---------------------------------------------------------------------------
# Core bug: write metric must NOT equal the engagement metric
# ---------------------------------------------------------------------------


def test_read_only_memory_calls_dont_count_as_writes(plugin_api):
    """A session with only ``action=remove`` must produce write=0, events>=1.

    Direct reproduction of #26927: pre-fix, this returned write == events.
    """
    msgs = [
        {"role": "user", "content": "clean up"},
        {
            "role": "assistant",
            "tool_calls": [
                _memory_call("remove"),
                _memory_call("remove"),
            ],
        },
    ]
    snap = plugin_api.analyze_messages("s", "t", msgs)
    assert snap["memory_events"] == 2, (
        "Every memory call must still count toward Memory Keeper "
        "engagement (memory_events is unchanged by the fix)."
    )
    assert snap["memory_write_events"] == 0, (
        "Pure ``remove`` calls produce no writes — this is the core "
        "regression for #26927."
    )


def test_add_and_replace_count_as_writes(plugin_api):
    msgs = [
        {
            "role": "assistant",
            "tool_calls": [
                _memory_call("add"),
                _memory_call("replace"),
                _memory_call("remove"),
            ],
        },
    ]
    snap = plugin_api.analyze_messages("s", "t", msgs)
    assert snap["memory_events"] == 3
    assert snap["memory_write_events"] == 2, (
        "``add`` and ``replace`` are the only write actions; ``remove`` "
        "is a deletion."
    )


def test_dict_shaped_arguments_are_supported(plugin_api):
    """Some adapters parse ``arguments`` into a dict before persistence."""
    msgs = [
        {
            "role": "assistant",
            "tool_calls": [
                _memory_call_with_dict_args("add"),
                _memory_call_with_dict_args("remove"),
            ],
        },
    ]
    snap = plugin_api.analyze_messages("s", "t", msgs)
    assert snap["memory_events"] == 2
    assert snap["memory_write_events"] == 1


def test_case_insensitive_action_value(plugin_api):
    """Models occasionally upper-case the action; tolerate that."""
    msgs = [
        {
            "role": "assistant",
            "tool_calls": [
                _memory_call("ADD"),
                _memory_call("Replace"),
            ],
        },
    ]
    snap = plugin_api.analyze_messages("s", "t", msgs)
    assert snap["memory_write_events"] == 2


def test_missing_or_malformed_arguments_dont_count_as_writes(plugin_api):
    """When we can't tell what the action was, default to non-write.

    Better to under-count than to repeat the #26927 over-count bug.
    """
    msgs = [
        {
            "role": "assistant",
            "tool_calls": [
                # No arguments at all
                {"function": {"name": "memory"}},
                # Bad JSON
                {"function": {"name": "memory", "arguments": "{not valid"}},
                # Non-object JSON
                {"function": {"name": "memory", "arguments": "42"}},
                # Empty string
                {"function": {"name": "memory", "arguments": ""}},
                # Action missing from otherwise-valid args
                {
                    "function": {
                        "name": "memory",
                        "arguments": '{"target": "memory"}',
                    }
                },
            ],
        },
    ]
    snap = plugin_api.analyze_messages("s", "t", msgs)
    assert snap["memory_events"] == 5
    assert snap["memory_write_events"] == 0


def test_mnemosyne_remember_is_still_a_write(plugin_api):
    """The external Mnemosyne plugin uses ``mnemosyne_remember`` as a
    write-only tool name; that part of the heuristic still works."""
    msgs = [
        {
            "role": "assistant",
            "tool_calls": [
                {"function": {"name": "mnemosyne_remember"}},
                {"function": {"name": "mnemosyne_recall"}},
            ],
        },
    ]
    snap = plugin_api.analyze_messages("s", "t", msgs)
    # ``memory_events`` substring-matches "mnemosyne" — both should count.
    assert snap["memory_events"] == 2
    # Only the write-specific name counts as a write.
    assert snap["memory_write_events"] == 1


def test_mixed_memory_and_mnemosyne_writes_are_summed(plugin_api):
    """The two write-tool sources are summed, not max'd or duplicated."""
    msgs = [
        {
            "role": "assistant",
            "tool_calls": [
                _memory_call("add"),
                _memory_call("add"),
                {"function": {"name": "mnemosyne_remember"}},
            ],
        },
    ]
    snap = plugin_api.analyze_messages("s", "t", msgs)
    assert snap["memory_write_events"] == 3


def test_non_memory_tools_never_count_as_memory_writes(plugin_api):
    """Defense in depth: a ``terminal`` call with ``action=add`` in its
    args (unusual but possible) must NOT count as a memory write."""
    msgs = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "terminal",
                        "arguments": '{"action": "add", "command": "ls"}',
                    }
                },
            ],
        },
    ]
    snap = plugin_api.analyze_messages("s", "t", msgs)
    assert snap["memory_events"] == 0
    assert snap["memory_write_events"] == 0


def test_tool_arguments_helper_handles_all_shapes(plugin_api):
    """Lock the helper's contract so future edits don't reintroduce the bug."""
    h = plugin_api._tool_arguments_from_call
    assert h({"arguments": {"action": "add"}}) == {"action": "add"}
    assert h({"function": {"arguments": '{"action": "add"}'}}) == {"action": "add"}
    assert h({"function": {"arguments": "{bad json"}}) == {}
    assert h({"function": {"arguments": "42"}}) == {}
    assert h({"function": {}}) == {}
    assert h({}) == {}
    assert h(None) == {}
    assert h("not a dict") == {}


# ---------------------------------------------------------------------------
# Defensive parsing: non-dict ``function`` field must not abort the scan
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_function_field", [
    "memory",                       # legacy string-only shape
    42,                             # accidentally an int (model adapter bug)
    ["memory", {"action": "add"}],  # list — saw this in one corrupted DB
    True,                           # truthy but not a dict
])
def test_helpers_tolerate_non_dict_function_field(plugin_api, bad_function_field):
    """Both helpers must short-circuit when ``call["function"]`` isn't a dict.

    Regression for the Copilot review on PR #26936: ``call.get("function")
    or {}`` only filters out *falsy* values, so a truthy non-dict (string,
    int, list, bool) used to crash ``fn.get(...)`` and abort the entire
    session scan instead of just under-counting the malformed call.
    """
    call = {"name": "memory", "function": bad_function_field, "arguments": {"action": "add"}}
    # Neither helper may raise — both must degrade gracefully.
    assert plugin_api._tool_name_from_call(call) == "memory"
    # Top-level ``arguments`` is still honoured even though ``function`` is junk.
    assert plugin_api._tool_arguments_from_call(call) == {"action": "add"}


def test_analyze_messages_does_not_abort_on_corrupt_function_field(plugin_api):
    """A single malformed call must not zero out the whole session's stats."""
    msgs = [
        {
            "role": "assistant",
            "tool_calls": [
                {"name": "memory", "function": "memory", "arguments": {"action": "add"}},
                _memory_call("add"),
            ],
        },
    ]
    # Pre-fix this raised AttributeError on ``"memory".get("arguments")``;
    # post-fix the malformed call is parsed via the top-level ``arguments``
    # and counted, and the well-formed call counts too.
    snap = plugin_api.analyze_messages("s", "t", msgs)
    assert snap["memory_events"] == 2
    assert snap["memory_write_events"] == 2



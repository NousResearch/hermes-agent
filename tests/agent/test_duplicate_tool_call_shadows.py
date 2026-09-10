"""Regression tests for duplicate/shadow tool-call normalization."""

import unittest

from pydantic import BaseModel, ConfigDict

from agent.message_sanitization import uniquify_tool_call_ids


_MISSING = object()


def _call(call_id, name, arguments_marker="{}"):
    function = {"name": name}
    if arguments_marker is not _MISSING:
        function["arguments"] = arguments_marker
    return {"id": call_id, "function": function}


class DuplicateToolCallShadowTests(unittest.TestCase):
    def test_exact_duplicate_executes_once(self):
        first = _call("dup", "tool_search", '{"queries":["x"]}')
        duplicate = _call("dup", "tool_search", '{"queries":["x"]}')

        calls = [first, duplicate]
        result = uniquify_tool_call_ids(calls)

        self.assertIs(result, calls)
        self.assertEqual(calls, [first])

    def test_populated_then_empty_shadow_keeps_populated(self):
        populated = _call("dup", "kanban_create", '{"title":"child","assignee":"worker"}')
        shadow = _call("dup", "kanban_create", "{}")

        calls = [populated, shadow]
        uniquify_tool_call_ids(calls)

        self.assertEqual(calls, [populated])
        self.assertEqual(populated["id"], "dup")

    def test_empty_shadow_then_populated_keeps_populated(self):
        populated = _call("dup", "kanban_create", '{"title":"child","assignee":"worker"}')
        for empty_arguments in ("{}", "", None, _MISSING):
            with self.subTest(empty_arguments=empty_arguments):
                shadow = _call("dup", "kanban_create", empty_arguments)
                calls = [shadow, populated]

                uniquify_tool_call_ids(calls)

                self.assertEqual(calls, [populated])
                self.assertEqual(populated["id"], "dup")

    def test_two_meaningful_same_function_calls_are_preserved(self):
        first = _call("dup", "tool_call", '{"name":"one"}')
        second = _call("dup", "tool_call", '{"name":"two"}')

        calls = [first, second]
        uniquify_tool_call_ids(calls)

        self.assertEqual(calls, [first, second])
        self.assertEqual([tc["id"] for tc in calls], ["dup", "dup_d2"])

    def test_same_id_different_functions_are_preserved(self):
        first = _call("dup", "tool_search", '{"queries":["x"]}')
        second = _call("dup", "tool_describe", '{"names":["x"]}')

        calls = [first, second]
        uniquify_tool_call_ids(calls)

        self.assertEqual(calls, [first, second])
        self.assertEqual([tc["id"] for tc in calls], ["dup", "dup_d2"])

    def test_lone_malformed_call_remains_visible(self):
        malformed = _call("bad", "tool_call", '{"name"')

        calls = [malformed]
        uniquify_tool_call_ids(calls)

        self.assertEqual(calls, [malformed])
        self.assertEqual(malformed["function"]["arguments"], '{"name"')

    def test_frozen_pydantic_call_is_rebuilt_for_duplicate_id_repair(self):
        class FrozenFunction(BaseModel):
            model_config = ConfigDict(frozen=True)
            name: str
            arguments: str

        class FrozenToolCall(BaseModel):
            model_config = ConfigDict(frozen=True)
            id: str
            call_id: str
            function: FrozenFunction

        first = FrozenToolCall(
            id="dup", call_id="dup", function=FrozenFunction(name="tool_call", arguments='{"name":"one"}'),
        )
        second = FrozenToolCall(
            id="dup", call_id="dup", function=FrozenFunction(name="tool_call", arguments='{"name":"two"}'),
        )
        calls = [first, second]

        result = uniquify_tool_call_ids(calls)

        self.assertIs(result, calls)
        self.assertIs(calls[0], first)
        self.assertIsNot(calls[1], second)
        self.assertEqual(calls[1].id, "dup_d2")
        self.assertEqual(calls[1].call_id, "dup_d2")

    def test_existing_suffix_and_composite_pairing_semantics_are_preserved(self):
        calls = [
            _call("call_x|fc_1", "one", '{"value":1}'),
            _call("call_x_d2|fc_existing", "existing", '{"value":2}'),
            _call("call_x|fc_2", "two", '{"value":3}'),
        ]

        uniquify_tool_call_ids(calls)

        self.assertEqual(
            [tc["id"] for tc in calls],
            ["call_x|fc_1", "call_x_d2|fc_existing", "call_x_d3|fc_2"],
        )


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Regression test: the inline delegate_task forwarder must pass `profile`.

Root cause caught Jun 2026: `AIAgent._dispatch_delegate_task` (run_agent.py) is
the single call site the gateway's concurrent/sequential/inline tool paths use to
invoke `delegate_task`. It forwarded goal/context/toolsets/tasks/max_iterations/
acp_command/acp_args/role but DROPPED `profile`. Result: a top-level
`delegate_task(goal=..., profile="dual-review")` silently lost its profile, so the
dual-review expansion never fired and only ONE lane (the default model) ran —
reported as a normal success. The per-task form (`tasks=[{profile:"dual-review"}]`)
worked only because `profile` rode inside the forwarded `tasks` dict.

This test pins the forwarding contract: every top-level scalar arg the schema
exposes (notably `profile`) must reach `delegate_task`. It fails on the old code
(no `profile=` line) and passes once the forwarder includes it.

Run:  python -m pytest tests/run_agent/test_dispatch_delegate_task_profile.py -v
"""

import unittest
from unittest.mock import MagicMock, patch


class TestDispatchDelegateTaskForwardsProfile(unittest.TestCase):
    def _call_dispatch(self, function_args: dict):
        """Invoke AIAgent._dispatch_delegate_task in isolation, capturing the
        kwargs the inline forwarder passes down to tools.delegate_tool.delegate_task.
        """
        from run_agent import AIAgent

        captured = {}

        def _fake_delegate_task(**kwargs):
            captured.update(kwargs)
            return "{}"

        # Bind the real method to a stand-in self; the method only touches
        # `self` via parent_agent=self (which delegate_task is mocked away from)
        # and `self._delegate_depth` to decide background vs synchronous
        # dispatch. Pin the depth to 0 (top-level model) so the int comparison
        # works on the MagicMock stand-in.
        fake_self = MagicMock()
        fake_self._delegate_depth = 0
        with patch("tools.delegate_tool.delegate_task", _fake_delegate_task):
            AIAgent._dispatch_delegate_task(fake_self, function_args)
        return captured

    def test_top_level_profile_is_forwarded(self):
        captured = self._call_dispatch(
            {"goal": "review X", "context": "ctx", "profile": "dual-review"}
        )
        self.assertEqual(
            captured.get("profile"),
            "dual-review",
            "inline forwarder dropped top-level 'profile' — dual-review expansion "
            "would never fire (see run_agent.AIAgent._dispatch_delegate_task)",
        )

    def test_all_schema_top_level_args_are_forwarded(self):
        """Every top-level scalar/array arg the model can set must be forwarded.

        Guards against the same drop recurring for a future field: compares the
        forwarder's forwarded keys against the schema's top-level properties
        (minus per-task-only 'tasks', which is itself forwarded).
        """
        from tools.delegate_tool import DELEGATE_TASK_SCHEMA

        schema_props = set(DELEGATE_TASK_SCHEMA["parameters"]["properties"].keys())
        # max_iterations is intentionally not model-exposed but IS forwarded;
        # include any property the model can set.
        captured = self._call_dispatch(
            {
                "goal": "g",
                "context": "c",
                "toolsets": ["web"],
                "tasks": None,
                "acp_command": None,
                "acp_args": None,
                "role": "leaf",
                "profile": "dual-review",
            }
        )
        forwarded = set(captured.keys())
        # Every model-facing top-level property must be forwarded.
        for prop in schema_props:
            self.assertIn(
                prop,
                forwarded,
                f"schema exposes top-level '{prop}' but the inline forwarder "
                f"does not pass it to delegate_task",
            )


if __name__ == "__main__":
    unittest.main()

"""Regression tests for length-truncated dispatcher-owned Kanban workers.

Extract the pure prompt helper via AST to avoid importing the full agent/model
stack while still exercising the production function and literals unchanged.
"""
from __future__ import annotations

import ast
import unittest
from pathlib import Path
from typing import Optional
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "agent" / "conversation_loop.py"
CONSTANTS = {
    "_LENGTH_CONTINUATION_NETWORK_STUB",
    "_LENGTH_CONTINUATION_OUTPUT_LIMIT",
    "_LENGTH_CONTINUATION_KANBAN_OUTPUT_LIMIT",
    "_LENGTH_CONTINUATION_DROPPED_TOOLS_PREFIX",
}


def _helper():
    tree = ast.parse(SOURCE.read_text(), filename=str(SOURCE))
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in CONSTANTS
            for target in node.targets
        ):
            nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name == "_get_continuation_prompt":
            nodes.append(node)
    assert len(nodes) == len(CONSTANTS) + 1, "Missing helper or constant"
    namespace = {"Optional": Optional}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace["_get_continuation_prompt"]


class KanbanLengthContinuationTests(unittest.TestCase):
    def setUp(self):
        self.continue_prompt = _helper()

    def test_truncated_worker_prompt_requires_real_terminal_tool(self):
        with patch("agent.delegation_context.owned_kanban_task", return_value="t_probe"):
            prompt = self.continue_prompt(False)
        self.assertIn("REAL tool calls", prompt)
        self.assertIn("kanban_request_review", prompt)
        self.assertIn("kanban_complete", prompt)
        self.assertIn("kanban_block", prompt)
        self.assertIn("content argument for write_file", prompt)
        self.assertNotIn("Finish the answer directly", prompt)

    def test_unowned_session_keeps_original_byte_identical_prompt(self):
        with patch("agent.delegation_context.owned_kanban_task", return_value=""):
            prompt = self.continue_prompt(False)
        self.assertEqual(
            prompt,
            "[System: Your previous response was truncated by the output length limit. "
            "Continue exactly where you left off. Do not restart or repeat prior text. "
            "Finish the answer directly.]",
        )

    def test_network_stub_for_worker_unmodified(self):
        with patch("agent.delegation_context.owned_kanban_task", return_value="t_probe"):
            prompt = self.continue_prompt(True)
        self.assertIn("network error mid-stream", prompt)

    def test_dropped_large_tool_arguments_keep_split_instruction(self):
        with patch("agent.delegation_context.owned_kanban_task", return_value="t_probe"):
            prompt = self.continue_prompt(True, ["write_file"])
        self.assertIn("write_file", prompt)
        self.assertIn("multiple smaller tool", prompt)

    def test_delegated_child_identity_does_not_claim_worker(self):
        with patch("agent.delegation_context.owned_kanban_task", return_value=""):
            prompt = self.continue_prompt(False)
        self.assertNotIn("REAL tool calls", prompt)


if __name__ == "__main__":
    unittest.main()

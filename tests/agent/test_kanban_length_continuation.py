"""Regression tests for length-truncated dispatcher-owned Kanban workers.

Exercises both the behavioral runtime function and the structural AST contracts
to ensure length-truncated workers receive actionable tool instructions and
their synthetic recovery nudges are preserved across context compression.
"""
from __future__ import annotations

import ast
import unittest
from pathlib import Path
from typing import Optional
from unittest.mock import patch

from agent.conversation_loop import (
    _LENGTH_CONTINUATION_KANBAN_OUTPUT_LIMIT,
    _LENGTH_CONTINUATION_OUTPUT_LIMIT,
    _get_continuation_prompt,
)

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "agent" / "conversation_loop.py"
CONSTANTS = {
    "_LENGTH_CONTINUATION_NETWORK_STUB",
    "_LENGTH_CONTINUATION_OUTPUT_LIMIT",
    "_LENGTH_CONTINUATION_KANBAN_OUTPUT_LIMIT",
    "_LENGTH_CONTINUATION_DROPPED_TOOLS_PREFIX",
}


def _ast_helper():
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
        self.continue_prompt = _get_continuation_prompt
        self.ast_continue_prompt = _ast_helper()

    def test_truncated_worker_prompt_requires_all_terminal_tools(self):
        with patch("agent.delegation_context.owned_kanban_task", return_value="t_probe"):
            prompt = self.continue_prompt(False)
        self.assertIn("REAL tool calls", prompt)
        self.assertIn("kanban_complete", prompt)
        self.assertIn("kanban_request_review", prompt)
        self.assertIn("kanban_request_changes", prompt)
        self.assertIn("kanban_schedule", prompt)
        self.assertIn("kanban_block", prompt)
        self.assertIn("content argument for write_file", prompt)
        self.assertNotIn("Finish the answer directly", prompt)

    def test_ast_extracted_helper_matches_live_function(self):
        with patch("agent.delegation_context.owned_kanban_task", return_value="t_probe"):
            live = self.continue_prompt(False)
            extracted = self.ast_continue_prompt(False)
        self.assertEqual(live, extracted)

    def test_unowned_session_keeps_original_byte_identical_prompt(self):
        with patch("agent.delegation_context.owned_kanban_task", return_value=""):
            prompt = self.continue_prompt(False)
        self.assertEqual(
            prompt,
            "[System: Your previous response was truncated by the output length limit. "
            "Continue exactly where you left off. Do not restart or repeat prior text. "
            "Finish the answer directly.]",
        )

    def test_context_compressor_classifies_worker_nudge_as_internal(self):
        # SessionDB strips synthetic metadata, so the compressor must recognize
        # the new Kanban continuation message by its exact content marker.
        context_source = (ROOT / "agent" / "context_compressor.py").read_text()
        tree = ast.parse(context_source)
        method = next(
            node for cls in tree.body if isinstance(cls, ast.ClassDef)
            and cls.name == "ContextCompressor"
            for node in cls.body if isinstance(node, ast.FunctionDef)
            and node.name == "_is_synthetic_compression_user_turn"
        )
        imports = [
            name.name for node in ast.walk(method)
            if isinstance(node, ast.ImportFrom)
            and node.module == "agent.conversation_loop"
            for name in node.names
        ]
        self.assertIn("_LENGTH_CONTINUATION_KANBAN_OUTPUT_LIMIT", imports)
        matches = [
            node for node in ast.walk(method)
            if isinstance(node, ast.Name)
            and node.id == "_LENGTH_CONTINUATION_KANBAN_OUTPUT_LIMIT"
        ]
        self.assertEqual(len(matches), 1)

    def test_context_compressor_runtime_recognizes_worker_nudge(self):
        from agent.context_compressor import ContextCompressor

        with patch("agent.delegation_context.owned_kanban_task", return_value="t_probe"):
            prompt = self.continue_prompt(False)
        self.assertTrue(ContextCompressor._is_synthetic_compression_user_turn(
            {"role": "user", "content": prompt}
        ))

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

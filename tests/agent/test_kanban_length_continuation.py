"""Regression tests for length-truncated dispatcher-owned Kanban workers.

Exercises the runtime behavior selecting continuation prompts and classifying
synthetic recovery nudges across context compression.
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

from agent.conversation_loop import (
    _get_continuation_prompt,
)

class KanbanLengthContinuationTests(unittest.TestCase):
    def setUp(self):
        self.continue_prompt = _get_continuation_prompt

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

    def test_unowned_session_keeps_original_byte_identical_prompt(self):
        with patch("agent.delegation_context.owned_kanban_task", return_value=""):
            prompt = self.continue_prompt(False)
        self.assertEqual(
            prompt,
            "[System: Your previous response was truncated by the output length limit. "
            "Continue exactly where you left off. Do not restart or repeat prior text. "
            "Finish the answer directly.]",
        )

    def test_context_compressor_runtime_recognizes_worker_nudge(self):
        from agent.context_compressor import ContextCompressor

        with patch("agent.delegation_context.owned_kanban_task", return_value="t_probe"):
            prompt = self.continue_prompt(False)
        self.assertTrue(ContextCompressor._is_synthetic_compression_user_turn(
            {"role": "user", "content": prompt}
        ))

    def test_non_synthetic_user_text_is_not_classified_as_internal(self):
        from agent.context_compressor import ContextCompressor

        self.assertFalse(ContextCompressor._is_synthetic_compression_user_turn(
            {"role": "user", "content": "Please continue my Kanban task."}
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

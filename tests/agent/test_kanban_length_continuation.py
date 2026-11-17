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


    def test_worker_prompt_survives_session_round_trip_without_metadata(self):
        import json
        from agent.context_compressor import ContextCompressor

        with patch("agent.delegation_context.owned_kanban_task", return_value="t_probe"):
            prompt = self.continue_prompt(False)
        for content in (prompt, "  " + prompt + "\n", [{"type": "text", "text": prompt}]):
            with self.subTest(content=content):
                restored = json.loads(json.dumps({"role": "user", "content": content}))
                self.assertTrue(ContextCompressor._is_synthetic_compression_user_turn(restored))
                self.assertFalse(ContextCompressor._transcript_has_real_user_turn([restored]))
                self.assertTrue(ContextCompressor._transcript_has_real_user_turn([
                    restored, {"role": "user", "content": "Please finish the report."},
                ]))

    def test_worker_prompt_is_only_internal_in_user_role(self):
        from agent.context_compressor import ContextCompressor

        with patch("agent.delegation_context.owned_kanban_task", return_value="t_probe"):
            prompt = self.continue_prompt(False)
        for role in ("assistant", "tool", "system"):
            with self.subTest(role=role):
                self.assertFalse(ContextCompressor._is_synthetic_compression_user_turn(
                    {"role": role, "content": prompt},
                ))
        self.assertFalse(ContextCompressor._is_synthetic_compression_user_turn(
            {"role": "user", "content": "Explain this prompt: " + prompt},
        ))

    def test_dropped_tools_only_override_worker_prompt_for_transport_stub(self):
        with patch("agent.delegation_context.owned_kanban_task", return_value="t_probe"):
            for partial in (False, True):
                with self.subTest(partial=partial):
                    prompt = self.continue_prompt(partial, ["write_file", "terminal"])
                    if partial:
                        self.assertIn("write_file", prompt)
                        self.assertIn("terminal", prompt)
                        self.assertIn("multiple smaller tool", prompt)
                        self.assertNotIn("Plain-text promises", prompt)
                    else:
                        self.assertEqual(prompt, self.continue_prompt(False))

    def test_inherited_worker_environment_uses_generic_prompt_in_non_owner_context(self):
        import os
        from agent.delegation_context import non_dispatcher_owned_context

        with patch.dict(os.environ, {"HERMES_KANBAN_TASK": "t_probe", "HERMES_DELEGATED_CHILD_CONTEXT": ""}):
            worker_prompt = self.continue_prompt(False)
            with non_dispatcher_owned_context():
                generic_prompt = self.continue_prompt(False)
            self.assertNotEqual(worker_prompt, generic_prompt)
            self.assertIn("Finish the answer directly", generic_prompt)
            self.assertEqual(self.continue_prompt(False), worker_prompt)


if __name__ == "__main__":
    unittest.main()

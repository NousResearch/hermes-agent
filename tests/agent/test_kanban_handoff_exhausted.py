"""Regression tests for the kanban worker handoff-exhaustion guard.

A dispatcher-owned worker that ends a turn on prose (no terminal board call) after its
stop nudges were exhausted must NOT exit cleanly: the stop gate flags the turn so the
caller fails it honestly and the process exits non-zero — the dispatcher then books a
crash instead of a ``protocol_violation``. This is the "missing terminal-action recovery"
half of the length-truncation fix (see ``test_kanban_length_continuation.py`` for the
continuation-prompt half).

Written unittest-style so it runs under the same direct-run path as the AST-based
``test_kanban_length_continuation.py`` while still being collected by pytest in CI.
"""

from __future__ import annotations

import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.kanban_stop import (
    build_kanban_stop_nudge,
    kanban_handoff_exhausted,
    session_called_kanban_terminal,
)

_ENV_WORKER = {"HERMES_KANBAN_TASK": "t_probe"}

_NARRATED = [{"role": "assistant", "content": "I will call kanban_complete next."}]
_TERMINAL = [
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "1",
                "type": "function",
                "function": {"name": "kanban_complete", "arguments": "{}"},
            }
        ],
    },
    {"role": "tool", "name": "kanban_complete", "tool_call_id": "1", "content": "done"},
]
_TERMINAL_FAILED = [
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "1",
                "type": "function",
                "function": {"name": "kanban_complete", "arguments": "{}"},
            }
        ],
    },
    {
        "role": "tool",
        "name": "kanban_complete",
        "tool_call_id": "1",
        "content": json.dumps({"error": "kanban_complete blocked: missing required deliverables"}),
    },
    {"role": "assistant", "content": "I encountered an error trying to complete."},
]
_TERMINAL_WRONG_TASK = [
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "1",
                "type": "function",
                "function": {"name": "kanban_complete", "arguments": json.dumps({"task_id": "t_other"})},
            }
        ],
    },
    {
        "role": "tool",
        "name": "kanban_complete",
        "tool_call_id": "1",
        "content": json.dumps({"ok": True, "task_id": "t_other"}),
    },
    {"role": "assistant", "content": "Completed t_other."},
]


class KanbanHandoffExhaustedTests(unittest.TestCase):
    """The pure predicate: worker identity + attempts + presence of a successful terminal call."""

    def setUp(self):
        self._env = patch.dict(os.environ, dict(_ENV_WORKER), clear=False)
        self._env.start()
        self.addCleanup(self._env.stop)

    def test_exhausted_nudges_without_terminal_call(self):
        self.assertTrue(kanban_handoff_exhausted(_NARRATED, attempts=2))

    def test_below_max_attempts_is_not_exhausted(self):
        self.assertFalse(kanban_handoff_exhausted(_NARRATED, attempts=1))

    def test_terminal_call_present_is_never_exhausted(self):
        self.assertFalse(kanban_handoff_exhausted(_TERMINAL, attempts=2))

    def test_failed_terminal_tool_does_not_satisfy_handoff(self):
        # A tool call whose execution produced an error did not hand off the card.
        self.assertFalse(session_called_kanban_terminal(_TERMINAL_FAILED))
        self.assertTrue(kanban_handoff_exhausted(_TERMINAL_FAILED, attempts=2))

    def test_terminal_tool_targeting_wrong_task_does_not_satisfy_handoff(self):
        # Completing an external or child task does not hand off the assigned task.
        self.assertFalse(session_called_kanban_terminal(_TERMINAL_WRONG_TASK))
        self.assertTrue(kanban_handoff_exhausted(_TERMINAL_WRONG_TASK, attempts=2))

    def test_orphan_assistant_terminal_call_is_not_handoff(self):
        messages = [_TERMINAL[0]]
        self.assertFalse(session_called_kanban_terminal(messages))
        self.assertTrue(kanban_handoff_exhausted(messages, attempts=2))

    def test_mismatched_tool_call_id_is_not_handoff(self):
        messages = [_TERMINAL[0], {
            "role": "tool", "name": "kanban_complete",
            "tool_call_id": "unrelated", "content": "done",
        }]
        self.assertFalse(session_called_kanban_terminal(messages))

    def test_mismatched_tool_name_is_not_handoff(self):
        messages = [_TERMINAL[0], {
            "role": "tool", "name": "kanban_block",
            "tool_call_id": "1", "content": "done",
        }]
        self.assertFalse(session_called_kanban_terminal(messages))

    def test_standalone_cross_task_success_is_not_handoff(self):
        messages = [{
            "role": "tool", "name": "kanban_complete",
            "tool_call_id": "1",
            "content": json.dumps({"ok": True, "task_id": "t_other"}),
        }]
        self.assertFalse(session_called_kanban_terminal(messages))

    def test_matched_call_with_foreign_result_task_is_not_handoff(self):
        messages = [_TERMINAL[0], {
            "role": "tool", "name": "kanban_complete",
            "tool_call_id": "1",
            "content": json.dumps({"ok": True, "task_id": "t_other"}),
        }]
        self.assertFalse(session_called_kanban_terminal(messages))

    def test_error_then_orphan_call_is_not_handoff(self):
        messages = list(_TERMINAL_FAILED) + [{
            "role": "assistant", "tool_calls": [{
                "id": "2", "type": "function",
                "function": {"name": "kanban_complete", "arguments": "{}"},
            }],
        }]
        self.assertFalse(session_called_kanban_terminal(messages))

    def test_empty_json_error_field_is_not_failure(self):
        messages = [_TERMINAL[0], {
            "role": "tool", "name": "kanban_complete",
            "tool_call_id": "1",
            "content": json.dumps({"ok": True, "error": "", "task_id": "t_probe"}),
        }]
        self.assertTrue(session_called_kanban_terminal(messages))

    def test_json_explicit_error_still_fails(self):
        messages = [_TERMINAL[0], {
            "role": "tool", "name": "kanban_complete",
            "tool_call_id": "1",
            "content": json.dumps({"ok": False, "error": "", "task_id": "t_probe"}),
        }]
        self.assertFalse(session_called_kanban_terminal(messages))

    def test_malformed_result_is_not_handoff(self):
        messages = [_TERMINAL[0], {
            "role": "tool", "name": "kanban_complete",
            "tool_call_id": "1", "content": "{unparseable",
        }]
        self.assertFalse(session_called_kanban_terminal(messages))

    def test_retry_after_failed_terminal_tool_succeeds(self):
        # Initial failure followed by successful completion counts as handed off.
        messages = list(_TERMINAL_FAILED) + [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "2",
                        "type": "function",
                        "function": {"name": "kanban_complete", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "name": "kanban_complete",
                "tool_call_id": "2",
                "content": json.dumps({"ok": True, "task_id": "t_probe"}),
            },
        ]
        self.assertTrue(session_called_kanban_terminal(messages))
        self.assertFalse(kanban_handoff_exhausted(messages, attempts=2))

    def test_not_a_worker_is_never_exhausted(self):
        # HERMES_KANBAN_TASK cleared → owned_kanban_task() == "" → not a worker.
        with patch.dict(os.environ, {"HERMES_KANBAN_TASK": ""}):
            self.assertFalse(kanban_handoff_exhausted(_NARRATED, attempts=2))


def _agent(*, nudges):
    """Minimal stop-gate agent double, matching the shape ``apply_stop_gates`` reads."""
    return SimpleNamespace(
        _kanban_stop_nudges=nudges,
        _lsp_turn_end_notes=3,  # short-circuit the late-LSP import probe
        _interim_content_was_streamed=lambda _text: False,
        _emit_interim_assistant_message=lambda _msg: None,
        _flush_messages_to_session_db=lambda *_a: None,
        _emit_diagnostic_status=lambda *_a, **_k: None,
        _has_content_after_think_block=lambda text: bool(text),
        _strip_think_blocks=lambda text: text,
        _emit_pending_fallback_notice=lambda: None,
        _clear_status_buffer=lambda: None,
        _build_assistant_message=lambda msg, finish_reason: {
            "role": "assistant",
            "content": getattr(msg, "content", ""),
        },
        valid_tool_names=[],
        api_mode="default",
        model="mock",
        provider="mock",
        log_prefix="",
        _vprint=lambda *_a, **_k: None,
        _cleanup_task_resources=MagicMock(),
        _persist_session=MagicMock(),
    )


class ApplyStopGatesHandoffTests(unittest.TestCase):
    """The stop gate flags the exhausted worker so the turn fails instead of completing."""

    def setUp(self):
        self._env = patch.dict(
            os.environ, dict(_ENV_WORKER, HERMES_VERIFY_ON_STOP="0"), clear=False
        )
        self._env.start()
        self.addCleanup(self._env.stop)

    def _apply(self, *, agent, messages):
        from agent.turn_stop_gates import apply_stop_gates

        return apply_stop_gates(
            agent,
            {"role": "assistant", "content": messages[-1].get("content", "")},
            final_response=messages[-1].get("content", ""),
            messages=messages,
            conversation_history=[],
            pending_verification_response=None,
            pending_verification_response_previewed=False,
        )

    def test_exhausted_worker_is_flagged_for_failure(self):
        verdict = self._apply(agent=_agent(nudges=2), messages=list(_NARRATED))
        self.assertFalse(verdict.continue_turn)
        self.assertTrue(verdict.kanban_handoff_failed)

    def test_worker_with_remaining_nudges_is_continued_not_failed(self):
        verdict = self._apply(agent=_agent(nudges=0), messages=list(_NARRATED))
        self.assertTrue(verdict.continue_turn)
        self.assertFalse(verdict.kanban_handoff_failed)

    def test_worker_that_called_a_terminal_tool_is_not_failed(self):
        verdict = self._apply(agent=_agent(nudges=2), messages=list(_TERMINAL))
        self.assertFalse(verdict.continue_turn)
        self.assertFalse(verdict.kanban_handoff_failed)

    def test_worker_with_failed_terminal_tool_is_flagged_for_failure_when_exhausted(self):
        verdict = self._apply(agent=_agent(nudges=2), messages=list(_TERMINAL_FAILED))
        self.assertFalse(verdict.continue_turn)
        self.assertTrue(verdict.kanban_handoff_failed)


class FinishTextResponseHandoffFailureTests(unittest.TestCase):
    """Integration: finish_text_response returns a failed result when handoff nudges exhaust."""

    def setUp(self):
        self._env = patch.dict(
            os.environ, dict(_ENV_WORKER, HERMES_VERIFY_ON_STOP="0"), clear=False
        )
        self._env.start()
        self.addCleanup(self._env.stop)

    def test_handoff_exhausted_returns_failed_result(self):
        from agent.turn_final_response import finish_text_response

        agent = _agent(nudges=2)
        agent._cleanup_task_resources = MagicMock()
        agent._persist_session = MagicMock()

        messages = list(_NARRATED)
        assistant_msg = SimpleNamespace(
            content=messages[-1]["content"],
            tool_calls=[],
            reasoning_content=None,
        )

        verdict = finish_text_response(
            agent,
            assistant_message=assistant_msg,
            response=None,
            finish_reason="stop",
            messages=messages,
            api_messages=[],
            conversation_history=[],
            api_call_count=1,
            user_message="Work task t_probe",
            active_system_prompt="system",
            final_response=assistant_msg.content,
            _turn_exit_reason=None,
            _preflight_compression_blocked=False,
            codex_ack_continuations=0,
            truncated_response_parts=[],
            length_continue_retries=0,
            _pending_verification_response=None,
            _pending_verification_response_previewed=False,
            effective_task_id="t_probe",
        )

        self.assertEqual(verdict.action, "return")
        result = verdict.result
        self.assertIsNotNone(result)
        self.assertTrue(result.get("failed"))
        self.assertFalse(result.get("completed"))
        self.assertEqual(result.get("failure_reason"), "loop_error")
        self.assertIn("This kanban task could not be handed off", str(result.get("error")))
        agent._cleanup_task_resources.assert_called_once_with("t_probe")
        agent._persist_session.assert_called_once_with(messages, [])
        self.assertEqual(messages, _NARRATED)


class TerminalResultContractTests(unittest.TestCase):
    """Only a correlated, successful response releases the worker's card."""

    @staticmethod
    def transcript(content, *, name="kanban_complete", arguments=None):
        return [
            {"role": "assistant", "tool_calls": [{
                "id": "handoff", "type": "function", "function": {
                    "name": name,
                    "arguments": arguments if arguments is not None else {"task_id": "t_probe"},
                },
            }]},
            {"role": "tool", "name": name, "tool_call_id": "handoff", "content": content},
        ]

    def test_each_terminal_action_accepts_text_and_structured_success(self):
        for name in (
            "kanban_complete", "kanban_request_review", "kanban_request_changes",
            "kanban_schedule", "kanban_block",
        ):
            for content in ("Task handed off", {"ok": True, "task_id": "t_probe"},
                            json.dumps({"success": True, "id": "t_probe"})):
                with self.subTest(name=name, content=content):
                    self.assertTrue(session_called_kanban_terminal(
                        self.transcript(content, name=name), task_id="t_probe",
                    ))

    def test_error_and_empty_results_do_not_release_card(self):
        for content in (
            None, "", "  \n", {"error": "missing artifacts"}, {"ok": False},
            {"success": False}, '{"success": false}', '{"error": "denied"}',
            "  ERROR: denied", "Error executing tool", "SCHEMA_VALIDATION_ERROR: content",
            "Tool error: rejected", "Failed: invalid state", "Rejected: wrong owner",
            "Exception: unavailable", "{invalid json}",
        ):
            with self.subTest(content=content):
                self.assertFalse(session_called_kanban_terminal(
                    self.transcript(content), task_id="t_probe",
                ))

    def test_result_id_alias_cannot_release_another_card(self):
        for content in ({"ok": True, "id": "t_other"},
                        json.dumps({"success": True, "id": "t_other"})):
            with self.subTest(content=content):
                self.assertFalse(session_called_kanban_terminal(
                    self.transcript(content), task_id="t_probe",
                ))

    def test_explicit_task_takes_precedence_over_environment(self):
        messages = self.transcript("done")
        with patch.dict(os.environ, {"HERMES_KANBAN_TASK": "t_other"}):
            self.assertTrue(session_called_kanban_terminal(messages, task_id="t_probe"))
            self.assertFalse(session_called_kanban_terminal(messages))

    def test_sdk_and_flat_calls_preserve_task_matching(self):
        for arguments in ({"task_id": "t_other"}, json.dumps({"task_id": "t_other"})):
            for call in (
                SimpleNamespace(id="handoff", function=SimpleNamespace(
                    name="kanban_complete", arguments=arguments)),
                SimpleNamespace(id="handoff", name="kanban_complete", arguments=arguments),
                {"id": "handoff", "name": "kanban_complete", "arguments": arguments},
            ):
                with self.subTest(arguments=arguments, call=call):
                    messages = self.transcript("done")
                    messages[0]["tool_calls"] = [call]
                    self.assertFalse(session_called_kanban_terminal(messages, task_id="t_probe"))
                    self.assertTrue(session_called_kanban_terminal(messages, task_id="t_other"))

    def test_implicit_task_can_use_result_identity(self):
        # Legacy adapters omit task_id and let the tool default to the owned card.
        for arguments in ("{}", {}):
            with self.subTest(arguments=arguments):
                messages = self.transcript({"ok": True, "task_id": "t_probe"})
                messages[0]["tool_calls"][0]["function"]["arguments"] = arguments
                self.assertTrue(session_called_kanban_terminal(messages, task_id="t_probe"))

    def test_nonterminal_tool_success_does_not_release_card(self):
        self.assertFalse(session_called_kanban_terminal(
            self.transcript("done", name="write_file"), task_id="t_probe",
        ))

    def test_missing_call_id_cannot_match_a_result(self):
        for call_id in (None, "", "  "):
            with self.subTest(call_id=call_id):
                messages = self.transcript("done")
                messages[0]["tool_calls"][0]["id"] = call_id
                messages[1]["tool_call_id"] = call_id
                self.assertFalse(session_called_kanban_terminal(messages, task_id="t_probe"))

    def test_empty_or_unrelated_transcript_is_not_handoff(self):
        for messages in (None, [], [None, "done", {"role": "user", "content": "done"}]):
            with self.subTest(messages=messages):
                self.assertFalse(session_called_kanban_terminal(messages, task_id="t_probe"))

    def test_transcript_iterator_matches_success(self):
        self.assertTrue(session_called_kanban_terminal(
            iter(self.transcript("done")), task_id="t_probe",
        ))


class HandoffBudgetBoundaryTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {
            "HERMES_KANBAN_TASK": "t_probe", "HERMES_KANBAN_STOP_NUDGE": "1",
            "HERMES_DELEGATED_CHILD_CONTEXT": "",
        })
        env.start()
        self.addCleanup(env.stop)

    def test_custom_budget_switches_from_nudge_to_failure_at_boundary(self):
        for maximum in (0, 1, 3):
            for attempts in (maximum - 1, maximum, maximum + 1):
                with self.subTest(maximum=maximum, attempts=attempts):
                    exhausted = attempts >= maximum
                    self.assertEqual(kanban_handoff_exhausted(
                        _NARRATED, attempts=attempts, max_attempts=maximum,
                    ), exhausted)
                    self.assertEqual(build_kanban_stop_nudge(
                        messages=_NARRATED, attempts=attempts, max_attempts=maximum,
                    ) is None, exhausted)

    def test_disabled_guard_neither_nudges_nor_fails(self):
        for value in ("0", "false", "NO", " Off "):
            with self.subTest(value=value), patch.dict(os.environ, {"HERMES_KANBAN_STOP_NUDGE": value}):
                self.assertIsNone(build_kanban_stop_nudge(messages=_NARRATED))
                self.assertFalse(kanban_handoff_exhausted(_NARRATED, attempts=99))

    def test_inherited_task_does_not_fail_non_dispatcher_execution(self):
        from agent.delegation_context import non_dispatcher_owned_context

        with non_dispatcher_owned_context():
            self.assertIsNone(build_kanban_stop_nudge(messages=_NARRATED))
            self.assertFalse(kanban_handoff_exhausted(_NARRATED, attempts=99))
        self.assertTrue(kanban_handoff_exhausted(_NARRATED, attempts=99))

    def test_successful_handoff_suppresses_nudge_even_with_budget_remaining(self):
        for name in ("kanban_request_review", "kanban_request_changes", "kanban_schedule"):
            with self.subTest(name=name):
                messages = TerminalResultContractTests.transcript("done", name=name)
                self.assertIsNone(build_kanban_stop_nudge(messages=messages, attempts=0))
                self.assertFalse(kanban_handoff_exhausted(messages, attempts=99))

    def test_failed_result_remains_eligible_for_recovery_nudge(self):
        messages = TerminalResultContractTests.transcript({"success": False})
        nudge = build_kanban_stop_nudge(messages=messages, attempts=1)
        self.assertIsNotNone(nudge)
        self.assertIn("t_probe", nudge)


if __name__ == "__main__":
    unittest.main()

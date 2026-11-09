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

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agent.kanban_stop import kanban_handoff_exhausted

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


class KanbanHandoffExhaustedTests(unittest.TestCase):
    """The pure predicate: worker identity + attempts + absence of a terminal call."""

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


if __name__ == "__main__":
    unittest.main()

"""Unit tests for gateway/platforms/cards/streaming_controller.py.

Tests the 6-phase state machine, accumulator blocks, flush throttle,
and terminal transitions (completed / aborted / terminated).
No network I/O — client is mocked.
"""

import asyncio
import json
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from gateway.platforms.cards.streaming_controller import (
    Phase,
    StreamingCardController,
    TERMINAL_PHASES,
    _build_final_card,
    _build_streaming_card,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _run(coro):
    """Run a coroutine synchronously in the current event loop."""
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_controller(message_id="om_test"):
    mock_client = MagicMock()
    mock_client.request = MagicMock(return_value=MagicMock(code=0, msg="ok"))
    return StreamingCardController(message_id=message_id, client=mock_client)


# ---------------------------------------------------------------------------
# State machine: phase transitions
# ---------------------------------------------------------------------------

class TestPhaseTransitions(unittest.TestCase):
    """Tests for the state machine phase transition logic."""

    def test_initial_phase_is_idle(self):
        ctrl = _make_controller()
        self.assertEqual(ctrl.phase, Phase.IDLE)

    def test_add_text_chunk_transitions_idle_to_streaming(self):
        ctrl = _make_controller()
        with patch.object(ctrl, "_perform_flush", new=AsyncMock()):
            _run(ctrl.add_text_chunk("hello"))
        self.assertEqual(ctrl.phase, Phase.STREAMING)

    def test_add_reasoning_chunk_transitions_idle_to_streaming(self):
        ctrl = _make_controller()
        with patch.object(ctrl, "_perform_flush", new=AsyncMock()):
            _run(ctrl.add_reasoning_chunk("thinking..."))
        self.assertEqual(ctrl.phase, Phase.STREAMING)

    def test_mark_completed_transitions_streaming_to_completed(self):
        ctrl = _make_controller()
        with patch.object(ctrl, "_perform_flush", new=AsyncMock()), \
             patch.object(ctrl, "_flush_final", new=AsyncMock()):
            _run(ctrl.add_text_chunk("text"))
            _run(ctrl.mark_completed())
        self.assertEqual(ctrl.phase, Phase.COMPLETED)

    def test_abort_transitions_to_aborted(self):
        ctrl = _make_controller()
        with patch.object(ctrl, "_perform_flush", new=AsyncMock()), \
             patch.object(ctrl, "_flush_final", new=AsyncMock()):
            _run(ctrl.add_text_chunk("partial"))
            _run(ctrl.abort())
        self.assertEqual(ctrl.phase, Phase.ABORTED)
        self.assertTrue(ctrl.is_aborted)

    def test_terminate_transitions_to_terminated(self):
        ctrl = _make_controller()
        _run(ctrl.terminate("message deleted"))
        self.assertEqual(ctrl.phase, Phase.TERMINATED)
        self.assertEqual(ctrl.terminal_reason, "message deleted")

    def test_is_terminal_phase_true_after_completed(self):
        ctrl = _make_controller()
        with patch.object(ctrl, "_perform_flush", new=AsyncMock()), \
             patch.object(ctrl, "_flush_final", new=AsyncMock()):
            _run(ctrl.add_text_chunk("done"))
            _run(ctrl.mark_completed())
        self.assertTrue(ctrl.is_terminal_phase)

    def test_transition_rejected_from_completed_to_streaming(self):
        """Backward transitions must be silently rejected."""
        ctrl = _make_controller()
        # Force the controller into COMPLETED phase by manipulating _phase directly
        # so we don't trigger the full lifecycle (flush, etc.)
        ctrl._phase = Phase.COMPLETED
        accepted = ctrl._transition(Phase.STREAMING, "test")
        self.assertFalse(accepted)
        self.assertEqual(ctrl.phase, Phase.COMPLETED)

    def test_mark_completed_noop_when_already_terminal(self):
        ctrl = _make_controller()
        with patch.object(ctrl, "_flush_final", new=AsyncMock()):
            _run(ctrl.terminate("deleted"))
            # Should not raise
            _run(ctrl.mark_completed())
        self.assertEqual(ctrl.phase, Phase.TERMINATED)

    def test_abort_noop_when_already_terminal(self):
        ctrl = _make_controller()
        with patch.object(ctrl, "_flush_final", new=AsyncMock()):
            _run(ctrl.terminate("gone"))
            _run(ctrl.abort())
        self.assertEqual(ctrl.phase, Phase.TERMINATED)


# ---------------------------------------------------------------------------
# Accumulator blocks
# ---------------------------------------------------------------------------

class TestAccumulatorBlocks(unittest.TestCase):
    """Tests for text / reasoning / tool-use accumulator logic."""

    def test_add_text_chunk_accumulates_text(self):
        ctrl = _make_controller()
        with patch.object(ctrl, "_perform_flush", new=AsyncMock()):
            _run(ctrl.add_text_chunk("Hello "))
            _run(ctrl.add_text_chunk("world!"))
        self.assertEqual(ctrl.text.accumulated_text, "Hello world!")

    def test_add_reasoning_chunk_accumulates_reasoning(self):
        ctrl = _make_controller()
        with patch.object(ctrl, "_perform_flush", new=AsyncMock()):
            _run(ctrl.add_reasoning_chunk("step 1 "))
            _run(ctrl.add_reasoning_chunk("step 2"))
        self.assertEqual(ctrl.reasoning.accumulated_reasoning_text, "step 1 step 2")

    def test_reasoning_phase_flag_set_during_reasoning(self):
        ctrl = _make_controller()
        with patch.object(ctrl, "_perform_flush", new=AsyncMock()):
            _run(ctrl.add_reasoning_chunk("thinking"))
        self.assertTrue(ctrl.reasoning.is_reasoning_phase)

    def test_reasoning_phase_flag_cleared_on_text_chunk(self):
        ctrl = _make_controller()
        with patch.object(ctrl, "_perform_flush", new=AsyncMock()):
            _run(ctrl.add_reasoning_chunk("thinking"))
            _run(ctrl.add_text_chunk("answer"))
        self.assertFalse(ctrl.reasoning.is_reasoning_phase)

    def test_start_tool_use_sets_tool_name(self):
        ctrl = _make_controller()
        ctrl.start_tool_use("web_search", args={"query": "test"})
        self.assertEqual(ctrl.tool_use.tool_name, "web_search")
        self.assertTrue(ctrl.tool_use.is_active)

    def test_complete_tool_use_appends_step(self):
        ctrl = _make_controller()
        ctrl.start_tool_use("web_search", args={"query": "test"})
        ctrl.complete_tool_use(result="search results")
        self.assertEqual(len(ctrl.tool_use.steps), 1)
        self.assertEqual(ctrl.tool_use.steps[0]["name"], "web_search")
        self.assertFalse(ctrl.tool_use.is_active)

    def test_complete_tool_use_resets_active_fields(self):
        ctrl = _make_controller()
        ctrl.start_tool_use("tool_x")
        ctrl.complete_tool_use()
        self.assertIsNone(ctrl.tool_use.tool_name)
        self.assertIsNone(ctrl.tool_use.tool_args)
        self.assertFalse(ctrl.tool_use.is_active)

    def test_add_text_chunk_ignored_when_terminal(self):
        ctrl = _make_controller()
        with patch.object(ctrl, "_flush_final", new=AsyncMock()):
            _run(ctrl.terminate("gone"))
        initial_text = ctrl.text.accumulated_text
        with patch.object(ctrl, "_perform_flush", new=AsyncMock()):
            _run(ctrl.add_text_chunk("this should be ignored"))
        self.assertEqual(ctrl.text.accumulated_text, initial_text)

    def test_completed_text_set_on_mark_completed(self):
        ctrl = _make_controller()
        with patch.object(ctrl, "_perform_flush", new=AsyncMock()), \
             patch.object(ctrl, "_flush_final", new=AsyncMock()):
            _run(ctrl.add_text_chunk("final answer"))
            _run(ctrl.mark_completed())
        self.assertEqual(ctrl.text.completed_text, "final answer")


# ---------------------------------------------------------------------------
# flush throttle
# ---------------------------------------------------------------------------

class TestFlushThrottle(unittest.TestCase):
    """Tests for flush throttling behavior."""

    def test_flush_skips_when_no_message_id(self):
        """Controller with no message_id should not call sdk.request."""
        mock_client = MagicMock()
        ctrl = StreamingCardController(message_id=None, client=mock_client)
        _run(ctrl.flush())
        mock_client.request.assert_not_called()

    def test_flush_skips_when_no_client(self):
        """Controller with no client should not raise."""
        ctrl = StreamingCardController(message_id="om_x", client=None)
        # Should not raise
        _run(ctrl.flush())

    def test_perform_flush_skips_when_content_unchanged(self):
        """_perform_flush is a no-op when card JSON equals last_flushed_text."""
        ctrl = _make_controller()
        # Set up the controller in STREAMING phase with some text
        ctrl._phase = Phase.STREAMING
        ctrl.text.accumulated_text = "hello"

        # Pre-compute the card JSON that _perform_flush would send
        card_json = json.dumps(ctrl.to_card_json(), ensure_ascii=False)
        # Mark it as already flushed
        ctrl.text.last_flushed_text = card_json

        with patch.object(ctrl, "_patch_message", new=AsyncMock()) as mock_patch:
            _run(ctrl._perform_flush())

        # Content is identical to last flush — _patch_message must NOT be called
        mock_patch.assert_not_called()

    def test_pending_flush_flag_prevents_duplicate_schedules(self):
        """Setting _pending_flush=True should cause _schedule_flush to return early."""
        ctrl = _make_controller()
        ctrl._pending_flush = True
        with patch.object(ctrl, "_perform_flush", new=AsyncMock()) as mock_pf:
            _run(ctrl._schedule_flush())
        # _perform_flush should not be called because _pending_flush was already set
        mock_pf.assert_not_called()

    def test_on_enter_terminal_phase_clears_pending_flush(self):
        ctrl = _make_controller()
        ctrl._pending_flush = True
        ctrl._on_enter_terminal_phase()
        self.assertFalse(ctrl._pending_flush)


# ---------------------------------------------------------------------------
# to_card_json
# ---------------------------------------------------------------------------

class TestToCardJson(unittest.TestCase):
    """Tests for to_card_json() snapshot method."""

    def test_streaming_card_shows_cursor_when_no_text(self):
        ctrl = _make_controller()
        card = ctrl.to_card_json()
        # Use ensure_ascii=False so the half-block cursor character survives serialisation
        card_str = json.dumps(card, ensure_ascii=False)
        # Streaming card with no text shows a cursor placeholder
        self.assertIn("▌", card_str)

    def test_streaming_card_contains_text_when_accumulated(self):
        ctrl = _make_controller()
        with patch.object(ctrl, "_perform_flush", new=AsyncMock()):
            _run(ctrl.add_text_chunk("my answer"))
        card = ctrl.to_card_json()
        self.assertIn("my answer", json.dumps(card))

    def test_final_card_returned_after_completed(self):
        ctrl = _make_controller()
        with patch.object(ctrl, "_perform_flush", new=AsyncMock()), \
             patch.object(ctrl, "_flush_final", new=AsyncMock()):
            _run(ctrl.add_text_chunk("done"))
            _run(ctrl.mark_completed())
        card = ctrl.to_card_json()
        card_str = json.dumps(card)
        self.assertIn("Done", card_str)

    def test_aborted_card_shows_aborted_status(self):
        ctrl = _make_controller()
        with patch.object(ctrl, "_perform_flush", new=AsyncMock()), \
             patch.object(ctrl, "_flush_final", new=AsyncMock()):
            _run(ctrl.add_text_chunk("partial"))
            _run(ctrl.abort())
        card = ctrl.to_card_json()
        card_str = json.dumps(card)
        self.assertIn("Aborted", card_str)


# ---------------------------------------------------------------------------
# _build_streaming_card and _build_final_card helpers
# ---------------------------------------------------------------------------

class TestCardBuilders(unittest.TestCase):
    """Tests for module-level card builder helper functions."""

    def test_build_streaming_card_schema_is_2_0(self):
        card = _build_streaming_card(text="hello", is_streaming=True)
        self.assertEqual(card.get("schema"), "2.0")
        self.assertIn("body", card)

    def test_build_streaming_card_includes_generating_note(self):
        card = _build_streaming_card(text="partial", is_streaming=True)
        card_str = json.dumps(card)
        self.assertIn("Generating", card_str)

    def test_build_final_card_includes_done_note(self):
        card = _build_final_card(text="final answer", elapsed_ms=1500)
        card_str = json.dumps(card)
        self.assertIn("Done", card_str)

    def test_build_final_card_includes_reasoning_section_when_provided(self):
        card = _build_final_card(
            text="answer",
            reasoning_text="my thinking",
            reasoning_elapsed_ms=2000,
        )
        card_str = json.dumps(card)
        self.assertIn("my thinking", card_str)
        self.assertIn("Reasoning", card_str)

    def test_build_final_card_is_aborted_shows_aborted_status(self):
        card = _build_final_card(text="", is_aborted=True, elapsed_ms=500)
        card_str = json.dumps(card)
        self.assertIn("Aborted", card_str)

    def test_build_final_card_includes_tool_steps(self):
        steps = [{"name": "web_search", "elapsed_ms": 300}]
        card = _build_final_card(text="answer", tool_steps=steps)
        card_str = json.dumps(card)
        self.assertIn("web_search", card_str)


if __name__ == "__main__":
    unittest.main()

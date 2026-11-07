"""Cancel-safe ticketed redirects and checkpoint appends (no model, no storage)."""
import threading
import unittest
from unittest.mock import patch

from agent.correction_tickets import checkpoint_input
from agent.interrupt_control import InterruptControlMixin


def agent():
    obj = InterruptControlMixin()
    obj._pending_steer_lock = threading.Lock()
    obj._pending_redirect_lock = threading.Lock()
    obj._pending_steer = obj._pending_redirect = None
    obj._execution_thread_id = None
    obj._interrupt_requested = False
    obj._model_request_active = threading.Event()
    obj._executing_tools = False
    return obj


class RedirectTicketTests(unittest.TestCase):
    def test_redirect_ticket_withdraws_before_consume(self):
        obj = agent()
        obj._model_request_active.set()
        self.assertTrue(obj.redirect("A", ticket="a"))
        self.assertTrue(obj.withdraw_steer("a"))
        self.assertIsNone(obj._pending_redirect)

    def test_consumed_redirect_ticket_cannot_also_be_withdrawn(self):
        obj = agent()
        obj._model_request_active.set()
        self.assertTrue(obj.redirect("A", ticket="a"))
        self.assertEqual(obj._drain_pending_redirect(), "A")
        self.assertFalse(obj.withdraw_steer("a"))  # consumed: never requeue as well

    def test_second_correction_after_abort_joins_the_same_buffer(self):
        obj = agent()
        obj._model_request_active.set()
        obj.redirect("A")
        obj._model_request_active.clear()
        obj.clear_interrupt(preserve_redirect=True)
        self.assertTrue(obj.redirect("B"))  # no re-armed stop flag, same rebuild
        self.assertFalse(obj._interrupt_requested)
        self.assertEqual(obj._drain_pending_redirect().count("Additional user correction"), 1)

    def test_turn_end_keeps_collected_redirect_for_the_next_turn(self):
        obj = agent()
        obj._model_request_active.set()
        obj.redirect("A", ticket="a")
        obj._model_request_active.clear()
        obj.redirect("B")
        obj.clear_interrupt()
        kept = obj._drain_pending_steer()
        self.assertIn("A", kept)
        self.assertIn("B", kept)
        self.assertIsNone(obj._pending_redirect)

    def test_hard_cancel_drops_collected_redirect(self):
        obj = agent()
        obj._model_request_active.set()
        obj.redirect("A")
        obj._model_request_active.clear()
        obj.redirect("B")
        obj.clear_interrupt(hard_cancel=True)
        self.assertIsNone(obj._pending_redirect)
        self.assertFalse(obj._drain_pending_steer())

    def test_tool_period_redirect_degrades_to_a_ticketed_steer(self):
        obj = agent()
        obj._executing_tools = True
        self.assertTrue(obj.redirect("A", ticket="a"))
        self.assertEqual(obj._drain_pending_steer(), "A")
        self.assertFalse(obj.withdraw_steer("a"))  # the steer drain consumed it


class CheckpointTests(unittest.TestCase):
    def test_before_tools_consumes_the_steer_buffer_first(self):
        obj = agent()
        obj.steer("correction", ticket="c")
        messages = [{"role": "user", "content": "task"}]
        with patch("agent.conversation_loop._apply_active_turn_redirect") as apply:
            self.assertTrue(checkpoint_input(obj, messages, stage="before_tools"))
        apply.assert_called_once()
        self.assertIsNone(obj._pending_steer)

    def test_before_request_appends_without_touching_the_cached_prefix(self):
        messages = [{"role": "user", "content": "task"},
                    {"role": "assistant", "content": ""},
                    {"role": "tool", "content": "done", "tool_call_id": "t"}]
        prefix = [dict(row) for row in messages]
        obj = agent()
        obj.steer("correction")
        self.assertTrue(checkpoint_input(obj, messages, stage="before_request",
                                        text=obj._drain_pending_steer()))
        self.assertEqual(messages[:len(prefix)], prefix)
        self.assertIn("correction", messages[-1]["content"])

    def test_before_request_false_without_text(self):
        self.assertFalse(checkpoint_input(agent(), [], stage="before_request", text=None))


if __name__ == "__main__":
    unittest.main()

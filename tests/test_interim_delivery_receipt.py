"""Regression tests for asynchronous interim-message delivery receipts."""
import unittest
from gateway.stream_consumer import InterimDeliveryReceipt
class InterimDeliveryReceiptTests(unittest.TestCase):
    def test_states(self):
        for state in ("delivered","failed","ambiguous"):
            r=InterimDeliveryReceipt(); seen=[]; r.add_done_callback(seen.append); r.resolve(state); r.resolve("failed")
            self.assertEqual(r.state,state); self.assertEqual(seen,[state])

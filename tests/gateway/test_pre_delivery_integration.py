from __future__ import annotations

from types import SimpleNamespace
import unittest
from typing import Any, cast

from gateway.pre_delivery import PreDeliveryGate
from gateway.run_turn_runner import TurnRunner


class FakeStreamConsumer:
    def __init__(self):
        self.calls = []
        self.suppressed = False

    def suppress_final_delivery(self):
        self.suppressed = True

    def finish(self, *args):
        self.calls.append(args)


class PreDeliveryTurnIntegrationTests(unittest.TestCase):
    def make_turn(self, gate):
        runner = SimpleNamespace(pre_delivery_gate=gate)
        ctx = SimpleNamespace(
            result_holder=[None],
            source=SimpleNamespace(platform="test", chat_id="chat-1"),
        )
        return TurnRunner(cast(Any, runner), cast(Any, ctx)), ctx

    def test_strict_approval_replaces_final_payload_before_finish(self):
        def policy(*, final_text, metadata):
            return {"allowed": True, "final_text": "approved", "status": "passed", "evidence_ref": "run:approved"}

        turn, ctx = self.make_turn(PreDeliveryGate(mode="strict", policy=policy))
        stream = FakeStreamConsumer()
        result = {"final_response": "unvalidated", "messages": [], "completed": True}
        turn._finish_stream_consumer(result, [], stream)
        self.assertEqual(stream.calls, [("approved",)])
        self.assertEqual(result["final_response"], "approved")
        self.assertFalse(stream.suppressed)
        self.assertIs(ctx.result_holder[0], result)

    def test_strict_rejection_suppresses_stream_and_returns_safe_notice(self):
        def policy(*, final_text, metadata):
            return {"allowed": False, "final_text": None, "status": "blocked", "reason": "unsafe"}

        turn, _ = self.make_turn(PreDeliveryGate(mode="strict", policy=policy))
        stream = FakeStreamConsumer()
        result = {"final_response": "unvalidated", "messages": [], "completed": True}
        turn._finish_stream_consumer(result, [], stream)
        self.assertEqual(stream.calls, [()])
        self.assertTrue(stream.suppressed)
        self.assertTrue(result["pre_delivery_blocked"])
        self.assertNotEqual(result["final_response"], "unvalidated")
        self.assertIn("withheld", result["final_response"])

    def test_without_gate_keeps_legacy_finish_payload(self):
        turn, _ = self.make_turn(None)
        stream = FakeStreamConsumer()
        result = {"final_response": "legacy", "messages": [], "completed": True}
        turn._finish_stream_consumer(result, [], stream)
        self.assertEqual(stream.calls, [("legacy",)])
        self.assertFalse(stream.suppressed)

    def test_strict_gate_applies_when_streaming_is_disabled(self):
        def policy(*, final_text, metadata):
            return {"allowed": True, "final_text": "approved", "status": "passed"}

        turn, _ = self.make_turn(PreDeliveryGate(mode="strict", policy=policy))
        result = {"final_response": "unvalidated", "messages": [], "completed": True}
        turn._finish_stream_consumer(result, [], None)
        self.assertEqual(result["final_response"], "approved")


if __name__ == "__main__":
    unittest.main()

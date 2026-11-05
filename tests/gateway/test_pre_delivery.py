from __future__ import annotations

import unittest

from gateway.pre_delivery import PreDeliveryGate


class PreDeliveryGateTests(unittest.IsolatedAsyncioTestCase):
    async def test_legacy_preserves_response_without_policy(self):
        decision = await PreDeliveryGate(mode="legacy").evaluate(final_text="answer")
        self.assertTrue(decision.allowed)
        self.assertEqual(decision.final_text, "answer")
        self.assertEqual(decision.status, "passed")

    async def test_strict_returns_approved_revised_text(self):
        def policy(*, final_text, metadata):
            return {"allowed": True, "final_text": "revised", "status": "passed", "evidence_ref": "run:1"}

        decision = await PreDeliveryGate(mode="strict", policy=policy).evaluate(final_text="draft")
        self.assertTrue(decision.allowed)
        self.assertEqual(decision.final_text, "revised")
        self.assertEqual(decision.evidence_ref, "run:1")

    async def test_strict_blocks_without_fallback(self):
        def policy(*, final_text, metadata):
            return {"allowed": False, "final_text": None, "status": "blocked", "reason": "unsafe"}

        decision = await PreDeliveryGate(mode="strict", policy=policy).evaluate(final_text="unsafe")
        self.assertFalse(decision.allowed)
        self.assertIsNone(decision.final_text)
        self.assertEqual(decision.status, "blocked")

    async def test_strict_malformed_policy_is_inconclusive_and_stops(self):
        def policy(*, final_text, metadata):
            return {"allowed": True, "final_text": "unverified"}

        decision = await PreDeliveryGate(mode="strict", policy=policy).evaluate(final_text="draft")
        self.assertFalse(decision.allowed)
        self.assertIsNone(decision.final_text)
        self.assertEqual(decision.status, "inconclusive")

    async def test_async_policy_is_supported(self):
        async def policy(*, final_text, metadata):
            return {"allowed": True, "final_text": final_text, "status": "passed"}

        decision = await PreDeliveryGate(mode="strict", policy=policy).evaluate(final_text="answer")
        self.assertTrue(decision.allowed)

    async def test_shadow_records_decision_but_preserves_original_text(self):
        def policy(*, final_text, metadata):
            return {"allowed": False, "final_text": None, "status": "blocked", "evidence_ref": "run:shadow"}

        decision = await PreDeliveryGate(mode="shadow", policy=policy).evaluate(final_text="draft")
        self.assertTrue(decision.allowed)
        self.assertEqual(decision.final_text, "draft")
        self.assertEqual(decision.status, "blocked")
        self.assertEqual(decision.evidence_ref, "run:shadow")

    async def test_policy_exception_is_safe_stop_in_strict(self):
        def policy(*, final_text, metadata):
            raise RuntimeError("validator unavailable")

        decision = await PreDeliveryGate(mode="strict", policy=policy).evaluate(final_text="draft")
        self.assertFalse(decision.allowed)
        self.assertIsNone(decision.final_text)
        self.assertEqual(decision.status, "inconclusive")


if __name__ == "__main__":
    unittest.main()

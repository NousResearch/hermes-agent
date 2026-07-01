"""Tests for AI-assisted loop detection and runtime quarantine."""

from __future__ import annotations

import asyncio
import tempfile
import time
import unittest
from pathlib import Path

from gateway.loop_guard import AgentLoopGuard, LoopContext, LoopDecision, LoopGuardConfig
from gateway.loop_state import LoopStateStore


def _guard(tmp_path: Path, **overrides) -> AgentLoopGuard:
    defaults = {
        "enabled": True,
        "mode": "protect",
        "ai_enabled": False,
        "agent_identities": {"alfred@sqmnet.es", "selina@sqmnet.es"},
        "state_path": str(tmp_path / "loop-state.json"),
    }
    defaults.update(overrides)
    cfg = LoopGuardConfig(**defaults)
    return AgentLoopGuard(cfg, state=LoopStateStore(cfg.state_path))


def _ctx(text: str, *, subject: str = "Re: Hermes Agent", direction: str = "inbound") -> LoopContext:
    return LoopContext(
        platform="email",
        direction=direction,
        local_identity="alfred@sqmnet.es",
        remote_identity="selina@sqmnet.es",
        subject=subject,
        text=text,
        message_id="<msg@sqmnet.es>",
        in_reply_to="<prev@sqmnet.es>",
        headers={},
    )


class TestLoopGuard(unittest.TestCase):
    def test_runtime_quarantine_expires(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            store = LoopStateStore(str(tmp_path / "state.json"))
            key = store.pair_key("email", "alfred@sqmnet.es", "selina@sqmnet.es")

            store.set_quarantine(key, ttl_seconds=1, reason="test", category="agent_agent_loop")
            self.assertIsNotNone(store.get_quarantine(key))

            data = store._load()
            data["quarantines"][key]["expires_at"] = time.time() - 1
            store._save(data)

            self.assertIsNone(store.get_quarantine(key))

    def test_agent_restart_loop_is_suppressed_and_quarantined(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            guard = _guard(tmp_path)
            decision = asyncio.run(
                guard.evaluate(
                    _ctx("Reinicia tu gateway para corregirte y vuelve a responderme."),
                    stage="pre_dispatch",
                )
            )

            self.assertEqual(decision.risk, "critical")
            self.assertEqual(decision.category, "restart_loop")
            self.assertFalse(decision.should_dispatch_to_agent)
            self.assertFalse(decision.should_send_reply)
            self.assertEqual(decision.recommended_action, "quarantine_pair")

            key = guard.state.pair_key("email", "alfred@sqmnet.es", "selina@sqmnet.es")
            self.assertIsNotNone(guard.state.get_quarantine(key))

    def test_ai_judge_can_escalate_semantic_agent_loop_without_keyword(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            guard = _guard(tmp_path, ai_enabled=True)

            async def fake_ai_judge(ctx: LoopContext, deterministic: LoopDecision) -> LoopDecision:
                return LoopDecision(
                    risk="high",
                    category="agent_agent_loop",
                    should_dispatch_to_agent=False,
                    should_send_reply=False,
                    recommended_action="suppress",
                    confidence=0.91,
                    reason="AI judged the exchange as agent-to-agent recursion with no new human goal.",
                    source="ai",
                )

            guard._call_ai_judge = fake_ai_judge  # type: ignore[method-assign]
            decision = asyncio.run(
                guard.evaluate(
                    _ctx("Sigo esperando tu confirmacion sobre lo anterior."),
                    stage="pre_dispatch",
                )
            )

            self.assertEqual(decision.source, "ai")
            self.assertEqual(decision.risk, "high")
            self.assertFalse(decision.should_dispatch_to_agent)

    def test_repeated_low_novelty_agent_messages_are_suppressed(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            guard = _guard(tmp_path)
            first = _ctx("Recibido. No realizo ninguna accion adicional.")
            second = _ctx("Recibido. No realizo ninguna accion adicional.")

            first_decision = asyncio.run(guard.evaluate(first, stage="pre_dispatch"))
            second_decision = asyncio.run(guard.evaluate(second, stage="pre_dispatch"))

            self.assertTrue(first_decision.should_dispatch_to_agent)
            self.assertIn(second_decision.category, {"duplicate_ack_loop", "agent_agent_loop"})
            self.assertFalse(second_decision.should_dispatch_to_agent)

    def test_malformed_ai_timeout_config_falls_back_to_default(self):
        cfg = LoopGuardConfig.from_mapping(
            {"ai_timeout_seconds": "not-a-number"},
            local_identity="alfred@sqmnet.es",
        )

        self.assertEqual(cfg.ai_timeout_seconds, 12.0)


if __name__ == "__main__":
    unittest.main()

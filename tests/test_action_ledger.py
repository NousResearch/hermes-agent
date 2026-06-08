"""
Tests for the action ledger module.
"""
import json
import tempfile
import time
from pathlib import Path
from unittest import TestCase
from agent.action_ledger import ActionLedger, ContextAction


class TestContextAction(TestCase):
    """Tests for ContextAction dataclass."""
    
    def test_to_dict(self):
        action = ContextAction(
            timestamp=1234567890.0,
            action_type="compression",
            session_id="test-123",
            context_tokens=50000,
            compression_ratio=0.3
        )
        data = action.to_dict()
        self.assertEqual(data["action_type"], "compression")
        self.assertEqual(data["session_id"], "test-123")
        self.assertEqual(data["context_tokens"], 50000)
        self.assertEqual(data["compression_ratio"], 0.3)
    
    def test_from_dict(self):
        data = {
            "timestamp": 1234567890.0,
            "action_type": "compaction",
            "session_id": "test-456",
            "message_id": "msg-1",
            "context_tokens": 60000,
            "compression_ratio": 0.25,
            "summary_tokens": 5000,
            "reason": "Test reason",
            "details": {"key": "value"}
        }
        action = ContextAction.from_dict(data)
        self.assertEqual(action.action_type, "compaction")
        self.assertEqual(action.session_id, "test-456")
        self.assertEqual(action.context_tokens, 60000)
        self.assertEqual(action.details["key"], "value")
    
    def test_round_trip(self):
        original = ContextAction(
            timestamp=time.time(),
            action_type="test",
            session_id="session-1",
            context_tokens=10000,
            compression_ratio=0.5,
            reason="Round trip test"
        )
        restored = ContextAction.from_dict(original.to_dict())
        self.assertEqual(original.action_type, restored.action_type)
        self.assertEqual(original.session_id, restored.session_id)
        self.assertEqual(original.context_tokens, restored.context_tokens)


class TestActionLedger(TestCase):
    """Tests for ActionLedger class."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.ledger_file = Path(self.temp_dir) / "test_ledger.json"
        self.ledger = ActionLedger(ledger_file=str(self.ledger_file))
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_record_action(self):
        self.ledger.record_action(
            action_type="compression",
            session_id="test-session",
            context_tokens=50000,
            compression_ratio=0.3
        )
        self.assertEqual(len(self.ledger.actions), 1)
        self.assertEqual(self.ledger.actions[0].action_type, "compression")
        self.assertEqual(self.ledger.actions[0].session_id, "test-session")
    
    def test_record_pre_compression(self):
        self.ledger.record_pre_compression(
            n_messages=100,
            current_tokens=80000,
            session_id="test-session"
        )
        self.assertEqual(len(self.ledger.actions), 1)
        self.assertEqual(self.ledger.actions[0].action_type, "pre_compression")
        self.assertEqual(self.ledger.actions[0].details["n_messages"], 100)
        self.assertEqual(self.ledger.actions[0].context_tokens, 80000)
    
    def test_record_post_compression(self):
        self.ledger.record_post_compression(
            n_messages=30,
            new_tokens=20000,
            saved_tokens=60000,
            savings_pct=75.0,
            summary_used=True,
            fallback_used=False,
            session_id="test-session"
        )
        self.assertEqual(len(self.ledger.actions), 1)
        self.assertEqual(self.ledger.actions[0].action_type, "post_compression")
        self.assertEqual(self.ledger.actions[0].details["savings_pct"], 75.0)
        self.assertEqual(self.ledger.actions[0].details["summary_used"], True)
    
    def test_get_session_actions(self):
        self.ledger.record_action(action_type="test", session_id="session-a")
        self.ledger.record_action(action_type="test", session_id="session-b")
        self.ledger.record_action(action_type="test", session_id="session-a")
        
        actions_a = self.ledger.get_session_actions("session-a")
        self.assertEqual(len(actions_a), 2)
        self.assertTrue(all(a.session_id == "session-a" for a in actions_a))
    
    def test_get_recent_actions(self):
        for i in range(10):
            self.ledger.record_action(action_type=f"test-{i}", session_id="test")
        
        recent = self.ledger.get_recent_actions(limit=5)
        self.assertEqual(len(recent), 5)
    
    def test_get_compaction_actions(self):
        self.ledger.record_action(action_type="compaction", session_id="test")
        self.ledger.record_action(action_type="compression", session_id="test")
        self.ledger.record_action(action_type="other", session_id="test")
        
        compactions = self.ledger.get_compaction_actions("test")
        self.assertEqual(len(compactions), 2)
    
    def test_get_compaction_summary(self):
        self.ledger.record_action(
            action_type="compression", session_id="test",
            context_tokens=50000, compression_ratio=0.3
        )
        self.ledger.record_action(
            action_type="compression", session_id="test",
            context_tokens=60000, compression_ratio=0.4
        )
        
        summary = self.ledger.get_compaction_summary("test")
        self.assertEqual(summary["total_compactions"], 2)
        self.assertEqual(summary["total_context_tokens"], 110000)
        self.assertAlmostEqual(summary["average_compression_ratio"], 0.35, places=2)
    
    def test_clear_session_actions(self):
        self.ledger.record_action(action_type="test", session_id="session-a")
        self.ledger.record_action(action_type="test", session_id="session-b")
        
        self.ledger.clear_session_actions("session-a")
        self.assertEqual(len(self.ledger.actions), 1)
        self.assertEqual(self.ledger.actions[0].session_id, "session-b")
    
    def test_persists_to_file(self):
        self.ledger.record_action(action_type="test", session_id="test")
        
        # Create a new ledger instance pointing to the same file
        new_ledger = ActionLedger(ledger_file=str(self.ledger_file))
        self.assertEqual(len(new_ledger.actions), 1)
        self.assertEqual(new_ledger.actions[0].action_type, "test")
    
    def test_loads_empty_file(self):
        # Create a new ledger without saving anything
        ledger_file = Path(self.temp_dir) / "empty_ledger.json"
        ledger = ActionLedger(ledger_file=str(ledger_file))
        self.assertEqual(len(ledger.actions), 0)
    
    def test_compression_ratio_calculation(self):
        # Test the compression ratio calculation in record_post_compression
        self.ledger.record_post_compression(
            n_messages=30,
            new_tokens=20000,
            saved_tokens=60000,
            savings_pct=75.0,
            summary_used=True,
            fallback_used=False,
            session_id="test"
        )
        action = self.ledger.actions[-1]
        # Ratio should be 1.0 - (75.0 / 100.0) = 0.25
        self.assertAlmostEqual(action.compression_ratio, 0.25, places=2)


if __name__ == "__main__":
    import unittest
    unittest.main()
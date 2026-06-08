"""
Tests for the compaction guard module.
"""
import time
from unittest import TestCase
from agent.compaction_guard import CompactionGuard, CompactionGuardConfig


class TestCompactionGuardConfig(TestCase):
    """Tests for CompactionGuardConfig defaults."""
    
    def test_defaults(self):
        config = CompactionGuardConfig()
        self.assertEqual(config.phone_min_tail_messages, 30)
        self.assertEqual(config.phone_min_tail_tokens, 40000)
        self.assertEqual(config.phone_threshold_percent, 0.60)
        self.assertEqual(config.min_compaction_interval, 120.0)
        self.assertEqual(config.max_auto_compressions, 5)
        self.assertIn("telegram", config.phone_platforms)
        self.assertIn("whatsapp", config.phone_platforms)


class TestCompactionGuard(TestCase):
    """Tests for CompactionGuard class."""
    
    def setUp(self):
        self.guard = CompactionGuard()
    
    def test_is_phone_platform_telegram(self):
        self.assertTrue(self.guard.is_phone_platform("telegram"))
    
    def test_is_phone_platform_whatsapp(self):
        self.assertTrue(self.guard.is_phone_platform("whatsapp"))
    
    def test_is_phone_platform_web(self):
        self.assertFalse(self.guard.is_phone_platform("web"))
    
    def test_is_phone_platform_none(self):
        self.assertFalse(self.guard.is_phone_platform(None))
    
    def test_is_phone_platform_case_insensitive(self):
        self.assertTrue(self.guard.is_phone_platform("Telegram"))
        self.assertTrue(self.guard.is_phone_platform("WHATSAPP"))
    
    def test_should_block_compaction_cooldown(self):
        # First compaction should not be blocked
        should_block, reason = self.guard.should_block_compaction(
            platform="web",
            messages=[{"role": "user", "content": "test"}] * 100,
            current_tokens=50000
        )
        self.assertFalse(should_block)
        
        # Record compaction
        self.guard.record_compaction()
        
        # Second compaction immediately should be blocked
        should_block, reason = self.guard.should_block_compaction(
            platform="web",
            messages=[{"role": "user", "content": "test"}] * 100,
            current_tokens=50000
        )
        self.assertTrue(should_block)
        self.assertIn("cooldown", reason.lower())
    
    def test_should_block_compaction_max_auto(self):
        # Create a guard with low max and no cooldown for testing
        config = CompactionGuardConfig(max_auto_compressions=2, min_compaction_interval=0)
        guard = CompactionGuard(config)
        
        # Record 2 compressions
        guard.record_compaction()
        guard.record_compaction()
        
        # Next should be blocked for phone
        should_block, reason = guard.should_block_compaction(
            platform="telegram",
            messages=[{"role": "user", "content": "test"}] * 100,
            current_tokens=50000
        )
        self.assertTrue(should_block)
        self.assertIn("Max auto-compressions", reason)
        
        # But not for web
        should_block, reason = guard.should_block_compaction(
            platform="web",
            messages=[{"role": "user", "content": "test"}] * 100,
            current_tokens=50000
        )
        self.assertFalse(should_block)
    
    def test_should_block_compaction_min_tail_messages(self):
        config = CompactionGuardConfig(phone_min_tail_messages=50)
        guard = CompactionGuard(config)
        
        # Too few messages for phone
        should_block, reason = guard.should_block_compaction(
            platform="telegram",
            messages=[{"role": "user", "content": "test"}] * 30,
            current_tokens=50000
        )
        self.assertTrue(should_block)
        self.assertIn("Too few messages", reason)
        
        # Enough messages
        should_block, reason = guard.should_block_compaction(
            platform="telegram",
            messages=[{"role": "user", "content": "test"}] * 100,
            current_tokens=50000
        )
        self.assertFalse(should_block)
    
    def test_get_phone_threshold(self):
        # Phone session should use higher threshold
        phone_threshold = self.guard.get_phone_threshold(128000, "telegram")
        web_threshold = self.guard.get_phone_threshold(128000, "web")
        
        self.assertGreater(phone_threshold, web_threshold)
        self.assertEqual(phone_threshold, int(128000 * 0.60))
        self.assertEqual(web_threshold, int(128000 * 0.50))
    
    def test_get_phone_tail_budget(self):
        # Phone session should use larger tail budget
        phone_budget = self.guard.get_phone_tail_budget(76800, "telegram")
        web_budget = self.guard.get_phone_tail_budget(76800, "web")
        
        self.assertGreaterEqual(phone_budget, web_budget)
        self.assertGreaterEqual(phone_budget, 40000)  # Minimum phone tail
    
    def test_record_compaction(self):
        self.guard.record_compaction()
        self.assertEqual(self.guard._compaction_count, 1)
        self.assertGreater(self.guard._last_compaction_time, 0)
    
    def test_reset(self):
        self.guard.record_compaction()
        self.guard.record_compaction()
        
        self.guard.reset()
        self.assertEqual(self.guard._compaction_count, 0)
        self.assertEqual(self.guard._last_compaction_time, 0)
        self.assertTrue(self.guard._session_active)
    
    def test_get_status(self):
        status = self.guard.get_status()
        self.assertIn("compaction_count", status)
        self.assertIn("session_active", status)
        self.assertIn("time_since_last_compaction", status)
        self.assertIn("max_auto_compressions", status)
        self.assertIn("min_compaction_interval", status)
        
        self.assertEqual(status["compaction_count"], 0)
        self.assertTrue(status["session_active"])


class TestCreatePhoneCompactionGuard(TestCase):
    """Tests for factory function."""
    
    def test_create_default(self):
        guard = CompactionGuard()
        self.assertIsInstance(guard, CompactionGuard)
    
    def test_create_with_config(self):
        config = CompactionGuardConfig(phone_min_tail_messages=20)
        guard = CompactionGuard(config)
        self.assertEqual(guard.config.phone_min_tail_messages, 20)


if __name__ == "__main__":
    import unittest
    unittest.main()
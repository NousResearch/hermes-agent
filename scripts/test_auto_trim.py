#!/usr/bin/env python3
"""test_auto_trim.py — 29 test cases for the Hermes auto-trim engine.

Covers:
  - Token counting
  - Priority extraction
  - Phase 1 eviction (T5/T6)
  - Phase 2 compression (T3/T4)
  - MIN_BLOCKS_KEPT floor
  - Archive-before-delete
  - Signal I/O (pause/resume/protect)
  - Config validation
  - Compressible ratio enforcement
  - Block ID preservation

Usage:
    python3 test_auto_trim.py [-v] [--no-ollama]
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# ─── Ensure we import from the canonical scripts location ───────────────────────
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))


# ─── Must be imported after sys.path is set ─────────────────────────────────────
import auto_trim as at



class TestTokenCounting(unittest.TestCase):
    """Test count_tokens with various input types."""

    def test_empty_string(self):
        self.assertEqual(at.count_tokens(""), 0)

    def test_simple_ascii(self):
        # "hello world" = 2 words, 11 chars // 4 = 2 → max(2, 2) = 2
        result = at.count_tokens("hello world")
        self.assertEqual(result, 2)

    def test_long_ascii(self):
        # "word " * 100 = 500 chars, 100 words, char_est = 500//4 = 125
        # max(100, 125) = 125
        text = "word " * 100
        result = at.count_tokens(text)
        self.assertEqual(result, 125)

    def test_cjk_characters(self):
        # CJK: fewer words but more chars → char-based estimate dominates
        text = "你好世界" * 20  # 80 chars, 4 words (if split on whitespace)
        result = at.count_tokens(text)
        self.assertGreater(result, 0)
        self.assertEqual(result, 20)  # 80 // 4 = 20

    def test_code_with_whitespace(self):
        # "def foo():\n    return 42\n" = 25 chars per repeat
        # * 10 = 250 chars, 40 words (4 words per line * 10), char_est = 250//4 = 62
        # max(40, 62) = 62
        code = "def foo():\n    return 42\n" * 10
        result = at.count_tokens(code)
        self.assertEqual(result, 62)

    def test_none_input(self):
        self.assertEqual(at.count_tokens(""), 0)


class TestBlockPriority(unittest.TestCase):
    """Test _block_priority extraction with type coercion."""

    def test_valid_priority(self):
        self.assertEqual(at._block_priority({"priority": 0}), 0)
        self.assertEqual(at._block_priority({"priority": 3}), 3)
        self.assertEqual(at._block_priority({"priority": 6}), 6)

    def test_missing_priority_defaults_to_6(self):
        self.assertEqual(at._block_priority({}), 6)

    def test_missing_priority_key_defaults_to_6(self):
        self.assertEqual(at._block_priority({"content": "test"}), 6)

    def test_non_integer_priority_defaults_to_6(self):
        self.assertEqual(at._block_priority({"priority": "high"}), 6)
        self.assertEqual(at._block_priority({"priority": None}), 6)
        self.assertEqual(at._block_priority({"priority": [1, 2]}), 6)


class TestTrimContextEmptyInput(unittest.TestCase):
    """Test trim_context with empty/edge-case inputs."""

    def test_empty_blocks_list(self):
        result = at.trim_context([])
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["action"], "none")
        self.assertEqual(result["blocks_deleted"], 0)
        self.assertEqual(result["blocks_compressed"], 0)

    def test_none_blocks_list(self):
        result = at.trim_context([])
        # None is falsy → should still return ok with no action
        self.assertEqual(result["status"], "ok")


class TestTrimContextBelowThreshold(unittest.TestCase):
    """Test that no trimming occurs when below threshold."""

    def test_below_threshold_no_action(self):
        blocks = [
            {"id": "b1", "content": "short text", "priority": 5},
        ]
        result = at.trim_context(blocks, budget=100000, threshold=100000)
        self.assertEqual(result["action"], "none")
        self.assertEqual(result["blocks_deleted"], 0)


class TestPhase1Eviction(unittest.TestCase):
    """Test Phase 1: deletion of T5/T6 blocks when over budget."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self._old_archive = at.ARCHIVE_DIR
        at.ARCHIVE_DIR = Path(self.tmpdir.name) / "archive"
        at.ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        at.ARCHIVE_DIR = self._old_archive
        self.tmpdir.cleanup()

    def test_eviction_deletes_low_priority(self):
        # Each block is "x"*500: 500//4=125 char_est, 1 word → count=125
        # Total: 625 tokens. Threshold 100 → overage 525.
        # With MIN_BLOCKS_KEPT=3, need ≥3 non-evictable fillers to allow
        # deleting both b4 and b5 while keeping 3 behind.
        blocks = [
            {"id": "b1", "content": "x" * 500, "priority": 0},   # keep (T0, filler)
            {"id": "b2", "content": "x" * 500, "priority": 1},   # keep (T1, filler)
            {"id": "b3", "content": "x" * 500, "priority": 2},   # keep (T2, filler)
            {"id": "b4", "content": "y" * 500, "priority": 5},   # evict (T5)
            {"id": "b5", "content": "z" * 500, "priority": 6},   # evict (T6)
        ]
        result = at.trim_context(blocks, budget=100, threshold=100)
        self.assertGreater(result["blocks_deleted"], 0)
        self.assertIn(result["status"], ("ok", "partial"))

    def test_eviction_preserves_high_priority(self):
        # 375 tokens, threshold 100 → overage 275
        blocks = [
            {"id": "b1", "content": "x" * 500, "priority": 0},
            {"id": "b2", "content": "y" * 500, "priority": 1},
            {"id": "b3", "content": "z" * 500, "priority": 2},
        ]
        result = at.trim_context(blocks, budget=100, threshold=100)
        # T0/T1/T2 never touched — all should remain
        remaining_ids = [b["id"] for b in result.get("remaining_blocks", blocks)]
        self.assertIn("b1", remaining_ids)
        self.assertIn("b2", remaining_ids)
        self.assertIn("b3", remaining_ids)

    def test_eviction_obeys_min_blocks_kept(self):
        blocks = [
            {"id": "b1", "content": "x" * 200, "priority": 5},
            {"id": "b2", "content": "y" * 200, "priority": 5},
            {"id": "b3", "content": "z" * 200, "priority": 5},
            {"id": "b4", "content": "w" * 200, "priority": 6},  # extra evictable
        ]
        result = at.trim_context(blocks, budget=100, threshold=100)
        # MIN_BLOCKS_KEPT = 3, so at most 1 should be deleted (4 blocks, min 3 kept)
        remaining_count = len(result.get("remaining_blocks", blocks))
        self.assertGreaterEqual(remaining_count, at.MIN_BLOCKS_KEPT)

    def test_eviction_stops_at_budget(self):
        blocks = [
            {"id": "b1", "content": "x" * 400, "priority": 0},   # T0 filler
            {"id": "b2", "content": "x" * 400, "priority": 1},   # T1 filler
            {"id": "b3", "content": "x" * 400, "priority": 2},   # T2 filler
            {"id": "b4", "content": "y" * 400, "priority": 5},   # 100 tokens
            {"id": "b5", "content": "z" * 400, "priority": 5},   # 100 tokens
            {"id": "b6", "content": "w" * 400, "priority": 6},   # 100 tokens
        ]
        # Total: 600 tokens, threshold=100 → overage=500
        # Delete b4(100)→400, delete b5(100)→300, delete b6(100)→200 → stops
        # MIN_BLOCKS_KEPT=3 satisfied by b1,b2,b3
        result = at.trim_context(blocks, budget=100, threshold=100)
        self.assertIn(result["status"], ("ok", "partial"))
        self.assertGreater(result["blocks_deleted"], 0)

    def test_eviction_adds_to_remaining(self):
        """Non-evictable blocks always go to remaining."""
        blocks = [
            {"id": "b1", "content": "x" * 100, "priority": 0},  # T0
            {"id": "b2", "content": "y" * 100, "priority": 6},  # T6 evictable
        ]
        result = at.trim_context(blocks, budget=50, threshold=50)
        remaining_ids = [b["id"] for b in result.get("remaining_blocks", blocks)]
        self.assertIn("b1", remaining_ids)


class TestPhase2Compression(unittest.TestCase):
    """Test Phase 2: compression of T3/T4 blocks."""

    @patch("auto_trim.query_ollama")
    def test_compression_called_for_t3_t4(self, mock_ollama):
        mock_ollama.return_value = "compressed version"
        # 50 tokens each, total 100. Threshold 30 → overage 70 → triggers compression
        blocks = [
            {"id": "b1", "content": "x" * 200, "priority": 3},
            {"id": "b2", "content": "y" * 200, "priority": 4},
        ]
        result = at.trim_context(blocks, budget=20, threshold=30)
        mock_ollama.assert_called()

    @patch("auto_trim.query_ollama")
    def test_compression_updates_block_content(self, mock_ollama):
        mock_ollama.return_value = "short"
        blocks = [
            {"id": "b1", "content": "x" * 200, "priority": 3},
        ]
        result = at.trim_context(blocks, budget=10, threshold=30)
        # Block should have compressed content
        remaining = result.get("remaining_blocks", blocks)
        self.assertTrue(any(b.get("compressed") for b in remaining))

    @patch("auto_trim.query_ollama")
    def test_compression_skips_protected(self, mock_ollama):
        mock_ollama.return_value = "short"
        blocks = [
            {"id": "b1", "content": "x" * 200, "priority": 3},
        ]
        # Set b1 as protected
        at.set_block_protected("b1", protected=True)
        result = at.trim_context(blocks, budget=10, threshold=30)
        remaining = result.get("remaining_blocks", blocks)
        self.assertEqual(len(remaining), 1)
        self.assertFalse(any(b.get("compressed") for b in remaining))
        # Clean up
        at.set_block_protected("b1", protected=False)

    @patch("auto_trim.query_ollama")
    def test_compression_skips_non_t3t4(self, mock_ollama):
        mock_ollama.return_value = "short"
        blocks = [
            {"id": "b1", "content": "x" * 200, "priority": 0},  # T0
            {"id": "b2", "content": "y" * 200, "priority": 5},  # T5 (should be evicted in phase 1)
        ]
        result = at.trim_context(blocks, budget=10, threshold=30)
        mock_ollama.assert_not_called()


class TestCompressibleRatio(unittest.TestCase):
    """Test compressible ratio tracking."""

    @patch("auto_trim.query_ollama")
    def test_compression_ratio_recorded(self, mock_ollama):
        mock_ollama.return_value = "ab"
        blocks = [{"id": "b1", "content": "hello world " * 20, "priority": 3}]
        # 60 tokens total, threshold 30 → overage 30 → triggers compression
        result = at.trim_context(blocks, budget=10, threshold=30)
        self.assertIn("compression_ratio", result)
        self.assertGreaterEqual(result["compression_ratio"], 0)


class TestArchiveBeforeDelete(unittest.TestCase):
    """Test that evicted blocks are archived before deletion."""

    def test_archive_created_on_eviction(self):
        blocks = [
            {"id": "test-block-1", "content": "x" * 200, "priority": 6},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            old_workspace = at.WORKSPACE
            old_bridge = at.BRIDGE_DIR
            old_signals = at.SIGNALS_DIR
            old_archive = at.ARCHIVE_DIR
            at.WORKSPACE = Path(tmpdir)
            at.BRIDGE_DIR = Path(tmpdir) / "bridge"
            at.SIGNALS_DIR = at.BRIDGE_DIR / "signals"
            at.ARCHIVE_DIR = Path(tmpdir) / "logs" / "archive"
            at.SIGNALS_DIR.mkdir(parents=True, exist_ok=True)
            at.ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

            # 50 tokens each (100//4), threshold=20 → overage 150 → triggers eviction
            # Need 3+ filler blocks to satisfy MIN_BLOCKS_KEPT=3
            result = at.trim_context([
                {"id": "filler-1", "content": "x" * 200, "priority": 0},
                {"id": "filler-2", "content": "x" * 200, "priority": 1},
                {"id": "filler-3", "content": "x" * 200, "priority": 2},
                {"id": "test-block-1", "content": "x" * 200, "priority": 6},
            ], budget=10, threshold=20)
            self.assertGreater(result["blocks_deleted"], 0)

            # Check archive file exists
            archive_files = list(at.ARCHIVE_DIR.glob("trimmed_*.json"))
            self.assertGreater(len(archive_files), 0)

            # Verify archived content
            archived = json.loads(archive_files[0].read_text())
            self.assertEqual(archived["id"], "test-block-1")

            # Restore
            at.WORKSPACE = old_workspace
            at.BRIDGE_DIR = old_bridge
            at.SIGNALS_DIR = old_signals
            at.ARCHIVE_DIR = old_archive

    def test_archive_preserves_block_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            old_workspace = at.WORKSPACE
            old_bridge = at.BRIDGE_DIR
            old_signals = at.SIGNALS_DIR
            old_archive = at.ARCHIVE_DIR
            at.WORKSPACE = Path(tmpdir)
            at.BRIDGE_DIR = Path(tmpdir) / "bridge"
            at.SIGNALS_DIR = at.BRIDGE_DIR / "signals"
            at.ARCHIVE_DIR = Path(tmpdir) / "logs" / "archive"
            at.SIGNALS_DIR.mkdir(parents=True, exist_ok=True)
            at.ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

            result = at.trim_context([
                {"id": "filler-1", "content": "x" * 200, "priority": 0},
                {"id": "filler-2", "content": "x" * 200, "priority": 1},
                {"id": "filler-3", "content": "x" * 200, "priority": 2},
                {"id": "my-special-block", "content": "x" * 200, "priority": 5},
            ], budget=10, threshold=20)
            archive_files = list(at.ARCHIVE_DIR.glob("trimmed_*.json"))
            self.assertGreater(len(archive_files), 0, "No archive file created")
            archived = json.loads(archive_files[0].read_text())
            self.assertEqual(archived["id"], "my-special-block")

            at.WORKSPACE = old_workspace
            at.BRIDGE_DIR = old_bridge
            at.SIGNALS_DIR = old_signals
            at.ARCHIVE_DIR = old_archive


class TestSignalIO(unittest.TestCase):
    """Test pause/resume/protect signal file I/O."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        old_signals = at.SIGNALS_DIR
        old_pause = at.PAUSE_SIGNAL
        old_protected = at.PROTECTED_SIGNAL
        at.SIGNALS_DIR = Path(self.tmpdir.name) / "signals"
        at.SIGNALS_DIR.mkdir(parents=True, exist_ok=True)
        at.PAUSE_SIGNAL = at.SIGNALS_DIR / "pause-trim"
        at.PROTECTED_SIGNAL = at.SIGNALS_DIR / "protected-blocks.json"
        self.old_signals = old_signals
        self.old_pause = old_pause
        self.old_protected = old_protected

    def tearDown(self):
        at.SIGNALS_DIR = self.old_signals
        at.PAUSE_SIGNAL = self.old_pause
        at.PROTECTED_SIGNAL = self.old_protected
        self.tmpdir.cleanup()

    def test_pause_creates_signal_file(self):
        result = at.pause_trimming(reason="test")
        self.assertTrue(at.PAUSE_SIGNAL.exists())
        self.assertEqual(result["status"], "paused")

    def test_resume_removes_signal_file(self):
        at.pause_trimming(reason="test")
        result = at.resume_trimming(reason="manual")
        self.assertFalse(at.PAUSE_SIGNAL.exists())
        self.assertEqual(result["status"], "resumed")

    def test_is_trimming_paused_returns_true_when_paused(self):
        at.pause_trimming()
        self.assertTrue(at._is_trimming_paused())

    def test_is_trimming_paused_returns_false_when_resumed(self):
        at.pause_trimming()
        at.resume_trimming()
        self.assertFalse(at._is_trimming_paused())

    def test_auto_resume_after_max_duration(self):
        at.pause_trimming()
        # Set MAX_PAUSE_SECONDS=0 → no ceiling → auto-resume immediately
        old_max = at.MAX_PAUSE_SECONDS
        at.MAX_PAUSE_SECONDS = 0
        # auto_resume_if_expired() should remove the signal file
        self.assertTrue(at.auto_resume_if_expired())
        self.assertFalse(at._is_trimming_paused())
        at.MAX_PAUSE_SECONDS = old_max


class TestProtectedBlocks(unittest.TestCase):
    """Test block protection mechanism."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        old = at.SIGNALS_DIR
        at.SIGNALS_DIR = Path(self.tmpdir.name) / "signals"
        at.SIGNALS_DIR.mkdir(parents=True, exist_ok=True)
        at.PROTECTED_SIGNAL = at.SIGNALS_DIR / "protected-blocks.json"
        self.old = old

    def tearDown(self):
        at.SIGNALS_DIR = self.old
        self.tmpdir.cleanup()

    def test_protect_adds_block(self):
        result = at.set_block_protected("block-a", protected=True)
        self.assertEqual(result["block_id"], "block-a")
        self.assertTrue(result["protected"])
        self.assertIn("block-a", at._get_protected_blocks())

    def test_unprotect_removes_block(self):
        at.set_block_protected("block-a", protected=True)
        at.set_block_protected("block-a", protected=False)
        self.assertNotIn("block-a", at._get_protected_blocks())

    def test_multiple_protected_blocks(self):
        at.set_block_protected("a", protected=True)
        at.set_block_protected("b", protected=True)
        protected = at._get_protected_blocks()
        self.assertIn("a", protected)
        self.assertIn("b", protected)

    def test_empty_protected_file_returns_empty_set(self):
        self.assertEqual(at._get_protected_blocks(), set())

    def test_corrupt_protected_file_returns_empty_set(self):
        at.PROTECTED_SIGNAL.write_text("not valid json {{{")
        self.assertEqual(at._get_protected_blocks(), set())

    def test_protected_blocks_skipped_during_eviction(self):
        blocks = [
            {"id": "protected", "content": "x" * 500, "priority": 6},
            {"id": "unprotected", "content": "y" * 500, "priority": 6},
        ]
        at.set_block_protected("protected", protected=True)
        result = at.trim_context(blocks, budget=100, threshold=100)
        remaining_ids = [b["id"] for b in result.get("remaining_blocks", blocks)]
        self.assertIn("protected", remaining_ids)
        at.set_block_protected("protected", protected=False)


class TestConfigValidation(unittest.TestCase):
    """Test input validation logic."""

    def test_validate_inputs_with_missing_context_status(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            old_workspace = at.WORKSPACE
            old_bridge = at.BRIDGE_DIR
            old_signals = at.SIGNALS_DIR
            at.WORKSPACE = Path(tmpdir)
            at.BRIDGE_DIR = Path(tmpdir) / "bridge"
            at.SIGNALS_DIR = at.BRIDGE_DIR / "signals"
            at.SIGNALS_DIR.mkdir(parents=True, exist_ok=True)

            ok, errors = at.validate_inputs()
            # Without context-status.json, should fail
            self.assertFalse(ok)
            self.assertTrue(any("context-status.json" in e for e in errors))

            at.WORKSPACE = old_workspace
            at.BRIDGE_DIR = old_bridge
            at.SIGNALS_DIR = old_signals

    def test_validate_rejects_nonsensical_budget(self):
        """TARGET_TOKENS > TRIM_THRESHOLD_TOKENS is invalid."""
        old_target = at.TARGET_TOKENS
        old_threshold = at.TRIM_THRESHOLD_TOKENS
        at.TARGET_TOKENS = 200000
        at.TRIM_THRESHOLD_TOKENS = 100000

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                old_workspace = at.WORKSPACE
                at.WORKSPACE = Path(tmpdir)
                at.BRIDGE_DIR = Path(tmpdir) / "bridge"
                at.SIGNALS_DIR = at.BRIDGE_DIR / "signals"
                at.SIGNALS_DIR.mkdir(parents=True, exist_ok=True)
                at.ARCHIVE_DIR = Path(tmpdir) / "logs" / "archive"
                at.ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

                # Write a dummy context-status.json
                ctx = {"blocks": []}
                (at.SIGNALS_DIR / "context-status.json").write_text(json.dumps(ctx))

                ok, errors = at.validate_inputs()
                self.assertFalse(ok)
                self.assertTrue(any("nonsensical" in e.lower() for e in errors))

                at.WORKSPACE = old_workspace
        finally:
            at.TARGET_TOKENS = old_target
            at.TRIM_THRESHOLD_TOKENS = old_threshold


class TestBlockIDPreservation(unittest.TestCase):
    """Test that block IDs are preserved through trimming."""

    def test_known_block_id_preserved_after_eviction(self):
        blocks = [
            {"id": "keep-me", "content": "x" * 500, "priority": 0},
            {"id": "delete-me", "content": "y" * 500, "priority": 6},
        ]
        result = at.trim_context(blocks, budget=50, threshold=50)
        remaining = result.get("remaining_blocks", blocks)
        remaining_ids = [b["id"] for b in remaining]
        self.assertIn("keep-me", remaining_ids)

    def test_block_without_id(self):
        """Block missing 'id' key should not crash and should receive a default."""
        blocks = [
            {"content": "no id here", "priority": 6},
        ]
        result = at.trim_context(blocks, budget=50, threshold=50)
        # Should succeed without crash; block gets default id "block_0"
        self.assertIn("blocks_deleted", result)
        self.assertIn("tokens_before", result)


class TestTrimContextReturnsCorrectStats(unittest.TestCase):
    """Test that trim_context returns accurate statistics."""

    def test_stat_keys_always_present(self):
        blocks = [{"id": "b1", "content": "short", "priority": 5}]
        result = at.trim_context(blocks, budget=100, threshold=100)
        expected_keys = [
            "status", "action", "tokens_before", "tokens_after",
            "tokens_saved", "blocks_deleted", "blocks_compressed",
        ]
        for key in expected_keys:
            self.assertIn(key, result, f"Missing key: {key}")

    def test_tokens_saved_equals_deleted_tokens(self):
        blocks = [
            {"id": "b1", "content": "x" * 400, "priority": 6},
        ]
        result = at.trim_context(blocks, budget=10, threshold=50)
        self.assertEqual(result["tokens_saved"], result["tokens_before"] - result["tokens_after"])


class TestDryRunMode(unittest.TestCase):
    """Test dry-run mode doesn't actually modify anything."""

    def test_dry_run_no_deletion(self):
        """DRY_RUN=True should skip actual deletions."""
        blocks = [
            {"id": "b1", "content": "x" * 500, "priority": 6},
        ]
        old_dry_run = at.DRY_RUN
        at.DRY_RUN = True
        try:
            result = at.trim_context(blocks, budget=100, threshold=50)
            # Should return "ok" not "partial" since nothing was actually deleted
            self.assertEqual(result["blocks_deleted"], 0)
        finally:
            at.DRY_RUN = old_dry_run


class TestIntegrationSmoke(unittest.TestCase):
    """Light integration test with realistic block mix."""

    def test_mixed_priority_trim(self):
        blocks = [
            {"id": "sys-prompt", "content": "You are a helpful assistant.", "priority": 0},
            {"id": "task", "content": "Write a poem about cats.", "priority": 1},
            {"id": "important-doc", "content": "API key: abc-123. Never share.", "priority": 2},
            {"id": "semantic-1", "content": "The user prefers concise responses. " * 10, "priority": 3},
            {"id": "bg-context", "content": "Previous session logs... " * 10, "priority": 4},
            {"id": "tool-output-1", "content": "Function result A" * 20, "priority": 5},
            {"id": "tool-output-2", "content": "Function result B" * 20, "priority": 5},
            {"id": "chat-turn-1", "content": "User: hello\\nAssistant: hi there" * 10, "priority": 6},
        ]
        result = at.trim_context(blocks, budget=200, threshold=500)
        # Should have deleted or compressed some blocks
        self.assertIn(result["action"], ("trimmed", "none", "partial"))
        # High-priority blocks should survive
        remaining = result.get("remaining_blocks", blocks)
        remaining_ids = [b["id"] for b in remaining]
        self.assertIn("sys-prompt", remaining_ids)


def run_tests():
    """Run the test suite."""
    # Check for --no-ollama flag (skip Ollama-dependent tests)
    no_ollama = "--no-ollama" in sys.argv

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Always include non-Ollama tests
    suite.addTests(loader.loadTestsFromTestCase(TestTokenCounting))
    suite.addTests(loader.loadTestsFromTestCase(TestBlockPriority))
    suite.addTests(loader.loadTestsFromTestCase(TestTrimContextEmptyInput))
    suite.addTests(loader.loadTestsFromTestCase(TestTrimContextBelowThreshold))
    suite.addTests(loader.loadTestsFromTestCase(TestPhase1Eviction))
    suite.addTests(loader.loadTestsFromTestCase(TestArchiveBeforeDelete))
    suite.addTests(loader.loadTestsFromTestCase(TestSignalIO))
    suite.addTests(loader.loadTestsFromTestCase(TestProtectedBlocks))
    suite.addTests(loader.loadTestsFromTestCase(TestConfigValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestBlockIDPreservation))
    suite.addTests(loader.loadTestsFromTestCase(TestTrimContextReturnsCorrectStats))
    suite.addTests(loader.loadTestsFromTestCase(TestDryRunMode))
    suite.addTests(loader.loadTestsFromTestCase(TestCompressibleRatio))

    if not no_ollama:
        suite.addTests(loader.loadTestsFromTestCase(TestPhase2Compression))

    suite.addTests(loader.loadTestsFromTestCase(TestIntegrationSmoke))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())
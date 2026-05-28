#!/usr/bin/env python3
"""
test_pause_protection.py — Tests for pause/protection features across
context_orchestrator.py, auto_trim.py, and gateway_integration.py.

Run:  AUTO_TRIM_PATH=/path/to/auto_trim.py python3 -m pytest scripts/test_pause_protection.py -v
     (or just:  python3 -m pytest scripts/test_pause_protection.py -v)
"""
import json
import os
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# ── Locate source trees ────────────────────────────────────────────────────────
# Primary: pipeline root on Linux production  |  Fallback: Mac staging
PIPELINE_ROOT = str(Path("/home/gerald/ai-team-shared/hermes-pipeline").resolve())
MAC_SCRIPTS    = str(Path(os.path.expanduser("~/.hermes/hermes-agent/scripts")).resolve())
MAC_HERMES     = str(Path(os.path.expanduser("~/.hermes")).resolve())
LINUX_SCRIPTS  = PIPELINE_ROOT

sys.path.insert(0, PIPELINE_ROOT)
sys.path.insert(0, MAC_SCRIPTS)
sys.path.insert(0, str(Path(os.path.expanduser("~/.hermes/scripts")).resolve()))

# Resolve AUTO_TRIM_LOC cross-platform: prefer Mac git-repo scripts,
# then Linux pipeline, then Linux prod mirrors, and fall back.
_mac_autotrim     = Path(MAC_SCRIPTS) / "auto_trim.py"
_linux_autotrim   = Path(PIPELINE_ROOT) / "auto_trim.py"
_linux_prod_auto  = Path(MAC_HERMES) / "linux_prod" / "auto_trim.py"
_linux_prod_auto2 = Path(MAC_HERMES) / "linux_production" / "auto_trim.py"

if os.environ.get("AUTO_TRIM_PATH"):
    AUTO_TRIM_LOC = os.environ["AUTO_TRIM_PATH"]
elif _mac_autotrim.exists():
    AUTO_TRIM_LOC = str(_mac_autotrim)
elif _linux_autotrim.exists():
    AUTO_TRIM_LOC = str(_linux_autotrim)
elif _linux_prod_auto.exists():
    AUTO_TRIM_LOC = str(_linux_prod_auto)
elif _linux_prod_auto2.exists():
    AUTO_TRIM_LOC = str(_linux_prod_auto2)
else:
    AUTO_TRIM_LOC = str(_mac_autotrim)
AUTO_TRIM_DIR = str(Path(AUTO_TRIM_LOC).parent.parent)


# ═══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════════════

_BASE_PATCHES_STARTED = False


def _base_context_patches():
    """Return a dict of patchers that suppress memory_palace module-level calls.

    Idempotent — patches are only started once across all test classes.
    """
    global _BASE_PATCHES_STARTED
    targets = {
        "auto_prune": lambda: None,
        "set_working": lambda **kw: None,
        "get_working": lambda: {},
        "clear_working": lambda: None,
        "recall_episodes": lambda **kw: [],
        "recall_facts": lambda *a, **kw: [],
        "store_episode": lambda *a, **kw: None,
        "prune_expired": lambda: {},
        "get_stats": lambda: {},
    }
    patchers = {}
    for name, dummy in targets.items():
        p = patch(f"context_orchestrator.{name}", dummy)
        # Only start if not already active
        if not hasattr(p, '_mock') or p._mock is None:
            p.start()
            patchers[name] = p
        else:
            patchers[name] = p
    _BASE_PATCHES_STARTED = True
    return patchers


def _mock_orch():
    """Create a mock orchestrator with the right pause/protect signatures."""
    m = MagicMock()
    m.pause_trimming.return_value = {"status": "paused", "reason": "mocked"}
    m.resume_trimming.return_value = {"status": "resumed", "reason": "mocked"}
    m.set_block_protected.return_value = {"block_id": "mock", "protected": True}
    m.is_paused.return_value = False
    m.get_pause_info.return_value = None
    m._active_blocks = []
    return m


def _setup_autotrim_module(tmp_path: Path):
    """
    Create a temp directory layout that lets auto_trim._resolve_base_dir()
    find run_bridge.py, then load and return the module.
    """
    (tmp_path / "run_bridge.py").write_text("# dummy — not executed")
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "auto_trim.py").write_text(Path(AUTO_TRIM_LOC).read_text())
    (tmp_path / "bridge" / "signals").mkdir(parents=True)

    spec = __import__("importlib").util.spec_from_file_location(
        "auto_trim_test", str(scripts / "auto_trim.py")
    )
    mod = __import__("importlib").util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ═══════════════════════════════════════════════════════════════════════════════
# context_orchestrator.py — pause / resume / auto-resume
# ═══════════════════════════════════════════════════════════════════════════════

class TestOrchestratorPauseResume(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._patches = _base_context_patches()

    @classmethod
    def tearDownClass(cls):
        for p in cls._patches.values():
            p.stop()

    def setUp(self):
        from context_orchestrator import ContextOrchestrator
        self.orch = ContextOrchestrator("t1")

    # ── Basic lifecycle ────────────────────────────────────────────

    def test_pause_sets_flags(self):
        r = self.orch.pause_trimming(reason="manual")
        self.assertEqual(r["status"], "paused")
        self.assertTrue(self.orch.is_paused())
        self.assertIsNotNone(self.orch._pause_start_time)

    def test_resume_clears_flags(self):
        self.orch.pause_trimming()
        r = self.orch.resume_trimming(reason="done")
        self.assertEqual(r["status"], "resumed")
        self.assertFalse(self.orch.is_paused())
        self.assertIsNone(self.orch._pause_reason)
        self.assertIsNone(self.orch._pause_start_time)

    def test_get_pause_info_paused(self):
        self.orch.pause_trimming(reason="zzz")
        info = self.orch.get_pause_info()
        self.assertEqual(info["reason"], "zzz")
        self.assertIn("elapsed_seconds", info)
        self.assertIn("auto_resume_in_seconds", info)

    def test_get_pause_info_not_paused(self):
        self.assertIsNone(self.orch.get_pause_info())

    # ── Paused trimming is suppressed ──────────────────────────────

    def test_paused_trim_returns_paused(self):
        with patch.object(self.orch, "_est_tokens", return_value=5000):
            self.orch._active_blocks = [
                {"id": "t5", "tier": 5, "content": "x", "tokens": 5000, "persist": False},
            ]
            self.orch.pause_trimming(reason="operator pause")
            r = self.orch.trim_context(current_usage_tokens=100000)

        self.assertEqual(r["status"], "paused")
        self.assertIn("paused", r["message"].lower())
        self.assertEqual(len(self.orch._active_blocks), 1)

    # ── Auto-resume: let real code run (no mock on resume_trimming) ─

    def test_auto_resume_when_pause_exceeded(self):
        """
        When elapsed time > MAX_PAUSE_DURATION, trim_context() calls the
        real resume_trimming(), clears the pause flag, then trims normally.
        """
        self.orch.MAX_PAUSE_DURATION = 1  # 1-second ceiling
        self.orch.pause_trimming(reason="old")

        with patch.object(self.orch, "_est_tokens", return_value=5000):
            self.orch._active_blocks = [
                {"id": "b1", "tier": 5, "content": "test", "tokens": 5000, "persist": False},
            ]
            # Backdate pause start so it looks 5 hours old
            self.orch._pause_start_time = time.time() - 18000

            # Do NOT mock resume_trimming; let real code execute
            with patch("time.time", return_value=time.time()):
                r = self.orch.trim_context(current_usage_tokens=100000, force=True)

        # Auto-resume should have run, clearing the flag
        self.assertFalse(self.orch._trimming_paused,
                         "Auto-resume should have cleared the pause flag")
        self.assertEqual(r["trimmed_blocks"], 1)

    def test_normal_trim_when_not_paused(self):
        with patch.object(self.orch, "_est_tokens", return_value=5000):
            self.orch._active_blocks = [
                {"id": "b1", "tier": 5, "content": "x", "tokens": 5000, "persist": False},
            ]
            r = self.orch.trim_context(current_usage_tokens=100000, force=True)
        self.assertEqual(r["trimmed_blocks"], 1)


# ═══════════════════════════════════════════════════════════════════════════════
# context_orchestrator.py — block protection
# ═══════════════════════════════════════════════════════════════════════════════

class TestOrchestratorProtection(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._patches = _base_context_patches()

    @classmethod
    def tearDownClass(cls):
        for p in cls._patches.values():
            p.stop()

    def setUp(self):
        from context_orchestrator import ContextOrchestrator
        self.orch = ContextOrchestrator("t2")

    def _est(self, n=10000):
        return patch.object(self.orch.__class__, "_est_tokens", return_value=n)

    def test_protect_toggle(self):
        self.assertEqual(self.orch.set_block_protected("a", protected=True),
                         {"block_id": "a", "protected": True})
        self.assertIn("a", self.orch._protected_blocks)
        self.assertEqual(self.orch.set_block_protected("a", protected=False),
                         {"block_id": "a", "protected": False})
        self.assertNotIn("a", self.orch._protected_blocks)

    def test_protected_skips_eviction(self):
        self.orch._protected_blocks = {"safe"}
        with self._est(10000):
            self.orch._active_blocks = [
                {"id": "safe", "tier": 5, "content": "x", "tokens": 10000, "persist": False},
                {"id": "doomed", "tier": 5, "content": "y", "tokens": 10000, "persist": False},
            ]
            r = self.orch.trim_context(current_usage_tokens=100000, force=True)
        remaining = {b["id"] for b in self.orch._active_blocks}
        self.assertIn("safe", remaining)
        self.assertNotIn("doomed", remaining)
        self.assertEqual(r["trimmed_blocks"], 1)

    def test_protected_skips_compression(self):
        """Protected blocks are NOT compressed; unprotected ones are."""
        self.orch._protected_blocks = {"t3p"}

        # Need enough content so _compress_block doesn't reject it as too short.
        # We mock _compress_block to return a short string, but the block
        # content must be >= 20 chars for the real guard to pass.
        with self._est(20000):
            self.orch._active_blocks = [
                {"id": "t3p", "tier": 3, "content": "prot" * 100, "tokens": 20000, "persist": False},
                {"id": "t3o", "tier": 3, "content": "open" * 100, "tokens": 20000, "persist": False},
            ]
            with patch.object(self.orch, "_compress_block", return_value="c"):
                r = self.orch.trim_context(current_usage_tokens=100000, force=True,
                                           target_model="deepseek-v4-pro")

                prot_block = next(b for b in self.orch._active_blocks if b["id"] == "t3p")
                # Protected block should NOT have _was_compressed
                self.assertNotIn("_was_compressed", prot_block,
                                 "Protected block should NOT be compressed")

                open_block = next(b for b in self.orch._active_blocks if b["id"] == "t3o")
                # Non-protected should be compressed
                self.assertIn("_was_compressed", open_block,
                              "Unprotected block should be compressed")


# ═══════════════════════════════════════════════════════════════════════════════
# auto_trim.py — file-signal pause / protect
# ═══════════════════════════════════════════════════════════════════════════════

class TestAutoTrimFileSignals(unittest.TestCase):
    """Load auto_trim into a temp directory so _resolve_base_dir() succeeds."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.tmpdir.name)
        self.scripts_dir = self.tmp / "scripts"
        self.signals_dir = self.tmp / "bridge" / "signals"
        self.pause_signal = self.signals_dir / "pause-trim"
        self.protected_signal = self.signals_dir / "protected-blocks.json"

        (self.tmp / "run_bridge.py").write_text("# dummy")
        self.scripts_dir.mkdir()
        (self.scripts_dir / "auto_trim.py").write_text(Path(AUTO_TRIM_LOC).read_text())
        self.signals_dir.mkdir(parents=True)

        # Patch HOME so auto_trim signal paths resolve to our tmpdir
        self._p_home = patch.dict(os.environ, {"HOME": str(self.tmp)})
        self._p_home.start()

        spec = __import__("importlib").util.spec_from_file_location(
            "auto_trim_test", str(self.scripts_dir / "auto_trim.py")
        )
        mod = __import__("importlib").util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.at = mod
        self.at.DRY_RUN = True

    def tearDown(self):
        self._p_home.stop()
        self.tmpdir.cleanup()

    # ── Pause signal file ──────────────────────────────────────────

    def test_pause_creates_file(self):
        r = self.at.pause_trimming(reason="test")
        self.assertEqual(r["status"], "paused")
        self.assertTrue(self.pause_signal.exists())
        data = json.loads(self.pause_signal.read_text())
        self.assertEqual(data["reason"], "test")
        self.assertIn("paused_at", data)

    def test_resume_removes_file(self):
        self.at.pause_trimming()
        self.assertTrue(self.pause_signal.exists())
        r = self.at.resume_trimming(reason="done")
        self.assertEqual(r["status"], "resumed")
        self.assertFalse(self.pause_signal.exists())

    def test_is_paused_toggles(self):
        self.assertFalse(self.at._is_trimming_paused())
        self.at.pause_trimming()
        self.assertTrue(self.at._is_trimming_paused())
        self.at.resume_trimming()
        self.assertFalse(self.at._is_trimming_paused())

    def test_auto_resume_after_expiry(self):
        """Expired pause signal → auto_resume_if_expired removes file."""
        self.at.MAX_PAUSE_SECONDS = 1
        self.at.pause_trimming(reason="stale")
        age = time.time() - 3600
        os.utime(self.at.PAUSE_SIGNAL, (age, age))

        lifted = self.at.auto_resume_if_expired()
        self.assertTrue(lifted)
        self.assertFalse(self.at._is_trimming_paused())
        self.assertFalse(self.at.PAUSE_SIGNAL.exists())

    # ── Protected blocks signal file ───────────────────────────────

    def test_protect_creates_file(self):
        r = self.at.set_block_protected("b42", protected=True)
        self.assertEqual(r, {"block_id": "b42", "protected": True})
        self.assertTrue(self.protected_signal.exists())
        data = json.loads(self.protected_signal.read_text())
        self.assertIn("b42", data["protected"])

    def test_unprotect_removes_from_file(self):
        self.at.set_block_protected("b42", protected=True)
        self.at.set_block_protected("b99", protected=True)
        self.at.set_block_protected("b42", protected=False)
        self.assertEqual(self.at._get_protected_blocks(), {"b99"})

    def test_empty_protected_returns_empty(self):
        self.assertEqual(self.at._get_protected_blocks(), set())

    def test_corrupted_signal_graceful(self):
        self.protected_signal.write_text("!!!not json")
        self.assertEqual(self.at._get_protected_blocks(), set())


# ═══════════════════════════════════════════════════════════════════════════════
# auto_trim.py — trim_context with pause + protect integration
# ═══════════════════════════════════════════════════════════════════════════════

class TestAutoTrimIntegration(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.tmpdir.name)
        self.signals_dir = self.tmp / "bridge" / "signals"
        self.pause_signal = self.signals_dir / "pause-trim"
        self.protected_signal = self.signals_dir / "protected-blocks.json"

        (self.tmp / "run_bridge.py").write_text("# dummy")
        scripts = self.tmp / "scripts"
        scripts.mkdir()
        (scripts / "auto_trim.py").write_text(Path(AUTO_TRIM_LOC).read_text())
        self.signals_dir.mkdir(parents=True)

        self._p_home = patch.dict(os.environ, {"HOME": str(self.tmp)})
        self._p_home.start()

        spec = __import__("importlib").util.spec_from_file_location(
            "auto_trim_int", str(scripts / "auto_trim.py")
        )
        mod = __import__("importlib").util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.at = mod
        self.at.DRY_RUN = True

    def tearDown(self):
        self._p_home.stop()
        self.tmpdir.cleanup()

    def test_paused_skips_trimming(self):
        self.at.pause_trimming()
        blocks = [{"id": "b1", "content": "x" * 10000, "priority": 5}]
        r = self.at.trim_context(blocks, budget=50000, threshold=50000)
        self.assertEqual(r["status"], "paused")
        self.assertEqual(r["action"], "none")
        self.assertEqual(r["blocks_deleted"], 0)

    def test_protected_block_survives(self):
        self.at.DRY_RUN = False
        self.at.MIN_BLOCKS_KEPT = 1
        self.at.set_block_protected("safe")
        blocks = [
            {"id": "safe", "content": "x" * 10000, "priority": 5},
            {"id": "doomed", "content": "y" * 10000, "priority": 5},
        ]
        with patch.object(self.at, "count_tokens", return_value=10000):
            r = self.at.trim_context(blocks, budget=5000, threshold=5000)
        self.assertEqual(r["blocks_deleted"], 1)
        self.assertIn("protected_blocks", r)
        self.assertIn("safe", r["protected_blocks"])

    def test_result_schema(self):
        """Verify auto_trim.trim_context() always includes pause/protect metadata.

        Set threshold low enough that trimming actually executes and the
        pause/protect fields are present in the result dict.
        """
        blocks = [
            {"id": "b1", "content": "x" * 1000, "priority": 6},
            {"id": "b2", "content": "y" * 1000, "priority": 6},
        ]
        with patch.object(self.at, "count_tokens", return_value=1000):
            r = self.at.trim_context(blocks, budget=5000, threshold=50)
        self.assertIn("status", r)
        self.assertIn("paused", r, "trim_context should report pause state")
        self.assertIn("protected_blocks", r, "trim_context should report protected blocks")
        self.assertFalse(r["paused"])
        self.assertIsInstance(r["protected_blocks"], list)


# ═══════════════════════════════════════════════════════════════════════════════
# gateway_integration.py — pause/protect wrappers
# ═══════════════════════════════════════════════════════════════════════════════

import gateway_integration as _gi  # noqa: E402 — module is importable


class TestGatewayPauseProtect(unittest.TestCase):
    """Test the wrapper functions in gateway_integration.py."""

    _patches_started = False

    @classmethod
    def setUpClass(cls):
        """Start mock patches once for all gateway tests."""
        if not TestGatewayPauseProtect._patches_started:
            cls._patches = _base_context_patches()
            TestGatewayPauseProtect._patches_started = True

    @classmethod
    def tearDownClass(cls):
        pass  # Do NOT stop patches — shared across test classes

    @classmethod
    def tearDownClass(cls):
        for p in cls._patches.values():
            p.stop()

    def setUp(self):
        self.mock_orch = _mock_orch()
        # Patch get_orchestrator at the gateway_integration module level
        self._p_get = patch.object(_gi, "get_orchestrator", return_value=self.mock_orch)
        self._p_get.start()
        # Pre-resolve the session key so _get_all_sessions returns the right key
        self.gate_sess = "gate_test_123"
        self._gate_sess_key = _gi._ensure_session_key(self.gate_sess)
        # Patch _get_all_sessions to include our resolved session key
        self._p_all = patch.object(
            _gi, "_get_all_sessions",
            return_value={self._gate_sess_key: self.mock_orch},
        )
        self._p_all.start()

    def tearDown(self):
        self._p_get.stop()
        self._p_all.stop()

    def test_gateway_pause(self):
        r = _gi.gateway_pause_trimming(reason="high ctx", gateway_session_id=self.gate_sess)
        self.mock_orch.pause_trimming.assert_called_once_with(reason="high ctx")
        self.assertEqual(r["status"], "paused")

    def test_gateway_resume(self):
        r = _gi.gateway_resume_trimming(reason="done", gateway_session_id=self.gate_sess)
        self.mock_orch.resume_trimming.assert_called_once_with(reason="done")
        self.assertEqual(r["status"], "resumed")

    def test_gateway_set_protect(self):
        r = _gi.gateway_set_block_protected("t0_id", protected=True,
                                            gateway_session_id=self.gate_sess)
        self.mock_orch.set_block_protected.assert_called_once_with("t0_id", protected=True)
        self.assertEqual(r, {"block_id": "mock", "protected": True})

    def test_gateway_is_paused(self):
        self.mock_orch.is_paused.return_value = True
        r = _gi.gateway_is_paused(gateway_session_id=self.gate_sess)
        self.assertTrue(r)
        self.mock_orch.is_paused.assert_called_once()

    def test_gateway_pause_info(self):
        self.mock_orch.get_pause_info.return_value = {
            "reason": "manual", "elapsed_seconds": 60, "auto_resume_in_seconds": 3540,
        }
        r = _gi.gateway_get_pause_info(gateway_session_id=self.gate_sess)
        self.assertEqual(r["reason"], "manual")
        self.assertEqual(r["elapsed_seconds"], 60)

    def test_gateway_status_paused(self):
        self.mock_orch.is_paused.return_value = True
        self.mock_orch.get_pause_info.return_value = {
            "reason": "manual", "elapsed_seconds": 50, "auto_resume_in_seconds": 3550,
        }
        r = _gi.gateway_status(gateway_session_id=self.gate_sess)
        self.assertTrue(r["is_paused"])
        self.assertEqual(r["pause_info"]["reason"], "manual")

    def test_gateway_status_not_paused(self):
        self.mock_orch.is_paused.return_value = False
        self.mock_orch.get_pause_info.return_value = None
        r = _gi.gateway_status(gateway_session_id=self.gate_sess)
        self.assertFalse(r["is_paused"])
        self.assertIsNone(r["pause_info"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
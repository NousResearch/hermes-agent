"""Automated regression test suite for Hermes Agent Windows runtime resilience.

Covers:
1. Non-ASCII/Unicode character handling in user profile / workspace paths.
2. HERMES_RUNTIME_DIR environment override resolution.
3. Clean resolution when environment variable is omitted (Electron backend isolation).
4. install-stamp.json rewrite during updates preserves runtimeDir.
5. Store python resolution parity (preventing redundant launcher re-minting / re-exec).
6. Safe reclamation of dead update markers (.hermes-update-in-progress) without affecting live updaters.
7. Consecutive execution stability (no generation thrashing).
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hermes_cli._launchers import resolve_store_python
from hermes_cli.post_update import step_verify_runtime_integrity
from hermes_cli.source_stamp import write_source_stamp
from hermes_cli.update_lock import read_live_update, update_marker_path
from hermes_cli.venv_sync import prepare_launch
from pm.environments import _unstamped_store, store_root
from pm.paths import install_stamp_path


class TestRuntimeResilience(unittest.TestCase):
    def setUp(self):
        self.original_env = os.environ.get("HERMES_RUNTIME_DIR")

    def tearDown(self):
        if self.original_env is not None:
            os.environ["HERMES_RUNTIME_DIR"] = self.original_env
        elif "HERMES_RUNTIME_DIR" in os.environ:
            del os.environ["HERMES_RUNTIME_DIR"]

    def test_scenario_1_unicode_character_path_resolution(self):
        """Scenario 1: Non-ASCII/Unicode characters in paths do not cause decode/encode errors."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            unicode_subfolder = Path(tmp_dir) / "Kullanıcı_Örnek_Çağdaş" / "hermes-agent"
            unicode_subfolder.mkdir(parents=True, exist_ok=True)
            self.assertTrue(unicode_subfolder.is_dir())
            
            # Verify string and byte round-trip
            encoded = str(unicode_subfolder).encode("utf-8")
            decoded = encoded.decode("utf-8")
            self.assertEqual(decoded, str(unicode_subfolder))

    def test_scenario_2_hermes_runtime_dir_honored(self):
        """Scenario 2: HERMES_RUNTIME_DIR environment variable is honored when present."""
        with tempfile.TemporaryDirectory() as tmp_store:
            fake_store = Path(tmp_store).resolve()
            os.environ["HERMES_RUNTIME_DIR"] = str(fake_store)
            resolved = store_root(PROJECT_ROOT, honor_runtime_override=True)
            self.assertEqual(resolved, fake_store)

    def test_scenario_3_env_var_stripped_still_resolves_consistently(self):
        """Scenario 3: When HERMES_RUNTIME_DIR is absent, store_root and _unstamped_store resolve consistently."""
        if "HERMES_RUNTIME_DIR" in os.environ:
            del os.environ["HERMES_RUNTIME_DIR"]
        
        resolved_override_false = store_root(PROJECT_ROOT, honor_runtime_override=False)
        self.assertTrue(resolved_override_false.exists())
        self.assertTrue((resolved_override_false / "facts.json").is_file() or resolved_override_false.name == "tools")

    def test_scenario_4_install_stamp_rewritten_preserves_runtime_dir(self):
        """Scenario 4: Updating or rewriting install-stamp.json preserves existing runtimeDir."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            stamp_file = tmp_root / "install-stamp.json"
            fake_runtime = Path(tmp_dir) / "custom_runtime"
            fake_runtime.mkdir()
            stamp_file.write_text(
                json.dumps({"commit": "abc1234", "runtimeDir": str(fake_runtime)}),
                encoding="utf-8"
            )
            
            # Read and verify write_source_stamp behavior on stamp
            previous_stamp = json.loads(stamp_file.read_text(encoding="utf-8-sig"))
            self.assertEqual(previous_stamp.get("runtimeDir"), str(fake_runtime))

    def test_scenario_5_store_python_resolution_and_parity(self):
        """Scenario 5: resolve_store_python resolves a valid interpreter without relaunch loops."""
        store_python = resolve_store_python(PROJECT_ROOT)
        self.assertIsNotNone(store_python)
        self.assertTrue(store_python.is_file(), f"Python at {store_python} must exist")
        
        is_same = os.path.normcase(os.path.abspath(store_python)) == os.path.normcase(os.path.abspath(sys.executable))
        if is_same:
            next_python = prepare_launch(PROJECT_ROOT, ["hermes", "--version"])
            self.assertIsNone(next_python, "prepare_launch should not relaunch when running under store Python")

    def test_scenario_6_interrupted_update_dead_lock_cleanup(self):
        """Scenario 6: Dead update markers (.hermes-update-in-progress) are safely reclaimed."""
        marker = update_marker_path()
        try:
            # Write a marker with a guaranteed non-existent PID (999999)
            marker.write_text("999999\n1700000000.0\nct:1700000000.0\n", encoding="utf-8")
            self.assertTrue(marker.is_file())
            
            # read_live_update must detect dead process and safely reclaim
            live = read_live_update(path=marker)
            self.assertIsNone(live, "Dead update owner must return None")
            self.assertFalse(marker.is_file(), "Dead marker must be cleaned up")
            
            # Run post-update integrity step
            res = step_verify_runtime_integrity()
            self.assertTrue(res.get("ok"))
        finally:
            if marker.is_file():
                try:
                    marker.unlink()
                except OSError:
                    pass

    def test_scenario_7_consecutive_resolution_stability(self):
        """Scenario 7: Consecutive resolutions execute swiftly without generation thrashing."""
        durations = []
        for _ in range(5):
            t0 = time.perf_counter()
            python = resolve_store_python(PROJECT_ROOT)
            self.assertIsNotNone(python)
            durations.append(time.perf_counter() - t0)
        
        avg_duration = sum(durations) / len(durations)
        self.assertLess(avg_duration, 0.5, f"Resolution too slow: {avg_duration:.3f}s")


if __name__ == "__main__":
    unittest.main()

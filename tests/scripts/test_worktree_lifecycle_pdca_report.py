from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "worktree-lifecycle" / "pdca_report.py"


class PdcaReportScriptTests(unittest.TestCase):
    def test_light_and_cleanup_modes_record_cadence(self):
        with tempfile.TemporaryDirectory() as folder:
            registry = Path(folder) / "registry.json"
            light = subprocess.run(
                [sys.executable, str(SCRIPT), "--mode", "light", "--registry", str(registry)],
                text=True, capture_output=True, check=False,
            )
            cleanup = subprocess.run(
                [sys.executable, str(SCRIPT), "--mode", "cleanup", "--registry", str(registry)],
                text=True, capture_output=True, check=False,
            )
            data = json.loads(registry.read_text(encoding="utf-8"))
        self.assertEqual(0, light.returncode, light.stdout + light.stderr)
        self.assertEqual(0, cleanup.returncode, cleanup.stdout + cleanup.stderr)
        self.assertIsNotNone(data["pdca"]["last_light_check_at"])
        self.assertIsNotNone(data["pdca"]["last_cleanup_review_at"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

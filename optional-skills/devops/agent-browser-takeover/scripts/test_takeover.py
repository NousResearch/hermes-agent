#!/usr/bin/env python3
"""Static checks for the takeover launcher. No live VNC required."""
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "templates" / "takeover_view.sh"


class TakeoverLauncherTests(unittest.TestCase):
    def test_launcher_has_no_hardcoded_mesh_ip(self):
        text = LAUNCHER.read_text()
        self.assertNotIn("10.66.67", text)
        self.assertNotIn("facebook", text.lower())

    def test_rejects_public_bind(self):
        env = os.environ.copy()
        env["BIND_IP"] = "0.0.0.0"
        env["TAKEOVER_BASE"] = tempfile.mkdtemp(prefix="takeover-test-")
        r = subprocess.run(
            ["bash", str(LAUNCHER), "start"],
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("Never 0.0.0.0", r.stderr)

    def test_skill_has_no_sitrep_or_intel_paths(self):
        skill = (ROOT / "SKILL.md").read_text().lower()
        self.assertNotIn("highlights", skill)
        self.assertNotIn("facebook", skill)
        self.assertNotIn("sitrep", skill)


if __name__ == "__main__":
    unittest.main()

"""Regression tests for hardware_check meminfo guard."""

import unittest
from pathlib import Path


class TestHardwareCheckMeminfoGuard(unittest.TestCase):
    """hardware_check must guard against malformed /proc/meminfo lines."""

    def test_memtotal_split_guard_in_source(self):
        """Source must contain a length check before indexing line.split()[1]."""
        repo_root = Path(__file__).resolve().parent
        source_path = repo_root / "skills" / "creative" / "comfyui" / "scripts" / "hardware_check.py"
        with open(source_path) as f:
            source = f.read()
        self.assertIn("parts = line.split()", source)
        self.assertIn("len(parts) >= 2", source)


if __name__ == "__main__":
    unittest.main()

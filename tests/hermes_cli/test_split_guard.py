"""Regression tests for split() guard fixes."""

import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


class TestSplitGuard(unittest.TestCase):
    """Verify split()[0] guards exist in cli.py for empty/whitespace strings."""

    def test_split_zero_guard_exists(self):
        """cli.py must contain guards against empty split results."""
        source = (REPO_ROOT / "cli.py").read_text()
        # Check for the guard pattern we added
        guards = [
            "text.split()[0] if text.strip() else",
            "cmd_lower.split()[0].lstrip(\"/\") if cmd_lower.strip() else",
            "cmd_lower.split()[0] if cmd_lower.strip() else",
        ]
        for guard in guards:
            self.assertIn(guard, source, f"Missing guard: {guard}")


if __name__ == "__main__":
    unittest.main()

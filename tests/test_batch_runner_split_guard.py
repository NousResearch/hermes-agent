"""Regression tests for batch_runner split guard."""

import unittest
from pathlib import Path


class TestBatchRunnerSplitGuard(unittest.TestCase):
    """batch_runner must guard batch_file.stem.split('_')[1]."""

    def test_batch_split_guard_in_source(self):
        """Source must contain a length check before indexing split result."""
        repo_root = Path(__file__).resolve().parent
        source_path = repo_root / "batch_runner.py"
        with open(source_path) as f:
            source = f.read()
        self.assertIn("batch_file.stem.split(\"_\")", source)
        self.assertIn("parts[1] if len(parts) > 1", source)


if __name__ == "__main__":
    unittest.main()

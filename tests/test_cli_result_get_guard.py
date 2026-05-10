"""Regression test for cli.py result.get() guard."""
import unittest

class TestCliResultGetGuard(unittest.TestCase):
    def test_get_guard_in_source(self):
        import pathlib
        repo_root = pathlib.Path(__file__).resolve().parents[1]
        cli_path = repo_root / "cli.py"
        with open(cli_path) as f:
            source = f.read()
        self.assertIn("result.get('error', 'Unknown error')", source)
        self.assertIn("result.get('restored_to', '?')", source)
        self.assertIn("result.get('reason', '')", source)

if __name__ == "__main__":
    unittest.main()

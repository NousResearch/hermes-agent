"""Regression test for cli.py cost float guard."""
import unittest

class TestCliCostFloatGuard(unittest.TestCase):
    def test_float_guard_in_source(self):
        import pathlib
        repo_root = pathlib.Path(__file__).resolve().parents[1]
        cli_path = repo_root / "cli.py"
        with open(cli_path) as f:
            source = f.read()
        self.assertIn("float(cost_result.amount_usd)", source)
        self.assertIn("except (ValueError, TypeError):", source)

if __name__ == "__main__":
    unittest.main()

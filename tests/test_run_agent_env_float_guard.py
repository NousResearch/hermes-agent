"""Regression test for run_agent env float guard."""
import unittest

class TestRunAgentEnvFloatGuard(unittest.TestCase):
    def test_or_fallback_in_source(self):
        import pathlib
        repo_root = pathlib.Path(__file__).resolve().parents[1]
        run_agent_path = repo_root / "run_agent.py"
        with open(run_agent_path) as f:
            source = f.read()
        self.assertIn("or 1800.0)", source)
        self.assertIn("or 120.0)", source)
        self.assertIn("or 180.0)", source)

if __name__ == "__main__":
    unittest.main()

"""Regression test for cli.py result.get() guard."""
import unittest

class TestCliResultGetGuard(unittest.TestCase):
    def test_get_guard_in_source(self):
        with open("/tmp/hermes-agent-fork/cli.py") as f:
            source = f.read()
        self.assertIn("result.get('error', 'Unknown error')", source)
        self.assertIn("result.get('restored_to', '?')", source)
        self.assertIn("result.get('reason', '')", source)

if __name__ == "__main__":
    unittest.main()

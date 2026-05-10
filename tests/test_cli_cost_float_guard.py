"""Regression test for cli.py cost float guard."""
import unittest

class TestCliCostFloatGuard(unittest.TestCase):
    def test_float_guard_in_source(self):
        with open("/tmp/hermes-agent-fork/cli.py") as f:
            source = f.read()
        self.assertIn("float(cost_result.amount_usd)", source)
        self.assertIn("except (ValueError, TypeError):", source)

if __name__ == "__main__":
    unittest.main()

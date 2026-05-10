"""Regression test for maps_client int guard."""
import unittest

class TestMapsClientIntGuard(unittest.TestCase):
    def test_try_except_in_source(self):
        with open("/tmp/hermes-agent-fork/skills/productivity/maps/scripts/maps_client.py") as f:
            source = f.read()
        self.assertIn("try:", source)
        self.assertIn("except (ValueError, TypeError):", source)
        self.assertIn("int(args.radius)", source)

if __name__ == "__main__":
    unittest.main()

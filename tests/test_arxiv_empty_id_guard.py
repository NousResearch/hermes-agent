"""Regression test for arxiv empty ID guard."""
import unittest

class TestArxivEmptyIdGuard(unittest.TestCase):
    def test_split_guard_in_source(self):
        with open("/tmp/hermes-agent-fork/skills/research/arxiv/scripts/search_arxiv.py") as f:
            source = f.read()
        self.assertIn("full_id.split('v')[0] if full_id", source)

if __name__ == "__main__":
    unittest.main()

"""Regression test for arxiv empty ID guard."""
import unittest

class TestArxivEmptyIdGuard(unittest.TestCase):
    def test_split_guard_in_source(self):
        import pathlib
        repo_root = pathlib.Path(__file__).resolve().parents[1]
        source_path = repo_root / "skills" / "research" / "arxiv" / "scripts" / "search_arxiv.py"
        with open(source_path) as f:
            source = f.read()
        self.assertIn("full_id.split('v')[0] if full_id", source)

if __name__ == "__main__":
    unittest.main()
